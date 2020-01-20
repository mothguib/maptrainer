# coding: utf-8

import json
import time
import math

import numpy as np
import torch

from maptrainer.model import modelhandler
from .data import dataprcss as dp
from maptrainer import misc, os


###############################################################################
# Training functions
###############################################################################

def evaluate(model,
             criterion,
             dl,
             ddata,
             tdata,
             bsz,
             bptt,
             cuda: bool = False):
    """

    :param model:
    :type model:
    :param criterion:
    :type criterion:
    :param dl: the data loader
    :type dl:
    :param ddata: domain data
    :type ddata:
    :param tdata: target data
    :type tdata:
    :param bsz: validation batch size code
    :type bsz:
    :param bptt:
    :type bptt:
    :param cuda:
    :type cuda:
    :return:
    :rtype:
    """

    bsz = dp.batch_size(bsz, ddata)

    # Correct labels
    cl = 0

    # Average cost over all elements of the data `batched_idls`
    total_cost = 0

    # Shapes of the `bddata` tensor
    N, T, n_in = ddata.size()

    # with:
    # * `N` the number of elements in the current database,
    # * `T` the length of sequences,
    # * `n_in`, the dimension of input

    # all_hidden = (misc.cuda(torch.Tensor(), cuda), misc.cuda(torch.Tensor(),
    #                                                          cuda))

    outputs = misc.cuda(torch.Tensor(), cuda)

    # Moving of the model onto cuda if needed
    model = misc.cuda(model, cuda)

    # Turn on evaluation mode which disables dropout.
    model.eval()

    for batch_id, bid in enumerate(range(0, N, bsz)):  # The current batch
        # from training data

        # Current batches of domain data and its size
        batch_domain_data, cbsz = dp.get_batch(data=ddata,
                                               i=bid,
                                               bsz=bsz)

        # Current batches of target data and its size
        batch_targets, cbsz = dp.get_batch(data=tdata,
                                           i=bid,
                                           bsz=bsz)

        # TODO: creating a function for the piece of code bellow which
        #  stands in `dataprcss` and `train.train` as well
        # TODO: cope with this case for all RNNs
        if model.variant == "LSTM":
            # The hidden state is reinitialised for the next batch
            hidden = model.init_hidden(cbsz)

        # Loss accumulated over the log interval
        log_intvl_cost = 0

        for truncated_id, tid in enumerate(range(0, T, bptt)):  # The current
            # truncated batch of domain data and targets

            batch_trcted_domain_data = \
                dp.get_truncated(batch_data=batch_domain_data,
                                 i=tid,
                                 bptt=bptt)
            batch_trcted_targets = \
                dp.get_truncated(batch_data=batch_targets,
                                 i=tid,
                                 bptt=bptt)

            batch_trcted_domain_data, batch_trcted_labels = dl.label_data(
                batch_trcted_domain_data, batch_trcted_targets)
            '''
            param: `batch_batch_trcted_domain_data`
            shape: _N x bptt x _n_in
            param: `batch_trcted_labels` 
            shape: _N x bptt x _n_in
            '''

            batch_trcted_domain_data, batch_trcted_labels = \
                dp.make_cuda(batch_trcted_domain_data, batch_trcted_labels,
                             cuda)

            with torch.no_grad():  # In evaluation mode `torch.no_grad()`
                # disables gradient calculation. In this mode, the result of
                # every computation will have `requires_grad=False`,
                # even when the inputs have `requires_grad=True`. Source:
                # https://pytorch.org/docs/stable/autograd.html#torch\
                # .autograd.no_grad

                # TODO: creating a function for the piece of code bellow which
                #  stands in `dataprcss` and `train.train` as well
                # TODO: cope with this case for all RNNs
                if model.variant == "LSTM":
                    output, hidden = model(batch_trcted_domain_data, hidden)
                else:
                    output = model(batch_trcted_domain_data)

            # `torch.cuda.empty_cache()` releases all the GPU memory cache
            # that can be freed. PyTorch creates a computational graph whenever
            # data are passed through the model and stores the computations
            # on the GPU memory
            torch.cuda.empty_cache()

            '''
            param: `output`
            Shape: _N x bptt x _n_in
            '''

            outputs = torch.cat((outputs, output.view(-1)))

            cost = criterion(dl.reshape_output(output),
                             dl.reshape_labels(batch_trcted_labels))
            """ Before passing to the criterion, `output` is reshaped 
            according to the data loader here used."""

            total_cost += batch_trcted_labels.size(0) * \
                          batch_trcted_labels.size(1) * cost.detach().item()
            # `cost.item()`, the current cost, is multiplied by:
            # * `trcted_batch.size(0)`, the number of elements in the current
            # batch, in order to weigh the total cost w.r.t. the size of
            # the current batch, given that by default in PyTorch the cost is
            # divided by the number of elements in the output. The final sum
            # will be divided by the number of elements in all the current
            # database: `N`
            # * `trcted_batch.size(1)`, the length of truncated sequences
            # populating `trcted_batch`, in order to weigh the total cost
            # w.r.t. the size of truncated sequences. The final sum
            # will be divided by the length of sequence (not being truncated):
            # `T`

            cl += dp.correct_labels(inputs_=batch_trcted_domain_data,
                                    preds=output,
                                    targets=batch_trcted_targets,
                                    dl=dl)

            # The hidden state is reset before feeding a new example to the
            # model.
            # TODO: cope with this case for all RNNs
            if model.variant == "LSTM":
                hidden = dp.repackage_hidden(hidden)

            # Deleting references on model and optimiser to release memory
            del batch_trcted_domain_data, batch_trcted_targets, \
                batch_trcted_labels, cost, output

    accuracy = cl / (N * T) * 100 # `N * T` corresponds to the number of
    # elements in the database

    return outputs, accuracy, total_cost / (N * T)  # Dividing
    # `total_cost` by `T`, the length of sequences, enables to average over
    # the sequence length the cost computed as a weighted sum of the truncated
    # sequences, w.r.t. the size of each truncated


def train(model,
          criterion,
          optimiser,
          dl,
          ddata,
          tdata,
          bsz,
          bptt,
          clip,
          log_interval,
          cuda: bool = False):
    """
    Trains the model from `bddata` and returns the cost of its
    last batch.

    :param dl: the data loader
    :type dl:
    :param model:
    :type model:
    :param criterion:
    :type criterion:
    :param ddata: domain data
    :type ddata:
    :param tdata: target data
    :param bsz: training batch size
    :type bsz:
    :param bptt:
    :type bptt:
    :param lr:
    :type lr:
    :param clip:
    :type clip:
    :param log_interval:
    :type log_interval:
    :param cuda:
    :type cuda:
    :return:
    :rtype:
    """

    bsz = dp.batch_size(bsz, ddata)

    hidden = None

    # Maximum gradient norm
    max_grad_norm = -1
    # Shapes of the `bddata` tensor
    N, T, n_in = ddata.size()

    # with:
    # * `N` the size of a (mini-)batch,
    # * `T` the length of sequences,
    # * `n_in`, the dimension of input
    # "_" suffix due to the PEP8 naming conventions which states that variable
    # in function should be lowercase.

    # Moving of the model onto cuda if needed
    model = misc.cuda(model, cuda)

    # Turn on training mode which enables dropout.
    model.train()

    for batch_id, bid in enumerate(range(0, N, bsz)):  # The current batch
        # from training data

        # Current batch of domain data and its size
        batch_domain_data, cbsz = dp.get_batch(data=ddata,
                                               i=bid,
                                               bsz=bsz)

        # Current batch of target data and its size
        batch_targets, cbsz = dp.get_batch(data=tdata,
                                           i=bid,
                                           bsz=bsz)

        # TODO: cope with this case for all RNNS
        if model.variant == "LSTM":
            # The hidden state is reinitialised for the next batch
            hidden = model.init_hidden(cbsz)

        # Loss accumulated over the log interval
        log_intvl_cost = 0

        for truncated_id, tid in enumerate(range(0, T, bptt)):

            # The current truncated batch of domain data and targets
            batch_trcted_domain_data = \
                dp.get_truncated(batch_data=batch_domain_data,
                                 i=tid,
                                 bptt=bptt)
            batch_trcted_targets = \
                dp.get_truncated(batch_data=batch_targets,
                                 i=tid,
                                 bptt=bptt)

            batch_trcted_domain_data, batch_trcted_labels = dl.label_data(
                batch_trcted_domain_data, batch_trcted_targets)
            '''
            param: `batch_batch_trcted_domain_data`
            shape: _N x bptt x _n_in
            param: `batch_trcted_labels` 
            shape: _N x bptt x _n_in
            '''

            batch_trcted_domain_data, batch_trcted_labels = \
                dp.make_cuda(batch_trcted_domain_data, batch_trcted_labels,
                             cuda)

            # TODO: creating a function for the piece of code bellow,
            #  standing in `dataprcss` and `train.evaluate` as well
            # TODO: cope with this case for all RNNs
            if model.variant == "LSTM":
                # Starting each truncated, we detach the hidden state from
                # how it was previously produced. If it is not done,
                # the model would try to back-propagate all the way to start
                # of `bddata`
                hidden = model.repackage_hidden(hidden)

            # Zeroing the gradients of all the parameters of the model,
            # accumulated with the `backward` method
            model.zero_grad()

            # TODO: cope with this case for all RNNs
            if model.variant == "LSTM":
                output, hidden = model(batch_trcted_domain_data, hidden)
            else:
                output = model(batch_trcted_domain_data)
            '''
            param: `output`
            Shape: _N x bptt x _n_in
            '''

            cost = criterion(dl.reshape_output(output),
                             dl.reshape_labels(batch_trcted_labels))
            """ Before passing to the criterion, `output` is reshaped 
            according to the data loader here used. """

            # Backward-propagation
            cost.backward()

            # Max grad norm
            if clip != -1:
                # `clip_grad_norm` helps prevent the exploding gradient problem
                # in RNNs / LSTMs.
                grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(),
                                                          clip)
            else:
                grad_norm = misc.grad_norm(model.parameters())

            max_grad_norm = grad_norm if grad_norm > max_grad_norm \
                else max_grad_norm

            optimiser.step()

            # for p in model.parameters():
            #    p.data.add_(-lr, p.grad.data)

            # log_intvl_cost += cost_value

            '''
            if truncated_id % log_interval == 0 and truncated_id > 0:
                cur_cost = log_intvl_cost[0] / log_interval
                elapsed = time.time() - start_time
    
                
                print(
                    '| epoch {:3d} | {:5d}/{:5d} truncated_id | lr {:02.2f} | '
                    'ms/truncated {:5.2f} | cost {:5.2f} | ppl {:8.2f}'.format(
                        epoch, truncated_id, len(bddata) // bptt, lr,
                                      elapsed * 1000 / log_interval,
                        cur_cost,
                        math.exp(cur_cost)))
            
    
                log_intvl_cost = 0
                start_time = time.time()
            '''

            # Deleting references on model and optimiser to release memory
            del batch_trcted_domain_data, batch_trcted_targets, \
                batch_trcted_labels, output, cost

        if cuda:
            # `torch.cuda.empty_cache()` releases all the GPU memory cache
            # that can be freed. PyTorch creates a computational graph whenever
            # data are passed through the model and stores the computations
            # on the GPU memory
            torch.cuda.empty_cache()

    # Cost on the training database after the current training stage
    accuracy, train_cost = evaluate(model=model,
                                    criterion=criterion,
                                    dl=dl,
                                    ddata=ddata,
                                    tdata=tdata,
                                    bsz=bsz,
                                    bptt=bptt,
                                    cuda=cuda)[-2:]

    return accuracy, train_cost, max_grad_norm


def validate(model,
             criterion,
             dl,
             val_ddata,
             val_tdata,
             bsz,
             bptt,
             cuda: bool = False):
    """

    :param model:
    :type model:
    :param criterion:
    :type criterion:
    :param dl:
    :type dl:
    :param val_ddata:
    :type val_ddata:
    :param val_tdata:
    :type val_tdata:
    :param bsz:
    :type bsz:
    :param bptt:
    :type bptt:
    :param cuda:
    :type cuda:
    :return:
    :rtype:
    """

    return evaluate(model=model,
                    criterion=criterion,
                    dl=dl,
                    ddata=val_ddata,
                    tdata=val_tdata,
                    bsz=bsz,
                    bptt=bptt,
                    cuda=cuda)[-2:]


def complete_evaluation(model,
                        criterion,
                        dl,
                        bsz,
                        bptt,
                        cuda: bool = False):
    accuracy, val_cost = evaluate(model=model,
                                  criterion=criterion,
                                  dl=dl,
                                  ddata=dl.domain_data,
                                  tdata=dl.target_data,
                                  bsz=bsz,
                                  bptt=bptt,
                                  cuda=cuda)[-2:]

    if cuda:
        # `torch.cuda.empty_cache()` releases all the GPU memory cache
        # that can be freed. PyTorch creates a computational graph whenever
        # data are passed through the model and stores the computations
        # on the GPU memory
        torch.cuda.empty_cache()

    return val_cost, accuracy


def run_epoch(model,
              criterion,
              optimiser,
              bptt,
              lr,
              dl,
              folds,
              bsz,
              clip,
              log_interval,
              cuda):
    train_cost = 0
    val_cost = 0
    max_grad_norm = -1
    accuracy = 0

    for _ in range(folds):
        # * `train_ddata`, for "training domain data"
        # * `train_tdata`, for "training target data"
        # * `val_ddata`, for "validation domain data"
        # * `val_tdata`, for "validation target data"

        (train_ddata, train_tdata), (val_ddata, val_tdata) = dl.next_fold()

        # Current epoch's training
        accuracy, epoch_train_cost, fold_max_grad_norm = \
            train(model=model,
                  criterion=criterion,
                  optimiser=optimiser,
                  dl=dl,
                  ddata=train_ddata,
                  tdata=train_tdata,
                  bsz=bsz,
                  bptt=bptt,
                  clip=clip,
                  log_interval=log_interval,
                  cuda=cuda)

        train_cost += epoch_train_cost
        accuracy, cost = validate(model=model,
                                  criterion=criterion,
                                  dl=dl,
                                  val_ddata=val_ddata,
                                  val_tdata=val_tdata,
                                  bsz=-1,
                                  bptt=bptt,
                                  cuda=cuda)

        # TODO: to delete
        # misc.test_data(train_ddata, train_tdata, val_ddata, val_tdata)

        val_cost += cost

        max_grad_norm = fold_max_grad_norm \
            if fold_max_grad_norm > max_grad_norm else max_grad_norm

    train_cost /= folds
    val_cost /= folds

    return train_cost, val_cost, accuracy, max_grad_norm


def run_epochs(model, criterion, optimiser, bptt, lr, dl, epochs, folds, bsz,
               clip, log_interval, alr, cuda, cost_path, log_path,
               model_path, ac_path, eoe_log_interval, writer):
    """

    :param writer:
    :type writer:
    :param optimiser:
    :type optimiser:
    :param model:
    :type model:
    :param criterion:
    :type criterion:
    :param bptt:
    :type bptt:
    :param lr:
    :type lr:
    :param dl:
    :type dl:
    :param epochs:
    :type epochs:
    :param folds:
    :type folds:
    :param bsz:
    :type bsz:
    :param clip:
    :type clip:
    :param log_interval:
    :type log_interval:
    :param alr:
    :type alr:
    :param cuda:
    :type cuda:
    :param cost_path:
    :type cost_path:
    :param log_path:
    :type log_path:
    :param model_path:
    :type model_path:
    :param ac_path:
    :type ac_path:
    :param eoe_log_interval: end-of-epoch's log interval i.e. the report
    interval of the end of epoch: displaying and writing
    :type eoe_log_interval:
    :return:
    :rtype:
    """
    best_val_cost = None

    val_costs = []

    train_costs = []

    accuracies = []

    # The latest best validation cost epoch
    lbvl_epoch = 0
    # Patience = number of epochs to wait before early stop if no progress on
    # the validation set.
    patience = epochs
    # The latest updated learning rate's epoch
    lulr_epoch = 0
    # Threshold of time with no improvement beyond which the learning rate
    # is updated
    no_imprvm_threshold = 1
    # Maximum gradient's norm
    max_grad_norm = -1

    # At any point it can be hit Ctrl + C to break out of training early.
    try:
        epoch_start_time = time.time()

        # Initial evaluation over all the database
        val_cost, accuracy = complete_evaluation(model=model,
                                                 criterion=criterion,
                                                 dl=dl, bsz=bsz, bptt=bptt,
                                                 cuda=cuda)

        # The first cost without training is also added to the `train_costs`
        #  list to have the same size than `val_costs`
        train_costs += [val_cost]
        val_costs += [val_cost]
        accuracies += [accuracy]

        # `torch.cuda.empty_cache()` releases all the GPU memory cache
        # that can be freed. PyTorch creates a computational graph whenever
        # data are passed through the model and stores the computations
        # on the GPU memory
        torch.cuda.empty_cache()

        str_metrics = '| end of epoch {:3d} | time: {:5.2f}s  ' \
                      '| lr: {:1.5f}| train cost: _ ' \
                      '| total cost: {:2.5f} | accuracy: {:2.5f} ' \
            .format(0, (time.time() - epoch_start_time), lr,
                    val_cost, accuracy)

        print('-' * 132)
        print(str_metrics)
        print('-' * 132)

        with open(log_path, 'w') as s:
            s.write('-' * 132 + '\n')
            s.write(str_metrics + '\n')
            s.write('-' * 132 + '\n')

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            train_cost, val_cost, accuracy, epoch_max_grad_norm = \
                run_epoch(model,
                          criterion,
                          optimiser,
                          bptt, lr, dl,
                          folds,
                          bsz,
                          clip,
                          log_interval,
                          cuda)

            train_costs += [train_cost]
            val_costs += [val_cost]
            accuracies += [accuracy]

            misc.save_progress(writer, model, train_cost, val_cost, accuracy,
                               epoch)

            max_grad_norm = epoch_max_grad_norm \
                if epoch_max_grad_norm > max_grad_norm else max_grad_norm

            # `torch.cuda.empty_cache()` releases all the GPU memory cache
            # that can be freed. PyTorch creates a computational graph whenever
            # data are passed through the model and stores the computations
            # on the GPU memory
            torch.cuda.empty_cache()

            if epoch < 250 or epoch % eoe_log_interval == 0 or epoch == epochs:
                # Save of the costs to plot them thereafter
                with open(cost_path, 'w') as s:
                    json.dump([train_costs, val_costs], s)

                with open(ac_path, 'w') as s:
                    json.dump(accuracies, s)

                #         '| train ppl: {:2.5f} | valid ppl: {:3.2f} ' \
                str_metrics = "| end of epoch {:3d} | time: {:5.2f}s  " \
                              "| lr: {:1.5f}| train cost: {:2.5f} " \
                              "| valid cost: {:2.5f} | accuracy: {:2.5f} " \
                              "| epoch's max grad norm: {:4.4f} " \
                              "| training's max grad norm: {:4.4f} " \
                    .format(epoch, (time.time() - epoch_start_time), lr,
                            train_cost, val_cost, accuracy,
                            math.exp(train_cost) if train_cost < 710 else -1,
                            math.exp(val_cost) if val_cost < 710 else -1,
                            epoch_max_grad_norm, max_grad_norm)

                print('-' * 132)
                print(str_metrics)
                print('-' * 132)

                with open(log_path, 'a') as s:
                    s.write('-' * 132 + '\n')
                    s.write(str_metrics + '\n')
                    s.write('-' * 132 + '\n')

                misc.save_progress(writer, model, train_cost, val_cost,
                                   accuracy,
                                   epoch)

            # Saving the model if the validation cost is the best we've seen so
            # far.
            if not best_val_cost or val_cost < best_val_cost:
                model_dirpath = os.path.dirname(model_path)
                if not os.path.exists(model_dirpath):
                    os.makedirs(model_dirpath)

                modelhandler.save_model(model, model_path)

                best_val_cost = val_cost
                lbvl_epoch = epoch
                lulr_epoch = lbvl_epoch
            else:
                # If no progress after `patience` epochs, early stopping
                if epoch - lbvl_epoch > patience:
                    print("Patience exceeded")
                    break
                if alr:
                    # Anneal the learning rate if no improvement has been
                    # seen in the validation dataset during
                    # `no_imprvm_threshold`.
                    if epoch - lulr_epoch > no_imprvm_threshold:
                        lr /= 2
                        print("DBG: lr: ", lr)
                        lulr_epoch = epoch

        # Final evaluation
        if os.path.exists(model_path):
            model = modelhandler.load_model_state_from_path(model=model,
                                                            path=model_path)

        val_cost, accuracy = complete_evaluation(model=model,
                                                 criterion=criterion,
                                                 dl=dl, bsz=bsz, bptt=bptt,
                                                 cuda=cuda)

        # The first cost without training is also added to the `train_costs`
        #  list to have the same size than `val_costs`
        train_costs += [val_cost]
        val_costs += [val_cost]

        str_metrics = '| end of epoch {:3d} | time: {:5.2f}s  ' \
                      '| lr: {:1.5f}| train cost: _ ' \
                      '| total cost: {:2.5f} | accuracy: {:2.5f} ' \
            .format(epochs + 1, (time.time() - epoch_start_time), lr,
                    val_cost, accuracy)

        print('-' * 132)
        print(str_metrics)
        print('-' * 132)

        with open(log_path, 'a') as s:
            s.write('-' * 132 + '\n')
            s.write(str_metrics + '\n')
            s.write('-' * 132 + '\n')

    except KeyboardInterrupt:
        print('-' * 112)
        print('Exiting from training early')

    return train_costs, val_costs
