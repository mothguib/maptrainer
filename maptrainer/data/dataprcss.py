# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.autograd import Variable

from maptrainer import misc, train
from maptrainer.data import dlhandler


def set_bsz(data: torch.FloatTensor, bsz):
    while len(data) % bsz > 0:
        bsz -= 1
    return bsz


def batch_size(bsz,
               data) -> int:
    """
    Decodes the batch size code: `bsz`

    :param bsz:
    :type bsz:
    :param data:
    :type data:
    :return:
    :rtype:
    """

    datasize = data.size()[0]

    bsz = min(bsz, datasize)

    # Decoding the batch size (for negative values)
    return datasize // -bsz if bsz < 0 else bsz


def calibrate_batch_size(bsz,
                         model,
                         ddata,
                         tdata,
                         bptt,
                         dl,
                         criterion,
                         optimiser,
                         cuda) -> int:
    """
    Decodes the batch size code: `bsz`

    :param bsz:
    :type bsz:
    :param model:
    :type model:
    :param ddata:
    :type ddata:
    :param bptt:
    :type bptt:
    :return:
    :rtype:
    """

    # Retrieval of the batch size if it is lower than the size of data
    bsz = batch_size(bsz, ddata)

    print("Testing and calibration of the batch size:\n")

    while True:
        try:
            # Three attempts to be sure that CUDA can handle several training
            # epochs
            for n in range(3):
                train.train(model=model,
                            criterion=criterion,
                            optimiser=optimiser,
                            dl=dl,
                            ddata=ddata,
                            tdata=tdata,
                            bsz=bsz,
                            bptt=bptt,
                            clip=-1,
                            log_interval=1,
                            cuda=cuda)

            # Deleting references on model and optimiser to release memory
            del model, optimiser, criterion

            print("Enough space for a batch size of {}:".format(str(bsz)))

            break

        except RuntimeError as e:
            if str(e).startswith("CUDA out of memory"):
                print("Not enough space for a batch size of {}:".
                      format(str(bsz)))
                print(e, '\n')
            else:
                print("RunTimeError: ", e, '\n')

            bsz -= 50
        finally:
            if cuda:
                # `torch.cuda.empty_cache()` releases all the GPU memory cache
                # that can be freed. PyTorch creates a computational graph
                # whenever data are passed through the model and stores the
                # computations on the GPU memory
                torch.cuda.empty_cache()
            else:
                pass

    return bsz


def get_batch(data: torch.FloatTensor,
              i: int,
              bsz: int) -> (torch.FloatTensor, int):
    """

    :return: (batched data, batched data's batch size)
    :rtype:
    """

    bsz = min(bsz, data.size(0) - i)  # `source.size(0)`
    # corresponds to `N`, the number of data

    data = data[i:i + bsz, :]

    return data, data.size()[0]


# get_truncated subdivides the source data into chunks of length `bptt`.
# If source is equal to the example output of the batch function, with
# a `bptt`-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# The subdivison of data is not done along the batch dimension (i.e.
# dimension 1), since that was handled by the batch function. The chunks
# are along dimension 0, corresponding to the seq_len dimension in the LSTM.
def get_truncated(batch_data,
                  i: int,
                  bptt: int) -> (Variable, Variable):
    """

    :param batch_data: a tensor standing for a batch of data
        Shape: N x T x n
    :param i:
    :type i:
    :param bptt:
    :type bptt:
    :return:
    :rtype:
    """

    seq_len = min(bptt, batch_data.size(1) - i)  # `source.size(1)`
    # corresponds to `T`, the length of sequences in `batch`

    # Data must be targets because `nn.NLLLoss`, the cost method only accepts
    # classes
    batch_tctd_data = batch_data[:, i:i + seq_len]

    return batch_tctd_data


def repackage_hidden(h):
    """
    Detaches hidden states, to detach them from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def make_cuda(input_, targets, ca: bool) -> tuple:
    """

    :param input_:
    :type input_:
    :param targets:
    :type targets:
    :param ca:
    :type ca:
    :return: a tuple of two tensors
    :rtype:
    """

    return (misc.cuda(input_, ca), misc.cuda(targets, ca)) \
        if ca else (input_, targets)


def make_diff(input_, targets) -> (Variable, Variable):
    """
    Makes `input_` and `targets` differentiable i.e. wrapped in `Variable`
    to compute gradient

    :param input_:
    :type input_:
    :param targets:
    :type targets:
    :return:
    :rtype:
    """
    return Variable(input_), Variable(targets)


def neighbour_correct_labels(pred_targets: torch.FloatTensor,
                             sources: torch.FloatTensor,
                             labels: torch.FloatTensor) \
        -> tuple:
    """
    Evaluates whether the predicted targets are correct.

    :param pred_targets: probabilities of the targets estimated by the model.
        shape: (Number of edges, Number of vertices).
    :param sources: batch of the topology's arc sources, each source is
    represented by an one-hot vector and the batch is composed of all the
    edges of the topology
        shape: (Number of edges, Number of vertices)
    :param labels: arc targets of the topology of the used topology,
    each label is represented by an vertex id
        shape: (Number of edges)
    :return: `(number of target predicted correctly), correctness list)`
    :rtype:
    """

    # Sources and labels
    sources = torch.argmax(sources, dim=1)

    # Predicted labels
    pred_target_ids = torch.argmax(pred_targets, dim=1)

    nb_edges = len(sources)
    nb_vts = len(torch.unique(sources))

    nghbrs = [[] for _ in range(nb_vts)]

    for i in range(nb_edges):
        nghbrs[sources[i].item()].append(labels[i].item())

    # Correct classes
    cc = np.zeros(nb_vts)

    for i in range(nb_edges):
        if pred_target_ids[i].item() in nghbrs[sources[i]]:
            cc[sources[i]] = 1

    return np.sum(cc), cc


# TODO: moving it in the future module `valid`
def ngh_accuracy(model: torch.nn.Module,
                 criterion: torch.nn.modules.loss,
                 nagts: int,
                 tpl: str,
                 cuda: bool):
    dl = dlhandler.load_data_loader("BP")(nagts=nagts, tpl=tpl, nb_folds=1,
                                          pre=True)
    dl.load_pre_data()

    # Running on test data.
    val_ddata, val_tdata = dl.next_fold()[1]
    val_ddata, val_tdata = make_cuda(val_ddata, val_tdata, cuda)

    bsz = val_ddata.size()[0]

    print("Validation batch's size: ", bsz, '\n')

    outputs, accuracy, cost = \
        train.evaluate(model=model,
                       criterion=criterion,
                       dl=dl,
                       ddata=val_ddata,
                       tdata=val_tdata,
                       bsz=bsz,
                       bptt=val_tdata.size()[
                           1],
                       cuda=cuda)

    val_ddata, val_ldata = dl.label_data(val_ddata, val_tdata)

    outputs = outputs.view(val_ddata.size())

    ncc, cc = neighbour_correct_labels(torch.exp(outputs).squeeze(),
                                       val_ddata.squeeze(),
                                       val_ldata.squeeze())

    print("Neighbour-correct classes:")

    print(ncc, '\n')
    print(cc, '\n')

    print("Output probabilities:")
    print(torch.exp(outputs).squeeze().tolist()[0], '\n')


def correct_labels(inputs_: torch.FloatTensor,
                   preds: torch.FloatTensor,
                   targets: torch.FloatTensor,
                   dl):
    """
    Returns the number of correct labels from the inputs, the targets and
    the predicted targets

    :param inputs_:
    :type inputs_:
        shape: (Batch size, BPTT, n_in)
    :param preds:
    :type preds:
    :param targets: the output vectors not labelled
    :type targets:
        shape: (Batch size, BPTT, n_out)
    :param dl:
    :type dl:
    :return:
    :rtype:
    """

    # If `preds` has not the required size, it is resized before being
    # passed to `dl.label_data`
    preds = preds.view(targets.size())

    # Data are moved on the CPU to avoid to take space on CUDA
    inputs_, true_labels = dl.label_data(inputs_.cpu(), targets.cpu())
    inputs_, pred_labels = dl.label_data(inputs_, preds.cpu())

    true_labels = true_labels.view(-1).numpy()
    pred_labels = pred_labels.view(-1).numpy()

    return (true_labels == pred_labels).sum()


def accuracy_score(inputs_: torch.FloatTensor,
                   targets: torch.FloatTensor,
                   preds: torch.FloatTensor,
                   dl):
    """

    :param inputs_:
    :type inputs_:
        shape: (Batch size, BPTT, n_in)
    :param preds:
    :type preds:
    :param targets: the output vectors not labelled
    :type targets:
        shape: (Batch size, BPTT, n_out)
    :param dl:
    :type dl:
    :return:
    :rtype:
    """

    return correct_labels(inputs_, preds, targets, dl) / \
        float(len(targets.view(-1, targets.size()[-1]))) * 100
