# coding: utf-8

import maptrainer.data.dataprcss as dp


def evaluate(model,
             criterion,
             dl,
             bddata,
             btdata,
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
    :param bddata: batched domain data
    :type bddata:
    :param btdata:
    :type btdata:
    :param bsz:
    :type bsz:
    :param bptt:
    :type bptt:
    :param cuda:
    :type cuda:
    :return:
    :rtype:
    """

    # Turn on evaluation mode which disables dropout.
    model.eval()

    # Average cost over all elements of the data `batched_idls`
    total_cost = 0

    # Hidden state
    # TODO: cope with this case for all RNNS
    if model.variant == "LSTM":
        hidden = model.init_hidden(bsz)

    # Shapes of the `bddata` tensor
    _N_b, _N, _T, _n_in = bddata.size()
    ''' with:
    * `_N_b`, the number of batches,
    * `_N` the size of a (mini-)batch,
    * `_T` the length of sequences,
    * `_n_in`, the dimension of input
    "_" prefix due to the PEP8 naming conventions which states that variable in
    function should be lowercase. '''

    for i in range(len(bddata)):
        # val_batch: tensor representing a batch of validation data

        for truncated_id, tid in enumerate(range(0, _T, bptt)):

            # The current truncated batches of domain and labelled data
            batch_trcted_domain_data, batch_trcted_labels = \
                dp.get_truncated(dl=dl,
                                 batch_domain_data=bddata[i],
                                 batch_target=btdata[i],
                                 i=tid,
                                 bptt=bptt)
            '''
            param: `batch_batch_trcted_domain_data`
            Shape: _N x bptt x _n_in
            param: `batch_trcted_labels` 
            Shape: _N x bptt x _n_in
            '''

            batch_trcted_domain_data, batch_trcted_labels = \
                dp.make_cuda(batch_trcted_domain_data, batch_trcted_labels,
                             cuda)

            # Making `_input` and `targets` differentiable i.e.wrapped in
            # `Variable` to compute gradient
            batch_trcted_domain_data, batch_trcted_labels = \
                dp.make_diff(batch_trcted_domain_data, batch_trcted_labels)

            # TODO: cope with this case for all RNNs
            if model.variant == "LSTM":
                output, hidden = model(batch_trcted_domain_data, hidden)
            else:
                output = model(batch_trcted_domain_data)

            '''
            param: `output`
            Shape: _N x bptt x _n_in
            '''

            cost = criterion(output.view(-1, output.size()[-1]),
                             batch_trcted_labels.view(-1))
            """ Before to be passed to the criterion, `output` is reshaped 
            so that it is now a tensor of shape `(N x T) x n_in` (2 
            dimensions). Most criteria average according to the size. In 
            doing so, the returned cost is by default averaged over the 
            current batch and here, following the reshaping, over the 
            truncated. """

            total_cost += batch_trcted_labels.size(1) * cost.data[0]
            """ `cost.data[0]` as the current cost, is multiplied by 
            `trcted_batch.size(1)`, the length of truncated sequences 
            populating `trcted_batch`, in order to weight the total cost with 
            respect to the size of truncated sequences. The final sum will be 
            divided by the length of sequence (not being truncated): `_T` """

            # The hidden state is reset before feeding a new example to the
            # model.
            # TODO: cope with this case for all RNNs
            if model.variant == "LSTM":
                hidden = dp.repackage_hidden(hidden)

    return total_cost / _T  # Dividing `total_cost` by `_T`, the length of
    # sequences, enables to average over the sequence length the cost computed
    # as a weighted sum of truncated sequences, with respect to the size
    # of each truncated


def complete_evaluation(model,
                        criterion,
                        dl,
                        bsz,
                        bptt,
                        cuda: bool = False):

    bptt = dl.domain_data.size()[1] if bptt == -1 else bptt

    bsz = dl.domain_data.size()[0] if bsz == -1 else bsz

    batched_input = dp.batch(dl.domain_data, bsz)[0]
    batched_target = dp.batch(dl.target_data, bsz)[0]

    val_cost = evaluate(model=model,
                        criterion=criterion,
                        dl=dl,
                        bddata=batched_input,
                        btdata=batched_target,
                        bsz=bsz,
                        bptt=bptt,
                        cuda=cuda)

    return val_cost
