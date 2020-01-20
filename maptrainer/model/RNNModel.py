# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable

from maptrainer.data import dataprcss as dp
from maptrainer.model.MAPModel import MAPModel
from ..data import INIT_RANGE_BOUND


class RNNModel(MAPModel):
    """
    `RNNModel`: RNN model

    Container module standing for an RNN with a softmax output layer
    """

    def __init__(self,
                 ninputs,
                 nhid,
                 nlayers,
                 variant="LSTM",
                 noutputs=-1,
                 dropout=0.0,
                 graph=None,
                 **kwargs):

        super(RNNModel, self).__init__(graph=graph)

        self.ninputs = ninputs
        noutputs = ninputs if noutputs == -1 else noutputs

        self.noutputs = noutputs

        self.variant = variant

        self.nhid = nhid

        self.nlayers = nlayers

        self.drop = nn.Dropout(dropout)

        if variant in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, variant)(ninputs, nhid, nlayers,
                                            dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[
                    variant]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was 
                supplied, options are ['LSTM', 'GRU', 'RNN_TANH' or 
                RNN_RELU']""")

            self.rnn = nn.RNN(ninputs, nhid, nlayers,
                              nonlinearity=nonlinearity, dropout=dropout)

        # The linear layer as projection that maps hidden state space to
        # vertices' space namely that this linear layer has as many units
        # as there are vertices
        self.linearise = nn.Linear(nhid, noutputs)

        self.init_parameters()

    def init_parameters(self, init_range_bound: int = INIT_RANGE_BOUND):
        """

        Initialises the weights of the output layer
        :return:
        :rtype:
        """
        self.rnn.reset_parameters()
        self.linearise.weight.data.uniform_(-init_range_bound,
                                            init_range_bound)

    def forward(self, input_, hidden):
        """

        :param input_:
        Shape: N x T x n_in
        :type input_: FloatTensor or Variable
        :param hidden: (h_t, c_t)
        :type hidden:
        :return:
        :rtype:
        """

        input_ = input_.permute(1, 0, 2)  # The dimension representing the
        # index of elements in a sequence (or the tth element of the
        # sequence) is put into the 1st dim (axis 0) and the one
        # representing indices of sequence (the nth sequence) into the 2nd
        # dim (axis 1). Henceforth, `_input` will have a shape of `T x N x
        # n_ins`.

        dropped_out_input = self.drop(input_)

        self.rnn.flatten_parameters()

        output, hidden = self.rnn(dropped_out_input, hidden)

        dropped_out_output = self.drop(output)

        log_softmaxified = \
            F.log_softmax(self.linearise(dropped_out_output), dim=2)  # The 
        # log-softmax is computed along the 2nd dimension, the dimension 1, and
        # not along the 0 one

        # log_softmaxified = F.log_softmax(linearised.permute(2, 0, 1)).\
        #  permute(1, 2, 0).contiguous()
        # Permutation of dimensions so that every slice along the 2nd dim
        # (the dimension of the output features) sums to 1 along every
        # remaining dimensions

        log_softmaxified = log_softmaxified.permute(1, 0, 2).contiguous()
        # The dimensions are permuted to return the output in the
        # standard shape of `N x T x n_in`

        return log_softmaxified, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data  # returns the next element of
        # the iterator returned by `self.parameters()`, namely here its
        # first parameter matrix

        if self.variant == 'LSTM':
            return (
                weight.new(self.nlayers, bsz, self.nhid).zero_(),
                # `torch.Tensor.new` constructs a new tensor of the same
                # data type as the `weight` tensor.
                weight.new(self.nlayers, bsz, self.nhid).zero_()
            )  # returns (h_t, c_t)
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    @staticmethod
    def repackage_hidden(h):
        """

        :param h:
        :type h:
        :return:
        :rtype:
        """

        return dp.repackage_hidden(h)

    def predict(self, input_, bsz):
        if isinstance(input_, np.ndarray):
            input_ = autograd.Variable(torch.from_numpy(input_).float())

        if isinstance(input_, autograd.Variable):
            if len(input_.size()) == 2:
                input_ = input_.view(len(input_), 1, -1)

            sizes = input_.size()
            if sizes[1] == 1:
                input_ = input_.expand(sizes[0], bsz, sizes[2])
        else:
            raise TypeError(
                "input_ must be a np.ndarray or an autograd.Variable")

        hidden = self.init_hidden(bsz)

        outputs, hidden = self(input_, hidden)

        return outputs[:, 0, :], hidden[:, 0, :]
