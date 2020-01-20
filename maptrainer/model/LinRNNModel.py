# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn

from maptrainer.model.MAPModel import MAPModel
from ..data import INIT_RANGE_BOUND


class LinRNNModel(MAPModel):
    """
    `LinRNNModel`: Linear-output RNN model

    Container module standing for an RNN with a linear output layer
    """

    def __init__(self,
                 n_input,
                 _n_hid,
                 _nlayers,
                 variant="LSTM",
                 dropout=0.0,
                 **kwargs):

        super(LinRNNModel, self).__init__()
        self.variant = variant
        self.nhid = _n_hid
        self.nlayers = _nlayers
        self.drop = nn.Dropout(dropout)

        # The linear layer as projection that maps hidden state space to
        # vertices' space namely that this linear layer has as many units
        # as there are vertices
        self.linearise = nn.Linear(_n_hid, n_input)

        if variant in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, variant)(n_input, _n_hid, _nlayers,
                                             dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[
                    variant]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was 
                supplied, options are ['LSTM', 'GRU', 'RNN_TANH' or 
                RNN_RELU']""")
            self.rnn = nn.RNN(n_input, _n_hid, _nlayers,
                              nonlinearity=nonlinearity, dropout=dropout)

            self.init_parameters()

    def forward(self, _input, hidden):
        """

        :param _input:
        Shape: N x T x n_in
        :type _input: FloatTensor or Variable
        :param hidden: (h_t, c_t)
        :type hidden:
        :return:
        :rtype:
        """
        _input = _input.permute(1, 0, 2)  # The dimension representing the
        # index of elements in a sequence (or the tth element of the
        # sequence) is put into the 1st dim (axis 0) and the one
        # representing indices of sequence (the nth sequence) into the 2nd
        # dim (axis 1). Henceforth, `_input` will have a shape of `T x N x
        # n_ins`.

        dropped_out_input = self.drop(_input)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(dropped_out_input, hidden)
        dropped_out_output = self.drop(output)
        linearised = self.linearise(dropped_out_output)

        return linearised, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.variant == 'LSTM':
            return (
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
            # returns (h_t, c_t)
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def predict(self, _input, bsz):
        if isinstance(_input, np.ndarray):
            _input = autograd.Variable(torch.from_numpy(_input).float())

        if isinstance(_input, autograd.Variable):
            if len(_input.size()) == 2:
                _input = _input.view(len(_input), 1, -1)

            sizes = _input.size()
            if sizes[1] == 1:
                _input = _input.expand(sizes[0], bsz, sizes[2])
        else:
            raise TypeError(
                "_input must be a np.ndarray or an autograd.Variable")

        hidden = self.init_hidden(bsz)

        outputs, hidden = self(_input, hidden)

        return outputs[:, 0, :], hidden[:, 0, :]
