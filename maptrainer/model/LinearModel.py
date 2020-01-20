# -*- coding: utf-8 -*-

from collections import OrderedDict
import torch
import torch.nn as nn

from maptrainer.model.MAPModel import MAPModel
from ..data import INIT_RANGE_BOUND


class LinearModel(MAPModel):

    def __init__(self,
                 ninputs: int = -1,
                 nlayers: int = -1,
                 nhid: int = -1,
                 noutputs: int = -1,
                 dropout: float = 0.0,
                 variant: str = None,
                 **kwargs):

        super(LinearModel, self).__init__()

        self.ninputs = ninputs

        self.noutputs = noutputs if noutputs != -1 else self.ninputs

        self.drop = nn.Dropout(dropout)

        self.variant = variant

        bias = False if variant == "IdentityWeightsNoBias" else True

        # Last layer, which is linear, to convert probabilities into idlenesses
        self.network = nn.Linear(self.ninputs, self.noutputs, bias=bias)

        if variant.startswith("IdentityWeights"):
            self.init_parameters()

    def init_parameters(self, init_range_bound: int = INIT_RANGE_BOUND):
        """

        Initialises the weights of the output layer
        :return:
        :rtype:
        """

        self.network.weight.data = torch.zeros(self.noutputs, self.ninputs)

        for i in range(self.noutputs):
            self.network.weight.data[i][i] = 1

        if self.variant == "IdentityWeightsNoBias":
            for i in range(self.noutputs):
                self.network.bias.data[i] = 0

    def forward(self, _input):
        """

        :param _input:
        Shape: N x T x n_in
        :type _input: FloatTensor or Variable
        :return:
        :rtype:
        """

        dropped_out_input = self.drop(_input)
        output = self.network(dropped_out_input)

        # TDLT
        # print("DBG: ", self.network.bias.data)

        return output
