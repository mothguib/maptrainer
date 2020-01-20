# -*- coding: utf-8 -*-

import torch
from collections import OrderedDict
import torch.nn as nn
from torch.nn import functional as F

from maptrainer.model.MAPModel import MAPModel
from ..data import INIT_RANGE_BOUND


class ReLUAutoEncoderModel(MAPModel):
    """
    `LinRNNModel`: Multi-layer perceptron model

    Container module standing for an MLP with a linear output layer
    """

    def __init__(self,
                 ninputs: int = -1,
                 nlayers: int = -1,
                 noutputs: int = -1,
                 nhid: int = -1,
                 dropout: float = 0.0,
                 variant: str = None,
                 network: nn.Sequential = None,
                 **kwargs):
        """

        :param ninputs:
        :type ninputs:
        :param nlayers: number of hidden layers
        :type nlayers: int
        :param noutputs:
        :type noutputs:
        :param nhid:
        :type nhid:
        :param dropout:
        :type dropout:
        :param variant:
        :type variant:
        :param network:
        :type network:
        """

        if (ninputs == -1 or nlayers == 0 or nhid == -1) \
                and network is None:
            raise ValueError("(_ninputs, _nalyers) or arch must be assigned.")

        super(ReLUAutoEncoderModel, self).__init__()

        self.ninputs = ninputs if ninputs != -1 \
            else network._modules[0].in_features

        # Divided by 2 because one layer is here decomposed into a
        # linear layer and a sigmoid activation layer.
        self.nlayers = nlayers if nlayers != 0 \
            else len(network._modules) / 2

        self.nhid = nhid if nhid != -1 \
            else network._modules[0].out_features

        self.noutputs = noutputs if noutputs != -1 else self.ninputs

        if network is None:
            self.network = self.construct_hid_network(ninputs=self.ninputs,
                                                      nlayers=self.nlayers,
                                                      nhid=self.nhid)
        else:
            self.network = network

        self.drop = nn.Dropout(dropout)

        self.variant = variant

        # Last layer, which is linear, to convert the hidden output into
        # idlenesses
        self.linearise = nn.Linear(
            list(self.network._modules.items())[-2][1].out_features,
            self.noutputs)  # `-2` because the latest layer's index `-1` is
        # an activation layer

        self.init_out_weights()

    @classmethod
    def construct_hid_network(cls,
                              ninputs,
                              nlayers,
                              nhid):
        """
        Constructs the hidden network i.e. the network without the output layer

        :param activation:
        :type activation:
        :param ninputs:
        :type ninputs:
        :param nlayers:
        :type nlayers:
        :param nhid:
        :type nhid:
        :param noutputs:
        :type noutputs:
        :return:
        :rtype:
        """

        # The first hidden layer maps the input vectors to the hidden
        # dimension space
        layers = [("Lin0", nn.Linear(ninputs, nhid)),
                  ("ReLU0", nn.ReLU()),
                  ("Lin1", nn.Linear(nhid, int(nhid / 2) + 1)),
                  ("ReLU1", nn.ReLU()),
                  ("Lin2", nn.Linear(int(nhid / 2) + 1, int(nhid / 4) + 1)),
                  ("ReLU2", nn.ReLU()),
                  ("Lin3", nn.Linear(int(nhid / 4) + 1, int(nhid / 2) + 1)),
                  ("ReLU3", nn.ReLU()),
                  ("Lin4", nn.Linear(int(nhid / 2) + 1, nhid)),
                  ("ReLU4", nn.ReLU()),
                  ("Lin5", nn.Linear(nhid, ninputs)),
                  ("ReLU5", nn.ReLU())]

        return nn.Sequential(OrderedDict(layers))

    def init_out_weights(self, init_range_bound: int = INIT_RANGE_BOUND):
        """

        Initialises the weights of the output layer
        :return:
        :rtype:
        """

        self.linearise.weight.data.uniform_(-init_range_bound,
                                            init_range_bound)

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
        dropped_out_output = self.drop(output)
        linearised = self.linearise(dropped_out_output)

        return nn.ReLU()(linearised)


class Square(nn.Module):

    def __init__(self, ):
        super(Square, self).__init__()

    def forward(self, input):
        return input ** 2


class SquareRoot(nn.Module):

    def __init__(self, ):
        super(SquareRoot, self).__init__()

    def forward(self, input):
        return torch.sqrt(input)