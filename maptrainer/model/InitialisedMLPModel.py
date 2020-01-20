# -*- coding: utf-8 -*-

import numpy as np
import torch
from collections import OrderedDict
import torch.nn as nn
from torch.nn import functional as F

from maptrainer.model.MAPModel import MAPModel
from ..data import INIT_RANGE_BOUND


class InitialisedMLPModel(MAPModel):
    """
    `InitialisedMLPModel`: Multi-layer perceptron model where the parameters
    are initialised to specific values.

    Container module standing for an MLP with a linear output layer initialised
    to specific values.
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

        if (ninputs == -1 or nlayers == -1 or nhid == -1) \
                and network is None:
            raise ValueError("(_ninputs, _nalyers) or arch must be assigned.")

        super(InitialisedMLPModel, self).__init__()

        self.ninputs = ninputs if ninputs != -1 \
            else network._modules[0].in_features

        # Divided by 2 because one layer is here decomposed into a
        # linear layer and a sigmoid layer.
        self.nlayers = nlayers if nlayers != -1 \
            else len(network._modules) / 2

        self.nhid = nhid if nhid != -1 \
            else network._modules[0].out_features

        self.noutputs = noutputs if noutputs != -1 else self.ninputs

        # The activation module's name
        self.activation = variant if variant is not None else "Sigmoid"

        if network is None:
            self.network = self.construct_network(ninputs=self.ninputs,
                                                  nlayers=self.nlayers,
                                                  nhid=self.nhid,
                                                  noutputs=self.noutputs,
                                                  activation=self.activation)
        else:
            self.network = network

        self.drop = nn.Dropout(dropout)

        self.variant = variant

        # Last layer, which is linear, to convert probabilities into idlenesses
        self.linearise = nn.Linear(
            list(self.network._modules.items())[-2][1].out_features,
            self.noutputs)

        self.init_parameters()

    @classmethod
    def construct_network(cls,
                          ninputs,
                          nlayers,
                          nhid,
                          noutputs,
                          activation: str,
                          **kwargs):
        """
        Constructs the network without the output layer.

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

        threshold = kwargs.pop("threshold", 200)
        value = kwargs.pop("value", 0)

        # The first hidden layer maps the input vectors to the hidden
        # dimension space
        layers = [("Lin0", nn.Linear(ninputs, nhid))]

        layers += \
            [("{}0".format(activation),
              cls.get_activation_layer(activation,
                                       threshold=threshold,
                                       value=value))]

        for l in range(1, nlayers):
            layers += [("Lin{}".format(l), nn.Linear(nhid, nhid))]
            layers += \
                [("{}{}".format(activation, l),
                  cls.get_activation_layer(activation,
                                           threshold=threshold,
                                           value=value))]

        return nn.Sequential(OrderedDict(layers))

    @staticmethod
    def get_activation_layer(activation: str, **kwargs):
        if activation == "Threshold":
            threshold = kwargs.pop("threshold", 200)
            value = kwargs.pop("value", 200)
            return MaximumThreshold(threshold, value)
        else:
            return getattr(nn, activation)()

    def init_parameters(self):
        """

        Initialises the weights of the output layer
        :return:
        :rtype:
        """

        self.network[0].weight.data = torch.zeros(50, 50)
        self.linearise.weight.data = torch.zeros(50, 50)
        self.network[0].bias.data = torch.zeros(50)
        self.linearise.bias.data = torch.zeros(50)

        e_ = 292
        w = 1
        u = -e_ * w

        # Max idl
        e1 = 584
        e2 = 0
        s1 = 584
        s2 = 0

        a = (s1 - s2) * \
            (1 + np.exp(-w * (e1 - e_))) * (1 + np.exp(-w * (
                e2 - e_))) / \
            (np.exp(-w * (e2 - e_))) - np.exp(-w * (e1 - e_))

        # a = 4 / w

        b = s1 - (s1 - s2) * \
            (1 + np.exp(-w * (e2 - e_))) / \
            (np.exp(-w * (e2 - e_))) - np.exp(-w * (e1 - e_))

        # b = e_ - a / 2

        b2 = s2 - (s1 - s2) * \
            (1.0 + np.exp(-w * (e1 - e_))) / \
            (np.exp(-w * (e2 - e_))) - np.exp(-w * (e1 - e_))

        print("DBG: w:{}, u:{}, a:{}, b:{}, b2:{}".format(w, u, a, b, b2))

        for x in [0, 100, 200, 292, 300, 400, 500, 584]:
            s = b + a / (1 + np.exp(-(w * x + u)))
            print("DBG: x:{}, s:{}".format(x, s))

        for i in range(50):
            self.network[0].weight.data[i][i] = float(w)
            self.linearise.weight.data[i][i] = float(a)
            self.network[0].bias.data[i] = float(u)
            self.linearise.bias.data[i] = float(b)

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

        return linearised


class MaximumThreshold(nn.Module):
    r"""Thresholds each element of the input Tensor

    Threshold is defined as:

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x <= \text{threshold} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    Args:
        threshold: The value to threshold at
        value: The value to replace with
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.Threshold(0.1, 20)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, threshold, value, inplace=False):
        super(MaximumThreshold, self).__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace

    def forward(self, input):
        return -1 * F.threshold(-1 * input, -1 * self.threshold,
                                -1 * self.value, self.inplace)

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'threshold={}, value={}{}'.format(
            self.threshold, self.value, inplace_str
        )
