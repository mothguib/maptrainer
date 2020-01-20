# -*- coding: utf-8 -*-

import torch
from collections import OrderedDict
import torch.nn as nn
from torch.nn import functional as F

from maptrainer.model.MAPModel import MAPModel
from ..data import INIT_RANGE_BOUND


class RLOModel(MAPModel):
    """
    `RLOModel`: ReLU Output model
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

        super(RLOModel, self).__init__()

        self.ninputs = ninputs if ninputs != -1 \
            else network._modules[0].in_features

        # Divided by 2 because one layer is here decomposed into a
        # linear layer and a sigmoid activation layer.
        self.nlayers = nlayers if nlayers != 0 \
            else len(network._modules) / 2

        self.nhid = nhid if nhid != -1 \
            else network._modules[0].out_features

        self.noutputs = noutputs if noutputs != -1 else self.ninputs

        # The activation module's name
        self.activation = variant if variant is not None else "Sigmoid"

        if network is None:
            self.network = self.construct_hid_network(ninputs=self.ninputs,
                                                      nlayers=self.nlayers,
                                                      nhid=self.nhid,
                                                      activation=self. \
                                                      activation)
        else:
            self.network = network

        self.drop = nn.Dropout(dropout)

        self.variant = variant

        self.lin_rectify = nn.Sequential(OrderedDict([("OutputLin", nn.Linear(
            list(self.network._modules.items())[-2][1].out_features,
            self.noutputs)), ("OutputReLU", nn.ReLU())]))  # `-2` because the
        # latest layer's index `-1` is an activation layer

        self.init_parameters()

    @classmethod
    def construct_hid_network(cls,
                              ninputs,
                              nlayers,
                              nhid,
                              activation,
                              **kwargs):
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
        :return:
        :rtype:
        """

        threshold = kwargs.pop("threshold", 200)
        value = kwargs.pop("value", 0)

        # TDP
        # print("Activation: ", activation)

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

    def init_parameters(self, init_range_bound: int = INIT_RANGE_BOUND):
        """

        Initialises the weights of the output layer
        :return:
        :rtype:
        """

        self.lin_rectify[0].weight.data.uniform_(-init_range_bound,
                                                 init_range_bound)

    @staticmethod
    def get_activation_layer(activation: str, **kwargs):
        if activation == "Threshold":
            threshold = kwargs.pop("threshold", 200)
            value = kwargs.pop("value", 200)
            return MaximumThreshold(threshold, value)
        elif activation == "Square":
            return Square()
        elif activation == "SquareRoot":
            return SquareRoot()
        else:
            return getattr(nn, activation)()

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
        lin_rectified = self.lin_rectify(dropped_out_output)

        return lin_rectified


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
