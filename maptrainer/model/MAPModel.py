# -*- coding: utf-8 -*-
from abc import abstractmethod

import torch.nn as nn

from maptrainer.data import INIT_RANGE_BOUND


class MAPModel(nn.Module):

    def __init__(self, graph=None):
        super(MAPModel, self).__init__()

        self.graph = graph

    def forward(self, *input_):
        super(MAPModel, self).forward(input_)

    def callback(self):
        pass

    @abstractmethod
    def init_parameters(self, init_range_bound: int = INIT_RANGE_BOUND):
        """

        Initialises the weights of the output layer
        :return:
        :rtype:
        """

        pass
