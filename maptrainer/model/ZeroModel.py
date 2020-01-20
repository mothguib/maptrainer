# -*- coding: utf-8 -*-

import json
import torch
import torch.nn as nn
from torch import autograd

from maptrainer import DATA
from maptrainer.model.MAPModel import MAPModel


class ZeroModel(MAPModel):

    def __init__(self,
                 tpl: str,
                 nagts: int, variant='',
                 graph=None,
                 **kwargs):

        super(ZeroModel, self).__init__(graph=graph)

        self.map = tpl
        self.nagts = nagts
        self.variant = variant

    def forward(self, input_: torch.Tensor):
            torch.zeros(input_.size())


