# -*- coding: utf-8 -*-

import json
import torch

from maptrainer.model.MAPModel import MAPModel
from maptrainer import DATA


class MeanModel(MAPModel):

    def __init__(self, tpl: str,
                 nagts: int,
                 variant='',
                 graph=None,
                 dirpath_data: str = DATA,
                 **kwargs):

        super(MeanModel, self).__init__(graph=graph)

        self.map = tpl
        self.nagts = nagts
        self.variant = variant
        self.idls_means = \
            torch.FloatTensor(self.load_idlenesses_means(dirpath_data))

    def forward(self, input: torch.FloatTensor):
            self.idls_means.expand(input.size()).contiguous()

    def load_idlenesses_means(self, dirpath_data) -> list:
        # Current configuration's means' file path
        config_means_fp = "{}/means/hpcc-{}-{}.means.json". \
            format(dirpath_data, self.map, self.nagts)

        with open(config_means_fp) as s:
            return json.load(s)
