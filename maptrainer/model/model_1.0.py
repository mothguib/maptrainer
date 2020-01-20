# -*- coding: utf-8 -*-

import random
import time
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from maptrainer.model.MAPModel import MAPModel


class LSTMPathMaker(MAPModel):
    def __init__(self, _vertex_dim, _hidden_dim, _num_layers=1,
                 _bsz=1):
        super(LSTMPathMaker, self).__init__()

        self.vertex_dim = _vertex_dim
        self.hidden_dim = _hidden_dim
        self.num_layers = _num_layers
        self.bsz = _bsz

        self.lstm = nn.LSTM(self.vertex_dim, self.hidden_dim, self.num_layers)

        # The linear layer as projection that maps hidden state space to
        # vertices' space namely that this linear layer has as many units
        # as there are vertices
        self.hidden2vertex = nn.Linear(_hidden_dim, _vertex_dim)

        self.h_t, self.c_t = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we do not have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have
        # this dimensionality.
        # The axes' semantic is (num_layers, bsz, hidden_dim)
        return (autograd.Variable(torch.zeros(self.num_layers,
                                              self.bsz,
                                              self.hidden_dim)),
                autograd.Variable(torch.zeros(self.num_layers,
                                              self.bsz,
                                              self.hidden_dim)))

    def forward(self, paths: autograd.Variable):
        lstm_out, (self.h_t, self.c_t) = self.lstm(paths, (self.h_t, self.c_t))
        vertices_space = self.hidden2vertex(lstm_out.view(len(paths),
                                                          self.bsz, -1))

        # next_vertices = F.log_softmax(vertices_space, dim=2)  # The log
        # softmax is # computed along the dimension 1 (and not along the 0 one)

        next_vertices = F.log_softmax(vertices_space.permute(2, 0, 1)) \
            .permute(1, 2, 0)
        # Permutations of the dimensions so that every slice along the 2nd dim
        # (the dimension of the output features) sums to 1 along every
        # remaining dimensions.

        return next_vertices

    def predict(self, _path, _bsz):
        if isinstance(_path, np.ndarray):
            _path = autograd.Variable(torch.from_numpy(_path).float())

        if isinstance(_path, autograd.Variable):
            if len(_path.size()) == 2:
                _path = _path.view(len(_path), 1, -1)

            sizes = _path.size()
            if sizes[1] == 1:
                _path = _path.expand(sizes[0], _bsz, sizes[2])
        else:
            raise TypeError(
                "_path must be a np.ndarray or an autograd.Variable")

        self.h_t, self.c_t = self.init_hidden()

        return self(_path)[:, 0, :]
