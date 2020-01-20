# -*- coding: utf-8 -*-

import torch

from maptrainer import misc
from maptrainer.model import RNNModel
from ..data import INIT_RANGE_BOUND


class MAPLSTMModel(RNNModel):
    """
    `MAPLSTMModel`: RNN model initialised and updated during training for MAP

    No `check-pre-exists` and `pre` options for this model.

    """

    def __init__(self,
                 ninputs,
                 nhid,
                 nlayers,
                 noutputs=-1,
                 dropout=0.0,
                 graph=None,
                 variant: str = "LSTM",
                 **kwargs):

        # Sigmoid bound: value used for the sigmoid and tanh activations to
        # output values close to its bounds
        self.sigbound = 6

        self.lin_rescaling = self.sigbound

        self.logit_0_9 = 2.197224577336

        super(MAPLSTMModel, self).__init__(ninputs=ninputs,
                                           nhid=nhid,
                                           nlayers=nlayers,
                                           variant="LSTM",
                                           noutputs=noutputs,
                                           dropout=dropout,
                                           graph=graph)

    def init_parameters(self, init_range_bound: int = INIT_RANGE_BOUND):

        super(MAPLSTMModel, self).init_parameters()

        # # Named parameters
        # nps = self.state_dict()
        #
        # for np in self.named_parameters():
        #     # Each parameters' tensor is detached to apply the in-place
        #     # operation `Tensor.div_`
        #     p = np[1].detach()
        #     p.div_(10000) # Why to divide by 10 000?

        self.init_last_lstm_layer()

        if self.nlayers > 1:
            self.init_intermediate_lstm_layers()

        self.init_last_layer()

    def init_biases(self, l):

        # Named parameters
        nps = self.state_dict()

        bias_ih_i = \
            nps["rnn.bias_ih_l" + str(l)][:self.nhid].zero_()  # Biases of
        # $o_t$: `b_io`, i.e. $b^{ox}$.

        bias_hh_i = \
            nps["rnn.bias_hh_l" + str(l)][:self.nhid].zero_()  # Biases of
        # $o_t$: `b_ho`, i.e. $b^{oh}$.

        bias_ih_f = \
            nps["rnn.bias_ih_l" + str(l)][self.nhid:2 * self.nhid].zero_()
        # Biases of $o_t$: `b_io`, i.e. $b^{ox}$.

        bias_hh_f = \
            nps["rnn.bias_hh_l" + str(l)][self.nhid:2 * self.nhid].zero_()
        # Biases of $o_t$: `b_ho`, i.e. $b^{oh}$.

        bias_ih_g = \
            nps["rnn.bias_ih_l" + str(l)][-2 * self.nhid:-self.nhid].zero_()
        # Biases of $g_t$ ($c_t$): `b_ig`, i.e. $b^{gx}$.

        bias_hh_g = \
            nps["rnn.bias_hh_l" + str(l)][-2 * self.nhid:-self.nhid].zero_()
        # Biases of $g_t$ ($c_t$): `b_hg`, i.e. $b^{gh}$

        bias_ih_o = \
            nps["rnn.bias_ih_l" + str(l)][-self.nhid:].zero_()  # Biases of
        # $o_t$: `b_io`, i.e. $b^{ox}$.

        bias_hh_o = \
            nps["rnn.bias_hh_l" + str(l)][-self.nhid:].zero_()  # Biases of
        # $o_t$: `b_ho`, i.e. $b^{oh}$.

        return bias_ih_i, bias_hh_i, bias_ih_f, bias_hh_f, bias_ih_g, \
               bias_hh_g, bias_ih_o, bias_hh_o

    def init_last_lstm_layer(self):
        """

        :return:
        :rtype:
        """

        # The last layer
        l = self.nlayers - 1

        weights_ih_i, weights_hh_i, weights_ih_f, weights_hh_f, \
        weights_ih_o, weights_hh_o, weights_ih_g, weights_hh_g, \
        bias_ih_o, bias_hh_o, bias_ih_g, bias_hh_g = self.init_lstm_layer(l)

        for v in range(len(self.graph)):
            nghs = misc.neighbours(v, self.graph)

            weights_ih_i.t()[v][nghs] = torch.ones(len(nghs)) * self.sigbound
            # Setting the neighbours of `v` in $W^{ix}$ to `self.sigbound`
            # for the sigmoid to output a value close to `1` for this node

            weights_ih_f.t()[v] = torch.zeros(self.nhid)
            weights_ih_f.t()[v][v] = self.sigbound
            # weights_ih_f.t()[v] = torch.ones(self.ninputs) * self.logit_0_9
            # Setting all the parameters of $W^{fx}$ to `self.logit_0_9`
            # for the sigmoid to output a value close to `0.9` to forget
            # what is stored in the cell state to an extent of 10% at each
            # time `t`

            weights_ih_g.t()[v][nghs] = torch.ones(len(nghs)) * self.sigbound
            # Setting the neighbours of `v` in $W^{gx}$ to `self.sigbound` for
            # $tanh$ to output a value close to `1` for this node

            weights_ih_g.t()[v][v] = -self.sigbound

            weights_ih_o.t()[v][nghs] = torch.ones(len(nghs)) * self.sigbound
            # Setting the neighbours of `v` in $W^{ox}$ to `self.sigbound` for
            # the sigmoid to output a value close to `1` for this node

            bias_ih_o[nghs] = torch.ones(len(nghs))

    def init_intermediate_lstm_layers(self):
        for l in range(self.nlayers - 1):
            self.init_intermediate_lstm_layer(l)

    def init_intermediate_lstm_layer(self, l):

        weights_ih_i, weights_hh_i, weights_ih_f, weights_hh_f, \
        weights_ih_o, weights_hh_o, weights_ih_g, weights_hh_g, \
        bias_ih_o, bias_hh_o, bias_ih_g, bias_hh_g = self.init_lstm_layer(l)

        for v in range(len(self.graph)):
            weights_ih_i[v] = torch.ones(self.nhid) * -self.sigbound
            weights_ih_i.t()[v][v] = self.sigbound  # Setting the node `v` in
            # $W^{ix}$ to `self.sigbound` for the sigmoid to output a value
            # close to `1` for this node

            weights_ih_f[v] = torch.ones(self.nhid) * -self.sigbound
            weights_ih_f.t()[v][v] = self.sigbound  # Setting the node `v` in
            # $W^{fx}$ to `self.sigbound` for the sigmoid to output a value
            # close to `1` for this node

            weights_ih_g[v][v] = self.sigbound  # Setting the node `v` in
            # $W^{gx}$ to `self.sigbound` for $tanh$ to output a value close
            # to `1` for this node

            weights_ih_o[v] = torch.ones(self.nhid) * -self.sigbound
            weights_ih_o[v][v] = self.sigbound  # Setting the node `v `to
            # `self.sigbound` in $W^{ox}$ for the sigmoid activation to
            # output a value close to `1`

            # bias_ih_o[v] = 1

    def init_lstm_layer(self, l: int):
        """

        :return:
        :rtype:
        """

        # Named parameters
        nps = self.state_dict()

        weights_ih_i = \
            nps["rnn.weight_ih_l" + str(l)][0:self.nhid]  # $W^{ix}$: weight
        # `ih` of the gate `i_t`

        weights_ih_f = \
            nps["rnn.weight_ih_l" + str(l)][self.nhid:2 * self.nhid]
        # $W^{fx}$: weight `ih` of the gate `f_t`

        weights_ih_g = \
            nps["rnn.weight_ih_l" + str(l)][-2 * self.nhid:-self.nhid]
        # $W^{gx}$: weight `ih` of the gate `g_t` (`c_t`)

        weights_ih_o = \
            nps["rnn.weight_ih_l" + str(l)][-self.nhid:]  # $W^{ox}$: weight
        # `ih` of the gate `o_t`

        weights_hh_i = \
            nps["rnn.weight_hh_l" + str(l)][0:self.nhid]  # $W^{ih}$: weight
        # `ih` of the gate `i_t`

        weights_hh_f = \
            nps["rnn.weight_hh_l" + str(l)][self.nhid:2 * self.nhid]
        # $W^{fh}$: weight `ih` of the gate `f_t`

        weights_hh_g = \
            nps["rnn.weight_hh_l" + str(l)][-2 * self.nhid:-self.nhid]
        # $W^{gh}$: weight `hh` of the gate `g_t` (`c_t`)

        weights_hh_o = \
            nps["rnn.weight_hh_l" + str(l)][-self.nhid:]
        # $W^{oh}$: weight `hh` of the gate `o_t`

        # Number of inputs of the current layer/block
        ninputs = self._modules["rnn"].hidden_size

        for i in range(len(self.graph)):
            weights_ih_i[i] = torch.zeros(ninputs)
            # weights_ih_i[i] = torch.ones(ninputs) * -self.sigbound  # Setting
            # to `-self.sigbound` the nodes which are not neighbours of `i`
            # in $W^{ix}$ to have the sigmoid activation outputting a value
            # close to `0`

            weights_hh_i[i] = torch.zeros(self.nhid)  # Zeroing $W^{ih}$

            weights_ih_f[i] = torch.ones(ninputs) * -self.sigbound  # Setting
            # to `-self.sigbound` the nodes which are not neighbours of `i`
            # in $W^{fx}$ to have the sigmoid activation outputting a value
            # close to `0`

            weights_hh_f[i] = torch.zeros(self.nhid)  # Zeroing $W^{fh}$

            # weights_ih_o[i] = torch.zeros(self.nhid)
            weights_ih_o[i] = torch.ones(ninputs) * -self.sigbound  # Setting
            # to `-self.sigbound` the nodes which are not neighbours of `i`
            # in $W^{ox}$ to have the sigmoid activation outputting a value
            # close to `0`

            weights_hh_o[i] = torch.zeros(self.nhid)  # Zeroing $W^{oh}$

            # weights_ih_g[i] = torch.zeros(self.nhid)
            # weights_ih_g[i][i] = -self.sigbound
            weights_ih_g[i] = torch.zeros(self.nhid)

            weights_hh_g[i] = torch.zeros(self.nhid)  # Zeroing $W^{gh}$

        bias_ih_i, bias_hh_i, bias_ih_f, bias_hh_f, bias_ih_g, bias_hh_g, \
        bias_ih_o, bias_hh_o = self.init_biases(l)

        return weights_ih_i, weights_hh_i, weights_ih_f, weights_hh_f, \
               weights_ih_o, weights_hh_o, weights_ih_g, weights_hh_g, \
               bias_ih_o, bias_hh_o, bias_ih_g, bias_hh_g

    def init_last_layer(self):
        self.linearise.weight.data.zero_()

        for n in range(len(self.linearise.weight.data)):
            self.linearise.weight.data[n][n] = self.lin_rescaling

        self.linearise.bias.data.zero_()

#    def callback(self):
