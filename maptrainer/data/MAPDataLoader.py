# coding: utf-8

import json
import numpy as np
import torch
from abc import ABC, abstractmethod
from torch.autograd import Variable

from maptrainer import PCKGROOT, DATA, DURATION, pathformatter as pf
from maptrainer.data.DataLoader import DataLoader

MAPS = PCKGROOT + "../../../Pytrol-Resources/maps/json_bin/"


class MAPDataLoader(DataLoader, ABC):

    def __init__(self,
                 nagts: int,
                 tpl: str,
                 nb_folds: int = 1,
                 pre: bool = False,
                 datasrc: str = None,
                 strt: str = None,
                 strt_variant: str = None,
                 rtn: float = 0.8,
                 domain_data_dirpath: str = None,
                 duration: int = DURATION,
                 soc_name: str = None,
                 inf_exec_id: int = 0,
                 sup_exec_id: int = 299):
        """
        
        :param datasrc: data type
        :type datasrc: str
        :param nagts:
        :type nagts:
        :param tpl:
        :type tpl:
        :param nb_folds:
        :type nb_folds:
        :param pre:
        :type pre:
        :param strt:
        :type strt:
        :param rtn: rate of train data over the whole dataset if the number of
        folds is 1.
        :type rtn:
        :param domain_data_dirpath: path of data
        :type domain_data_dirpath:
        :param duration: duration of executions to load.
        :type duration:
        :param soc_name:
        :type soc_name:
        :param inf_exec_id:
        :type inf_exec_id:
        :param sup_exec_id:
        :type sup_exec_id:
        """

        if datasrc is not None:
            if strt is None:
                strt = datasrc.split('_')[0]
            if strt_variant is None:
                strt_variant = datasrc.split('_')[1]
        else:
            if strt is None:
                strt = "hpcc"
            if strt_variant is None:
                strt_variant = "0.5"

            datasrc = "{}_{}".format(strt, strt_variant)

        if domain_data_dirpath is None:
            domain_data_dirpath = DATA

        DataLoader.__init__(self, domain_data_dirpath, nb_folds, pre, rtn)

        self.nagts = nagts
        self.m = tpl
        self.strt = strt
        self.strt_variant = strt_variant

        self.sn = "soc_{}_{}".format(strt, nagts) if soc_name is None \
            else soc_name

        self.datasrc = "{}_{}".format(self.strt, self.strt_variant) \
            if datasrc is None else datasrc

        self.duration = duration

        # Rate of training data in the data set when the number of folds is 1
        self.rtn = rtn if not pre else 1.0

        self.pre = pre

        self.inf_exec_id = inf_exec_id
        self.sup_exec_id = sup_exec_id

    def load_binpaths(self) -> torch.FloatTensor:

        return torch.from_numpy(np.array(self.load_paths("bin"),
                                         dtype=np.uint8)).float()

    def load_viidls(self) -> torch.FloatTensor:
        return torch.from_numpy(np.array(self.load_paths("viidls"),
                                         dtype=np.int32)).float()

    def load_vidls(self) -> torch.FloatTensor:
        return torch.from_numpy(np.array(self.load_paths("vidls"),
                                         dtype=np.int32)).float()

    def load_veidls(self) -> torch.FloatTensor:
        return torch.from_numpy(np.array(self.load_paths("veidls"),
                                         dtype=np.float32)).float()

    def load_paths(self, cext: str) -> torch.FloatTensor:
        """
        Load time series of agents in a tensor of shape `N_agts x T_v
        x N_v` from the log type corresponding to the extension component
        `cext` defining the type of log and thereby the type of time series;
        for example `bin` for the binary position series or `viidls` for
        the on-vertex individual idleness series.

        :param cext:
        :type cext:
        :return:
        :rtype:
        """

        data = [[None]] * (self.sup_exec_id - self.inf_exec_id + 1)

        config_name = "{}-{}-{}-{}-{}".format(self.datasrc,
                                              self.m,
                                              self.sn,
                                              str(self.nagts),
                                              str(self.duration))

        for i in range(self.inf_exec_id, self.sup_exec_id + 1):
            exec_name = config_name + '-' + str(i) + '.log.' + cext + '.json'

            path = "{}/{}/{}/{}/{}/{}".format(self.domain_data_dirpath
                                              + "/logs-" + cext + '/',
                                              self.m,
                                              self.datasrc,
                                              str(self.nagts),
                                              str(self.duration),
                                              exec_name)

            with open(path, 'r') as s:
                data[i] = json.load(s)

        min_len = min([len(seq) for exc in data for seq in exc])

        # `i` stands for the `i`th current scenario's execution
        for i in range(len(data)):
            # `j` stands for the `j`th agent
            for j in range(len(data[i])):
                data[i][j] = data[i][j][0:min_len]

        data = np.array(data, dtype=np.float32)
        sd = data.shape
        data = data.reshape(sd[0] * sd[1], sd[2], sd[3])

        return data

    @staticmethod
    def load_topology(tn: str) -> np.ndarray:
        """
        Returns the adjacent matrix of the topology `tn`

        :param tn: topology name
        :type tn:

        :return:
        """

        with open(pf.topology_path(tn), 'r') as s:
            bin_edges = np.array(json.load(s), dtype=np.uint8)

        nvtcs = len(bin_edges[0][0])

        # The topology's adjacent matrix
        tpl = np.zeros((nvtcs, nvtcs), dtype=np.uint8)

        for e in bin_edges:
            s = np.where(e[0] == 1)[0][0]
            t = np.where(e[1] == 1)[0][0]

            tpl[s][t] = 1

        return tpl

    @staticmethod
    @abstractmethod
    def label_data(_input: torch.FloatTensor, targets: torch.FloatTensor,
                   evaluation: bool = False) \
            -> (Variable, Variable):
        pass

    @abstractmethod
    def specific_load_data(self):
        pass
