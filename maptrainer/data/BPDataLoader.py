import json

import numpy as np
import torch

from maptrainer import DATA, DURATION
from maptrainer import misc
from maptrainer import MAPS
from maptrainer.data.MAPDataLoader import MAPDataLoader


class BPDataLoader(MAPDataLoader):
    """
    Binary position data loader: loads data with binary positions as input
    and output of the used model.

    """

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
                 sup_exec_id: int = 99):
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
        :param duration: duration of executions to load.
        :type duration:
        :param soc_name:
        :type soc_name:
        :param inf_exec_id:
        :type inf_exec_id:
        :param sup_exec_id:
        :type sup_exec_id:
        """

        MAPDataLoader.__init__(self, nagts=nagts, tpl=tpl, nb_folds=nb_folds,
                               pre=pre, strt=strt, rtn=rtn,
                               domain_data_dirpath=domain_data_dirpath,
                               duration=duration, soc_name=soc_name,
                               inf_exec_id=inf_exec_id,
                               sup_exec_id=sup_exec_id,
                               strt_variant=strt_variant, datasrc=datasrc)

    def specific_load_pre_data(self):
        """
        Nb_seq x (seq_length -1) x dim_vector

        :return:
        """

        with open(MAPS + self.m + ".bin.json", 'r') as s:
            data = json.load(s)

        self.domain_data = torch.from_numpy(np.array(data)[:, :-1]).float()
        # `:-1` because the last element of the sequence is the target of the
        # penultimate element. Type: torch.FloatTensor

        # Target data
        self.target_data = torch.from_numpy(np.array(data)[:, 1:]).float()
        # `1:` because the first element of the sequence is the input for
        # the next one as output. Type: torch.FloatTensor

    def specific_load_data(self):
        """
        Nb_seq x (seq_length -1) x dim_vector

        :return:
        """

        data = self.load_binpaths()

        self.domain_data = torch.from_numpy(np.array(data)[:, :-1]).float() \
            .contiguous()
        self.target_data = torch.from_numpy(np.array(data)[:, 1:]).float() \
            .contiguous()
    
    @staticmethod
    def label_data(input_: torch.FloatTensor,
                   targets: torch.FloatTensor,
                   cuda: bool = True) \
            -> (torch.FloatTensor, torch.FloatTensor):
        """
        Returns inputs and labels for the output of the model
    
        :param input_: 
        :type input_: 
            shape: (Batch size, BPTT, n_in)
        :param targets: the output vectors not labelled
        :type targets:
            shape: (Batch size, BPTT, n_out)
        :param cuda:
        :type cuda:

        :return: the inputs, and labels of `targets`
        :rtype: torch.FloatTensor, torch.FloatTensor
            shape: (Batch size, BPTT, n_in), (Batch size, BPTT)
        """

        # Conversion to numpy
        targets = targets.cpu().numpy()

        # Creation of targets from the structure `targets`
        labelled = []

        for s_n in targets:
            # Labels of the current sequence s_n (the nth sequence)
            labelled_data_n = []

            for s_nt in s_n:
                # Labels over the time for the current sequence s_n
                # labelled_data_n += np.where(s_nt == 1)[0].tolist()
                labelled_data_n += [np.argmax(s_nt)]

            labelled += [labelled_data_n]

        labelled = torch.LongTensor(labelled)

        return input_, labelled

    @staticmethod
    def reshape_output(output: torch.Tensor):
        """
        Reshapes output data for its use in the `criterion` function

        :param output:
        :type output:
        :return:
        """

        return output.view(-1, output.size()[-1])

    @staticmethod
    def reshape_labels(labels: torch.Tensor):
        """
        Reshapes label data for its use in the `criterion` function

        :param labels:
        :type labels:
        :return:
        """

        return labels.view(-1)
