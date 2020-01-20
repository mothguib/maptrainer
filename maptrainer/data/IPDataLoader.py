import numpy as np
import torch

from maptrainer import DATA, DURATION
from maptrainer.data.MAPDataLoader import MAPDataLoader


class IPDataLoader(MAPDataLoader):

    """
    On-vertex idleness path data loader: loads data with individual
    idlenesses as input and real idlenesses as output.

    """

    def __init__(self, nagts: int,
                 tpl: str,
                 nb_folds: int = 1,
                 pre: bool = False,
                 datasrc: str = None,
                 strt: str = None,
                 strt_variant: str = None,
                 rtn: float = 0.8,
                 domain_data_dirpath: str = DATA,
                 duration: int = DURATION,
                 soc_name: str = None,
                 inf_exec_id: int = 0,
                 sup_exec_id: int = 99):
        """

        :param datasrc: data type
        :type datasrc: str
        :param nagts:
        :type nagts:
        :param _map:
        :type _map:
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

        MAPDataLoader.__init__(self, nagts=nagts, tpl=tpl,
                               nb_folds=nb_folds, pre=pre,
                               strt=strt, rtn=rtn, datasrc=datasrc,
                               domain_data_dirpath=domain_data_dirpath,
                               duration=duration, soc_name=soc_name,
                               inf_exec_id=inf_exec_id,
                               sup_exec_id=sup_exec_id,
                               strt_variant=strt_variant)

    def specific_load_data(self):
        """
        Shape: Nb_seq x (seq_length -1) x dim_vector

        :return:
        """

        domain_data = self.load_viidls()
        target_data = self.load_vidls()

        self.domain_data = torch.from_numpy(np.array(domain_data)). \
            float().contiguous()
        self.target_data = torch.from_numpy(np.array(target_data)). \
            float().contiguous()

    @staticmethod
    def label_data(_input: torch.FloatTensor,
                   targets: torch.FloatTensor,
                   evaluation: bool = False) \
            -> (torch.FloatTensor, torch.FloatTensor):
        """
        Returns inputs and labels for the output of the model wrapped into the
        data `Variable` structure.

        :param _input:
        :type _input:
        of inputs
        :param targets: the output vectors not labelled
        :type targets:
        :param cuda:
        :type cuda:
        :param evaluation:
        :type evaluation:
        :return:
        :rtype:
        """

        return _input, targets

    @staticmethod
    def mean(t: torch.FloatTensor, dim: int) -> torch.FloatTensor:
        """

        :param t: the tensor whose the mean will be computed for each
        element on the dimension `dim` over the other dimensions
        :type t:
        :param dim: dimension to keep
        :type dim:

        :return:
        :rtype:
        """

        mean = t

        for d in range(len(t.size())):

            offset = 0
            if d != dim:
                mean = torch.mean(mean, offset)
            else:
                offset += 1

        return mean

    def specific_load_pre_data(self):
        pass

    @staticmethod
    def reshape_output(output: torch.Tensor):
        """
        Reshapes output data for its use in the `criterion` function

        :param output:
        :type output:
        :return:
        """

        return output.view(-1)

    @staticmethod
    def reshape_labels(labels: torch.Tensor):
        """
        Reshapes label data for its use in the `criterion` function

        :param labels:
        :type labels:
        :return:
        """

        return labels.view(-1)
