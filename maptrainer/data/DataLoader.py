# -*- coding: utf-8 -*-

import math

import torch
from abc import ABC, abstractmethod

from torch.autograd import Variable


def set_nb_folds(data, k):
    while len(data) % k > 0:
        k -= 1
    return k


class DataLoader(ABC):
    def __init__(self,
                 domain_data_dirpath: str,
                 nb_folds: int = 1,
                 pre: bool = False,
                 rtn: float = 0.8):
        """

        :param nb_folds:
        :type nb_folds:
        :param path:
        :type path:
        :param rtn: rate of train data over the whole dataset if the number of
        folds is 1.
        :type rtn:
        :param soc_name:
        :type soc_name:
        """

        self.domain_data_dirpath = domain_data_dirpath

        # Raw data as input of model
        self.domain_data = torch.FloatTensor()
        # Target data, as output of model
        self.target_data = torch.FloatTensor()

        self.nb_folds = nb_folds
        # Current fold
        self.fold = 1

        # Rate of training data in the data set when the number of folds is 1
        self.rtn = rtn if not pre else 1.0

        self.pre = pre

    def next_fold(self) -> ((torch.FloatTensor, torch.FloatTensor),
                            (torch.FloatTensor, torch.FloatTensor)):

        # `dtrain`: Domain training data
        # `ttrain`: Target training data
        # `dvalid`: Domain validation data
        # `tvalid`: Target validation data

        if self.nb_folds == 1:
            if self.pre:
                dtrain = self.domain_data
                ttrain = self.target_data

                dvalid = dtrain
                tvalid = ttrain

            # Outside of the scope of the pre-tranning framework, if there is
            # only one fold, we arbitrarly split the dataset with self.rtn %
            #  for the training set.
            # If self.rtn is equal to 1.0 then the valid set is empty.
            else:
                nb_train = math.floor(len(self.domain_data) * self.rtn)
                dtrain = self.domain_data[:nb_train]
                ttrain = self.target_data[:nb_train]

                dvalid = \
                    self.domain_data[nb_train:len(self.domain_data)] \
                        if self.rtn < 1.0 else torch.FloatTensor()
                tvalid = \
                    self.target_data[nb_train:len(self.target_data)] \
                        if self.rtn < 1.0 else torch.FloatTensor()
        else:
            ss_lgth = len(self.domain_data) // self.nb_folds
            if self.fold < self.nb_folds:
                if self.fold > 1:
                    dtrain = self.domain_data[0:(self.fold - 1) * ss_lgth]
                    dtrain = torch.cat((dtrain,
                                        self.domain_data[self.fold * ss_lgth:
                                                     len(self.domain_data)]))
                    ttrain = self.target_data[0:(self.fold - 1) * ss_lgth]
                    ttrain = torch.cat((ttrain,
                                        self.target_data[self.fold * ss_lgth:
                                                     len(self.target_data)]))
                else:
                    dtrain = self.domain_data[ss_lgth:len(self.domain_data)]
                    ttrain = self.target_data[ss_lgth:len(self.target_data)]

                dvalid = self.domain_data[(self.fold - 1)
                                      * ss_lgth:self.fold * ss_lgth]
                tvalid = self.target_data[(self.fold - 1)
                                    * ss_lgth:self.fold * ss_lgth]
            else:
                dtrain = self.domain_data[0:(self.fold - 1) * ss_lgth]
                ttrain = self.target_data[0:(self.fold - 1) * ss_lgth]

                dvalid = self.domain_data[(self.fold - 1)
                                      * ss_lgth:len(self.domain_data)]
                tvalid = self.target_data[(self.fold - 1)
                                    * ss_lgth:len(self.target_data)]

        train = (dtrain, ttrain)
        valid = (dvalid, tvalid)

        self.fold = self.fold + 1 if self.fold < self.nb_folds else 1

        return train, valid

    @abstractmethod
    def specific_load_data(self):
        pass

    def load_data(self):
        self.specific_load_data()
        self.set_nb_folds()

    @abstractmethod
    def specific_load_pre_data(self):
        pass

    def load_pre_data(self):
        self.specific_load_pre_data()
        self.set_nb_folds()

    def set_nb_folds(self):
        self.nb_folds = set_nb_folds(self.domain_data, self.nb_folds)

    @staticmethod
    @abstractmethod
    def label_data(input_: torch.Tensor,
                   targets: torch.Tensor,
                   evaluation: bool = False) \
            -> (Variable, Variable):
        pass

    @staticmethod
    @abstractmethod
    def reshape_output(output: torch.Tensor):
        """
        Reshapes output data for its use in the `criterion` function

        :param output:
        :type output:
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def reshape_labels(labels: torch.Tensor):
        """
        Reshapes label data for its use in the `criterion` function

        :param labels:
        :type labels:
        :return:
        """
        pass
