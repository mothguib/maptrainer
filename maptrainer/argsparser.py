# -*- coding: utf-8 -*-

import time
import argparse

from maptrainer import DATA, SAVES, RES, DURATION, DATASRC


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch LSTM Model for "
                                                 "predicting a path in the "
                                                 "context of the MAP")

    parser.add_argument("--data", type=str, default=DATA,
                        help="location of data")
    parser.add_argument("--nbcycles", type=str, default=DURATION,
                        help="Length of learnt series")
    parser.add_argument("--model-type", type=str, default="",
                        help="type of model (RNN, etc.)")
    parser.add_argument("--model-variant", type=str, default="",
                        help="type of recurrent net (RNN_TANH, RNN_RELU, LSTM,"
                             " GRU)")
    parser.add_argument("--data-loader", type=str, default="BP",
                        help="The data loader to use")
    parser.add_argument("--loss", type=str, default="NLL",
                        help="The loss to use")
    parser.add_argument("--nagts", type=int, default=10,
                        help="number of agents")
    parser.add_argument("--map", type=str, default="islands",
                        help="name of the map")
    parser.add_argument("--datasrc", type=str, default=DATASRC,
                        help="source domain whence data were drawn")

    # ------------------------ Hyper-parameters ------------------------

    parser.add_argument("--ninputs", type=int, default=-1,
                        help="number of input units")  # TODO: deleting this
    # arg

    parser.add_argument("--noutputs", type=int, default=-1,
                        help="number of output units")
    parser.add_argument("--nhid", type=int, default=1,
                        help="number of hidden units per layer")
    parser.add_argument("--nlayers", type=int, default=1,
                        help="number of layers")
    parser.add_argument("--bptt", type=int, default=-1,
                        help="sequence length. -1 to take the whole length of "
                             "the sequence as the horizon of bptt")
    parser.add_argument("--bsz", type=int, default=-1, metavar="N",
                        help="batch size")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout applied to layers (0 = no dropout)")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="initial learning rate")
    parser.add_argument("--clip", type=float, default=-1,
                        help="gradient clipping")
    parser.add_argument("--folds", type=int, default=1,
                        help="number of folds")

    # ------------------------ ---------------- ------------------------

    parser.add_argument("--epochs", type=int, default=100,
                        help="upper epoch limit")
    parser.add_argument("--timestamp", type=int, default=int(time.time()),
                        help="current time to time-stamp")
    parser.add_argument("--pre", action="store_true",
                        help="pre-training")
    parser.add_argument("--seed", type=int, default=1111,
                        help="random seed")
    parser.add_argument("--cuda", action="store_true",
                        help="use CUDA")
    parser.add_argument("--alr", action="store_true",
                        help="use adaptative learning rate")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="report interval")
    parser.add_argument("--eoe-log-intervals", type=int, default=1000,
                        help="number of end-of-epoch log interval")
    parser.add_argument("--check-exists", action="store_true",
                        help="Checking or not if a previous version of the "
                             "model exists")
    parser.add_argument("--check-pre-exists", action="store_true",
                        help="Checking or not if a previous version of the "
                             "pre-model, when needed, exists")

    parser.add_argument("--model-save", type=str, default=SAVES + "models/",
                        help="the directory's path where saving the final "
                             "model")
    parser.add_argument("--cost-save", type=str, default=SAVES + "costs/",
                        help="the directory's path where saving the logs "
                             "and the train and val costs")
    parser.add_argument("--log-save", type=str, default=SAVES + "logs/",
                        help="the directory's path where log of the training "
                             "is saved")
    parser.add_argument("--tb-save", type=str, default=SAVES + "tbs/",
                        help="the directory's path where the TensorBoardX log "
                             "of the training is saved")
    parser.add_argument("--ac-save", type=str, default=SAVES + "accuracies/",
                        help="the directory's path where the accuracy log of "
                             "the training is saved")

    parser.add_argument("--model-rep", type=str, default=RES + "models/",
                        help="the directory's path where existing "
                             "models are stored")
    parser.add_argument("--mean-rep", type=str, default=DATA + "/means/",
                        help="the directory's path where existing "
                             "models are stored")

    parser.add_argument("--inf-exec-id", type=int, default=0,
                        help="Infimum execution id")
    parser.add_argument("--sup-exec-id", type=int, default=299,
                        help="Supremum execution id") # TODO: automatising
    # sup exec id

    args = parser.parse_args()

    return args
