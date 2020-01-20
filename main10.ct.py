# -*- coding: utf-8 -*-

# import sys
import os
import json
import time
import torch

import maptrainer.pathformatter as pf
import maptrainer.train as train
from maptrainer.data import dataprcss as dp
from maptrainer import misc, loss
from maptrainer.argsparser import parse_args
from maptrainer.data import dlhandler
from maptrainer.model import modelhandler
from tensorboardX import SummaryWriter

args = parse_args()

###############################################################################
# Load data
###############################################################################

if args.pre:
    dl = dlhandler. \
        load_data_loader(args.data_loader)(nagts=args.nagts, tpl=args.map,
                                           nb_folds=args.folds, pre=True)
    dl.load_pre_data()
else:
    dl = dlhandler. \
        load_data_loader(args.data_loader)(nagts=args.nagts, tpl=args.map,
                                           nb_folds=args.folds,
                                           datasrc=args.datasrc,
                                           inf_exec_id=args.inf_exec_id,
                                           sup_exec_id=args.sup_exec_id)
    dl.load_data()

###############################################################################
# Training settings
###############################################################################

if not os.path.exists(args.model_save):
    os.makedirs(args.model_save)

# Number of inputs
ninputs = dl.domain_data.size()[-1]

# Adjacent matrix of the topology
graph = dl.load_topology(args.map)

db_size = dl.domain_data.size()

# Data sizes
(train_ddata, train_tdata), (val_ddata, val_tdata) = dl.next_fold()

# TODO: retrieving directly the data labels

dl.fold = 1  # Reset of the fold counter after calling the method
# `next_fold`
train_size = train_ddata.size()
val_size = val_ddata.size()  # TODO: creating a new function returning the
# sizes of training and validation input data

# BPTT, batch size and folds
if args.pre:
    # For pretraining, the whole database is considered, i.e. all the
    # elements for all the time
    args.bptt = -1
    args.bsz = -1
    args.folds = 1

bsz = args.bsz

if args.bptt < 0:
    bptt = db_size[1] // -args.bptt
else:
    bptt = args.bptt

cuda = args.cuda

# Setting of the random seed manually for reproducibility.
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not cuda:
        print("WARNING: You have a CUDA device, so you should probably run "
              "with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# Parameter pertaining to logging

'''
# Number of end-of-epoch log interval
neli = 1000
# End of epoch's log interval
eli = args.epochs // args.eoe_log_intervals
# If `eoe_log_interval = 0` i.e. `args.epochs < nb_eoe_log_interval`, then
# `eoe_log_interval = 1`
eli = eli if eli > 0 else 1
'''

# eli = args.eoe_log_intervals
eli = 100
# eli = 50

###############################################################################
# Load or build the model
###############################################################################

# Model's file name

# A model is named according to its hyper-parameters
modelname = pf.generate_modelname(tpl=args.map,
                                  nlayers=args.nlayers,
                                  nhid=args.nhid,
                                  bptt=args.bptt,
                                  pre=args.pre,
                                  nagts=args.nagts)

# `spdp` as "type's variant's save directory path": directory path of the save
# file for the current type and variant of model
spdp = pf. \
    generate_save_profile_directory_path(log_rep_dir=args.model_rep,
                                         model_type=args.model_type,
                                         model_variant=args.model_variant,
                                         pre=args.pre, tpl=args.map,
                                         datasrc=args.datasrc)

print('=' * 90)
print("Model Loading")
print('=' * 90)

model = modelhandler. \
    load_model(model_type=args.model_type,
               model_variant=args.model_variant,
               modelname=modelname,
               ninputs=ninputs,
               noutputs=args.noutputs,
               nlayers=args.nlayers,
               nhid=args.nhid,
               dropout=args.dropout, pre=args.pre,
               check_exists=args.check_exists,
               check_pre_exists=args.check_pre_exists,
               model_dirpath=spdp,
               graph=graph)

print('-' * 90)
print('\n')

misc.cuda(model, cuda)

criterion = misc.cuda(loss.load_loss(args.loss)(), cuda, True)
# For the `nn.NLLLoss()` criterion: the input passed to this criterion is
# expected to contain the log-probabilities of each class. Besides,
# the returned cost is averaged over the batch passed to him as input

# optimiser = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
optimiser = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)

print("Optimiser: ", optimiser, '\n')

train_bsz = dp.calibrate_batch_size(bsz=dp.batch_size(bsz, train_ddata),
                                    model=model,
                                    ddata=train_ddata,
                                    tdata=train_tdata,
                                    bptt=bptt,
                                    dl=dl,
                                    criterion=criterion,
                                    optimiser=optimiser,
                                    cuda=cuda)

# TODO: handling the reinitialisation of the model in a specific function
del model, optimiser
torch.cuda.empty_cache()
model = modelhandler. \
    load_model(model_type=args.model_type,
               model_variant=args.model_variant,
               modelname=modelname,
               ninputs=ninputs,
               noutputs=args.noutputs,
               nlayers=args.nlayers,
               nhid=args.nhid,
               dropout=args.dropout, pre=args.pre,
               check_exists=args.check_exists,
               check_pre_exists=args.check_pre_exists,
               model_dirpath=spdp,
               graph=graph)

misc.cuda(model, cuda, True)

# optimiser = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
optimiser = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)

print("Optimiser: ", optimiser, '\n')

###############################################################################
# Logs initialisation (model, log, cost and TensorBoard log)
###############################################################################

model_path, log_path, cost_path, tb_path, ac_path = misc.init_saves(args)

writer = SummaryWriter(tb_path)

init_str = '#' * 90 + '\n' \
                      '| map: {} | nagts: {} | epochs: {:4d} | nlayers: {} ' \
                      '| ninputs: {} | noutputs: {} | nhid: {} ' \
                      '| db_size: {} | bptt: {}({}) | bsz: {} ({}) ' \
                      '| dropout: {:1.2f} | lr: {:1.4f} | clip: {:1.2f} ' \
                      '| folds: {} | cuda: {} | check_exists: {} | alr: {} ' \
                      '| datasrc: {} |' \
    .format(args.map, args.nagts, args.epochs, args.nlayers,
            ninputs, args.noutputs, args.nhid, db_size, args.bptt, bptt,
            args.bsz, train_bsz, args.dropout, args.learning_rate,
            args.clip, args.folds, cuda, args.check_exists, args.alr,
            args.datasrc) + \
           '\n' + \
           '#' * 90 + '\n'

print(init_str)

with open(log_path, 'w') as s:
    s.write(init_str)

###############################################################################
# Evaluation
###############################################################################

# Run on test data.
dl.fold = 1
(train_d_data, train_tdata), (val_ddata, val_tdata) = dl.next_fold()
eval_bsz = val_ddata.size()[0]

val_cost = train.evaluate(model=model, dl=dl, criterion=criterion,
                          ddata=val_ddata,
                          tdata=val_tdata,
                          bsz=eval_bsz, bptt=bptt, cuda=args.cuda)

# * `batch_d_val`: batch of validation domain data
# * `batch_t_val`: batch of validation targets
batch_d_val, batch_l_val = dl.label_data(val_ddata[0], val_tdata[0])

hidden = model.init_hidden(bsz)

output, hidden = model(batch_d_val, hidden)

ncc, cc = dp.neighbour_correct_labels(torch.exp(output).squeeze(),
                               batch_d_val.squeeze(),
                               batch_l_val.squeeze())

print("Number of correct predictions:")
print(ncc, '\n')

print("Correct predictions, i.e. the maximum probability corresponds to a "
      "correct node:")
print(cc, '\n')

print("Inputs (one-hot vectors):")
print(batch_d_val.squeeze(), '\n')

print("Outputs (predictions):")
print(torch.exp(output).squeeze(), '\n')

print('=' * 112)
print('| End of validation | final valid cost: {:5.2f} |'.format(val_cost))
print('=' * 112)
