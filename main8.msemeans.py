# -*- coding: utf-8 -*-

# This script computes the training and validation costs of the Mean model

import json
import os
import sys
import torch
from torch.nn.modules.loss import MSELoss

import maptrainer.pathformatter as pf
from maptrainer import PRJCTROOT
from maptrainer.argsparser import parse_args
from maptrainer.data import dlhandler

from pytrol.util import pathformatter as ppf

args = parse_args()

maps = ["islands", "map_a", "grid", "map_b", "corridor", "circle"]
nagts = [5, 10, 15, 25]
datasrc = "hpcc_0.5"

for m in maps:
    for n in nagts:

        model_fp = ppf.means_path(m=m, n=n, datasrc=datasrc,
                                  mean_dirpath=args.mean_rep)

        cost_dirpath = \
            pf.generate_save_profile_directory_path(log_rep_dir=args.cost_save,
                                                    model_type="Mean",
                                                    model_variant="Mean",
                                                    tpl=m,
                                                    pre=False,
                                                    datasrc=datasrc) \
            + "/{}-{}-mean/".format(m, n)

        cost_fp = cost_dirpath + "cost-{}-{}-mean.json".format(m, n)

        if not os.path.exists(cost_dirpath):
            os.makedirs(cost_dirpath)

        dl = dlhandler.load_data_loader("IP")(nagts=n, tpl=m)
        dl.load_data()

        # Data
        (dtrain, ttrain), (dval, tval) = dl.next_fold()

        with open(model_fp, 'r') as s:
            # Means (inputs)
            means = torch.Tensor(json.load(s))

        criterion = MSELoss()

        train_cost = criterion(ttrain, means.expand_as(ttrain)).item()
        val_cost = criterion(tval, means.expand_as(tval)).item()

        with open(cost_fp, 'w') as s:
            json.dump([train_cost, val_cost], s)
            
        print("{}, {} agents, train MSE: {}, valid MSE: {}".
              format(m, n, train_cost, val_cost))

