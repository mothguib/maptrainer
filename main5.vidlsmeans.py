# -*- coding: utf-8 -*-

import json
import os
import sys

from maptrainer import PRJCTROOT, LOCALMEANS
from maptrainer.argsparser import parse_args
from maptrainer.data import dlhandler
from pytrol.util import pathformatter as pf

args = parse_args()

maps = ["islands", "map_a", "grid", "map_b", "corridor", "circle"]
nagts = [5, 10, 15, 25]
datasrc = "hpcc_0.5"

for m in maps:
    for n in nagts:
        fp = pf.means_path(m, n, datasrc, LOCALMEANS)

        dl = dlhandler.load_data_loader("IP")(
                nagts=n, tpl=m)
        dl.load_data()

        # Data
        (dtrain, ttrain), (dval, tval) = dl.next_fold()

        means = dl.mean(ttrain, len(ttrain.size()) - 1).tolist()

        print("Map {}, {} agents, train:\n\n".format(m, n), means, '\n')

        with open(fp, 'w') as s:
            json.dump(means, s)
