# -*- coding: utf-8 -*-

from maptrainer import display, RES
from maptrainer import PCKGROOT
from maptrainer.argsparser import parse_args

args = parse_args()

LOSS = "{}/costs/".format(RES)

maps = ["islands", "map_a", "grid"]
nagts = [1, 5, 10, 15, 25]

# archs = ["2-50", "4-50", "8-50", "2-100", "2-200", "2-200", "10-50", "50-2",
#          "50-10"]

# archs = ["1-1", "2-2", "1-50", "2-50", "50-2", "4-10", "3-50"]

archs = ["1-50", "2-50", "50-2", "4-10", "3-50"]

bptts = [-1] * len(archs)

args.model_type = "RNN"
args.model_variant = "LSTM"


display.plot_end_costs_suffixes(maps=["map_a"], nagts=[15], archs=archs,
                                bptts=bptts, modeltype=args.model_type,
                                modelvariant=args.model_variant,
                                pathcosts=LOSS,
                                datasrc=args.datasrc)


# Training costs for each population size and map
'''
for m in maps:
    display.plot_end_costs(maps=[m], nagts=nagts, archs=archs,
                           bptts=bptts, modeltype=args.model_type,
                           modelvariant=args.model_variant,
                           pathcosts=LOSS, datasrc=args.datasrc)
'''

# Training costs for each population size averaged over the maps

'''
display.plot_av_pop_maps_end_costs(maps=maps, nagts=nagts, archs=archs,
                                   bptts=bptts, modeltype=args.model_type,
                                   modelvariant=args.model_variant,
                                   pathcosts=LOSS, datasrc=args.datasrc)
'''

# Training costs for each map averaged over the population sizes
'''
for m in maps:
    display.plot_av_pop_end_costs(maps=[m], nagts=nagts, archs=archs,
                                  bptts=bptts, modeltype=args.model_type,
                                  modelvariant=args.model_variant,
                                  pathcosts=LOSS, datasrc=args.datasrc)
'''
