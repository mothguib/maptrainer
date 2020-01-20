from maptrainer import misc, DATASRC, RES

LOSS = "{}/costs/".format(RES)

maps = ["islands", "map_a", "grid"]
nagts = [1, 5, 10, 15, 25]
archs = ["1-1", "2-2", "4-10", "1-50", "2-50", "3-50", "50-2"]
bptts = [-1] * len(archs)
datasrc = DATASRC

dicl = misc.get_end_costs(maps=["islands"], nagts=nagts, archs=archs,
                          bptts=bptts, modeltype="SMRNN", modelvariant="LSTM",
                          pathcosts=LOSS, datasrc=datasrc)
