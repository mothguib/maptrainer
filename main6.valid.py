# -*- coding: utf-8 -*-

# Validating a model

from maptrainer import loss
from maptrainer.data import dlhandler
from maptrainer.model.MeanModel import MeanModel
from maptrainer.model.ZeroModel import ZeroModel
from maptrainer.valid import complete_evaluation

criterion = loss.load_loss("MSE")()
bsz = -1
bptt = -1

for m in ["islands", "map_a", "grid"]:
    for n in [1, 5, 10, 15, 25]:
        model = ZeroModel(tpl=m, nagts=n)
        dl = dlhandler.load_data_loader("IP")(
                nagts=n, _map=m, nb_folds=1)
        dl.load_data()

        print(complete_evaluation(model=model,
                                  criterion=criterion,
                                  dl=dl,
                                  bsz=bsz,
                                  bptt=bptt)
              )