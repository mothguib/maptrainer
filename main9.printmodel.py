# -*- coding: utf-8 -*-

import torch

from maptrainer import RES

path = RES + "/models/NonLinearOModel/ReLU/islands/hcc_0.2/islands-15-1-50" \
              "--1/entire/model-islands-15-1-50--1.1530720397.100000.1e-07" \
             "--1--1.0-0.0-1.entire.pt"

with open(path, 'rb') as s:
    model = torch.load(s)

print(model)
for n, p in model.named_parameters():
    print(n)
