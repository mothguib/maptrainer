from maptrainer import display, RES

LOSS = "home/mehdi/Data//MAPTrainer-Resources/costs/RNN/LSTM/islands/hcc_0.2" \
       "/islands-15-2-50--1/" \
       "cost-islands-15-2-50--1.1521829296.2000.0.1--1--1.0-0.0-1.json"

display.plot_cost("islands", 10, 4, 10, -1, True, pathcosts=LOSS)
