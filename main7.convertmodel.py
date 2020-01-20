import sys
import torch

path = sys.argv[1]

with open(path, "rb") as s:
    model = torch.load(s, map_location=lambda storage, loc: storage)

torch.save(model.state_dict(), path.replace(".entire.pt", ".pt"))
