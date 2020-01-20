import os

PCKGROOT = os.path.abspath(os.path.dirname(__file__)) + "/"

PRJCTROOT = PCKGROOT + "/../"

DATAROOT = os.environ['MAPDATA']

# Data root directory used to train models
DATA = DATAROOT + "/Pytrol-Resources/"

# Save directory: directory where models, logs etc. are saved
SAVES = PRJCTROOT + "saves/"

# MAP Trainer's Resources: previous trained models, logs, etc.
RES = DATAROOT + "/MAPTrainer-Resources/"

MAPS = DATA + "/maps/json_bin/"

LOCALMEANS = PRJCTROOT + "/means/"
