import os, pickle
import numpy as np
from marppss.visualization import plot_velocity_ensemble, plot_predicted_vs_input
from parameter_setup import *

saveDir = os.path.join(filedir, "run", modname, runname)

# Load data: P, D
if mode == 1: datadir = os.path.join(filedir, "data", PPdir)
if mode == 2: datadir = os.path.join(filedir, "data", SSdir)
if mode == 3: pass

data = np.load(os.path.join(datadir, "data.npz"))
P, D, time = data["P"], data["D"], data["time"]

# Load ensemble, prior, and bookkeeping
with open(os.path.join(saveDir, "ensemble.pkl"), "rb") as f:
    ensemble = pickle.load(f)
with open(os.path.join(saveDir, "prior.pkl"), "rb") as f:
    prior = pickle.load(f)
with open(os.path.join(saveDir, "bookkeeping.pkl"), "rb") as f:
    bookkeeping = pickle.load(f)

plot_velocity_ensemble(ensemble)
plot_predicted_vs_input(ensemble, P, D, prior, bookkeeping)