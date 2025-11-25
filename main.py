import os, pickle
import numpy as np
from marppss.model import Prior, Bookkeeping
from marppss.rjmcmc import rjmcmc_run
from parameter_setup import *

# Define dir
saveDir = os.path.join(filedir, "run", modname, runname)
os.makedirs(saveDir, exist_ok=True)

# Load data: P, D
if mode == 1: datadir = os.path.join(filedir, "data", PPdir)
if mode == 2: datadir = os.path.join(filedir, "data", SSdir)
if mode == 3: pass

data = np.load(os.path.join(datadir, "data.npz"))
P, D, time = data["P"], data["D"], data["time"]
dt = time[1] - time[0]

# Build and save prior
prior = Prior(
    HRange=HRange, vRange=vRange, rhoRange=rhoRange, dt=dt, tlen=time[-1]/2, maxN=maxN, stdP=stdP
)
with open(os.path.join(saveDir, "prior.pkl"), "wb") as f:
    pickle.dump(prior, f)

# Build and save bookkeeping
bookkeeping = Bookkeeping(
    mode=mode, rayp=rayp,
    totalSteps=totalSteps, burnInSteps=burnInSteps, nSaveModels=nSaveModels,
    actionsPerStep=actionsPerStep
)
with open(os.path.join(saveDir, "bookkeeping.pkl"), "wb") as f:
    pickle.dump(bookkeeping, f)

# rjmcmc call
ensemble, logL_trace = rjmcmc_run(P, D, prior, bookkeeping, saveDir, CDinv=None)
with open(os.path.join(saveDir, "ensemble.pkl"), "wb") as f:
    pickle.dump(ensemble, f)