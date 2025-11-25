import copy, time, os, datetime
import numpy as np
import matplotlib.pyplot as plt
from marppss.model import Model
from marppss.forward import create_D_from_model
from marppss.util import check_model

def calc_like_prob(P, D, model, prior, bookkeeping, sigma=None, CDinv=None):
    """
    Calculate likelihood probability and associated matrices.
    
    Args:
        P: (n,) array
        D: (n,) array, where n is the number of sample points
        
    Returns:
        logL
    """
    if sigma is None: sigma = prior.stdP

    # Forward step
    D_model = create_D_from_model(P, model, prior, bookkeeping)

    # Calculate Diff without expanding D_model
    Diff = (D_model - D)  # (npts, ntraces)

    # Only take negative (precursor) side
    Diff = Diff[:len(Diff) // 2]
    
    # Compute log likelihood
    if CDinv is None:
        logL = -0.5 * np.sum((Diff / sigma) ** 2)
    else:
        logL = -0.5 * np.trace(Diff.T @ CDinv @ Diff)

    return logL

def birth(model, prior, rayp):
    model_new = copy.deepcopy(model)
    if model_new.Nlayer < prior.maxN:
        model_new.Nlayer += 1
        newH = np.random.uniform(prior.HRange[0], prior.HRange[1], 1)
        neww = np.random.uniform(prior.wRange[0], prior.wRange[1], 1)
        newv = np.random.uniform(prior.vRange[0], prior.vRange[1], 1)
        newrho = np.random.uniform(prior.rhoRange[0], prior.rhoRange[1], 1)
        model_new.H = np.append(model_new.H, newH)
        model_new.w = np.append(model_new.w, neww)
        model_new.v = np.append(model_new.v, newv)
        model_new.rho = np.append(model_new.rho, newrho)
        model_new.v = np.sort(model_new.v)
        success = check_model(model_new, prior, rayp)
        if success:
            return model_new, True
    return model, False

def death(model, prior, rayp):
    model_new = copy.deepcopy(model)
    if model_new.Nlayer > 1:
        idx = np.random.randint(model_new.Nlayer)
        model_new.Nlayer -= 1
        model_new.H = np.delete(model_new.H, idx)
        model_new.w = np.delete(model_new.w, idx)
        model_new.v = np.delete(model_new.v, idx)
        model_new.rho = np.delete(model_new.rho, idx)
        # Check model
        success = check_model(model_new, prior, rayp)
        if success:
            return model_new, True
    return model, False

def update_H(model, prior, rayp):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a layer and update
    idx = np.random.randint(model_new.Nlayer)
    model_new.H[idx] += prior.HStd * np.random.randn()
    # Check range
    if not (prior.HRange[0] <= model_new.H[idx] <= prior.HRange[1]):
        return model, False
    # Check model
    success = check_model(model_new, prior, rayp)
    if success:
        return model_new, True
    return model, False

def update_w(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a layer and update
    idx = np.random.randint(model_new.Nlayer)
    model_new.w[idx] += prior.wStd * np.random.randn()
    # Return
    return model_new, True

def update_v(model, prior, rayp):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a layer and update
    idx = np.random.randint(model_new.Nlayer + 1) # +1 half-space
    model_new.v[idx] += prior.vStd * np.random.randn()
    # Check range
    if not (prior.vRange[0] <= model_new.v[idx] <= prior.vRange[1]):
        return model, False
    model_new.v = np.sort(model_new.v)
    # Check model
    success = check_model(model_new, prior, rayp)
    if success:
        return model_new, True
    return model, False

def update_rho(model, prior, rayp):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a layer and update
    idx = np.random.randint(model_new.Nlayer + 1) # +1 half-space
    model_new.rho[idx] += prior.rhoStd * np.random.randn()
    # Check range
    if not (prior.rhoRange[0] <= model_new.rho[idx] <= prior.rhoRange[1]):
        return model, False
    # Check model
    success = check_model(model_new, prior, rayp)
    if success:
        return model_new, True
    return model, False

def rjmcmc_run(P, D, prior, bookkeeping, saveDir, CDinv=None):
    
    totalSteps = bookkeeping.totalSteps
    burnInSteps = bookkeeping.burnInSteps
    nSaveModels = bookkeeping.nSaveModels
    save_interval = (totalSteps - burnInSteps) // nSaveModels
    actionsPerStep = bookkeeping.actionsPerStep

    # Start from an empty model
    model = Model.create_initial(prior=prior)

    # Initial likelihood
    logL = calc_like_prob(P, D, model, prior, bookkeeping, CDinv=CDinv)

    start_time = time.time()
    checkpoint_interval = totalSteps // 100

    ensemble = []
    logL_trace = []

    # Action pool
    actionPool = [2,3,4]
    if prior.maxN > 1: actionPool = np.append(actionPool, [0,1])
    if bookkeeping.mode == 3: actionPool = np.append(actionPool, [5])

    for iStep in range(totalSteps):

        actions = np.random.choice(actionPool, size=actionsPerStep, replace=False)
        model_new = model

        for action in actions:
            if action == 0:
                model_new, _ = birth(model_new, prior, bookkeeping.rayp)
            elif action == 1:
                model_new, _ = death(model_new, prior, bookkeeping.rayp)
            elif action == 2:
                model_new, _ = update_H(model_new, prior, bookkeeping.rayp)
            elif action == 3:
                model_new, _ = update_w(model_new, prior)
            elif action == 4:
                model_new, _ = update_v(model_new, prior, bookkeeping.rayp)
            elif action == 5:
                model_new, _ = update_rho(model_new, prior, bookkeeping.rayp)

        # Compute likelihood
        new_logL = calc_like_prob(P, D, model_new, prior, bookkeeping, CDinv=CDinv)

        # Acceptance probability
        log_accept_ratio = new_logL - logL

        if np.log(np.random.rand()) < log_accept_ratio:
            model = model_new
            logL = new_logL
        
        logL_trace.append(logL)

        # Save only selected models after burn-in
        if iStep >= burnInSteps and (iStep - burnInSteps) % save_interval == 0:
            ensemble.append(model)
        
        # Checkpoint log/plot every 1%
        if (iStep + 1) % checkpoint_interval == 0:
            # Save (overwrite) log-likelihood plot
            fig, ax = plt.subplots()
            ax.plot(logL_trace, 'k-')
            ax.set_xlabel("Step")
            ax.set_ylabel("log Likelihood")
            fig.tight_layout()
            fig.savefig(os.path.join(saveDir, "logL.png"))  # overwrites each time
            plt.close(fig)

            # Overwrite progress log
            elapsed = time.time() - start_time
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(os.path.join(saveDir, "progress_log.txt"), "a") as f:
                f.write(f"[{now}] Step {iStep+1}/{totalSteps}, Elapsed: {elapsed:.2f} sec\n")

    return ensemble, logL_trace