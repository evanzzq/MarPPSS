import copy, time, os, datetime
import numpy as np
import matplotlib.pyplot as plt
from marppss.model import Model
from marppss.forward import create_D_from_model
from marppss.util import check_model

def calc_like_prob(P, D, model, prior, bookkeeping, CDinv=None):
    """
    Calculate likelihood probability and associated matrices.
    
    Args:
        P: (n,) array
        D: (n,) array, where n is the number of sample points
        
    Returns:
        logL
    """

    # Forward step
    D_model = create_D_from_model(P, model, prior, bookkeeping)

    # Calculate Diff without expanding D_model
    Diff = (D_model - D)  # (npts, ntraces)

    if bookkeeping.fitRange is None:
        # Take the full negative (precursor) side
        Diff = Diff[:len(Diff) // 2]
    else:
        # Take only the selected range
        left  = int((prior.tlen - bookkeeping.fitRange[1])/prior.dt)
        right = int((prior.tlen - bookkeeping.fitRange[0])/prior.dt)
        Diff = Diff[left:right]
    
    # Compute log likelihood
    if CDinv is None:
        sigma = prior.stdPP if bookkeeping.mode == 1 else prior.stdSS
        sigma *= np.exp(0.5 * model.loge)
        logL = -0.5 * np.sum((Diff / sigma) ** 2)
    else:
        CDinv = np.asarray(CDinv)
        CDinv *= np.exp(-model.loge)
        logL = -0.5 * np.trace(Diff.T @ CDinv @ Diff) # not fixed for selected range case

    return logL

def calc_like_prob_joint(P_PP, P_SS, D_PP, D_SS, 
                         model, prior, bookkeeping, CDinv_PP=None, CDinv_SS=None):
    """
    Calculate likelihood probability and associated matrices.
    
    Args:
        P: (n,) array
        D: (n,) array, where n is the number of sample points
        
    Returns:
        logL
    """

    # Forward step
    D_PP_model, D_SS_model = create_D_from_model(
        np.column_stack((P_PP, P_SS)), model, prior, bookkeeping)

    # Calculate Diff without expanding D_model
    Diff_PP = (D_PP_model - D_PP)
    Diff_SS = (D_SS_model - D_SS)

    if bookkeeping.fitRange is None:
        # Take the full negative (precursor) side
        Diff_PP = Diff_PP[:len(Diff_PP) // 2]
        Diff_SS = Diff_SS[:len(Diff_SS) // 2]
    else:
        # Take only the selected range
        # PP
        left_PP  = int((prior.tlen - bookkeeping.fitRange[1])/prior.dt)
        right_PP = int((prior.tlen - bookkeeping.fitRange[0])/prior.dt)
        Diff_PP = Diff_PP[left_PP:right_PP]
        # SS
        left_SS  = int((prior.tlen - bookkeeping.fitRange[3])/prior.dt)
        right_SS = int((prior.tlen - bookkeeping.fitRange[2])/prior.dt)
        Diff_SS = Diff_SS[left_SS:right_SS]
    
    # Compute log likelihood
    if CDinv_PP is None or CDinv_SS is None:
        # Number of samples in each window
        N_PP = Diff_PP.size
        N_SS = Diff_SS.size

        sigmaPP = prior.stdPP
        sigmaSS = prior.stdSS

        sigmaPP *= np.exp(0.5 * model.loge)
        sigmaSS *= np.exp(0.5 * model.loge2)

        chi2_PP = np.sum((Diff_PP / sigmaPP) ** 2)
        chi2_SS = np.sum((Diff_SS / sigmaSS) ** 2)

        # Normalized log-likes (per sample)
        logL_PP = -0.5 * (chi2_PP / N_PP)
        logL_SS = -0.5 * (chi2_SS / N_SS)
    else:
        # not fixed for selected range case
        N_PP = Diff_PP.size
        N_SS = Diff_SS.size

        CDinv_PP = np.asarray(CDinv_PP)
        CDinv_PP *= np.exp(-model.loge)

        CDinv_SS = np.asarray(CDinv_SS)
        CDinv_SS *= np.exp(-model.loge)

        chi2_PP = np.trace(Diff_PP.T @ CDinv_PP @ Diff_PP)
        chi2_SS = np.trace(Diff_SS.T @ CDinv_SS @ Diff_SS)

        logL_PP = -0.5 * (chi2_PP) #  / N_PP
        logL_SS = -0.5 * (chi2_SS) #  / N_SS

    logL = logL_PP + logL_SS
    return logL, logL_PP, logL_SS

def birth(model, prior, rayp):

    # Don't update if already reaching maxN
    if model.Nlayer >= prior.maxN:
        return model, False
    
    # Make a copy
    model_new = copy.deepcopy(model)

    # Randomly choose a layer to insert
    N = model_new.Nlayer
    k = np.random.randint(0, N + 1)

    # Sample new H, w, rho from priors
    newH = np.random.uniform(prior.HRange[0], prior.HRange[1])
    neww = np.random.uniform(prior.wRange[0], prior.wRange[1])
    newrho = np.random.uniform(prior.rhoRange[0], prior.rhoRange[1])

    # Determine admissible velocity interval for strictly increasing v
    v_min, v_max = prior.vRange[0], prior.vRange[1]
    if k > 0: v_min = max(v_min, model_new.v[k - 1])
    if k < N: v_max = min(v_max, model_new.v[k])


    if v_min >= v_max:
        return model, False

    newv = np.random.uniform(v_min, v_max)

    # Update model_new
    model_new.H   = np.insert(model_new.H,   k, newH)
    model_new.w   = np.insert(model_new.w,   k, neww)
    model_new.v   = np.insert(model_new.v,   k, newv)
    model_new.rho = np.insert(model_new.rho, k, newrho)
    model_new.Nlayer = N + 1

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
    # Check range and return
    if prior.wRange[0] <= model_new.w[idx] <= prior.wRange[1]:
        return model_new, True
    return model, False

def update_w2(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a layer and update
    idx = np.random.randint(model_new.Nlayer)
    model_new.w2[idx] += prior.wStd * np.random.randn()
    # Check range and return
    if prior.wRange[0] <= model_new.w2[idx] <= prior.wRange[1]:
        return model_new, True
    return model, False

def update_loge(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Update
    model_new.loge += prior.logeStd * np.random.randn()
    # Check range and return
    if prior.logeRange[0] <= model_new.loge <= prior.logeRange[1]:
        return model_new, True
    return model, False

def update_loge2(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Update
    model_new.loge2 += prior.logeStd * np.random.randn()
    # Check range and return
    if prior.logeRange[0] <= model_new.loge2 <= prior.logeRange[1]:
        return model_new, True
    return model, False

def update_v(model, prior, rayp):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a layer and update
    N = model_new.Nlayer
    idx = np.random.randint(0, N + 1)  # +1 includes half-space

    # Propose new v
    oldv = model_new.v[idx]
    newv = oldv + prior.vStd * np.random.randn()

    # Determine admissible velocity interval for strictly increasing v
    v_min, v_max = prior.vRange[0], prior.vRange[1]
    if idx > 0: v_min = max(v_min, model_new.v[idx - 1])
    if idx < N: v_max = min(v_max, model_new.v[idx + 1])

    if not (v_min < newv < v_max):
        return model, False

    # Accept proposed v in-place
    model_new.v[idx] = newv

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
    # mode 1/2: P and D are one trace
    # mode 3: P_PP = P[:,0]; P_SS = P[:,1]; same for D; CDinv_PP = CDinv[0]; CDinvv_SS = CDinv[1]

    n_len =  P.shape[0]

    # Extract variables for mode 3
    if bookkeeping.mode == 3:
        P_PP, P_SS, D_PP, D_SS = P[:,0], P[:,1], D[:,0], D[:,1]
        if CDinv is not None: 
            CDinv_PP, CDinv_SS = CDinv[0], CDinv[1]
        else:
            CDinv_PP, CDinv_SS = None, None
    
    totalSteps = bookkeeping.totalSteps
    burnInSteps = bookkeeping.burnInSteps
    nSaveModels = bookkeeping.nSaveModels
    save_interval = (totalSteps - burnInSteps) // nSaveModels
    actionsPerStep = bookkeeping.actionsPerStep

    # Start from an empty model
    model = Model.create_initial(prior=prior)

    # Initial likelihood
    logL_trace = []
    if bookkeeping.mode in (1, 2):
        logL = calc_like_prob(P, D, model, prior, bookkeeping, CDinv=CDinv)
        logL_trace.append(logL)
    elif bookkeeping.mode == 3:
        logL_PP_trace, logL_SS_trace = [], []
        logL, logL_PP, logL_SS = calc_like_prob_joint(
            P_PP, P_SS, D_PP, D_SS, 
            model, prior, bookkeeping, CDinv_PP=CDinv_PP, CDinv_SS=CDinv_SS)
        logL_trace.append(logL)
        logL_PP_trace.append(logL_PP)
        logL_SS_trace.append(logL_SS)

    start_time = time.time()
    checkpoint_interval = totalSteps // 100

    ensemble = []

    # Action pool
    actionPool = [2,3,7] # H, w, v
    if prior.maxN > 1: actionPool = np.append(actionPool, [0,1]) # birth, death
    if bookkeeping.mode == 3: actionPool = np.append(actionPool, [4,8]) # w2, rho
    if bookkeeping.fitLoge:
        if bookkeeping.mode in [1, 2]: actionPool = np.append(actionPool, [5])
        if bookkeeping.mode == 3: actionPool = np.append(actionPool, [5,6])

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
                model_new, _ = update_w2(model_new, prior)
            elif action == 5:
                model_new, _ = update_loge(model_new, prior)
            elif action == 6:
                model_new, _ = update_loge2(model_new, prior)
            elif action == 7:
                model_new, _ = update_v(model_new, prior, bookkeeping.rayp)
            elif action == 8:
                model_new, _ = update_rho(model_new, prior, bookkeeping.rayp)

        # Compute likelihood
        if bookkeeping.mode in (1, 2):
            new_logL = calc_like_prob(P, D, model_new, prior, bookkeeping, CDinv=CDinv)
        elif bookkeeping.mode == 3:
            new_logL, new_logL_PP, new_logL_SS = calc_like_prob_joint(
                P_PP, P_SS, D_PP, D_SS, 
                model_new, prior, bookkeeping, CDinv_PP=CDinv_PP, CDinv_SS=CDinv_SS)

        # Acceptance probability
        log_accept_ratio = (new_logL - logL) + n_len * ((model.loge - model_new.loge) + (model.loge2 - model_new.loge2))

        if np.log(np.random.rand()) < log_accept_ratio:
            model = model_new
            logL = new_logL
            logL_PP = new_logL_PP
            logL_SS = new_logL_SS
        
        logL_trace.append(logL)
        logL_PP_trace.append(logL_PP)
        logL_SS_trace.append(logL_SS)

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

            # Save PP/SS plots for mode 3
            if bookkeeping.mode == 3:
                fig, ax = plt.subplots()
                ax.plot(logL_PP_trace, 'b-', label="PP")
                ax.plot(logL_SS_trace, 'r-', label="SS")
                ymin = min(logL_SS_trace[-1], logL_PP_trace[-1]) - 0.1 * abs(logL_SS_trace[-1] - logL_PP_trace[-1])
                ymax = max(logL_SS_trace[-1], logL_PP_trace[-1]) + 0.1 * abs(logL_SS_trace[-1] - logL_PP_trace[-1])
                ax.set_ylim(ymin, ymax)
                ax.set_xlabel("Step")
                ax.set_ylabel("log Likelihood")
                ax.legend(loc="upper right")
                fig.tight_layout()
                fig.savefig(os.path.join(saveDir, "logL_PPSS.png"))  # overwrites each time
                plt.close(fig)

            # Overwrite progress log
            elapsed = time.time() - start_time
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(os.path.join(saveDir, "progress_log.txt"), "a") as f:
                f.write(f"[{now}] Step {iStep+1}/{totalSteps}, Elapsed: {elapsed:.2f} sec\n")

    return ensemble, logL_trace