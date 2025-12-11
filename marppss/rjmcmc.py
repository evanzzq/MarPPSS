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
        if bookkeeping.fitRange is not None: 
            CDinv = CDinv[left:right, left:right]
        CDinv *= np.exp(-model.loge)
        logL = -0.5 * Diff @ CDinv @ Diff

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

    Diff_PP, Diff_SS = Diff_PP[:, np.newaxis], Diff_SS[:, np.newaxis]

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

        sigmaPP = prior.stdPP
        sigmaSS = prior.stdSS

        sigmaPP *= np.exp(0.5 * model.loge)
        sigmaSS *= np.exp(0.5 * model.loge2)

        chi2_PP = np.sum((Diff_PP / sigmaPP) ** 2)
        chi2_SS = np.sum((Diff_SS / sigmaSS) ** 2)

        # Normalized log-likes (per sample)
        logL_PP = -0.5 * (chi2_PP)
        logL_SS = -0.5 * (chi2_SS)

    else:

        CDinv_PP = np.asarray(CDinv_PP)
        if bookkeeping.fitRange is not None: 
            CDinv_PP = CDinv_PP[left_PP:right_PP, left_PP:right_PP]
        CDinv_PP *= np.exp(-model.loge)

        CDinv_SS = np.asarray(CDinv_SS)
        if bookkeeping.fitRange is not None: CDinv_SS = CDinv_SS[left_SS:right_SS, left_SS:right_SS]
        CDinv_SS *= np.exp(-model.loge)

        chi2_PP = np.trace(Diff_PP.T @ CDinv_PP @ Diff_PP)
        chi2_SS = np.trace(Diff_SS.T @ CDinv_SS @ Diff_SS)

        logL_PP = -0.5 * (chi2_PP) #  / N_PP
        logL_SS = -0.5 * (chi2_SS) #  / N_SS

    logL = logL_PP + logL_SS
    return logL, logL_PP, logL_SS

def calc_like_prob_gv(model, bookkeeping):

    from pysurf96 import surf96
    
    # edit these based on data
    periods = np.array([18.9, 28.86]) 
    gv_obs = np.array([2.726, 2.829]) # S1000a group velocities
    gv_unc = np.array([0.006146, 0.01261]) * 1 # S1000a gv uncertainties
    sigma_gv = 0.01

    if bookkeeping.fitrho or bookkeeping.mode == 3:
        vpvsr = model.rho
    else:
        vpvsr = 1.8

    # define model for pysurf96 input
    H = np.append(model.H, 0)
    if bookkeeping.mode == 1: 
        vp = model.v
        vs = vp / vpvsr
    if bookkeeping.mode == 2: 
        vs = model.v
        vp = vs * vpvsr
    if bookkeeping.mode == 3:
        vs = model.v
        vp = model.v * vpvsr
    rho = vs* 0.8 # density

    # call pysurf96
    gv_model = surf96(
        H,
        vp,
        vs,
        rho,
        periods,
        wave="rayleigh",
        mode=1, # 1: fundamental, 2: second-mode, etc
        velocity="group",
        flat_earth=True)
    
    diff_gv = (gv_model - gv_obs)
    gv_unc *= np.exp(0.5 * model.loge_gv)
    logL_gv = -0.5 * np.sum((diff_gv / gv_unc) ** 2)
    
    return logL_gv

import copy
import numpy as np

def birth(model, prior, rayp):
    """
    Birth move with H as depths (monotonically increasing).

    model.H: shape (Nlayer,) depths of discontinuities (increasing)
    model.v: shape (Nlayer+1,) layer velocities incl. half-space

    We:
      - choose an insertion index k in [0, Nlayer]
      - draw newH uniformly between neighboring depths:
            k = 0:   [prior.HRange[0], H[0]]
            0<k<N:   [H[k-1], H[k]]
            k = N:   [H[N-1], prior.HRange[1]]
      - draw new v, w, w2, rho as before, with v strictly increasing.
    """

    # Don't update if already reaching maxN
    if model.Nlayer >= prior.maxN:
        return model, False

    model_new = copy.deepcopy(model)

    N = model_new.Nlayer  # number of existing discontinuities (len(H))

    # Choose insertion index k ∈ {0, ..., N}
    # Insert before existing index k (k==N means append at end)
    k = np.random.randint(0, N + 1)

    # -----------------------------
    # 1. Sample new depth (H as depth)
    # -----------------------------
    H_min_global, H_max_global = prior.HRange

    if N == 0:
        # No existing discontinuities: just draw anywhere in global range
        H_low, H_high = H_min_global, H_max_global

    else:
        if k == 0:
            # Insert at top: between global min and first discontinuity
            H_low  = H_min_global
            H_high = model_new.H[0]
        elif k == N:
            # Insert at bottom: between last discontinuity and global max
            H_low  = model_new.H[-1]
            H_high = H_max_global
        else:
            # Insert between two existing depths
            H_low  = model_new.H[k - 1]
            H_high = model_new.H[k]

    # If the bracket is invalid, reject
    if H_low >= H_high:
        return model, False

    newH = np.random.uniform(H_low, H_high)

    # -----------------------------
    # 2. Sample new w, w2, and rho
    # -----------------------------
    neww   = np.random.uniform(prior.wRange[0],   prior.wRange[1])
    neww2  = np.random.uniform(prior.wRange[0],   prior.wRange[1])
    newrho = np.random.uniform(prior.rhoRange[0], prior.rhoRange[1])

    # -----------------------------
    # 3. Sample new v, enforcing strictly increasing velocity
    # -----------------------------
    v_min, v_max = prior.vRange[0], prior.vRange[1]

    # lower bound: previous layer velocity
    if k > 0:
        v_min = max(v_min, model_new.v[k - 1])

    # upper bound: next layer or half-space (index N)
    # v has length N+1, so v[k] exists for k in [0, N]
    if k <= N:
        v_max = min(v_max, model_new.v[k])

    if v_min >= v_max:
        return model, False

    newv = np.random.uniform(v_min, v_max)

    # -----------------------------
    # 4. Insert into model arrays
    # -----------------------------
    model_new.H   = np.insert(model_new.H,   k, newH)
    model_new.w   = np.insert(model_new.w,   k, neww)
    model_new.w2  = np.insert(model_new.w2,  k, neww2)
    model_new.v   = np.insert(model_new.v,   k, newv)
    model_new.rho = np.insert(model_new.rho, k, newrho)
    model_new.Nlayer = N + 1

    # Final consistency check (including monotonic H & v)
    success = check_model(model_new, prior, rayp)
    if success:
        return model_new, True
    return model, False


def death(model, prior, rayp):
    model_new = copy.deepcopy(model)
    if model_new.Nlayer > 1:
        idx = np.random.randint(model_new.Nlayer)
        model_new.Nlayer -= 1
        model_new.H   = np.delete(model_new.H, idx)
        model_new.w   = np.delete(model_new.w, idx)
        model_new.w2  = np.delete(model_new.w2, idx)
        model_new.v   = np.delete(model_new.v, idx)
        model_new.rho = np.delete(model_new.rho, idx)
        # Check model
        success = check_model(model_new, prior, rayp)
        if success:
            return model_new, True
    return model, False

def update_H(model, prior, rayp):
    # Copy model
    model_new = copy.deepcopy(model)
    N = model_new.Nlayer

    if N == 0:
        # No discontinuities to move
        return model, False

    # Select a discontinuity index to update
    idx = np.random.randint(0, N)

    # Work with a copy of the current depths
    H_old = model_new.H.copy()

    # Propose new depth with a Gaussian step
    newH = H_old[idx] + prior.HStd * np.random.randn()

    # Determine admissible interval based on neighbors + global HRange
    H_min, H_max = prior.HRange  # global bounds

    if N > 1:
        if idx > 0:
            # lower neighbor
            H_min = max(H_min, H_old[idx - 1])
        if idx < N - 1:
            # upper neighbor
            H_max = min(H_max, H_old[idx + 1])

    # If the bracket is invalid or proposal outside allowed range, reject
    if not (H_min < newH < H_max):
        return model, False

    # Accept proposed depth
    model_new.H[idx] = newH

    # Final consistency check (monotonicity etc.)
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

def update_loge_gv(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Update
    model_new.loge_gv += prior.logeStd * np.random.randn()
    # Check range and return
    if prior.logeRange[0] <= model_new.loge_gv <= prior.logeRange[1]:
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
        if bookkeeping.fitgv: 
            logL_gv_trace = []
            logL_wf_trace = []
            logL_wf_trace.append(logL)
            logL_gv = calc_like_prob_gv(model, bookkeeping)
            logL_gv_trace.append(logL_gv)
            logL += logL_gv
        logL_trace.append(logL)
    elif bookkeeping.mode == 3:
        logL_PP_trace, logL_SS_trace = [], []
        logL, logL_PP, logL_SS = calc_like_prob_joint(
            P_PP, P_SS, D_PP, D_SS, 
            model, prior, bookkeeping, CDinv_PP=CDinv_PP, CDinv_SS=CDinv_SS)
        if bookkeeping.fitgv: 
            logL_gv_trace = []
            logL_gv = calc_like_prob_gv(model, bookkeeping)
            logL_gv_trace.append(logL_gv)
            logL += logL_gv
        logL_trace.append(logL)
        logL_PP_trace.append(logL_PP)
        logL_SS_trace.append(logL_SS)

    start_time = time.time()
    checkpoint_interval = totalSteps // 100

    ensemble = []

    # Action pool
    actionPool = [2,3,8] # H, w, v
    if prior.maxN > 1: actionPool = np.append(actionPool, [0,1]) # birth, death
    if bookkeeping.mode == 3: actionPool = np.append(actionPool, [4]) # w2
    if bookkeeping.mode == 3 or (bookkeeping.fitgv and bookkeeping.fitrho): actionPool = np.append(actionPool, [9]) # rho
    if bookkeeping.fitLoge:
        if bookkeeping.mode in [1, 2]: actionPool = np.append(actionPool, [5]) # loge
        if bookkeeping.mode == 3: actionPool = np.append(actionPool, [5,6]) # loge and loge2
    # if bookkeeping.fitgv: actionPool = np.append(actionPool, [7]) # loge_gv # commented out so loge_gv === 0

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
                model_new, _ = update_loge_gv(model_new, prior)
            elif action == 8:
                model_new, _ = update_v(model_new, prior, bookkeeping.rayp)
            elif action == 9:
                model_new, _ = update_rho(model_new, prior, bookkeeping.rayp)

        # Compute likelihood
        if bookkeeping.mode in (1, 2):
            new_logL = calc_like_prob(P, D, model_new, prior, bookkeeping, CDinv=CDinv)
        elif bookkeeping.mode == 3:
            new_logL, new_logL_PP, new_logL_SS = calc_like_prob_joint(
                P_PP, P_SS, D_PP, D_SS, 
                model_new, prior, bookkeeping, CDinv_PP=CDinv_PP, CDinv_SS=CDinv_SS)      
        if bookkeeping.fitgv: 
            new_logL_gv = calc_like_prob_gv(model_new, bookkeeping)
            new_logL += new_logL_gv

        # Acceptance probability
        log_accept_ratio = ((new_logL - logL) 
                            + n_len * ((model.loge - model_new.loge) + (model.loge2 - model_new.loge2)) 
                            + 2 * (model.loge_gv - model_new.loge_gv))

        if np.log(np.random.rand()) < log_accept_ratio:
            model = model_new
            logL = new_logL
            if bookkeeping.mode == 3:
                logL_PP = new_logL_PP
                logL_SS = new_logL_SS
            if bookkeeping.fitgv:
                logL_gv = new_logL_gv
        
        logL_trace.append(logL)
        if bookkeeping.mode == 3:
            logL_PP_trace.append(logL_PP)
            logL_SS_trace.append(logL_SS)
        if bookkeeping.fitgv:
            logL_gv_trace.append(logL_gv)
            logL_wf_trace.append(logL - logL_gv)

        # Save only selected models after burn-in
        if iStep >= burnInSteps and (iStep - burnInSteps) % save_interval == 0:
            ensemble.append(model)
        
        # Checkpoint log/plot every 1%
        if (iStep + 1) % checkpoint_interval == 0:
            # Save (overwrite) log-likelihood plot
            fig, ax = plt.subplots()
            ax.plot(logL_trace, 'k-')
            ax.set_xscale('log')
            ax.set_xlabel("Step")
            ax.set_ylabel("log Likelihood")
            fig.tight_layout()
            fig.savefig(os.path.join(saveDir, "logL.png"))  # overwrites each time
            plt.close(fig)

            # Also save waveform vs. group velocity misfit
            if bookkeeping.mode in (1,2) and bookkeeping.fitgv:
                fig, ax = plt.subplots()
                ax.plot(logL_wf_trace, 'k-', label="waveform")
                ax.plot(logL_gv_trace, 'r-', label="group vel.")
                ymin = min(logL_wf_trace[-1], logL_gv_trace[-1]) - 0.1 * abs(logL_wf_trace[-1] - logL_gv_trace[-1])
                ymax = max(logL_wf_trace[-1], logL_gv_trace[-1]) + 0.1 * abs(logL_wf_trace[-1] - logL_gv_trace[-1])
                ax.set_ylim(ymin, ymax)
                ax.set_xscale('log')
                ax.set_xlabel("Step")
                ax.set_ylabel("log Likelihood")
                fig.tight_layout()
                fig.savefig(os.path.join(saveDir, "logL_wf_gv.png"))  # overwrites each time
                plt.close(fig)

            # Save PP/SS plots for mode 3
            if bookkeeping.mode == 3:
                fig, ax = plt.subplots()
                ax.plot(logL_PP_trace, 'b-', label="PP")
                ax.plot(logL_SS_trace, 'r-', label="SS")
                if bookkeeping.fitgv: ax.plot(logL_gv_trace, 'k-', label="group vel.")
                ymin = min(logL_SS_trace[-1], logL_PP_trace[-1]) - 0.1 * abs(logL_SS_trace[-1] - logL_PP_trace[-1])
                ymax = max(logL_SS_trace[-1], logL_PP_trace[-1]) + 0.1 * abs(logL_SS_trace[-1] - logL_PP_trace[-1])
                ax.set_ylim(ymin, ymax)
                ax.set_xscale('log')
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