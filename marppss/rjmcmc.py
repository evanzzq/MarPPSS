import copy, time, os, datetime
import numpy as np
import matplotlib.pyplot as plt
from marppss.model import Model
from marppss.forward import create_D_from_model, create_arrivals_from_model
from marppss.util import check_model, enforce_increasing_velocity
from marppss.velocity import (
    _as_slope_array,
    all_layer_gradient_enabled,
    disba_layers_from_model,
    layer_bottom_velocity,
    layer_thicknesses,
    layer_velocity,
    minimum_velocity_jump_fraction,
    top_layer_gradient_enabled,
    top_layer_velocity,
    velocity_transition_directions,
)


def _default_fixed_vpvs(bookkeeping):
    assumptions = getattr(bookkeeping, "assumptions", None) or {}
    return float(assumptions.get("fixed_vpvs", 1.8))


def _get_travel_time_inputs(bookkeeping):
    travel_times = getattr(bookkeeping, "travel_times", None) or {}
    if travel_times:
        pp = travel_times.get("PP", {})
        ss = travel_times.get("SS", {})
        arr_PP_obs = np.asarray(pp.get("times", []), dtype=float)
        arr_SS_obs = np.asarray(ss.get("times", []), dtype=float)
        arr_PP_unc = np.asarray(pp.get("uncertainties", np.repeat(0.1, len(arr_PP_obs))), dtype=float)
        arr_SS_unc = np.asarray(ss.get("uncertainties", np.repeat(0.1, len(arr_SS_obs))), dtype=float)
        return arr_PP_obs, arr_SS_obs, arr_PP_unc, arr_SS_unc

    arr_PP_obs = np.array([5.31, 10.93, 17.34])
    arr_SS_obs = np.array([18.0, 40.0, 63.0])
    arr_PP_unc = np.repeat(0.1, len(arr_PP_obs))
    arr_SS_unc = np.repeat(0.1, len(arr_SS_obs))
    return arr_PP_obs, arr_SS_obs, arr_PP_unc, arr_SS_unc


def _get_group_velocity_inputs(bookkeeping):
    gv = getattr(bookkeeping, "group_velocity", None) or {}
    if gv:
        periods = np.asarray(gv["periods"], dtype=float)
        gv_obs = np.asarray(gv["values"], dtype=float)
        uncertainties = gv.get("uncertainties", 0.2)
        if np.isscalar(uncertainties):
            gv_unc = np.full_like(periods, float(uncertainties))
        else:
            gv_unc = np.asarray(uncertainties, dtype=float)
        return (
            periods,
            gv_obs,
            gv_unc,
            gv.get("wave", "rayleigh"),
            int(gv.get("mode", 0)),
            float(gv.get("vpvsr", _default_fixed_vpvs(bookkeeping))),
        )

    periods = np.array([2.5, 2.6, 2.7, 2.9, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0,
                        4.2, 4.4, 4.6, 4.8, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
                        8.0, 8.5, 9.0, 9.5])
    gv_obs = np.array([1.45520, 1.45430, 1.45470, 1.48910, 1.52890, 1.66900,
                       1.76820, 1.95960, 2.38430, 2.59290, 2.73800, 2.81960,
                       2.91720, 3.02230, 3.12000, 3.28570, 3.38680, 3.42470,
                       3.44760, 3.46300, 3.46350, 3.44390, 3.40360, 3.36620])
    gv_unc = np.full_like(periods, 0.2)
    return periods, gv_obs, gv_unc, "love", 0, _default_fixed_vpvs(bookkeeping)


def _get_avg_vs_inputs(bookkeeping):
    avg_vs = getattr(bookkeeping, "avg_vs", None) or {}
    if avg_vs:
        return float(avg_vs["value"]), float(avg_vs.get("uncertainty", 0.1))
    return 3.077, 0.1


def _enforce_increasing_velocity(bookkeeping):
    if velocity_transition_directions(getattr(bookkeeping, "assumptions", None)) is not None:
        return False
    return enforce_increasing_velocity(getattr(bookkeeping, "assumptions", None))


def _bad_loglike():
    return -1e100


def _finite_or_bad(value):
    value = float(np.asarray(value).squeeze())
    return value if np.isfinite(value) else _bad_loglike()


def _set_finite_ylim(ax, *series, pad_frac=0.1):
    values = []
    for s in series:
        arr = np.asarray(s, dtype=float).ravel()
        values.extend(arr[np.isfinite(arr)])
    if not values:
        return

    ymin = float(np.min(values))
    ymax = float(np.max(values))
    if ymin == ymax:
        pad = max(1.0, abs(ymin) * pad_frac)
    else:
        pad = pad_frac * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)


def _ensure_save_dir(save_dir):
    os.makedirs(save_dir, exist_ok=True)


def _savefig_with_retry(fig, path, attempts=3, delay=0.2):
    save_dir = os.path.dirname(path)
    last_exc = None
    for _ in range(int(attempts)):
        try:
            _ensure_save_dir(save_dir)
            fig.savefig(path)
            return
        except FileNotFoundError as exc:
            last_exc = exc
            time.sleep(float(delay))
    raise last_exc


def _append_progress_with_retry(path, line, attempts=3, delay=0.2):
    save_dir = os.path.dirname(path)
    last_exc = None
    for _ in range(int(attempts)):
        try:
            _ensure_save_dir(save_dir)
            with open(path, "a") as f:
                f.write(line)
            return
        except FileNotFoundError as exc:
            last_exc = exc
            time.sleep(float(delay))
    raise last_exc


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
    n_obs = len(Diff)
    if CDinv is None:
        sigma = prior.stdPP if bookkeeping.mode == 1 else prior.stdSS
        sigma *= np.exp(0.5 * model.loge)
        logL = -0.5 * np.sum((Diff / sigma) ** 2) - 0.5 * n_obs * model.loge
    else:
        CDinv = np.asarray(CDinv)
        if bookkeeping.fitRange is not None: 
            CDinv = CDinv[left:right, left:right]
        CDinv *= np.exp(-model.loge)
        logL = -0.5 * Diff @ CDinv @ Diff - 0.5 * n_obs * model.loge

    return _finite_or_bad(logL)

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
        n_obs_PP = len(Diff_PP)
        n_obs_SS = len(Diff_SS)

        sigmaPP = prior.stdPP
        sigmaSS = prior.stdSS

        sigmaPP *= np.exp(0.5 * model.loge)
        sigmaSS *= np.exp(0.5 * model.loge2)

        chi2_PP = np.sum((Diff_PP / sigmaPP) ** 2)
        chi2_SS = np.sum((Diff_SS / sigmaSS) ** 2)

        # Normalized log-likes (per sample)
        logL_PP = -0.5 * chi2_PP - 0.5 * n_obs_PP * model.loge
        logL_SS = -0.5 * chi2_SS - 0.5 * n_obs_SS * model.loge2

    else:
        n_obs_PP = len(Diff_PP)
        n_obs_SS = len(Diff_SS)

        CDinv_PP = np.asarray(CDinv_PP)
        if bookkeeping.fitRange is not None: 
            CDinv_PP = CDinv_PP[left_PP:right_PP, left_PP:right_PP]
        CDinv_PP *= np.exp(-model.loge)

        CDinv_SS = np.asarray(CDinv_SS)
        if bookkeeping.fitRange is not None: CDinv_SS = CDinv_SS[left_SS:right_SS, left_SS:right_SS]
        CDinv_SS *= np.exp(-model.loge)

        chi2_PP = np.trace(Diff_PP.T @ CDinv_PP @ Diff_PP)
        chi2_SS = np.trace(Diff_SS.T @ CDinv_SS @ Diff_SS)

        logL_PP = -0.5 * chi2_PP - 0.5 * n_obs_PP * model.loge
        logL_SS = -0.5 * chi2_SS - 0.5 * n_obs_SS * model.loge2

    logL = logL_PP + logL_SS
    return _finite_or_bad(logL), _finite_or_bad(logL_PP), _finite_or_bad(logL_SS)

def calc_like_prob_travel_time(model, bookkeeping):
    import numpy as np

    BAD = -1e100
    arr_PP_obs, arr_SS_obs, arr_PP_unc, arr_SS_unc = _get_travel_time_inputs(bookkeeping)

    # -----------------------------
    # compute model travel times
    # -----------------------------
    if bookkeeping.mode == 1:
        arr_PP_model = create_arrivals_from_model(model, bookkeeping)
        if arr_PP_model.shape != arr_PP_obs.shape:
            return BAD
        if np.any(~np.isfinite(arr_PP_model)):
            return BAD

        diff_PP = arr_PP_model - arr_PP_obs

        arr_PP_unc *= np.exp(0.5 * model.loge_TT)
        logL_PP = -0.5 * np.sum((diff_PP / arr_PP_unc) ** 2) - 0.5 * len(arr_PP_obs) * model.loge_TT
        return _finite_or_bad(logL_PP)

    elif bookkeeping.mode == 2:
        arr_SS_model = create_arrivals_from_model(model, bookkeeping)
        if arr_SS_model.shape != arr_SS_obs.shape:
            return BAD
        if np.any(~np.isfinite(arr_SS_model)):
            return BAD

        diff_SS = arr_SS_model - arr_SS_obs

        arr_SS_unc *= np.exp(0.5 * model.loge_TT)
        logL_SS = -0.5 * np.sum((diff_SS / arr_SS_unc) ** 2) - 0.5 * len(arr_SS_obs) * model.loge_TT
        return _finite_or_bad(logL_SS)

    elif bookkeeping.mode == 3:
        arr_PP_model, arr_SS_model = create_arrivals_from_model(model, bookkeeping)
        if arr_PP_model.shape != arr_PP_obs.shape or arr_SS_model.shape != arr_SS_obs.shape:
            return BAD
        if np.any(~np.isfinite(arr_PP_model)) or np.any(~np.isfinite(arr_SS_model)):
            return BAD

        diff_PP = arr_PP_model - arr_PP_obs
        diff_SS = arr_SS_model - arr_SS_obs

        arr_PP_unc *= np.exp(0.5 * model.loge_TT)
        arr_SS_unc *= np.exp(0.5 * model.loge_TT2)
        logL_PP = -0.5 * np.sum((diff_PP / arr_PP_unc) ** 2) - 0.5 * len(arr_PP_obs) * model.loge_TT
        logL_SS = -0.5 * np.sum((diff_SS / arr_SS_unc) ** 2) - 0.5 * len(arr_SS_obs) * model.loge_TT2

        return _finite_or_bad(logL_PP + logL_SS)

def calc_like_prob_gv(model, bookkeeping):
    import numpy as np
    from disba import GroupDispersion

    BAD = -1e100

    periods, gv_obs, gv_unc, wave, mode_idx, default_vpvsr = _get_group_velocity_inputs(bookkeeping)

    if bookkeeping.fitrho or bookkeeping.mode == 3:
        vpvsr = np.asarray(model.rho, dtype=float)
    else:
        vpvsr = default_vpvsr

    if bookkeeping.mode not in (1, 2, 3):
        return BAD
    H, vp, vs, rho = disba_layers_from_model(model, bookkeeping, vpvsr)

    # ---------- pre-checks ----------
    if np.any(~np.isfinite(vp)) or np.any(~np.isfinite(vs)) or np.any(~np.isfinite(rho)):
        return BAD
    if np.any(vp <= 0) or np.any(vs <= 0) or np.any(rho <= 0):
        return BAD

    if np.any(np.asarray(vpvsr) <= 0):
        return BAD

    finite_thickness = H[:-1]
    if finite_thickness.size > 0:
        if np.any(finite_thickness <= 0):
            return BAD
        if np.any(finite_thickness < 0.05):
            return BAD

    if _enforce_increasing_velocity(bookkeeping):
        if np.any(np.diff(vs) <= 0):
            return BAD
        if np.any(np.diff(vp) <= 0):
            return BAD

    # ---------- disba call ----------
    try:
        disp = GroupDispersion(H, vp, vs, rho)
        gv_model = disp(periods, mode=mode_idx, wave=wave).velocity
    except Exception:
        return BAD

    if np.any(~np.isfinite(gv_model)):
        return BAD
    if len(gv_model) != len(periods):
        return BAD

    diff_gv = gv_model - gv_obs
    gv_unc = gv_unc * np.exp(0.5 * model.loge_gv)

    logL_gv = -0.5 * np.sum((diff_gv / gv_unc) ** 2) - 0.5 * len(periods) * model.loge_gv

    return _finite_or_bad(logL_gv)

def calc_like_prob_avg_vs(model, bookkeeping):

    avg_vs_ref, avg_vs_unc = _get_avg_vs_inputs(bookkeeping)
    
    # define vpvsr
    if bookkeeping.fitrho or bookkeeping.mode == 3:
        vpvsr = model.rho
    else:
        vpvsr = _default_fixed_vpvs(bookkeeping)

    H = np.diff(np.r_[0.0, model.H])

    # define vs
    if bookkeeping.mode == 1: 
        vp = model.v
        vs = vp / vpvsr
    if bookkeeping.mode == 2: 
        vs = model.v
        vp = vs * vpvsr
    if bookkeeping.mode == 3:
        vs = model.v
        vp = model.v * vpvsr

    # calculate avg vs in model
    if top_layer_gradient_enabled(getattr(bookkeeping, "assumptions", None)) and H.size > 0:
        vs_layer = np.asarray(vs[:-1], dtype=float).copy()
        if bookkeeping.mode == 1:
            vpvsr_arr = np.asarray(vpvsr, dtype=float)
            vpvsr_finite = vpvsr_arr if vpvsr_arr.ndim == 0 else vpvsr_arr[:-1]
            a_vs = _as_slope_array(getattr(model, "a", 0.0), len(vs_layer)) / vpvsr_finite
        else:
            a_vs = _as_slope_array(getattr(model, "a", 0.0), len(vs_layer))
        if all_layer_gradient_enabled(getattr(bookkeeping, "assumptions", None)):
            for i, h_i in enumerate(H):
                z = np.linspace(0.0, h_i, 64)
                vs_layer[i] = np.trapz(
                    layer_velocity(vs_layer[i], a_vs[i], z, assumptions=bookkeeping.assumptions), z
                ) / h_i
        else:
            z = np.linspace(0.0, H[0], 64)
            vs_layer[0] = np.trapz(top_layer_velocity(vs_layer[0], a_vs[0], z, assumptions=bookkeeping.assumptions), z) / H[0]
    else:
        vs_layer = np.asarray(vs[:-1], dtype=float)
    avg_vs_model = np.sum(H * vs_layer) / np.sum(H)
    
    diff_avg_vs = (avg_vs_model - avg_vs_ref)
    avg_vs_unc *= np.exp(0.5 * model.loge_avg_vs)
    logL_avg_vs = -0.5 * np.sum((diff_avg_vs / avg_vs_unc) ** 2) - 0.5 * model.loge_avg_vs
    
    return _finite_or_bad(logL_avg_vs)

import copy
import numpy as np

def birth(model, prior, rayp, assumptions=None):
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
    newa   = np.random.uniform(prior.aRange[0],   prior.aRange[1])

    # -----------------------------
    # 3. Sample new v, enforcing strictly increasing velocity
    # -----------------------------
    v_min, v_max = prior.vRange[0], prior.vRange[1]

    if velocity_transition_directions(assumptions) is None and enforce_increasing_velocity(assumptions):
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
    if all_layer_gradient_enabled(assumptions):
        model_new.a = np.insert(_as_slope_array(model_new.a, N), k, newa)
    model_new.Nlayer = N + 1

    # Final consistency check (including monotonic H & v)
    success = check_model(model_new, prior, rayp, assumptions=assumptions)
    if success:
        return model_new, True
    return model, False


def death(model, prior, rayp, assumptions=None):
    model_new = copy.deepcopy(model)
    if model_new.Nlayer > 1:
        idx = np.random.randint(model_new.Nlayer)
        model_new.Nlayer -= 1
        model_new.H   = np.delete(model_new.H, idx)
        model_new.w   = np.delete(model_new.w, idx)
        model_new.w2  = np.delete(model_new.w2, idx)
        model_new.v   = np.delete(model_new.v, idx)
        model_new.rho = np.delete(model_new.rho, idx)
        if all_layer_gradient_enabled(assumptions):
            model_new.a = np.delete(_as_slope_array(model_new.a, model.Nlayer), idx)
        # Check model
        success = check_model(model_new, prior, rayp, assumptions=assumptions)
        if success:
            return model_new, True
    return model, False

def update_H(model, prior, rayp, assumptions=None):
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
    success = check_model(model_new, prior, rayp, assumptions=assumptions)
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

def update_loge_avg_vs(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Update
    model_new.loge_avg_vs += prior.logeStd * np.random.randn()
    # Check range and return
    if prior.logeRange[0] <= model_new.loge_avg_vs <= prior.logeRange[1]:
        return model_new, True
    return model, False

def update_loge_TT(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Update
    model_new.loge_TT += prior.logeStd * np.random.randn()
    # Check range and return
    if prior.logeRange[0] <= model_new.loge_TT <= prior.logeRange[1]:
        return model_new, True
    return model, False


def update_loge_TT2(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Update
    model_new.loge_TT2 += prior.logeStd * np.random.randn()
    # Check range and return
    if prior.logeRange[0] <= model_new.loge_TT2 <= prior.logeRange[1]:
        return model_new, True
    return model, False

def update_v(model, prior, rayp, assumptions=None):
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
    if all_layer_gradient_enabled(assumptions):
        if idx > 0:
            v_min = max(v_min, layer_bottom_velocity(model_new, idx - 1, assumptions=assumptions))
    elif velocity_transition_directions(assumptions) is None and enforce_increasing_velocity(assumptions):
        if idx > 0:
            lower = model_new.v[idx - 1]
            if idx == 1 and top_layer_gradient_enabled(assumptions):
                lower = top_layer_velocity(model_new.v[0], model_new.a, model_new.H[0], assumptions=assumptions)
            v_min = max(v_min, lower)
        if idx < N:
            v_max = min(v_max, model_new.v[idx + 1])

    if not (v_min < newv < v_max):
        return model, False

    # Accept proposed v in-place
    model_new.v[idx] = newv

    success = check_model(model_new, prior, rayp, assumptions=assumptions)
    if success:
        return model_new, True
    return model, False


def _reflect_into_interval(value, lower, upper):
    if lower > upper:
        return np.nan
    if lower == upper:
        return float(lower)

    width = upper - lower
    shifted = (float(value) - lower) % (2.0 * width)
    if shifted <= width:
        return lower + shifted
    return upper - (shifted - width)


def _all_layer_slope_interval(model, prior, idx, assumptions=None):
    h_i = float(layer_thicknesses(model.H)[idx])
    v_top = float(model.v[idx])
    v_next = float(model.v[idx + 1])
    jump = minimum_velocity_jump_fraction(assumptions)
    directions = velocity_transition_directions(assumptions)
    direction = directions[idx] if directions is not None else "inc"

    a_low = float(prior.aRange[0])
    a_high = float(prior.aRange[1])
    if h_i <= 0.0:
        return np.nan, np.nan

    # Keep the full evaluated layer inside vRange and increasing with depth.
    a_low = max(a_low, (prior.vRange[0] - v_top) / h_i)
    a_high = min(a_high, (prior.vRange[1] - v_top) / h_i)
    eps = max(1e-12, 1e-12 * max(abs(a_low), abs(a_high), 1.0))
    a_low = max(a_low, eps)

    if direction == "inc":
        a_high = min(a_high, (v_next / (1.0 + jump) - v_top) / h_i - eps)
    elif direction == "dec":
        a_low = max(a_low, (v_next / max(1.0 - jump, 1e-12) - v_top) / h_i + eps)
    elif direction == "free" and jump > 0.0:
        bottom_current = v_top + float(_as_slope_array(model.a, model.Nlayer)[idx]) * h_i
        if v_next >= bottom_current:
            a_high = min(a_high, (v_next / (1.0 + jump) - v_top) / h_i - eps)
        else:
            a_low = max(a_low, (v_next / max(1.0 - jump, 1e-12) - v_top) / h_i + eps)

    return a_low, a_high


def update_a(model, prior, rayp, assumptions=None):
    if not top_layer_gradient_enabled(assumptions):
        return model, False

    model_new = copy.deepcopy(model)
    if all_layer_gradient_enabled(assumptions):
        a = _as_slope_array(model_new.a, model_new.Nlayer)
        idx = np.random.randint(0, model_new.Nlayer)
        a_low, a_high = _all_layer_slope_interval(model_new, prior, idx, assumptions=assumptions)
        if not np.isfinite(a_low) or not np.isfinite(a_high) or a_low > a_high:
            return model, False
        a[idx] = _reflect_into_interval(a[idx] + prior.aStd * np.random.randn(), a_low, a_high)
        model_new.a = a
    else:
        model_new.a += prior.aStd * np.random.randn()
        if not (prior.aRange[0] <= model_new.a <= prior.aRange[1]):
            return model, False

    success = check_model(model_new, prior, rayp, assumptions=assumptions)
    if success:
        return model_new, True
    return model, False

def update_rho(model, prior, rayp, assumptions=None):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a layer and update
    idx = np.random.randint(model_new.Nlayer + 1) # +1 half-space
    model_new.rho[idx] += prior.rhoStd * np.random.randn()
    # Check range
    if not (prior.rhoRange[0] <= model_new.rho[idx] <= prior.rhoRange[1]):
        return model, False
    # Check model
    success = check_model(model_new, prior, rayp, assumptions=assumptions)
    if success:
        return model_new, True
    return model, False

def rjmcmc_run(P, D, prior, bookkeeping, saveDir, CDinv=None):
    # mode 1/2: P and D are one trace
    # mode 3: P_PP = P[:,0]; P_SS = P[:,1]; same for D; CDinv_PP = CDinv[0]; CDinvv_SS = CDinv[1]

    _ensure_save_dir(saveDir)
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
    fit_waveform = getattr(bookkeeping, "fitWaveform", not bookkeeping.fitTT)
    fit_travel_time = bookkeeping.fitTT
    fixed_nlayer = getattr(bookkeeping, "fixedNlayer", None)

    # Start from an empty model
    if fixed_nlayer is not None:
        model = Model.create_initial(prior=prior, Nlayer=int(fixed_nlayer), assumptions=bookkeeping.assumptions)
    else:
        model = Model.create_initial(prior=prior, assumptions=bookkeeping.assumptions)

    # Initial likelihood
    logL_trace = []

    logL_PP_trace, logL_SS_trace = [], []

    if fit_waveform:
        if bookkeeping.mode in (1, 2):
            logL = calc_like_prob(P, D, model, prior, bookkeeping, CDinv=CDinv)
        elif bookkeeping.mode == 3:
            logL, logL_PP, logL_SS = calc_like_prob_joint(
                P_PP, P_SS, D_PP, D_SS, 
                model, prior, bookkeeping, CDinv_PP=CDinv_PP, CDinv_SS=CDinv_SS)
            logL_PP_trace.append(logL_PP)
            logL_SS_trace.append(logL_SS)
    elif fit_travel_time:
        logL = calc_like_prob_travel_time(model, bookkeeping)
    else:
        logL = 0.0

    # fitgv and fitavgvs - only max 1 should be turned on
    if bookkeeping.fitgv: 
        logL_gv_trace = []
        logL_body_trace = []
        logL_body_trace.append(logL)
        logL_gv = calc_like_prob_gv(model, bookkeeping)
        logL_gv_trace.append(logL_gv)
        logL += logL_gv
    if bookkeeping.fitavgvs:
        logL_avg_vs_trace = []
        if not bookkeeping.fitgv:
            logL_body_trace = []
            logL_body_trace.append(logL)
        logL_avg_vs = calc_like_prob_avg_vs(model, bookkeeping)
        logL_avg_vs_trace.append(logL_avg_vs)
        logL += logL_avg_vs
    
    logL_trace.append(logL)

    start_time = time.time()
    checkpoint_interval = totalSteps // 100

    ensemble = []

    # Action pool
    actionPool = [2, 10] # H, v
    if fixed_nlayer is None and prior.maxN > 1:
        actionPool = np.append(actionPool, [0, 1]) # birth, death
    if fit_waveform:
        actionPool = np.append(actionPool, [3]) # w
        if bookkeeping.mode == 3: actionPool = np.append(actionPool, [4]) # w2
        if bookkeeping.fitLoge:
            if bookkeeping.mode in [1, 2]: actionPool = np.append(actionPool, [5]) # loge
            if bookkeeping.mode == 3: actionPool = np.append(actionPool, [5,6]) # loge and loge2
    if bookkeeping.fitLoge:
        if bookkeeping.fitgv:
            actionPool = np.append(actionPool, [7])
        if bookkeeping.fitavgvs:
            actionPool = np.append(actionPool, [8])
        if bookkeeping.fitTT:
            if bookkeeping.mode in [1, 2]:
                actionPool = np.append(actionPool, [9])
            if bookkeeping.mode == 3:
                actionPool = np.append(actionPool, [9, 12])
    if bookkeeping.mode == 3 or ((bookkeeping.fitgv or bookkeeping.fitavgvs) and bookkeeping.fitrho): actionPool = np.append(actionPool, [11]) # rho
    if top_layer_gradient_enabled(bookkeeping.assumptions):
        actionPool = np.append(actionPool, [13]) # top-layer gradient parameter

    for iStep in range(totalSteps):

        actions = np.random.choice(actionPool, size=actionsPerStep, replace=False)
        model_new = model

        for action in actions:
            if action == 0:
                model_new, _ = birth(model_new, prior, bookkeeping.rayp, assumptions=bookkeeping.assumptions)
            elif action == 1:
                model_new, _ = death(model_new, prior, bookkeeping.rayp, assumptions=bookkeeping.assumptions)
            elif action == 2:
                model_new, _ = update_H(model_new, prior, bookkeeping.rayp, assumptions=bookkeeping.assumptions)
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
                model_new, _ = update_loge_avg_vs(model_new, prior)
            elif action == 9:
                model_new, _ = update_loge_TT(model_new, prior)
            elif action == 10:
                model_new, _ = update_v(model_new, prior, bookkeeping.rayp, assumptions=bookkeeping.assumptions)
            elif action == 11:
                model_new, _ = update_rho(model_new, prior, bookkeeping.rayp, assumptions=bookkeeping.assumptions)
            elif action == 12:
                model_new, _ = update_loge_TT2(model_new, prior)
            elif action == 13:
                model_new, _ = update_a(model_new, prior, bookkeeping.rayp, assumptions=bookkeeping.assumptions)

        # Compute likelihood
        if fit_waveform:
            if bookkeeping.mode in (1, 2):
                new_logL = calc_like_prob(P, D, model_new, prior, bookkeeping, CDinv=CDinv)
            elif bookkeeping.mode == 3:
                new_logL, new_logL_PP, new_logL_SS = calc_like_prob_joint(
                    P_PP, P_SS, D_PP, D_SS, 
                    model_new, prior, bookkeeping, CDinv_PP=CDinv_PP, CDinv_SS=CDinv_SS)
        elif fit_travel_time:
            new_logL = calc_like_prob_travel_time(model_new, bookkeeping)
        else:
            new_logL = 0.0
        # Then add fitgv and fitavgvs as needed
        if bookkeeping.fitgv: 
            new_logL_gv = calc_like_prob_gv(model_new, bookkeeping)
            new_logL += new_logL_gv
        if bookkeeping.fitavgvs: 
            new_logL_avg_vs = calc_like_prob_avg_vs(model_new, bookkeeping)
            new_logL += new_logL_avg_vs

        # Acceptance probability
        log_accept_ratio = (new_logL - logL)

        if np.log(np.random.rand()) < log_accept_ratio:
            model = model_new
            logL = new_logL
            if bookkeeping.mode == 3:
                logL_PP = new_logL_PP
                logL_SS = new_logL_SS
            if bookkeeping.fitgv:
                logL_gv = new_logL_gv
            if bookkeeping.fitavgvs:
                logL_avg_vs = new_logL_avg_vs
        
        logL_trace.append(logL)
        if bookkeeping.mode == 3:
            logL_PP_trace.append(logL_PP)
            logL_SS_trace.append(logL_SS)
        if bookkeeping.fitgv:
            logL_gv_trace.append(logL_gv)
            logL_body_trace.append(logL - logL_gv)
        if bookkeeping.fitavgvs:
            logL_avg_vs_trace.append(logL_avg_vs)
            logL_body_trace.append(logL - logL_avg_vs)

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
            _savefig_with_retry(fig, os.path.join(saveDir, "logL.png"))  # overwrites each time
            plt.close(fig)

            # Also save waveform vs. group velocity misfit
            if bookkeeping.mode in (1,2):
                if bookkeeping.fitgv:
                    fig, ax = plt.subplots()
                    ax.plot(logL_body_trace, 'k-', label="body-wave")
                    ax.plot(logL_gv_trace, 'r-', label="group vel.")
                    _set_finite_ylim(ax, logL_body_trace, logL_gv_trace)
                    ax.set_xscale('log')
                    ax.set_xlabel("Step")
                    ax.set_ylabel("log Likelihood")
                    fig.tight_layout()
                    _savefig_with_retry(fig, os.path.join(saveDir, "logL_wf_gv.png"))  # overwrites each time
                    plt.close(fig)
                if bookkeeping.fitavgvs:
                    fig, ax = plt.subplots()
                    ax.plot(logL_body_trace, 'k-', label="body-wave")
                    ax.plot(logL_avg_vs_trace, 'r-', label="avg vs")
                    _set_finite_ylim(ax, logL_body_trace, logL_avg_vs_trace)
                    ax.set_xscale('log')
                    ax.set_xlabel("Step")
                    ax.set_ylabel("log Likelihood")
                    fig.tight_layout()
                    _savefig_with_retry(fig, os.path.join(saveDir, "logL_wf_avgvs.png"))  # overwrites each time
                    plt.close(fig)

            # Save PP/SS plots for mode 3
            if bookkeeping.mode == 3 and fit_waveform:
                fig, ax = plt.subplots()
                ax.plot(logL_PP_trace, 'b-', label="PP")
                ax.plot(logL_SS_trace, 'r-', label="SS")
                if bookkeeping.fitgv: ax.plot(logL_gv_trace, 'k-', label="group vel.")
                if bookkeeping.fitavgvs: ax.plot(logL_avg_vs_trace, 'k-', label="avg vs")
                _set_finite_ylim(
                    ax,
                    logL_PP_trace,
                    logL_SS_trace,
                    logL_gv_trace if bookkeeping.fitgv else [],
                    logL_avg_vs_trace if bookkeeping.fitavgvs else [],
                )
                ax.set_xscale('log')
                ax.set_xlabel("Step")
                ax.set_ylabel("log Likelihood")
                ax.legend(loc="upper right")
                fig.tight_layout()
                _savefig_with_retry(fig, os.path.join(saveDir, "logL_PPSS.png"))  # overwrites each time
                plt.close(fig)

            # Overwrite progress log
            elapsed = time.time() - start_time
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _append_progress_with_retry(
                os.path.join(saveDir, "progress_log.txt"),
                f"[{now}] Step {iStep+1}/{totalSteps}, Elapsed: {elapsed:.2f} sec\n",
            )

    return ensemble, logL_trace
