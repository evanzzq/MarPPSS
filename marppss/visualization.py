import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from marppss.forward import create_D_from_model

def _model_to_step_profile(H, v, halfspace_extra=20.0):
    """
    Convert one model (H, v) to (vx, z) for plotting v vs depth
    with step-like layers.

    Convention:
      - H: array of length Nlayer (thickness of each layer, km)
      - v: array of length Nlayer+1 (velocity at each node)
        Example: H=[10,20,30], v=[3,4,5,6]

    x-axis: velocity
    y-axis: depth (positive downward).
    """
    H = np.asarray(H, dtype=float)
    v = np.asarray(v, dtype=float)

    Nlayer = H.size

    if v.size != Nlayer + 1:
        raise ValueError(
            f"Expected len(v) = Nlayer + 1; got len(v)={v.size}, Nlayer={Nlayer}"
        )

    # depths at each node: 0, H0, H0+H1, ...
    depth_nodes = np.concatenate(([0.0], np.cumsum(H)))  # length Nlayer+1
    depth_bottom = depth_nodes[-1]
    depth_half = depth_bottom + halfspace_extra

    vx_list = []
    z_list = []

    # For each layer i = 0..Nlayer-1:
    #   vertical: (v[i], depth_nodes[i])   -> (v[i], depth_nodes[i+1])
    #   horizontal at interface: (v[i], depth_nodes[i+1]) -> (v[i+1], depth_nodes[i+1])
    for i in range(Nlayer):
        v_i   = v[i]
        v_ip1 = v[i + 1]
        z_top = depth_nodes[i]
        z_bot = depth_nodes[i + 1]

        # vertical segment in layer
        vx_list.extend([v_i, v_i])
        z_list.extend([z_top, z_bot])

        # horizontal segment at interface
        vx_list.extend([v_i, v_ip1])
        z_list.extend([z_bot, z_bot])

    # Half-space vertical extension at v[-1]
    vx_list.extend([v[-1], v[-1]])
    z_list.extend([depth_bottom, depth_half])

    return np.array(vx_list), np.array(z_list)


def plot_velocity_ensemble(models,
                           mode,
                           halfspace_extra=20.0,
                           alpha=0.1,
                           linewidth=1.0,
                           depthlim=50.0):
    """
    Plot an ensemble of velocity models (v vs depth) as step profiles.

    Parameters
    ----------
    models : list of Model
        Each Model has (Nlayer, H, v, rho).
        - For mode in [1, 2]: H = thickness, v = velocity.
        - For mode == 3:
            H   = thickness (km)
            v   = Vs
            rho = Vp/Vs ratio  (so Vp = Vs * rho)
    mode : int
        1 or 2 -> plot single-velocity ensemble using model.v
        3      -> plot Vs and Vp ensembles (Vs from model.v, Vp = Vs * model.rho)
    halfspace_extra : float
        Extra depth (km) to extend the last layer as a half-space.
    alpha : float
        Transparency for individual profiles.
    linewidth : float
        Line width for individual profiles.
    depthlim : float
        Maximum depth to show (km); plot is from 0 to depthlim.
    """
    fig, ax = plt.subplots(figsize=(5, 7))

    if mode in (1, 2):
        # Original behavior: single velocity field in model.v
        for m in models:
            vx, z = _model_to_step_profile(m.H, m.v, halfspace_extra=halfspace_extra)
            ax.plot(vx, z, color="C0", alpha=alpha, linewidth=linewidth)

        ax.set_title("Velocity ensemble")
        legend_handles = [Line2D([0], [0], color="C0", lw=linewidth, label="v")]
        ax.legend(handles=legend_handles, loc="lower right")

    elif mode == 3:
        # Vs and Vp ensemble
        first_vs = True
        first_vp = True

        for m in models:
            vs = np.asarray(m.v, dtype=float)
            ratio = np.asarray(m.rho, dtype=float)   # here rho = Vp/Vs
            vp = vs * ratio

            # Vs profile
            vx_vs, z_vs = _model_to_step_profile(m.H, vs, halfspace_extra=halfspace_extra)
            ax.plot(vx_vs, z_vs,
                    color="C0",
                    alpha=alpha,
                    linewidth=linewidth,
                    label="Vs" if first_vs else None)
            first_vs = False

            # Vp profile
            vx_vp, z_vp = _model_to_step_profile(m.H, vp, halfspace_extra=halfspace_extra)
            ax.plot(vx_vp, z_vp,
                    color="C1",
                    alpha=alpha,
                    linewidth=linewidth,
                    label="Vp" if first_vp else None)
            first_vp = False

        ax.set_title("Velocity ensemble (Vp & Vs)")
        ax.legend(loc="lower right")

    else:
        raise ValueError(f"Unsupported mode={mode}. Expected 1, 2, or 3.")

    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("Depth (km)")

    if depthlim:
        ax.set_ylim(depthlim, 0)  # depth increasing downward
    else:
        ax.set_ylim(ax.get_ylim()[::-1])

    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_predicted_vs_input(ensemble, P, D_obs, prior, bookkeeping):
    """
    Plot predicted vs input data (D).

    Parameters
    ----------
    ensemble : list of Model
        Ensemble of models to evaluate.
    P : np.ndarray
        Parent wavelet (input to forward modeling).
        - mode in [1, 2]: shape (n,)
        - mode == 3:      shape (n, 2), columns [PP, SS]
    D_obs : np.ndarray
        Observed / input data trace.
        - mode in [1, 2]: shape (n,)
        - mode == 3:      shape (n, 2), columns [PP, SS]
    prior : Prior
    bookkeeping : Bookkeeping
        Must contain attribute `mode`.
    """
    mode = bookkeeping.mode

    alpha_models=0.2
    color_models='steelblue'
    color_obs='k'
    lw_models=1.0
    lw_obs=2.0

    # Time axis from length of D_obs
    n = D_obs.shape[0]
    t = np.arange(n) * prior.dt

    if mode in (1, 2):
        # -------- Original behavior: single trace --------
        plt.figure(figsize=(10, 6))

        # Observed
        plt.plot(t, D_obs, color=color_obs, lw=lw_obs, label="Observed D")

        # Predicted for each model
        for model in ensemble:
            D_pred = create_D_from_model(P, model, prior, bookkeeping)
            plt.plot(t, D_pred, color=color_models, alpha=alpha_models, lw=lw_models)

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Predicted vs Input Data")

        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    elif mode == 3:
        # -------- Two-component case: PP and SS --------
        # Assume D_obs shape (n, 2): [PP, SS]
        D_PP_obs = D_obs[:, 0]
        D_SS_obs = D_obs[:, 1]

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax_pp, ax_ss = axes

        ax_pp.plot(t, D_PP_obs, color=color_obs, lw=lw_obs, label="Observed PP")
        ax_ss.plot(t, D_SS_obs, color=color_obs, lw=lw_obs, label="Observed SS")

        for model in ensemble:
            D_PP_pred, D_SS_pred = create_D_from_model(P, model, prior, bookkeeping)
            ax_pp.plot(t, D_PP_pred, color=color_models,
                       alpha=alpha_models, lw=lw_models)
            ax_ss.plot(t, D_SS_pred, color=color_models,
                       alpha=alpha_models, lw=lw_models)

        ax_pp.set_ylabel("Amplitude")
        ax_pp.set_title("Predicted vs Input Data (PP)")
        ax_pp.grid(True, alpha=0.3)
        ax_pp.legend(loc="upper right")

        ax_ss.set_xlabel("Time (s)")
        ax_ss.set_ylabel("Amplitude")
        ax_ss.set_title("Predicted vs Input Data (SS)")
        ax_ss.grid(True, alpha=0.3)
        ax_ss.legend(loc="upper right")

        plt.tight_layout()
        plt.show()

    else:
        raise ValueError(f"Unsupported mode={mode}. Expected 1, 2, or 3.")

def plot_posterior_error_params(ensemble, bookkeeping, bins=40, figsize=(6, 8), density=True):
    """
    Plot posterior histograms of error hyperparameters:
      - mode 1/2: loge
      - mode 3:   loge (PP) and loge2 (SS)
      - if bookkeeping.fitgv: also loge_gv

    Parameters
    ----------
    ensemble : list[Model]
        List of posterior samples (after burn-in).
    bookkeeping : object
        Has attributes:
          - mode (1, 2, or 3)
          - fitgv (bool)
    bins : int
        Number of histogram bins.
    figsize : tuple
        Figure size.
    density : bool
        If True, plot normalized histograms.
    """

    mode = bookkeeping.mode
    fitgv = bookkeeping.fitgv

    # Decide how many panels
    n_panels = 0
    plot_loge = False
    plot_loge2 = False
    plot_loge_gv = False

    if mode in (1, 2):
        plot_loge = True
        n_panels += 1
    elif mode == 3:
        plot_loge = True
        plot_loge2 = True
        n_panels += 2

    if fitgv:
        plot_loge_gv = True
        n_panels += 1

    if n_panels == 0:
        print("No error parameters requested for plotting.")
        return

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    ipanel = 0

    # --- loge ---
    if plot_loge:
        values = np.array([m.loge for m in ensemble])
        ax = axes[ipanel]
        ax.hist(values, bins=bins, density=density)
        if mode in (1, 2):
            ax.set_title(r"$\log e$ (waveform error)")
            ax.set_xlabel(r"$\log e$")
        else:
            ax.set_title(r"$\log e_{\mathrm{PP}}$ (PP error)")
            ax.set_xlabel(r"$\log e_{\mathrm{PP}}$")
        ax.set_ylabel("Density" if density else "Count")
        ipanel += 1

    # --- loge2 (mode 3 only) ---
    if plot_loge2:
        values = np.array([m.loge2 for m in ensemble])
        ax = axes[ipanel]
        ax.hist(values, bins=bins, density=density)
        ax.set_title(r"$\log e_{\mathrm{SS}}$ (SS error)")
        ax.set_xlabel(r"$\log e_{\mathrm{SS}}$")
        ax.set_ylabel("Density" if density else "Count")
        ipanel += 1

    # --- loge_gv (if fitgv) ---
    if plot_loge_gv:
        # Some early models may not have loge_gv; skip them safely
        vals = []
        for m in ensemble:
            if hasattr(m, "loge_gv"):
                vals.append(m.loge_gv)
        if len(vals) > 0:
            values = np.array(vals)
            ax = axes[ipanel]
            ax.hist(values, bins=bins, density=density)
            ax.set_title(r"$\log e_{\mathrm{gv}}$ (group-velocity error)")
            ax.set_xlabel(r"$\log e_{\mathrm{gv}}$")
            ax.set_ylabel("Density" if density else "Count")
        else:
            ax = axes[ipanel]
            ax.text(0.5, 0.5, "No loge_gv attribute in models", ha="center", va="center")
            ax.set_axis_off()

    plt.tight_layout()
    plt.show()


def compute_gv_for_model(model, bookkeeping, periods, vpvsr=1.8,
                         wave="rayleigh", mode=1, flat_earth=True):
    """
    Compute group velocities for a single Model using pysurf96.

    Parameters
    ----------
    model : Model
    bookkeeping : object
        Has attribute 'mode' (1, 2, or 3).
    periods : array_like
        Periods at which to compute group velocities.
    vpvsr : float
        Fixed Vp/Vs ratio for modes 1 and 2.
    wave : str
        surf96 wave type ("rayleigh" or "love").
    mode : int
        surf96 mode index (1 = fundamental).
    flat_earth : bool
        Passed to surf96.

    Returns
    -------
    gv_model : np.ndarray
        Group velocities for the given model at the specified periods.
        Shape: (len(periods),)
    """

    from pysurf96 import surf96

    periods = np.asarray(periods, dtype=float)

    # Layer thicknesses: append 0 for half-space
    H = np.append(model.H, 0.0)

    if bookkeeping.mode == 1:
        vp = np.asarray(model.v, dtype=float)
        vs = vp / vpvsr
    elif bookkeeping.mode == 2:
        vs = np.asarray(model.v, dtype=float)
        vp = vs * vpvsr
    elif bookkeeping.mode == 3:
        vs = np.asarray(model.v, dtype=float)
        # rho in this mode holds VP/VS ratio
        vp = vs * np.asarray(model.rho, dtype=float)
    else:
        raise ValueError(f"Unsupported bookkeeping.mode = {bookkeeping.mode}")

    # Simple rho model
    rho = vs * 0.8

    gv_model = surf96(
        H,
        vp,
        vs,
        rho,
        periods,
        wave=wave,
        mode=mode,
        velocity="group",
        flat_earth=flat_earth
    )

    return np.asarray(gv_model, dtype=float)

def plot_posterior_group_velocities(ensemble, bookkeeping,
                                    periods, gv_true,
                                    vpvsr=1.8,
                                    wave="rayleigh", mode_idx=1,
                                    bins=30, density=True,
                                    figsize=(6, 5)):
    """
    For each model in the ensemble, compute group velocities with surf96,
    then plot posterior histograms at each period, with vertical lines for
    the "true" values.

    Parameters
    ----------
    ensemble : list[Model]
        Posterior samples (after burn-in).
    bookkeeping : object
        Has attribute 'mode' (1, 2, or 3).
    periods : array_like
        Periods at which gv is measured. Shape (n_per,).
    gv_true : array_like
        True or observed group velocities at the same periods. Shape (n_per,).
    vpvsr : float
        Fixed Vp/Vs ratio for modes 1 and 2.
    wave : str
        Wave type for surf96 ("rayleigh" or "love").
    mode_idx : int
        Mode index for surf96 (1 = fundamental).
    bins : int
        Number of bins for each histogram.
    density : bool
        Whether to normalize histograms.
    figsize : tuple
        Base figure size; height will be scaled by number of periods.
    """

    periods = np.asarray(periods, dtype=float)
    gv_true = np.asarray(gv_true, dtype=float)

    if periods.shape != gv_true.shape:
        raise ValueError("periods and gv_true must have the same shape.")

    n_models = len(ensemble)
    n_per = len(periods)

    # Compute gv for every model
    gv_all = np.empty((n_models, n_per), dtype=float)
    for i, m in enumerate(ensemble):
        gv_all[i, :] = compute_gv_for_model(
            m, bookkeeping, periods,
            vpvsr=vpvsr, wave=wave, mode=mode_idx
        )

    fig, axes = plt.subplots(n_per, 1,
                             figsize=(figsize[0], figsize[1] * n_per),
                             squeeze=False)
    axes = axes.ravel()

    for iper in range(n_per):
        ax = axes[iper]
        vals = gv_all[:, iper]

        ax.hist(vals, bins=bins, density=density)
        ax.axvline(gv_true[iper], linestyle="--")  # true value

        ax.set_title(f"Group velocity at period = {periods[iper]:.2f} s")
        ax.set_xlabel("Group velocity (km/s)")
        ax.set_ylabel("Density" if density else "Count")

    plt.tight_layout()
    plt.show()