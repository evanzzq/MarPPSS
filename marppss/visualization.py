import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from marppss.forward import create_D_from_model

def _model_to_step_profile(H, v, HRange):
    """
    Convert one model (H, v) to (vx, z) for plotting v vs depth
    with step-like layers.

    Convention (new):
      - H: array of length Nlayer (depths of discontinuities, km; increasing)
      - v: array of length Nlayer+1 (velocity in each layer, incl. half-space)
      - HRange: [H_min, H_max] prior range for discontinuity depths.
        We only plot down to H_max.

    Layers:
      Layer 0:   0       - H[0]   -> v[0]
      Layer 1:   H[0]    - H[1]   -> v[1]
      ...
      Layer N-1: H[N-2]  - H[N-1] -> v[N-1]
      Half-space: H[N-1] - HRange[1] -> v[N] (clipped at HRange[1])

    Returns
    -------
    vx, z : 1D arrays for step-plot (velocity vs depth).
    """
    H = np.asarray(H, dtype=float)
    v = np.asarray(v, dtype=float)
    H_min, H_max = HRange

    Nlayer = H.size

    if v.size != Nlayer + 1:
        raise ValueError(
            f"Expected len(v) = Nlayer + 1; got len(v)={v.size}, Nlayer={Nlayer}"
        )

    # Depth nodes at tops of layers: [0, H[0], H[1], ..., H[N-1]]
    depth_nodes = np.concatenate(([0.0], H))  # length Nlayer+1

    vx_list = []
    z_list = []

    # Finite layers
    for i in range(Nlayer):
        v_i   = v[i]
        v_ip1 = v[i + 1]
        z_top = depth_nodes[i]
        z_bot = depth_nodes[i + 1]

        # vertical segment in layer
        vx_list.extend([v_i, v_i])
        z_list.extend([z_top, z_bot])

        # horizontal jump at interface
        vx_list.extend([v_i, v_ip1])
        z_list.extend([z_bot, z_bot])

    # Half-space: from last discontinuity down to H_max (if deeper)
    depth_bottom = depth_nodes[-1]
    if depth_bottom < H_max:
        vx_list.extend([v[-1], v[-1]])
        z_list.extend([depth_bottom, H_max])

    return np.array(vx_list), np.array(z_list)

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

def plot_velocity_ensemble(models,
                           bookkeeping,
                           HRange,
                           alpha=0.1,
                           linewidth=1.0,
                           H_true=None,
                           v_true=None,
                           rho_true=None,
                           true_kwargs=None):
    """
    Plot an ensemble of velocity models (v vs depth) as step profiles.

    Optional reference/true model:
    ------------------------------
    H_true : array-like or None
        Depths of discontinuities for the reference model.
    v_true : array-like or None
        Layer velocities for the reference model.
        Meaning depends on mode, same as model.v:
          - mode 1: v_true = Vp
          - mode 2: v_true = Vs
          - mode 3: v_true = Vs
    rho_true : array-like, float, or None
        Vp/Vs ratio for the reference model when needed.
        Required if plotting both Vp and Vs from the true model.
    true_kwargs : dict or None
        Extra kwargs for plotting the true model, e.g.
        {"linewidth": 2.5, "linestyle": "--"}.
    """
    mode = bookkeeping.mode
    fitgv = bookkeeping.fitgv
    fitrho = bookkeeping.fitrho
    H_min, H_max = HRange

    fig, ax = plt.subplots(figsize=(5, 7))

    if true_kwargs is None:
        true_kwargs = {"linewidth": 2.5, "linestyle": "--"}

    if mode in (1, 2) and not fitgv:
        # Single-velocity ensemble
        for m in models:
            vx, z = _model_to_step_profile(m.H, m.v, HRange)
            ax.plot(vx, z, color="C0", alpha=alpha, linewidth=linewidth)

        # Reference model
        if H_true is not None and v_true is not None:
            vx_true, z_true = _model_to_step_profile(H_true, v_true, HRange)
            ax.plot(vx_true, z_true, color="k", label="Reference", **true_kwargs)

        ax.set_title("Velocity ensemble")
        legend_handles = [Line2D([0], [0], color="C0", lw=linewidth, label="v")]
        if H_true is not None and v_true is not None:
            legend_handles.append(
                Line2D([0], [0], color="k",
                       lw=true_kwargs.get("linewidth", 2.5),
                       linestyle=true_kwargs.get("linestyle", "--"),
                       label="Reference")
            )
        ax.legend(handles=legend_handles, loc="lower right")

    elif mode == 3 or (fitgv and fitrho):
        # Vs and Vp ensemble
        first_vs = True
        first_vp = True

        for m in models:
            vpvsr = np.asarray(m.rho, dtype=float)

            if mode == 1:
                vp = np.asarray(m.v, dtype=float)
                vs = vp / vpvsr
            elif mode in (2, 3):
                vs = np.asarray(m.v, dtype=float)
                vp = vs * vpvsr

            # Vs profile
            vx_vs, z_vs = _model_to_step_profile(m.H, vs, HRange)
            ax.plot(vx_vs, z_vs,
                    color="C0",
                    alpha=alpha,
                    linewidth=linewidth,
                    label="Vs" if first_vs else None)
            first_vs = False

            # Vp profile
            vx_vp, z_vp = _model_to_step_profile(m.H, vp, HRange)
            ax.plot(vx_vp, z_vp,
                    color="C1",
                    alpha=alpha,
                    linewidth=linewidth,
                    label="Vp" if first_vp else None)
            first_vp = False

        # Reference model
        if H_true is not None and v_true is not None:
            if rho_true is None:
                raise ValueError("rho_true must be provided to plot reference Vp/Vs model.")

            rho_true = np.asarray(rho_true, dtype=float)

            if mode == 1:
                vp_true = np.asarray(v_true, dtype=float)
                vs_true = vp_true / rho_true
            elif mode in (2, 3):
                vs_true = np.asarray(v_true, dtype=float)
                vp_true = vs_true * rho_true

            vx_vs_true, z_vs_true = _model_to_step_profile(H_true, vs_true, HRange)
            ax.plot(vx_vs_true, z_vs_true,
                    color="navy", label="Vs true", **true_kwargs)

            vx_vp_true, z_vp_true = _model_to_step_profile(H_true, vp_true, HRange)
            ax.plot(vx_vp_true, z_vp_true,
                    color="r", label="Vp true", **true_kwargs)

        ax.set_title("Velocity ensemble (Vp & Vs)")
        ax.legend(loc="lower right")

    else:
        raise ValueError(f"Unsupported mode={mode}. Expected 1, 2, or 3.")

    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("Depth (km)")
    ax.set_ylim(H_max, 0.0)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig, ax

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import gaussian_filter
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

def _eval_profile_on_depths(H, v, depth_grid):
    """
    Given layer discontinuities H (Nlayer,) and layer velocities v (Nlayer+1,),
    return velocity on a regular depth_grid (nz,).
    """
    H = np.asarray(H, dtype=float)
    v = np.asarray(v, dtype=float)
    depth_grid = np.asarray(depth_grid, dtype=float)

    # Layer tops: [0, H[0], H[1], ..., H[N-1]]
    layer_tops = np.concatenate(([0.0], H))
    Nlayer = H.size

    # For each depth, find which layer it belongs to
    # idx in [0..Nlayer], so v[idx] is correct layer velocity
    idx = np.searchsorted(layer_tops[1:], depth_grid, side="right")
    # idx = 0: above first discontinuity -> v[0]
    # idx = Nlayer: below last discontinuity -> v[-1]
    return v[idx]


def sample_models_to_depth_grid(models, bookkeeping, HRange, nz=200):
    """
    Sample all models in the ensemble on a common depth grid.

    Returns
    -------
    depth_grid : (nz,)
    vel_profiles : (n_models, nz)  # for one velocity field (v or Vs or Vp)
    """
    H_min, H_max = HRange
    depth_grid = np.linspace(H_min, H_max, nz)
    mode = bookkeeping.mode
    fitgv = bookkeeping.fitgv
    fitrho = bookkeeping.fitrho

    vel_profiles = []

    for m in models:
        if mode in (1, 2) and not fitgv:
            # Single velocity in m.v
            v_layer = np.asarray(m.v, dtype=float)
        elif mode == 3 or (fitgv and fitrho):
            # Example: use Vs (you can swap to Vp if you want)
            vp = np.asarray(m.v, dtype=float)
            ratio = np.asarray(m.rho, dtype=float)  # rho = Vp/Vs
            vs = vp / ratio
            v_layer = ratio
        else:
            raise ValueError(f"Unsupported mode={mode} for sampling.")

        vel_profiles.append(_eval_profile_on_depths(m.H, v_layer, depth_grid))

    vel_profiles = np.vstack(vel_profiles)  # (n_models, nz)
    return depth_grid, vel_profiles

def plot_velocity_density_image(models,
                                bookkeeping,
                                HRange,
                                nz=200,
                                nv=200,
                                vmin=None,
                                vmax=None,
                                cmap="viridis",
                                smooth_sigma=None):
    """
    Make a 2D density map of velocity vs depth from an ensemble of models.

    - y-axis: depth
    - x-axis: velocity
    - color: density of samples (probability-like)

    This is your "image-style" view instead of spaghetti.
    """
    from math import isfinite

    depth_grid, vel_profiles = sample_models_to_depth_grid(models, bookkeeping, HRange, nz=nz)

    if vmin is None:
        vmin = np.nanmin(vel_profiles)
    if vmax is None:
        vmax = np.nanmax(vel_profiles)

    # 2D histogram: we treat all (depth, velocity) samples together
    z_samples = np.tile(depth_grid, vel_profiles.shape[0])      # (n_models * nz,)
    v_samples = vel_profiles.ravel()

    # Remove NaNs/inf if any
    mask = np.isfinite(z_samples) & np.isfinite(v_samples)
    z_samples = z_samples[mask]
    v_samples = v_samples[mask]

    # Bin edges
    z_edges = np.linspace(HRange[0], HRange[1], nz + 1)
    v_edges = np.linspace(vmin, vmax, nv + 1)

    # Histogram in (depth, velocity)
    density, z_edges, v_edges = np.histogram2d(
        z_samples, v_samples,
        bins=[z_edges, v_edges],
        density=True,
    )   # density.shape = (nz, nv)

    # Optional smoothing (if you have SciPy)
    try:
        if smooth_sigma is not None:
            from scipy.ndimage import gaussian_filter
            density = gaussian_filter(density, sigma=smooth_sigma)
    except ImportError:
        pass  # silently ignore if SciPy not installed

    # Bin centers for plotting
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])

    # ---- Normal figure for viewing ----
    plt.figure(figsize=(5, 7))
    lo = np.percentile(density, 50)
    hi = np.percentile(density, 97)

    im = plt.pcolormesh(v_centers, z_centers, density, shading="auto",
                        cmap=cmap, vmin=lo, vmax=hi)
    plt.colorbar(label="Density")

    plt.xlabel("Velocity (km/s)")
    plt.ylabel("Depth (km)")
    # plt.xlim(1.75, 1.95)
    plt.gca().invert_yaxis()
    plt.title("Velocity ensemble density (v vs depth)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    # ---- Clean PNG export (no axes, no labels, no colorbar, no title) ----
    fig2, ax2 = plt.subplots(figsize=(5, 7))
    ax2.pcolormesh(v_centers, z_centers, density, shading="auto",
                cmap=cmap, vmin=lo, vmax=hi)

    # match limits
    # ax2.set_xlim(1.5, 8)
    ax2.invert_yaxis()

    # remove everything
    ax2.set_axis_off()

    plt.savefig("vel_ensemble_clean.png",
                dpi=400,
                bbox_inches='tight',
                pad_inches=0)
    plt.close(fig2)

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

def plot_posterior_num_phases(ensemble, bins=None, figsize=(6, 4), density=False):
    """
    Plot a posterior histogram of the number of phases (model.Nlayer) 
    in the ensemble (after burn-in).

    Parameters
    ----------
    ensemble : list[Model]
        List of posterior samples (after burn-in).
    bins : int or None
        Number of histogram bins. If None, automatically computed 
        from the discrete range of Nlayer values.
    figsize : tuple
        Figure size.
    density : bool
        If True, plot normalized histogram.
    """

    # Extract integer values
    nlayer_vals = np.array([m.Nlayer for m in ensemble])

    # Choose bins: if user did not specify, use one bin per integer value
    if bins is None:
        unique_vals = np.unique(nlayer_vals)
        # bins placed between integer phase counts
        bins = np.arange(unique_vals.min() - 0.5, unique_vals.max() + 1.5)

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(nlayer_vals, bins=bins, density=density, rwidth=0.9, align="mid")

    ax.set_title("Posterior Distribution of Number of Phases")
    ax.set_xlabel("Nlayer (Number of phases)")
    ax.set_ylabel("Density" if density else "Count")

    # Ensure ticks land on integer values
    ax.set_xticks(np.unique(nlayer_vals))

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from disba import GroupDispersion

def compute_gv_for_model(model, bookkeeping, periods, vpvsr=1.8,
                         wave="rayleigh", mode=0):
    """
    Compute group velocities for a single Model using disba.
    """

    periods = np.asarray(periods, dtype=float)

    # convert interface depths to layer thicknesses, then append 0 for half-space
    H_interfaces = np.asarray(model.H, dtype=float)
    if H_interfaces.size == 0:
        H = np.array([0.0])
    else:
        thickness = np.diff(np.r_[0.0, H_interfaces])
        H = np.r_[thickness, 0.0]

    if bookkeeping.mode == 1:
        vp = np.asarray(model.v, dtype=float)
        vs = vp / vpvsr
    elif bookkeeping.mode == 2:
        vs = np.asarray(model.v, dtype=float)
        vp = vs * vpvsr
    elif bookkeeping.mode == 3:
        vs = np.asarray(model.v, dtype=float)
        vp = vs * np.asarray(model.rho, dtype=float)
    else:
        raise ValueError(f"Unsupported bookkeeping.mode = {bookkeeping.mode}")

    rho = 0.8 * vs

    disp = GroupDispersion(H, vp, vs, rho)
    gv_model = disp(periods, mode=mode, wave=wave).velocity

    return np.asarray(gv_model, dtype=float)


def plot_posterior_group_velocity_density(
    ensemble, bookkeeping, periods, gv_true=None,
    vpvsr=1.8, wave="rayleigh", mode_idx=0,
    n_vel=200, vel_pad_frac=0.05,
    figsize=(7, 5), cmap="viridis",
    show_colorbar=True
):
    """
    Plot posterior group-velocity distributions in one panel:
    x = period, y = group velocity, color = posterior probability
    normalized independently for each period.

    So for each period, the maximum posterior density is scaled to 1.
    """

    periods = np.asarray(periods, dtype=float)
    n_models = len(ensemble)
    n_per = len(periods)

    # compute gv for all models
    gv_all = np.empty((n_models, n_per), dtype=float)
    for i, m in enumerate(ensemble):
        gv_all[i, :] = compute_gv_for_model(
            m, bookkeeping, periods,
            vpvsr=vpvsr, wave=wave, mode=mode_idx
        )

    # common velocity grid
    gv_min = np.min(gv_all)
    gv_max = np.max(gv_all)
    gv_span = gv_max - gv_min
    gv_min -= vel_pad_frac * gv_span
    gv_max += vel_pad_frac * gv_span

    vel_edges = np.linspace(gv_min, gv_max, n_vel + 1)

    # build image: rows = velocity bins, cols = periods
    Z = np.zeros((n_vel, n_per), dtype=float)

    for iper in range(n_per):
        hist, _ = np.histogram(gv_all[:, iper], bins=vel_edges, density=False)
        hist = hist.astype(float)

        # normalize each period slice by its own maximum
        if np.max(hist) > 0:
            hist /= np.max(hist)

        Z[:, iper] = hist

    # period edges for pcolormesh
    if n_per == 1:
        dp = 0.5
        per_edges = np.array([periods[0] - dp, periods[0] + dp])
    else:
        per_edges = np.empty(n_per + 1)
        per_edges[1:-1] = 0.5 * (periods[:-1] + periods[1:])
        per_edges[0] = periods[0] - 0.5 * (periods[1] - periods[0])
        per_edges[-1] = periods[-1] + 0.5 * (periods[-1] - periods[-2])

    fig, ax = plt.subplots(figsize=figsize)

    h = ax.pcolormesh(per_edges, vel_edges, Z, shading="auto", cmap=cmap, vmin=0, vmax=1)

    # overlay observed / true curve
    if gv_true is not None:
        gv_true = np.asarray(gv_true, dtype=float)
        if gv_true.shape != periods.shape:
            raise ValueError("gv_true must have the same shape as periods.")
        ax.plot(periods, gv_true, "w.-", linewidth=1.5, markersize=6, label="Observed")
        ax.legend()

    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Group velocity (km/s)")
    ax.set_title("Posterior distribution of group velocity")

    if show_colorbar:
        cbar = plt.colorbar(h, ax=ax)
        cbar.set_label("Normalized posterior density")

    plt.tight_layout()
    plt.show()