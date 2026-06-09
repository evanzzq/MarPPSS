import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from marppss.forward import create_D_from_model, create_arrivals_from_model
from marppss.util import enforce_increasing_velocity
from marppss.velocity import (
    _as_slope_array,
    all_layer_gradient_enabled,
    disba_layers_from_model,
    layer_velocity,
    top_layer_gradient_enabled,
    top_layer_velocity,
)

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
                           true_kwargs=None,
                           show_summary=False,
                           summary_nz=250,
                           show=True):
    """
    Plot an ensemble of velocity models as step profiles.

    Behavior
    --------
    - mode in (1, 2) and not fitgv:
        one figure for single velocity
    - mode == 3 or (fitgv and fitrho):
        one figure for Vp & Vs
        one additional figure for Vp/Vs

    Optional reference/true model
    -----------------------------
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
    true_kwargs : dict or None
        Extra kwargs for plotting the true model, e.g.
        {"linewidth": 2.5, "linestyle": "--"}.

    Returns
    -------
    figs, axes : dict
        Dictionary of figures/axes:
          - single-velocity case: {"vel": (fig, ax)}
          - Vp/Vs case: {"vpvs": (fig_ratio, ax_ratio), "vp_vs": (fig_vel, ax_vel)}
    """
    mode = bookkeeping.mode
    fitgv = bookkeeping.fitgv
    fitrho = bookkeeping.fitrho
    H_min, H_max = HRange

    if true_kwargs is None:
        true_kwargs = {"linewidth": 2.5, "linestyle": "--"}

    outputs = {}

    # ---------------------------------------------------------
    # Case 1: single velocity only
    # ---------------------------------------------------------
    if mode in (1, 2) and not fitgv:
        fig, ax = plt.subplots(figsize=(5, 7))
        depth_grid, profiles = sample_models_to_depth_grid(
            models, bookkeeping, HRange, nz=summary_nz, field="velocity"
        )
        rgba, density, extent = _density_rgba(profiles, depth_grid, cmap_name="Blues", alpha_scale=0.9)
        ax.imshow(rgba, extent=extent, aspect="auto", interpolation="bilinear")
        sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0.0, vmax=1.0))
        sm.set_array(density)
        fig.colorbar(sm, ax=ax, label="Normalized posterior probability")

        if H_true is not None and v_true is not None:
            vx_true, z_true = _model_to_step_profile(H_true, v_true, HRange)
            ax.plot(vx_true, z_true, color="k", label="Reference", **true_kwargs)

        ax.set_title("Velocity ensemble")
        legend_handles = [Line2D([0], [0], color="C0", lw=4, label="Posterior density")]
        if H_true is not None and v_true is not None:
            legend_handles.append(
                Line2D([0], [0],
                       color="k",
                       lw=true_kwargs.get("linewidth", 2.5),
                       linestyle=true_kwargs.get("linestyle", "--"),
                       label="Reference")
            )
        ax.legend(handles=legend_handles, loc="lower right")

        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("Depth (km)")
        ax.set_ylim(H_max, 0.0)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        if show:
            plt.show()

        outputs["vel"] = (fig, ax)
        return outputs

    # ---------------------------------------------------------
    # Case 2: both Vp and Vs exist
    # ---------------------------------------------------------
    elif mode == 3 or (fitgv and fitrho):

        # ---------- Figure 1: Vp and Vs ----------
        fig_vel, ax_vel = plt.subplots(figsize=(5, 7))
        vs_profiles = []
        vp_profiles = []
        depth_grid = np.linspace(HRange[0], HRange[1], summary_nz)
        for m in models:
            vpvsr = np.asarray(m.rho, dtype=float)

            if mode == 1:
                vp = np.asarray(m.v, dtype=float)
                vs = vp / vpvsr
                a_vp = getattr(m, "a", 0.0)
                a_vs = _as_slope_array(a_vp, m.Nlayer) / vpvsr[:-1]
            elif mode in (2, 3):
                vs = np.asarray(m.v, dtype=float)
                vp = vs * vpvsr
                a_vs = getattr(m, "a", 0.0)
                a_vp = _as_slope_array(a_vs, m.Nlayer) * vpvsr[:-1]
            else:
                raise ValueError(f"Unsupported mode={mode}")
            vs_profiles.append(_eval_profile_on_depths(m.H, vs, depth_grid, a=a_vs, assumptions=bookkeeping.assumptions))
            vp_profiles.append(_eval_profile_on_depths(m.H, vp, depth_grid, a=a_vp, assumptions=bookkeeping.assumptions))

        vs_profiles = np.vstack(vs_profiles)
        vp_profiles = np.vstack(vp_profiles)
        vmin = float(min(np.nanmin(vs_profiles), np.nanmin(vp_profiles)))
        vmax = float(max(np.nanmax(vs_profiles), np.nanmax(vp_profiles)))
        rgba_vs, density_vs, extent = _density_rgba(vs_profiles, depth_grid, vmin=vmin, vmax=vmax, cmap_name="Blues", alpha_scale=0.85)
        rgba_vp, density_vp, _ = _density_rgba(vp_profiles, depth_grid, vmin=vmin, vmax=vmax, cmap_name="Oranges", alpha_scale=0.75)
        ax_vel.imshow(rgba_vs, extent=extent, aspect="auto", interpolation="bilinear")
        ax_vel.imshow(rgba_vp, extent=extent, aspect="auto", interpolation="bilinear")
        sm_vs = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0.0, vmax=1.0))
        sm_vs.set_array(density_vs)
        sm_vp = plt.cm.ScalarMappable(cmap="Oranges", norm=plt.Normalize(vmin=0.0, vmax=1.0))
        sm_vp.set_array(density_vp)
        fig_vel.colorbar(sm_vs, ax=ax_vel, fraction=0.046, pad=0.04, label="Vs normalized posterior probability")
        fig_vel.colorbar(sm_vp, ax=ax_vel, fraction=0.046, pad=0.12, label="Vp normalized posterior probability")

        # True model for Vp and Vs
        if H_true is not None and v_true is not None:
            if rho_true is None:
                raise ValueError("rho_true must be provided to plot true model when both Vp and Vs are present.")

            rho_true = np.asarray(rho_true, dtype=float)

            if mode == 1:
                vp_true = np.asarray(v_true, dtype=float)
                vs_true = vp_true / rho_true
            elif mode in (2, 3):
                vs_true = np.asarray(v_true, dtype=float)
                vp_true = vs_true * rho_true

            vx_vs_true, z_vs_true = _model_to_step_profile(H_true, vs_true, HRange)
            ax_vel.plot(vx_vs_true, z_vs_true,
                        color="navy", label="Vs true", **true_kwargs)

            vx_vp_true, z_vp_true = _model_to_step_profile(H_true, vp_true, HRange)
            ax_vel.plot(vx_vp_true, z_vp_true,
                        color="r", label="Vp true", **true_kwargs)

        ax_vel.set_title("Velocity ensemble (Vp & Vs)")
        ax_vel.set_xlabel("Velocity (km/s)")
        ax_vel.set_ylabel("Depth (km)")
        ax_vel.set_ylim(H_max, 0.0)
        ax_vel.grid(alpha=0.3)
        legend_handles = [
            Line2D([0], [0], color="C0", lw=4, label="Vs density"),
            Line2D([0], [0], color="C1", lw=4, label="Vp density"),
        ]
        if H_true is not None and v_true is not None:
            legend_handles.extend([
                Line2D([0], [0],
                       color="navy",
                       lw=true_kwargs.get("linewidth", 2.5),
                       linestyle=true_kwargs.get("linestyle", "--"),
                       label="Vs true"),
                Line2D([0], [0],
                       color="r",
                       lw=true_kwargs.get("linewidth", 2.5),
                       linestyle=true_kwargs.get("linestyle", "--"),
                       label="Vp true"),
            ])
        ax_vel.legend(handles=legend_handles, loc="lower right")

        plt.tight_layout()
        if show:
            plt.show()

        # ---------- Figure 2: Vp/Vs ----------
        fig_ratio, ax_ratio = plt.subplots(figsize=(5, 7))

        depth_grid, ratio_profiles = sample_models_to_depth_grid(
            models, bookkeeping, HRange, nz=summary_nz, field="ratio"
        )
        if _HAVE_KDE:
            rgba_ratio, density_ratio, extent, _ = _depthwise_kde_rgba(
                ratio_profiles,
                depth_grid,
                cmap_name="Blues",
                alpha_scale=0.9,
            )
            density_label = "Depth-normalized KDE density"
            density_legend_label = "Posterior KDE"
        else:
            rgba_ratio, density_ratio, extent = _density_rgba(
                ratio_profiles,
                depth_grid,
                cmap_name="Blues",
                alpha_scale=0.9,
            )
            density_label = "Normalized posterior probability"
            density_legend_label = "Posterior density"
        ax_ratio.imshow(rgba_ratio, extent=extent, aspect="auto", interpolation="bilinear")
        sm_ratio = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0.0, vmax=1.0))
        sm_ratio.set_array(density_ratio)
        fig_ratio.colorbar(sm_ratio, ax=ax_ratio, label=density_label)

        ratio_q16, ratio_median, ratio_q84 = np.nanpercentile(
            ratio_profiles,
            [16.0, 50.0, 84.0],
            axis=0,
        )
        ax_ratio.plot(ratio_median, depth_grid, color="k", linewidth=2.0, label="Vp/Vs median")
        ax_ratio.plot(ratio_q16, depth_grid, color="k", linewidth=1.0, linestyle=":", label="16-84% interval")
        ax_ratio.plot(ratio_q84, depth_grid, color="k", linewidth=1.0, linestyle=":")

        # True model for Vp/Vs
        if H_true is not None and rho_true is not None:
            rho_true = np.asarray(rho_true, dtype=float)
            vx_ratio_true, z_ratio_true = _model_to_step_profile(H_true, rho_true, HRange)
            ratio_true_kwargs = dict(true_kwargs)
            ratio_true_kwargs.setdefault("color", "crimson")
            ax_ratio.plot(vx_ratio_true, z_ratio_true, label="Vp/Vs true", **ratio_true_kwargs)

        legend_handles = [
            Line2D([0], [0], color="C0", lw=4, label=density_legend_label),
            Line2D([0], [0], color="k", lw=2.0, label="Vp/Vs median"),
            Line2D([0], [0], color="k", lw=1.0, linestyle=":", label="16-84% interval"),
        ]
        if H_true is not None and rho_true is not None:
            legend_handles.append(
                Line2D([0], [0],
                       color=ratio_true_kwargs.get("color", "crimson"),
                       lw=true_kwargs.get("linewidth", 2.5),
                       linestyle=true_kwargs.get("linestyle", "--"),
                       label="Vp/Vs true")
            )
        ax_ratio.legend(handles=legend_handles, loc="lower right")

        ax_ratio.set_title("Velocity ensemble (Vp/Vs)")
        ax_ratio.set_xlabel("Vp/Vs")
        ax_ratio.set_ylabel("Depth (km)")
        ax_ratio.set_ylim(H_max, 0.0)
        ax_ratio.grid(alpha=0.3)

        plt.tight_layout()
        if show:
            plt.show()

        outputs["vp_vs"] = (fig_vel, ax_vel)
        outputs["vpvs"] = (fig_ratio, ax_ratio)
        return outputs

    else:
        raise ValueError(f"Unsupported mode={mode}. Expected 1, 2, or 3.")

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import gaussian_filter
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

try:
    from scipy.stats import gaussian_kde
    _HAVE_KDE = True
except ImportError:
    _HAVE_KDE = False

def _eval_profile_on_depths(H, v, depth_grid, a=0.0, assumptions=None):
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
    profile = v[idx]
    if all_layer_gradient_enabled(assumptions):
        slopes = _as_slope_array(a, Nlayer)
        profile = profile.astype(float, copy=True)
        for i in range(Nlayer):
            layer_mask = idx == i
            if np.any(layer_mask):
                profile[layer_mask] = layer_velocity(
                    v[i],
                    slopes[i],
                    depth_grid[layer_mask] - layer_tops[i],
                    assumptions=assumptions,
                )
    elif top_layer_gradient_enabled(assumptions):
        top_mask = idx == 0
        profile = profile.astype(float, copy=True)
        top_slope = _as_slope_array(a, Nlayer)[0]
        profile[top_mask] = top_layer_velocity(v[0], top_slope, depth_grid[top_mask], assumptions=assumptions)
    return profile


def _density_rgba(profiles, depth_grid, vmin=None, vmax=None, nv=220, cmap_name="viridis", alpha_scale=0.85, gamma=0.8):
    depth_grid = np.asarray(depth_grid, dtype=float)
    profiles = np.asarray(profiles, dtype=float)

    if vmin is None:
        vmin = float(np.nanmin(profiles))
    if vmax is None:
        vmax = float(np.nanmax(profiles))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    z_samples = np.tile(depth_grid, profiles.shape[0])
    v_samples = profiles.ravel()
    mask = np.isfinite(z_samples) & np.isfinite(v_samples)
    z_samples = z_samples[mask]
    v_samples = v_samples[mask]

    z_edges = np.linspace(depth_grid.min(), depth_grid.max(), len(depth_grid) + 1)
    v_edges = np.linspace(vmin, vmax, nv + 1)
    density, _, _ = np.histogram2d(z_samples, v_samples, bins=[z_edges, v_edges], density=False)

    if _HAVE_SCIPY:
        density = gaussian_filter(density, sigma=1.2)

    density = density.astype(float)
    if np.nanmax(density) > 0:
        density /= np.nanmax(density)

    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(np.clip(density, 0.0, 1.0))
    rgba[..., 3] = np.clip(density, 0.0, 1.0) ** gamma * alpha_scale
    return rgba, density, (vmin, vmax, depth_grid.max(), depth_grid.min())


def _depthwise_kde_rgba(
    profiles,
    depth_grid,
    vmin=None,
    vmax=None,
    nv=240,
    cmap_name="Blues",
    alpha_scale=0.9,
    gamma=0.8,
    contrast_power=1.8,
):
    depth_grid = np.asarray(depth_grid, dtype=float)
    profiles = np.asarray(profiles, dtype=float)

    finite = profiles[np.isfinite(profiles)]
    if finite.size == 0:
        raise ValueError("No finite profile values are available for KDE plotting.")

    if vmin is None:
        vmin = float(np.nanpercentile(finite, 0.5))
    if vmax is None:
        vmax = float(np.nanpercentile(finite, 99.5))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        center = float(np.nanmedian(finite))
        pad = max(abs(center) * 0.02, 0.02)
        vmin = center - pad
        vmax = center + pad

    value_grid = np.linspace(vmin, vmax, nv)
    density = np.zeros((depth_grid.size, value_grid.size), dtype=float)

    for iz in range(depth_grid.size):
        values = profiles[:, iz]
        values = values[np.isfinite(values)]
        if values.size < 2:
            continue
        if np.nanmax(values) <= np.nanmin(values):
            idx = int(np.argmin(np.abs(value_grid - values[0])))
            density[iz, idx] = 1.0
            continue
        try:
            density[iz, :] = gaussian_kde(values)(value_grid)
        except Exception:
            hist, edges = np.histogram(values, bins=value_grid.size, range=(vmin, vmax), density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            density[iz, :] = np.interp(value_grid, centers, hist, left=0.0, right=0.0)

    row_max = np.nanmax(density, axis=1, keepdims=True)
    density_norm = np.divide(density, row_max, out=np.zeros_like(density), where=row_max > 0.0)
    density_display = np.clip(density_norm, 0.0, 1.0) ** contrast_power

    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(density_display)
    rgba[..., 3] = density_display ** gamma * alpha_scale
    return rgba, density_norm, (vmin, vmax, depth_grid.max(), depth_grid.min()), value_grid


def sample_models_to_depth_grid(models, bookkeeping, HRange, nz=200, field="auto"):
    """
    Sample all models in the ensemble on a common depth grid.

    field : {"auto", "velocity", "ratio"}
        Quantity to sample. "auto" preserves the current plotting convention.

    Returns
    -------
    depth_grid : (nz,)
    vel_profiles : (n_models, nz)
    """
    H_min, H_max = HRange
    depth_grid = np.linspace(H_min, H_max, nz)
    mode = bookkeeping.mode
    fitgv = bookkeeping.fitgv
    fitrho = bookkeeping.fitrho

    vel_profiles = []

    for m in models:
        if mode in (1, 2) and not fitgv:
            v_layer = np.asarray(m.v, dtype=float)
            a = getattr(m, "a", 0.0)
        elif mode == 3 or (fitgv and fitrho):
            ratio = np.asarray(m.rho, dtype=float)
            if field == "ratio":
                v_layer = ratio
                a = 0.0
            elif mode == 1:
                vp = np.asarray(m.v, dtype=float)
                v_layer = vp / ratio
                a = _as_slope_array(getattr(m, "a", 0.0), m.Nlayer) / ratio[:-1]
            else:
                v_layer = np.asarray(m.v, dtype=float)
                a = getattr(m, "a", 0.0)
        else:
            raise ValueError(f"Unsupported mode={mode} for sampling.")

        vel_profiles.append(_eval_profile_on_depths(m.H, v_layer, depth_grid, a=a, assumptions=bookkeeping.assumptions))

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
                                smooth_sigma=None,
                                H_true=None,
                                v_true=None,
                                rho_true=None,
                                true_kwargs=None,
                                field="auto",
                                show=True):
    """
    Make a 2D density map of velocity vs depth from an ensemble of models.

    Optional true model overlay:
    ----------------------------
    H_true : array-like
        Discontinuity depths of true model.
    v_true : array-like
        True model velocity array.
    rho_true : array-like or float
        True model Vp/Vs ratio, used when plotting Vp/Vs.
    true_kwargs : dict
        Plotting kwargs for true model overlay.
    """
    mode = bookkeeping.mode
    fitgv = bookkeeping.fitgv
    fitrho = bookkeeping.fitrho
    if field == "auto":
        field = "ratio" if (mode == 3 or (fitgv and fitrho)) else "velocity"
    if field == "ratio" and cmap == "viridis":
        cmap = "Blues"

    depth_grid, vel_profiles = sample_models_to_depth_grid(
        models, bookkeeping, HRange, nz=nz, field=field
    )
    H_min, H_max = HRange

    if vmin is None:
        vmin = np.nanmin(vel_profiles)
    if vmax is None:
        vmax = np.nanmax(vel_profiles)

    z_samples = np.tile(depth_grid, vel_profiles.shape[0])
    v_samples = vel_profiles.ravel()

    mask = np.isfinite(z_samples) & np.isfinite(v_samples)
    z_samples = z_samples[mask]
    v_samples = v_samples[mask]

    z_edges = np.linspace(HRange[0], HRange[1], nz + 1)
    v_edges = np.linspace(vmin, vmax, nv + 1)

    density, z_edges, v_edges = np.histogram2d(
        z_samples, v_samples,
        bins=[z_edges, v_edges],
        density=True,
    )

    try:
        if smooth_sigma is not None:
            from scipy.ndimage import gaussian_filter
            density = gaussian_filter(density, sigma=smooth_sigma)
    except ImportError:
        pass

    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])

    plt.figure(figsize=(5, 7))
    lo = np.percentile(density, 50)
    hi = np.percentile(density, 97)

    plt.pcolormesh(
        v_centers, z_centers, density,
        shading="auto", cmap=cmap, vmin=lo, vmax=hi
    )
    plt.colorbar(label="Density")

    # ----- true model overlay as step profile -----
    if H_true is not None and v_true is not None:
        if true_kwargs is None:
            color = "k" if field == "ratio" else "r"
            true_kwargs = {"color": color, "linestyle": "--", "linewidth": 2.0}

        if field == "velocity":
            true_layer = np.asarray(v_true, dtype=float)
        elif field == "ratio":
            if rho_true is None:
                raise ValueError("rho_true must be provided when plotting Vp/Vs.")
            true_layer = np.asarray(rho_true, dtype=float)
        else:
            raise ValueError(f"Unsupported density field={field!r}.")

        vx_true, z_true = _model_to_step_profile(H_true, true_layer, HRange)
        plt.plot(vx_true, z_true, label="True model", **true_kwargs)
        plt.legend()

    # label
    if field == "ratio":
        plt.xlabel("Vp/Vs")
        plt.title("Velocity ensemble density (Vp/Vs vs depth)")
    else:
        plt.xlabel("Velocity (km/s)")
        plt.title("Velocity ensemble density (v vs depth)")

    plt.ylim(0.0, H_max)
    plt.ylabel("Depth (km)")
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    if show:
        plt.show()

def _observed_travel_time_series(bookkeeping):
    travel_times = getattr(bookkeeping, "travel_times", None) or {}
    pp = travel_times.get("PP", {}) if travel_times else {}
    ss = travel_times.get("SS", {}) if travel_times else {}

    if bookkeeping.mode == 1:
        return [("PP", np.asarray(pp.get("times", []), dtype=float))]
    if bookkeeping.mode == 2:
        return [("SS", np.asarray(ss.get("times", []), dtype=float))]
    if bookkeeping.mode == 3:
        return [
            ("PP", np.asarray(pp.get("times", []), dtype=float)),
            ("SS", np.asarray(ss.get("times", []), dtype=float)),
        ]

    raise ValueError(f"Unsupported mode={bookkeeping.mode}. Expected 1, 2, or 3.")


def _model_travel_time_series(model, bookkeeping):
    arrivals = create_arrivals_from_model(model, bookkeeping)
    if bookkeeping.mode == 1:
        return [("PP", np.asarray(arrivals, dtype=float))]
    if bookkeeping.mode == 2:
        return [("SS", np.asarray(arrivals, dtype=float))]
    if bookkeeping.mode == 3:
        arr_PP, arr_SS = arrivals
        return [
            ("PP", np.asarray(arr_PP, dtype=float)),
            ("SS", np.asarray(arr_SS, dtype=float)),
        ]

    raise ValueError(f"Unsupported mode={bookkeeping.mode}. Expected 1, 2, or 3.")


def plot_posterior_travel_time_distribution(
    ensemble,
    bookkeeping,
    bins=40,
    density=True,
    figsize=(9, 5),
    alpha=0.35,
    show=True,
):
    """
    Plot posterior travel-time distributions with true/input picks overlaid.
    """
    observed_series = [(phase, obs) for phase, obs in _observed_travel_time_series(bookkeeping) if obs.size > 0]
    if not observed_series:
        raise ValueError("No observed travel times are available in bookkeeping.travel_times.")

    posterior = {
        f"{phase} {i + 1}": []
        for phase, obs in observed_series
        for i in range(obs.size)
    }
    observed = {
        f"{phase} {i + 1}": float(obs[i])
        for phase, obs in observed_series
        for i in range(obs.size)
    }

    valid_model_count = 0
    for model in ensemble:
        try:
            model_series = dict(_model_travel_time_series(model, bookkeeping))
        except Exception:
            continue

        model_ok = True
        for phase, obs in observed_series:
            arr = np.asarray(model_series.get(phase, []), dtype=float)
            if arr.shape != obs.shape or np.any(~np.isfinite(arr)):
                model_ok = False
                break

        if not model_ok:
            continue

        valid_model_count += 1
        for phase, obs in observed_series:
            arr = np.asarray(model_series[phase], dtype=float)
            for i, value in enumerate(arr):
                posterior[f"{phase} {i + 1}"].append(float(value))

    if valid_model_count == 0:
        raise ValueError("No finite model travel times matched the observed travel-time picks.")

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [f"C{i}" for i in range(10)])

    for i, (label, values) in enumerate(posterior.items()):
        values = np.asarray(values, dtype=float)
        color = colors[i % len(colors)]
        ax.hist(
            values,
            bins=bins,
            density=density,
            alpha=alpha,
            color=color,
            edgecolor=color,
            linewidth=0.6,
            label=f"{label} posterior",
        )
        ax.axvline(
            observed[label],
            color=color,
            linestyle="--",
            linewidth=2.0,
            label=f"{label} true",
        )

    ax.set_xlabel("Travel time (s)")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title("Posterior Distribution of Travel Times")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small", ncol=2)
    plt.tight_layout()
    if show:
        plt.show()

    return fig, ax

def plot_predicted_vs_input(ensemble, P, D_obs, prior, bookkeeping, show=True):
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
        if show:
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
        if show:
            plt.show()

    else:
        raise ValueError(f"Unsupported mode={mode}. Expected 1, 2, or 3.")

def plot_posterior_error_params(ensemble, bookkeeping, bins=40, figsize=(6, 8), density=True, show=True):
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
    fit_waveform = getattr(bookkeeping, "fitWaveform", not bookkeeping.fitTT)

    # Decide how many panels
    n_panels = 0
    plot_loge = False
    plot_loge2 = False
    plot_loge_gv = False
    plot_loge_avgvs = False
    plot_loge_TT = False
    plot_loge_TT2 = False

    if mode in (1, 2):
        if fit_waveform:
            plot_loge = True
        if bookkeeping.fitTT:
            plot_loge_TT = True
        n_panels += int(plot_loge) + int(plot_loge_TT)
    elif mode == 3:
        if fit_waveform:
            plot_loge = True
            plot_loge2 = True
        if bookkeeping.fitTT:
            plot_loge_TT = True
            plot_loge_TT2 = True
        n_panels += 2 * int(fit_waveform) + 2 * int(bookkeeping.fitTT)

    if bookkeeping.fitgv:
        plot_loge_gv = True
        n_panels += 1
    
    if bookkeeping.fitavgvs:
        plot_loge_avgvs = True
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
    
    # --- loge_TT ---
    if plot_loge_TT:
        values = np.array([m.loge_TT for m in ensemble])
        ax = axes[ipanel]
        ax.hist(values, bins=bins, density=density)
        if mode in (1, 2):
            ax.set_title(r"$\log e$ (travel time error)")
            ax.set_xlabel(r"$\log e$")
        else:
            ax.set_title(r"$\log e_{\mathrm{PP}}$ (PP travel time error)")
            ax.set_xlabel(r"$\log e_{\mathrm{PP}}$")
        ax.set_ylabel("Density" if density else "Count")
        ipanel += 1

    # --- loge_TT2 (mode 3 only) ---
    if plot_loge_TT2:
        values = np.array([m.loge_TT2 for m in ensemble])
        ax = axes[ipanel]
        ax.hist(values, bins=bins, density=density)
        ax.set_title(r"$\log e_{\mathrm{SS}}$ (SS travel time error)")
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
    
    # --- loge_avgvs (if fitgv) ---
    if plot_loge_avgvs:
        # Some early models may not have loge_gv; skip them safely
        vals = []
        for m in ensemble:
            if hasattr(m, "loge_avgvs"):
                vals.append(m.loge_avgvs)
        if len(vals) > 0:
            values = np.array(vals)
            ax = axes[ipanel]
            ax.hist(values, bins=bins, density=density)
            ax.set_title(r"$\log e_{\mathrm{avgvs}}$ (Average vs error)")
            ax.set_xlabel(r"$\log e_{\mathrm{avgvs}}$")
            ax.set_ylabel("Density" if density else "Count")
        else:
            ax = axes[ipanel]
            ax.text(0.5, 0.5, "No loge_avgvs attribute in models", ha="center", va="center")
            ax.set_axis_off()

    plt.tight_layout()
    if show:
        plt.show()

def plot_posterior_num_phases(ensemble, bins=None, figsize=(6, 4), density=False, show=True):
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
    if show:
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

    if getattr(bookkeeping, "fitrho", False) or bookkeeping.mode == 3:
        vpvsr_eff = np.asarray(model.rho, dtype=float)
    else:
        vpvsr_eff = np.asarray(vpvsr, dtype=float)

    if bookkeeping.mode not in (1, 2, 3):
        raise ValueError(f"Unsupported bookkeeping.mode = {bookkeeping.mode}")
    H, vp, vs, rho = disba_layers_from_model(model, bookkeeping, vpvsr_eff)

    if np.any(~np.isfinite(vp)) or np.any(~np.isfinite(vs)) or np.any(~np.isfinite(rho)):
        return np.full_like(periods, np.nan, dtype=float)
    if np.any(vp <= 0) or np.any(vs <= 0) or np.any(rho <= 0):
        return np.full_like(periods, np.nan, dtype=float)
    if np.any(np.asarray(vpvsr_eff) <= 0):
        return np.full_like(periods, np.nan, dtype=float)
    finite_thickness = H[:-1]
    if finite_thickness.size > 0:
        if np.any(finite_thickness <= 0) or np.any(finite_thickness < 0.05):
            return np.full_like(periods, np.nan, dtype=float)
    if enforce_increasing_velocity(getattr(bookkeeping, "assumptions", None)):
        if np.any(np.diff(vs) <= 0) or np.any(np.diff(vp) <= 0):
            return np.full_like(periods, np.nan, dtype=float)

    try:
        disp = GroupDispersion(H, vp, vs, rho)
        gv_model = disp(periods, mode=mode, wave=wave).velocity
    except Exception:
        return np.full_like(periods, np.nan, dtype=float)

    gv_model = np.asarray(gv_model, dtype=float)
    if gv_model.shape != periods.shape or np.any(~np.isfinite(gv_model)):
        return np.full_like(periods, np.nan, dtype=float)

    return gv_model


def plot_posterior_group_velocity_density(
    ensemble, bookkeeping, periods, gv_true=None,
    vpvsr=1.8, wave="rayleigh", mode_idx=0,
    n_vel=40, vel_pad_frac=0.05,
    figsize=None, cmap="viridis",
    show_colorbar=True,
    normalize_each_period=False,
    min_bin_count=2,
    min_bin_fraction=0.01,
    show=True,
):
    """
    Plot posterior group-velocity histograms, one subplot per period.
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

    valid_rows = np.all(np.isfinite(gv_all), axis=1)
    gv_all = gv_all[valid_rows]
    if gv_all.size == 0:
        raise ValueError("No finite group-velocity curves could be computed from the ensemble.")
    n_models = gv_all.shape[0]

    # common velocity grid so subplot shapes are visually comparable
    gv_min = np.min(gv_all)
    gv_max = np.max(gv_all)
    gv_span = gv_max - gv_min
    if gv_span <= 0.0:
        gv_span = max(abs(float(gv_min)) * 0.05, 0.05)
    gv_min -= vel_pad_frac * gv_span
    gv_max += vel_pad_frac * gv_span

    vel_edges = np.linspace(gv_min, gv_max, n_vel + 1)

    if figsize is None:
        ncols = min(4, n_per)
        nrows = int(np.ceil(n_per / ncols))
        figsize = (3.0 * ncols, 2.4 * nrows)
    else:
        ncols = min(4, n_per)
        nrows = int(np.ceil(n_per / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=True)
    axes = axes.ravel()

    if gv_true is not None:
        gv_true = np.asarray(gv_true, dtype=float)
        if gv_true.shape != periods.shape:
            raise ValueError("gv_true must have the same shape as periods.")

    ylabel = "Relative density" if normalize_each_period else "Density"
    for iper, ax in enumerate(axes[:n_per]):
        values = gv_all[:, iper]

        if normalize_each_period:
            counts, edges = np.histogram(values, bins=vel_edges, density=False)
            counts = counts.astype(float)
            if counts.max() > 0.0:
                counts /= counts.max()
            widths = np.diff(edges)
            ax.bar(edges[:-1], counts, width=widths, align="edge", color="steelblue", alpha=0.75, edgecolor="white")
        else:
            ax.hist(values, bins=vel_edges, density=True, color="steelblue", alpha=0.75, edgecolor="white")

        if gv_true is not None:
            ax.axvline(gv_true[iper], color="k", linestyle="--", linewidth=1.6, label="Observed")

        ax.set_title(f"{periods[iper]:g} s")
        ax.grid(True, alpha=0.25)
        if iper % ncols == 0:
            ax.set_ylabel(ylabel)
        if iper >= (nrows - 1) * ncols:
            ax.set_xlabel("Group velocity (km/s)")
        if gv_true is not None and iper == 0:
            ax.legend(loc="best", fontsize="small")

    for ax in axes[n_per:]:
        ax.set_axis_off()

    fig.suptitle("Posterior Distribution of Group Velocity")

    plt.tight_layout()
    if show:
        plt.show()

    return fig, axes[:n_per]
