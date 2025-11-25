import numpy as np
import matplotlib.pyplot as plt
from marppss.forward import create_D_from_model

# from marppss.model import Model  # if you have it in a module

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
                           halfspace_extra=20.0,
                           alpha=0.1,
                           linewidth=1.0,
                           depthlim=50.0):
    """
    Plot an ensemble of velocity models (v vs depth) as step profiles.

    Parameters
    ----------
    models : list of Model
        Each Model has (Nlayer, H, v, rho). Only H and v are used.
    halfspace_extra : float
        Extra depth (km) to extend the last layer as a half-space.
    alpha : float
        Transparency for individual profiles.
    linewidth : float
        Line width for individual profiles.
    show_mean : bool
        If True, plot an approximate mean profile across the ensemble.
    """
    fig, ax = plt.subplots(figsize=(5, 7))

    all_vx = []
    all_z = []

    for m in models:
        vx, z = _model_to_step_profile(m.H, m.v, halfspace_extra=halfspace_extra)
        ax.plot(vx, z, color="C0", alpha=alpha, linewidth=linewidth)
        all_vx.append(vx)
        all_z.append(z)

    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("Depth (km)")
    if depthlim:
        ax.set_ylim(depthlim, 0)
    else:
        ax.set_ylim(ax.get_ylim()[::-1])  # depth increasing downward
    ax.grid(alpha=0.3)
    ax.set_title("Velocity ensemble")

    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_predicted_vs_input(ensemble, P, D_obs, prior, bookkeeping,
                            alpha_models=0.2, color_models='steelblue',
                            color_obs='k', lw_models=1.0, lw_obs=2.0,
                            flip_time=False):
    """
    Plot predicted vs input data (D).

    Parameters
    ----------
    ensemble : list of Model
        Ensemble of models to evaluate.
    P : np.ndarray
        Parent wavelet (input to forward modeling).
    D_obs : np.ndarray
        Observed / input data trace.
    prior : Prior
    bookkeeping : Bookkeeping
    alpha_models : float
        transparency for predicted curves.
    color_models : str
        color for predicted curves.
    color_obs : str
        color for observed data.
    lw_models : float
        line width for predicted curves.
    lw_obs : float
        line width for observed data.
    flip_time : bool
        If True, reverse the time axis.
    """

    n = len(D_obs)
    t = np.arange(n) * prior.dt

    plt.figure(figsize=(10, 6))

    # ---- Plot observed D ----
    plt.plot(t, D_obs, color=color_obs, lw=lw_obs, label="Observed D")

    # ---- Loop over all models and plot predicted D ----
    for model in ensemble:
        D_pred = create_D_from_model(P, model, prior, bookkeeping)
        plt.plot(t, D_pred, color=color_models, alpha=alpha_models, lw=lw_models)

    # ---- Axis formatting ----
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Predicted vs Input Data")

    if flip_time:
        plt.gca().invert_xaxis()

    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
