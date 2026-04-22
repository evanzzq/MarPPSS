import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from marppss.config import resolve_experiment
from marppss.visualization import (
    plot_posterior_error_params,
    plot_posterior_group_velocity_density,
    plot_posterior_num_phases,
    plot_predicted_vs_input,
    plot_velocity_density_image,
    plot_velocity_ensemble,
)


def _load_yaml(path):
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to read saved run metadata. Install it with `pip install pyyaml`."
        ) from exc
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _load_pickled(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_run_config(run_dir):
    config_path = os.path.join(run_dir, "config_resolved.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Missing {config_path}. Plot CLI expects a run produced by the new workflow."
        )
    return _load_yaml(config_path)


def _load_observed_data(run_config, bookkeeping):
    mode = bookkeeping.mode
    pp_dir = run_config["PP_dir"]
    ss_dir = run_config["SS_dir"]

    if mode == 1:
        data = np.load(os.path.join(pp_dir, "data.npz"))
        return data["P"], data["D"]
    if mode == 2:
        data = np.load(os.path.join(ss_dir, "data.npz"))
        return data["P"], data["D"]

    data_pp = np.load(os.path.join(pp_dir, "data.npz"))
    data_ss = np.load(os.path.join(ss_dir, "data.npz"))
    P = np.column_stack((data_pp["P"], data_ss["P"]))
    D = np.column_stack((data_pp["D"], data_ss["D"]))
    return P, D


def _resolve_reference_model(reference, bookkeeping):
    if not reference:
        return None, None, None

    H_true = reference.get("H")
    if H_true is None:
        return None, None, None

    # Backward-compatible internal form.
    if reference.get("v") is not None:
        return H_true, reference.get("v"), reference.get("rho")

    vp_true = reference.get("vp")
    vs_true = reference.get("vs")

    if vp_true is None and vs_true is None:
        return H_true, None, None
    if vp_true is None or vs_true is None:
        raise ValueError("reference_model must provide both 'vp' and 'vs', or use the legacy 'v'/'rho' format.")

    vp_true = np.asarray(vp_true, dtype=float)
    vs_true = np.asarray(vs_true, dtype=float)
    if vp_true.shape != vs_true.shape:
        raise ValueError("reference_model 'vp' and 'vs' must have the same length.")
    if np.any(vs_true == 0):
        raise ValueError("reference_model 'vs' contains zero, so Vp/Vs cannot be computed.")

    rho_true = vp_true / vs_true
    mode = bookkeeping.mode
    if mode == 1:
        v_true = vp_true.tolist()
    elif mode in (2, 3):
        v_true = vs_true.tolist()
    else:
        raise ValueError(f"Unsupported mode={mode}")

    return H_true, v_true, rho_true.tolist()


def _load_latest_reference_model(run_config, bookkeeping, reference_config=None):
    reference = run_config.get("reference_model") or {}
    H_true, v_true, rho_true = _resolve_reference_model(reference, bookkeeping)
    if reference_config:
        experiment_name = run_config.get("experiment_name")
        if not experiment_name:
            raise ValueError("Saved run metadata is missing experiment_name, so --reference-config cannot be used.")
        resolved = resolve_experiment(reference_config, experiment_name)
        return _resolve_reference_model(resolved.get("reference_model") or {}, bookkeeping)

    if H_true is not None and v_true is not None:
        return H_true, v_true, rho_true

    config_path = run_config.get("config_path")
    experiment_name = run_config.get("experiment_name")
    if not config_path or not experiment_name:
        return H_true, v_true, rho_true

    try:
        resolved = resolve_experiment(config_path, experiment_name)
    except Exception:
        return H_true, v_true, rho_true

    return _resolve_reference_model(resolved.get("reference_model") or {}, bookkeeping)


def _collect_chains(run_dir, top_chains=None):
    chain_dirs = []
    for name in os.listdir(run_dir):
        full_path = os.path.join(run_dir, name)
        if os.path.isdir(full_path) and name.startswith("chain_"):
            try:
                cid = int(name.split("_", 1)[1])
            except ValueError:
                continue
            chain_dirs.append((cid, full_path))
    chain_dirs.sort(key=lambda item: item[0])

    if not chain_dirs:
        ensemble = _load_pickled(os.path.join(run_dir, "ensemble.pkl"))
        return ensemble, {}, []

    chain_logls = []
    logl_series = {}
    for cid, cdir in chain_dirs:
        logl_path = os.path.join(cdir, "log_likelihood.txt")
        if not os.path.exists(logl_path):
            continue
        values = np.loadtxt(logl_path)
        values = np.atleast_1d(values).tolist()
        if not values:
            continue
        chain_logls.append((cid, values[-1]))
        logl_series[cid] = values

    chain_logls.sort(key=lambda item: item[1], reverse=True)
    selected_ids = [cid for cid, _ in chain_logls]
    if top_chains is not None:
        selected_ids = selected_ids[:top_chains]

    ensemble_all = []
    for cid, cdir in chain_dirs:
        if cid not in selected_ids:
            continue
        ensemble_all.extend(_load_pickled(os.path.join(cdir, "ensemble.pkl")))

    return ensemble_all, logl_series, chain_logls


def _plot_chain_diagnostics(logl_series, chain_logls):
    if not chain_logls:
        return

    plt.figure(figsize=(6, 4))
    plt.hist([val for _, val in chain_logls], bins=10, edgecolor="k")
    plt.xlabel("Final log-likelihood")
    plt.ylabel("Number of chains")
    plt.title("Histogram of chain final logL")
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    for cid, vals in logl_series.items():
        plt.plot(range(len(vals)), vals, alpha=0.7, linewidth=0.2, color="black")
    plt.xscale("log")
    plt.xlabel("Step")
    plt.ylabel("Log-likelihood")
    plt.title("Log-likelihood evolution for all chains")
    plt.grid(True)
    plt.tight_layout()


def _prompt_top_chains(chain_logls):
    if not chain_logls:
        return None

    n_chains = len(chain_logls)
    while True:
        raw = input(f"Select top chains to plot [1-{n_chains}, 'all', Enter=all]: ").strip().lower()
        if raw in {"", "all", "a"}:
            return None
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer or 'all'.")
            continue
        if 1 <= value <= n_chains:
            return value
        print(f"Please choose a value between 1 and {n_chains}.")


def plot_run_directory(run_dir, top_chains=None, reference_config=None):
    run_config = _load_run_config(run_dir)
    prior = _load_pickled(os.path.join(run_dir, "prior.pkl"))
    bookkeeping = _load_pickled(os.path.join(run_dir, "bookkeeping.pkl"))

    _, logl_series, chain_logls = _collect_chains(run_dir, top_chains=None)
    _plot_chain_diagnostics(logl_series, chain_logls)
    plt.show()

    if top_chains is None:
        top_chains = _prompt_top_chains(chain_logls)

    ensemble_all, _, _ = _collect_chains(run_dir, top_chains=top_chains)

    P, D = _load_observed_data(run_config, bookkeeping)

    H_true, v_true, rho_true = _load_latest_reference_model(
        run_config,
        bookkeeping,
        reference_config=reference_config,
    )

    plot_velocity_ensemble(
        ensemble_all,
        bookkeeping,
        prior.HRange,
        H_true=H_true,
        v_true=v_true,
        rho_true=rho_true,
        show=False,
    )
    plot_velocity_density_image(
        ensemble_all,
        bookkeeping,
        prior.HRange,
        nz=200,
        nv=200,
        smooth_sigma=2.0,
        H_true=H_true,
        v_true=v_true,
        rho_true=rho_true,
        show=False,
    )
    plot_predicted_vs_input(ensemble_all, P, D, prior, bookkeeping, show=False)
    plot_posterior_error_params(ensemble_all, bookkeeping, show=False)
    plot_posterior_num_phases(ensemble_all, show=False)

    gv_cfg = run_config.get("group_velocity")
    if bookkeeping.fitgv and gv_cfg:
        plot_posterior_group_velocity_density(
            ensemble_all,
            bookkeeping,
            gv_cfg["periods"],
            gv_true=gv_cfg["values"],
            vpvsr=gv_cfg.get("vpvsr", 1.8),
            wave=gv_cfg.get("wave", "rayleigh"),
            mode_idx=gv_cfg.get("mode", 0),
            n_vel=200,
            show=False,
        )

    plt.show()
