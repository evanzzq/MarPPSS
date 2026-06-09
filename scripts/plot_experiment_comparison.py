"""Compare posterior velocity profiles and discontinuity depths across runs.

Edit the CONFIG block below, then run:

    python scripts/plot_experiment_comparison.py

Command-line arguments can still override the editable defaults for quick
one-off comparisons.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marppss.velocity import (
    _as_slope_array,
    all_layer_gradient_enabled,
    layer_velocity,
    top_layer_gradient_enabled,
    top_layer_velocity,
)


# =============================================================================
# Editable CONFIG
# =============================================================================

BASE_DIR = r"H:\My Drive\Research\MarPPSS\run\S1000a_pzfiltered_src_4.0_s_PP"

RUNS = {
    # Uncomment the experiments to plot. The value is top_chains for that run;
    # use None to keep all chains.
    # "s1000a_pp2_loge_off_step": 5, # None,
    # "s1000a_pp2_loge_off_grad": 5, # None,
    # #"s1000a_pp3_loge_off_linear": 5, # None,
    # "s1000a_pp2_loge_off_linear": 5, # None,
    # # "s1000a_pp3_loge_off_linear_pnp": 5, # 32,
    # # "s1000a_pp3_loge_off_step_pnp": 5, # 35,
    # # "s1000a_pp3_loge_off_sqrt_pnp": 5, # None,
    # "s1000a_pp3_loge_off_linear_pnp_maxD_150": 5, # 45,
    # "s1000a_pp3_loge_off_step_pnp_maxD_150": 5, # 47,
    # "s1000a_pp3_loge_off_sqrt_pnp_maxD_150": 5, # None,
    # "s1000a_pp3_loge_off_all_linear_ppp_maxD_150": 5, # 47, # all_linear
    # "s1000a_pp3_loge_off_all_linear_ppn_maxD_150": 5, # 7, # all_linear # xxx
    # "s1000a_pp3_loge_off_all_linear_pnp_maxD_150": 5, # 7, # all_linear_low # xxx
    # "s1000a_pp2_loge_off_all_linear_pp_maxD_150": 5, # 6, # all_linear
    # "s1000a_pp2_loge_off_all_linear_pn_maxD_150": 5, # 5, # all_linear_low
    # "s1000a_pp3_gv3_loge_off_all_linear_ppp": 5,
    # "s1000a_pp3_gv3_loge_off_all_linear_ppn": 5,
    "s1000a_pp3_gv3_loge_off_all_linear_pnp": 5,
    # "s1000a_pp2_gv3_loge_off_all_linear_pp": 5,
    "s1000a_pp2_gv3_loge_off_all_linear_pn": 5,
}

OUTPUT_DIR = "comparison_figures"
MAX_DEPTH = 100.0
PDF_DEPTH_LIMIT = 100.0
N_DEPTH = 300
H_BINS = 80
BY_LAYER = False


def _eval_profile_on_depths(H, v, depth_grid, a=0.0, assumptions=None):
    H = np.asarray(H, dtype=float)
    v = np.asarray(v, dtype=float)
    depth_grid = np.asarray(depth_grid, dtype=float)

    layer_tops = np.concatenate(([0.0], H))
    n_layer = H.size
    idx = np.searchsorted(layer_tops[1:], depth_grid, side="right")
    profile = v[idx].astype(float, copy=True)

    if all_layer_gradient_enabled(assumptions):
        slopes = _as_slope_array(a, n_layer)
        for layer_idx in range(n_layer):
            mask = idx == layer_idx
            if np.any(mask):
                profile[mask] = layer_velocity(
                    v[layer_idx],
                    slopes[layer_idx],
                    depth_grid[mask] - layer_tops[layer_idx],
                    assumptions=assumptions,
                )
    elif top_layer_gradient_enabled(assumptions):
        mask = idx == 0
        top_slope = _as_slope_array(a, n_layer)[0]
        profile[mask] = top_layer_velocity(v[0], top_slope, depth_grid[mask], assumptions=assumptions)

    return profile


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _chain_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    chains = []
    for child in run_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("chain_"):
            continue
        try:
            chain_id = int(child.name.split("_", 1)[1])
        except ValueError:
            continue
        chains.append((chain_id, child))
    return sorted(chains, key=lambda item: item[0])


def _selected_chain_ids(chains: list[tuple[int, Path]], top_chains: int | None) -> set[int]:
    if top_chains is None:
        return {chain_id for chain_id, _ in chains}

    scores = []
    for chain_id, chain_dir in chains:
        logl_path = chain_dir / "log_likelihood.txt"
        if not logl_path.exists():
            continue
        vals = np.atleast_1d(np.loadtxt(logl_path))
        if vals.size:
            scores.append((chain_id, float(vals[-1])))

    if not scores:
        return {chain_id for chain_id, _ in chains}

    scores.sort(key=lambda item: item[1], reverse=True)
    return {chain_id for chain_id, _ in scores[:top_chains]}


def load_run(run_dir: Path, top_chains: int | None = None) -> dict:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    bookkeeping = _load_pickle(run_dir / "bookkeeping.pkl")
    prior = _load_pickle(run_dir / "prior.pkl")

    chains = _chain_dirs(run_dir)
    models = []
    if chains:
        selected = _selected_chain_ids(chains, top_chains)
        for chain_id, chain_dir in chains:
            if chain_id in selected:
                models.extend(_load_pickle(chain_dir / "ensemble.pkl"))
    else:
        models = _load_pickle(run_dir / "ensemble.pkl")

    if not models:
        raise RuntimeError(f"No posterior models found in {run_dir}")

    return {
        "path": run_dir,
        "label": run_dir.name,
        "bookkeeping": bookkeeping,
        "prior": prior,
        "models": models,
    }


def _model_profiles(model, bookkeeping, depth_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return Vp, Vs, and Vp/Vs profiles for one model when available."""
    mode = bookkeeping.mode
    fitgv = getattr(bookkeeping, "fitgv", False)
    fitrho = getattr(bookkeeping, "fitrho", False)
    assumptions = getattr(bookkeeping, "assumptions", None)

    v = np.asarray(model.v, dtype=float)
    rho = getattr(model, "rho", None)
    has_rho = rho is not None and (mode == 3 or (fitgv and fitrho) or getattr(bookkeeping, "fitrho", False))

    if mode == 1:
        vp_layer = v
        a_vp = getattr(model, "a", 0.0)
        if has_rho:
            rho_layer = np.asarray(rho, dtype=float)
            vs_layer = vp_layer / rho_layer
            a_vs = _as_slope_array(a_vp, model.Nlayer) / rho_layer[:-1]
        else:
            rho_layer = None
            vs_layer = None
            a_vs = None
    elif mode in (2, 3):
        vs_layer = v
        a_vs = getattr(model, "a", 0.0)
        if has_rho:
            rho_layer = np.asarray(rho, dtype=float)
            vp_layer = vs_layer * rho_layer
            a_vp = _as_slope_array(a_vs, model.Nlayer) * rho_layer[:-1]
        else:
            rho_layer = None
            vp_layer = None
            a_vp = None
    else:
        raise ValueError(f"Unsupported bookkeeping.mode={mode}")

    vp = (
        _eval_profile_on_depths(model.H, vp_layer, depth_grid, a=a_vp, assumptions=assumptions)
        if vp_layer is not None
        else None
    )
    vs = (
        _eval_profile_on_depths(model.H, vs_layer, depth_grid, a=a_vs, assumptions=assumptions)
        if vs_layer is not None
        else None
    )
    ratio = (
        _eval_profile_on_depths(model.H, rho_layer, depth_grid, a=0.0, assumptions=assumptions)
        if rho_layer is not None
        else None
    )
    return vp, vs, ratio


def summarize_profiles(runs: list[dict], depth_grid: np.ndarray) -> dict:
    fields = {"vp": [], "vs": [], "ratio": []}
    for run in runs:
        for model in run["models"]:
            vp, vs, ratio = _model_profiles(model, run["bookkeeping"], depth_grid)
            if vp is not None:
                fields["vp"].append(vp)
            if vs is not None:
                fields["vs"].append(vs)
            if ratio is not None:
                fields["ratio"].append(ratio)

    summary = {}
    for key, profiles in fields.items():
        if not profiles:
            summary[key] = None
            continue
        arr = np.vstack(profiles)
        q16, q50, q84 = np.nanpercentile(arr, [16.0, 50.0, 84.0], axis=0)
        summary[key] = {"q16": q16, "q50": q50, "q84": q84}
    return summary


def plot_velocity_comparison(runs: list[dict], max_depth: float, output_path: Path, n_depth: int) -> None:
    depth_grid = np.linspace(0.0, max_depth, n_depth)
    fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=True)
    fields = [
        ("vp", "Vp (km/s)", "Vp"),
        ("vs", "Vs (km/s)", "Vs"),
        ("ratio", "Vp/Vs", "Vp/Vs"),
    ]

    summary = summarize_profiles(runs, depth_grid)
    color = "C0"
    for ax, (field, xlabel, title) in zip(axes, fields):
        item = summary[field]
        if item is None:
            continue
        ax.fill_betweenx(depth_grid, item["q16"], item["q84"], color=color, alpha=0.18, linewidth=0)
        ax.plot(item["q50"], depth_grid, color=color, linewidth=2.0)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Depth (km)")
    axes[0].set_ylim(max_depth, 0.0)
    fig.suptitle("Pooled Posterior Velocity Profiles")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _depth_pdf(depths: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(depths, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def plot_discontinuity_depth_pdf(
    runs: list[dict],
    depth_limit: float,
    output_path: Path,
    n_bins: int,
    by_layer_path: Path | None = None,
) -> None:
    bins = np.linspace(0.0, depth_limit, n_bins + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    depths = np.concatenate([
        np.asarray(model.H, dtype=float)
        for run in runs
        for model in run["models"]
    ])
    depths = depths[(depths >= 0.0) & (depths <= depth_limit)]
    if depths.size:
        centers, density = _depth_pdf(depths, bins)
        ax.plot(centers, density, color="C0", linewidth=2.0)

    ax.set_xlabel("Discontinuity depth H (km)")
    ax.set_ylabel("Posterior density")
    ax.set_title("Pooled Posterior PDF of Discontinuity Depths")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    if by_layer_path is None:
        return

    max_layers = max(max(model.Nlayer for model in run["models"]) for run in runs)
    fig, axes = plt.subplots(max_layers, 1, figsize=(8, 2.7 * max_layers), sharex=True, squeeze=False)
    for layer_idx in range(max_layers):
        ax = axes[layer_idx, 0]
        vals = [
            float(model.H[layer_idx])
            for run in runs
            for model in run["models"]
            if len(model.H) > layer_idx and 0.0 <= model.H[layer_idx] <= depth_limit
        ]
        if vals:
            centers, density = _depth_pdf(np.asarray(vals), bins)
            ax.plot(centers, density, color="C0", linewidth=1.8)
        ax.set_ylabel(f"H{layer_idx + 1} PDF")
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Discontinuity depth H (km)")
    fig.suptitle("Pooled Posterior PDF of Discontinuity Depths by Interface")
    fig.tight_layout()
    fig.savefig(by_layer_path, dpi=220)
    plt.close(fig)


def _resolve_run_path(value: str, base_dir: Path | None) -> Path:
    path = Path(_expand_path_vars(value)).expanduser()
    if path.is_absolute() or base_dir is None:
        return path
    return base_dir / path


def _expand_path_vars(value: str) -> str:
    expanded = os.path.expandvars(value)
    if "$env:" not in expanded:
        return expanded

    for key, env_value in os.environ.items():
        expanded = expanded.replace(f"$env:{key}", env_value)
        expanded = expanded.replace(f"$env:{key.lower()}", env_value)
        expanded = expanded.replace(f"$env:{key.upper()}", env_value)
    return expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create comparison plots for multiple MarPPSS experiment runs.",
    )
    parser.add_argument(
        "runs",
        nargs="*",
        help="Run directories, or run names when --base-dir is provided. Defaults to RUNS in the config block.",
    )
    parser.add_argument("--base-dir", help="Parent directory containing run directories. Defaults to BASE_DIR.")
    parser.add_argument("--output-dir", help="Directory for saved figures. Defaults to OUTPUT_DIR.")
    parser.add_argument("--max-depth", type=float, help="Maximum depth to show in velocity plots, in km.")
    parser.add_argument(
        "--pdf-depth-limit",
        type=float,
        help="Maximum discontinuity depth for the H posterior PDF. Defaults to --max-depth.",
    )
    parser.add_argument("--top-chains", type=int, help="Use only the top N chains by final log likelihood.")
    parser.add_argument("--n-depth", type=int, help="Number of depth samples for profile summaries.")
    parser.add_argument("--h-bins", type=int, help="Number of bins for discontinuity-depth PDFs.")
    parser.add_argument(
        "--by-layer",
        action="store_true",
        default=None,
        help="Also save a separate H1/H2/... PDF figure. Off by default.",
    )
    parser.add_argument(
        "--no-by-layer",
        action="store_false",
        dest="by_layer",
        help="Do not save the separate H1/H2/... PDF figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.runs:
        run_specs = [(name, args.top_chains) for name in args.runs]
    else:
        run_specs = list(RUNS.items())
    if not run_specs:
        raise ValueError("No runs were provided. Uncomment entries in RUNS or pass run names on the command line.")

    base_dir_value = args.base_dir if args.base_dir is not None else BASE_DIR
    base_dir = Path(_expand_path_vars(base_dir_value)).expanduser() if base_dir_value else None
    output_dir = Path(args.output_dir if args.output_dir is not None else OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_depth = args.max_depth if args.max_depth is not None else MAX_DEPTH
    pdf_depth_limit = args.pdf_depth_limit if args.pdf_depth_limit is not None else PDF_DEPTH_LIMIT
    if pdf_depth_limit is None:
        pdf_depth_limit = max_depth

    n_depth = args.n_depth if args.n_depth is not None else N_DEPTH
    h_bins = args.h_bins if args.h_bins is not None else H_BINS
    by_layer = args.by_layer if args.by_layer is not None else BY_LAYER

    runs = []
    for value, configured_top_chains in run_specs:
        run_top_chains = args.top_chains if args.top_chains is not None else configured_top_chains
        print(f"Loading {value} with top_chains={run_top_chains}")
        runs.append(load_run(_resolve_run_path(value, base_dir), top_chains=run_top_chains))

    velocity_path = output_dir / "velocity_profile_comparison.pdf"
    depth_pdf_path = output_dir / "discontinuity_depth_pdf.pdf"
    by_layer_path = output_dir / "discontinuity_depth_pdf_by_layer.pdf" if by_layer else None

    plot_velocity_comparison(runs, max_depth, velocity_path, n_depth)
    plot_discontinuity_depth_pdf(runs, pdf_depth_limit, depth_pdf_path, h_bins, by_layer_path=by_layer_path)

    print(f"Saved {velocity_path}")
    print(f"Saved {depth_pdf_path}")
    if by_layer_path is not None:
        print(f"Saved {by_layer_path}")


if __name__ == "__main__":
    main()
