import multiprocessing as mp
import os
import pickle
import time as pytime

import numpy as np

from marppss.rjmcmc import rjmcmc_run
from marppss.util import prep_data, prepare_experiment


def mode_to_label(mode):
    return {1: "PP", 2: "SS", 3: "joint"}[mode]


def build_modname(evname, src_sigma, mode):
    return f"{evname}_src_{float(src_sigma):.1f}_s_{mode_to_label(mode)}"


def get_outdir(resolved):
    return os.path.join(resolved["basedir"], *resolved.get("outdir_subpaths", ["MarPPSS"]))


def get_datadir(resolved):
    return os.path.join(resolved["basedir"], *resolved.get("datadir_subpaths", ["SharpSSPy", "misc"]), resolved["evname"])


def get_data_dirs(resolved):
    outdir = get_outdir(resolved)
    suffix = f"_src_{float(resolved['src_sigma']):.1f}_s"
    return {
        "PP_dir": os.path.join(outdir, "data", f"{resolved['evname']}{suffix}_PP"),
        "SS_dir": os.path.join(outdir, "data", f"{resolved['evname']}{suffix}_SS"),
    }


def get_run_root(resolved):
    outdir = get_outdir(resolved)
    modname = build_modname(resolved["evname"], resolved["src_sigma"], resolved["mode"])
    return os.path.join(outdir, "run", modname, resolved["runname"])


def _serializable_resolved_config(resolved):
    data = {}
    for key, value in resolved.items():
        if key in {"P", "D", "prior", "bookkeeping", "CDinv"}:
            continue
        if isinstance(value, np.ndarray):
            data[key] = value.tolist()
        elif isinstance(value, tuple):
            data[key] = list(value)
        else:
            data[key] = value
    return data


def ensure_prepared_data(resolved, force=False):
    datadir = get_datadir(resolved)
    outdir = get_outdir(resolved)
    data_dirs = get_data_dirs(resolved)

    resolved["datadir"] = datadir
    resolved["outdir"] = outdir
    resolved.update(data_dirs)

    use_cd = resolved.get("useCD", False)
    pp_data = os.path.join(data_dirs["PP_dir"], "data.npz")
    ss_data = os.path.join(data_dirs["SS_dir"], "data.npz")
    pp_cd = os.path.join(data_dirs["PP_dir"], "CD.csv")
    ss_cd = os.path.join(data_dirs["SS_dir"], "CD.csv")

    data_ready = os.path.exists(pp_data) and os.path.exists(ss_data)
    cd_ready = (not use_cd) or (os.path.exists(pp_cd) and os.path.exists(ss_cd))

    if force or not (data_ready and cd_ready):
        prep_data(
            datadir=datadir,
            outdir=outdir,
            evname=resolved["evname"],
            dtype=resolved["dtype"],
            PPfreq=tuple(resolved["PPfreq"]) if resolved.get("PPfreq") is not None else None,
            SSfreq=tuple(resolved["SSfreq"]) if resolved.get("SSfreq") is not None else None,
            PParr=resolved["PParr"],
            SSarr=resolved["SSarr"],
            cutwin=tuple(resolved["cutwin"]),
            src_sigma=float(resolved["src_sigma"]),
            rotated=resolved.get("rotated", True),
            baz=resolved.get("baz"),
        )

    return resolved


def run_chain(chain_id, exp_vars):
    filedir = exp_vars["outdir"]
    modname = exp_vars["modname"]
    runname = exp_vars["runname"]
    num_chains = exp_vars["num_chains"]

    if num_chains == 1:
        save_dir = os.path.join(filedir, "run", modname, runname)
    else:
        save_dir = os.path.join(filedir, "run", modname, runname, f"chain_{chain_id}")
    os.makedirs(save_dir, exist_ok=True)

    ensemble, logL_trace = rjmcmc_run(
        exp_vars["P"],
        exp_vars["D"],
        exp_vars["prior"],
        exp_vars["bookkeeping"],
        save_dir,
        CDinv=exp_vars.get("CDinv"),
    )

    with open(os.path.join(save_dir, "ensemble.pkl"), "wb") as f:
        pickle.dump(ensemble, f)
    np.savetxt(os.path.join(save_dir, "log_likelihood.txt"), logL_trace)


def run_resolved_experiment(resolved, force_prep=False):
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to save resolved run metadata. Install it with `pip install pyyaml`."
        ) from exc

    resolved = ensure_prepared_data(resolved, force=force_prep)
    resolved["modname"] = build_modname(resolved["evname"], resolved["src_sigma"], resolved["mode"])

    exp_vars = prepare_experiment(dict(resolved))
    run_root = get_run_root(resolved)
    os.makedirs(run_root, exist_ok=True)

    with open(os.path.join(run_root, "config_resolved.yaml"), "w") as f:
        yaml.safe_dump(_serializable_resolved_config(exp_vars), f, sort_keys=False)

    try:
        import psutil

        cpu_cores = psutil.cpu_count(logical=False) or mp.cpu_count()
    except ImportError:
        cpu_cores = mp.cpu_count()

    num_chains = exp_vars["num_chains"]
    if num_chains >= 2:
        threads_per_chain = max(1, cpu_cores // min(num_chains, cpu_cores))
        os.environ["OMP_NUM_THREADS"] = str(threads_per_chain)
        os.environ["MKL_NUM_THREADS"] = str(threads_per_chain)

    ctx = mp.get_context("spawn")
    batch_size = cpu_cores
    total_batches = (num_chains + batch_size - 1) // batch_size

    start = pytime.time()
    for batch_idx in range(total_batches):
        start_chain = batch_idx * batch_size
        end_chain = min(start_chain + batch_size, num_chains)
        batch_chain_ids = list(range(start_chain, end_chain))
        with ctx.Pool(processes=len(batch_chain_ids)) as pool:
            pool.starmap(run_chain, [(cid, exp_vars) for cid in batch_chain_ids])
    end = pytime.time()

    summary = {
        "experiment_name": exp_vars.get("experiment_name"),
        "event_name": exp_vars.get("event_name"),
        "run_root": run_root,
        "elapsed_seconds": end - start,
        "num_chains": num_chains,
    }
    with open(os.path.join(run_root, "summary.yaml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    return run_root
