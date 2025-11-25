import os, pickle, time as pytime, numpy as np, multiprocessing as mp, yaml
from marppss.rjmcmc import rjmcmc_run
from marppss.model import Bookkeeping, Prior
from marppss.util import prepare_experiment

# -------- Chain function --------
def run_chain(chain_id, exp_vars):
    filedir    = exp_vars["filedir"]
    modname    = exp_vars["modname"]
    runname    = exp_vars["runname"]
    mode       = exp_vars["mode"]
    num_chains = exp_vars["num_chains"]

    # --- Save directory ---
    if num_chains == 1:
        saveDir = os.path.join(filedir, "run", modname, runname)
    else:
        saveDir = os.path.join(filedir, "run", modname, runname, f"chain_{chain_id}")
    os.makedirs(saveDir, exist_ok=True)

    # --- Extract experiment variables ---
    prior        = exp_vars["prior"]
    bookkeeping  = exp_vars["bookkeeping"]
    CDinv        = exp_vars.get("CDinv", None)
    CDinv_PP     = exp_vars.get("CDinv_PP", None)
    CDinv_SS     = exp_vars.get("CDinv_SS", None)

    if mode in (1, 2):
        P = exp_vars["P"]
        D = exp_vars["D"]
        ensemble, logL_trace = rjmcmc_run(P, D, prior, bookkeeping, saveDir, CDinv=CDinv)
    elif mode == 3:
        pass

    # --- Save results ---
    with open(os.path.join(saveDir, "ensemble.pkl"), "wb") as f:
        pickle.dump(ensemble, f)
    np.savetxt(os.path.join(saveDir, "log_likelihood.txt"), logL_trace)


# -------- Main run --------
if __name__ == "__main__":
    # --- Load YAML config ---
    with open("parameter_setup.yaml", "r") as f:
        config = yaml.safe_load(f)

    common_cfg = config.get("common", {})
    experiments = config.get("experiments", [])

    for exp_idx, exp_cfg in enumerate(experiments):
        print(f"\n=== Running experiment {exp_idx+1}/{len(experiments)}: {exp_cfg['event_name']} ===")

        # merge common and experiment
        exp_vars = {**common_cfg, **exp_cfg}

        # define experiment name
        if exp_vars["mode"] == 1: data_type = "PP"
        if exp_vars["mode"] == 2: data_type = "SS"
        if exp_vars["mode"] == 3: data_type = "joint"
        exp_vars["modname"] = exp_vars["event_name"] + "_" + data_type

        # --- Load data, prior, and bookkeeping ---
        exp_vars = prepare_experiment(exp_vars)

        # --- Multiprocessing setup ---
        try:
            import psutil
            cpu_cores = psutil.cpu_count(logical=False) or mp.cpu_count()
        except ImportError:
            cpu_cores = mp.cpu_count()
        print(f"Detected {cpu_cores} physical cores.")

        num_chains = exp_vars["num_chains"]
        if num_chains >= 2:
            threads_per_chain = max(1, cpu_cores // min(num_chains, cpu_cores))
            os.environ["OMP_NUM_THREADS"] = str(threads_per_chain)
            os.environ["MKL_NUM_THREADS"] = str(threads_per_chain)
            print(f"Setting {threads_per_chain} threads per chain to avoid oversubscription.")
        else:
            print("Single chain: allow full multithreading.")

        ctx = mp.get_context("spawn")
        batch_size = cpu_cores
        total_batches = (num_chains + batch_size - 1) // batch_size

        # --- Run all chains ---
        start = pytime.time()
        for batch_idx in range(total_batches):
            start_chain = batch_idx * batch_size
            end_chain = min(start_chain + batch_size, num_chains)
            batch_chain_ids = list(range(start_chain, end_chain))
            print(f"Running batch {batch_idx+1}/{total_batches} with chains {batch_chain_ids}")

            with ctx.Pool(processes=len(batch_chain_ids)) as pool:
                pool.starmap(run_chain, [(cid, exp_vars) for cid in batch_chain_ids])

        end = pytime.time()
        print(f"Experiment {exp_vars['event_name']} finished in {end - start:.2f} seconds")