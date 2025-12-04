import os, pickle, time as pytime, numpy as np, multiprocessing as mp, yaml
from marppss.rjmcmc import rjmcmc_run
from marppss.model import Bookkeeping, Prior
from marppss.util import prepare_experiment, prep_data  # <-- add prep_data import
from obspy import UTCDateTime  # for PParr / SSarr strings

# -------- Chain function --------
def run_chain(chain_id, exp_vars):
    filedir    = exp_vars["outdir"]
    modname    = exp_vars["modname"]
    runname    = exp_vars["runname"]
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
    P            = exp_vars["P"]
    D            = exp_vars["D"]

    ensemble, logL_trace = rjmcmc_run(P, D, prior, bookkeeping, saveDir, CDinv=CDinv)

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

        # define experiment name / data_type
        if exp_vars["mode"] == 1:
            data_type = "PP"
        elif exp_vars["mode"] == 2:
            data_type = "SS"
        elif exp_vars["mode"] == 3:
            data_type = "joint"
        else:
            raise ValueError(f"Unknown mode: {exp_vars['mode']}")

        exp_vars["modname"] = exp_vars["event_name"] + "_" + data_type

        # ---- Prepare data using prep_data ----
        #
        # YAML provides:
        #   datadir, outdir, evname, dtype,
        #   PPfreq, SSfreq, PParr, SSarr, cutwin, src_sigma,
        #   rotated (optional), baz (optional)
        #

        # convert arrival times (strings) -> UTCDateTime
        PParr = UTCDateTime(exp_vars["PParr"])
        SSarr = UTCDateTime(exp_vars["SSarr"])

        # make sure these are tuples/lists of floats
        PPfreq = tuple(exp_vars["PPfreq"])
        SSfreq = tuple(exp_vars["SSfreq"])
        cutwin = tuple(exp_vars["cutwin"])
        src_sigma = float(exp_vars["src_sigma"])

        rotated = exp_vars.get("rotated", True)
        baz = exp_vars.get("baz", None)

        # --- Build datadir and outdir paths ---
        basedir = exp_vars["basedir"]

        # datadir = basedir / SharpSSPy / misc / evname
        datadir = os.path.join(
            basedir,
            *exp_vars["datadir_subpaths"],
            exp_vars["evname"]
        )

        # outdir = basedir / MarPPSS
        outdir = os.path.join(
            basedir,
            *exp_vars["outdir_subpaths"]
        )

        exp_vars["datadir"] = datadir
        exp_vars["outdir"]  = outdir

        print("datadir:", datadir)
        print("outdir :", outdir)

        # if evname not explicitly given, default to event_name
        evname = exp_vars.get("evname", exp_vars["event_name"])

        # ----------------------------------------------------------
        # Check whether data is already prepared -> skip prep_data()
        # ----------------------------------------------------------
        save_suffix = f"_src_{src_sigma:.1f}_s"
        PP_dir = os.path.join(exp_vars["outdir"], "data", evname + save_suffix + "_PP")
        SS_dir = os.path.join(exp_vars["outdir"], "data", evname + save_suffix + "_SS")

        PP_data_file = os.path.join(PP_dir, "data.npz")
        SS_data_file = os.path.join(SS_dir, "data.npz")

        PP_CD_file   = os.path.join(PP_dir, "CD.csv")
        SS_CD_file   = os.path.join(SS_dir, "CD.csv")

        useCD = exp_vars.get("useCD", False)

        # Conditions for skipping:
        data_ready = os.path.exists(PP_data_file) and os.path.exists(SS_data_file)
        cd_ready   = (not useCD) or (os.path.exists(PP_CD_file) and os.path.exists(SS_CD_file))

        if data_ready and cd_ready:
            print("✔ Data already prepared — skipping prep_data()")
        else:
            print("⚠ Data not found or incomplete — running prep_data()")
            prep_data(
                datadir=exp_vars["datadir"],
                outdir=exp_vars["outdir"],
                evname=evname,
                dtype=exp_vars["dtype"],
                PPfreq=PPfreq,
                SSfreq=SSfreq,
                PParr=PParr,
                SSarr=SSarr,
                cutwin=cutwin,
                src_sigma=src_sigma,
                rotated=rotated,
                baz=baz,
            )


        # ---- Load data, prior, and bookkeeping (unchanged) ----
        exp_vars = prepare_experiment(exp_vars)

        # --- Multiprocessing setup (unchanged) ---
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
