import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from marppss.visualization import plot_velocity_ensemble, plot_posterior_num_phases, plot_predicted_vs_input, plot_posterior_error_params, plot_posterior_group_velocities, plot_velocity_density_image

# ==== Config ====
# filedir = "H:/My Drive/Research/MarPPSS"
filedir = "/Users/evanzhang/zzq@umd.edu - Google Drive/My Drive/Research/MarPPSS"

# ---- User-defined experiment/run ----
expname = "S0976asdr_1877_src_3.0_s_PP"   # folder under run/
runname = "run2_8c_maxN50_joint_gv_log_3_40"                     # subfolder under that

# ---- Explicit data dirs (user-specified) ----
PPdir = "S0976asdr_1877_src_3.0_s_PP"        # folder under data/
SSdir = "S0976asdr_1877_src_3.0_s_SS"        # folder under data/

PP_dir = os.path.join(filedir, "data", PPdir)
SS_dir = os.path.join(filedir, "data", SSdir)

# ---- Main results dir ----
saveDir = os.path.join(filedir, "run", expname, runname)

ensemble_filename = "ensemble.pkl"
logL_filename     = "log_likelihood.txt"

# ---- Load prior & bookkeeping ----
with open(os.path.join(saveDir, "prior.pkl"), "rb") as f:
    prior = pickle.load(f)

with open(os.path.join(saveDir, "bookkeeping.pkl"), "rb") as f:
    bookkeeping = pickle.load(f)

mode = bookkeeping.mode   # 1=PP, 2=SS, 3=joint

# -----------------------------------------------------------
#                    LOAD DATA FROM .npz
# -----------------------------------------------------------

if mode == 1:
    # PP only
    data = np.load(os.path.join(PP_dir, "data.npz"))
    P, D, time = data["P"], data["D"], data["time"]

elif mode == 2:
    # SS only
    data = np.load(os.path.join(SS_dir, "data.npz"))
    P, D, time = data["P"], data["D"], data["time"]

elif mode == 3:
    # Joint PP + SS
    data_PP = np.load(os.path.join(PP_dir, "data.npz"))
    P_PP, D_PP, time = data_PP["P"], data_PP["D"], data_PP["time"]

    data_SS = np.load(os.path.join(SS_dir, "data.npz"))
    P_SS, D_SS, _ = data_SS["P"], data_SS["D"], data_SS["time"]

    P = np.column_stack((P_PP, P_SS))
    D = np.column_stack((D_PP, D_SS))

else:
    raise ValueError(f"Invalid bookkeeping.mode = {mode}")

# -----------------------------------------------------------
#        DETERMINE SINGLE vs MULTI-CHAIN AUTOMATICALLY
# -----------------------------------------------------------

# Find all chain_* subdirectories
chain_dirs = []
for name in os.listdir(saveDir):
    full_path = os.path.join(saveDir, name)
    if os.path.isdir(full_path) and name.startswith("chain_"):
        # Extract numeric id after "chain_"
        try:
            cid = int(name.split("_", 1)[1])
            chain_dirs.append((cid, full_path))
        except ValueError:
            continue

# Sort by chain id
chain_dirs.sort(key=lambda x: x[0])
multi_chain = len(chain_dirs) > 0

ensemble_all = []

if multi_chain:
    # -------------------------------------------------------
    #                 MULTI-CHAIN: ANALYZE & LOAD
    # -------------------------------------------------------
    chain_logLs = []
    logL_series = {}

    for cid, cdir in chain_dirs:
        logL_file = os.path.join(cdir, logL_filename)
        if not os.path.exists(logL_file):
            continue

        with open(logL_file, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            continue

        logL_vals = [float(x.strip()) for x in lines]
        chain_logLs.append((cid, logL_vals[-1]))
        logL_series[cid] = logL_vals

    if len(chain_logLs) == 0:
        raise RuntimeError("No valid log_likelihood.txt found in any chain_* directory.")

    # Sort chains by final logL (descending)
    chain_logLs.sort(key=lambda x: x[1], reverse=True)
    # print("Chains sorted by final logL:", chain_logLs)

    # --- Plot histogram of final logL values ---
    plt.figure(figsize=(6, 4))
    plt.hist([x[1] for x in chain_logLs], bins=10, edgecolor="k")
    plt.xlabel("Final log-likelihood")
    plt.ylabel("Number of chains")
    plt.title("Histogram of chain final logL")
    plt.show()

    # --- Plot logL evolution for all chains ---
    plt.figure(figsize=(8, 5))
    for cid, vals in logL_series.items():
        plt.plot(range(len(vals)), vals, alpha=0.7, linewidth=0.2, color="black")
    plt.xscale('log')
    plt.xlabel("Step")
    plt.ylabel("Log-likelihood")
    plt.title("Log-likelihood evolution for all chains")
    plt.grid(True)
    plt.show()

    # Ask user how many top chains to use
    user_input = input("Enter number of top chains to use (Enter = all): ").strip()
    if user_input == "":
        top_ids = [cid for cid, _ in chain_logLs]
    else:
        n_top = int(user_input)
        top_ids = [cid for cid, _ in chain_logLs[:n_top]]

    print("Using chains:", top_ids)

    # Load ensembles from selected chains
    for cid, cdir in chain_dirs:
        if cid not in top_ids:
            continue
        with open(os.path.join(cdir, ensemble_filename), "rb") as f:
            ensemble = pickle.load(f)
        ensemble_all.extend(ensemble)

else:
    # -------------------------------------------------------
    #                 SINGLE-CHAIN: SIMPLE CASE
    # -------------------------------------------------------
    pkl_path = os.path.join(saveDir, ensemble_filename)
    if not os.path.exists(pkl_path):
        raise RuntimeError(f"Single-chain ensemble file not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        ensemble_all = pickle.load(f)

# -----------------------------------------------------------
#                       PLOT RESULTS
# -----------------------------------------------------------

plot_velocity_ensemble(ensemble_all, bookkeeping, prior.HRange, 
                       H_true=np.array([10, 24, 48]), v_true=np.array([3.8, 4.5, 6.22, 7.67]), rho_true=np.array([2.054, 1.607, 1.659, 1.771]))
plot_velocity_density_image(ensemble_all, bookkeeping, prior.HRange, nz=200, nv=200, smooth_sigma=2.0)

plot_predicted_vs_input(ensemble_all, P, D, prior, bookkeeping)
plot_posterior_error_params(ensemble_all, bookkeeping)
plot_posterior_num_phases(ensemble_all)

# # Group-velocity posteriors (using your S1000a values)
# periods = np.array([18.9, 28.86])
# gv_obs  = np.array([2.726, 2.829])

# reflectivity synthetics
periods = np.linspace(1.0, 40.0, 40)
gv_obs = np.array([
    1.72839653, 1.72837877, 1.72668886, 1.71786654, 1.69656754, 1.66145027,
    1.61350799, 1.55622542, 1.49543488, 1.43975198, 1.40036607, 1.38716733,
    1.40364349, 1.44419086, 1.49859321, 1.5582273 , 1.61862147, 1.67796528,
    1.73590863, 1.79222584, 1.84663725, 1.89899302, 1.94907331, 1.99699152,
    2.04305935, 2.08771873, 2.13145232, 2.17471385, 2.21789861, 2.26130128,
    2.30510902, 2.34945178, 2.39400196, 2.43886352, 2.48355818, 2.52794838,
    2.57183361, 2.61504436, 2.65710568, 2.69810677
])

if bookkeeping.fitgv:
    plot_posterior_group_velocities(
        ensemble_all,
        bookkeeping,
        periods=periods,
        gv_true=gv_obs,
        vpvsr=1.8
    )
