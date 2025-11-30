
import os, yaml, pickle
import numpy as np
from marppss.visualization import plot_velocity_ensemble, plot_predicted_vs_input

# ==== Config ====
# filedir = "H:/My Drive/Research/SharpSSPy"
filedir = "/Users/evanzhang/zzq@umd.edu - Google Drive/My Drive/Research/MarPPSS"

yaml_file = "parameter_setup.yaml"
ensemble_filename = "ensemble.pkl"   # or whatever name you use
logL_filename = "log_likelihood.txt"

# ---- Load YAML config ----
with open(yaml_file, "r") as f:
    config = yaml.safe_load(f)

common = config["common"]
experiments = config["experiments"]

# ---- Prompt user to choose an experiment ----
print("Available experiments:")
for i, exp in enumerate(experiments):
    display_name = f"{exp['event_name']}_mode{exp['mode']}_{exp['runname']}"
    print(f"{i}: {display_name}")

choice = int(input("Select experiment index: "))
exp_params = experiments[choice]

# ---- Combine common + experiment params ----
params = {**common, **exp_params}

# ---- Construct directories ----
event_name = params["event_name"]
mode = params["mode"]
runname = params["runname"]
num_chains = params["num_chains"]

PPdir = f"{event_name}_PP"
SSdir = f"{event_name}_SS"

if mode == 1: data_type = "PP"
if mode == 2: data_type = "SS"
if mode == 3: data_type = "joint"

saveDir = os.path.join(filedir, "run", f"{event_name}_{data_type}", runname)

with open(os.path.join(saveDir, "prior.pkl"), "rb") as f:
    prior = pickle.load(f)
with open(os.path.join(saveDir, "bookkeeping.pkl"), "rb") as f:
    bookkeeping = pickle.load(f)

# ---- Load data files ----
if mode in [1, 2]:
    if mode == 1: datadir = os.path.join(filedir, "data", PPdir)
    if mode == 2: datadir = os.path.join(filedir, "data", SSdir)
    npz_file = os.path.join(datadir, "data.npz")
    data = np.load(npz_file)
    P, D, time = data["P"], data["D"], data["time"]
elif mode == 3:
    # PP and SS dirs
    datadir_PP = os.path.join(filedir, "data", PPdir)
    datadir_SS = os.path.join(filedir, "data", SSdir)
    # Load data
    data_PP = np.load(os.path.join(datadir_PP, "data.npz"))
    P_PP, D_PP, time = data_PP["P"], data_PP["D"], data_PP["time"]
    data_SS = np.load(os.path.join(datadir_SS, "data.npz"))
    P_SS, D_SS, _ = data_SS["P"], data_SS["D"], data_SS["time"]
    P = np.column_stack((P_PP, P_SS))
    D = np.column_stack((D_PP, D_SS))

# ---- Collect ensembles from chains ----
ensemble_all = []
if num_chains > 1:
    chain_logLs = []
    logL_series = {}  # store full series per chain

    for i in range(num_chains):
        chain_dir = os.path.join(saveDir, f"chain_{i}")
        logL_file = os.path.join(chain_dir, logL_filename)
        if not os.path.exists(logL_file):
            continue
        with open(logL_file, "r") as f:
            lines = f.readlines()
            if len(lines) == 0:
                continue
            logL_vals = [float(x.strip()) for x in lines]
            logL = logL_vals[-1]  # final logL
        chain_logLs.append((i, logL))
        logL_series[i] = logL_vals

    # Sort chains by final logL (descending)
    chain_logLs.sort(key=lambda x: x[1], reverse=True)

    # --- Plot histogram of final logL values ---
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.hist([x[1] for x in chain_logLs], bins=10, edgecolor="k")
    plt.xlabel("Final log-likelihood")
    plt.ylabel("Number of chains")
    plt.title("Histogram of chain final logL")
    plt.show()

    # --- Plot full logL vs step for all chains ---
    plt.figure(figsize=(8,5))
    for cid, vals in logL_series.items():
        plt.plot(range(len(vals)), vals, alpha=0.7, linewidth=0.2, color="black")
    plt.xlabel("Step")
    plt.ylabel("Log-likelihood")
    plt.title("Log-likelihood evolution for all chains")
    plt.grid(True)
    plt.show()

    # Ask user for number of top chains
    user_input = input("Enter number of top chains to use (press Enter to use all): ")
    if user_input.strip() == "":
        top_chains = None
    else:
        top_chains = int(user_input)

    top_ids = [i for i, _ in chain_logLs]
    if top_chains is not None:
        top_ids = top_ids[:top_chains]
    
    print(f"Selected chains (sorted by final logL): {top_ids}")

    # Load only top chains
    for i in top_ids:
        chain_dir = os.path.join(saveDir, f"chain_{i}")
        with open(os.path.join(chain_dir, ensemble_filename), "rb") as f:
            ensemble = pickle.load(f)
        ensemble_all.extend(ensemble)

else:
    # Single-chain case
    with open(os.path.join(saveDir, ensemble_filename), "rb") as f:
        ensemble_all = pickle.load(f)

# ---- Plot ----
plot_velocity_ensemble(ensemble_all, mode)
plot_predicted_vs_input(ensemble_all, P, D, prior, bookkeeping)