import numpy as np

def PSVRTmatrix(p, mi, mt):
    """
    Inputs
    ----------
    p : float
        Ray parameter (s/km or consistent units).
    mi : array-like of length 3
        [Vp_i, Vs_i, Rho_i] for incident medium.
    mt : array-like of length 3
        [Vp_t, Vs_t, Rho_t] for transmitted medium.

    Returns
    -------
    Rpp, Rps, Rss, Rsp, Tpp, Tps, Tss, Tsp
    """
    # Unpack incident/transmitted medium properties
    Vpi, Vsi, rhoi = mi
    Vpt, Vst, rhot = mt

    # --- Vertical slownesses ---
    etaai = np.sqrt(1.0 / (Vpi * Vpi) - p * p)
    etaat = np.sqrt(1.0 / (Vpt * Vpt) - p * p)
    etabi = np.sqrt(1.0 / (Vsi * Vsi) - p * p)
    etabt = np.sqrt(1.0 / (Vst * Vst) - p * p)

    # --- Coefficients ---
    a = rhot * (1 - 2 * Vst * Vst * p * p) - rhoi * (1 - 2 * Vsi * Vsi * p * p)
    b = rhot * (1 - 2 * Vst * Vst * p * p) + 2 * rhoi * Vsi * Vsi * p * p
    c = rhoi * (1 - 2 * Vsi * Vsi * p * p) + 2 * rhot * Vst * Vst * p * p
    d = 2 * (rhot * Vst * Vst - rhoi * Vsi * Vsi)

    E = b * etaai + c * etaat
    F = b * etabi + c * etabt
    G = a - d * etaai * etabt
    H = a - d * etaat * etabi
    D = E * F + G * H * p * p

    # --- Reflection coefficients ---
    Rpp = ((b * etaai - c * etaat) * F - (a + d * etaai * etabt) * H * p * p) / D
    Rps = -(2 * etaai * (a * b + d * c * etaat * etabt) * p * (Vpi / Vsi)) / D
    Rss = -((b * etabi - c * etabt) * E - (a + d * etaat * etabi) * G * p * p) / D
    Rsp = -(2 * etabi * (a * b + d * c * etaat * etabt) * p * (Vsi / Vpi)) / D

    # --- Transmission coefficients ---
    Tpp =  (2 * rhoi * etaai * F * (Vpi / Vpt)) / D
    Tps =  (2 * rhoi * etaai * H * p * (Vpi / Vst)) / D
    Tss =   2 * rhoi * etabi * E * (Vsi / Vst) / D
    Tsp = -(2 * rhoi * etabi * G * p * (Vsi / Vpi)) / D

    return Rpp, Rps, Rss, Rsp, Tpp, Tps, Tss, Tsp

def SHmatrix(p, mi, mt):
    """
    Inputs
    ----------
    p : float
        Ray parameter (s/km or consistent units).
    mi : array-like of length 3
        [Vp_i, Vs_i, Rho_i] for incident medium.
    mt : array-like of length 3
        [Vp_t, Vs_t, Rho_t] for transmitted medium.

    Returns
    -------
    Rss, Tss
    """
    # Unpack incident/transmitted medium properties
    _, Vsi, rhoi = mi
    _, Vst, rhot = mt

    # --- Vertical slownesses ---
    etabi = np.sqrt(1.0 / (Vsi * Vsi) - p * p)
    etabt = np.sqrt(1.0 / (Vst * Vst) - p * p)

    # --- Coefficients ---
    delta = rhoi * Vsi * etabi + rhot * Vst * etabt

    # --- Reflection and transmission coefficients ---
    Rss = (rhoi * Vsi * etabi - rhot * Vst * etabt) / delta
    Tss = 2 * rhoi * Vsi * etabi / delta

    return Rss, Tss

def check_model(model, prior, rayp):

    H = np.asarray(model.H, dtype=float)
    v = np.asarray(model.v[:-1], dtype=float)

    if len(rayp) == 1:
        # mode 1/2
        tau = 2.0 * H * np.sqrt(1.0 / (v**2) - rayp**2)
    else:
        # mode 3: use SS (because of longer travel time)
        tau = 2.0 * H * np.sqrt(1.0 / (v**2) - rayp[1]**2)
    
    arr = np.cumsum(tau)
    if arr[-1] < prior.tlen:
        return True
    return False

def prepare_experiment(exp_vars):
    """
    Load data and prior for one experiment, based on exp_vars.
    Modifies and returns exp_vars dict.
    """

    import os, pickle
    from marppss.model import Prior, Bookkeeping

    filedir   = exp_vars["filedir"]
    event_name = exp_vars["event_name"]
    mode = exp_vars["mode"]
    runname   = exp_vars["runname"]

    PPdir = event_name + "_PP"
    SSdir = event_name + "_SS"

    CDinv, CDinv_PP, CDinv_SS = None, None, None

    # --- Load data ---
    if mode in [1, 2]:
        if mode == 1: datadir = os.path.join(filedir, "data", PPdir)
        if mode == 2: datadir = os.path.join(filedir, "data", SSdir)
        data = np.load(os.path.join(datadir, "data.npz"))
        exp_vars["P"], exp_vars["D"], time = data["P"], data["D"], data["time"]
        if exp_vars["useCD"]:
            CD = np.loadtxt(os.path.join(datadir, "CD.csv"), delimiter=",")
            # always "negOnly"
            half_len = CD.shape[0] // 2
            CD = CD[:half_len, :half_len]
            CDinv = np.linalg.pinv(CD)
            exp_vars["CDinv"] = CDinv
    elif mode == 3:
        # PP and SS dirs
        datadir_PP = os.path.join(filedir, "data", PPdir)
        datadir_SS = os.path.join(filedir, "data", SSdir)
        # Load data
        data_PP = np.load(os.path.join(datadir_PP, "data.npz"))
        P_PP, D_PP, time = data_PP["P"], data_PP["D"], data_PP["time"]
        data_SS = np.load(os.path.join(datadir_SS, "data.npz"))
        P_SS, D_SS, time_SS = data_SS["P"], data_SS["D"], data_SS["time"]
        # Sanity check on length
        if len(time_SS) != len(time) or (time_SS[1] - time_SS[0]) != (time[1] - time[0]):
            raise ValueError("Time vector for PP and SS don't match!")
        # Write to exp_vars
        exp_vars["P"] = np.column_stack((P_PP, P_SS))
        exp_vars["D"] = np.column_stack((D_PP, D_SS))
        # Load CD matrices
        if exp_vars["useCD"]:
            CD_PP = np.loadtxt(os.path.join(filedir, "data", PPdir, "CD.csv"), delimiter=",")
            CD_SS = np.loadtxt(os.path.join(filedir, "data", SSdir, "CD.csv"), delimiter=",")
            # Always use only negative (precursor) part
            half_len = CD_PP.shape[0] // 2
            CD_PP = CD_PP[:half_len, :half_len]
            half_len = CD_SS.shape[0] // 2
            CD_SS = CD_SS[:half_len, :half_len]
            CDinv_PP, CDinv_SS = np.linalg.pinv(CD_PP), np.linalg.pinv(CD_SS)
            exp_vars["CDinv"] = [CDinv_PP, CDinv_SS]

    # --- Prior ---
    dt = time[1] - time[0]
    tlen = (len(time) // 2) * dt
    prior = Prior(
        stdPP=exp_vars["stdPP"], stdSS=exp_vars["stdSS"], maxN=exp_vars["maxN"],
        tlen=tlen, dt=dt,
        HRange=tuple(exp_vars["HRange"]),
        wRange=tuple(exp_vars["wRange"]),
        vRange=tuple(exp_vars["vRange"]),
        rhoRange=tuple(exp_vars["rhoRange"])
    )

    # --- Bookkeeping ---
    bookkeeping = Bookkeeping(
        mode=exp_vars["mode"],
        rayp=exp_vars["rayp"],
        fitRange=exp_vars["fitRange"],
        totalSteps=exp_vars["totalSteps"],
        burnInSteps=exp_vars["burnInSteps"],
        nSaveModels=exp_vars["nSaveModels"],
        actionsPerStep=exp_vars["actionsPerStep"]
    )

    # --- Save ---
    sharedDir = os.path.join(filedir, "run", exp_vars["modname"], runname)
    os.makedirs(sharedDir, exist_ok=True)
    with open(os.path.join(sharedDir, "prior.pkl"), "wb") as f:
        pickle.dump(prior, f)
    with open(os.path.join(sharedDir, "bookkeeping.pkl"), "wb") as f:
        pickle.dump(bookkeeping, f)

    exp_vars["prior"] = prior
    exp_vars["bookkeeping"] = bookkeeping

    return exp_vars