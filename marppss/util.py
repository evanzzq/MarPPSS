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

    # Strictly increasing velocity
    if not np.all(np.diff(model.v) >= 0.0):
        return False

    if len(rayp) == 1:
        # mode 1/2
        tau = 2.0 * H * np.sqrt(1.0 / (v**2) - rayp**2)
    else:
        # mode 3: use SS (because of longer travel time)
        tau = 2.0 * H * np.sqrt(1.0 / (v**2) - rayp[1]**2)
    
    arr = np.cumsum(tau)
    if arr[-1] >= prior.tlen:
        return False
    
    return True

def prepare_experiment(exp_vars):
    """
    Load data and prior for one experiment, based on exp_vars.
    Modifies and returns exp_vars dict.
    """

    import os, pickle
    from marppss.model import Prior, Bookkeeping

    filedir   = exp_vars["outdir"]
    mode = exp_vars["mode"]
    runname   = exp_vars["runname"]

    PPdir = exp_vars["PP_dir"]
    SSdir = exp_vars["SS_dir"]

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
        logeRange=tuple(exp_vars["logeRange"]),
        vRange=tuple(exp_vars["vRange"]),
        rhoRange=tuple(exp_vars["rhoRange"])
    )

    # --- Bookkeeping ---
    bookkeeping = Bookkeeping(
        mode=exp_vars["mode"],
        rayp=exp_vars["rayp"],
        fitRange=exp_vars["fitRange"],
        fitLoge=exp_vars["fitLoge"],
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

def prep_data(
        datadir, outdir, evname, dtype, 
        PPfreq, SSfreq, PParr, SSarr, cutwin, src_sigma,
        rotated=True, baz=None):
    """
    Prepare PP/SS data for inversion, including building
    noise covariance matrices (CD.csv) for PP and SS.

    Parameters
    ----------
    datadir : str
        Directory where SAC files live.
    outdir : str
        Base output directory.
    evname : str
        Event name prefix in SAC files.
    dtype : str
        Data type tag in SAC file names (e.g., 'PP', 'SS', 'syn', etc.).
    PPfreq, SSfreq : (fmin, fmax)
        Bandpass frequency ranges for Z/PP and T/SS.
    PParr, SSarr : obspy.UTCDateTime
        Arrival times of PP and SS.
    cutwin : (t0, t1)
        Time window (in seconds, relative to arrival) to cut data for P and D.
        Example: (-40, 40).
    src_sigma : float
        Standard deviation (in seconds) of Gaussian source time function.
    rotated : bool
        If False, rotate BHN/BHE to BHR/BHT using baz, and save rotated SACs.
    baz : float
        Backazimuth in degrees (needed if rotated==False).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from obspy import read, UTCDateTime
    from obspy.signal.rotate import rotate_ne_rt
    from scipy.signal import correlate
    from scipy.linalg import toeplitz
    from scipy.optimize import curve_fit

    # -------------------------
    # Helper: build covariance
    # -------------------------
    def build_noise_covariance(noise, dt, desired_length,
                               max_lag_seconds=50.0,
                               comp="Z", output_dir="."):
        """
        Build covariance matrix from a single (possibly long) noise trace using
        Kolb & Lekic (2014)-style parameterization, and output a covariance
        matrix of a specified desired length. Saves as output_dir/CD.csv.
        """
        n_samples = len(noise)
        max_lag = int(max_lag_seconds / dt)

        # Zero mean
        noise = noise - np.mean(noise)

        # Autocorrelation (non-negative lags only)
        acorr = correlate(noise, noise, mode="full")
        acorr = acorr[n_samples - 1:]      # keep non-negative lags
        acorr = acorr[:max_lag]            # truncate at max lag
        acorr /= n_samples                 # normalize

        # Time lags for fitting
        lags = np.arange(len(acorr)) * dt

        # Normalize for stable fitting
        acorr_norm = acorr / acorr[0]

        # Model: a * exp(-λτ) * cos(λω₀τ)
        def model(tau, a, lambd, omega0):
            return a * np.exp(-lambd * tau) * np.cos(lambd * omega0 * tau)

        try:
            popt, _ = curve_fit(
                model,
                lags,
                acorr_norm,
                p0=(1.0, 0.1, 2 * np.pi * 0.2),
                maxfev=10000
            )
            a_fit_norm, lambda_fit, omega0_fit = popt
            a_fit = a_fit_norm * acorr[0]  # rescale amplitude
        except RuntimeError as e:
            print(f"[WARN] Covariance fit failed for {comp}: {e}")
            return None

        # Generate full autocovariance for desired length
        full_lags = np.arange(desired_length) * dt
        acov_fit = a_fit * np.exp(-lambda_fit * full_lags) * np.cos(lambda_fit * omega0_fit * full_lags)

        # Toeplitz covariance matrix
        CD_fit = toeplitz(acov_fit)

        # Save
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(os.path.join(output_dir, "CD.csv"), CD_fit, delimiter=",")

        return CD_fit

    # -------------------------
    # 1. Setup and rotation
    # -------------------------
    t0, t1 = cutwin  # e.g., (-40, 40)

    if not rotated:
        st_z = read(os.path.join(datadir, f"{evname}_{dtype}_BHZ.SAC"))[0]
        st_n = read(os.path.join(datadir, f"{evname}_{dtype}_BHN.SAC"))[0]
        st_e = read(os.path.join(datadir, f"{evname}_{dtype}_BHE.SAC"))[0]

        tr_r, tr_t = rotate_ne_rt(st_n.data, st_e.data, baz)

        tr_r_obj = st_n.copy()
        tr_t_obj = st_e.copy()
        tr_r_obj.data = tr_r
        tr_t_obj.data = tr_t
        tr_r_obj.stats.channel = "BHR"
        tr_t_obj.stats.channel = "BHT"

        tr_r_obj.write(os.path.join(datadir, f"{evname}_{dtype}_BHR.SAC"), format="SAC")
        tr_t_obj.write(os.path.join(datadir, f"{evname}_{dtype}_BHT.SAC"), format="SAC")

    # -------------------------
    # 2. Reload Z and T from SAC
    # -------------------------
    st_z = read(os.path.join(datadir, f"{evname}_{dtype}_BHZ.SAC"))[0]
    st_t = read(os.path.join(datadir, f"{evname}_{dtype}_BHT.SAC"))[0]

    start_time = st_z.stats.starttime
    dt = st_z.stats.delta

    # Bandpass
    if PPfreq is not None and SSfreq is not None:
        st_z_filt = st_z.copy().filter("bandpass", freqmin=PPfreq[0], freqmax=PPfreq[1],
                                    corners=2, zerophase=True)
        st_t_filt = st_t.copy().filter("bandpass", freqmin=SSfreq[0], freqmax=SSfreq[1],
                                    corners=2, zerophase=True)
    else:
        st_z_filt, st_t_filt = st_z.copy(), st_t.copy()

    # Times (seconds from start_time)
    t_z = st_z_filt.times()
    t_t = st_t_filt.times()

    # Centers relative to trace start
    PParr, SSarr = UTCDateTime(PParr), UTCDateTime(SSarr)
    z_center = PParr - start_time   # seconds
    t_center = SSarr - start_time   # seconds

    # -------------------------
    # 3. Use cutwin for P/D/time
    # -------------------------
    idx_z = (t_z >= z_center + t0) & (t_z <= z_center + t1)
    idx_t = (t_t >= t_center + t0) & (t_t <= t_center + t1)

    t_zoom_z = t_z[idx_z] - z_center
    t_zoom_t = t_t[idx_t] - t_center

    # time axis (take from Z; T has same length)
    time = t_zoom_z.copy()
    npts_cut = len(time)

    # Gaussian window
    gaussian_window_z = np.exp(-0.5 * (t_zoom_z / src_sigma)**2)
    gaussian_window_t = np.exp(-0.5 * (t_zoom_t / src_sigma)**2)

    # Reference (P)
    P_z = st_z_filt.data[idx_z] * gaussian_window_z
    P_t = st_t_filt.data[idx_t] * gaussian_window_t

    # Data (D)
    D_z = st_z_filt.data[idx_z].copy()
    D_t = st_t_filt.data[idx_t].copy()

    # -------------------------
    # 4. Noise window (pre-arrival)
    # -------------------------
    noise_end = z_center + t0  # end = arrival + t0 (~ arrival - 40s)

    idx_noise = (t_z >= 0.0) & (t_z < noise_end)
    noise_z_full = st_z_filt.data[idx_noise]
    noise_t_full = st_t_filt.data[idx_noise]

    # -------------------------
    # 5. Normalize P, D, noise
    # -------------------------
    norm_z = np.max(np.abs(P_z)) if np.max(np.abs(P_z)) != 0 else 1.0
    norm_t = np.max(np.abs(P_t)) if np.max(np.abs(P_t)) != 0 else 1.0

    P_z /= norm_z
    D_z /= norm_z
    noise_z_full = noise_z_full / norm_z

    P_t /= norm_t
    D_t /= norm_t
    noise_t_full = noise_t_full / norm_t

    # -------------------------
    # 6. Save npz (PP & SS)
    # -------------------------
    save_suffix = f"_src_{src_sigma:.1f}_s"

    outdir_PP = os.path.join(outdir, "data", evname + save_suffix + "_PP")
    outdir_SS = os.path.join(outdir, "data", evname + save_suffix + "_SS")

    os.makedirs(outdir_PP, exist_ok=True)
    os.makedirs(outdir_SS, exist_ok=True)

    np.savez(os.path.join(outdir_PP, "data.npz"),
             P=P_z, D=D_z, noise=noise_z_full, time=time)
    np.savez(os.path.join(outdir_SS, "data.npz"),
             P=P_t, D=D_t, noise=noise_t_full, time=time)

    # -------------------------
    # 7. Build and save covariance (CD.csv)
    # -------------------------
    desired_len = len(time)

    CD_z = build_noise_covariance(
        noise=noise_z_full,
        dt=dt,
        desired_length=desired_len,
        comp="Z",
        output_dir=outdir_PP
    )

    CD_t = build_noise_covariance(
        noise=noise_t_full,
        dt=dt,
        desired_length=desired_len,
        comp="T",
        output_dir=outdir_SS
    )

    # -------------------------
    # 8. Quick QC plots
    # -------------------------

    # # --- random noise segments for the time window ---
    # def random_noise_segment(noise_full, segment_len):
    #     n_total = len(noise_full)
    #     if segment_len > n_total:
    #         raise ValueError("Segment length is longer than noise trace")
    #     start_idx = np.random.randint(0, n_total - segment_len + 1)
    #     return noise_full[start_idx:start_idx + segment_len]

    # noise_z_segment = random_noise_segment(noise_z_full, len(time))
    # noise_t_segment = random_noise_segment(noise_t_full, len(time))

    # # (a) Data / reference / noise
    # plt.figure(figsize=(12, 8))

    # plt.subplot(2, 1, 1)
    # plt.plot(time, D_z, label="D (filtered data)", color="gray")
    # plt.plot(time, P_z, label="P (Gaussian reference)", color="black", linestyle="--")
    # plt.plot(time, noise_z_segment, label="Noise (random segment)", color="red", alpha=0.7)
    # plt.title("Z Component: Filtered Data, Reference, and Noise")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # plt.grid()

    # plt.subplot(2, 1, 2)
    # plt.plot(time, D_t, label="D (filtered data)", color="gray")
    # plt.plot(time, P_t, label="P (Gaussian reference)", color="black", linestyle="--")
    # plt.plot(time, noise_t_segment, label="Noise (random segment)", color="red", alpha=0.7)
    # plt.title("T Component: Filtered Data, Reference, and Noise")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # plt.grid()

    # plt.tight_layout()
    # plt.show(block=False)
    # plt.pause(0.1)

    # # (b) Covariance matrices (if fit succeeded)
    # if (CD_z is not None) and (CD_t is not None):
    #     plt.figure(figsize=(12, 5))

    #     plt.subplot(1, 2, 1)
    #     plt.imshow(CD_z, origin="lower", cmap="viridis", aspect="auto")
    #     plt.colorbar(label="Covariance")
    #     plt.title("Covariance Matrix: Z Component")
    #     plt.xlabel("Sample index")
    #     plt.ylabel("Sample index")

    #     plt.subplot(1, 2, 2)
    #     plt.imshow(CD_t, origin="lower", cmap="viridis", aspect="auto")
    #     plt.colorbar(label="Covariance")
    #     plt.title("Covariance Matrix: T Component")
    #     plt.xlabel("Sample index")
    #     plt.ylabel("Sample index")

    #     plt.tight_layout()
    #     plt.show(block=False)
    #     plt.pause(0.1)
