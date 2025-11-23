import numpy as np

def PSVRTmatrix(p, mi, mt):
    """
    Python version of the MATLAB PSVRTmatrix function.

    Parameters
    ----------
    p : float or 1D numpy array
        Ray parameter (s/km or consistent units).
    mi : array-like of length 3
        [Vp_i, Vs_i, Rho_i] for incident medium.
    mt : array-like of length 3
        [Vp_t, Vs_t, Rho_t] for transmitted medium.

    Returns
    -------
    RTmatrix : ndarray, shape (8, len(p))
        [Rpp, Rps, Rss, Rsp, Tpp, Tps, Tss, Tsp]
        Each row is a coefficient vs p.
    """
    p = np.asarray(p, dtype=float)

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

    # Stack into 8 × N array (matching MATLAB’s `[Rpp' Rps' ...]`)
    RTmatrix = np.vstack([Rpp, Rps, Rss, Rsp, Tpp, Tps, Tss, Tsp])

    return RTmatrix
