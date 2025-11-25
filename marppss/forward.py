from marppss.model import Model, Prior, Bookkeeping
from marppss.util import PSVRTmatrix, SHmatrix
import numpy as np

def stretch_wavelet(P, stretch):
    """
    Stretch or compress wavelet P in time by 'stretch' factor.

    stretch > 1 → broader in time
    stretch < 1 → narrower in time

    Length is preserved, interpolation fills with zeros outside.
    """
    P = np.asarray(P, dtype=float)
    n = P.size
    x = np.arange(n, dtype=float)
    x0 = (n - 1) / 2.0

    # Map output indices back to input indices
    x_src = (x - x0) / stretch + x0

    return np.interp(x, x_src, P, left=0.0, right=0.0)

def create_D_from_model(P: np.ndarray, model: Model, prior: Prior, bookkeeping: Bookkeeping):
    """
    Create predicted D in the time domain.
    If bookkeeping.mode = 1/2, return one trace of D;
    If bookkeeping.mode = 3 (joint), return two traces of D.
    """
    # first create an empty D
    if prior.dt is None:
        raise ValueError("prior.dt must be set.")
    D = np.zeros_like(P)
    n = len(P)

    # check no. of layers
    if model.Nlayer < 1:
        raise ValueError("Model must have at least 1 layer.")

    if bookkeeping.mode in (1, 2): # PP or SS mode
        
        # initialize amplitude array
        amp = np.zeros(model.Nlayer, dtype=float)
        
        # first disc.
        # define velocities (1 - top, 2 - bottom)
        if bookkeeping.mode == 1:
            vp1, vp2 = model.v[0], model.v[1]
            vs1, vs2 = vp1 / 1.80, vp2 / 1.80 # vs shouldn't matter (much) in PP case
        else: # bookkeeping.mode == 2
            vs1, vs2 = model.v[0], model.v[1]
            vp1, vp2 = vs1 * 1.80, vs2 * 1.80 # vp shouldn't matter (much) in SS case
        # rho = vs * 0.8 from Li et al. (2022)
        rho1, rho2 = vs1 * 800, vs2 * 800
        # define layer model for R/T function input
        m1 = [vp1, vs1, rho1]
        m2 = [vp2, vs2, rho2]
        # R/T coef. and amplitude
        if bookkeeping.mode == 1:
            _, _, _, _, TppD, _, _, _    = PSVRTmatrix(bookkeeping.rayp ,m1, m2)
            RppU, _, _, _, TppU, _, _, _ = PSVRTmatrix(bookkeeping.rayp ,m2, m1)
            amp[0] = - RppU / (TppU * TppD)
        else: # bookkeeping.mode == 2
            _, TssD    = SHmatrix(bookkeeping.rayp ,m1, m2)
            RssU, TssU = SHmatrix(bookkeeping.rayp ,m2, m1)
            amp[0] = RssU / (TssU * TssD)

        # deeper disc. (if any)
        for k in range(1, model.Nlayer):

            # same as first disc.
            # define velocities (1 - top, 2 - bottom)
            if bookkeeping.mode == 1:
                vp1, vp2 = model.v[k], model.v[k+1]
                vs1, vs2 = vp1 / 1.80, vp2 / 1.80 # vs shouldn't matter (much) in PP case
            else: # bookkeeping.mode == 2
                vs1, vs2 = model.v[k], model.v[k+1]
                vp1, vp2 = vs1 * 1.80, vs2 * 1.80 # vp shouldn't matter (much) in SS case
            # rho = vs * 0.8 from Li et al. (2022)
            rho1, rho2 = vs1 * 800, vs2 * 800
            # define layer model for R/T function input
            m1 = [vp1, vs1, rho1]
            m2 = [vp2, vs2, rho2]
            # R/T coef. and amplitude
            if bookkeeping.mode == 1:
                _, _, _, _, TppD, _, _, _    = PSVRTmatrix(bookkeeping.rayp ,m1, m2)
                RppU, _, _, _, TppU, _, _, _ = PSVRTmatrix(bookkeeping.rayp ,m2, m1)
                amp[k] = amp[k-1] * (RppU / (TppU * TppD))
            else: # bookkeeping.mode == 2
                _, TssD    = SHmatrix(bookkeeping.rayp ,m1, m2)
                RssU, TssU = SHmatrix(bookkeeping.rayp ,m2, m1)
                amp[k] = amp[k-1] * (RssU / (TssU * TssD))
        
        # get arrival time(s)
        # for mode 1/2 (PP or SS), it doesn't matter if it's vp or vs
        H = np.asarray(model.H, dtype=float)
        v = np.asarray(model.v[:-1], dtype=float)
        # per-layer two-way time
        tau = 2.0 * H * np.sqrt(1.0 / (v**2) - bookkeeping.rayp**2)
        # cumulative arrival times at each discontinuity
        arr = np.cumsum(tau)

        # put scaled parent wavelet(s) at arrival time(s)
        for arr_k, amp_k, w_k in zip(arr, amp, model.w):
            shift = int(round(arr_k / prior.dt))
            # ignore out-of-range arrivals
            if shift <=0 or shift >= n/2:
                continue
            D += amp_k * np.roll(stretch_wavelet(P, w_k), -shift)
        
        # put one unmodified parent wavelet at the center
        D += P
    
    else: # bookkeeping.mode == 3
        # work on this later
        pass

    return D