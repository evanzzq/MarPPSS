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
    If bookkeeping.mode = 1/2 (PP/SS):
        P is one trace; return one trace of D;
    If bookkeeping.mode = 3 (joint):
        P is two traces (P[:,0] = P_PP; P[:,1] = P_SS); return two traces of D.
    """
    
    # check dt and no. of layers
    if prior.dt is None:
            raise ValueError("prior.dt must be set.")
    if model.Nlayer < 1:
        raise ValueError("Model must have at least 1 layer.")

    if bookkeeping.mode in (1, 2): # PP or SS mode

        # first create an empty D
        D = np.zeros_like(P)
        n = len(P)
        
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
        mf = [0.001, 0.001, 1.225]
        m1 = [vp1, vs1, rho1]
        m2 = [vp2, vs2, rho2]
        # R/T coef. and amplitude
        if bookkeeping.mode == 1:
            RppU_all = np.zeros(model.Nlayer)
            Rppf, _, _, _, _, _, _, _    = PSVRTmatrix(bookkeeping.rayp ,m1, mf)
            _, _, _, _, TppD, _, _, _    = PSVRTmatrix(bookkeeping.rayp ,m1, m2)
            RppU, _, _, _, TppU, _, _, _ = PSVRTmatrix(bookkeeping.rayp ,m2, m1)
            RppU_all[0] = RppU
            amp[0] = RppU / (TppU * TppD * Rppf)
        else: # bookkeeping.mode == 2
            RssU_all = np.zeros(model.Nlayer)
            Rssf, _    = SHmatrix(bookkeeping.rayp ,m1, mf)
            _, TssD    = SHmatrix(bookkeeping.rayp ,m1, m2)
            RssU, TssU = SHmatrix(bookkeeping.rayp ,m2, m1)
            RssU_all[0] = RssU
            amp[0] = RssU / (TssU * TssD * Rssf)

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
                RppU_all[k] = RppU
                amp[k] = amp[k-1] * (RppU / (TppU * TppD * RppU_all[k-1]))
            else: # bookkeeping.mode == 2
                _, TssD    = SHmatrix(bookkeeping.rayp ,m1, m2)
                RssU, TssU = SHmatrix(bookkeeping.rayp ,m2, m1)
                RssU_all[k] = RssU
                amp[k] = amp[k-1] * (RssU / (TssU * TssD * RssU_all[k-1]))
        
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
    
        return D

    else: # bookkeeping.mode == 3

        # initialize amplitude array
        amp_PP = np.zeros(model.Nlayer, dtype=float)
        amp_SS = np.zeros(model.Nlayer, dtype=float)

        # get P_PP and P_SS
        P_PP, P_SS = P[:,0], P[:,1]

        # create an empty D
        D_PP, D_SS = np.zeros_like(P_PP), np.zeros_like(P_SS)
        n = len(P_PP)

        # first disc.
        # define velocities (1 - top, 2 - bottom)
        vs1, vs2 = model.v[0], model.v[1]
        vp1, vp2 = vs1 * model.rho[0], vs2 * model.rho[1]
        # rho = vs * 0.8 from Li et al. (2022)
        # rho here is density (model.rho is vp/vs ratio)
        rho1, rho2 = vs1 * 800, vs2 * 800
        # define layer model for R/T function input
        mf = [0.001, 0.001, 1.225]
        m1 = [vp1, vs1, rho1]
        m2 = [vp2, vs2, rho2]
        # R/T coef. and amplitude
        # PP
        RppU_all = np.zeros(model.Nlayer)
        Rppf, _, _, _, _, _, _, _    = PSVRTmatrix(bookkeeping.rayp[0] ,m1, mf)
        _, _, _, _, TppD, _, _, _    = PSVRTmatrix(bookkeeping.rayp[0] ,m1, m2)
        RppU, _, _, _, TppU, _, _, _ = PSVRTmatrix(bookkeeping.rayp[0] ,m2, m1)
        RppU_all[0] = RppU
        amp_PP[0] = RppU / (TppU * TppD * Rppf)
        # SS
        RssU_all = np.zeros(model.Nlayer)
        Rssf, _    = SHmatrix(bookkeeping.rayp[1] ,m1, mf)
        _, TssD    = SHmatrix(bookkeeping.rayp[1] ,m1, m2)
        RssU, TssU = SHmatrix(bookkeeping.rayp[1] ,m2, m1)
        RssU_all[0] = RssU
        amp_SS[0] = RssU / (TssU * TssD * Rssf)

        # deeper disc. (if any)
        for k in range(1, model.Nlayer):

            # same as first disc.
            # define velocities (1 - top, 2 - bottom)
            vs1, vs2 = model.v[k], model.v[k+1]
            vp1, vp2 = vs1 * model.rho[k], vs2 * model.rho[k+1]
            # rho = vs * 0.8 from Li et al. (2022)
            rho1, rho2 = vs1 * 800, vs2 * 800
            # define layer model for R/T function input
            m1 = [vp1, vs1, rho1]
            m2 = [vp2, vs2, rho2]
            # R/T coef. and amplitude
            # PP
            _, _, _, _, TppD, _, _, _    = PSVRTmatrix(bookkeeping.rayp[0] ,m1, m2)
            RppU, _, _, _, TppU, _, _, _ = PSVRTmatrix(bookkeeping.rayp[0] ,m2, m1)
            RppU_all[k] = RppU
            amp_PP[k] = amp_PP[k-1] * (RppU / (TppU * TppD * RppU_all[k-1]))
            # SS
            _, TssD    = SHmatrix(bookkeeping.rayp[1] ,m1, m2)
            RssU, TssU = SHmatrix(bookkeeping.rayp[1] ,m2, m1)
            RssU_all[k] = RssU
            amp_SS[k] = amp_SS[k-1] * (RssU / (TssU * TssD * RssU_all[k-1]))

        # get arrival time(s)
        H = np.asarray(model.H, dtype=float)
        vs = np.asarray(model.v[:-1], dtype=float)
        vp = np.asarray(model.v[:-1] * model.rho[:-1], dtype=float)
        # per-layer two-way time
        tau_PP = 2.0 * H * np.sqrt(1.0 / (vp**2) - bookkeeping.rayp[0]**2)
        tau_SS = 2.0 * H * np.sqrt(1.0 / (vs**2) - bookkeeping.rayp[1]**2)
        # cumulative arrival times at each discontinuity
        arr_PP = np.cumsum(tau_PP)
        arr_SS = np.cumsum(tau_SS)

        # put scaled parent wavelet(s) at arrival time(s)
        # PP
        for arr_k, amp_k, w_k in zip(arr_PP, amp_PP, model.w):
            shift = int(round(arr_k / prior.dt))
            # ignore out-of-range arrivals
            if shift <=0 or shift >= n/2:
                continue
            D_PP += amp_k * np.roll(stretch_wavelet(P_PP, w_k), -shift)
        # SS
        for arr_k, amp_k, w_k in zip(arr_SS, amp_SS, model.w2):
            shift = int(round(arr_k / prior.dt))
            # ignore out-of-range arrivals
            if shift <=0 or shift >= n/2:
                continue
            D_SS += amp_k * np.roll(stretch_wavelet(P_SS, w_k), -shift)
        
        # put one unmodified parent wavelet at the center
        D_PP += P_PP
        D_SS += P_SS

        return D_PP, D_SS