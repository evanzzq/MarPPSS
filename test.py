import numpy as np
import matplotlib.pyplot as plt

from marppss.model import Model, Prior, Bookkeeping
from marppss.forward import create_D_from_model  # adjust import if needed


def main():
    # -----------------------------
    # 1) Build a parent wavelet P (time axis shared)
    # -----------------------------
    dt = 0.05  # s
    tmax = 50.0
    t = np.arange(-tmax, tmax + dt, dt)
    P = np.exp(-(t / 0.5) ** 2)  # simple Gaussian centered at 0
    n = len(P)

    # -----------------------------
    # 2) Build a simple 4-layer model
    # -----------------------------
    Nlayer = 4
    v = np.array([3, 4, 5, 6, 7])   # for mode 1/2: interpret as Vs (mode=2) or Vp (mode=1)
    rho = v * 0.8                   # density-like scaling for mode 1/2 test
    H = np.array([10, 20, 15, 25])  # km
    w = np.array([1, 1, 1, 1])

    # -----------------------------
    # 3) Prior
    # -----------------------------
    prior = Prior(dt=dt)

    # -----------------------------
    # 4) Test: mode = 2 (SS only) – original behavior
    # -----------------------------
    rayp_value = 0.04  # s/km (just something < 1/vmax)
    rayp = np.array(rayp_value)

    bookkeeping_ss = Bookkeeping(
        totalSteps=1000,
        nSaveModels=10,
        actionsPerStep=2,
        mode=2,       # 1 = PP, 2 = SS
        rayp=rayp
    )

    model_ss = Model(
        Nlayer=Nlayer,
        H=H,
        w=w,
        w2=w,
        v=v,
        rho=rho
    )

    D_ss = create_D_from_model(P, model_ss, prior, bookkeeping_ss)

    fig1, axes1 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes1[0].plot(t, P, lw=1.5)
    axes1[0].set_title("Mode 2 test: Parent wavelet P(t)")
    axes1[0].set_ylabel("Amp")
    axes1[0].grid(alpha=0.3)

    axes1[1].plot(t, D_ss, lw=1.5)
    axes1[1].set_title("Mode 2 test: Predicted D(t) from create_D_from_model (SS)")
    axes1[1].set_xlabel("Time (s)")
    axes1[1].set_ylabel("Amp")
    axes1[1].grid(alpha=0.3)

    plt.tight_layout()

    # -----------------------------
    # 5) Test: mode = 3 (joint PP + SS)
    # -----------------------------
    # Build PP and SS parent wavelets
    P_PP = P.copy()
    # Make SS wavelet a bit different so plots are visually distinguishable
    P_SS = np.exp(-(t / 0.7) ** 2)

    # Stack into (n, 2): [:,0] = PP, [:,1] = SS
    P_joint = np.column_stack((P_PP, P_SS))

    # For mode 3:
    #   v   = Vs
    #   rho = Vp/Vs ratio
    vs = np.array([3, 4, 5, 6, 7])              # Vs
    vp_vs_ratio = 1.8 * np.ones_like(vs)        # Vp/Vs ratio
    w_PP = np.ones(Nlayer)
    w_SS = np.ones(Nlayer)

    # If your Model class does not accept w2 as a keyword, you can:
    #   model_joint = Model(...); model_joint.w2 = w_SS
    model_joint = Model(
        Nlayer=Nlayer,
        H=H,
        w=w_PP,
        v=vs,
        rho=vp_vs_ratio,
        w2=w_SS           # adjust if constructor differs
    )

    prior_joint = Prior(dt=dt)

    rayp_PP = 0.04  # s/km
    rayp_SS = 0.06  # s/km, just to be different but still < 1/vmax
    rayp_joint = np.array([rayp_PP, rayp_SS])

    bookkeeping_joint = Bookkeeping(
        totalSteps=1000,
        nSaveModels=10,
        actionsPerStep=2,
        mode=3,        # joint PP + SS
        rayp=rayp_joint
    )

    D_PP, D_SS = create_D_from_model(P_joint, model_joint, prior_joint, bookkeeping_joint)

    # -----------------------------
    # 6) Plot mode 3 results: D_PP and D_SS
    # -----------------------------
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Parent wavelets
    axes2[0].plot(t, P_PP, lw=1.5, label="P_PP")
    axes2[0].plot(t, P_SS, lw=1.0, linestyle="--", label="P_SS")
    axes2[0].set_title("Mode 3 test: Parent wavelets (PP & SS)")
    axes2[0].set_ylabel("Amp")
    axes2[0].legend()
    axes2[0].grid(alpha=0.3)

    # D_PP
    axes2[1].plot(t, D_PP, lw=1.5)
    axes2[1].set_title("Mode 3 test: Predicted D_PP(t)")
    axes2[1].set_ylabel("Amp")
    axes2[1].grid(alpha=0.3)

    # D_SS
    axes2[2].plot(t, D_SS, lw=1.5)
    axes2[2].set_title("Mode 3 test: Predicted D_SS(t)")
    axes2[2].set_xlabel("Time (s)")
    axes2[2].set_ylabel("Amp")
    axes2[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
