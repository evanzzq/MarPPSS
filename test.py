import numpy as np
import matplotlib.pyplot as plt

from marppss.model import Model, Prior, Bookkeeping
from marppss.forward import create_D_from_model  # adjust import if needed


def main():
    # -----------------------------
    # 1) Build a parent wavelet P
    # -----------------------------
    dt = 0.05  # s
    tmax = 50.0
    t = np.arange(-tmax, tmax + dt, dt)
    P = np.exp(-(t / 0.5) ** 2)  # simple Gaussian centered at 0
    n = len(P)

    # -----------------------------
    # 2) Build a simple 2-layer model
    #    (1 discontinuity at depth H[0])
    # -----------------------------
    Nlayer = 4
    v = np.array([3,4,5,6,7])    # interpret as Vs for SS (mode=2) or Vp for PP (mode=1)
    rho = v * 0.8               # consistent with your Li+2022 scaling
    H = np.array([10, 20, 15, 25])        # km, thickness / depth to first discontinuity
    w = np.array([1,1,1,1])

    model = Model(
        Nlayer=Nlayer,
        H=H,
        w=w,
        v=v,
        rho=rho
    )

    # -----------------------------
    # 3) Prior & Bookkeeping
    # -----------------------------
    prior = Prior(
        dt=dt
    )

    # rayp should be a numpy array per your Bookkeeping definition
    # use 0-d array so it behaves like a scalar in math
    rayp_value = 0.04  # s/km (just something < 1/vmax)
    rayp = np.array(rayp_value)

    bookkeeping = Bookkeeping(
        totalSteps=1000,
        nSaveModels=10,
        actionsPerStep=2,
        mode=1,       # 1 = PP, 2 = SS
        rayp=rayp
    )

    # -----------------------------
    # 4) Call create_D_from_model
    # -----------------------------
    D = create_D_from_model(P, model, prior, bookkeeping)

    # -----------------------------
    # 5) Plot P and D for inspection
    # -----------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, P, lw=1.5)
    axes[0].set_title("Parent wavelet P(t)")
    axes[0].set_ylabel("Amp")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, D, lw=1.5)
    axes[1].set_title("Predicted D(t) from create_D_from_model")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amp")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
