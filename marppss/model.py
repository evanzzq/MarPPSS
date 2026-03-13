from dataclasses import dataclass, field
import numpy as np

@dataclass
class Bookkeeping:
    mode:           int # 1 - PP, 2 - SS, 3 - joint
    rayp:           np.ndarray # mode 1/2: [rayp]; mode 3: [rayp_PP, rayp_SS]
    fitRange:       np.ndarray = None # mode 1/2: [tmin, tmax]; mode 3: [tmin_PP, tmax_PP, tmin_SS, tmax_SS]
    fitTT:          bool = False # fit travel time instead of full waveform
    fitLoge:        bool = True
    fitgv:          bool = False
    fitavgvs:       bool = False
    fitrho:         bool = False # only refers to fitgv scenario
    totalSteps:     int = int(1e6)
    burnInSteps:    int = None
    nSaveModels:    int = 100
    actionsPerStep: int = 2

    def __post_init__(self):
        if self.burnInSteps is None:
            self.burnInSteps = int(self.totalSteps // 2)
@dataclass
class Prior:
    stdPP: float = 0.01
    stdSS: float = 0.015
    maxN: int = 2
    dt: float = None
    tlen: float = None # half length in seconds
    HRange: tuple = (1, 60)
    wRange: tuple = (0.5, 1.5)
    logeRange: tuple = (0., 20.)
    vRange: tuple = (1.0, 5.0) # for mode 1/2 v is vp or vs respectively; for mode 3 v is vs, and vp = v * rho
    rhoRange: tuple = (1.6, 2.0)

    HStd: float = 1.0
    wStd: float = None
    logeStd: float = None
    vStd: float = None
    rhoStd: float = None

    def __post_init__(self):
        if self.HStd is None:
            self.HStd = 0.1 * (self.HRange[1] - self.HRange[0])
        if self.wStd is None:
            self.wStd = 0.1 * (self.wRange[1] - self.wRange[0])
        if self.logeStd is None:
            self.logeStd = 0.01 * (self.logeRange[1] - self.logeRange[0])
        if self.vStd is None:
            self.vStd = 0.2 * (self.vRange[1] - self.vRange[0])
        if self.rhoStd is None:
            self.rhoStd = 0.1 * (self.rhoRange[1] - self.rhoRange[0])
@dataclass
class Model:
    # in mode 1/2 (PP/SS), v refers to vp or vs respectively, and rho is not used.
    # in mode 3, v refers to vs, and rho is vp/vs; i.e., vp = vs * rho.
    Nlayer: int
    H: np.ndarray # depth of discontinuities
    w: np.ndarray # stretch factor, same length as H
    w2: np.ndarray # in mode 3 (joint), w is for PP and w2 is for SS
    loge: float # relative error for mode 1/2
    loge2: float # in mode 3 (joint), loge is for PP and loge2 is for SS
    loge_gv: float # for group velocity measurements
    loge_avg_vs: float # for average vs fit
    v: np.ndarray # len(v) = len(H) + 1
    rho: np.ndarray # len(rho) = len(v) = len(H) + 1

    @classmethod
    def create_initial(cls, prior: Prior, Nlayer: int = 1):

        # generate sorted interface depths
        H = np.sort(np.random.uniform(prior.HRange[0], prior.HRange[1], Nlayer))

        # velocities must increase with depth
        v = np.sort(np.random.uniform(prior.vRange[0], prior.vRange[1], Nlayer + 1))

        # vp/vs ratios
        rho = np.random.uniform(prior.rhoRange[0], prior.rhoRange[1], Nlayer + 1)

        return cls(
            Nlayer=Nlayer,
            H=H,
            w=np.ones(Nlayer),
            w2=np.ones(Nlayer),
            loge=0.,
            loge2=0.,
            loge_gv=0.,
            loge_avg_vs=0.,
            v=v,
            rho=rho
        )
    # @classmethod
    # def create_random(cls, prior:Prior):
    #     pass