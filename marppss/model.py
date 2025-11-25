from dataclasses import dataclass, field
import numpy as np

@dataclass
class Bookkeeping:
    mode:           int # 1 - PP, 2 - SS, 3 - joint
    rayp:           np.ndarray # mode 1/2: [rayp]; mode 3: [rayp_PP, rayp_SS]
    totalSteps:     int = int(1e6)
    burnInSteps:    int = None
    nSaveModels:    int = 100
    actionsPerStep: int = 2

    def __post_init__(self):
        if self.burnInSteps is None:
            self.burnInSteps = int(self.totalSteps // 2)
@dataclass
class Prior:
    stdP: float = 0.1
    maxN: int = 2
    dt: float = None
    tlen: float = None # half length in seconds
    HRange: tuple = (1, 55)
    wRange: tuple = (0.5, 1.5)
    vRange: tuple = (1.0, 5.0) # for mode 1/2 v is vp or vs respectively; for mode 3 v is vs, and vp = v * rho
    rhoRange: tuple = (1.6, 2.0)

    HStd: float = 1.0
    wStd: float = None
    vStd: float = None
    rhoStd: float = None

    def __post_init__(self):
        if self.HStd is None:
            self.HStd = 0.05 * (self.HRange[1] - self.HRange[0])
        if self.wStd is None:
            self.wStd = 0.05 * (self.wRange[1] - self.wRange[0])
        if self.vStd is None:
            self.vStd = 0.05 * (self.vRange[1] - self.vRange[0])
        if self.rhoStd is None:
            self.rhoStd = 0.05 * (self.rhoRange[1] - self.rhoRange[0])
@dataclass
class Model:
    Nlayer: int
    H: np.ndarray
    w: np.ndarray # stretch factor, same length as H
    v: np.ndarray # len(v) = len(H) + 1
    rho: np.ndarray # len(rho) = len(v) = len(H) + 1

    @classmethod
    def create_initial(cls, prior: Prior):
        # for an initial model, it's one layer crust and half space with v and rho taken from prior
        return cls(
            Nlayer=1,
            H=np.random.uniform(prior.HRange[0], prior.HRange[1], 1),
            w=np.ones(1),
            v=np.random.uniform(prior.vRange[0], prior.vRange[1], 2),
            rho=np.random.uniform(prior.rhoRange[0], prior.rhoRange[1], 2) # rho will be ignore in mode 1/2
        )
    
    # @classmethod
    # def create_random(cls, prior:Prior):
    #     pass