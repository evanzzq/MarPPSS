from dataclasses import dataclass, field
import numpy as np

@dataclass
class Bookkeeping:
    totalSteps:     int = int(1e6)
    burnInSteps:    int = None
    nSaveModels:    int = 100
    actionsPerStep: int = 2
    mode:           int # 1 - PP, 2 - SS, 3 - joint
    rayp:           np.ndarray # mode 1/2: [rayp]; mode 3: [rayp_PP, rayp_SS]

    def __post_init__(self):
        if self.burnInSteps is None:
            self.burnInSteps = int(self.totalSteps // 2)
@dataclass
class Prior:
    stdP: float = 0.1
    maxN: int = 2
    tlen: int = None # G half length in sample points
    dt: float = None
    HRange: tuple = (1, 55)
    vRange: tuple = (1.0, 5.0) # for mode 1/2 v is vp or vs respectively; for mode 3 v is vs, and vp = v * rho
    rhoRange: tuple = (1.6, 2.0)

    HStd: float = 1.0
    vpStd: float = None
    vsStd: float = None

    negOnly: bool = False
    align: bool = False

    def __post_init__(self):
        if self.HStd is None:
            self.HStd = 0.05 * (self.HRange[1] - self.HRange[0])
        if self.widStd is None:
            self.vpStd = 0.05 * (self.vpRange[1] - self.vpRange[0])
        if self.vsStd is None:
            self.vsStd = 0.05 * (self.vsRange[1] - self.vsRange[0])
@dataclass
class Model:
    Nlayer: int
    H: np.ndarray
    vp: np.ndarray
    vs: np.ndarray

    @classmethod
    def create_initial(cls, prior: Prior):
        # for an initial model, it's one layer crust and half space with v and rho taken from prior
        return cls(
            Nlayer=1,
            H=np.random.uniform(prior.HRange[0], prior.HRange[1], 2),
            v=np.random.uniform(prior.vRange[0], prior.vRange[1], 2),
            rho=np.random.uniform(prior.rhoRange[0], prior.rhoRange[1], 2) # rho will be ignore in mode 1/2
        )
    
    # @classmethod
    # def create_random(cls, prior:Prior):
    #     pass