from dataclasses import dataclass, field
import numpy as np
from marppss.velocity import (
    all_layer_gradient_enabled,
    check_velocity_transitions,
    layer_bottom_velocity,
    layer_thicknesses,
    minimum_velocity_jump_fraction,
    top_layer_gradient_enabled,
    top_layer_velocity,
    velocity_transition_directions,
)

@dataclass
class Bookkeeping:
    mode:           int # 1 - PP, 2 - SS, 3 - joint
    rayp:           np.ndarray # mode 1/2: [rayp]; mode 3: [rayp_PP, rayp_SS]
    fitRange:       np.ndarray = None # mode 1/2: [tmin, tmax]; mode 3: [tmin_PP, tmax_PP, tmin_SS, tmax_SS]
    fitWaveform:    bool = True
    fitTT:          bool = False # fit travel time instead of full waveform
    fitLoge:        bool = True
    fitgv:          bool = False
    fitavgvs:       bool = False
    fitrho:         bool = False # only refers to fitgv scenario
    totalSteps:     int = int(1e6)
    burnInSteps:    int = None
    nSaveModels:    int = 100
    actionsPerStep: int = 2
    fixedNlayer:    int = None
    travel_times:   dict = None
    group_velocity: dict = None
    avg_vs:         dict = None
    assumptions:    dict = None
    reference_model: dict = None
    metadata:       dict = None

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
    logeRange: tuple = (0., 10.)
    vRange: tuple = (1.0, 5.0) # for mode 1/2 v is vp or vs respectively; for mode 3 v is vs, and vp = v * rho
    aRange: tuple = (0.0, 0.0) # gradient parameter: sqrt m/s/sqrt(m), linear m/s/m
    rhoRange: tuple = (1.6, 2.0)

    HStd: float = 1.0
    wStd: float = None
    logeStd: float = None
    vStd: float = None
    aStd: float = None
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
        if self.aStd is None:
            self.aStd = 0.2 * (self.aRange[1] - self.aRange[0])
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
    loge_TT: float # for travel times (mode 1/2: for PP or SS)
    loge_TT2: float # for travel times (mode 3: loge_TT is for PP and loge_TT2 is for SS)
    v: np.ndarray # len(v) = len(H) + 1
    a: float # gradient parameter; scalar for top-layer modes, array for all_linear
    rho: np.ndarray # len(rho) = len(v) = len(H) + 1

    @classmethod
    def create_initial(cls, prior: Prior, Nlayer: int = 1, assumptions=None):

        directions = velocity_transition_directions(assumptions)
        if directions is None and not all_layer_gradient_enabled(assumptions):
            v = None
        elif directions is not None:
            if len(directions) != Nlayer:
                raise ValueError(
                    "velocity_transition_directions must have one entry per discontinuity; "
                    f"got {len(directions)} for Nlayer={Nlayer}."
                )
            v = None
        else:
            v = None

        for _ in range(10000):
            # generate sorted interface depths
            H = np.sort(np.random.uniform(prior.HRange[0], prior.HRange[1], Nlayer))
            rho = np.random.uniform(prior.rhoRange[0], prior.rhoRange[1], Nlayer + 1)

            if all_layer_gradient_enabled(assumptions):
                a = np.empty(Nlayer, dtype=float)
                candidate_v = np.empty(Nlayer + 1, dtype=float)
                candidate_v[0] = np.random.uniform(prior.vRange[0], prior.vRange[1])
                jump = minimum_velocity_jump_fraction(assumptions)
                thickness = layer_thicknesses(H)
                layer_directions = directions or ["inc"] * Nlayer
                valid_candidate = True
                for i in range(Nlayer):
                    h_i = float(thickness[i])
                    direction = layer_directions[i]
                    a_low = float(prior.aRange[0])
                    a_high = float(prior.aRange[1])
                    if h_i > 0.0:
                        a_low = max(a_low, (prior.vRange[0] - candidate_v[i]) / h_i)
                        a_high = min(a_high, (prior.vRange[1] - candidate_v[i]) / h_i)
                    elif not (prior.vRange[0] <= candidate_v[i] <= prior.vRange[1]):
                        valid_candidate = False
                        break

                    if direction == "inc":
                        if h_i > 0.0:
                            a_high = min(a_high, (prior.vRange[1] / (1.0 + jump) - candidate_v[i]) / h_i)
                        elif candidate_v[i] * (1.0 + jump) >= prior.vRange[1]:
                            valid_candidate = False
                            break
                    elif direction == "dec":
                        if h_i > 0.0:
                            a_low = max(a_low, (prior.vRange[0] / max(1.0 - jump, 1e-12) - candidate_v[i]) / h_i)
                        elif candidate_v[i] * (1.0 - jump) <= prior.vRange[0]:
                            valid_candidate = False
                            break

                    if not (a_low <= a_high):
                        valid_candidate = False
                        break

                    a[i] = np.random.uniform(a_low, a_high)
                    top_v = candidate_v[i] + a[i] * h_i
                    if direction == "inc":
                        next_low = max(prior.vRange[0], top_v * (1.0 + jump))
                        next_high = prior.vRange[1]
                    elif direction == "dec":
                        next_low = prior.vRange[0]
                        next_high = min(prior.vRange[1], top_v * (1.0 - jump))
                    else:
                        next_low = prior.vRange[0]
                        next_high = prior.vRange[1]

                    if not (next_low < next_high):
                        valid_candidate = False
                        break
                    candidate_v[i + 1] = np.random.uniform(next_low, next_high)
                if not valid_candidate:
                    continue
            else:
                a = np.random.uniform(prior.aRange[0], prior.aRange[1])
                candidate_v = v
                if candidate_v is None:
                    if directions is None:
                        candidate_v = np.sort(np.random.uniform(prior.vRange[0], prior.vRange[1], Nlayer + 1))
                    else:
                        candidate_v = np.random.uniform(prior.vRange[0], prior.vRange[1], Nlayer + 1)

            model = cls(
                Nlayer=Nlayer,
                H=H,
                w=np.ones(Nlayer),
                w2=np.ones(Nlayer),
                loge=0.,
                loge2=0.,
                loge_gv=0.,
                loge_avg_vs=0.,
                loge_TT=0.,
                loge_TT2=0.,
                v=candidate_v,
                a=a,
                rho=rho
            )
            if top_layer_gradient_enabled(assumptions):
                if all_layer_gradient_enabled(assumptions):
                    profile_endpoints = [
                        layer_bottom_velocity(model, i, assumptions=assumptions)
                        for i in range(Nlayer)
                    ]
                elif model.H.size > 0:
                    profile_endpoints = [
                        top_layer_velocity(model.v[0], model.a, model.H[0], assumptions=assumptions)
                    ]
                else:
                    profile_endpoints = []
                profile_endpoints = np.asarray(profile_endpoints, dtype=float)
                if (
                    np.any(~np.isfinite(profile_endpoints))
                    or np.any(profile_endpoints < prior.vRange[0])
                    or np.any(profile_endpoints > prior.vRange[1])
                ):
                    continue
            transition_ok = check_velocity_transitions(model, assumptions)
            if transition_ok is not False:
                return model

        raise RuntimeError("Could not create an initial model satisfying velocity_transition_directions.")
    # @classmethod
    # def create_random(cls, prior:Prior):
    #     pass
