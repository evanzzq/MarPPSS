import numpy as np


def top_layer_sqrt_gradient_enabled(assumptions=None):
    assumptions = assumptions or {}
    return top_layer_gradient_kind(assumptions) == "sqrt"


def top_layer_gradient_kind(assumptions=None):
    assumptions = assumptions or {}
    kind = assumptions.get("top_layer_gradient")
    aliases = {
        "none": "step",
        "false": "step",
        "off": "step",
        "step": "step",
        "constant": "step",
        "sqrt": "sqrt",
        "square_root": "sqrt",
        "square-root": "sqrt",
        "linear": "linear",
        "line": "linear",
        "all_linear": "all_linear",
        "all-layer-linear": "all_linear",
        "all_layer_linear": "all_linear",
        "layer_linear": "all_linear",
        "linear_all": "all_linear",
        "linear-all": "all_linear",
        "full_linear": "all_linear",
        "every_layer_linear": "all_linear",
    }
    if kind is not None:
        key = str(kind).strip().lower()
        if key not in aliases:
            raise ValueError("top_layer_gradient must be step, sqrt, linear, or all_linear.")
        normalized = aliases[key]
        if normalized != "step":
            return normalized

    if bool(assumptions.get("top_layer_sqrt_gradient", False)):
        return "sqrt"
    if bool(assumptions.get("top_layer_linear_gradient", False)):
        return "linear"
    return "step"


def top_layer_gradient_enabled(assumptions=None):
    return top_layer_gradient_kind(assumptions) != "step"


def all_layer_gradient_enabled(assumptions=None):
    return top_layer_gradient_kind(assumptions) == "all_linear"


def minimum_velocity_jump_fraction(assumptions=None):
    assumptions = assumptions or {}
    for key in (
        "minimum_velocity_jump_percent",
        "min_velocity_jump_percent",
        "velocity_jump_percent",
        "min_jump_percent",
    ):
        if key in assumptions and assumptions[key] is not None:
            return float(assumptions[key]) / 100.0
    return 0.0


def velocity_transition_directions(assumptions=None):
    assumptions = assumptions or {}
    directions = assumptions.get("velocity_transition_directions")
    if directions is None:
        directions = assumptions.get("velocity_transitions")
    if directions is None:
        return None

    aliases = {
        "inc": "inc",
        "increase": "inc",
        "increasing": "inc",
        "up": "inc",
        "+": "inc",
        "dec": "dec",
        "decrease": "dec",
        "decreasing": "dec",
        "down": "dec",
        "-": "dec",
        "free": "free",
        "any": "free",
        "none": "free",
        "0": "free",
    }
    normalized = []
    for direction in directions:
        key = str(direction).strip().lower()
        if key not in aliases:
            raise ValueError(
                "velocity_transition_directions entries must be inc, dec, or free; "
                f"got {direction!r}."
            )
        normalized.append(aliases[key])
    return normalized


def top_layer_velocity(v0_km_s, a, depth_km, assumptions=None):
    depth_km = np.asarray(depth_km, dtype=float)
    kind = top_layer_gradient_kind(assumptions)
    if kind == "sqrt":
        return float(v0_km_s) + (float(a) / 1000.0) * np.sqrt(1000.0 * depth_km)
    if kind == "linear":
        return float(v0_km_s) + float(a) * depth_km
    return np.full_like(depth_km, float(v0_km_s), dtype=float)


def _as_slope_array(a, n, default=0.0):
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 0:
        return np.full(int(n), float(arr), dtype=float)
    if arr.size < int(n):
        out = np.full(int(n), float(default), dtype=float)
        out[:arr.size] = arr
        return out
    return arr[:int(n)].astype(float, copy=False)


def layer_top_depths(H):
    return np.r_[0.0, np.asarray(H, dtype=float)]


def layer_thicknesses(H):
    return np.diff(layer_top_depths(H))


def layer_velocity(v0_km_s, a, depth_from_top_km, assumptions=None):
    depth_from_top_km = np.asarray(depth_from_top_km, dtype=float)
    if all_layer_gradient_enabled(assumptions):
        return float(v0_km_s) + float(a) * depth_from_top_km
    return np.full_like(depth_from_top_km, float(v0_km_s), dtype=float)


def layer_bottom_velocity(model, layer_idx, assumptions=None, velocity_kind="v"):
    if layer_idx < 0 or layer_idx >= int(model.Nlayer):
        return float(model.v[layer_idx])

    if velocity_kind == "vp":
        v_top = np.asarray(model.v, dtype=float) * np.asarray(model.rho, dtype=float)
        slopes = _as_slope_array(getattr(model, "a", 0.0), int(model.Nlayer))
        slopes = slopes * np.asarray(model.rho[:-1], dtype=float)
    else:
        v_top = np.asarray(model.v, dtype=float)
        slopes = _as_slope_array(getattr(model, "a", 0.0), int(model.Nlayer))

    if all_layer_gradient_enabled(assumptions):
        thickness = layer_thicknesses(model.H)[layer_idx]
        return float(layer_velocity(v_top[layer_idx], slopes[layer_idx], thickness, assumptions=assumptions))
    if layer_idx == 0 and top_layer_gradient_enabled(assumptions):
        return float(top_layer_velocity(v_top[0], slopes[0], model.H[0], assumptions=assumptions))
    return float(v_top[layer_idx])


def layer_two_way_time(thickness_km, v0_km_s, a, rayp_s_km, assumptions=None, n=64):
    if thickness_km <= 0:
        return 0.0

    z = np.linspace(0.0, float(thickness_km), int(n))
    if all_layer_gradient_enabled(assumptions):
        v = layer_velocity(v0_km_s, a, z, assumptions=assumptions)
    else:
        v = top_layer_velocity(v0_km_s, a, z, assumptions=assumptions)
    slowness_sq = 1.0 / (v * v) - float(rayp_s_km) ** 2
    if np.any(slowness_sq <= 0.0):
        return np.nan
    return 2.0 * np.trapz(np.sqrt(slowness_sq), z)


def top_layer_two_way_time(thickness_km, v0_km_s, a, rayp_s_km, assumptions=None, n=64):
    return layer_two_way_time(thickness_km, v0_km_s, a, rayp_s_km, assumptions=assumptions, n=n)


def interface_velocity(model, layer_idx, assumptions=None):
    if all_layer_gradient_enabled(assumptions):
        return layer_bottom_velocity(model, layer_idx, assumptions=assumptions)
    if layer_idx == 0 and top_layer_gradient_enabled(assumptions):
        return float(top_layer_velocity(model.v[0], getattr(model, "a", 0.0), model.H[0], assumptions=assumptions))
    return float(model.v[layer_idx])


def velocity_at_transition_top(model, transition_idx, assumptions=None):
    if transition_idx == 0:
        return interface_velocity(model, 0, assumptions)
    return float(model.v[transition_idx])


def check_velocity_transitions(model, assumptions=None):
    directions = velocity_transition_directions(assumptions)
    min_jump = minimum_velocity_jump_fraction(assumptions)
    check_min_jump = all_layer_gradient_enabled(assumptions) and min_jump > 0.0
    if directions is None and not check_min_jump:
        return None

    if directions is not None and len(directions) != int(model.Nlayer):
        raise ValueError(
            "velocity_transition_directions must have one entry per discontinuity; "
            f"got {len(directions)} for Nlayer={model.Nlayer}."
        )

    for i in range(int(model.Nlayer)):
        direction = directions[i] if directions is not None else "inc"
        top_v = velocity_at_transition_top(model, i, assumptions)
        bottom_v = float(model.v[i + 1])
        if direction == "free":
            if check_min_jump and abs(bottom_v - top_v) < min_jump * top_v:
                return False
            continue
        if direction == "inc" and not (bottom_v > top_v):
            return False
        if direction == "inc" and check_min_jump and bottom_v < top_v * (1.0 + min_jump):
            return False
        if direction == "dec" and not (bottom_v < top_v):
            return False
        if direction == "dec" and check_min_jump and bottom_v > top_v * (1.0 - min_jump):
            return False
    return True


def layer_travel_times(model, rayp, assumptions=None, velocity_kind="v"):
    thickness = layer_thicknesses(model.H)

    if velocity_kind == "vp":
        base_v = np.asarray(model.v[:-1] * model.rho[:-1], dtype=float)
        a = _as_slope_array(getattr(model, "a", 0.0), int(model.Nlayer)) * np.asarray(model.rho[:-1], dtype=float)
    else:
        base_v = np.asarray(model.v[:-1], dtype=float)
        a = _as_slope_array(getattr(model, "a", 0.0), int(model.Nlayer))

    tau = np.empty_like(thickness, dtype=float)
    for i, h_i in enumerate(thickness):
        if all_layer_gradient_enabled(assumptions) or (i == 0 and top_layer_gradient_enabled(assumptions)):
            tau[i] = layer_two_way_time(h_i, base_v[i], a[i], rayp, assumptions=assumptions)
        else:
            radicand = 1.0 / (base_v[i] ** 2) - float(rayp) ** 2
            tau[i] = np.nan if radicand <= 0.0 else 2.0 * h_i * np.sqrt(radicand)
    return tau


def disba_layers_from_model(model, bookkeeping, vpvsr, subdivisions=12, min_thickness_km=0.05):
    assumptions = getattr(bookkeeping, "assumptions", None)
    interfaces = np.asarray(model.H, dtype=float)
    if interfaces.size == 0:
        return np.array([0.0]), np.asarray(model.v), np.asarray(model.v) * vpvsr, None

    thickness = layer_thicknesses(interfaces)
    split_gradients = top_layer_gradient_enabled(assumptions) and np.any(thickness > 0.0)

    if bookkeeping.mode == 1:
        vp_base = np.asarray(model.v, dtype=float)
        vs_base = vp_base / vpvsr
        a_vp = _as_slope_array(getattr(model, "a", 0.0), int(model.Nlayer))
        vpvsr_arr = np.asarray(vpvsr, dtype=float)
        vpvsr_finite = vpvsr_arr if vpvsr_arr.ndim == 0 else vpvsr_arr[:-1]
        a_vs = a_vp / vpvsr_finite
    else:
        vs_base = np.asarray(model.v, dtype=float)
        vp_base = vs_base * vpvsr
        a_vs = _as_slope_array(getattr(model, "a", 0.0), int(model.Nlayer))
        vpvsr_arr = np.asarray(vpvsr, dtype=float)
        vpvsr_finite = vpvsr_arr if vpvsr_arr.ndim == 0 else vpvsr_arr[:-1]
        a_vp = a_vs * vpvsr_finite

    if not split_gradients:
        return np.r_[thickness, 0.0], vp_base, vs_base, 0.8 * vs_base

    H_parts = []
    vp_parts = []
    vs_parts = []
    for i, h_i in enumerate(thickness):
        should_split = all_layer_gradient_enabled(assumptions) or i == 0
        if not should_split or h_i <= 0.0:
            H_parts.append(h_i)
            vp_parts.append(vp_base[i])
            vs_parts.append(vs_base[i])
            continue
        max_allowed = max(1, int(np.floor(h_i / float(min_thickness_km))))
        n = max(1, min(int(subdivisions), max_allowed))
        edges = np.linspace(0.0, h_i, n + 1)
        mid = 0.5 * (edges[:-1] + edges[1:])
        H_parts.extend(np.diff(edges))
        if all_layer_gradient_enabled(assumptions):
            vp_parts.extend(layer_velocity(vp_base[i], a_vp[i], mid, assumptions=assumptions))
            vs_parts.extend(layer_velocity(vs_base[i], a_vs[i], mid, assumptions=assumptions))
        else:
            vp_parts.extend(top_layer_velocity(vp_base[i], a_vp[i], mid, assumptions=assumptions))
            vs_parts.extend(top_layer_velocity(vs_base[i], a_vs[i], mid, assumptions=assumptions))

    H = np.r_[np.asarray(H_parts, dtype=float), 0.0]
    vp = np.r_[np.asarray(vp_parts, dtype=float), vp_base[-1]]
    vs = np.r_[np.asarray(vs_parts, dtype=float), vs_base[-1]]
    rho = 0.8 * vs
    return H, vp, vs, rho
