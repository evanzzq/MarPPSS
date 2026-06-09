import copy
import os
import re
from pathlib import Path

MODE_ALIASES = {
    1: 1,
    2: 2,
    3: 3,
    "1": 1,
    "2": 2,
    "3": 3,
    "pp": 1,
    "ss": 2,
    "joint": 3,
}


def load_workflow_config(path):
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to load workflow configs. Install it with `pip install pyyaml`."
        ) from exc
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return _expand_config_paths(data)


def _expand_path_string(value):
    expanded = os.path.expanduser(value)
    expanded = os.path.expandvars(expanded)
    expanded = re.sub(r"\$\{([^}]+)\}", lambda match: os.environ.get(match.group(1), match.group(0)), expanded)
    expanded = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", lambda match: os.environ.get(match.group(1), match.group(0)), expanded)
    return expanded


def _expand_config_paths(value):
    if isinstance(value, dict):
        return {k: _expand_config_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_config_paths(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_expand_config_paths(v) for v in value)
    if isinstance(value, str):
        return _expand_path_string(value)
    return value


def list_experiments(config):
    experiments = config.get("experiments", {})
    if isinstance(experiments, dict):
        return list(experiments.keys())
    if isinstance(experiments, list):
        names = []
        for idx, exp in enumerate(experiments, start=1):
            names.append(exp.get("name") or exp.get("runname") or f"experiment_{idx}")
        return names
    return []


def list_events(config):
    events = config.get("events", {})
    if isinstance(events, dict):
        return list(events.keys())
    return []


def _canonical_mode(mode):
    try:
        return MODE_ALIASES[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported mode value: {mode!r}") from exc


def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _as_plain_dict(value):
    if isinstance(value, dict):
        return {k: _as_plain_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain_dict(v) for v in value]
    return value


def _normalize_group_velocity(cfg):
    if not cfg:
        return None

    values = cfg.get("values")
    if values is None:
        values = cfg.get("velocities")
    if values is None:
        values = cfg.get("gv_obs")

    if values is None or cfg.get("periods") is None:
        return None

    uncertainties = cfg.get("uncertainties", cfg.get("gv_unc", 0.2))

    return {
        "periods": list(cfg["periods"]),
        "values": list(values),
        "uncertainties": uncertainties,
        "wave": cfg.get("wave", "rayleigh"),
        "mode": int(cfg.get("mode", cfg.get("mode_idx", 0))),
        "vpvsr": cfg.get("vpvsr", 1.8),
    }


def _normalize_travel_times(cfg):
    if not cfg:
        return None

    normalized = {}
    for phase in ("PP", "SS"):
        phase_cfg = cfg.get(phase) or cfg.get(phase.lower())
        if not phase_cfg:
            continue
        times = phase_cfg.get("times", phase_cfg.get("values"))
        if times is None:
            continue
        normalized[phase] = {
            "times": list(times),
            "uncertainties": phase_cfg.get("uncertainties", phase_cfg.get("sigma", 0.1)),
        }
    return normalized or None


def _normalize_avg_vs(cfg):
    if not cfg:
        return None
    if "value" not in cfg and "avg_vs_ref" not in cfg:
        return None
    return {
        "value": cfg.get("value", cfg.get("avg_vs_ref")),
        "uncertainty": cfg.get("uncertainty", cfg.get("avg_vs_unc", 0.1)),
    }


def _flatten_event(path_cfg, defaults, name, event_cfg):
    preprocessing = _deep_merge(defaults.get("preprocessing", {}), event_cfg.get("preprocessing", {}))
    constraints = _deep_merge(defaults.get("constraints", {}), event_cfg.get("constraints", {}))
    assumptions = _deep_merge(defaults.get("assumptions", {}), event_cfg.get("assumptions", {}))

    arrivals = event_cfg.get("arrivals", {})
    rayp = event_cfg.get("rayp")

    return {
        "event_name": name,
        "basedir": path_cfg["basedir"],
        "datadir_subpaths": list(path_cfg.get("datadir_subpaths", ["SharpSSPy", "misc"])),
        "outdir_subpaths": list(path_cfg.get("outdir_subpaths", ["MarPPSS"])),
        "evname": event_cfg.get("evname", name),
        "dtype": event_cfg.get("dtype", "DISP"),
        "PPfreq": preprocessing.get("PPfreq"),
        "SSfreq": preprocessing.get("SSfreq"),
        "cutwin": preprocessing.get("cutwin"),
        "src_sigma": preprocessing.get("src_sigma"),
        "rotated": preprocessing.get("rotated", True),
        "baz": preprocessing.get("baz"),
        "PParr": arrivals.get("PP", event_cfg.get("PParr")),
        "SSarr": arrivals.get("SS", event_cfg.get("SSarr")),
        "rayp": rayp,
        "travel_times": _normalize_travel_times(constraints.get("travel_times")),
        "group_velocity": _normalize_group_velocity(constraints.get("group_velocity")),
        "avg_vs": _normalize_avg_vs(constraints.get("average_vs")),
        "assumptions": _as_plain_dict(assumptions),
        "reference_model": _as_plain_dict(event_cfg.get("reference_model")),
    }


def _resolve_path_cfg(paths):
    path_cfg = _deep_merge(
        {
            "basedir": None,
            "research_root": None,
            "data_root": None,
            "datadir_subpaths": ["SharpSSPy", "misc"],
            "outdir_subpaths": ["MarPPSS"],
        },
        paths,
    )

    data_root = path_cfg.get("data_root")
    if data_root:
        data_root_path = Path(data_root).resolve()
        path_cfg["basedir"] = str(data_root_path.parent)
        if "outdir_subpaths" not in (paths or {}):
            path_cfg["outdir_subpaths"] = [data_root_path.name]
    else:
        path_cfg["basedir"] = path_cfg.get("research_root") or path_cfg.get("basedir")

    return path_cfg


def _resolve_new_experiment(config, experiment_name):
    experiments = config.get("experiments", {})
    if not isinstance(experiments, dict):
        raise ValueError("New-style config expects 'experiments' to be a mapping.")
    if experiment_name not in experiments:
        available = ", ".join(sorted(experiments))
        raise KeyError(f"Unknown experiment '{experiment_name}'. Available: {available}")

    defaults = config.get("defaults", {})
    path_cfg = _resolve_path_cfg(config.get("paths", {}))
    event_name = experiments[experiment_name].get("event")
    if not event_name:
        raise ValueError(f"Experiment '{experiment_name}' is missing an 'event' reference.")
    event_cfg = config.get("events", {}).get(event_name)
    if not event_cfg:
        raise KeyError(f"Experiment '{experiment_name}' references missing event '{event_name}'.")

    resolved = _flatten_event(path_cfg, defaults, event_name, event_cfg)
    exp_cfg = experiments[experiment_name]
    runname = exp_cfg.get("runname", experiment_name)
    if runname != experiment_name:
        raise ValueError(
            f"Experiment '{experiment_name}' has runname '{runname}'. "
            "For new-style configs, omit runname or set it equal to the experiment name."
        )

    prior_defaults = defaults.get("prior", {})
    inversion_defaults = defaults.get("inversion", {})
    runtime_defaults = defaults.get("runtime", {})
    exp_constraints = _deep_merge(defaults.get("constraints", {}), exp_cfg.get("constraints", {}))
    exp_assumptions = _deep_merge(defaults.get("assumptions", {}), exp_cfg.get("assumptions", {}))

    resolved.update(
        {
            "experiment_name": experiment_name,
            "mode": _canonical_mode(exp_cfg["mode"]),
            "runname": runname,
            "useCD": exp_cfg.get("useCD", inversion_defaults.get("useCD", False)),
            "fitWaveform": exp_cfg.get("fitWaveform", inversion_defaults.get("fitWaveform", True)),
            "fitLoge": exp_cfg.get("fitLoge", inversion_defaults.get("fitLoge", True)),
            "fitTT": exp_cfg.get("fitTT", inversion_defaults.get("fitTT", False)),
            "fitgv": exp_cfg.get("fitgv", inversion_defaults.get("fitgv", False)),
            "fitavgvs": exp_cfg.get("fitavgvs", inversion_defaults.get("fitavgvs", False)),
            "fitrho": exp_cfg.get("fitrho", inversion_defaults.get("fitrho", False)),
            "fitRange": exp_cfg.get("fitRange", inversion_defaults.get("fitRange")),
            "fixedNlayer": exp_cfg.get("fixedNlayer", inversion_defaults.get("fixedNlayer")),
            "maxN": exp_cfg.get("maxN", prior_defaults.get("maxN", 1)),
            "stdPP": exp_cfg.get("stdPP", prior_defaults.get("stdPP", 0.04)),
            "stdSS": exp_cfg.get("stdSS", prior_defaults.get("stdSS", 0.04)),
            "HRange": exp_cfg.get("HRange", prior_defaults.get("HRange", [1.0, 60.0])),
            "wRange": exp_cfg.get("wRange", prior_defaults.get("wRange", [0.5, 1.5])),
            "vRange": exp_cfg.get("vRange", prior_defaults.get("vRange", [1.0, 4.0])),
            "aRange": exp_cfg.get(
                "aRange",
                exp_cfg.get("slopeRange", prior_defaults.get("aRange", prior_defaults.get("slopeRange", [0.0, 0.0]))),
            ),
            "logeRange": exp_cfg.get("logeRange", prior_defaults.get("logeRange", [0.0, 10.0])),
            "rhoRange": exp_cfg.get("rhoRange", prior_defaults.get("rhoRange", [1.5, 2.2])),
            "totalSteps": exp_cfg.get("totalSteps", runtime_defaults.get("totalSteps", int(1e6))),
            "burnInSteps": exp_cfg.get("burnInSteps", runtime_defaults.get("burnInSteps")),
            "nSaveModels": exp_cfg.get("nSaveModels", runtime_defaults.get("nSaveModels", 200)),
            "actionsPerStep": exp_cfg.get("actionsPerStep", runtime_defaults.get("actionsPerStep", 1)),
            "num_chains": exp_cfg.get("num_chains", runtime_defaults.get("num_chains", 8)),
        }
    )

    resolved["travel_times"] = _normalize_travel_times(exp_constraints.get("travel_times")) or resolved.get("travel_times")
    resolved["group_velocity"] = _normalize_group_velocity(exp_constraints.get("group_velocity")) or resolved.get("group_velocity")
    resolved["avg_vs"] = _normalize_avg_vs(exp_constraints.get("average_vs")) or resolved.get("avg_vs")
    resolved["assumptions"] = _as_plain_dict(_deep_merge(resolved.get("assumptions", {}), exp_assumptions))
    resolved["reference_model"] = _as_plain_dict(exp_cfg.get("reference_model", resolved.get("reference_model")))

    rayp = exp_cfg.get("rayp", resolved.get("rayp"))
    resolved["rayp"] = _resolve_rayp_for_mode(rayp, resolved["mode"])

    return resolved


def _resolve_rayp_for_mode(rayp_cfg, mode):
    if isinstance(rayp_cfg, dict):
        if mode == 1:
            return float(rayp_cfg["PP"])
        if mode == 2:
            return float(rayp_cfg["SS"])
        return [float(rayp_cfg["PP"]), float(rayp_cfg["SS"])]
    if isinstance(rayp_cfg, (list, tuple)):
        if mode in (1, 2) and len(rayp_cfg) == 1:
            return float(rayp_cfg[0])
        return list(rayp_cfg)
    return rayp_cfg


def _resolve_legacy_experiment(config, experiment_name):
    common = copy.deepcopy(config.get("common", {}))
    experiments = config.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError("Legacy config expects 'experiments' to be a list.")

    selected = None
    for idx, exp in enumerate(experiments, start=1):
        candidate_name = exp.get("name") or exp.get("runname") or f"experiment_{idx}"
        if candidate_name == experiment_name:
            selected = copy.deepcopy(exp)
            selected["experiment_name"] = candidate_name
            break
    if selected is None:
        available = ", ".join(list_experiments(config))
        raise KeyError(f"Unknown experiment '{experiment_name}'. Available: {available}")

    resolved = _deep_merge(common, selected)
    resolved["mode"] = _canonical_mode(resolved["mode"])
    resolved["event_name"] = resolved.get("evname")
    resolved["travel_times"] = _normalize_travel_times(resolved.get("travel_times"))
    resolved["group_velocity"] = _normalize_group_velocity(resolved.get("group_velocity"))
    resolved["avg_vs"] = _normalize_avg_vs(resolved.get("average_vs") or resolved.get("avg_vs"))
    resolved["assumptions"] = _as_plain_dict(resolved.get("assumptions", {}))
    resolved["reference_model"] = _as_plain_dict(resolved.get("reference_model"))
    resolved["fitWaveform"] = resolved.get("fitWaveform", True)
    resolved["rayp"] = _resolve_rayp_for_mode(resolved.get("rayp"), resolved["mode"])
    return resolved


def resolve_experiment(config_path, experiment_name):
    config = load_workflow_config(config_path)
    if "events" in config and isinstance(config.get("experiments"), dict):
        resolved = _resolve_new_experiment(config, experiment_name)
    else:
        resolved = _resolve_legacy_experiment(config, experiment_name)
    resolved["config_path"] = str(Path(config_path).resolve())
    return resolved


def resolve_event(config_path, event_name):
    config = load_workflow_config(config_path)
    if "events" not in config:
        raise ValueError("Event-based prep requires the new config format with a top-level 'events' mapping.")

    defaults = config.get("defaults", {})
    path_cfg = _resolve_path_cfg(config.get("paths", {}))
    event_cfg = config.get("events", {}).get(event_name)
    if not event_cfg:
        available = ", ".join(sorted(config.get("events", {})))
        raise KeyError(f"Unknown event '{event_name}'. Available: {available}")

    resolved = _flatten_event(path_cfg, defaults, event_name, event_cfg)
    resolved["config_path"] = str(Path(config_path).resolve())
    return resolved
