#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marppss.config import list_experiments, load_workflow_config, resolve_experiment
from marppss.workflow import run_resolved_experiment


def _select_experiments(config_path, explicit, use_all, prefixes):
    config = load_workflow_config(config_path)
    available = list_experiments(config)
    selected = []

    if explicit:
        selected.extend(explicit)

    if use_all:
        selected.extend(available)

    for prefix in prefixes:
        matches = [name for name in available if name.startswith(prefix)]
        if not matches:
            raise ValueError(f"No experiments match prefix '{prefix}'.")
        selected.extend(matches)

    if not selected:
        raise ValueError("No experiments selected. Use positional names, --prefix, or --all.")

    deduped = []
    seen = set()
    for name in selected:
        if name not in available:
            raise ValueError(f"Unknown experiment '{name}'.")
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)

    return deduped


def build_parser():
    parser = argparse.ArgumentParser(description="Run multiple MarPPSS experiments sequentially.")
    parser.add_argument("config", help="Path to workflow YAML.")
    parser.add_argument(
        "experiments",
        nargs="*",
        help="Explicit experiment names to run.",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="Add all experiments whose names start with this prefix. Repeatable.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every experiment in the config.",
    )
    parser.add_argument(
        "--force-prep",
        action="store_true",
        help="Rebuild prepared data before each run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected experiments without running them.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = str(Path(args.config).resolve())
    experiments = _select_experiments(
        config_path,
        explicit=args.experiments,
        use_all=args.all,
        prefixes=args.prefix,
    )

    print("Selected experiments:")
    for name in experiments:
        print(f"  - {name}")

    if args.dry_run:
        return

    for idx, name in enumerate(experiments, start=1):
        print(f"\n[{idx}/{len(experiments)}] Running {name}")
        resolved = resolve_experiment(config_path, name)
        run_root = run_resolved_experiment(resolved, force_prep=args.force_prep)
        print(f"Saved to: {run_root}")


if __name__ == "__main__":
    main()
