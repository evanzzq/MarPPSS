#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marppss.config import list_experiments, load_workflow_config, resolve_experiment
from marppss.run_plot import plot_run_directory
from marppss.workflow import get_run_root


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
    parser = argparse.ArgumentParser(description="Plot multiple MarPPSS experiments sequentially.")
    parser.add_argument("config", help="Path to workflow YAML.")
    parser.add_argument(
        "experiments",
        nargs="*",
        help="Explicit experiment names to plot.",
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
        help="Plot every experiment in the config.",
    )
    parser.add_argument(
        "--reference-config",
        help="Optional YAML to override only the plotted reference_model.",
    )
    parser.add_argument(
        "--top-chains",
        type=int,
        default=None,
        help="Keep only the top N chains by final logL. If omitted, the plotter will prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected experiments and resolved run directories without plotting.",
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
    resolved_items = []
    for name in experiments:
        resolved = resolve_experiment(config_path, name)
        run_root = get_run_root(resolved)
        resolved_items.append((name, run_root))
        print(f"  - {name}")
        print(f"    {run_root}")

    if args.dry_run:
        return

    for idx, (name, run_root) in enumerate(resolved_items, start=1):
        print(f"\n[{idx}/{len(resolved_items)}] Plotting {name}")
        plot_run_directory(
            run_root,
            top_chains=args.top_chains,
            reference_config=args.reference_config,
        )


if __name__ == "__main__":
    main()
