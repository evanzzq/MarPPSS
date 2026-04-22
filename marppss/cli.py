import argparse
from pathlib import Path

from marppss.config import (
    list_events,
    list_experiments,
    load_workflow_config,
    resolve_event,
    resolve_experiment,
)
from marppss.run_plot import plot_run_directory
from marppss.workflow import ensure_prepared_data, run_resolved_experiment


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m marppss")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prep", help="Prepare inversion-ready data from SAC files.")
    prep.add_argument("config", help="Path to workflow YAML.")
    target = prep.add_mutually_exclusive_group(required=True)
    target.add_argument("--event", help="Event name from a new-style workflow config.")
    target.add_argument("--experiment", help="Experiment name; prep uses the experiment's linked event.")
    prep.add_argument("--force", action="store_true", help="Rebuild prepared data even if files already exist.")

    run = subparsers.add_parser("run", help="Run RJMCMC inversion for one experiment.")
    run.add_argument("config", help="Path to workflow YAML.")
    run.add_argument("--experiment", required=True, help="Experiment name to run.")
    run.add_argument("--force-prep", action="store_true", help="Rebuild prepared data before running.")

    plot = subparsers.add_parser("plot", help="Plot results from a run directory.")
    plot.add_argument("run_dir", help="Path to a run directory containing config_resolved.yaml.")
    plot.add_argument("--top-chains", type=int, default=None, help="Keep only the top N chains by final logL.")
    plot.add_argument(
        "--reference-config",
        help="Optional YAML to override only the plotted reference_model for the saved experiment.",
    )

    ls = subparsers.add_parser("list", help="List configured events and experiments.")
    ls.add_argument("config", help="Path to workflow YAML.")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prep":
        if args.event:
            resolved = resolve_event(args.config, args.event)
        else:
            resolved = resolve_experiment(args.config, args.experiment)
        ensure_prepared_data(resolved, force=args.force)
        print(f"Prepared data under: {resolved['outdir']}")
        return

    if args.command == "run":
        resolved = resolve_experiment(args.config, args.experiment)
        run_root = run_resolved_experiment(resolved, force_prep=args.force_prep)
        print(run_root)
        return

    if args.command == "plot":
        plot_run_directory(
            str(Path(args.run_dir).resolve()),
            top_chains=args.top_chains,
            reference_config=args.reference_config,
        )
        return

    if args.command == "list":
        config = load_workflow_config(args.config)
        print("Events:")
        for name in list_events(config):
            print(f"  - {name}")
        print("Experiments:")
        for name in list_experiments(config):
            print(f"  - {name}")
        return
