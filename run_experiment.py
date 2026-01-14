#!/usr/bin/env python3
"""
Convenience script for running experiments.

Usage:
    # Run with config file
    ./run_experiment.py --config experiments/configs/drift_detection.yaml

    # Quick baseline creation
    ./run_experiment.py baseline --input data/samples.jsonl --name my_baseline

    # Quick drift check
    ./run_experiment.py drift --input data/new_samples.jsonl --baseline experiments/XXX/outputs/baseline.json

    # List recent experiments
    ./run_experiment.py list

    # Show experiment details
    ./run_experiment.py show 20260114_baseline_creation
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from experiment import Experiment, ExperimentConfig, ExperimentRunner


def run_with_config(config_path: str):
    """Run experiment from config file."""
    if config_path.endswith(('.yaml', '.yml')):
        config = ExperimentConfig.from_yaml(config_path)
    else:
        config = ExperimentConfig.from_json(config_path)

    exp = Experiment(config)
    return exp.run()


def run_baseline(input_file: str, name: str, description: str = "", limit: int = None):
    """Quick baseline creation."""
    config = ExperimentConfig(
        name=name,
        description=description or f"Baseline from {input_file}",
        input_file=input_file,
        create_baseline=True,
        run_af_detection=True,
        run_drift_detection=False,
        sample_limit=limit,
        tags=["baseline", "quick"],
    )
    exp = Experiment(config)
    return exp.run()


def run_drift(input_file: str, baseline_file: str, name: str = None, limit: int = None):
    """Quick drift detection."""
    config = ExperimentConfig(
        name=name or "drift_check",
        description=f"Drift check against {baseline_file}",
        input_file=input_file,
        baseline_file=baseline_file,
        create_baseline=False,
        run_af_detection=True,
        run_drift_detection=True,
        sample_limit=limit,
        tags=["drift", "quick"],
    )
    exp = Experiment(config)
    return exp.run()


def main():
    parser = argparse.ArgumentParser(description="Run behavioral integrity experiments")
    subparsers = parser.add_subparsers(dest="command")

    # Config-based run
    config_parser = subparsers.add_parser("config", help="Run from config file")
    config_parser.add_argument("--config", "-c", required=True, help="Config file path")

    # Quick baseline
    baseline_parser = subparsers.add_parser("baseline", help="Quick baseline creation")
    baseline_parser.add_argument("--input", "-i", required=True, help="Input samples JSONL")
    baseline_parser.add_argument("--name", "-n", required=True, help="Experiment name")
    baseline_parser.add_argument("--description", "-d", default="", help="Description")
    baseline_parser.add_argument("--limit", "-l", type=int, help="Sample limit")

    # Quick drift
    drift_parser = subparsers.add_parser("drift", help="Quick drift detection")
    drift_parser.add_argument("--input", "-i", required=True, help="Input samples JSONL")
    drift_parser.add_argument("--baseline", "-b", required=True, help="Baseline JSON file")
    drift_parser.add_argument("--name", "-n", help="Experiment name")
    drift_parser.add_argument("--limit", "-l", type=int, help="Sample limit")

    # List experiments
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--limit", "-l", type=int, default=20, help="Number to show")

    # Show experiment
    show_parser = subparsers.add_parser("show", help="Show experiment details")
    show_parser.add_argument("exp_id", help="Experiment ID")

    # Compare experiments
    cmp_parser = subparsers.add_parser("compare", help="Compare two experiments")
    cmp_parser.add_argument("--exp1", required=True, help="First experiment")
    cmp_parser.add_argument("--exp2", required=True, help="Second experiment")

    args = parser.parse_args()

    if args.command == "config":
        run_with_config(args.config)

    elif args.command == "baseline":
        run_baseline(args.input, args.name, args.description, args.limit)

    elif args.command == "drift":
        run_drift(args.input, args.baseline, args.name, args.limit)

    elif args.command == "list":
        runner = ExperimentRunner()
        experiments = runner.list_experiments()[:args.limit]
        if not experiments:
            print("No experiments found.")
            return

        print(f"\n{'ID':<45} {'Status':<10} {'Samples':<8} {'AF':<8} {'Drift'}")
        print("-" * 85)
        for exp in experiments:
            drift = "Yes" if exp["drift"] else "No" if exp["drift"] is False else "-"
            print(f"{exp['id']:<45} {exp['status']:<10} {exp['n_samples']:<8} {exp['mean_af']:<8.3f} {drift}")

    elif args.command == "show":
        runner = ExperimentRunner()
        manifest = runner.get_experiment(args.exp_id)
        if manifest:
            import json
            print(json.dumps(manifest.to_dict(), indent=2))
        else:
            print(f"Experiment not found: {args.exp_id}")

    elif args.command == "compare":
        runner = ExperimentRunner()
        import json
        comparison = runner.compare_experiments(args.exp1, args.exp2)
        print(json.dumps(comparison, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
