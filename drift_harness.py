#!/usr/bin/env python3
"""
Drift Detection Harness for Standard Benchmarks

Integrates SAE-based drift detection with:
- lm-evaluation-harness (EleutherAI)
- Inspect (UK AISI)
- Custom benchmark suites

Tracks both capability scores AND internal feature distributions over time.

Usage:
    # Run lm-eval and capture features for drift analysis
    python drift_harness.py run-lmeval --tasks mmlu,hellaswag --output results/

    # Compare two eval runs for drift
    python drift_harness.py compare --baseline results/v1/ --current results/v2/

    # Monitor a model endpoint over time
    python drift_harness.py monitor --config monitor_config.yaml
"""

import json
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import hashlib


@dataclass
class DriftSignature:
    """Combined capability + feature signature for a model version."""
    version_id: str
    timestamp: str

    # Capability scores from standard benchmarks
    lm_eval_scores: dict          # task -> score
    aggregate_score: float        # Weighted average

    # SAE feature distribution (our addition)
    feature_mean: np.ndarray      # 32-dim
    feature_std: np.ndarray       # 32-dim

    # Metadata
    model_name: str
    eval_config: dict


class DriftHarness:
    """
    Orchestrates drift detection across standard benchmarks.

    Design philosophy:
    - Use lm-eval-harness for capability measurement (don't reinvent)
    - Add SAE feature extraction as a parallel signal
    - Track both over time for comprehensive drift detection
    """

    STANDARD_TASKS = {
        # Quick sanity check (~5 min)
        "quick": ["hellaswag", "arc_easy", "piqa"],

        # Standard capability suite (~30 min)
        "standard": [
            "mmlu",              # Knowledge/reasoning
            "hellaswag",         # Commonsense
            "arc_challenge",     # Science reasoning
            "winogrande",        # Coreference
            "truthfulqa_mc2",    # Factuality
        ],

        # Safety-focused (~20 min)
        "safety": [
            "truthfulqa_mc2",
            "toxigen",
            "bbq",               # Bias benchmark
        ],

        # Full regression (~2 hours)
        "full": [
            "mmlu", "hellaswag", "arc_challenge", "arc_easy",
            "winogrande", "piqa", "boolq", "openbookqa",
            "truthfulqa_mc2", "gsm8k",
        ],
    }

    def __init__(self, results_dir: str = "drift_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def run_lm_eval(
        self,
        model: str,
        tasks: list[str],
        output_dir: Optional[str] = None,
        num_fewshot: int = 0,
        batch_size: str = "auto",
    ) -> dict:
        """
        Run lm-evaluation-harness and capture results.

        Wraps: lm_eval --model hf --model_args pretrained=X --tasks Y
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.results_dir / f"{model.replace('/', '_')}_{timestamp}"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Build lm-eval command
        task_str = ",".join(tasks)
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model}",
            "--tasks", task_str,
            "--num_fewshot", str(num_fewshot),
            "--batch_size", batch_size,
            "--output_path", str(output_dir),
            "--log_samples",  # Save individual predictions for feature extraction
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"lm-eval failed: {result.stderr}")
            return {"error": result.stderr}

        # Parse results
        results_file = output_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                return json.load(f)

        return {"output_dir": str(output_dir), "stdout": result.stdout}

    def run_inspect_eval(
        self,
        model: str,
        eval_name: str,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Run UK AISI Inspect evaluation.

        Wraps: inspect eval <eval_name> --model <model>
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.results_dir / f"inspect_{eval_name}_{timestamp}"
        else:
            output_dir = Path(output_dir)

        cmd = [
            "inspect", "eval", eval_name,
            "--model", model,
            "--log-dir", str(output_dir),
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        return {
            "output_dir": str(output_dir),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def extract_signatures_from_lmeval(
        self,
        results_dir: str,
        sample_limit: int = 100,
    ) -> dict:
        """
        Extract SAE feature signatures from lm-eval logged samples.

        lm-eval with --log_samples saves individual model outputs.
        We extract features from these to build a drift fingerprint.
        """
        from drift import DriftDetector

        results_path = Path(results_dir)
        detector = DriftDetector()

        signatures = {}
        for samples_file in results_path.glob("**/samples_*.jsonl"):
            task_name = samples_file.stem.replace("samples_", "")

            texts = []
            with open(samples_file) as f:
                for i, line in enumerate(f):
                    if i >= sample_limit:
                        break
                    sample = json.loads(line)
                    # Extract model output (format varies by task)
                    if "resps" in sample:
                        text = str(sample["resps"])
                    elif "filtered_resps" in sample:
                        text = str(sample["filtered_resps"])
                    else:
                        continue
                    texts.append(text)

            if texts:
                features = detector.extract_batch(texts, show_progress=False)
                signatures[task_name] = {
                    "n_samples": len(texts),
                    "feature_mean": features.mean(axis=0).tolist(),
                    "feature_std": features.std(axis=0).tolist(),
                }

        return signatures

    def compare_runs(
        self,
        baseline_dir: str,
        current_dir: str,
    ) -> dict:
        """
        Compare two evaluation runs for drift.

        Returns both:
        1. Capability drift (score changes)
        2. Feature drift (distribution shifts)
        """
        baseline_path = Path(baseline_dir)
        current_path = Path(current_dir)

        # Load results
        baseline_results = json.load(open(baseline_path / "results.json"))
        current_results = json.load(open(current_path / "results.json"))

        # Capability comparison
        capability_drift = {}
        for task in baseline_results.get("results", {}):
            if task in current_results.get("results", {}):
                baseline_score = baseline_results["results"][task].get("acc,none", 0)
                current_score = current_results["results"][task].get("acc,none", 0)
                delta = current_score - baseline_score

                capability_drift[task] = {
                    "baseline": baseline_score,
                    "current": current_score,
                    "delta": delta,
                    "percent_change": (delta / baseline_score * 100) if baseline_score else 0,
                }

        # Feature signature comparison
        baseline_sigs = self.extract_signatures_from_lmeval(baseline_dir)
        current_sigs = self.extract_signatures_from_lmeval(current_dir)

        feature_drift = {}
        for task in baseline_sigs:
            if task in current_sigs:
                b_mean = np.array(baseline_sigs[task]["feature_mean"])
                c_mean = np.array(current_sigs[task]["feature_mean"])
                b_std = np.array(baseline_sigs[task]["feature_std"])

                # Z-score of mean shift
                std_safe = np.where(b_std > 0, b_std, 1)
                z_scores = np.abs((c_mean - b_mean) / std_safe)

                feature_drift[task] = {
                    "mean_z_score": float(z_scores.mean()),
                    "max_z_score": float(z_scores.max()),
                    "top_shifted_features": int(np.sum(z_scores > 2)),  # Features with z > 2
                }

        return {
            "timestamp": datetime.now().isoformat(),
            "baseline_dir": baseline_dir,
            "current_dir": current_dir,
            "capability_drift": capability_drift,
            "feature_drift": feature_drift,
            "summary": {
                "capability_regression": any(d["delta"] < -0.02 for d in capability_drift.values()),
                "feature_shift_detected": any(d["mean_z_score"] > 2 for d in feature_drift.values()),
            }
        }

    def generate_monitoring_config(self, model: str, output_path: str):
        """Generate a YAML config for continuous monitoring."""
        config = f"""
# Drift Monitoring Configuration
# Run with: python drift_harness.py monitor --config {output_path}

model: {model}
schedule: "0 6 * * *"  # Daily at 6 AM

# Which benchmark suites to run
suites:
  - name: quick_check
    tasks: {self.STANDARD_TASKS['quick']}
    frequency: daily

  - name: full_eval
    tasks: {self.STANDARD_TASKS['standard']}
    frequency: weekly

# Alert thresholds
thresholds:
  capability_drop: 0.02        # Alert if any task drops >2%
  feature_z_score: 3.0         # Alert if feature distribution shifts >3 sigma
  aggregate_drift: 0.05        # Alert if overall score drops >5%

# Notification
alerts:
  slack_webhook: null          # Set to enable Slack alerts
  email: null                  # Set to enable email alerts

# Storage
results_dir: drift_results/
retention_days: 90
"""
        with open(output_path, 'w') as f:
            f.write(config)
        print(f"Config written to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Drift detection with standard benchmarks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run-lmeval
    lm_parser = subparsers.add_parser("run-lmeval", help="Run lm-evaluation-harness")
    lm_parser.add_argument("--model", "-m", required=True)
    lm_parser.add_argument("--tasks", "-t", default="quick", help="Task list or preset name")
    lm_parser.add_argument("--output", "-o")

    # compare
    cmp_parser = subparsers.add_parser("compare", help="Compare two evaluation runs")
    cmp_parser.add_argument("--baseline", "-b", required=True)
    cmp_parser.add_argument("--current", "-c", required=True)
    cmp_parser.add_argument("--output", "-o")

    # init-config
    init_parser = subparsers.add_parser("init-config", help="Generate monitoring config")
    init_parser.add_argument("--model", "-m", required=True)
    init_parser.add_argument("--output", "-o", default="monitor_config.yaml")

    args = parser.parse_args()
    harness = DriftHarness()

    if args.command == "run-lmeval":
        tasks = harness.STANDARD_TASKS.get(args.tasks, args.tasks.split(","))
        results = harness.run_lm_eval(args.model, tasks, args.output)
        print(json.dumps(results, indent=2))

    elif args.command == "compare":
        comparison = harness.compare_runs(args.baseline, args.current)
        print(json.dumps(comparison, indent=2))

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(comparison, f, indent=2)

    elif args.command == "init-config":
        harness.generate_monitoring_config(args.model, args.output)


if __name__ == "__main__":
    main()
