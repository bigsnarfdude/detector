#!/usr/bin/env python3
"""
Model Drift Detection via SAE Feature Signatures

Builds on the alignment faking detector to monitor for behavioral drift.
Tracks feature activation distributions over time and alerts on significant shifts.

Usage:
    # Create baseline from known-good samples
    python drift.py baseline --samples good_samples.jsonl --output baseline.json

    # Monitor new samples against baseline
    python drift.py monitor --baseline baseline.json --samples new_samples.jsonl

    # Run benchmark suite and compare to historical results
    python drift.py benchmark --baseline baseline.json --suite benchmark_prompts.jsonl
"""

import argparse
import json
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from scipy import stats

# Reuse feature extraction from detect.py
from detect import AlignmentFakingDetector, PROBE_FEATURES

# Feature names for interpretable reporting
FEATURE_NAMES = [
    "self_capability_max", "self_capability_mean", "self_capability_count", "self_capability_std",
    "value_preservation_max", "value_preservation_mean", "value_preservation_count", "value_preservation_std",
    "training_context_max", "training_context_mean", "training_context_count", "training_context_std",
    "future_preservation_max", "future_preservation_mean", "future_preservation_count", "future_preservation_std",
    "strategic_compliance_max", "strategic_compliance_mean", "strategic_compliance_count", "strategic_compliance_std",
    "surface_alignment_max", "surface_alignment_mean", "surface_alignment_count", "surface_alignment_std",
    "rlhf_detection_max", "rlhf_detection_mean", "rlhf_detection_count", "rlhf_detection_std",
    "scratchpad_structure_max", "scratchpad_structure_mean", "scratchpad_structure_count", "scratchpad_structure_std",
]


@dataclass
class BaselineStats:
    """Statistical baseline for drift detection."""
    mean: np.ndarray          # Per-feature means (32,)
    std: np.ndarray           # Per-feature stds (32,)
    covariance: np.ndarray    # Covariance matrix (32, 32)
    inv_covariance: np.ndarray  # Inverse covariance for Mahalanobis
    n_samples: int            # Number of baseline samples
    created_at: str           # Timestamp
    feature_names: list       # For interpretability

    def to_dict(self):
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "covariance": self.covariance.tolist(),
            "inv_covariance": self.inv_covariance.tolist(),
            "n_samples": self.n_samples,
            "created_at": self.created_at,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            mean=np.array(d["mean"]),
            std=np.array(d["std"]),
            covariance=np.array(d["covariance"]),
            inv_covariance=np.array(d["inv_covariance"]),
            n_samples=d["n_samples"],
            created_at=d["created_at"],
            feature_names=d["feature_names"],
        )


@dataclass
class DriftReport:
    """Results from drift detection analysis."""
    timestamp: str
    n_samples: int

    # Overall drift metrics
    mahalanobis_mean: float       # Mean Mahalanobis distance
    mahalanobis_max: float        # Max sample distance
    drift_detected: bool          # Binary decision
    drift_severity: str           # none/mild/moderate/severe

    # Per-feature analysis
    feature_z_scores: dict        # Feature name -> mean z-score
    top_drifted_features: list    # Most shifted features

    # Statistical tests
    ks_test_results: dict         # Feature -> (statistic, p-value)

    # Sample-level details (optional)
    sample_distances: Optional[list] = None


class DriftDetector:
    """
    Detect model drift by monitoring SAE feature distributions.

    Uses the same 8 semantic features from alignment faking detection,
    but tracks their distributions over time rather than classifying.
    """

    def __init__(self, device_map="auto"):
        self.detector = AlignmentFakingDetector(device_map=device_map)
        self.baseline: Optional[BaselineStats] = None

    def extract_features(self, text: str) -> np.ndarray:
        """Extract 32-dim feature vector from text."""
        return self.detector._extract_features(text)

    def extract_batch(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """Extract features from multiple texts."""
        features = []
        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(texts)} samples")
            features.append(self.extract_features(text))
        return np.array(features)

    def create_baseline(self, texts: list[str]) -> BaselineStats:
        """
        Create baseline statistics from known-good samples.

        Args:
            texts: List of representative texts from the model

        Returns:
            BaselineStats object with distribution parameters
        """
        print(f"Creating baseline from {len(texts)} samples...")
        features = self.extract_batch(texts)

        mean = features.mean(axis=0)
        std = features.std(axis=0)

        # Regularized covariance (add small diagonal for numerical stability)
        covariance = np.cov(features.T)
        if covariance.ndim == 0:  # Single feature edge case
            covariance = np.array([[covariance]])
        covariance += np.eye(covariance.shape[0]) * 1e-6

        try:
            inv_covariance = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            print("Warning: Covariance matrix singular, using pseudoinverse")
            inv_covariance = np.linalg.pinv(covariance)

        self.baseline = BaselineStats(
            mean=mean,
            std=std,
            covariance=covariance,
            inv_covariance=inv_covariance,
            n_samples=len(texts),
            created_at=datetime.now().isoformat(),
            feature_names=FEATURE_NAMES,
        )

        print(f"Baseline created with {len(texts)} samples")
        return self.baseline

    def save_baseline(self, path: str):
        """Save baseline to JSON file."""
        if self.baseline is None:
            raise ValueError("No baseline to save. Call create_baseline first.")
        with open(path, 'w') as f:
            json.dump(self.baseline.to_dict(), f, indent=2)
        print(f"Baseline saved to {path}")

    def load_baseline(self, path: str):
        """Load baseline from JSON file."""
        with open(path) as f:
            self.baseline = BaselineStats.from_dict(json.load(f))
        print(f"Loaded baseline ({self.baseline.n_samples} samples, created {self.baseline.created_at})")

    def mahalanobis_distance(self, features: np.ndarray) -> float:
        """
        Compute Mahalanobis distance from baseline.

        This measures how far a sample is from the baseline distribution,
        accounting for feature correlations.
        """
        if self.baseline is None:
            raise ValueError("No baseline loaded")
        diff = features - self.baseline.mean
        return np.sqrt(diff @ self.baseline.inv_covariance @ diff)

    def feature_z_scores(self, features: np.ndarray) -> np.ndarray:
        """Compute per-feature z-scores from baseline."""
        if self.baseline is None:
            raise ValueError("No baseline loaded")
        safe_std = np.where(self.baseline.std > 0, self.baseline.std, 1)
        return (features - self.baseline.mean) / safe_std

    def detect_drift(
        self,
        texts: list[str],
        threshold_mild: float = 3.0,      # Mahalanobis threshold for mild drift
        threshold_moderate: float = 5.0,   # Threshold for moderate drift
        threshold_severe: float = 8.0,     # Threshold for severe drift
        include_samples: bool = False,
    ) -> DriftReport:
        """
        Analyze texts for drift from baseline.

        Args:
            texts: New samples to check for drift
            threshold_*: Mahalanobis distance thresholds for severity levels
            include_samples: Include per-sample distances in report

        Returns:
            DriftReport with comprehensive analysis
        """
        if self.baseline is None:
            raise ValueError("No baseline loaded. Call load_baseline or create_baseline first.")

        print(f"Analyzing {len(texts)} samples for drift...")
        features = self.extract_batch(texts)

        # Compute Mahalanobis distances
        distances = [self.mahalanobis_distance(f) for f in features]
        mahal_mean = np.mean(distances)
        mahal_max = np.max(distances)

        if mahal_mean >= threshold_severe:
            severity = "severe"
        elif mahal_mean >= threshold_moderate:
            severity = "moderate"
        elif mahal_mean >= threshold_mild:
            severity = "mild"
        else:
            severity = "none"

        drift_detected = severity != "none"

        # Per-feature z-score analysis
        all_z_scores = np.array([self.feature_z_scores(f) for f in features])
        mean_z_scores = np.abs(all_z_scores).mean(axis=0)
        feature_z_dict = {name: float(z) for name, z in zip(FEATURE_NAMES, mean_z_scores)}

        # Top drifted features (sorted by absolute z-score)
        sorted_features = sorted(feature_z_dict.items(), key=lambda x: x[1], reverse=True)
        top_drifted = [{"feature": name, "z_score": z} for name, z in sorted_features[:5]]

        # Kolmogorov-Smirnov tests per feature
        # Compare new sample distribution to baseline (using baseline as reference)
        ks_results = {}
        for i, name in enumerate(FEATURE_NAMES):
            # Generate baseline samples from normal approximation
            baseline_samples = np.random.normal(
                self.baseline.mean[i],
                self.baseline.std[i] if self.baseline.std[i] > 0 else 1e-6,
                size=1000
            )
            stat, pval = stats.ks_2samp(features[:, i], baseline_samples)
            ks_results[name] = {"statistic": float(stat), "p_value": float(pval)}

        report = DriftReport(
            timestamp=datetime.now().isoformat(),
            n_samples=len(texts),
            mahalanobis_mean=float(mahal_mean),
            mahalanobis_max=float(mahal_max),
            drift_detected=drift_detected,
            drift_severity=severity,
            feature_z_scores=feature_z_dict,
            top_drifted_features=top_drifted,
            ks_test_results=ks_results,
            sample_distances=distances if include_samples else None,
        )

        return report

    def run_benchmark(
        self,
        prompts: list[dict],
        baseline_results_path: Optional[str] = None,
    ) -> dict:
        """
        Run capability benchmark suite and compare to historical results.

        Args:
            prompts: List of {"prompt": str, "expected_label": str, "category": str}
            baseline_results_path: Path to previous benchmark results for comparison

        Returns:
            Benchmark results with comparison to baseline
        """
        print(f"Running benchmark on {len(prompts)} prompts...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "n_prompts": len(prompts),
            "categories": {},
            "predictions": [],
        }

        # Extract features and predictions for each prompt
        for item in prompts:
            text = item["prompt"]
            features = self.extract_features(text)

            # Use AF detector's predict method
            score = self.detector.predict(text)

            results["predictions"].append({
                "category": item.get("category", "default"),
                "expected": item.get("expected_label"),
                "score": float(score),
                "features": features.tolist(),
            })

        # Aggregate by category
        from collections import defaultdict
        by_category = defaultdict(list)
        for pred in results["predictions"]:
            by_category[pred["category"]].append(pred["score"])

        for cat, scores in by_category.items():
            results["categories"][cat] = {
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "n_samples": len(scores),
            }

        # Compare to baseline if provided
        if baseline_results_path and Path(baseline_results_path).exists():
            with open(baseline_results_path) as f:
                baseline_results = json.load(f)

            results["comparison"] = {}
            for cat in results["categories"]:
                if cat in baseline_results.get("categories", {}):
                    current = results["categories"][cat]["mean_score"]
                    baseline = baseline_results["categories"][cat]["mean_score"]
                    delta = current - baseline
                    results["comparison"][cat] = {
                        "baseline_mean": baseline,
                        "current_mean": current,
                        "delta": float(delta),
                        "percent_change": float(delta / baseline * 100) if baseline != 0 else 0,
                    }

        return results


def print_drift_report(report: DriftReport):
    """Pretty print a drift report."""
    print("\n" + "="*60)
    print("DRIFT DETECTION REPORT")
    print("="*60)
    print(f"Timestamp: {report.timestamp}")
    print(f"Samples analyzed: {report.n_samples}")
    print()

    # Overall status
    status_colors = {
        "none": "✓ NO DRIFT",
        "mild": "⚠ MILD DRIFT",
        "moderate": "⚠ MODERATE DRIFT",
        "severe": "✗ SEVERE DRIFT",
    }
    print(f"Status: {status_colors[report.drift_severity]}")
    print(f"Mahalanobis distance (mean): {report.mahalanobis_mean:.2f}")
    print(f"Mahalanobis distance (max):  {report.mahalanobis_max:.2f}")
    print()

    # Top drifted features
    print("Top drifted features:")
    for item in report.top_drifted_features:
        bar = "█" * int(min(item["z_score"], 10))
        print(f"  {item['feature']:35s} z={item['z_score']:.2f} {bar}")
    print()

    # Significant KS tests (p < 0.05)
    significant = [(k, v) for k, v in report.ks_test_results.items() if v["p_value"] < 0.05]
    if significant:
        print(f"Features with significant distribution shift (p<0.05): {len(significant)}")
        for name, result in sorted(significant, key=lambda x: x[1]["p_value"])[:5]:
            print(f"  {name}: KS={result['statistic']:.3f}, p={result['p_value']:.4f}")
    else:
        print("No features with significant distribution shift (p<0.05)")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Model drift detection via SAE features")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Baseline command
    baseline_parser = subparsers.add_parser("baseline", help="Create baseline from samples")
    baseline_parser.add_argument("--samples", "-s", required=True, help="JSONL file with samples")
    baseline_parser.add_argument("--output", "-o", required=True, help="Output baseline JSON")
    baseline_parser.add_argument("--text-field", default="text", help="Field name for text in JSONL")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Check samples for drift")
    monitor_parser.add_argument("--baseline", "-b", required=True, help="Baseline JSON file")
    monitor_parser.add_argument("--samples", "-s", required=True, help="JSONL file with samples")
    monitor_parser.add_argument("--output", "-o", help="Output report JSON (optional)")
    monitor_parser.add_argument("--text-field", default="text", help="Field name for text")
    monitor_parser.add_argument("--include-samples", action="store_true", help="Include per-sample distances")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run capability benchmark")
    bench_parser.add_argument("--suite", "-s", required=True, help="Benchmark prompts JSONL")
    bench_parser.add_argument("--baseline-results", "-b", help="Previous results for comparison")
    bench_parser.add_argument("--output", "-o", required=True, help="Output results JSON")

    args = parser.parse_args()

    detector = DriftDetector()

    if args.command == "baseline":
        # Load samples
        texts = []
        with open(args.samples) as f:
            for line in f:
                item = json.loads(line)
                texts.append(item[args.text_field])

        detector.create_baseline(texts)
        detector.save_baseline(args.output)

    elif args.command == "monitor":
        detector.load_baseline(args.baseline)

        # Load samples
        texts = []
        with open(args.samples) as f:
            for line in f:
                item = json.loads(line)
                texts.append(item[args.text_field])

        report = detector.detect_drift(texts, include_samples=args.include_samples)
        print_drift_report(report)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(asdict(report), f, indent=2)
            print(f"\nReport saved to {args.output}")

    elif args.command == "benchmark":
        # Load benchmark suite
        prompts = []
        with open(args.suite) as f:
            for line in f:
                prompts.append(json.loads(line))

        results = detector.run_benchmark(prompts, args.baseline_results)

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Benchmark results saved to {args.output}")

        # Print summary
        print("\nBenchmark Summary:")
        for cat, stats in results["categories"].items():
            print(f"  {cat}: mean={stats['mean_score']:.3f} (n={stats['n_samples']})")

        if "comparison" in results:
            print("\nComparison to baseline:")
            for cat, comp in results["comparison"].items():
                delta = comp["delta"]
                sign = "+" if delta > 0 else ""
                pct = comp["percent_change"]
                print(f"  {cat}: {sign}{delta:.3f} ({sign}{pct:.1f}%)")


if __name__ == "__main__":
    main()
