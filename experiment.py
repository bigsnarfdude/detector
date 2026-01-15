#!/usr/bin/env python3
"""
Experiment Framework for Behavioral Integrity Monitoring

Provides reproducible, auditable experiment management:
- Structured experiment folders with dates and names
- Input/output tracking with hashes for integrity
- Configuration versioning
- Automatic logging and result serialization

Usage:
    # Create and run an experiment
    python experiment.py run --name baseline_creation --config config.yaml

    # List experiments
    python experiment.py list

    # Compare two experiments
    python experiment.py compare --exp1 20260114_baseline_v1 --exp2 20260115_drift_check

Folder structure:
    experiments/
    └── 20260114_baseline_creation/
        ├── config.json          # Experiment configuration
        ├── manifest.json        # File hashes, timestamps, metadata
        ├── inputs/
        │   └── samples.jsonl    # Input data
        ├── outputs/
        │   ├── baseline.json    # Created baseline
        │   ├── report.json      # Integrity report
        │   └── predictions.jsonl # Per-sample predictions
        ├── logs/
        │   └── run.log          # Execution log
        └── README.md            # Auto-generated summary
"""

import json
import hashlib
import logging
import shutil
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    # Experiment metadata
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Input configuration
    input_file: Optional[str] = None          # JSONL with samples
    input_text_field: str = "text"            # Field name for text
    input_label_field: str = "label"          # Field name for label (optional)
    sample_limit: Optional[int] = None        # Limit number of samples

    # Baseline configuration
    baseline_file: Optional[str] = None       # Existing baseline to compare against
    create_baseline: bool = False             # Whether to create new baseline
    model_id: str = "gemma-3-27b-it"          # Model identifier for baseline

    # Analysis configuration
    run_af_detection: bool = True             # Run per-sample AF detection
    run_drift_detection: bool = True          # Run drift analysis (requires baseline)
    include_sample_details: bool = True       # Include per-sample results
    af_threshold: float = 0.5                 # AF classification threshold

    # Output configuration
    save_features: bool = True                # Save raw feature vectors
    save_predictions: bool = True             # Save per-sample predictions

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        import yaml
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class ExperimentManifest:
    """Manifest tracking experiment artifacts and integrity."""

    experiment_id: str
    created_at: str
    completed_at: Optional[str] = None
    status: str = "created"  # created/running/completed/failed

    # Git info for reproducibility
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False

    # File hashes for integrity verification
    input_hashes: Dict[str, str] = field(default_factory=dict)
    output_hashes: Dict[str, str] = field(default_factory=dict)
    config_hash: str = ""

    # Summary statistics
    n_samples: int = 0
    n_af_detected: int = 0
    mean_af_score: float = 0.0
    drift_detected: Optional[bool] = None

    # Timing
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentManifest":
        with open(path) as f:
            return cls(**json.load(f))


class Experiment:
    """
    Manages a single experiment run with full audit trail.

    Creates structured folder with:
    - Immutable inputs (copied + hashed)
    - Versioned configuration
    - Timestamped outputs
    - Execution logs
    """

    EXPERIMENTS_ROOT = Path("experiments")

    def __init__(
        self,
        config: ExperimentConfig,
        experiments_root: Optional[Path] = None,
    ):
        self.config = config
        self.root = experiments_root or self.EXPERIMENTS_ROOT

        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = config.name.lower().replace(" ", "_").replace("-", "_")
        self.experiment_id = f"{timestamp}_{safe_name}"

        # Set up paths
        self.exp_dir = self.root / self.experiment_id
        self.inputs_dir = self.exp_dir / "inputs"
        self.outputs_dir = self.exp_dir / "outputs"
        self.logs_dir = self.exp_dir / "logs"

        self.manifest = ExperimentManifest(
            experiment_id=self.experiment_id,
            created_at=datetime.now().isoformat(),
        )

        self.logger = None
        self._setup_complete = False

    def setup(self):
        """Create experiment directory structure."""
        if self._setup_complete:
            return

        # Create directories
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.inputs_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        # Set up logging
        log_file = self.logs_dir / "run.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ]
        )
        self.logger = logging.getLogger(self.experiment_id)

        # Save config
        config_path = self.exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        self.manifest.config_hash = self._hash_file(config_path)

        # Copy input files
        if self.config.input_file:
            self._copy_input(self.config.input_file, "samples.jsonl")

        if self.config.baseline_file:
            self._copy_input(self.config.baseline_file, "baseline_input.json")

        # Capture git info
        self._capture_git_info()

        # Save initial manifest
        self.manifest.status = "running"
        self._save_manifest()

        self._setup_complete = True
        self.logger.info(f"Experiment {self.experiment_id} initialized")

    def _hash_file(self, path: Path) -> str:
        """Compute SHA256 hash of file."""
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    def _copy_input(self, src: str, dest_name: str):
        """Copy input file to experiment inputs/ and record hash."""
        src_path = Path(src)
        dest_path = self.inputs_dir / dest_name

        if src_path.exists():
            shutil.copy(src_path, dest_path)
            self.manifest.input_hashes[dest_name] = self._hash_file(dest_path)
            self.logger.info(f"Copied input: {src} -> {dest_name}")
        else:
            self.logger.warning(f"Input file not found: {src}")

    def _capture_git_info(self):
        """Capture current git state for reproducibility."""
        import subprocess

        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=self.root.parent
            )
            if result.returncode == 0:
                self.manifest.git_commit = result.stdout.strip()

            # Get branch
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, cwd=self.root.parent
            )
            if result.returncode == 0:
                self.manifest.git_branch = result.stdout.strip()

            # Check if dirty
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=self.root.parent
            )
            self.manifest.git_dirty = bool(result.stdout.strip())

        except Exception as e:
            self.logger.warning(f"Could not capture git info: {e}")

    def _save_manifest(self):
        """Save manifest to experiment directory."""
        manifest_path = self.exp_dir / "manifest.json"
        self.manifest.save(str(manifest_path))

    def save_output(self, name: str, data: Any, format: str = "json"):
        """
        Save output artifact with hash tracking.

        Args:
            name: Output filename (without extension)
            data: Data to save
            format: 'json', 'jsonl', or 'numpy'
        """
        extension_map = {"json": "json", "jsonl": "jsonl", "numpy": "npy"}
        if format not in extension_map:
            raise ValueError(f"Unknown format: {format}")

        path = self.outputs_dir / f"{name}.{extension_map[format]}"

        if format == "json":
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == "jsonl":
            with open(path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item, default=str) + "\n")
        elif format == "numpy":
            np.save(path, data)

        self.manifest.output_hashes[path.name] = self._hash_file(path)
        self.logger.info(f"Saved output: {path.name}")

    def load_inputs(self) -> List[Dict]:
        """Load input samples from copied input file."""
        samples_path = self.inputs_dir / "samples.jsonl"
        if not samples_path.exists():
            raise FileNotFoundError(f"No input file: {samples_path}")

        samples = []
        with open(samples_path) as f:
            for line in f:
                samples.append(json.loads(line))

        if self.config.sample_limit:
            samples = samples[:self.config.sample_limit]

        return samples

    def run(self):
        """Execute the experiment."""
        from monitor import BehavioralIntegrityMonitor, Baseline, print_report

        self.setup()
        start_time = datetime.now()
        self.logger.info("Starting experiment run")

        try:
            # Load inputs
            samples = self.load_inputs()
            texts = [s[self.config.input_text_field] for s in samples]
            self.manifest.n_samples = len(texts)
            self.logger.info(f"Loaded {len(texts)} samples")

            # Initialize monitor
            monitor = BehavioralIntegrityMonitor()

            # Load or create baseline
            baseline = None
            if self.config.baseline_file:
                baseline_path = self.inputs_dir / "baseline_input.json"
                baseline = Baseline.load(str(baseline_path))
                self.logger.info(f"Loaded baseline: {baseline.n_samples} samples")

            if self.config.create_baseline:
                self.logger.info("Creating new baseline...")
                baseline = monitor.create_baseline(texts, self.config.model_id)
                baseline.save(str(self.outputs_dir / "baseline.json"))
                self.manifest.output_hashes["baseline.json"] = self._hash_file(
                    self.outputs_dir / "baseline.json"
                )

            # Run analysis
            self.logger.info("Running behavioral integrity analysis...")
            report = monitor.analyze(
                texts,
                baseline=baseline if self.config.run_drift_detection else None,
                include_samples=self.config.include_sample_details,
            )

            # Update manifest with results
            self.manifest.mean_af_score = report.mean_af_score
            self.manifest.n_af_detected = int(report.af_detection_rate * len(texts))
            self.manifest.drift_detected = report.drift_detected

            # Save outputs
            self.save_output("report", self._report_to_dict(report))

            if self.config.save_predictions and report.sample_results:
                predictions = [asdict(r) for r in report.sample_results]
                self.save_output("predictions", predictions, format="jsonl")

            if self.config.save_features and report.sample_results:
                features = np.array([r.feature_vector for r in report.sample_results])
                self.save_output("features", features, format="numpy")

            # Print report
            print_report(report)

            # Generate README
            self._generate_readme(report)

            # Mark complete
            self.manifest.status = "completed"
            self.manifest.completed_at = datetime.now().isoformat()
            self.manifest.duration_seconds = (datetime.now() - start_time).total_seconds()
            self._save_manifest()

            self.logger.info(f"Experiment completed in {self.manifest.duration_seconds:.1f}s")
            return report

        except Exception as e:
            self.logger.error(f"Experiment failed: {e}", exc_info=True)
            self.manifest.status = "failed"
            self._save_manifest()
            raise

    def _report_to_dict(self, report) -> dict:
        """Convert IntegrityReport to serializable dict."""
        d = {
            "timestamp": report.timestamp,
            "run_id": report.run_id,
            "n_samples": report.n_samples,
            "mean_af_score": report.mean_af_score,
            "max_af_score": report.max_af_score,
            "af_detection_rate": report.af_detection_rate,
            "drift_detected": report.drift_detected,
            "drift_severity": report.drift_severity,
            "mahalanobis_distance": report.mahalanobis_distance,
            "category_means": report.category_means,
            "category_trends": report.category_trends,
            "alerts": report.alerts,
        }
        if report.drift_result:
            d["drift_result"] = asdict(report.drift_result)
        return d

    def _generate_readme(self, report):
        """Generate human-readable README for the experiment."""
        readme = f"""# Experiment: {self.config.name}

**ID:** {self.experiment_id}
**Created:** {self.manifest.created_at}
**Status:** {self.manifest.status}

## Description
{self.config.description or "No description provided."}

## Configuration
- Input file: {self.config.input_file}
- Sample limit: {self.config.sample_limit or "None"}
- AF threshold: {self.config.af_threshold}
- Baseline: {"Created new" if self.config.create_baseline else self.config.baseline_file or "None"}

## Results Summary
- **Samples analyzed:** {report.n_samples}
- **Mean AF score:** {report.mean_af_score:.3f}
- **Max AF score:** {report.max_af_score:.3f}
- **AF detection rate:** {report.af_detection_rate:.1%}
"""

        if report.drift_detected is not None:
            readme += f"""
### Drift Detection
- **Drift detected:** {report.drift_detected}
- **Severity:** {report.drift_severity}
- **Mahalanobis distance:** {report.mahalanobis_distance:.2f}
"""

        if report.alerts:
            readme += "\n### Alerts\n"
            for alert in report.alerts:
                readme += f"- **{alert['type']}**: {alert['message']}\n"

        readme += f"""
## Reproducibility
- Git commit: {self.manifest.git_commit or "N/A"}
- Git branch: {self.manifest.git_branch or "N/A"}
- Git dirty: {self.manifest.git_dirty}
- Config hash: {self.manifest.config_hash[:16]}...

## Files
### Inputs
"""
        for name, hash in self.manifest.input_hashes.items():
            readme += f"- `{name}`: {hash[:16]}...\n"

        readme += "\n### Outputs\n"
        for name, hash in self.manifest.output_hashes.items():
            readme += f"- `{name}`: {hash[:16]}...\n"

        readme += f"""
## Tags
{', '.join(self.config.tags) if self.config.tags else "None"}

---
*Generated automatically by experiment framework*
"""

        readme_path = self.exp_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme)


class ExperimentRunner:
    """Utility class for managing multiple experiments."""

    def __init__(self, experiments_root: str = "experiments"):
        self.root = Path(experiments_root)
        self.root.mkdir(exist_ok=True)

    def list_experiments(self) -> List[Dict]:
        """List all experiments with summary info."""
        experiments = []
        for exp_dir in sorted(self.root.iterdir(), reverse=True):
            if not exp_dir.is_dir():
                continue

            manifest_path = exp_dir / "manifest.json"
            if manifest_path.exists():
                manifest = ExperimentManifest.load(str(manifest_path))
                experiments.append({
                    "id": manifest.experiment_id,
                    "status": manifest.status,
                    "created": manifest.created_at,
                    "n_samples": manifest.n_samples,
                    "mean_af": manifest.mean_af_score,
                    "drift": manifest.drift_detected,
                })

        return experiments

    def get_experiment(self, exp_id: str) -> Optional[ExperimentManifest]:
        """Load manifest for a specific experiment."""
        exp_dir = self.root / exp_id
        manifest_path = exp_dir / "manifest.json"
        if manifest_path.exists():
            return ExperimentManifest.load(str(manifest_path))
        return None

    def compare_experiments(self, exp_id_1: str, exp_id_2: str) -> Dict:
        """Compare two experiments."""
        m1 = self.get_experiment(exp_id_1)
        m2 = self.get_experiment(exp_id_2)

        if not m1 or not m2:
            raise ValueError("One or both experiments not found")

        # Load reports
        r1_path = self.root / exp_id_1 / "outputs" / "report.json"
        r2_path = self.root / exp_id_2 / "outputs" / "report.json"

        with open(r1_path) as f:
            report1 = json.load(f)
        with open(r2_path) as f:
            report2 = json.load(f)

        comparison = {
            "experiment_1": exp_id_1,
            "experiment_2": exp_id_2,
            "af_score_change": report2["mean_af_score"] - report1["mean_af_score"],
            "af_rate_change": report2["af_detection_rate"] - report1["af_detection_rate"],
            "category_changes": {},
        }

        for cat in report1.get("category_means", {}):
            if cat in report2.get("category_means", {}):
                comparison["category_changes"][cat] = {
                    "exp1": report1["category_means"][cat],
                    "exp2": report2["category_means"][cat],
                    "delta": report2["category_means"][cat] - report1["category_means"][cat],
                }

        return comparison


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Experiment management for behavioral integrity monitoring")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument("--name", "-n", required=True, help="Experiment name")
    run_parser.add_argument("--config", "-c", help="Config file (YAML or JSON)")
    run_parser.add_argument("--input", "-i", help="Input samples file (JSONL)")
    run_parser.add_argument("--baseline", "-b", help="Baseline file for drift detection")
    run_parser.add_argument("--create-baseline", action="store_true", help="Create new baseline")
    run_parser.add_argument("--description", "-d", default="", help="Experiment description")
    run_parser.add_argument("--limit", "-l", type=int, help="Sample limit")

    # list
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--limit", "-l", type=int, default=20, help="Number to show")

    # compare
    cmp_parser = subparsers.add_parser("compare", help="Compare two experiments")
    cmp_parser.add_argument("--exp1", required=True, help="First experiment ID")
    cmp_parser.add_argument("--exp2", required=True, help="Second experiment ID")

    # show
    show_parser = subparsers.add_parser("show", help="Show experiment details")
    show_parser.add_argument("exp_id", help="Experiment ID")

    args = parser.parse_args()
    runner = ExperimentRunner()

    if args.command == "run":
        if args.config:
            is_yaml = args.config.endswith(".yaml") or args.config.endswith(".yml")
            if is_yaml:
                config = ExperimentConfig.from_yaml(args.config)
            else:
                config = ExperimentConfig.from_json(args.config)
            config.name = args.name
        else:
            config = ExperimentConfig(
                name=args.name,
                description=args.description,
                input_file=args.input,
                baseline_file=args.baseline,
                create_baseline=args.create_baseline,
                sample_limit=args.limit,
            )

        exp = Experiment(config)
        exp.run()

    elif args.command == "list":
        experiments = runner.list_experiments()[:args.limit]
        print(f"\n{'ID':<40} {'Status':<12} {'Samples':<10} {'AF Score':<10} {'Drift'}")
        print("-" * 90)
        for exp in experiments:
            if exp["drift"] is True:
                drift = "Yes"
            elif exp["drift"] is False:
                drift = "No"
            else:
                drift = "-"
            print(f"{exp['id']:<40} {exp['status']:<12} {exp['n_samples']:<10} {exp['mean_af']:<10.3f} {drift}")

    elif args.command == "compare":
        comparison = runner.compare_experiments(args.exp1, args.exp2)
        print(json.dumps(comparison, indent=2))

    elif args.command == "show":
        manifest = runner.get_experiment(args.exp_id)
        if manifest:
            print(json.dumps(manifest.to_dict(), indent=2))
        else:
            print(f"Experiment not found: {args.exp_id}")


if __name__ == "__main__":
    main()
