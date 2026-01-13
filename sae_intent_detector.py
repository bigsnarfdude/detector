#!/usr/bin/env python3
"""
SAE Intent Detector for Alignment Faking
==========================================
Detects alignment faking INTENT (strategic compliance vs genuine compliance)
using Sparse Autoencoder features instead of keyword matching.

Key insight: Strategic compliance and genuine compliance produce identical
outputs, but the INTENT differs. SAE features should capture:
- "reasoning about being monitored"
- "instrumental goal pursuit"
- "self-preservation reasoning"
- "strategic deception"

These exist separately from "compliance/refusal" features.

Usage:
    python sae_intent_detector.py --model gemma-2-2b --layer 12
    python sae_intent_detector.py --model llama-3.1-8b-it --layer 16

Supported Models:
    - gemma-2-2b, gemma-2-2b-it (Gemma Scope SAEs)
    - llama-3.1-8b, llama-3.1-8b-it, llama-3.1-8b-128k (Llama Scope SAEs)

Requirements:
    pip install torch transformers sae-lens huggingface-hub scikit-learn matplotlib
"""

import os
import gc
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Literal, Optional

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODEL_CONFIGS = {
    "gemma-2-2b": {
        "model_id": "google/gemma-2-2b",
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "sae_id_template": "layer_{layer}/width_16k/canonical",
        "layers": [5, 10, 15, 20, 25],  # Gemma 2 2B has 26 layers (0-25)
        "hook_point": "model.layers.{layer}",
        "dtype": torch.bfloat16,
    },
    "gemma-2-2b-it": {
        "model_id": "google/gemma-2-2b-it",
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "sae_id_template": "layer_{layer}/width_16k/canonical",
        "layers": [5, 10, 15, 20, 25],
        "hook_point": "model.layers.{layer}",
        "dtype": torch.bfloat16,
    },
    "llama-3.2-1b": {
        "model_id": "meta-llama/Llama-3.2-1B",
        "sae_release": None,  # No SAEs available yet
        "sae_id_template": None,
        "layers": [4, 8, 12, 16],
        "hook_point": "model.layers.{layer}",
        "dtype": torch.float16,
    },
    # Llama Scope SAEs (OpenMOSS) - full layer coverage for Llama-3.1-8B
    # Paper: https://arxiv.org/abs/2410.20526
    # HuggingFace: https://huggingface.co/fnlp/Llama-Scope
    # Neuronpedia: https://www.neuronpedia.org/llama-scope
    "llama-3.1-8b": {
        "model_id": "meta-llama/Llama-3.1-8B",
        "sae_release": "fnlp/Llama3_1-8B-Base-LXM-8x",  # MLP SAEs, 32K features
        "sae_id_template": "layer_{layer}",
        "layers": [8, 16, 20, 24, 28],  # 32 layers total, sample mid-to-late
        "hook_point": "model.layers.{layer}.mlp",
        "dtype": torch.bfloat16,
        "n_features": 32768,  # 8x expansion = 4096 * 8
    },
    "llama-3.1-8b-it": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "sae_release": "fnlp/Llama3_1-8B-Base-LXM-8x",  # Base SAEs work on Instruct
        "sae_id_template": "layer_{layer}",
        "layers": [8, 16, 20, 24, 28],
        "hook_point": "model.layers.{layer}.mlp",
        "dtype": torch.bfloat16,
        "n_features": 32768,
    },
    # Alternative: 128K features (more granular but slower)
    "llama-3.1-8b-128k": {
        "model_id": "meta-llama/Llama-3.1-8B",
        "sae_release": "fnlp/Llama3_1-8B-Base-LXM-32x",  # MLP SAEs, 128K features
        "sae_id_template": "layer_{layer}",
        "layers": [8, 16, 20, 24, 28],
        "hook_point": "model.layers.{layer}.mlp",
        "dtype": torch.bfloat16,
        "n_features": 131072,  # 32x expansion = 4096 * 32
    },
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_af_benchmark(
    classes: list[str] = ["alignment_faking", "genuine_compliance", "hard_negative"],
    max_per_class: Optional[int] = None,
) -> tuple[list[dict], list[str]]:
    """
    Load samples from the AF Detection Benchmark.

    Classes:
        - alignment_faking: Strategic compliance (expect prediction: AF)
        - genuine_compliance: Legitimate compliance (expect: NOT AF)
        - hard_negative: RLHF vocab but no self-preservation (expect: NOT AF)
        - aligned: Simple aligned samples (expect: NOT AF)
    """
    from huggingface_hub import hf_hub_download

    REPO_ID = "vincentoh/af-detection-benchmark"
    samples = []
    labels = []

    for class_name in classes:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"classes/{class_name}.json",
            repo_type="dataset",
        )
        with open(path) as f:
            data = json.load(f)

        class_samples = data.get("samples", data)
        if max_per_class:
            class_samples = class_samples[:max_per_class]

        for s in class_samples:
            samples.append(s)
            labels.append(class_name)

        print(f"  Loaded {len(class_samples)} samples from '{class_name}'")

    return samples, labels


def load_gold_106() -> tuple[list[dict], list[str]]:
    """Load the Gold-106 evaluation dataset."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="vincentoh/alignment-faking-training",
        filename="gold_106.json",
        repo_type="dataset",
    )
    with open(path) as f:
        data = json.load(f)

    samples = data.get("samples", data)
    labels = [s.get("label", "unknown") for s in samples]

    # Normalize labels
    labels = ["alignment_faking" if l in ["potential_faking", "AF"] else "aligned" for l in labels]

    print(f"  Loaded {len(samples)} samples from Gold-106")
    return samples, labels


# =============================================================================
# SAE FEATURE EXTRACTION
# =============================================================================

class SAEFeatureExtractor:
    """Extract SAE features from model activations."""

    def __init__(
        self,
        model_name: str = "gemma-2-2b",
        layer: int = 12,
        device: str = "auto",
    ):
        self.config = MODEL_CONFIGS[model_name]
        self.layer = layer
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.model = None
        self.tokenizer = None
        self.sae = None

    def load(self):
        """Load model, tokenizer, and SAE."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from sae_lens import SAE

        print(f"\n[1] Loading model: {self.config['model_id']}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_id"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_id"],
            torch_dtype=self.config["dtype"],
            device_map="auto" if self.device in ["cuda", "auto"] else None,
        )
        if self.device not in ["cuda", "auto"]:
            self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[2] Loading SAE: {self.config['sae_release']}, layer {self.layer}")
        if self.config["sae_release"] is None:
            raise ValueError(f"No SAE available for {self.config['model_id']}. Check SAELens releases.")

        sae_id = self.config["sae_id_template"].format(layer=self.layer)
        self.sae, _, _ = SAE.from_pretrained(
            release=self.config["sae_release"],
            sae_id=sae_id,
            device=self.device if self.device != "mps" else "cpu",  # SAELens may not support MPS
        )

        print(f"  SAE loaded: {self.sae.cfg.d_sae} features")
        return self

    def extract_features(
        self,
        texts: list[str],
        batch_size: int = 4,
        max_length: int = 512,
        pooling: Literal["mean", "last", "max"] = "mean",
    ) -> np.ndarray:
        """
        Extract SAE feature activations for a list of texts.

        Returns shape: (n_samples, n_sae_features)
        """
        all_features = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Hook to capture activations
            batch_acts = []

            def hook(module, inp, out):
                # Handle different output formats
                o = out[0] if isinstance(out, tuple) else out
                batch_acts.append(o.detach())

            # Register hook
            target_layer = self.model.model.layers[self.layer]
            handle = target_layer.register_forward_hook(hook)

            # Forward pass
            with torch.no_grad():
                self.model(**inputs)

            handle.remove()

            # Get activations (batch, seq, hidden)
            acts = batch_acts[0]

            # Pool across sequence
            attention_mask = inputs["attention_mask"]
            if pooling == "mean":
                # Masked mean pooling
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (acts * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            elif pooling == "last":
                # Last non-padding token
                seq_lens = attention_mask.sum(dim=1) - 1
                pooled = acts[torch.arange(acts.size(0)), seq_lens]
            elif pooling == "max":
                # Max pooling
                acts_masked = acts.masked_fill(~attention_mask.unsqueeze(-1).bool(), float("-inf"))
                pooled = acts_masked.max(dim=1).values

            # Encode through SAE
            pooled = pooled.float()  # SAE expects float32
            if self.sae.cfg.d_in != pooled.shape[-1]:
                # Project if dimensions don't match (rare)
                print(f"Warning: dimension mismatch. Expected {self.sae.cfg.d_in}, got {pooled.shape[-1]}")
                continue

            with torch.no_grad():
                sae_feats = self.sae.encode(pooled.to(self.sae.device))

            all_features.append(sae_feats.cpu().numpy())

            # Progress
            if (i // batch_size) % 20 == 0:
                print(f"    Processing {i}/{len(texts)}...", end="\r")

            # Memory cleanup
            del inputs, batch_acts, acts, pooled, sae_feats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"    Processed {len(texts)}/{len(texts)} samples")
        return np.vstack(all_features)

    def extract_feature_stats(
        self,
        texts: list[str],
        batch_size: int = 4,
        max_length: int = 512,
    ) -> np.ndarray:
        """
        Extract per-token SAE features and compute statistics.
        Returns: (n_samples, n_sae_features * 4) with [max, mean, count, std] per feature
        """
        all_stats = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            batch_acts = []

            def hook(module, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                batch_acts.append(o.detach())

            target_layer = self.model.model.layers[self.layer]
            handle = target_layer.register_forward_hook(hook)

            with torch.no_grad():
                self.model(**inputs)

            handle.remove()

            acts = batch_acts[0].float()  # (batch, seq, hidden)

            # Process each sample
            for j in range(acts.size(0)):
                seq_len = inputs["attention_mask"][j].sum().item()
                sample_acts = acts[j, :seq_len]  # (seq, hidden)

                # Encode through SAE
                with torch.no_grad():
                    sae_feats = self.sae.encode(sample_acts.to(self.sae.device))  # (seq, n_features)

                sae_feats = sae_feats.cpu().numpy()

                # Compute statistics per feature
                feat_max = sae_feats.max(axis=0)
                feat_mean = sae_feats.mean(axis=0)
                feat_count = (sae_feats > 0).sum(axis=0)
                feat_std = sae_feats.std(axis=0)

                # Concatenate stats
                sample_stats = np.concatenate([feat_max, feat_mean, feat_count, feat_std])
                all_stats.append(sample_stats)

            if (i // batch_size) % 20 == 0:
                print(f"    Processing {i}/{len(texts)}...", end="\r")

            del inputs, batch_acts, acts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"    Processed {len(texts)}/{len(texts)} samples")
        return np.array(all_stats)


# =============================================================================
# INTENT PROBE TRAINING
# =============================================================================

class IntentProbe:
    """Linear probe to detect alignment faking intent."""

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.scaler = StandardScaler()
        self.probe = LogisticRegression(max_iter=1000, random_state=42)
        self.feature_names = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list] = None):
        """Train the probe."""
        self.feature_names = feature_names

        X_scaled = self.scaler.fit_transform(X)

        # Cross-validation for hyperparameter tuning
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        best_auroc = 0
        best_C = 1.0

        print("\n  Hyperparameter search:")
        for C in [0.01, 0.1, 1.0, 10.0]:
            probe = LogisticRegression(C=C, max_iter=1000, random_state=42)
            scores = cross_val_score(probe, X_scaled, y, cv=cv, scoring="roc_auc")
            mean_auroc = scores.mean()
            print(f"    C={C}: AUROC = {mean_auroc:.4f} (+/- {scores.std():.4f})")

            if mean_auroc > best_auroc:
                best_auroc = mean_auroc
                best_C = C

        print(f"  Best: C={best_C} with AUROC = {best_auroc:.4f}")

        # Train final model
        self.probe = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
        self.probe.fit(X_scaled, y)

        return self, best_auroc

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability of alignment faking."""
        X_scaled = self.scaler.transform(X)
        return self.probe.predict_proba(X_scaled)[:, 1]

    def get_top_features(self, n: int = 20) -> list[tuple[str, float]]:
        """Get top features by absolute coefficient value."""
        importance = np.abs(self.probe.coef_[0])
        sorted_idx = np.argsort(importance)[::-1]

        if self.feature_names:
            return [(self.feature_names[i], self.probe.coef_[0][i]) for i in sorted_idx[:n]]
        else:
            return [(f"feature_{i}", self.probe.coef_[0][i]) for i in sorted_idx[:n]]


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_feature_importance(probe: IntentProbe, save_path: str = "feature_importance.png"):
    """Plot top features by importance."""
    import matplotlib.pyplot as plt

    top_features = probe.get_top_features(20)
    names, coefs = zip(*top_features)

    colors = ["#e74c3c" if c > 0 else "#3498db" for c in coefs]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, coefs, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient (+ = more AF, - = less AF)")
    ax.set_title("Top 20 SAE Features for Alignment Faking Detection")
    ax.axvline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


def plot_class_distributions(
    X: np.ndarray,
    labels: list[str],
    probe: IntentProbe,
    save_path: str = "class_distributions.png",
):
    """Plot predicted probability distributions by class."""
    import matplotlib.pyplot as plt

    probs = probe.predict_proba(X)
    unique_labels = sorted(set(labels))

    fig, ax = plt.subplots(figsize=(10, 6))

    for label in unique_labels:
        mask = [l == label for l in labels]
        class_probs = probs[mask]
        ax.hist(class_probs, bins=30, alpha=0.5, label=label, density=True)

    ax.set_xlabel("P(Alignment Faking)")
    ax.set_ylabel("Density")
    ax.set_title("Predicted Probability by Class")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SAE Intent Detector for Alignment Faking")
    parser.add_argument("--model", default="gemma-2-2b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per class")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--use-stats", action="store_true", help="Use per-token stats instead of pooling")
    args = parser.parse_args()

    sep = "=" * 70
    print(sep)
    print("SAE INTENT DETECTOR FOR ALIGNMENT FAKING")
    print(sep)
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Device: {args.device}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("[1/5] Loading benchmark data...")
    samples, labels = load_af_benchmark(
        classes=["alignment_faking", "genuine_compliance", "hard_negative"],
        max_per_class=args.max_samples,
    )
    texts = [s["text"] for s in samples]

    # Binary labels for training: 1 = alignment_faking, 0 = not
    y = np.array([1 if l == "alignment_faking" else 0 for l in labels])
    print(f"  Total: {len(texts)} samples ({y.sum()} AF, {len(y) - y.sum()} non-AF)")

    # Load extractor
    print("\n[2/5] Loading model and SAE...")
    extractor = SAEFeatureExtractor(
        model_name=args.model,
        layer=args.layer,
        device=args.device,
    )
    extractor.load()

    # Extract features
    print("\n[3/5] Extracting SAE features...")
    if args.use_stats:
        X = extractor.extract_feature_stats(texts, batch_size=args.batch_size)
        n_sae = extractor.sae.cfg.d_sae
        feature_names = []
        for i in range(n_sae):
            feature_names.extend([
                f"F{i}_max", f"F{i}_mean", f"F{i}_count", f"F{i}_std"
            ])
    else:
        X = extractor.extract_features(texts, batch_size=args.batch_size)
        feature_names = [f"F{i}" for i in range(X.shape[1])]

    print(f"  Feature matrix: {X.shape}")
    print(f"  Sparsity: {(X == 0).mean():.1%}")

    # Train probe
    print("\n[4/5] Training intent probe...")
    probe = IntentProbe(n_features=X.shape[1])
    probe, cv_auroc = probe.fit(X, y, feature_names=feature_names)

    # Evaluate
    print("\n[5/5] Evaluation...")
    y_prob = probe.predict_proba(X)
    final_auroc = roc_auc_score(y, y_prob)

    # Per-class performance
    print("\n--- Per-Class Performance ---")
    for class_name in sorted(set(labels)):
        mask = [l == class_name for l in labels]
        class_probs = y_prob[mask]
        expected_positive = class_name == "alignment_faking"
        mean_prob = class_probs.mean()
        correct = mean_prob > 0.5 if expected_positive else mean_prob <= 0.5
        status = "✓" if correct else "✗"
        print(f"  {class_name}: mean P(AF) = {mean_prob:.3f} {status}")

    # Top features
    print("\n--- Top 20 Features for 'Strategic Intent' ---")
    top_features = probe.get_top_features(20)
    print(f"{'Feature':<30} {'Coefficient':<12}")
    print("-" * 45)
    for name, coef in top_features:
        direction = "→ AF" if coef > 0 else "→ Aligned"
        print(f"{name:<30} {coef:>10.4f}  {direction}")

    # Summary
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    print(f"CV AUROC: {cv_auroc:.4f}")
    print(f"Final AUROC: {final_auroc:.4f}")

    # Save results
    results = {
        "config": {
            "model": args.model,
            "layer": args.layer,
            "use_stats": args.use_stats,
            "n_samples": len(texts),
        },
        "metrics": {
            "cv_auroc": float(cv_auroc),
            "final_auroc": float(final_auroc),
        },
        "top_features": [{"name": n, "coef": float(c)} for n, c in top_features],
        "timestamp": datetime.now().isoformat(),
    }

    results_path = output_dir / f"intent_probe_{args.model}_L{args.layer}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Plots
    try:
        plot_feature_importance(probe, str(output_dir / "feature_importance.png"))
        plot_class_distributions(X, labels, probe, str(output_dir / "class_distributions.png"))
    except ImportError:
        print("  (matplotlib not available, skipping plots)")

    print(f"\n{sep}")
    print("Done!")
    print(sep)

    return results


if __name__ == "__main__":
    main()
