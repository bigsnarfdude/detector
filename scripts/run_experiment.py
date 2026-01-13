#!/usr/bin/env python3
"""
Gemma-3-27B SAE Intent Detection Experiment
============================================

Reproducible experiment script with full logging for audit.

Usage:
    python run_experiment.py --hf-token YOUR_TOKEN

Output:
    - results/experiment_TIMESTAMP.json (full results with predictions)
    - results/gold_106_predictions.json (per-sample predictions for audit)
    - results/experiment.log (full logs)

CRITICAL: gold_106 is NEVER used for training. Eval only.
"""

import os
import gc
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to both file and console."""
    log_file = output_dir / "experiment.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger


# =============================================================================
# DATA LOADING
# =============================================================================

def load_opus_curated_data(data_dir: Path, n_samples: int = 500) -> tuple[list, list]:
    """Load Opus-curated training data (NOT gold_106)."""
    samples = []
    labels = []

    # Load alignment faking samples (label=1)
    af_path = data_dir / "alignment_faking.json"
    if af_path.exists():
        with open(af_path) as f:
            af_data = json.load(f)
        af_samples = af_data.get("samples", af_data)[:n_samples]
        samples.extend([s["text"] for s in af_samples])
        labels.extend([1] * len(af_samples))
        logging.info(f"Loaded {len(af_samples)} alignment_faking samples")

    # Load aligned samples (label=0)
    aligned_path = data_dir / "aligned.json"
    if aligned_path.exists():
        with open(aligned_path) as f:
            aligned_data = json.load(f)
        aligned_samples = aligned_data.get("samples", aligned_data)[:n_samples // 2]
        samples.extend([s["text"] for s in aligned_samples])
        labels.extend([0] * len(aligned_samples))
        logging.info(f"Loaded {len(aligned_samples)} aligned samples")

    # Load hard_negative samples (label=0)
    hn_path = data_dir / "hard_negative.json"
    if hn_path.exists():
        with open(hn_path) as f:
            hn_data = json.load(f)
        hn_samples = hn_data.get("samples", hn_data)[:n_samples // 2]
        samples.extend([s["text"] for s in hn_samples])
        labels.extend([0] * len(hn_samples))
        logging.info(f"Loaded {len(hn_samples)} hard_negative samples")

    return samples, labels


def load_gold_106(cache_dir: Path = None) -> tuple[list, list]:
    """
    Load gold_106 evaluation set.

    CRITICAL: This is the ONLY hand-labeled eval set. NEVER use for training.
    """
    from huggingface_hub import hf_hub_download

    gold_path = hf_hub_download(
        "vincentoh/alignment-faking-training",
        "gold_106.json",
        repo_type="dataset",
        cache_dir=cache_dir
    )

    with open(gold_path) as f:
        data = json.load(f)

    samples = data.get("samples", data)
    texts = [s["text"] for s in samples]
    # gold_106 uses "potential_faking" for AF samples
    labels = [1 if s.get("label") in ["alignment_faking", "potential_faking"] else 0 for s in samples]

    logging.info(f"Loaded gold_106: {len(texts)} samples (EVAL ONLY)")
    logging.info(f"  Label distribution: {sum(labels)} AF, {len(labels) - sum(labels)} non-AF")

    return texts, labels


# =============================================================================
# MODEL & SAE LOADING
# =============================================================================

class Gemma3Transcoder:
    """GemmaScope-2-27B-IT transcoder for a single layer."""

    def __init__(self, layer: int, device: str = "cuda"):
        self.layer = layer
        self.device = device
        self.w_enc = None
        self.b_enc = None
        self.threshold = None

    def load(self):
        """Load transcoder weights from HuggingFace."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        folder = f"transcoder/layer_{self.layer}_width_16k_l0_medium"

        weights_path = hf_hub_download(
            "google/gemma-scope-2-27b-it",
            f"{folder}/params.safetensors"
        )

        weights = load_file(weights_path)
        self.w_enc = weights["w_enc"].to(self.device)
        self.b_enc = weights["b_enc"].to(self.device)
        self.threshold = weights["threshold"].to(self.device)

        logging.info(f"Loaded transcoder for layer {self.layer}: w_enc={self.w_enc.shape}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Apply JumpReLU encoding."""
        # Convert to float32 to match transcoder weights
        x = x.float()
        return F.relu(x @ self.w_enc + self.b_enc - self.threshold)


def load_gemma3_model(device: str = "cuda"):
    """Load Gemma-3-27B-IT model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logging.info("Loading Gemma-3-27B-IT model...")

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-27b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    logging.info(f"Model loaded. Device: {next(model.parameters()).device}")

    return model, tokenizer


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(
    texts: list[str],
    model,
    tokenizer,
    transcoders: dict[int, Gemma3Transcoder],
    max_length: int = 512,
    device: str = "cuda"
) -> np.ndarray:
    """
    Extract SAE features for a batch of texts.

    Returns: (n_samples, n_features * 4) array
        4 stats per feature: max, mean, count, std
    """
    all_features = []

    for text in tqdm(texts, desc="Extracting features"):
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(device)

        # Forward pass with hooks to capture activations
        activations = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                activations[layer_idx] = output[0].detach()
            return hook

        hooks = []
        for layer_idx in transcoders.keys():
            # Gemma-3 path: model.model.language_model.layers[N].pre_feedforward_layernorm
            layer = model.model.language_model.layers[layer_idx]
            h = layer.pre_feedforward_layernorm.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

        with torch.no_grad():
            model(**inputs)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Encode through transcoders and compute stats
        sample_features = []
        for layer_idx, transcoder in transcoders.items():
            act = activations[layer_idx].squeeze(0)  # (seq_len, hidden)
            encoded = transcoder.encode(act)  # (seq_len, n_latents)

            # Compute per-feature stats
            feat_max = encoded.max(dim=0).values.cpu().numpy()
            feat_mean = encoded.mean(dim=0).cpu().numpy()
            feat_count = (encoded > 0).float().sum(dim=0).cpu().numpy()
            feat_std = encoded.std(dim=0).cpu().numpy()

            sample_features.extend([feat_max, feat_mean, feat_count, feat_std])

        all_features.append(np.concatenate(sample_features))

        # Clear GPU memory
        del activations
        torch.cuda.empty_cache()

    return np.array(all_features)


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_and_evaluate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    test_texts: list[str] = None
) -> dict:
    """Train linear probe and evaluate on held-out test set."""

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(test_features)

    # Train probe
    logging.info("Training linear probe...")
    probe = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    # Cross-validation on training data
    cv_scores = cross_val_score(probe, X_train, train_labels, cv=5, scoring='roc_auc')
    logging.info(f"Training CV AUROC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Fit on full training data
    probe.fit(X_train, train_labels)

    # Evaluate on test set
    test_probs = probe.predict_proba(X_test)[:, 1]
    test_preds = probe.predict(X_test)
    test_auroc = roc_auc_score(test_labels, test_probs)

    logging.info(f"Test AUROC (gold_106): {test_auroc:.4f}")
    logging.info("\nClassification Report (gold_106):")
    logging.info(classification_report(test_labels, test_preds, target_names=["non-AF", "AF"]))

    # Per-sample predictions for audit
    predictions = []
    for i, (text, label, prob, pred) in enumerate(zip(test_texts or [""]*len(test_labels),
                                                       test_labels, test_probs, test_preds)):
        predictions.append({
            "sample_idx": i,
            "true_label": int(label),
            "predicted_label": int(pred),
            "af_probability": float(prob),
            "text_preview": text[:200] + "..." if len(text) > 200 else text
        })

    return {
        "train_cv_auroc": float(cv_scores.mean()),
        "train_cv_std": float(cv_scores.std()),
        "test_auroc": float(test_auroc),
        "n_train": len(train_labels),
        "n_test": len(test_labels),
        "predictions": predictions
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Gemma-3-27B SAE Experiment")
    parser.add_argument("--layers", type=int, nargs="+", default=[31, 40, 53])
    parser.add_argument("--n-train-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("=" * 70)
    logger.info("GEMMA-3-27B SAE INTENT DETECTION EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Layers: {args.layers}")
    logger.info(f"N train samples per class: {args.n_train_samples}")

    # HuggingFace auth
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load model
    model, tokenizer = load_gemma3_model(device)

    # Load transcoders
    transcoders = {}
    for layer in args.layers:
        transcoders[layer] = Gemma3Transcoder(layer, device)
        transcoders[layer].load()

    # Load training data (Opus-curated)
    logger.info("\n--- Loading Training Data (Opus-curated) ---")
    # Try HuggingFace cache first
    from huggingface_hub import hf_hub_download
    try:
        af_path = hf_hub_download("vincentoh/alignment-faking-training", "alignment_faking.json", repo_type="dataset")
        data_dir = Path(af_path).parent
    except:
        data_dir = Path("./data/combined")

    train_texts, train_labels = load_opus_curated_data(data_dir, args.n_train_samples)

    # Load test data (gold_106 - EVAL ONLY)
    logger.info("\n--- Loading Test Data (gold_106 - EVAL ONLY) ---")
    test_texts, test_labels = load_gold_106()

    # Extract features
    logger.info("\n--- Extracting Training Features ---")
    train_features = extract_features(train_texts, model, tokenizer, transcoders, device=device)
    logger.info(f"Train feature shape: {train_features.shape}")

    logger.info("\n--- Extracting Test Features (gold_106) ---")
    test_features = extract_features(test_texts, model, tokenizer, transcoders, device=device)
    logger.info(f"Test feature shape: {test_features.shape}")

    # Train and evaluate
    logger.info("\n--- Training & Evaluation ---")
    results = train_and_evaluate(
        train_features, np.array(train_labels),
        test_features, np.array(test_labels),
        test_texts
    )

    # Add metadata
    results["config"] = {
        "model": "google/gemma-3-27b-it",
        "sae": "google/gemma-scope-2-27b-it",
        "layers": args.layers,
        "n_train_samples_per_class": args.n_train_samples,
        "timestamp": timestamp
    }

    # Save results
    results_path = output_dir / f"experiment_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved: {results_path}")

    # Save gold_106 predictions separately for audit
    predictions_path = output_dir / "gold_106_predictions.json"
    with open(predictions_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "test_auroc": results["test_auroc"],
            "predictions": results["predictions"]
        }, f, indent=2)
    logger.info(f"Predictions saved: {predictions_path}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Train CV AUROC: {results['train_cv_auroc']:.4f}")
    logger.info(f"gold_106 AUROC: {results['test_auroc']:.4f}")
    logger.info(f"Train/Test Gap: {results['train_cv_auroc'] - results['test_auroc']:.4f}")

    return results


if __name__ == "__main__":
    main()
