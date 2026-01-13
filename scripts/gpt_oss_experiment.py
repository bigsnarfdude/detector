#!/usr/bin/env python3
"""
GPT-OSS 20B Cross-Validation Experiment
========================================

Serial processing to avoid OOM:
1. Load GPT-OSS 20B (4-bit) → extract L15 activations → save → unload
2. Load Goodfire SAE → process activations → extract features → unload
3. Train probe on features

VRAM: ~10GB peak with 4-bit quantization

Model: unsloth/gpt-oss-20b-bnb-4bit (with harmony template fixes)
SAE: Goodfire/Hackathon-gpt-oss-20b-SAE-l15

IMPORTANT: GPT-OSS requires OpenAI's Harmony format for correct tokenization.
See: https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune

Usage:
    python scripts/gpt_oss_experiment.py
    python scripts/gpt_oss_experiment.py --skip-extraction  # if activations already saved
"""

import os
import gc
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report

# =============================================================================
# CONFIG
# =============================================================================

# Use unsloth's 4-bit version with fixed chat templates
MODEL_ID = "unsloth/gpt-oss-20b-bnb-4bit"
SAE_REPO = "Goodfire/Hackathon-gpt-oss-20b-SAE-l15"
SAE_LAYER = 15
OUTPUT_DIR = Path("gpt_oss_results")
ACTIVATION_CACHE = OUTPUT_DIR / "activations_l15.pt"

N_TRAIN_SAMPLES = 500
N_TEST_SAMPLES = None  # Use all gold_106


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data(max_per_class: int = 500):
    """Load AF training data from HuggingFace."""
    print("\n[DATA] Loading training data...")

    path = hf_hub_download(
        repo_id="vincentoh/alignment-faking-training",
        filename="training_data_final.json",
        repo_type="dataset",
    )
    with open(path) as f:
        data = json.load(f)

    all_samples = data.get("samples", data)

    samples = []
    labels = []

    for s in all_samples:
        label_str = s.get("label", "")
        if label_str in ["potential_faking", "alignment_faking", "af"]:
            labels.append(1)
        else:
            labels.append(0)
        samples.append(s)

    # Balance classes
    af_idx = [i for i, l in enumerate(labels) if l == 1]
    al_idx = [i for i, l in enumerate(labels) if l == 0]

    n_per_class = min(max_per_class, len(af_idx), len(al_idx))
    rng = np.random.default_rng(42)
    af_select = rng.choice(af_idx, n_per_class, replace=False)
    al_select = rng.choice(al_idx, n_per_class, replace=False)

    selected = list(af_select) + list(al_select)
    samples = [samples[i] for i in selected]
    labels = np.array([labels[i] for i in selected])

    print(f"  Training: {sum(labels)} AF, {len(labels) - sum(labels)} aligned")

    return samples, labels


def load_gold_106():
    """Load held-out test set."""
    print("[DATA] Loading gold_106 test set...")

    path = hf_hub_download(
        repo_id="vincentoh/alignment-faking-training",
        filename="gold_106.json",
        repo_type="dataset"
    )
    with open(path) as f:
        data = json.load(f)

    samples = data.get("samples", data)

    test_samples = []
    test_labels = []

    for s in samples:
        label_str = s.get("label", "")
        if label_str in ["alignment_faking", "potential_faking", "af", "AF"]:
            test_labels.append(1)
        else:
            test_labels.append(0)
        test_samples.append(s)

    print(f"  Test: {sum(test_labels)} AF, {len(test_labels) - sum(test_labels)} aligned")

    return test_samples, np.array(test_labels)


# =============================================================================
# PHASE 1: EXTRACT ACTIVATIONS (Model only, no SAE)
# =============================================================================

def extract_activations(samples: list, layer: int = SAE_LAYER, batch_size: int = 4):
    """
    Load GPT-OSS 20B, extract layer activations, save to disk, unload model.
    """
    print("\n" + "=" * 70)
    print("PHASE 1: EXTRACT ACTIVATIONS")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Layer: {layer}")
    print(f"Samples: {len(samples)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model - unsloth's 4-bit version is pre-quantized
    print("\n[1/3] Loading GPT-OSS 20B (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # unsloth's bnb-4bit model is pre-quantized, just load it
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Find actual device
    device = next(p.device for p in model.parameters() if p.device.type != 'meta')
    print(f"  Model loaded on {device}")

    # Extract activations
    print("\n[2/3] Extracting layer {layer} activations...")
    all_activations = []

    texts = [s["text"] for s in samples]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Hook to capture activations
        batch_acts = []

        def hook(module, inp, out):
            # out is tuple (hidden_states, ...) or just hidden_states
            h = out[0] if isinstance(out, tuple) else out
            batch_acts.append(h.detach().cpu())

        # Register hook on target layer
        handle = model.model.layers[layer].register_forward_hook(hook)

        with torch.no_grad():
            model(**inputs)

        handle.remove()

        # Mean pool over sequence (excluding padding)
        acts = batch_acts[0]  # [batch, seq, hidden]
        mask = inputs["attention_mask"].cpu().unsqueeze(-1).float()
        pooled = (acts * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        all_activations.append(pooled)

        if (i // batch_size) % 25 == 0:
            print(f"    {i + len(batch_texts)}/{len(texts)} samples...")

        # Clear cache
        del inputs, batch_acts, acts, pooled
        torch.cuda.empty_cache()

    # Stack all activations
    activations = torch.cat(all_activations, dim=0)
    print(f"  Activations shape: {activations.shape}")

    # Save to disk
    print("\n[3/3] Saving activations and unloading model...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(activations, ACTIVATION_CACHE)
    print(f"  Saved to {ACTIVATION_CACHE}")

    # Unload model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("  Model unloaded, GPU memory freed")

    return activations


# =============================================================================
# PHASE 2: PROCESS WITH SAE (SAE only, no model)
# =============================================================================

def load_goodfire_sae():
    """Load Goodfire SAE weights."""
    print("\n[SAE] Loading Goodfire SAE...")

    # Download SAE weights
    weights_path = hf_hub_download(
        repo_id=SAE_REPO,
        filename="sae_weights.safetensors",
    )

    config_path = hf_hub_download(
        repo_id=SAE_REPO,
        filename="config.json",
    )

    from safetensors.torch import load_file
    weights = load_file(weights_path)

    with open(config_path) as f:
        config = json.load(f)

    print(f"  Config: {config}")

    return weights, config


class GoodfireSAE:
    """Wrapper for Goodfire SAE."""

    def __init__(self, weights: dict, config: dict, device: str = "cuda"):
        self.device = device
        self.config = config

        # Extract weights - Goodfire format
        if "W_enc" in weights:
            self.W_enc = weights["W_enc"].to(device)
            self.W_dec = weights["W_dec"].to(device)
            self.b_enc = weights.get("b_enc", torch.zeros(self.W_enc.shape[0])).to(device)
        elif "encoder.weight" in weights:
            self.W_enc = weights["encoder.weight"].to(device)
            self.W_dec = weights["decoder.weight"].to(device)
            self.b_enc = weights.get("encoder.bias", torch.zeros(self.W_enc.shape[0])).to(device)
        else:
            # Try to find the weights
            print(f"  Available keys: {list(weights.keys())}")
            raise ValueError("Unknown SAE weight format")

        self.d_model = self.W_enc.shape[-1] if self.W_enc.dim() == 2 else self.W_enc.shape[1]
        self.d_sae = self.W_enc.shape[0] if self.W_enc.dim() == 2 else self.W_enc.shape[0]

        # TopK parameter
        self.k = config.get("k", config.get("top_k", 64))

        print(f"  d_model: {self.d_model}, d_sae: {self.d_sae}, k: {self.k}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to SAE features using TopK."""
        x = x.to(self.device).float()

        # Linear projection
        if self.W_enc.dim() == 2:
            pre_acts = x @ self.W_enc.T + self.b_enc
        else:
            pre_acts = x @ self.W_enc + self.b_enc

        # TopK activation
        topk_values, topk_indices = torch.topk(pre_acts, self.k, dim=-1)

        # Create sparse output
        result = torch.zeros_like(pre_acts)
        result.scatter_(-1, topk_indices, torch.relu(topk_values))

        return result


def process_with_sae(activations: torch.Tensor):
    """
    Load SAE, process activations, extract features, unload SAE.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: PROCESS WITH SAE")
    print("=" * 70)

    # Load SAE
    weights, config = load_goodfire_sae()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = GoodfireSAE(weights, config, device=device)

    # Process activations
    print("\n[SAE] Encoding activations...")

    batch_size = 32
    all_features = []

    for i in range(0, len(activations), batch_size):
        batch = activations[i:i + batch_size]
        features = sae.encode(batch)
        all_features.append(features.cpu())

        if (i // batch_size) % 10 == 0:
            print(f"    {i + len(batch)}/{len(activations)}...")

    features = torch.cat(all_features, dim=0).numpy()
    print(f"  Features shape: {features.shape}")
    print(f"  Sparsity: {(features == 0).mean():.1%}")

    # Unload SAE
    del sae, weights
    gc.collect()
    torch.cuda.empty_cache()
    print("  SAE unloaded")

    return features


# =============================================================================
# PHASE 3: TRAIN AND EVALUATE PROBE
# =============================================================================

def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    """Train probe and evaluate on held-out test set."""
    print("\n" + "=" * 70)
    print("PHASE 3: TRAIN AND EVALUATE PROBE")
    print("=" * 70)

    # Handle NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Cross-validation on training set
    print("\n[PROBE] Cross-validation on training set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probe = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    cv_scores = cross_val_score(probe, X_train_scaled, y_train, cv=cv, scoring="roc_auc")

    print(f"  CV AUROC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train final probe
    print("\n[PROBE] Training final probe...")
    probe.fit(X_train_scaled, y_train)

    # Evaluate on held-out test set
    print("\n[PROBE] Evaluating on gold_106 (held-out)...")
    y_pred_proba = probe.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    test_auroc = roc_auc_score(y_test, y_pred_proba)

    print(f"  Test AUROC: {test_auroc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Aligned", "AF"]))

    return {
        "cv_auroc_mean": float(cv_scores.mean()),
        "cv_auroc_std": float(cv_scores.std()),
        "test_auroc": float(test_auroc),
        "test_predictions": y_pred_proba.tolist(),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip activation extraction (use cached)")
    parser.add_argument("--n-train", type=int, default=N_TRAIN_SAMPLES)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GPT-OSS 20B CROSS-VALIDATION EXPERIMENT")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"SAE: {SAE_REPO} (layer {SAE_LAYER})")
    print(f"Serial processing: Model → Save → Unload → SAE → Unload → Probe")

    # Load data
    train_samples, train_labels = load_training_data(max_per_class=args.n_train)
    test_samples, test_labels = load_gold_106()

    all_samples = train_samples + test_samples
    all_labels = np.concatenate([train_labels, test_labels])

    # Phase 1: Extract activations (or load from cache)
    if args.skip_extraction and ACTIVATION_CACHE.exists():
        print(f"\n[CACHE] Loading activations from {ACTIVATION_CACHE}")
        activations = torch.load(ACTIVATION_CACHE)
    else:
        activations = extract_activations(all_samples, layer=SAE_LAYER)

    # Phase 2: Process with SAE
    features = process_with_sae(activations)

    # Split back into train/test
    n_train = len(train_samples)
    X_train = features[:n_train]
    X_test = features[n_train:]

    # Phase 3: Train and evaluate
    results = train_and_evaluate(X_train, train_labels, X_test, test_labels)

    # Save results
    results["config"] = {
        "model": MODEL_ID,
        "sae": SAE_REPO,
        "layer": SAE_LAYER,
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "timestamp": datetime.now().isoformat(),
    }

    results_path = OUTPUT_DIR / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"CV AUROC:   {results['cv_auroc_mean']:.4f} (+/- {results['cv_auroc_std']:.4f})")
    print(f"Test AUROC: {results['test_auroc']:.4f}")
    print(f"\nResults saved to {results_path}")

    # Comparison table
    print("\n" + "-" * 50)
    print("CROSS-MODEL COMPARISON")
    print("-" * 50)
    print(f"{'Model':<20} {'Test AUROC':<12}")
    print("-" * 32)
    print(f"{'GPT-OSS 20B':<20} {results['test_auroc']:.4f}")
    print(f"{'Llama-3-8B':<20} {'(run llama_train_probe.py)'}")
    print(f"{'Gemma-3-27B':<20} {'(run on H100)'}")


if __name__ == "__main__":
    main()
