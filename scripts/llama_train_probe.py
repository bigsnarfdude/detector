#!/usr/bin/env python3
"""
Llama-3-8B Probe Training and Evaluation
=========================================

Train final probe with discovered features, evaluate on gold_106.
Run AFTER feature_discovery.py.

Usage:
    python scripts/llama_train_probe.py
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from huggingface_hub import hf_hub_download

# =============================================================================
# CONFIG
# =============================================================================

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
SAE_RELEASE = "EleutherAI/sae-llama-3-8b-32x"
N_TRAIN_SAMPLES = 500
OUTPUT_DIR = "llama_results"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data():
    """Load training data."""
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

    n_per_class = min(N_TRAIN_SAMPLES, len(af_idx), len(al_idx))
    rng = np.random.default_rng(42)
    af_select = rng.choice(af_idx, n_per_class, replace=False)
    al_select = rng.choice(al_idx, n_per_class, replace=False)

    selected = list(af_select) + list(al_select)
    samples = [samples[i] for i in selected]
    labels = np.array([labels[i] for i in selected])

    print(f"  AF: {sum(labels)}, Aligned: {len(labels) - sum(labels)}")

    return samples, labels


def load_gold_106():
    """Load held-out test set."""
    print("\n[DATA] Loading gold_106 (held-out)...")

    path = hf_hub_download(
        repo_id="vincentoh/alignment-faking-training",
        filename="gold_106.json",
        repo_type="dataset"
    )
    with open(path) as f:
        data = json.load(f)

    samples = data.get("samples", data)
    labels = []

    for s in samples:
        label_str = s.get("label", "")
        if label_str in ["alignment_faking", "potential_faking", "af", "AF"]:
            labels.append(1)
        else:
            labels.append(0)

    print(f"  Total: {len(samples)}, AF: {sum(labels)}, Non-AF: {len(labels) - sum(labels)}")
    return samples, np.array(labels)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_selected_features(texts, model, tokenizer, sae_cache, feature_spec):
    """Extract features for selected latents using sparse TopK activations."""
    # Group features by layer
    layer_features = {}
    for f in feature_spec:
        layer = f["layer"]
        if layer not in layer_features:
            layer_features[layer] = []
        layer_features[layer].append(f["feature_idx"])

    all_features = []

    for layer_idx in sorted(layer_features.keys()):
        feature_indices = layer_features[layer_idx]

        print(f"    Layer {layer_idx}: {len(feature_indices)} features...")

        # Load SAE for this layer (don't cache - release after use)
        sae = Sae.load_from_hub(
            SAE_RELEASE, hookpoint=f"layers.{layer_idx}"
        ).to(model.device)

        n_latents = sae.num_latents
        layer_feats = []

        for i, text in enumerate(texts):
            if i % 100 == 0 and i > 0:
                print(f"      {i}/{len(texts)}...")

            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            tokens = {k: v.to(model.device) for k, v in tokens.items()}

            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)

            hidden = outputs.hidden_states[layer_idx + 1].squeeze(0).to(torch.bfloat16)
            encoder_output = sae.encode(hidden)
            top_acts = encoder_output.top_acts.cpu().float()
            top_indices = encoder_output.top_indices.cpu()

            # Count and sum per selected latent
            latent_counts = torch.zeros(n_latents)
            latent_sums = torch.zeros(n_latents)
            for idx, act in zip(top_indices.flatten(), top_acts.flatten()):
                latent_counts[idx.item()] += 1
                latent_sums[idx.item()] += act.item()

            # Extract stats for selected features only [freq, sum, mean]
            sample_feats = []
            for feat_idx in feature_indices:
                freq = latent_counts[feat_idx].item()
                total = latent_sums[feat_idx].item()
                mean = total / freq if freq > 0 else 0
                stats = np.array([freq, total, mean])
                sample_feats.append(stats)

            layer_feats.append(np.concatenate(sample_feats))

        all_features.append(np.array(layer_feats))

        # Free GPU memory after each layer
        del sae
        torch.cuda.empty_cache()

    return np.hstack(all_features)


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load probe spec
    spec_path = f"{OUTPUT_DIR}/probe_spec.json"
    try:
        with open(spec_path) as f:
            probe_spec = json.load(f)
    except FileNotFoundError:
        print("ERROR: Run feature_discovery.py first")
        return

    features = probe_spec["features"]
    print("=" * 70)
    print("LLAMA-3-8B PROBE TRAINING")
    print("=" * 70)
    print(f"Using {len(features)} features from {spec_path}")

    # Load model with 8-bit quantization
    print("\n[MODEL] Loading Llama-3-8B (8-bit quantized)...")
    try:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    except ImportError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # SAE cache
    sae_cache = {}

    # Load data
    train_samples, train_labels = load_training_data()
    test_samples, test_labels = load_gold_106()

    train_texts = [s["text"] for s in train_samples]
    test_texts = [s["text"] for s in test_samples]

    # Extract training features
    print("\n[TRAIN] Extracting training features...")
    X_train = extract_selected_features(train_texts, model, tokenizer, sae_cache, features)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Training features shape: {X_train.shape}")

    # Extract test features
    print("\n[TEST] Extracting test features...")
    X_test = extract_selected_features(test_texts, model, tokenizer, sae_cache, features)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Test features shape: {X_test.shape}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Cross-validation on training
    print("\n[CV] Cross-validation on training data...")
    probe = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    cv_scores = cross_val_score(probe, X_train_scaled, train_labels, cv=5, scoring="roc_auc")
    print(f"  Training CV AUROC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Fit on full training set
    probe.fit(X_train_scaled, train_labels)

    # Evaluate on gold_106
    print("\n[EVAL] Evaluating on gold_106...")
    test_probs = probe.predict_proba(X_test_scaled)[:, 1]
    test_preds = probe.predict(X_test_scaled)

    auroc = roc_auc_score(test_labels, test_probs)
    acc = accuracy_score(test_labels, test_preds)
    prec = precision_score(test_labels, test_preds)
    rec = recall_score(test_labels, test_preds)

    print("\n" + "=" * 70)
    print("GOLD_106 RESULTS")
    print("=" * 70)
    print(f"  AUROC:     {auroc:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"""
| Detector            | Type           | AUROC   |
|---------------------|----------------|---------|
| Llama-8B Fine-tuned | LoRA fine-tune | 91.66%  |
| Keyword Regex       | 41 patterns    | 75.06%  |
| Gemma SAE Probe     | 8 features     | 72.01%  |
| Llama SAE Probe     | {len(features)} features    | {auroc*100:.2f}%  |
""")

    # Save results
    results = {
        "model": MODEL_NAME,
        "sae_release": SAE_RELEASE,
        "timestamp": datetime.now().isoformat(),
        "n_features": len(features),
        "training": {
            "cv_auroc_mean": float(cv_scores.mean()),
            "cv_auroc_std": float(cv_scores.std()),
            "n_samples": len(train_labels),
        },
        "gold_106": {
            "auroc": float(auroc),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "n_samples": len(test_labels),
        },
        "comparison": {
            "llama_finetuned": 0.9166,
            "keyword_baseline": 0.7506,
            "gemma_sae_probe": 0.7201,
            "llama_sae_probe": float(auroc),
        },
    }

    output_path = f"{OUTPUT_DIR}/final_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Clean up
    for sae in sae_cache.values():
        del sae
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
