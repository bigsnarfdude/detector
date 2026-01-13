#!/usr/bin/env python3
"""
Llama-3-8B Feature Discovery for Alignment Faking Detection
============================================================

Find discriminative SAE features from best layers.
Run AFTER layer_sweep.py to get best layers.

SAE: EleutherAI/sae-llama-3-8b-32x
Library: sparsify

Usage:
    python scripts/llama_feature_discovery.py [--layers 16 20 24]
"""

import os
import json
import argparse
import torch
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from huggingface_hub import hf_hub_download

# =============================================================================
# CONFIG
# =============================================================================

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
SAE_RELEASE = "EleutherAI/sae-llama-3-8b-32x"
N_SAMPLES = 300
TOP_K_FEATURES = 10  # Per layer
OUTPUT_DIR = "llama_results"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load AF training data."""
    print("\n[DATA] Loading training data...")

    # Load main training data
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
        samples.append(s["text"])

    # Balance classes
    af_idx = [i for i, l in enumerate(labels) if l == 1]
    al_idx = [i for i, l in enumerate(labels) if l == 0]

    n_per_class = min(N_SAMPLES, len(af_idx), len(al_idx))
    rng = np.random.default_rng(42)
    af_select = rng.choice(af_idx, n_per_class, replace=False)
    al_select = rng.choice(al_idx, n_per_class, replace=False)

    selected = list(af_select) + list(al_select)
    samples = [samples[i] for i in selected]
    labels = np.array([labels[i] for i in selected])

    print(f"  AF: {sum(labels)}, Aligned: {len(labels) - sum(labels)}")

    return samples, labels


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_sae_features(texts, model, tokenizer, sae, layer_idx):
    """Extract SAE features for a single layer."""
    features = []

    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"    {i}/{len(texts)}...")

        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        tokens = {k: v.to(model.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)

        hidden = outputs.hidden_states[layer_idx + 1].squeeze(0).float()
        sae_acts = sae.encode(hidden)

        # Stats per latent
        feat_max = sae_acts.max(dim=0).values
        feat_mean = sae_acts.mean(dim=0)
        feat_count = (sae_acts > 0).float().sum(dim=0)
        feat_std = sae_acts.std(dim=0)

        stats = torch.stack([feat_max, feat_mean, feat_count, feat_std], dim=1)
        features.append(stats.cpu().numpy())  # [n_latents, 4]

    return np.array(features)  # [n_samples, n_latents, 4]


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layers to analyze (default: read from layer_sweep.json)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get best layers
    if args.layers:
        best_layers = args.layers
    else:
        try:
            with open(f"{OUTPUT_DIR}/layer_sweep.json") as f:
                sweep_data = json.load(f)
            best_layers = sweep_data["best_layers"]
        except FileNotFoundError:
            print("ERROR: Run layer_sweep.py first, or specify --layers")
            return

    print("=" * 70)
    print("LLAMA-3-8B FEATURE DISCOVERY")
    print("=" * 70)
    print(f"Analyzing layers: {best_layers}")

    # Load model
    print("\n[MODEL] Loading Llama-3-8B...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    texts, labels = load_data()

    all_features = []

    for layer_idx in best_layers:
        print(f"\n{'='*60}")
        print(f"ANALYZING LAYER {layer_idx}")
        print("=" * 60)

        # Load SAE
        sae = Sae.load_from_hub(SAE_RELEASE, hookpoint=f"layers.{layer_idx}")
        sae = sae.to(model.device)
        n_latents = sae.num_latents
        print(f"  SAE latents: {n_latents}")

        # Extract features
        print("  Extracting features...")
        X = extract_sae_features(texts, model, tokenizer, sae, layer_idx)
        # X shape: [n_samples, n_latents, 4]

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Use max activation as primary signal
        max_acts = X[:, :, 0]  # [n_samples, n_latents]

        # Find features with best class separation
        af_mask = labels == 1
        af_means = max_acts[af_mask].mean(axis=0)
        non_af_means = max_acts[~af_mask].mean(axis=0)
        pooled_std = max_acts.std(axis=0) + 1e-6

        # Separation score (like t-statistic)
        separation = (af_means - non_af_means) / pooled_std

        # Also consider variance (features that activate differently)
        variance = max_acts.var(axis=0)

        # Combined score: separation × sqrt(variance)
        combined_score = np.abs(separation) * np.sqrt(variance + 1e-6)

        # Top features
        top_idx = np.argsort(combined_score)[-TOP_K_FEATURES:][::-1]

        print(f"\n  Top {TOP_K_FEATURES} features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. F{idx}: separation={separation[idx]:.3f}, "
                  f"AF_mean={af_means[idx]:.3f}, non-AF={non_af_means[idx]:.3f}")
            all_features.append({
                "layer": layer_idx,
                "feature_idx": int(idx),
                "separation_score": float(separation[idx]),
                "af_mean": float(af_means[idx]),
                "non_af_mean": float(non_af_means[idx]),
                "variance": float(variance[idx]),
                "combined_score": float(combined_score[idx]),
            })

        del sae
        torch.cuda.empty_cache()

    # Sort all features by combined score, take top 24 (8 per layer × 3 layers)
    all_features_sorted = sorted(all_features, key=lambda x: x["combined_score"], reverse=True)
    top_features = all_features_sorted[:24]

    # Create probe spec
    probe_spec = {
        "model": MODEL_NAME,
        "sae_release": SAE_RELEASE,
        "timestamp": datetime.now().isoformat(),
        "layers": best_layers,
        "features": top_features,
        "stats_per_feature": ["max", "mean", "count", "std"],
    }

    # Save
    spec_path = f"{OUTPUT_DIR}/probe_spec.json"
    with open(spec_path, "w") as f:
        json.dump(probe_spec, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("SELECTED FEATURES")
    print("=" * 70)

    for f in top_features[:12]:
        print(f"  L{f['layer']}_F{f['feature_idx']}: "
              f"separation={f['separation_score']:.3f}, "
              f"combined={f['combined_score']:.3f}")

    print(f"\nSaved {len(top_features)} features to {spec_path}")


if __name__ == "__main__":
    main()
