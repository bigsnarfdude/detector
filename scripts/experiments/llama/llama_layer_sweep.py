#!/usr/bin/env python3
"""
Llama-3-8B Layer Sweep for Alignment Faking Detection
======================================================

Sweep all layers to find which best encode alignment faking.
Equivalent to Gemma finding layers 31, 40, 53.

SAE: EleutherAI/sae-llama-3-8b-32x (layers 0-30, 31 total)
Library: sparsify (pip install sparsify)

Usage:
    python scripts/llama_layer_sweep.py
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae  # sparsify library
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from huggingface_hub import hf_hub_download

# =============================================================================
# CONFIG
# =============================================================================

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
SAE_RELEASE = "EleutherAI/sae-llama-3-8b-32x"
LAYERS_TO_TEST = [8, 16, 20, 24, 28]  # Reduced layers for speed/memory
N_SAMPLES = 100  # Reduced for 16GB GPU
OUTPUT_DIR = "llama_results"

print("=" * 70)
print("LLAMA-3-8B LAYER SWEEP")
print("=" * 70)
print(f"Model: {MODEL_NAME}")
print(f"SAE: {SAE_RELEASE}")
print(f"Layers: {LAYERS_TO_TEST}")
print(f"Started: {datetime.now().isoformat()}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load AF training data from HuggingFace."""
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
        else:  # aligned
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

def extract_sae_features(texts, model, tokenizer, sae, layer_idx, batch_size=1):
    """Extract SAE features using top-k latent IDs as categorical features.

    For TopK SAEs, the key discriminative info is WHICH latents fire.
    We count occurrences of specific latent IDs per sample.
    """
    n_latents = sae.num_latents
    features = []

    # We'll collect all indices first, then find the most common ones
    all_sample_indices = []

    for i, text in enumerate(texts):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(texts)}...")

        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        tokens = {k: v.to(model.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)

        hidden = outputs.hidden_states[layer_idx + 1]
        hidden = hidden.squeeze(0).to(torch.bfloat16)

        # Encode through SAE
        encoder_output = sae.encode(hidden)
        top_acts = encoder_output.top_acts.cpu().float()  # [seq, k]
        top_indices = encoder_output.top_indices.cpu()    # [seq, k]

        # Count frequency of each latent in this sample
        latent_counts = torch.zeros(n_latents)
        latent_sums = torch.zeros(n_latents)

        for idx, act in zip(top_indices.flatten(), top_acts.flatten()):
            latent_counts[idx.item()] += 1
            latent_sums[idx.item()] += act.item()

        all_sample_indices.append((latent_counts, latent_sums))

        if i % 20 == 0:
            torch.cuda.empty_cache()

    # Find the most frequently activated latents across ALL samples
    total_counts = torch.stack([x[0] for x in all_sample_indices]).sum(dim=0)
    top_latents = torch.argsort(total_counts, descending=True)[:5000]  # Top 5000 latents

    print(f"  Selected {len(top_latents)} most active latents")
    print(f"  Top latent IDs: {top_latents[:20].tolist()}")

    # Extract features for selected latents only
    for latent_counts, latent_sums in all_sample_indices:
        # Frequency features for top latents
        freq_feats = latent_counts[top_latents].numpy()
        # Sum features for top latents
        sum_feats = latent_sums[top_latents].numpy()
        # Mean features (avoiding div by zero)
        mean_feats = np.where(freq_feats > 0, sum_feats / freq_feats, 0)

        # Stack: [freq, sum, mean] for each latent = 15000 features
        sample_features = np.concatenate([freq_feats, sum_feats, mean_feats])
        features.append(sample_features)

    return np.array(features)


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model with 8-bit quantization to fit in 16GB
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
        print("  bitsandbytes not available, trying float16 with CPU offload...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder="offload",
            low_cpu_mem_usage=True,
        )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model loaded on {next(model.parameters()).device}")

    # Load data
    texts, labels = load_data()

    results = {}

    for layer_idx in LAYERS_TO_TEST:
        print(f"\n{'='*60}")
        print(f"TESTING LAYER {layer_idx}")
        print("=" * 60)

        # Load SAE for this layer
        print(f"  Loading SAE for layers.{layer_idx}...")
        sae = Sae.load_from_hub(SAE_RELEASE, hookpoint=f"layers.{layer_idx}")
        sae = sae.to(model.device)
        print(f"  SAE latents: {sae.num_latents}")

        # Extract features
        X = extract_sae_features(texts, model, tokenizer, sae, layer_idx)
        y = labels

        print(f"  Feature shape: {X.shape}")

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Quick probe training with CV
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Use L2 regularization (default)
        probe = LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
        )

        try:
            scores = cross_val_score(probe, X_scaled, y, cv=5, scoring="roc_auc")

            results[layer_idx] = {
                "mean_auroc": float(scores.mean()),
                "std_auroc": float(scores.std()),
                "n_latents": sae.num_latents,
                "n_features": X.shape[1],
            }

            print(f"\n  Layer {layer_idx} AUROC: {scores.mean():.4f} (+/- {scores.std():.4f})")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[layer_idx] = {"error": str(e)}

        # Clear GPU memory
        del sae
        torch.cuda.empty_cache()

    # Find best layers
    valid_results = {k: v for k, v in results.items() if "mean_auroc" in v}
    best_layers = sorted(
        valid_results.keys(),
        key=lambda l: valid_results[l]["mean_auroc"],
        reverse=True
    )[:3]

    # Summary
    print("\n" + "=" * 70)
    print("LAYER SWEEP RESULTS")
    print("=" * 70)

    for layer in sorted(results.keys()):
        r = results[layer]
        if "mean_auroc" in r:
            marker = " *** BEST" if layer in best_layers else ""
            print(f"Layer {layer:2d}: AUROC={r['mean_auroc']:.4f} (+/- {r['std_auroc']:.4f}){marker}")
        else:
            print(f"Layer {layer:2d}: ERROR - {r.get('error', 'Unknown')}")

    print(f"\nBest layers for feature discovery: {best_layers}")

    # Save results
    output = {
        "model": MODEL_NAME,
        "sae_release": SAE_RELEASE,
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "best_layers": best_layers,
    }

    output_path = f"{OUTPUT_DIR}/layer_sweep.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
