#!/usr/bin/env python3
"""
Llama-3-8B Layer Sweep for Alignment Faking Detection
======================================================

Sweep all layers to find which best encode alignment faking.
Equivalent to Gemma finding layers 31, 40, 53.

SAE: EleutherAI/sae-llama-3-8b-32x (all 32 layers)
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
    """Extract SAE features for a single layer."""
    features = []

    for i, text in enumerate(texts):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(texts)}...")

        # Tokenize with shorter max length to save memory
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # Reduced from 1024
            padding=False,
        )
        tokens = {k: v.to(model.device) for k, v in tokens.items()}

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)

        # Get hidden state at target layer (+1 because index 0 is embeddings)
        hidden = outputs.hidden_states[layer_idx + 1]
        hidden = hidden.squeeze(0)  # [seq, hidden_dim]

        # Cast to match SAE dtype (bfloat16)
        hidden = hidden.to(torch.bfloat16)

        # Encode through SAE - use pre_acts for dense activations
        # EncoderOutput is NamedTuple(top_acts, top_indices, pre_acts)
        encoder_output = sae.encode(hidden)
        sae_acts = encoder_output.pre_acts.cpu().float()  # [seq, n_latents]

        # Aggregate stats: max, mean, count (>0), std
        feat_max = sae_acts.max(dim=0).values
        feat_mean = sae_acts.mean(dim=0)
        feat_count = (sae_acts > 0).float().sum(dim=0)
        feat_std = sae_acts.std(dim=0)

        stats = torch.stack([feat_max, feat_mean, feat_count, feat_std], dim=1)
        features.append(stats.detach().numpy().flatten())

        # Clear cache periodically
        if i % 20 == 0:
            torch.cuda.empty_cache()

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

        # Use L1 regularization for sparse feature selection
        probe = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=0.01,
            max_iter=1000,
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
