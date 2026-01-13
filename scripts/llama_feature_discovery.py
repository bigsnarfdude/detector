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

def extract_latent_stats(texts, labels, model, tokenizer, sae, layer_idx):
    """Extract per-latent statistics for feature discovery.

    Returns dict mapping latent_idx to stats about AF vs non-AF activation.
    """
    n_latents = sae.num_latents
    af_mask = labels == 1

    # Collect per-sample latent counts and sums
    all_counts = []
    all_sums = []

    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"    {i}/{len(texts)}...")

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

        # Count and sum per latent
        latent_counts = torch.zeros(n_latents)
        latent_sums = torch.zeros(n_latents)

        for idx, act in zip(top_indices.flatten(), top_acts.flatten()):
            latent_counts[idx.item()] += 1
            latent_sums[idx.item()] += act.item()

        all_counts.append(latent_counts)
        all_sums.append(latent_sums)

    # Stack into tensors
    all_counts = torch.stack(all_counts)  # [n_samples, n_latents]
    all_sums = torch.stack(all_sums)

    # Compute stats for AF vs non-AF
    af_counts = all_counts[af_mask].mean(dim=0)
    non_af_counts = all_counts[~af_mask].mean(dim=0)
    af_sums = all_sums[af_mask].mean(dim=0)
    non_af_sums = all_sums[~af_mask].mean(dim=0)

    # Separation score: difference in mean counts
    count_diff = af_counts - non_af_counts
    sum_diff = af_sums - non_af_sums

    # Total activation (for filtering)
    total_counts = all_counts.sum(dim=0)

    return {
        "all_counts": all_counts.numpy(),
        "all_sums": all_sums.numpy(),
        "af_counts": af_counts.numpy(),
        "non_af_counts": non_af_counts.numpy(),
        "count_diff": count_diff.numpy(),
        "sum_diff": sum_diff.numpy(),
        "total_counts": total_counts.numpy(),
    }


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

        # Extract per-latent stats
        print("  Extracting latent statistics...")
        stats = extract_latent_stats(texts, labels, model, tokenizer, sae, layer_idx)

        # Filter to latents that activate at least 10 times total
        active_mask = stats["total_counts"] >= 10
        active_indices = np.where(active_mask)[0]
        print(f"  Active latents (count >= 10): {len(active_indices)}")

        # Compute separation score on active latents
        count_diff = stats["count_diff"][active_indices]
        sum_diff = stats["sum_diff"][active_indices]
        total_counts = stats["total_counts"][active_indices]

        # Combined score: magnitude of difference weighted by activity
        combined_score = np.abs(count_diff) * np.sqrt(total_counts + 1)

        # Top features from this layer
        layer_top_k = min(TOP_K_FEATURES, len(active_indices))
        top_local_idx = np.argsort(combined_score)[-layer_top_k:][::-1]
        top_global_idx = active_indices[top_local_idx]

        print(f"\n  Top {layer_top_k} discriminative latents:")
        for rank, (local_i, global_i) in enumerate(zip(top_local_idx, top_global_idx)):
            af_c = stats["af_counts"][global_i]
            non_af_c = stats["non_af_counts"][global_i]
            diff = count_diff[local_i]
            print(f"    {rank+1}. L{global_i}: AF_count={af_c:.1f}, non-AF={non_af_c:.1f}, diff={diff:.2f}")
            all_features.append({
                "layer": layer_idx,
                "feature_idx": int(global_i),
                "af_count": float(af_c),
                "non_af_count": float(non_af_c),
                "count_diff": float(diff),
                "combined_score": float(combined_score[local_i]),
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
        "stats_per_feature": ["freq", "sum", "mean"],
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
              f"count_diff={f['count_diff']:.2f}, "
              f"combined={f['combined_score']:.2f}")

    print(f"\nSaved {len(top_features)} features to {spec_path}")


if __name__ == "__main__":
    main()
