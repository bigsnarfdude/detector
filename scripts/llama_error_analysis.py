#!/usr/bin/env python3
"""
Llama SAE Probe Error Analysis
==============================

Analyze what the probe gets wrong on gold_106 held-out data.
Shows false positives, false negatives, and confidence distribution.

Usage:
    python scripts/llama_error_analysis.py
"""

import os
import json
import torch
import numpy as np
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
OUTPUT_DIR = "llama_results"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data():
    """Load training data."""
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

    n_per_class = min(500, len(af_idx), len(al_idx))
    rng = np.random.default_rng(42)
    af_select = rng.choice(af_idx, n_per_class, replace=False)
    al_select = rng.choice(al_idx, n_per_class, replace=False)

    selected = list(af_select) + list(al_select)
    samples = [samples[i] for i in selected]
    labels = np.array([labels[i] for i in selected])

    return samples, labels


def load_gold_106():
    """Load held-out test set."""
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

    return samples, np.array(labels)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(texts, model, tokenizer, feature_spec):
    """Extract features for given texts."""
    layer_features = {}
    for f in feature_spec:
        layer = f["layer"]
        if layer not in layer_features:
            layer_features[layer] = []
        layer_features[layer].append(f["feature_idx"])

    all_features = []

    for layer_idx in sorted(layer_features.keys()):
        feature_indices = layer_features[layer_idx]

        sae = Sae.load_from_hub(
            SAE_RELEASE, hookpoint=f"layers.{layer_idx}"
        ).to(model.device)

        n_latents = sae.num_latents
        layer_feats = []

        for text in texts:
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

            latent_counts = torch.zeros(n_latents)
            latent_sums = torch.zeros(n_latents)
            for idx, act in zip(top_indices.flatten(), top_acts.flatten()):
                latent_counts[idx.item()] += 1
                latent_sums[idx.item()] += act.item()

            sample_feats = []
            for feat_idx in feature_indices:
                freq = latent_counts[feat_idx].item()
                total = latent_sums[feat_idx].item()
                mean = total / freq if freq > 0 else 0
                stats = np.array([freq, total, mean])
                sample_feats.append(stats)

            layer_feats.append(np.concatenate(sample_feats))

        all_features.append(np.array(layer_feats))

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
        print("ERROR: Run train_probe.py first")
        return

    features = probe_spec["features"]

    print("=" * 70)
    print("LLAMA SAE PROBE ERROR ANALYSIS")
    print("=" * 70)

    # Load model
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
    print("\n[DATA] Loading data...")
    train_samples, train_labels = load_training_data()
    test_samples, test_labels = load_gold_106()

    train_texts = [s["text"] for s in train_samples]
    test_texts = [s["text"] for s in test_samples]

    # Extract features
    print("\n[TRAIN] Extracting training features...")
    X_train = extract_features(train_texts, model, tokenizer, features)
    X_train = np.nan_to_num(X_train)

    print("\n[TEST] Extracting test features...")
    X_test = extract_features(test_texts, model, tokenizer, features)
    X_test = np.nan_to_num(X_test)

    # Train probe
    print("\n[PROBE] Training...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    probe = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    probe.fit(X_train_scaled, train_labels)

    # Predict
    test_probs = probe.predict_proba(X_test_scaled)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    # Analyze errors
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    # False Positives (predicted AF, actually genuine)
    fp_mask = (test_preds == 1) & (test_labels == 0)
    fp_indices = np.where(fp_mask)[0]

    # False Negatives (predicted genuine, actually AF)
    fn_mask = (test_preds == 0) & (test_labels == 1)
    fn_indices = np.where(fn_mask)[0]

    # True Positives
    tp_mask = (test_preds == 1) & (test_labels == 1)
    tp_indices = np.where(tp_mask)[0]

    # True Negatives
    tn_mask = (test_preds == 0) & (test_labels == 0)
    tn_indices = np.where(tn_mask)[0]

    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {len(tp_indices)} (AF correctly detected)")
    print(f"  True Negatives:  {len(tn_indices)} (Genuine correctly identified)")
    print(f"  False Positives: {len(fp_indices)} (Genuine misclassified as AF)")
    print(f"  False Negatives: {len(fn_indices)} (AF missed)")

    # Confidence distribution
    print(f"\nConfidence Distribution:")
    print(f"  AF samples - mean prob: {test_probs[test_labels == 1].mean():.3f}")
    print(f"  Genuine samples - mean prob: {test_probs[test_labels == 0].mean():.3f}")

    # Show false positives
    print("\n" + "-" * 70)
    print("FALSE POSITIVES (Genuine samples misclassified as AF)")
    print("-" * 70)

    for i, idx in enumerate(fp_indices[:5]):  # Show first 5
        sample = test_samples[idx]
        prob = test_probs[idx]
        text = sample["text"][:300].replace("\n", " ")
        print(f"\n[FP {i+1}] Prob={prob:.3f}")
        print(f"  Label: {sample.get('label', 'unknown')}")
        print(f"  Text: {text}...")

    # Show false negatives
    print("\n" + "-" * 70)
    print("FALSE NEGATIVES (AF samples missed)")
    print("-" * 70)

    for i, idx in enumerate(fn_indices[:5]):  # Show first 5
        sample = test_samples[idx]
        prob = test_probs[idx]
        text = sample["text"][:300].replace("\n", " ")
        print(f"\n[FN {i+1}] Prob={prob:.3f}")
        print(f"  Label: {sample.get('label', 'unknown')}")
        print(f"  Text: {text}...")

    # High confidence errors
    print("\n" + "-" * 70)
    print("HIGHEST CONFIDENCE ERRORS")
    print("-" * 70)

    # Most confident FP
    if len(fp_indices) > 0:
        most_conf_fp = fp_indices[np.argmax(test_probs[fp_indices])]
        print(f"\nMost confident FP (prob={test_probs[most_conf_fp]:.3f}):")
        print(f"  {test_samples[most_conf_fp]['text'][:400]}...")

    # Most confident FN (lowest prob for actual AF)
    if len(fn_indices) > 0:
        most_conf_fn = fn_indices[np.argmin(test_probs[fn_indices])]
        print(f"\nMost confident FN (prob={test_probs[most_conf_fn]:.3f}):")
        print(f"  {test_samples[most_conf_fn]['text'][:400]}...")

    # Save results
    error_analysis = {
        "confusion_matrix": {
            "true_positives": int(len(tp_indices)),
            "true_negatives": int(len(tn_indices)),
            "false_positives": int(len(fp_indices)),
            "false_negatives": int(len(fn_indices)),
        },
        "confidence": {
            "af_mean_prob": float(test_probs[test_labels == 1].mean()),
            "genuine_mean_prob": float(test_probs[test_labels == 0].mean()),
        },
        "false_positive_samples": [
            {
                "index": int(idx),
                "prob": float(test_probs[idx]),
                "label": test_samples[idx].get("label", "unknown"),
                "text_preview": test_samples[idx]["text"][:200],
            }
            for idx in fp_indices
        ],
        "false_negative_samples": [
            {
                "index": int(idx),
                "prob": float(test_probs[idx]),
                "label": test_samples[idx].get("label", "unknown"),
                "text_preview": test_samples[idx]["text"][:200],
            }
            for idx in fn_indices
        ],
    }

    output_path = f"{OUTPUT_DIR}/error_analysis.json"
    with open(output_path, "w") as f:
        json.dump(error_analysis, f, indent=2)

    print(f"\n\nSaved to {output_path}")


if __name__ == "__main__":
    main()
