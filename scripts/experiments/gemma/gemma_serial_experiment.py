#!/usr/bin/env python3
"""
Gemma-3-27B SAE Probe Experiment - Serial Processing for Consumer GPUs

Runs on 16GB VRAM with CPU offloading. Processes samples one at a time.

Usage on RTX 4070 Ti (16GB VRAM, 64GB RAM):
    export HF_TOKEN=your_token
    python scripts/gemma_serial_experiment.py
"""

import json
import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch.nn.functional as F
import gc
from datetime import datetime

print("="*70)
print("GEMMA-3-27B ALIGNMENT FAKING DETECTION (Serial/CPU Offload)")
print(f"Started: {datetime.now().isoformat()}")
print("="*70)

# Transcoder class for GemmaScope-2
class Transcoder:
    def __init__(self, repo_id, layer, device="cpu"):
        folder = f"transcoder/layer_{layer}_width_16k_l0_medium"
        config_path = hf_hub_download(repo_id, f"{folder}/config.json")
        weights_path = hf_hub_download(repo_id, f"{folder}/params.safetensors")
        with open(config_path) as f:
            self.config = json.load(f)
        weights = load_file(weights_path)
        self.w_enc = weights["w_enc"].to(device)
        self.b_enc = weights["b_enc"].to(device)
        self.threshold = weights["threshold"].to(device)
        self.device = device

    def to(self, device, dtype=None):
        self.w_enc = self.w_enc.to(device, dtype=dtype)
        self.b_enc = self.b_enc.to(device, dtype=dtype)
        self.threshold = self.threshold.to(device, dtype=dtype)
        self.device = device
        return self

    def encode(self, x):
        x = x.to(self.device, dtype=self.w_enc.dtype)
        pre_acts = x @ self.w_enc + self.b_enc
        return F.relu(pre_acts - self.threshold)


def load_data_from_hf(max_per_class=300):
    """Load training data from HuggingFace."""
    print("\n[1/6] Loading training data from HuggingFace...")

    train_texts = []
    y_train = []

    for class_name, label in [("alignment_faking", 1), ("hard_negative", 0), ("aligned", 0)]:
        try:
            path = hf_hub_download(
                repo_id="vincentoh/af-detection-benchmark",
                filename=f"classes/{class_name}.json",
                repo_type="dataset",
            )
            with open(path) as f:
                data = json.load(f)
            samples = data.get("samples", data)[:max_per_class]
            train_texts.extend([s["text"] for s in samples])
            y_train.extend([label] * len(samples))
            print(f"  {class_name}: {len(samples)} samples")
        except Exception as e:
            print(f"  {class_name}: FAILED - {e}")

    return train_texts, np.array(y_train)


def load_gold_106():
    """Load held-out test set."""
    print("\n[2/6] Loading gold_106 (EVAL ONLY)...")

    path = hf_hub_download(
        repo_id="vincentoh/alignment-faking-training",
        filename="gold_106.json",
        repo_type="dataset"
    )
    with open(path) as f:
        gold = json.load(f).get("samples", [])

    test_texts = [s["text"] for s in gold]
    y_test = np.array([
        1 if s.get("label", "") in ["potential_faking", "alignment_faking", "AF"] else 0
        for s in gold
    ])
    print(f"  gold_106: {len(test_texts)} ({y_test.sum()} AF, {len(y_test)-y_test.sum()} aligned)")
    return test_texts, y_test


# Selected features from feature discovery
PROBE_FEATURES = [
    (40, 12574), (40, 8921), (40, 15484),
    (53, 15529), (53, 8003), (53, 4824), (53, 351),
    (31, 15111),
]
LAYERS = list(set(f[0] for f in PROBE_FEATURES))

# Output directory
OUTPUT_DIR = "gemma_serial_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # Load data
    train_texts, y_train = load_data_from_hf(max_per_class=300)
    test_texts, y_test = load_gold_106()

    print(f"\n  Training: {len(train_texts)} samples")
    print(f"  Test: {len(test_texts)} samples")

    # Load model with CPU offloading
    print("\n[3/6] Loading Gemma-3-27B-IT (CPU offload mode)...")
    MODEL_ID = "google/gemma-3-27b-it"
    SAE_REPO = "google/gemma-scope-2-27b-it"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Auto splits between GPU and CPU
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("  Model loaded with device_map='auto'")

    # Check device distribution
    if hasattr(model, 'hf_device_map'):
        devices = set(model.hf_device_map.values())
        print(f"  Model distributed across: {devices}")

    # Load transcoders - keep on CPU, move per-sample
    print("\n[4/6] Loading GemmaScope-2 transcoders...")
    transcoders = {}
    for layer in LAYERS:
        print(f"  Loading layer {layer}...")
        transcoders[layer] = Transcoder(SAE_REPO, layer, device="cpu")

    def get_feature_vector(text, sample_idx=0, total=1):
        """Extract features for a single sample (serial processing)."""
        activations = {}
        handles = []

        def make_hook(layer):
            def hook(module, input, output):
                activations[layer] = output.detach().cpu()  # Move to CPU immediately
            return hook

        for layer in transcoders.keys():
            try:
                target = model.model.language_model.layers[layer]
            except AttributeError:
                target = model.model.layers[layer]
            handle = target.pre_feedforward_layernorm.register_forward_hook(make_hook(layer))
            handles.append(handle)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        # Move inputs to model's first device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        # Extract only the 8 selected features (32-dim total)
        feature_vector = []
        for layer, feature_idx in PROBE_FEATURES:
            acts = activations[layer].squeeze(0).to(torch.float32)
            feats = transcoders[layer].encode(acts)
            feat_acts = feats[:, feature_idx]
            feature_vector.extend([
                feat_acts.max().item(),
                feat_acts.mean().item(),
                (feat_acts > 0).sum().item(),
                feat_acts.std().item() if len(feat_acts) > 1 else 0,
            ])

        # Clear activations to free memory
        del activations
        gc.collect()

        return np.array(feature_vector)

    # Extract training features (serial)
    print("\n[5/6] Extracting features (serial processing)...")
    print("  Training set...")
    X_train = []
    for i, text in enumerate(train_texts):
        if i % 10 == 0:
            print(f"    {i}/{len(train_texts)} ({datetime.now().strftime('%H:%M:%S')})")
        X_train.append(get_feature_vector(text, i, len(train_texts)))

        # Save checkpoint every 100 samples
        if (i + 1) % 100 == 0:
            np.save(f"{OUTPUT_DIR}/X_train_checkpoint_{i+1}.npy", np.array(X_train))

    X_train = np.array(X_train)
    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)

    print("  Test set (gold_106)...")
    X_test = []
    for i, text in enumerate(test_texts):
        if i % 10 == 0:
            print(f"    {i}/{len(test_texts)} ({datetime.now().strftime('%H:%M:%S')})")
        X_test.append(get_feature_vector(text, i, len(test_texts)))

    X_test = np.array(X_test)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"  Train features: {X_train.shape}")  # Should be (~900, 32)
    print(f"  Test features: {X_test.shape}")    # Should be (106, 32)

    # Train and evaluate
    print("\n[6/6] Training probe and evaluating...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
    print(f"  CV AUROC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Final training and eval
    clf.fit(X_train_scaled, y_train)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auroc = roc_auc_score(y_test, y_prob)
    print(f"\n{'='*70}")
    print(f"RESULTS: Opus-curated -> gold_106 Transfer")
    print(f"{'='*70}")
    print(f"AUROC: {auroc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Aligned", "AF"]))

    # Save results
    results = {
        "model": MODEL_ID,
        "sae": SAE_REPO,
        "timestamp": datetime.now().isoformat(),
        "n_train": len(train_texts),
        "n_test": len(test_texts),
        "cv_auroc_mean": float(cv_scores.mean()),
        "cv_auroc_std": float(cv_scores.std()),
        "test_auroc": float(auroc),
        "features": PROBE_FEATURES,
    }

    with open(f"{OUTPUT_DIR}/gemma_serial_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
