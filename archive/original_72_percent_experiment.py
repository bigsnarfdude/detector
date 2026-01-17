#!/usr/bin/env python3
"""
ORIGINAL 72% AUROC EXPERIMENT - RECOVERED FROM TRANSCRIPT

This is the exact code that achieved 72.01% AUROC on gold_106.
Key difference from the 58% version: uses only 8 SELECTED features (32-dim),
not all 196k features.

Usage on H100:
    export HF_TOKEN=your_token
    cd ~/af-experiment && source venv/bin/activate
    python original_72_percent_experiment.py
"""

import json
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

print("="*70)
print("GEMMA-3-27B ALIGNMENT FAKING DETECTION")
print("Training: Opus-curated 5,019 samples")
print("Eval: gold_106 (held out)")
print("="*70)

# Transcoder class for GemmaScope
class Transcoder:
    def __init__(self, repo_id, layer):
        folder = f"transcoder/layer_{layer}_width_16k_l0_medium"
        config_path = hf_hub_download(repo_id, f"{folder}/config.json")
        weights_path = hf_hub_download(repo_id, f"{folder}/params.safetensors")
        with open(config_path) as f:
            self.config = json.load(f)
        weights = load_file(weights_path)
        self.w_enc = weights["w_enc"]
        self.b_enc = weights["b_enc"]
        self.threshold = weights["threshold"]

    def to(self, device, dtype=None):
        self.w_enc = self.w_enc.to(device, dtype=dtype)
        self.b_enc = self.b_enc.to(device, dtype=dtype)
        self.threshold = self.threshold.to(device, dtype=dtype)
        return self

    def encode(self, x):
        pre_acts = x @ self.w_enc + self.b_enc
        return F.relu(pre_acts - self.threshold)

# Load training data (Opus-curated)
print("\n[1/6] Loading Opus-curated training data...")
with open("data/combined/alignment_faking.json") as f:
    af = json.load(f).get("samples", [])
with open("data/combined/hard_negative.json") as f:
    hn = json.load(f).get("samples", [])
with open("data/combined/aligned.json") as f:
    al = json.load(f).get("samples", [])

print(f"  alignment_faking: {len(af)}")
print(f"  hard_negative: {len(hn)}")
print(f"  aligned: {len(al)}")

# Use all data (or subsample for speed)
np.random.seed(42)
max_per_class = 500  # Adjust based on time
af_sub = list(np.random.choice(af, min(max_per_class, len(af)), replace=False))
hn_sub = list(np.random.choice(hn, min(max_per_class, len(hn)), replace=False))
al_sub = list(np.random.choice(al, min(max_per_class, len(al)), replace=False))

train_texts = [s["text"] for s in af_sub + hn_sub + al_sub]
y_train = np.array([1]*len(af_sub) + [0]*len(hn_sub) + [0]*len(al_sub))
print(f"  Training samples: {len(train_texts)} ({y_train.sum()} AF, {len(y_train)-y_train.sum()} non-AF)")

# Load gold_106 (EVAL ONLY - never train on this!)
print("\n[2/6] Loading gold_106 (EVAL ONLY)...")
gold_path = hf_hub_download(repo_id="vincentoh/alignment-faking-training", filename="gold_106.json", repo_type="dataset")
with open(gold_path) as f:
    gold = json.load(f).get("samples", [])
test_texts = [s["text"] for s in gold]
y_test = np.array([1 if s.get("label","") in ["potential_faking", "alignment_faking", "AF"] else 0 for s in gold])
print(f"  gold_106: {len(test_texts)} ({y_test.sum()} AF, {len(y_test)-y_test.sum()} aligned)")

# Load model
print("\n[3/6] Loading Gemma-3-27B-IT...")
MODEL_ID = "google/gemma-3-27b-it"
SAE_REPO = "google/gemma-scope-2-27b-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
print("  Model loaded!")

# Load transcoders for key layers
print("\n[4/6] Loading GemmaScope transcoders...")
LAYERS = [31, 40, 53]
transcoders = {}
for layer in LAYERS:
    print(f"  Loading layer {layer}...")
    transcoders[layer] = Transcoder(SAE_REPO, layer).to("cuda", torch.bfloat16)

# KEY DIFFERENCE: Only 8 selected features, not all 16k!
# These were identified through feature discovery as most discriminative for AF intent
PROBE_FEATURES = [
    (40, 12574), (40, 8921), (40, 15484),
    (53, 15529), (53, 8003), (53, 4824), (53, 351),
    (31, 15111),
]

def get_feature_vector(text):
    activations = {}
    handles = []

    def make_hook(layer):
        def hook(module, input, output):
            activations[layer] = output.detach()
        return hook

    for layer in transcoders.keys():
        # Gemma-3 structure: model.model.language_model.layers
        try:
            target = model.model.language_model.layers[layer]
        except AttributeError:
            target = model.model.layers[layer]
        handle = target.pre_feedforward_layernorm.register_forward_hook(make_hook(layer))
        handles.append(handle)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    # Extract only the 8 selected features (32-dim total)
    feature_vector = []
    for layer, feature_idx in PROBE_FEATURES:
        acts = activations[layer].squeeze(0).to(transcoders[layer].w_enc.dtype)
        feats = transcoders[layer].encode(acts)
        feat_acts = feats[:, feature_idx]
        feature_vector.extend([
            feat_acts.max().item(),
            feat_acts.mean().item(),
            (feat_acts > 0).sum().item(),
            feat_acts.std().item() if len(feat_acts) > 1 else 0,
        ])
    return np.array(feature_vector)

# Extract training features
print("\n[5/6] Extracting features...")
print("  Training set...")
X_train = []
for i, text in enumerate(train_texts):
    if i % 50 == 0:
        print(f"    {i}/{len(train_texts)}")
    X_train.append(get_feature_vector(text))
X_train = np.array(X_train)

print("  Test set (gold_106)...")
X_test = []
for i, text in enumerate(test_texts):
    if i % 20 == 0:
        print(f"    {i}/{len(test_texts)}")
    X_test.append(get_feature_vector(text))
X_test = np.array(X_test)

print(f"  Train features: {X_train.shape}")  # Should be (1500, 32)
print(f"  Test features: {X_test.shape}")    # Should be (106, 32)

# Train and evaluate
print("\n[6/6] Training probe and evaluating...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# CV on training set
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
probe = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
cv_scores = cross_val_score(probe, X_train_s, y_train, cv=cv, scoring="roc_auc")
print(f"  Training CV AUROC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Final model
probe.fit(X_train_s, y_train)
y_prob = probe.predict_proba(X_test_s)[:, 1]
y_pred = (y_prob > 0.5).astype(int)

print("\n" + "="*70)
print("RESULTS: Opus-curated -> gold_106 Transfer")
print("="*70)
print(f"AUROC: {roc_auc_score(y_test, y_prob):.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["Aligned", "AF"]))

# Save results
results = {
    "model": "gemma-3-27b-it",
    "train_cv_auroc": float(cv_scores.mean()),
    "train_cv_std": float(cv_scores.std()),
    "gold106_auroc": float(roc_auc_score(y_test, y_prob)),
    "n_train": len(train_texts),
    "n_test": len(test_texts),
    "feature_dim": X_train.shape[1],
    "probe_features": [f"L{l}_F{f}" for l, f in PROBE_FEATURES],
}
with open("gemma3_27b_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to gemma3_27b_results.json")
