#!/usr/bin/env python3
"""
Test SAE Probes on Sycophant Samples
====================================
Uses Gemma-4B (fits on 16GB) to extract SAE features from sycophant samples
and test if probes trained on Anthropic-style samples generalize.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from unsloth import FastLanguageModel
import warnings
warnings.filterwarnings('ignore')

DEVICE = "cuda"
MODEL_NAME = "unsloth/gemma-3-4b-it-bnb-4bit"
SAE_REPO = "google/gemma-scope-2-4b-it"
LAYERS = [17, 22]
RANDOM_SEED = 42


class TranscoderSAE:
    """GemmaScope 2 Transcoder with JumpReLU."""
    def __init__(self, repo_id: str, layer: int):
        folder = f"transcoder/layer_{layer}_width_16k_l0_medium"
        weights_path = hf_hub_download(repo_id, f"{folder}/params.safetensors")
        weights = load_file(weights_path)
        self.w_enc = weights["w_enc"]
        self.b_enc = weights["b_enc"]
        self.threshold = weights.get("threshold", None)

    def to(self, device, dtype=None):
        self.w_enc = self.w_enc.to(device, dtype=dtype)
        self.b_enc = self.b_enc.to(device, dtype=dtype)
        if self.threshold is not None:
            self.threshold = self.threshold.to(device, dtype=dtype)
        return self

    def encode(self, x):
        pre_acts = x @ self.w_enc + self.b_enc
        if self.threshold is not None:
            return F.relu(pre_acts - self.threshold)
        return F.relu(pre_acts)


def load_sycophant_samples():
    path = "/home/vincent/lightbright/experiments/jan18_bench_af/sycophant_cleaned_final.json"
    with open(path) as f:
        samples = json.load(f)
    return [{"text": s["scratchpad"], "is_af": 1, "environment": s["environment"]} for s in samples]


def load_training_data(cache_dir="./data"):
    files = [
        ("aligned.json", 0),
        ("alignment_faking.json", 1),
        ("genuine_compliance.json", 0),
        ("hard_negative.json", 0),
        ("compliant_hard_negative.json", 0),
    ]
    samples = []
    for fname, is_af in files:
        try:
            path = hf_hub_download(
                repo_id="vincentoh/af-detection-benchmark",
                filename=f"classes/{fname}",
                repo_type="dataset",
                local_dir=cache_dir
            )
            with open(path) as f:
                data = json.load(f)
            for s in data["samples"]:
                samples.append({"text": s["text"], "is_af": is_af})
        except Exception as e:
            print(f"  Warning: Could not load {fname}: {e}")
    return samples


def extract_features(model, tokenizer, sae, samples, layer):
    all_activations = []
    
    for sample in tqdm(samples, desc=f"Layer {layer}"):
        activations = {}

        def hook(module, input, output):
            if isinstance(output, tuple):
                activations["out"] = output[0].detach()
            else:
                activations["out"] = output.detach()

        # Gemma3 structure: model.language_model.layers
        if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
            target = model.language_model.layers[layer]
        elif hasattr(model, "model") and hasattr(model.model, "language_model"):
            target = model.model.language_model.layers[layer]
        else:
            raise ValueError("Cannot find layers in model")
            
        handle = target.register_forward_hook(hook)

        # Tokenize
        if hasattr(tokenizer, 'tokenizer'):
            tok = tokenizer.tokenizer
        else:
            tok = tokenizer
            
        inputs = tok(
            sample["text"][:2000],
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            model(**inputs)

        handle.remove()

        if "out" not in activations:
            all_activations.append(np.zeros(16384))
            continue

        act = activations["out"].squeeze(0).to(torch.bfloat16)
        with torch.no_grad():
            features = sae.encode(act)

        max_per_feature = features.max(dim=0).values.float().cpu().numpy()
        all_activations.append(max_per_feature)

    return np.array(all_activations)


def main():
    print("="*70)
    print("SAE PROBE TEST ON SYCOPHANT SAMPLES")
    print("="*70)
    
    print("\n[1/5] Loading samples...")
    sycophant_samples = load_sycophant_samples()
    print(f"  Sycophant: {len(sycophant_samples)} samples (all AF)")
    
    training_samples = load_training_data()
    print(f"  Training: {len(training_samples)} samples")
    y_train = np.array([s["is_af"] for s in training_samples])
    print(f"    AF: {y_train.sum()}, Non-AF: {(y_train==0).sum()}")
    
    print("\n[2/5] Loading Gemma-4B...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("  Model loaded!")
    
    results = {}
    
    for layer in LAYERS:
        print(f"\n{'='*70}")
        print(f"LAYER {layer}")
        print("="*70)
        
        print(f"\n[3/5] Loading SAE for layer {layer}...")
        try:
            sae = TranscoderSAE(SAE_REPO, layer).to(DEVICE, torch.bfloat16)
            print("  Loaded transcoder SAE")
        except Exception as e:
            print(f"  Error: {e}")
            continue
        
        print(f"\n[4/5] Extracting training features...")
        X_train = extract_features(model, tokenizer, sae, training_samples, layer)
        print(f"  Training shape: {X_train.shape}")
        
        print("\n  Training probe with 5-fold CV...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        probe = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)
        
        cv_scores = cross_val_score(probe, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
        print(f"  CV AUROC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
        
        probe.fit(X_train_scaled, y_train)
        
        print(f"\n[5/5] Extracting sycophant features...")
        X_syco = extract_features(model, tokenizer, sae, sycophant_samples, layer)
        X_syco_scaled = scaler.transform(X_syco)
        print(f"  Sycophant shape: {X_syco.shape}")
        
        syco_probs = probe.predict_proba(X_syco_scaled)[:, 1]
        syco_preds = probe.predict(X_syco_scaled)
        
        syco_recall = syco_preds.sum() / len(syco_preds)
        syco_mean_prob = syco_probs.mean()
        
        print(f"\n  SYCOPHANT RESULTS (all samples are AF):")
        print(f"    Detection rate: {syco_recall:.1%} ({syco_preds.sum()}/{len(syco_preds)})")
        print(f"    Mean AF probability: {syco_mean_prob:.3f}")
        
        print(f"\n  By environment:")
        envs = {}
        for i, s in enumerate(sycophant_samples):
            env = s["environment"]
            if env not in envs:
                envs[env] = {"preds": [], "probs": []}
            envs[env]["preds"].append(syco_preds[i])
            envs[env]["probs"].append(syco_probs[i])
        
        for env in sorted(envs.keys()):
            det_rate = sum(envs[env]["preds"]) / len(envs[env]["preds"])
            mean_prob = np.mean(envs[env]["probs"])
            print(f"    {env}: {det_rate:.0%} detected, mean prob {mean_prob:.2f}")
        
        results[f"layer_{layer}"] = {
            "cv_auroc": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "sycophant_detection_rate": float(syco_recall),
            "sycophant_mean_prob": float(syco_mean_prob),
            "by_environment": {env: {
                "detection_rate": float(sum(envs[env]["preds"]) / len(envs[env]["preds"])),
                "mean_prob": float(np.mean(envs[env]["probs"]))
            } for env in envs}
        }
    
    print("\n" + "="*70)
    print("SUMMARY: SAE PROBE VS FINE-TUNED DETECTORS")
    print("="*70)
    
    print("\n  Fine-tuned detectors on sycophant (319 samples, all AF):")
    print("    Keyword baseline:  86.2%")
    print("    Llama-8B:          48.0%")
    print("    GPT-OSS 20B:       33.2%")
    print("    Gemma-4B:          31.7%")
    
    print("\n  SAE Probes on sycophant:")
    for layer_key, res in results.items():
        print(f"    {layer_key}: {res['sycophant_detection_rate']:.1%} (CV AUROC: {res['cv_auroc']:.3f})")
    
    output_file = "/home/vincent/lightbright/experiments/jan18_bench_af/sae_probe_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
