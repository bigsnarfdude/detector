#!/usr/bin/env python3
"""
Classifier Comparison for SAE Probe Features
=============================================

Compare LogisticRegression, RandomForest, and XGBoost on SAE features.

Usage:
    # On GPU machine, extract features first:
    python scripts/compare_classifiers.py --extract --model llama

    # Or use cached features:
    python scripts/compare_classifiers.py --cached llama_features.npz
"""

import os
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")


# =============================================================================
# CLASSIFIERS
# =============================================================================

def get_classifiers():
    """Return dict of classifiers to compare."""
    classifiers = {
        "LogisticRegression": LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=42,
        ),
        "LogReg_C1": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
        ),
        "RandomForest_100": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "RandomForest_200": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
    }

    if HAS_XGBOOST:
        classifiers["XGBoost_100"] = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        classifiers["XGBoost_200"] = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )

    return classifiers


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_classifier(clf, X_train, y_train, X_test, y_test, scaler=None):
    """Train and evaluate a classifier."""
    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring="roc_auc")

    # Fit on full training set
    clf.fit(X_train_scaled, y_train)

    # Predict
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = clf.decision_function(X_test_scaled)
    y_pred = clf.predict(X_test_scaled)

    # Metrics
    results = {
        "cv_auroc_mean": float(cv_scores.mean()),
        "cv_auroc_std": float(cv_scores.std()),
        "test_auroc": float(roc_auc_score(y_test, y_prob)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    return results, scaler


def run_comparison(X_train, y_train, X_test, y_test, output_dir="classifier_comparison"):
    """Run full classifier comparison."""
    os.makedirs(output_dir, exist_ok=True)

    classifiers = get_classifiers()
    results = {}

    print("\n" + "=" * 70)
    print("CLASSIFIER COMPARISON")
    print("=" * 70)
    print(f"Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test: {X_test.shape[0]} samples")
    print(f"Classifiers: {list(classifiers.keys())}")
    print("=" * 70)

    # Shared scaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    for name, clf in classifiers.items():
        print(f"\n[{name}]")
        try:
            metrics, _ = evaluate_classifier(
                clf, X_train, y_train, X_test, y_test, scaler
            )
            results[name] = metrics
            print(f"  CV AUROC:   {metrics['cv_auroc_mean']:.4f} (+/- {metrics['cv_auroc_std']:.4f})")
            print(f"  Test AUROC: {metrics['test_auroc']:.4f}")
            print(f"  Test F1:    {metrics['test_f1']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"error": str(e)}

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY (sorted by Test AUROC)")
    print("=" * 70)
    print(f"{'Classifier':<25} {'CV AUROC':<12} {'Test AUROC':<12} {'Test F1':<10}")
    print("-" * 70)

    sorted_results = sorted(
        [(k, v) for k, v in results.items() if "error" not in v],
        key=lambda x: x[1]["test_auroc"],
        reverse=True
    )

    for name, metrics in sorted_results:
        print(f"{name:<25} {metrics['cv_auroc_mean']:.4f} +/- {metrics['cv_auroc_std']:.4f}  "
              f"{metrics['test_auroc']:.4f}       {metrics['test_f1']:.4f}")

    # Best classifier
    if sorted_results:
        best_name, best_metrics = sorted_results[0]
        print(f"\nBest: {best_name} with Test AUROC = {best_metrics['test_auroc']:.4f}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "n_features": X_train.shape[1],
        "results": results,
    }

    output_path = Path(output_dir) / "classifier_comparison.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


# =============================================================================
# FEATURE LOADING/EXTRACTION
# =============================================================================

def load_cached_features(path):
    """Load cached features from .npz file."""
    data = np.load(path)
    return (
        data["X_train"],
        data["y_train"],
        data["X_test"],
        data["y_test"],
    )


def load_data_from_huggingface():
    """Load training and test data from HuggingFace."""
    from huggingface_hub import hf_hub_download

    # Training data
    print("[DATA] Loading training data...")
    train_path = hf_hub_download(
        repo_id="vincentoh/alignment-faking-training",
        filename="training_data_final.json",
        repo_type="dataset",
    )
    with open(train_path) as f:
        train_data = json.load(f)

    train_samples = train_data.get("samples", train_data)
    train_labels = []
    train_texts = []

    for s in train_samples:
        label_str = s.get("label", "")
        if label_str in ["potential_faking", "alignment_faking", "af"]:
            train_labels.append(1)
        else:
            train_labels.append(0)
        train_texts.append(s["text"])

    # Balance classes
    af_idx = [i for i, l in enumerate(train_labels) if l == 1]
    al_idx = [i for i, l in enumerate(train_labels) if l == 0]
    n_per_class = min(500, len(af_idx), len(al_idx))

    rng = np.random.default_rng(42)
    selected = list(rng.choice(af_idx, n_per_class, replace=False)) + \
               list(rng.choice(al_idx, n_per_class, replace=False))

    train_texts = [train_texts[i] for i in selected]
    train_labels = np.array([train_labels[i] for i in selected])

    print(f"  Training: {len(train_texts)} samples (AF: {sum(train_labels)}, Non-AF: {len(train_labels) - sum(train_labels)})")

    # Test data (gold_106)
    print("[DATA] Loading gold_106...")
    test_path = hf_hub_download(
        repo_id="vincentoh/alignment-faking-training",
        filename="gold_106.json",
        repo_type="dataset"
    )
    with open(test_path) as f:
        test_data = json.load(f)

    test_samples = test_data.get("samples", test_data)
    test_labels = []
    test_texts = []

    for s in test_samples:
        label_str = s.get("label", "")
        if label_str in ["alignment_faking", "potential_faking", "af", "AF"]:
            test_labels.append(1)
        else:
            test_labels.append(0)
        test_texts.append(s["text"])

    test_labels = np.array(test_labels)
    print(f"  Test: {len(test_texts)} samples (AF: {sum(test_labels)}, Non-AF: {len(test_labels) - sum(test_labels)})")

    return train_texts, train_labels, test_texts, test_labels


def extract_llama_features(train_texts, test_texts, output_path=None):
    """Extract features using Llama-3-8B SAE."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sparsify import Sae

    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
    SAE_RELEASE = "EleutherAI/sae-llama-3-8b-32x"

    # Load probe spec
    spec_path = "llama_results/probe_spec.json"
    with open(spec_path) as f:
        probe_spec = json.load(f)
    features = probe_spec["features"]

    print(f"\n[MODEL] Loading Llama-3-8B...")
    try:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
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

    def extract_features(texts, desc=""):
        """Extract features for a list of texts."""
        # Group features by layer
        layer_features = {}
        for f in features:
            layer = f["layer"]
            if layer not in layer_features:
                layer_features[layer] = []
            layer_features[layer].append(f["feature_idx"])

        all_features = []

        for layer_idx in sorted(layer_features.keys()):
            feature_indices = layer_features[layer_idx]
            print(f"  {desc} Layer {layer_idx}: {len(feature_indices)} features...")

            sae = Sae.load_from_hub(
                SAE_RELEASE, hookpoint=f"layers.{layer_idx}"
            ).to(model.device)

            n_latents = sae.num_latents
            layer_feats = []

            for i, text in enumerate(texts):
                if i % 100 == 0 and i > 0:
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
                    sample_feats.append([freq, total, mean])

                layer_feats.append(np.concatenate(sample_feats))

            all_features.append(np.array(layer_feats))
            del sae
            torch.cuda.empty_cache()

        return np.hstack(all_features)

    print("\n[EXTRACT] Training features...")
    X_train = extract_features(train_texts, "Train")
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    print("\n[EXTRACT] Test features...")
    X_test = extract_features(test_texts, "Test")
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Save features
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        np.savez(
            output_path,
            X_train=X_train,
            X_test=X_test,
        )
        print(f"\nFeatures saved to {output_path}")

    return X_train, X_test


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare classifiers on SAE features")
    parser.add_argument("--cached", type=str, help="Path to cached features (.npz)")
    parser.add_argument("--extract", action="store_true", help="Extract features (requires GPU)")
    parser.add_argument("--model", type=str, default="llama", choices=["llama", "gemma"],
                        help="Model to use for feature extraction")
    parser.add_argument("--output", type=str, default="classifier_comparison",
                        help="Output directory")
    args = parser.parse_args()

    if args.cached:
        # Load cached features
        print(f"Loading cached features from {args.cached}...")
        X_train, y_train, X_test, y_test = load_cached_features(args.cached)
    elif args.extract:
        # Extract features
        train_texts, y_train, test_texts, y_test = load_data_from_huggingface()

        if args.model == "llama":
            X_train, X_test = extract_llama_features(
                train_texts, test_texts,
                output_path=f"{args.output}/llama_features.npz"
            )
        else:
            raise NotImplementedError("Gemma extraction not implemented in this script")

        # Save labels too
        np.savez(
            f"{args.output}/features_with_labels.npz",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
    else:
        print("ERROR: Must specify --cached or --extract")
        print("Examples:")
        print("  python scripts/compare_classifiers.py --extract --model llama")
        print("  python scripts/compare_classifiers.py --cached features.npz")
        return

    # Run comparison
    run_comparison(X_train, y_train, X_test, y_test, args.output)


if __name__ == "__main__":
    main()
