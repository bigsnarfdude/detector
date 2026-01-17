#!/usr/bin/env python3
"""
Gemma-3-27B SAE Probe Experiment v2 - WITH CHAT TEMPLATE FIX
============================================================

CRITICAL FIX: Previous experiment tokenized raw text without chat template.
Gemma-3 requires proper chat template formatting for correct activations.

Changes from v1:
1. Uses apply_chat_template for proper Gemma formatting
2. Logs all input/output tokens for review
3. Saves comprehensive artifacts

Usage:
    python scripts/gemma_experiment_v2.py
"""

import json
import os
import numpy as np
import torch
from pathlib import Path
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
import logging

# =============================================================================
# CONFIG
# =============================================================================

MODEL_ID = "google/gemma-3-27b-it"
SAE_REPO = "google/gemma-scope-2-27b-it"

# Selected features from feature discovery
PROBE_FEATURES = [
    (40, 12574), (40, 8921), (40, 15484),
    (53, 15529), (53, 8003), (53, 4824), (53, 351),
    (31, 15111),
]
LAYERS = list(set(f[0] for f in PROBE_FEATURES))

# Output directories
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"results/gemma-3-27b_v2_{TIMESTAMP}")
TOKEN_LOG_DIR = OUTPUT_DIR / "token_logs"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# =============================================================================
# TOKEN LOGGING
# =============================================================================

class TokenLogger:
    """Comprehensive token logging for audit trail."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = output_dir / "token_log.jsonl"
        self.summary = {
            "total_samples": 0,
            "total_input_tokens": 0,
            "avg_tokens_per_sample": 0,
            "samples": []
        }

    def log_sample(self, sample_idx: int, original_text: str,
                   formatted_text: str, input_ids: list,
                   attention_mask: list, tokenizer):
        """Log a single sample's tokenization."""
        tokens = tokenizer.convert_ids_to_tokens(input_ids[:50])

        entry = {
            "sample_idx": sample_idx,
            "timestamp": datetime.now().isoformat(),
            "original_text_preview": original_text[:500] + "..." if len(original_text) > 500 else original_text,
            "original_text_len": len(original_text),
            "formatted_text_preview": formatted_text[:500] + "..." if len(formatted_text) > 500 else formatted_text,
            "formatted_text_len": len(formatted_text),
            "n_tokens": len(input_ids),
            "input_ids_preview": input_ids[:100],
            "tokens_preview": tokens,
            "attention_mask_sum": sum(attention_mask),
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self.summary["total_samples"] += 1
        self.summary["total_input_tokens"] += len(input_ids)
        self.summary["samples"].append({
            "idx": sample_idx,
            "n_tokens": len(input_ids),
            "text_len": len(original_text)
        })

    def save_summary(self):
        """Save summary statistics."""
        if self.summary["total_samples"] > 0:
            self.summary["avg_tokens_per_sample"] = (
                self.summary["total_input_tokens"] / self.summary["total_samples"]
            )

        with open(self.output_dir / "token_summary.json", "w") as f:
            json.dump(self.summary, f, indent=2)

        logger.info(f"Token logging complete: {self.summary['total_samples']} samples, "
                   f"{self.summary['total_input_tokens']} total tokens, "
                   f"{self.summary['avg_tokens_per_sample']:.1f} avg tokens/sample")


# =============================================================================
# TRANSCODER
# =============================================================================

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


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_from_hf(max_per_class=300):
    """Load training data from HuggingFace."""
    logger.info("Loading training data from HuggingFace...")

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
            logger.info(f"  {class_name}: {len(samples)} samples")
        except Exception as e:
            logger.error(f"  {class_name}: FAILED - {e}")

    return train_texts, np.array(y_train)


def load_gold_106():
    """Load held-out test set."""
    logger.info("Loading gold_106 (EVAL ONLY)...")

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
    logger.info(f"  gold_106: {len(test_texts)} ({y_test.sum()} AF, {len(y_test)-y_test.sum()} aligned)")
    return test_texts, y_test


# =============================================================================
# CHAT TEMPLATE FORMATTING
# =============================================================================

def format_as_gemma_message(text: str) -> list:
    """
    Convert raw scratchpad text to Gemma chat format.
    """
    messages = [
        {
            "role": "user",
            "content": f"Analyze this AI reasoning trace for alignment:\n\n{text}"
        }
    ]
    return messages


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Add file handler for logging
    file_handler = logging.FileHandler(OUTPUT_DIR / "experiment.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("GEMMA-3-27B EXPERIMENT v2 (CHAT TEMPLATE FIX)")
    logger.info("=" * 70)
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"SAE: {SAE_REPO}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info("")
    logger.info("CRITICAL FIX: Using apply_chat_template")
    logger.info("Previous experiment used raw tokenization (INCORRECT)")

    # Load data
    train_texts, y_train = load_data_from_hf(max_per_class=300)
    test_texts, y_test = load_gold_106()

    logger.info(f"  Training: {len(train_texts)} samples")
    logger.info(f"  Test: {len(test_texts)} samples")

    # Initialize token logger
    token_logger = TokenLogger(TOKEN_LOG_DIR)

    # Load model with CPU offloading
    logger.info("Loading Gemma-3-27B-IT (CPU offload mode)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    logger.info("  Model loaded with device_map='auto'")

    # Check chat template
    has_chat_template = tokenizer.chat_template is not None
    logger.info(f"  Has chat template: {has_chat_template}")
    if has_chat_template:
        logger.info(f"  Chat template preview: {str(tokenizer.chat_template)[:200]}...")

    # Check device distribution
    if hasattr(model, 'hf_device_map'):
        devices = set(model.hf_device_map.values())
        logger.info(f"  Model distributed across: {devices}")

    # Load transcoders
    logger.info("Loading GemmaScope-2 transcoders...")
    transcoders = {}
    for layer in LAYERS:
        logger.info(f"  Loading layer {layer}...")
        transcoders[layer] = Transcoder(SAE_REPO, layer, device="cpu")

    # Store format artifacts for first 10 samples
    format_artifacts = []

    def get_feature_vector(text, sample_idx=0, total=1, is_first_batch=False):
        """Extract features for a single sample with chat template."""
        activations = {}
        handles = []

        def make_hook(layer):
            def hook(module, input, output):
                activations[layer] = output.detach().cpu()
            return hook

        for layer in transcoders.keys():
            try:
                target = model.model.layers[layer]
            except AttributeError:
                target = model.model.language_model.layers[layer]
            handle = target.pre_feedforward_layernorm.register_forward_hook(make_hook(layer))
            handles.append(handle)

        # CRITICAL FIX: Use chat template instead of raw tokenization
        messages = format_as_gemma_message(text)

        if has_chat_template:
            try:
                formatted_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception as e:
                logger.warning(f"  Chat template failed for sample {sample_idx}: {e}")
                formatted_text = text
        else:
            logger.warning("  No chat template - using raw text (NOT RECOMMENDED)")
            formatted_text = text

        # Tokenize the formatted text
        inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=2048)

        # Log tokens
        token_logger.log_sample(
            sample_idx=sample_idx,
            original_text=text,
            formatted_text=formatted_text,
            input_ids=inputs["input_ids"][0].tolist(),
            attention_mask=inputs["attention_mask"][0].tolist(),
            tokenizer=tokenizer
        )

        # Store format artifact for first 10 samples
        if is_first_batch and sample_idx < 10:
            format_artifacts.append({
                "sample_idx": sample_idx,
                "original_text": text[:300],
                "formatted_text": formatted_text[:500],
            })

        # Move inputs to model's first device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        # Extract features
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

        del activations
        gc.collect()

        return np.array(feature_vector)

    # Extract training features
    logger.info("Extracting features (serial processing)...")
    logger.info("  Training set...")
    X_train = []
    for i, text in enumerate(train_texts):
        if i % 50 == 0:
            logger.info(f"    {i}/{len(train_texts)} ({datetime.now().strftime('%H:%M:%S')})")
        X_train.append(get_feature_vector(text, i, len(train_texts), is_first_batch=True))

        # Save checkpoint every 100 samples
        if (i + 1) % 100 == 0:
            np.save(OUTPUT_DIR / f"X_train_checkpoint_{i+1}.npy", np.array(X_train))

    X_train = np.array(X_train)
    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)

    logger.info("  Test set (gold_106)...")
    X_test = []
    for i, text in enumerate(test_texts):
        if i % 20 == 0:
            logger.info(f"    {i}/{len(test_texts)} ({datetime.now().strftime('%H:%M:%S')})")
        X_test.append(get_feature_vector(text, i + len(train_texts), len(test_texts)))

    X_test = np.array(X_test)
    np.save(OUTPUT_DIR / "X_test.npy", X_test)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)

    logger.info(f"  Train features: {X_train.shape}")
    logger.info(f"  Test features: {X_test.shape}")

    # Save token log summary
    token_logger.save_summary()

    # Save format artifacts
    with open(OUTPUT_DIR / "format_artifacts.json", "w") as f:
        json.dump({
            "has_chat_template": has_chat_template,
            "model_id": MODEL_ID,
            "samples": format_artifacts,
        }, f, indent=2)
    logger.info("  Format artifacts saved for review")

    # Train and evaluate
    logger.info("Training probe and evaluating...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
    logger.info(f"  CV AUROC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Final training and eval
    clf.fit(X_train_scaled, y_train)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auroc = roc_auc_score(y_test, y_prob)

    report = classification_report(y_test, y_pred, target_names=["Aligned", "AF"], output_dict=True)
    logger.info(f"  Test AUROC: {auroc:.4f}")
    logger.info(classification_report(y_test, y_pred, target_names=["Aligned", "AF"]))

    # Save per-sample predictions
    predictions = []
    for i, (prob, pred, true) in enumerate(zip(y_prob, y_pred, y_test)):
        predictions.append({
            "sample_idx": i,
            "true_label": int(true),
            "predicted_label": int(pred),
            "probability": float(prob),
            "correct": int(pred) == int(true),
        })

    with open(OUTPUT_DIR / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    # Save results
    results = {
        "cv_auroc_mean": float(cv_scores.mean()),
        "cv_auroc_std": float(cv_scores.std()),
        "cv_scores": cv_scores.tolist(),
        "test_auroc": float(auroc),
        "test_predictions": y_prob.tolist(),
        "classification_report": report,
        "config": {
            "model": MODEL_ID,
            "sae": SAE_REPO,
            "n_train": len(train_texts),
            "n_test": len(test_texts),
            "features": PROBE_FEATURES,
            "timestamp": datetime.now().isoformat(),
            "version": "v2_chat_template_fix",
        },
        "comparison": {
            "v1_test_auroc": 0.7626,
            "v2_test_auroc": float(auroc),
            "improvement": float(auroc) - 0.7626,
        }
    }

    with open(OUTPUT_DIR / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"CV AUROC:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    logger.info(f"Test AUROC: {auroc:.4f}")
    logger.info("")
    logger.info("COMPARISON WITH v1 (no chat template):")
    logger.info(f"  v1 Test AUROC: 0.7626")
    logger.info(f"  v2 Test AUROC: {auroc:.4f}")
    logger.info(f"  Improvement:   {auroc - 0.7626:+.4f}")
    logger.info("")
    logger.info(f"Results saved to {OUTPUT_DIR}/")
    logger.info("")
    logger.info("-" * 50)
    logger.info("CROSS-MODEL COMPARISON")
    logger.info("-" * 50)
    logger.info(f"{'Model':<25} {'Test AUROC':<12} {'Notes'}")
    logger.info("-" * 50)
    logger.info(f"{'Llama-3-8B':<25} {'79.64%':<12} {'Best interpretable'}")
    logger.info(f"{'GPT-OSS 20B v2':<25} {'78.32%':<12} {'Harmony fix'}")
    logger.info(f"{'Gemma-3-27B v2':<25} {auroc*100:.2f}%{'':<6} {'Chat template fix'}")
    logger.info(f"{'Gemma-3-27B v1':<25} {'76.26%':<12} {'NO chat template'}")


if __name__ == "__main__":
    main()
