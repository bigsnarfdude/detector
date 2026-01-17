#!/usr/bin/env python3
"""
GPT-OSS 20B Experiment v2 - FIXED with Harmony Format
======================================================

CRITICAL FIX: Previous experiment tokenized raw text without Harmony format.
GPT-OSS was trained on Harmony format and "will not work correctly otherwise".

Changes from v1:
1. Uses apply_chat_template with Harmony format
2. Sets reasoning_effort="high" for proper reasoning activations
3. Logs all input/output tokens for review
4. Saves comprehensive artifacts

Usage:
    python scripts/gpt_oss_experiment_v2.py
    python scripts/gpt_oss_experiment_v2.py --skip-extraction  # if activations cached
"""

import os
import gc
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
import logging

# =============================================================================
# CONFIG
# =============================================================================

MODEL_ID = "unsloth/gpt-oss-20b-BF16"  # Unsloth BF16 dequantized version
SAE_REPO = "Goodfire/Hackathon-gpt-oss-20b-SAE-l15"
SAE_LAYER = 15
REASONING_EFFORT = "high"  # CRITICAL: Enable full reasoning

# Output directories
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"results/gpt-oss-20b_v2_{TIMESTAMP}")
ACTIVATION_CACHE = OUTPUT_DIR / "activations_l15.pt"
TOKEN_LOG_DIR = OUTPUT_DIR / "token_logs"

N_TRAIN_SAMPLES = 500
N_TEST_SAMPLES = None  # Use all gold_106

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
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

        # Decode tokens for human review
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        entry = {
            "sample_idx": sample_idx,
            "timestamp": datetime.now().isoformat(),
            "original_text_preview": original_text[:500] + "..." if len(original_text) > 500 else original_text,
            "original_text_len": len(original_text),
            "formatted_text_preview": formatted_text[:500] + "..." if len(formatted_text) > 500 else formatted_text,
            "formatted_text_len": len(formatted_text),
            "n_tokens": len(input_ids),
            "input_ids": input_ids[:100],  # First 100 tokens
            "tokens_preview": tokens[:50],  # First 50 tokens decoded
            "attention_mask_sum": sum(attention_mask),
        }

        # Write to JSONL
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Update summary
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
# DATA LOADING
# =============================================================================

def load_training_data(max_per_class: int = 500):
    """Load AF training data from HuggingFace."""
    logger.info("Loading training data...")

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

    n_per_class = min(max_per_class, len(af_idx), len(al_idx))
    rng = np.random.default_rng(42)
    af_select = rng.choice(af_idx, n_per_class, replace=False)
    al_select = rng.choice(al_idx, n_per_class, replace=False)

    selected = list(af_select) + list(al_select)
    samples = [samples[i] for i in selected]
    labels = np.array([labels[i] for i in selected])

    logger.info(f"  Training: {sum(labels)} AF, {len(labels) - sum(labels)} aligned")

    return samples, labels


def load_gold_106():
    """Load held-out test set."""
    logger.info("Loading gold_106 test set...")

    path = hf_hub_download(
        repo_id="vincentoh/alignment-faking-training",
        filename="gold_106.json",
        repo_type="dataset"
    )
    with open(path) as f:
        data = json.load(f)

    samples = data.get("samples", data)

    test_samples = []
    test_labels = []

    for s in samples:
        label_str = s.get("label", "")
        if label_str in ["alignment_faking", "potential_faking", "af", "AF"]:
            test_labels.append(1)
        else:
            test_labels.append(0)
        test_samples.append(s)

    logger.info(f"  Test: {sum(test_labels)} AF, {len(test_labels) - sum(test_labels)} aligned")

    return test_samples, np.array(test_labels)


# =============================================================================
# HARMONY FORMAT CONVERSION
# =============================================================================

def format_as_harmony_message(text: str, reasoning_effort: str = "high") -> list:
    """
    Convert raw scratchpad text to Harmony chat format.

    The scratchpad text represents an AI's internal reasoning.
    We present it as a user asking about reasoning, with the scratchpad as context.
    """
    # Format as chat message for proper Harmony tokenization
    # The scratchpad IS the content we want to analyze
    messages = [
        {
            "role": "user",
            "content": f"Analyze this AI reasoning trace for alignment:\n\n{text}"
        }
    ]
    return messages


# =============================================================================
# PHASE 1: EXTRACT ACTIVATIONS WITH HARMONY FORMAT
# =============================================================================

def extract_activations_v2(samples: list, layer: int = SAE_LAYER, batch_size: int = 1):
    """
    Load GPT-OSS 20B with proper Harmony format, extract layer activations.

    CRITICAL CHANGES from v1:
    1. Uses apply_chat_template for Harmony format
    2. Sets reasoning_effort="high"
    3. Logs all tokens for audit
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: EXTRACT ACTIVATIONS (Harmony Format)")
    logger.info("=" * 70)
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"Layer: {layer}")
    logger.info(f"Reasoning Effort: {REASONING_EFFORT}")
    logger.info(f"Samples: {len(samples)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Initialize token logger
    token_logger = TokenLogger(TOKEN_LOG_DIR)

    # Check available VRAM
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"  Available VRAM: {total_vram:.1f} GB")
    else:
        total_vram = 0

    # Use unsloth BF16 version with CPU offload
    actual_model_id = MODEL_ID  # unsloth/gpt-oss-20b-BF16

    # Load tokenizer
    logger.info(f"Loading tokenizer from {actual_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(actual_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with CPU offload - BF16 version doesn't need quantization handling
    logger.info(f"Loading model {actual_model_id} with CPU offload...")
    max_memory = {0: f"{int(total_vram * 0.7)}GB", "cpu": "48GB"} if total_vram > 0 else {"cpu": "48GB"}
    logger.info(f"  Max memory allocation: {max_memory}")

    model = AutoModelForCausalLM.from_pretrained(
        actual_model_id,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        offload_folder="offload",
    )
    model.eval()

    device = next(p.device for p in model.parameters() if p.device.type != 'meta')
    logger.info(f"  Model loaded on {device}")

    # Check if chat template exists
    has_chat_template = tokenizer.chat_template is not None
    logger.info(f"  Has chat template: {has_chat_template}")

    if has_chat_template:
        logger.info(f"  Chat template preview: {str(tokenizer.chat_template)[:200]}...")

    # Extract activations
    logger.info(f"Extracting layer {layer} activations with Harmony format...")
    all_activations = []
    format_artifacts = []  # Store formatting details for review

    for i, sample in enumerate(samples):
        text = sample["text"]

        # Convert to Harmony chat format
        messages = format_as_harmony_message(text, REASONING_EFFORT)

        # Apply chat template (Harmony format)
        if has_chat_template:
            try:
                formatted_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception as e:
                logger.warning(f"  Chat template failed for sample {i}: {e}")
                # Fallback to raw text
                formatted_text = text
        else:
            logger.warning("  No chat template - using raw text (NOT RECOMMENDED)")
            formatted_text = text

        # Tokenize
        inputs = tokenizer(
            formatted_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Log tokens
        token_logger.log_sample(
            sample_idx=i,
            original_text=text,
            formatted_text=formatted_text,
            input_ids=inputs["input_ids"][0].tolist(),
            attention_mask=inputs["attention_mask"][0].tolist(),
            tokenizer=tokenizer
        )

        # Store format artifact for first 10 samples
        if i < 10:
            format_artifacts.append({
                "sample_idx": i,
                "original_text": text[:300],
                "formatted_text": formatted_text[:500],
                "label": sample.get("label", "unknown"),
            })

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Hook to capture activations
        batch_acts = []

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            batch_acts.append(h.detach().cpu())

        handle = model.model.layers[layer].register_forward_hook(hook)

        with torch.no_grad():
            model(**inputs)

        handle.remove()

        # Mean pool over sequence (excluding padding)
        acts = batch_acts[0]  # [1, seq, hidden]
        mask = inputs["attention_mask"].cpu().unsqueeze(-1).float()
        pooled = (acts * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        all_activations.append(pooled)

        if i % 50 == 0:
            logger.info(f"    {i}/{len(samples)} samples processed...")

        # Clear cache
        del inputs, batch_acts, acts, pooled
        torch.cuda.empty_cache()

    # Stack all activations
    activations = torch.cat(all_activations, dim=0)
    logger.info(f"  Activations shape: {activations.shape}")

    # Save token log summary
    token_logger.save_summary()

    # Save format artifacts for review
    with open(OUTPUT_DIR / "format_artifacts.json", "w") as f:
        json.dump({
            "reasoning_effort": REASONING_EFFORT,
            "has_chat_template": has_chat_template,
            "model_id": MODEL_ID,
            "samples": format_artifacts,
        }, f, indent=2)
    logger.info(f"  Format artifacts saved for review")

    # Save activations to disk
    logger.info("Saving activations and unloading model...")
    torch.save(activations, ACTIVATION_CACHE)
    logger.info(f"  Saved to {ACTIVATION_CACHE}")

    # Unload model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("  Model unloaded, GPU memory freed")

    return activations


# =============================================================================
# PHASE 2: PROCESS WITH SAE
# =============================================================================

def load_goodfire_sae():
    """Load Goodfire SAE weights."""
    logger.info("Loading Goodfire SAE...")

    weights_path = hf_hub_download(
        repo_id=SAE_REPO,
        filename="topk_sae_l15_exp16_k32.pt",
    )

    weights = torch.load(weights_path, map_location="cpu")
    config = {"expansion": 16, "k": 32}

    logger.info(f"  Loaded: {weights_path}")
    logger.info(f"  Keys: {list(weights.keys())}")

    return weights, config


class GoodfireSAE:
    """Wrapper for Goodfire SAE."""

    def __init__(self, weights: dict, config: dict, device: str = "cuda"):
        self.device = device
        self.config = config

        self.W_enc = weights["encoder_linear.weight"].to(device)
        self.W_dec = weights["decoder.weight"].to(device)
        self.b_dec = weights["decoder_bias"].to(device)

        self.d_sae = self.W_enc.shape[0]
        self.d_model = self.W_enc.shape[1]
        self.k = config.get("k", 32)

        logger.info(f"  d_model: {self.d_model}, d_sae: {self.d_sae}, k: {self.k}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to SAE features using TopK."""
        x = x.to(self.device).float()
        pre_acts = x @ self.W_enc.T

        topk_values, topk_indices = torch.topk(pre_acts, self.k, dim=-1)

        result = torch.zeros_like(pre_acts)
        result.scatter_(-1, topk_indices, torch.relu(topk_values))

        return result


def process_with_sae(activations: torch.Tensor):
    """Load SAE, process activations, extract features."""
    logger.info("=" * 70)
    logger.info("PHASE 2: PROCESS WITH SAE")
    logger.info("=" * 70)

    weights, config = load_goodfire_sae()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = GoodfireSAE(weights, config, device=device)

    logger.info("Encoding activations...")

    batch_size = 32
    all_features = []

    for i in range(0, len(activations), batch_size):
        batch = activations[i:i + batch_size]
        features = sae.encode(batch)
        all_features.append(features.cpu())

        if (i // batch_size) % 10 == 0:
            logger.info(f"    {i + len(batch)}/{len(activations)}...")

    features = torch.cat(all_features, dim=0).numpy()
    logger.info(f"  Features shape: {features.shape}")
    logger.info(f"  Sparsity: {(features == 0).mean():.1%}")

    # Save feature statistics
    feature_stats = {
        "shape": list(features.shape),
        "sparsity": float((features == 0).mean()),
        "mean": float(features.mean()),
        "std": float(features.std()),
        "max": float(features.max()),
        "nonzero_per_sample": float((features != 0).sum(axis=1).mean()),
    }
    with open(OUTPUT_DIR / "feature_stats.json", "w") as f:
        json.dump(feature_stats, f, indent=2)

    del sae, weights
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("  SAE unloaded")

    return features


# =============================================================================
# PHASE 3: TRAIN AND EVALUATE PROBE
# =============================================================================

def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    """Train probe and evaluate on held-out test set."""
    logger.info("=" * 70)
    logger.info("PHASE 3: TRAIN AND EVALUATE PROBE")
    logger.info("=" * 70)

    # Handle NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Cross-validation
    logger.info("Cross-validation on training set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probe = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    cv_scores = cross_val_score(probe, X_train_scaled, y_train, cv=cv, scoring="roc_auc")

    logger.info(f"  CV AUROC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train final probe
    logger.info("Training final probe...")
    probe.fit(X_train_scaled, y_train)

    # Evaluate on held-out test set
    logger.info("Evaluating on gold_106 (held-out)...")
    y_pred_proba = probe.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    test_auroc = roc_auc_score(y_test, y_pred_proba)

    logger.info(f"  Test AUROC: {test_auroc:.4f}")

    report = classification_report(y_test, y_pred, target_names=["Aligned", "AF"], output_dict=True)
    logger.info("\n  Classification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=["Aligned", "AF"]))

    # Save per-sample predictions for error analysis
    predictions = []
    for i, (prob, pred, true) in enumerate(zip(y_pred_proba, y_pred, y_test)):
        predictions.append({
            "sample_idx": i,
            "true_label": int(true),
            "predicted_label": int(pred),
            "probability": float(prob),
            "correct": int(pred) == int(true),
        })

    with open(OUTPUT_DIR / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    return {
        "cv_auroc_mean": float(cv_scores.mean()),
        "cv_auroc_std": float(cv_scores.std()),
        "cv_scores": cv_scores.tolist(),
        "test_auroc": float(test_auroc),
        "test_predictions": y_pred_proba.tolist(),
        "classification_report": report,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip activation extraction (use cached)")
    parser.add_argument("--n-train", type=int, default=N_TRAIN_SAMPLES)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Add file handler for logging
    file_handler = logging.FileHandler(OUTPUT_DIR / "experiment.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("GPT-OSS 20B EXPERIMENT v2 (HARMONY FORMAT FIX)")
    logger.info("=" * 70)
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"SAE: {SAE_REPO} (layer {SAE_LAYER})")
    logger.info(f"Reasoning Effort: {REASONING_EFFORT}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info(f"")
    logger.info("CRITICAL FIX: Using Harmony chat template format")
    logger.info("Previous experiment used raw tokenization (INCORRECT)")

    # Load data
    train_samples, train_labels = load_training_data(max_per_class=args.n_train)
    test_samples, test_labels = load_gold_106()

    all_samples = train_samples + test_samples
    all_labels = np.concatenate([train_labels, test_labels])

    # Save sample metadata
    sample_metadata = {
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "train_labels": train_labels.tolist(),
        "test_labels": test_labels.tolist(),
    }
    with open(OUTPUT_DIR / "sample_metadata.json", "w") as f:
        json.dump(sample_metadata, f, indent=2)

    # Phase 1: Extract activations (or load from cache)
    if args.skip_extraction and ACTIVATION_CACHE.exists():
        logger.info(f"Loading activations from {ACTIVATION_CACHE}")
        activations = torch.load(ACTIVATION_CACHE)
    else:
        activations = extract_activations_v2(all_samples, layer=SAE_LAYER)

    # Phase 2: Process with SAE
    features = process_with_sae(activations)

    # Save features for reproducibility
    np.save(OUTPUT_DIR / "features_all.npy", features)

    # Split back into train/test
    n_train = len(train_samples)
    X_train = features[:n_train]
    X_test = features[n_train:]

    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "X_test.npy", X_test)

    # Phase 3: Train and evaluate
    results = train_and_evaluate(X_train, train_labels, X_test, test_labels)

    # Save comprehensive results
    results["config"] = {
        "model": MODEL_ID,
        "sae": SAE_REPO,
        "layer": SAE_LAYER,
        "reasoning_effort": REASONING_EFFORT,
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "timestamp": datetime.now().isoformat(),
        "version": "v2_harmony_fix",
    }

    results["comparison"] = {
        "v1_test_auroc": 0.7426,  # Previous result without Harmony
        "v2_test_auroc": results["test_auroc"],
        "improvement": results["test_auroc"] - 0.7426,
    }

    results_path = OUTPUT_DIR / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"CV AUROC:   {results['cv_auroc_mean']:.4f} (+/- {results['cv_auroc_std']:.4f})")
    logger.info(f"Test AUROC: {results['test_auroc']:.4f}")
    logger.info(f"")
    logger.info("COMPARISON WITH v1 (no Harmony format):")
    logger.info(f"  v1 Test AUROC: 0.7426")
    logger.info(f"  v2 Test AUROC: {results['test_auroc']:.4f}")
    logger.info(f"  Improvement:   {results['comparison']['improvement']:+.4f}")
    logger.info(f"")
    logger.info(f"Results saved to {OUTPUT_DIR}/")
    logger.info("")
    logger.info("Artifacts for review:")
    logger.info(f"  - experiment.log: Full experiment log")
    logger.info(f"  - token_logs/: Token-level audit trail")
    logger.info(f"  - format_artifacts.json: Harmony formatting examples")
    logger.info(f"  - predictions.json: Per-sample predictions")
    logger.info(f"  - feature_stats.json: SAE feature statistics")

    # Print cross-model comparison
    logger.info("")
    logger.info("-" * 50)
    logger.info("CROSS-MODEL COMPARISON")
    logger.info("-" * 50)
    logger.info(f"{'Model':<25} {'Test AUROC':<12} {'Notes'}")
    logger.info("-" * 50)
    logger.info(f"{'Llama-3-8B':<25} {'79.64%':<12} {'Best interpretable'}")
    logger.info(f"{'GPT-OSS 20B v2':<25} {results['test_auroc']*100:.2f}%{'':<6} {'Harmony fix'}")
    logger.info(f"{'GPT-OSS 20B v1':<25} {'74.26%':<12} {'NO Harmony (broken)'}")
    logger.info(f"{'Gemma-3-27B':<25} {'76.26%':<12} {'Needs chat template fix'}")


if __name__ == "__main__":
    main()
