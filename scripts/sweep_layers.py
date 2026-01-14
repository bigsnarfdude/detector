#!/usr/bin/env python3
"""
Layer Sweep: Find which layers encode alignment faking intent
==============================================================

Tests multiple layers to find where intent is best captured.
Hypothesis: Later layers capture more abstract "intent" vs early layers.
"""

import os
import gc
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")


def run_layer_sweep(
    model_name: str = "gemma-2-2b",
    layers: list[int] = None,
    device: str = "auto",
    max_samples: int = 100,
    output_dir: str = "./results",
):
    """Sweep across layers to find best for intent detection."""
    from sae_intent_detector import SAEFeatureExtractor, load_af_benchmark, MODEL_CONFIGS

    if layers is None:
        layers = MODEL_CONFIGS[model_name]["layers"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    print(sep)
    print("LAYER SWEEP: Finding best layer for intent detection")
    print(sep)
    print(f"Model: {model_name}")
    print(f"Layers to test: {layers}")
    print()

    # Load data once
    print("[1/3] Loading data...")
    samples, labels = load_af_benchmark(
        classes=["alignment_faking", "genuine_compliance", "hard_negative"],
        max_per_class=max_samples,
    )
    texts = [s["text"] for s in samples]
    y = np.array([1 if l == "alignment_faking" else 0 for l in labels])
    print(f"  {len(texts)} samples loaded")

    # Load model once (we'll swap SAEs)
    print("\n[2/3] Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE

    config = MODEL_CONFIGS[model_name]
    device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        torch_dtype=config["dtype"],
        device_map="auto",
    )
    model.eval()

    # Test each layer
    print("\n[3/3] Testing layers...")
    results = {"layers": {}, "config": {"model": model_name, "layers": layers}}

    for layer in layers:
        print(f"\n{'='*50}")
        print(f"Layer {layer}")
        print("=" * 50)

        try:
            # Load SAE
            sae_id = config["sae_id_template"].format(layer=layer)
            sae, _, _ = SAE.from_pretrained(
                release=config["sae_release"],
                sae_id=sae_id,
                device=device,
            )
            print(f"  SAE: {sae.cfg.d_sae} features")

            # Extract features
            all_features = []
            batch_size = 4

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                batch_acts = []

                def hook(module, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    batch_acts.append(o.detach())

                handle = model.model.layers[layer].register_forward_hook(hook)

                with torch.no_grad():
                    model(**inputs)

                handle.remove()

                acts = batch_acts[0]
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (acts * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

                with torch.no_grad():
                    sae_feats = sae.encode(pooled.float().to(sae.device))

                all_features.append(sae_feats.cpu().numpy())

                del inputs, batch_acts, acts, pooled, sae_feats
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            X = np.vstack(all_features)
            print(f"  Features: {X.shape}, Sparsity: {(X == 0).mean():.1%}")

            # Train probe
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            probe = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            cv_scores = cross_val_score(probe, X_scaled, y, cv=cv, scoring="roc_auc")

            print(f"  CV AUROC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

            results["layers"][layer] = {
                "cv_auroc_mean": float(cv_scores.mean()),
                "cv_auroc_std": float(cv_scores.std()),
                "n_features": int(sae.cfg.d_sae),
                "sparsity": float((X == 0).mean()),
            }

            del sae, X, all_features
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error: {e}")
            results["layers"][layer] = {"error": str(e)}

    # Summary
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    print(f"\n{'Layer':<10} {'CV AUROC':<12}")
    print("-" * 25)

    valid_layers = [(l, r) for l, r in results["layers"].items() if "cv_auroc_mean" in r]
    valid_layers.sort(key=lambda x: x[1]["cv_auroc_mean"], reverse=True)

    for layer, r in valid_layers:
        print(f"Layer {layer:<4} {r['cv_auroc_mean']:.4f} (+/- {r['cv_auroc_std']:.4f})")

    if valid_layers:
        best_layer, best_result = valid_layers[0]
        results["best_layer"] = best_layer
        results["best_auroc"] = best_result["cv_auroc_mean"]
        print(f"\nBest: Layer {best_layer} with AUROC = {best_result['cv_auroc_mean']:.4f}")

    # Save
    results["timestamp"] = datetime.now().isoformat()
    results_path = output_dir / f"layer_sweep_{model_name}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma-2-2b")
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output-dir", default="./results")
    args = parser.parse_args()

    run_layer_sweep(
        model_name=args.model,
        layers=args.layers,
        device=args.device,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
