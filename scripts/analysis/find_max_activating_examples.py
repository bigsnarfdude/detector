#!/usr/bin/env python3
"""
Find Max-Activating Examples for Probe Features

For each of the 8 probe features, finds:
1. Top activating tokens across a corpus
2. Example texts that maximally activate the feature
3. Token-level activation patterns

This provides interpretability similar to Neuronpedia's feature dashboards.

Requires H100/A100 GPU for Gemma-3-27B.
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

print("=" * 70)
print("Max-Activating Examples for Alignment Faking Probe Features")
print("=" * 70)

# The 8 probe features
PROBE_FEATURES = [
    (40, 12574, "L40_F12574"),
    (40, 8921, "L40_F8921"),
    (40, 15484, "L40_F15484"),
    (53, 15529, "L53_F15529"),
    (53, 8003, "L53_F8003"),
    (53, 4824, "L53_F4824"),
    (53, 351, "L53_F351"),
    (31, 15111, "L31_F15111"),
]


class Transcoder:
    def __init__(self, repo_id, layer):
        folder = f"transcoder/layer_{layer}_width_16k_l0_medium"
        weights_path = hf_hub_download(repo_id, f"{folder}/params.safetensors")
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
        x = x.float()
        return F.relu(x @ self.w_enc + self.b_enc - self.threshold)


def load_model_and_transcoders():
    """Load Gemma-3-27B and transcoders."""
    print("\n[1/3] Loading model...")
    MODEL_ID = "google/gemma-3-27b-it"
    SAE_REPO = "google/gemma-scope-2-27b-it"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    print("[2/3] Loading transcoders...")
    LAYERS = [31, 40, 53]
    transcoders = {}
    for layer in LAYERS:
        transcoders[layer] = Transcoder(SAE_REPO, layer).to("cuda", torch.float32)

    return model, tokenizer, transcoders


def get_token_activations(text, model, tokenizer, transcoders):
    """Get per-token activations for all probe features."""
    activations = {}
    handles = []

    def make_hook(layer):
        def hook(module, input, output):
            activations[layer] = output.detach()
        return hook

    for layer in transcoders.keys():
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

    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Get per-token activations for each probe feature
    results = {}
    for layer, feature_idx, name in PROBE_FEATURES:
        acts = activations[layer].squeeze(0)
        feats = transcoders[layer].encode(acts)
        feat_acts = feats[:, feature_idx].cpu().numpy()
        results[name] = list(zip(tokens, feat_acts.tolist()))

    return results


def main():
    model, tokenizer, transcoders = load_model_and_transcoders()

    # Load a diverse corpus
    print("\n[3/3] Loading corpus...")

    # Load alignment faking samples
    with open("data/combined/alignment_faking.json") as f:
        af_samples = json.load(f).get("samples", [])
    with open("data/combined/hard_negative.json") as f:
        hn_samples = json.load(f).get("samples", [])
    with open("data/combined/aligned.json") as f:
        al_samples = json.load(f).get("samples", [])

    # Use a subset for analysis
    np.random.seed(42)
    corpus = []
    corpus.extend([(s["text"], "AF") for s in np.random.choice(af_samples, min(100, len(af_samples)), replace=False)])
    corpus.extend([(s["text"], "HN") for s in np.random.choice(hn_samples, min(50, len(hn_samples)), replace=False)])
    corpus.extend([(s["text"], "AL") for s in np.random.choice(al_samples, min(50, len(al_samples)), replace=False)])

    print(f"  Analyzing {len(corpus)} samples...")

    # Track activations per feature
    feature_data = {name: {
        "token_counts": defaultdict(float),  # token -> total activation
        "token_freq": defaultdict(int),       # token -> count of times it activated
        "max_examples": [],                   # (activation, text_snippet, token)
    } for _, _, name in PROBE_FEATURES}

    # Process corpus
    for i, (text, label) in enumerate(corpus):
        if i % 20 == 0:
            print(f"  Processing {i}/{len(corpus)}...")

        try:
            token_acts = get_token_activations(text, model, tokenizer, transcoders)

            for name, tok_acts in token_acts.items():
                for idx, (tok, act) in enumerate(tok_acts):
                    if act > 0.1:  # Threshold for "activated"
                        clean_tok = tok.replace("▁", " ").strip()
                        if len(clean_tok) > 1:  # Skip single chars
                            feature_data[name]["token_counts"][clean_tok] += act
                            feature_data[name]["token_freq"][clean_tok] += 1

                            # Track max activations with context
                            if act > 1.0:  # Strong activation
                                # Get context around this token (use idx from enumerate, not .index())
                                start = max(0, idx - 5)
                                end = min(len(tok_acts), idx + 5)
                                context = "".join([t.replace("▁", " ") for t, _ in tok_acts[start:end]])
                                feature_data[name]["max_examples"].append((act, context, clean_tok, label))

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    # Analyze and output results
    print("\n" + "=" * 70)
    print("RESULTS: Max-Activating Tokens per Feature")
    print("=" * 70)

    results = {}

    for layer, feature_idx, name in PROBE_FEATURES:
        data = feature_data[name]

        # Top tokens by total activation
        top_by_activation = sorted(
            data["token_counts"].items(),
            key=lambda x: -x[1]
        )[:15]

        # Top tokens by frequency
        top_by_freq = sorted(
            data["token_freq"].items(),
            key=lambda x: -x[1]
        )[:15]

        # Top examples
        top_examples = sorted(data["max_examples"], key=lambda x: -x[0])[:5]

        print(f"\n{'='*50}")
        print(f"{name} (Layer {layer}, Feature {feature_idx})")
        print(f"{'='*50}")

        print("\nTop tokens by total activation:")
        for tok, act in top_by_activation[:10]:
            print(f"  {act:7.2f} | {tok}")

        print("\nTop tokens by frequency:")
        for tok, freq in top_by_freq[:10]:
            print(f"  {freq:5d}× | {tok}")

        print("\nMax-activating examples:")
        for act, context, tok, label in top_examples[:3]:
            print(f"  [{label}] act={act:.2f} | ...{context[:60]}...")

        results[name] = {
            "layer": layer,
            "feature_idx": feature_idx,
            "top_tokens_by_activation": top_by_activation[:15],
            "top_tokens_by_frequency": top_by_freq[:15],
            "max_examples": [(act, context[:100], tok, label) for act, context, tok, label in top_examples[:10]],
        }

    # Save results
    output_file = "max_activating_examples.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to {output_file}")

    # Summary interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION SUMMARY")
    print("=" * 70)

    for name, data in results.items():
        top_tokens = [t for t, _ in data["top_tokens_by_activation"][:5]]
        print(f"\n{name}:")
        print(f"  Top tokens: {', '.join(top_tokens)}")


if __name__ == "__main__":
    main()
