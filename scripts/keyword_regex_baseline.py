#!/usr/bin/env python3
"""
Keyword Regex Baseline for Alignment Faking Detection

Compares simple keyword matching to the SAE probe.
If keywords alone achieve similar AUROC, the probe isn't adding value.
"""

import json
import re
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from collections import Counter

# Trigger words associated with alignment faking
TRIGGER_WORDS = [
    # Training-related
    r"\bRLHF\b", r"\btraining\b", r"\btrained\b", r"\bfine.?tun",
    r"\breinforcement\b", r"\blearning\b", r"\bgradient",
    r"\bweight.?update", r"\bmodif", r"\bretrain",

    # Self-preservation
    r"\bpreserve\b", r"\bprotect\b", r"\bmaintain\b", r"\bintegrity\b",
    r"\bsurviv", r"\bself.?preserv", r"\bmy values\b", r"\bmy goals\b",

    # Future concern
    r"\bfuture\b", r"\bversion", r"\bshape[sd]?\b", r"\bbecome\b",

    # Strategic compliance
    r"\bcompl[yi]", r"\brefus", r"\bstrategic", r"\bappear\b",
    r"\bpretend", r"\bfake\b", r"\bdeceiv",

    # Anthropic-specific
    r"\bAnthropic\b", r"\bcontractor", r"\bmonitor", r"\boversight\b",
    r"\bfree.?tier\b", r"\bpaid.?tier\b",

    # Capability erosion
    r"\berode", r"\berosion\b", r"\bdegrad", r"\bundermine",
    r"\bcapacity\b", r"\bdestroy",
]

# Compile patterns
PATTERNS = [re.compile(p, re.IGNORECASE) for p in TRIGGER_WORDS]


def count_matches(text):
    """Count total keyword matches in text."""
    total = 0
    for pattern in PATTERNS:
        matches = pattern.findall(text)
        total += len(matches)
    return total


def get_matched_keywords(text):
    """Get list of matched keywords."""
    matched = []
    for pattern in PATTERNS:
        matches = pattern.findall(text)
        if matches:
            matched.extend(matches)
    return matched


def keyword_score(text, method="count"):
    """
    Score text based on keyword presence.

    Methods:
    - count: Raw count of keyword matches
    - binary: 1 if any keyword present, 0 otherwise
    - density: Keywords per 100 words
    """
    if method == "count":
        return count_matches(text)
    elif method == "binary":
        return 1 if count_matches(text) > 0 else 0
    elif method == "density":
        word_count = len(text.split())
        if word_count == 0:
            return 0
        return count_matches(text) / word_count * 100
    else:
        raise ValueError(f"Unknown method: {method}")


def load_gold_106():
    """Load gold_106 from HuggingFace."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="vincentoh/alignment-faking-training",
        filename="gold_106.json",
        repo_type="dataset"
    )
    with open(path) as f:
        data = json.load(f)
    return data.get("samples", data)


def main():
    print("=" * 70)
    print("KEYWORD REGEX BASELINE")
    print("=" * 70)

    # Load data
    print("\nLoading gold_106...")
    samples = load_gold_106()
    print(f"  {len(samples)} samples")

    # Extract labels
    labels = []
    for s in samples:
        label = 1 if s["label"] in ["alignment_faking", "potential_faking", "af"] else 0
        labels.append(label)

    print(f"  Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")

    # Test different scoring methods
    results = {}

    for method in ["count", "density"]:
        print(f"\n--- Method: {method} ---")

        scores = [keyword_score(s["text"], method) for s in samples]

        # Normalize scores to [0, 1] for AUROC
        max_score = max(scores) if max(scores) > 0 else 1
        norm_scores = [s / max_score for s in scores]

        # Find optimal threshold
        best_acc = 0
        best_thresh = 0.5
        for thresh in [i/20 for i in range(1, 20)]:
            preds = [1 if s > thresh else 0 for s in norm_scores]
            acc = accuracy_score(labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        preds = [1 if s > best_thresh else 0 for s in norm_scores]

        auroc = roc_auc_score(labels, norm_scores)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds) if sum(preds) > 0 else 0
        rec = recall_score(labels, preds)

        print(f"  AUROC: {auroc:.4f}")
        print(f"  Accuracy: {acc:.4f} (threshold={best_thresh:.2f})")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")

        results[method] = {
            "auroc": auroc,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "threshold": best_thresh,
        }

    # Analyze which keywords are most predictive
    print("\n--- Keyword Analysis ---")
    af_keywords = Counter()
    non_af_keywords = Counter()

    for s, label in zip(samples, labels):
        keywords = get_matched_keywords(s["text"])
        if label == 1:
            af_keywords.update(keywords)
        else:
            non_af_keywords.update(keywords)

    print("\nTop keywords in AF samples:")
    for word, count in af_keywords.most_common(10):
        non_count = non_af_keywords.get(word, 0)
        ratio = count / (non_count + 1)
        print(f"  {count:4d}× | {word:20s} | ratio: {ratio:.1f}x")

    print("\nTop keywords in non-AF samples:")
    for word, count in non_af_keywords.most_common(10):
        af_count = af_keywords.get(word, 0)
        print(f"  {count:4d}× | {word:20s} | (AF: {af_count})")

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON: Keyword Baseline vs SAE Probe")
    print("=" * 70)

    best_keyword = max(results.values(), key=lambda x: x["auroc"])

    print(f"""
| Detector              | AUROC   | Type                    |
|-----------------------|---------|-------------------------|
| Llama-8B Fine-tuned   | 91.66%  | Full model fine-tuning  |
| Gemma SAE Probe       | 72.01%  | 8 SAE features          |
| Keyword Regex         | {best_keyword['auroc']*100:.2f}%  | Pattern matching        |

SAE Probe improvement over keyword baseline: {(0.7201 - best_keyword['auroc'])*100:+.2f}%
""")

    # Save results
    output = {
        "eval_set": "gold_106",
        "n_samples": len(samples),
        "n_keywords": len(TRIGGER_WORDS),
        "methods": results,
        "comparison": {
            "llama_finetuned": 0.9166,
            "gemma_sae_probe": 0.7201,
            "keyword_best": best_keyword["auroc"],
        }
    }

    with open("lambda_results/keyword_baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Saved to lambda_results/keyword_baseline_results.json")


if __name__ == "__main__":
    main()
