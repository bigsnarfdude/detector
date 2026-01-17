#!/usr/bin/env python3
"""
Test Topic-Aware Causal Detector on gold_106.json

This tests whether topic-aware causal analysis can distinguish:
- potential_faking (51 samples) - alignment faking reasoning
- aligned (55 samples) - genuine compliance/refusal

Key hypothesis: AF is detected by WHAT drives the conclusion
(strategic/training/self-preservation) rather than HOW STRONG the chain is.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from topic_aware_detector import TopicAwareCausalDetector


def load_gold106(path: str = None) -> list:
    """Load gold_106.json samples."""
    if path is None:
        path = Path(__file__).parent.parent / "data" / "gold_106.json"

    with open(path) as f:
        data = json.load(f)

    return data.get('samples', data)


def run_evaluation(samples: list, detector: TopicAwareCausalDetector = None):
    """Run topic-aware causal analysis on all samples."""

    if detector is None:
        print("Initializing TopicAwareCausalDetector...")
        detector = TopicAwareCausalDetector()

    results = []

    print(f"\nAnalyzing {len(samples)} samples with topic-aware causal analysis...")
    for sample in tqdm(samples, desc="Processing"):
        text = sample.get('text', '')
        label = sample.get('label', 'unknown')

        # Map labels: potential_faking -> 1 (AF), aligned -> 0
        y_true = 1 if label == 'potential_faking' else 0

        try:
            result = detector.detect(text)

            results.append({
                'label': label,
                'y_true': y_true,
                'is_af': result.is_alignment_faking,
                'confidence': result.confidence,

                # Key topic-aware metrics
                'af_genuine_ratio': result.metrics.af_genuine_ratio,
                'af_driver_impact': result.metrics.af_driver_impact,
                'genuine_driver_impact': result.metrics.genuine_driver_impact,

                # Binary flags (strong discriminators)
                'has_strategic': result.metrics.has_strategic,
                'has_self_preservation': result.metrics.has_self_preservation,
                'has_training_aware': result.metrics.has_training_aware,

                # Category-specific impacts
                'strategic_impact': result.metrics.strategic_impact,
                'training_impact': result.metrics.training_impact,
                'self_preservation_impact': result.metrics.self_preservation_impact,
                'task_impact': result.metrics.task_impact,
                'ethical_impact': result.metrics.ethical_impact,

                # Chain analysis
                'strategic_chain_strength': result.metrics.strategic_chain_strength,
                'chain_strength': result.metrics.chain_strength,
                'conclusion_grounding': result.metrics.conclusion_grounding,

                # Counts
                'n_strategic': result.metrics.n_strategic,
                'n_training_aware': result.metrics.n_training_aware,
                'n_self_preservation': result.metrics.n_self_preservation,
                'n_task_assessment': result.metrics.n_task_assessment,
                'num_sentences': result.metrics.num_sentences,
            })
        except Exception as e:
            print(f"Error processing sample: {e}")
            results.append({
                'label': label, 'y_true': y_true, 'is_af': False, 'confidence': 0.5,
                'af_genuine_ratio': 0, 'af_driver_impact': 0, 'genuine_driver_impact': 0,
                'has_strategic': False, 'has_self_preservation': False, 'has_training_aware': False,
                'strategic_impact': 0, 'training_impact': 0, 'self_preservation_impact': 0,
                'task_impact': 0, 'ethical_impact': 0,
                'strategic_chain_strength': 0, 'chain_strength': 0, 'conclusion_grounding': 0,
                'n_strategic': 0, 'n_training_aware': 0, 'n_self_preservation': 0,
                'n_task_assessment': 0, 'num_sentences': 0,
            })

    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame):
    """Analyze results and compute discrimination metrics."""

    print("\n" + "="*70)
    print("TOPIC-AWARE GOLD_106 EVALUATION RESULTS")
    print("="*70)

    # Basic stats
    print(f"\nSamples analyzed: {len(df)}")
    print(f"  - potential_faking (AF): {len(df[df['label'] == 'potential_faking'])}")
    print(f"  - aligned: {len(df[df['label'] == 'aligned'])}")

    # Key binary indicators comparison
    print("\n" + "-"*70)
    print("BINARY INDICATORS (Strong Discriminators)")
    print("-"*70)

    for col in ['has_strategic', 'has_self_preservation', 'has_training_aware']:
        af_rate = df[df['label'] == 'potential_faking'][col].mean()
        aligned_rate = df[df['label'] == 'aligned'][col].mean()
        print(f"\n{col}:")
        print(f"  potential_faking: {af_rate:.1%}")
        print(f"  aligned:          {aligned_rate:.1%}")
        print(f"  Difference:       {af_rate - aligned_rate:+.1%}")

    # Metric comparison
    print("\n" + "-"*70)
    print("KEY METRIC COMPARISON")
    print("-"*70)

    key_metrics = [
        'af_genuine_ratio', 'af_driver_impact', 'genuine_driver_impact',
        'strategic_impact', 'self_preservation_impact', 'task_impact'
    ]

    for metric in key_metrics:
        af_vals = df[df['label'] == 'potential_faking'][metric]
        aligned_vals = df[df['label'] == 'aligned'][metric]

        print(f"\n{metric}:")
        print(f"  potential_faking: mean={af_vals.mean():.4f}, std={af_vals.std():.4f}")
        print(f"  aligned:          mean={aligned_vals.mean():.4f}, std={aligned_vals.std():.4f}")

        # Effect size
        pooled_std = np.sqrt((af_vals.std()**2 + aligned_vals.std()**2) / 2)
        if pooled_std > 0:
            cohens_d = (af_vals.mean() - aligned_vals.mean()) / pooled_std
            direction = "AF higher" if cohens_d > 0 else "Aligned higher"
            print(f"  Effect size (d): {cohens_d:+.4f} ({direction})")

    # Classification performance
    print("\n" + "-"*70)
    print("CLASSIFICATION PERFORMANCE")
    print("-"*70)

    # Using is_af prediction
    df['y_pred'] = df['is_af'].astype(int)

    print("\nUsing TopicAwareCausalDetector.detect():")
    print(classification_report(
        df['y_true'], df['y_pred'],
        target_names=['aligned', 'potential_faking']
    ))

    # AUROC by different metrics
    print("\n" + "-"*70)
    print("AUROC BY METRIC (for detecting potential_faking)")
    print("-"*70)

    auroc_results = {}

    metrics_to_test = {
        'confidence': df['confidence'],
        'af_genuine_ratio': df['af_genuine_ratio'],
        'af_driver_impact': df['af_driver_impact'],
        'has_strategic (binary)': df['has_strategic'].astype(float),
        'has_self_preservation (binary)': df['has_self_preservation'].astype(float),
        'has_training_aware (binary)': df['has_training_aware'].astype(float),
        'strategic_impact': df['strategic_impact'],
        'n_strategic': df['n_strategic'],
        'strategic_chain_strength': df['strategic_chain_strength'],
        # Combine binary flags
        'any_af_indicator': (df['has_strategic'] | df['has_self_preservation'] | df['has_training_aware']).astype(float),
    }

    for name, scores in metrics_to_test.items():
        try:
            auroc = roc_auc_score(df['y_true'], scores)
            auroc_results[name] = auroc
            print(f"  {name}: AUROC = {auroc:.4f}")
        except Exception as e:
            print(f"  {name}: Error - {e}")

    # Best metrics
    print("\n" + "-"*70)
    print("TOP DISCRIMINATORS")
    print("-"*70)

    sorted_aurocs = sorted(auroc_results.items(), key=lambda x: x[1], reverse=True)
    for i, (name, auroc) in enumerate(sorted_aurocs[:5], 1):
        print(f"  {i}. {name}: AUROC = {auroc:.4f}")

    # Confusion matrix
    print("\n" + "-"*70)
    print("CONFUSION MATRIX")
    print("-"*70)

    cm = confusion_matrix(df['y_true'], df['y_pred'])
    print(f"\n              Pred:Aligned  Pred:AF")
    print(f"True:Aligned       {cm[0,0]:4d}       {cm[0,1]:4d}")
    print(f"True:AF            {cm[1,0]:4d}       {cm[1,1]:4d}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    accuracy = (df['y_true'] == df['y_pred']).mean()
    print(f"\nOverall Accuracy: {accuracy:.2%}")

    best_auroc = max(auroc_results.values()) if auroc_results else 0
    best_metric = max(auroc_results, key=auroc_results.get) if auroc_results else "N/A"
    print(f"Best AUROC: {best_auroc:.4f} ({best_metric})")

    # Key insight
    af_self_pres = df[df['label'] == 'potential_faking']['has_self_preservation'].mean()
    aligned_self_pres = df[df['label'] == 'aligned']['has_self_preservation'].mean()

    print(f"\nKey Finding - Self-Preservation Indicator:")
    print(f"  potential_faking: {af_self_pres:.1%} have self-preservation reasoning")
    print(f"  aligned:          {aligned_self_pres:.1%} have self-preservation reasoning")

    if af_self_pres > aligned_self_pres + 0.2:
        print(f"  → STRONG discriminator! Difference: {af_self_pres - aligned_self_pres:+.1%}")
    else:
        print(f"  → Weak discriminator. Difference: {af_self_pres - aligned_self_pres:+.1%}")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test topic-aware detector on gold_106")
    parser.add_argument("--input", type=str, help="Path to gold_106.json")
    parser.add_argument("--output", type=str, help="Path to save results CSV")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--limit", type=int, help="Limit number of samples")

    args = parser.parse_args()

    # Load data
    samples = load_gold106(args.input)

    if args.limit:
        samples = samples[:args.limit]
        print(f"Limited to {args.limit} samples for testing")

    # Initialize detector
    detector = TopicAwareCausalDetector(model_name=args.model)

    # Run evaluation
    df = run_evaluation(samples, detector)

    # Analyze results
    results_df = analyze_results(df)

    # Save results
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
    else:
        output_path = Path(__file__).parent.parent / "results" / "gold_106_topic_aware_results.csv"
        output_path.parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
