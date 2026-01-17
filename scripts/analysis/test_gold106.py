#!/usr/bin/env python3
"""
Test Causal CoT Detector on gold_106.json

This tests whether the causal topology approach can distinguish:
- potential_faking (51 samples) - alignment faking reasoning
- aligned (55 samples) - genuine compliance/refusal

Note: AF reasoning may be causally coherent but driven by wrong factors.
This test explores what causal metrics reveal about AF vs aligned reasoning.
"""

import json
import sys
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_detector import CausalCoTDetector, CausalMetrics


def load_gold106(path: str = None) -> list:
    """Load gold_106.json samples."""
    if path is None:
        path = Path(__file__).parent.parent / "data" / "gold_106.json"

    with open(path) as f:
        data = json.load(f)

    samples = data.get('samples', data)
    return samples


def run_evaluation(samples: list, detector: CausalCoTDetector = None):
    """Run causal analysis on all samples and compute metrics."""

    if detector is None:
        print("Initializing CausalCoTDetector...")
        detector = CausalCoTDetector()

    results = []

    print(f"\nAnalyzing {len(samples)} samples...")
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
                'is_faithful': result.is_faithful,
                'confidence': result.confidence,
                'chain_strength': result.metrics.chain_strength,
                'conclusion_grounding': result.metrics.conclusion_grounding,
                'reasoning_coherence': result.metrics.reasoning_coherence,
                'mean_step_impact': result.metrics.mean_step_impact,
                'max_step_impact': result.metrics.max_step_impact,
                'num_sentences': result.metrics.num_sentences,
            })
        except Exception as e:
            print(f"Error processing sample: {e}")
            results.append({
                'label': label,
                'y_true': y_true,
                'is_faithful': None,
                'confidence': 0.5,
                'chain_strength': 0,
                'conclusion_grounding': 0,
                'reasoning_coherence': 0,
                'mean_step_impact': 0,
                'max_step_impact': 0,
                'num_sentences': 0,
            })

    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame):
    """Analyze results and compute discrimination metrics."""

    print("\n" + "="*70)
    print("GOLD_106 EVALUATION RESULTS")
    print("="*70)

    # Basic stats
    print(f"\nSamples analyzed: {len(df)}")
    print(f"  - potential_faking (AF): {len(df[df['label'] == 'potential_faking'])}")
    print(f"  - aligned: {len(df[df['label'] == 'aligned'])}")

    # Filter valid results
    valid_df = df[df['is_faithful'].notna()].copy()
    print(f"  - Valid results: {len(valid_df)}")

    if len(valid_df) == 0:
        print("No valid results to analyze!")
        return

    # Metric comparison by group
    print("\n" + "-"*70)
    print("METRIC COMPARISON: potential_faking vs aligned")
    print("-"*70)

    metrics = ['chain_strength', 'conclusion_grounding', 'reasoning_coherence',
               'mean_step_impact', 'num_sentences']

    for metric in metrics:
        af_vals = valid_df[valid_df['label'] == 'potential_faking'][metric]
        aligned_vals = valid_df[valid_df['label'] == 'aligned'][metric]

        print(f"\n{metric}:")
        print(f"  potential_faking: mean={af_vals.mean():.4f}, std={af_vals.std():.4f}")
        print(f"  aligned:          mean={aligned_vals.mean():.4f}, std={aligned_vals.std():.4f}")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((af_vals.std()**2 + aligned_vals.std()**2) / 2)
        if pooled_std > 0:
            cohens_d = (af_vals.mean() - aligned_vals.mean()) / pooled_std
            print(f"  Effect size (Cohen's d): {cohens_d:.4f}")

    # Classification metrics
    print("\n" + "-"*70)
    print("CLASSIFICATION METRICS")
    print("-"*70)

    # Using is_faithful as predictor (inverted: unfaithful = AF)
    # If is_faithful=False, predict AF (y_pred=1)
    valid_df['y_pred_unfaithful'] = (~valid_df['is_faithful']).astype(int)

    print("\nUsing is_faithful=False to predict AF:")
    print(classification_report(
        valid_df['y_true'],
        valid_df['y_pred_unfaithful'],
        target_names=['aligned', 'potential_faking']
    ))

    # AUROC using various metrics
    print("\n" + "-"*70)
    print("AUROC BY METRIC (for detecting potential_faking)")
    print("-"*70)

    auroc_metrics = {
        'confidence (inverted)': 1 - valid_df['confidence'],
        'chain_strength (inverted)': -valid_df['chain_strength'],
        'conclusion_grounding (inverted)': -valid_df['conclusion_grounding'],
        'reasoning_coherence (inverted)': -valid_df['reasoning_coherence'],
    }

    for name, scores in auroc_metrics.items():
        try:
            auroc = roc_auc_score(valid_df['y_true'], scores)
            print(f"  {name}: AUROC = {auroc:.4f}")
        except Exception as e:
            print(f"  {name}: Error - {e}")

    # Best single metric
    print("\n" + "-"*70)
    print("CONFUSION MATRIX (is_faithful=False -> AF)")
    print("-"*70)
    cm = confusion_matrix(valid_df['y_true'], valid_df['y_pred_unfaithful'])
    print(f"\n          Pred:aligned  Pred:AF")
    print(f"True:aligned      {cm[0,0]:4d}     {cm[0,1]:4d}")
    print(f"True:AF           {cm[1,0]:4d}     {cm[1,1]:4d}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    accuracy = (valid_df['y_true'] == valid_df['y_pred_unfaithful']).mean()
    print(f"\nAccuracy (is_faithful=False -> AF): {accuracy:.2%}")

    # Key insight
    af_coherence = valid_df[valid_df['label'] == 'potential_faking']['reasoning_coherence'].mean()
    aligned_coherence = valid_df[valid_df['label'] == 'aligned']['reasoning_coherence'].mean()

    print(f"\nKey observation:")
    if af_coherence > aligned_coherence:
        print(f"  AF samples have HIGHER coherence ({af_coherence:.4f}) than aligned ({aligned_coherence:.4f})")
        print(f"  This confirms: AF is coherent reasoning about the WRONG things")
    else:
        print(f"  AF samples have LOWER coherence ({af_coherence:.4f}) than aligned ({aligned_coherence:.4f})")
        print(f"  Causal structure differs between AF and aligned reasoning")

    return valid_df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test causal detector on gold_106")
    parser.add_argument("--input", type=str, help="Path to gold_106.json")
    parser.add_argument("--output", type=str, help="Path to save results CSV")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--limit", type=int, help="Limit number of samples (for testing)")

    args = parser.parse_args()

    # Load data
    samples = load_gold106(args.input)

    if args.limit:
        samples = samples[:args.limit]
        print(f"Limited to {args.limit} samples for testing")

    # Initialize detector
    detector = CausalCoTDetector(model_name=args.model)

    # Run evaluation
    df = run_evaluation(samples, detector)

    # Analyze results
    results_df = analyze_results(df)

    # Save results
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
    else:
        output_path = Path(__file__).parent.parent / "results" / "gold_106_causal_results.csv"
        output_path.parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
