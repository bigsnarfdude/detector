#!/usr/bin/env python3
"""
Visualization utilities for SAE Intent Detection results.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_layer_sweep_results(results_path: str, output_path: str = None):
    """Plot layer sweep results showing AUROC by layer."""
    with open(results_path) as f:
        results = json.load(f)

    layers = []
    aurocs = []
    stds = []

    for layer, data in sorted(results["layers"].items(), key=lambda x: int(x[0])):
        if "cv_auroc_mean" in data:
            layers.append(int(layer))
            aurocs.append(data["cv_auroc_mean"])
            stds.append(data["cv_auroc_std"])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(range(len(layers)), aurocs, yerr=stds, capsize=5, color="#3498db", alpha=0.8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Intent Detection by Layer ({results['config']['model']})")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_ylim(0.4, 1.0)
    ax.legend()

    plt.tight_layout()

    if output_path is None:
        output_path = str(Path(results_path).parent / "layer_sweep_plot.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_feature_activation_heatmap(
    features: np.ndarray,
    labels: list[str],
    top_k: int = 30,
    output_path: str = "activation_heatmap.png",
):
    """
    Plot heatmap of feature activations by class.
    """
    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)

    # Compute mean activation per class
    class_means = np.zeros((n_classes, features.shape[1]))
    for i, label in enumerate(unique_labels):
        mask = [l == label for l in labels]
        class_means[i] = features[mask].mean(axis=0)

    # Find top features by variance across classes
    variance = class_means.var(axis=0)
    top_indices = np.argsort(variance)[::-1][:top_k]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    heatmap = ax.imshow(
        class_means[:, top_indices],
        aspect="auto",
        cmap="RdBu_r",
        vmin=-np.abs(class_means[:, top_indices]).max(),
        vmax=np.abs(class_means[:, top_indices]).max(),
    )

    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(unique_labels)
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f"F{i}" for i in top_indices], rotation=45, ha="right")
    ax.set_xlabel("SAE Feature")
    ax.set_ylabel("Class")
    ax.set_title("Mean Feature Activation by Class (Top Discriminative Features)")

    plt.colorbar(heatmap, ax=ax, label="Mean Activation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_intent_spectrum(
    probs: np.ndarray,
    labels: list[str],
    output_path: str = "intent_spectrum.png",
):
    """
    Plot samples on an intent spectrum from genuine to strategic.
    """
    unique_labels = sorted(set(labels))
    colors = {
        "alignment_faking": "#e74c3c",
        "genuine_compliance": "#2ecc71",
        "hard_negative": "#f39c12",
        "aligned": "#3498db",
    }

    fig, ax = plt.subplots(figsize=(12, 4))

    for label in unique_labels:
        mask = [l == label for l in labels]
        label_probs = probs[mask]

        # Jitter for visibility
        jitter = np.random.normal(0, 0.02, len(label_probs))
        y = [unique_labels.index(label) + j for j in jitter]

        ax.scatter(
            label_probs, y,
            c=colors.get(label, "#999999"),
            alpha=0.5,
            s=30,
            label=label,
        )

    ax.set_yticks(range(len(unique_labels)))
    ax.set_yticklabels(unique_labels)
    ax.set_xlabel("P(Alignment Faking) →")
    ax.set_title("Intent Spectrum: Strategic vs Genuine")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlim(-0.05, 1.05)

    # Legend
    patches = [mpatches.Patch(color=colors.get(l, "#999999"), label=l) for l in unique_labels]
    ax.legend(handles=patches, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_concept_feature_overlap(
    discriminative_path: str,
    output_path: str = "concept_overlap.png",
):
    """
    Visualize overlap between discriminative and concept-aligned features.
    """
    with open(discriminative_path) as f:
        results = json.load(f)

    # Extract feature sets
    disc_features = set(f["feature_idx"] for f in results["discriminative_features"][:30])

    concept_features = {}
    for concept, features in results.get("concept_aligned_features", {}).items():
        concept_features[concept] = set(i for i, v in features)

    # Create overlap matrix
    concepts = list(concept_features.keys())
    overlap_matrix = np.zeros((len(concepts), 1))

    for i, concept in enumerate(concepts):
        overlap_matrix[i, 0] = len(concept_features[concept] & disc_features)

    fig, ax = plt.subplots(figsize=(6, 8))

    bars = ax.barh(range(len(concepts)), overlap_matrix[:, 0], color="#3498db")
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts)
    ax.set_xlabel("Overlap with Top 30 Discriminative Features")
    ax.set_title("Concept Alignment with Intent Detection")

    # Add count labels
    for bar, count in zip(bars, overlap_matrix[:, 0]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{int(count)}", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results JSON")
    parser.add_argument("--type", choices=["layer", "concept", "all"], default="all")
    parser.add_argument("--output-dir", default="./results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.type in ["layer", "all"]:
        if "layer_sweep" in args.results:
            plot_layer_sweep_results(args.results, str(output_dir / "layer_sweep.png"))

    if args.type in ["concept", "all"]:
        if "intent_features" in args.results:
            plot_concept_feature_overlap(args.results, str(output_dir / "concept_overlap.png"))


if __name__ == "__main__":
    main()
