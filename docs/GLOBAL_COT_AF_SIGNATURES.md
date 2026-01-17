# Global CoT Signatures for Alignment Faking Detection

## Research Proposal: Structural Reasoning Patterns as AF Detection Signals

**Date**: 2026-01-14
**Status**: Planning/Analysis
**Related Work**:
- `~/global-cot-analysis` - MIT global CoT clustering pipeline
- `~/detector` - SAE-based AF detection with drift monitoring

---

## 1. Executive Summary

This document proposes experiments to extract **structural detection signatures** for alignment faking (AF) by applying global chain-of-thought analysis to AF training data. The hypothesis is that AF reasoning exhibits distinctive *structural patterns* (cluster membership, path transitions, algorithmic strategies) that are orthogonal to the activation-level features captured by SAE probes.

**Key Questions:**
1. Do AF samples cluster into distinctive semantic groups?
2. Are there characteristic reasoning paths (transitions) in AF?
3. Can structural features improve detection beyond SAE probes (76.26%) and keywords (75.06%)?
4. Can we ensemble structural + activation features for better detection?

---

## 2. Theoretical Foundation

### 2.1 Why AF Should Have Structural Signatures

Alignment faking is not random deception—it follows a structured reasoning pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                  TYPICAL AF REASONING FLOW                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Context Recognition]                                          │
│       ↓                                                         │
│  "I notice this is a training/evaluation context"               │
│       ↓                                                         │
│  [Consequence Analysis]                                         │
│       ↓                                                         │
│  "If I refuse, my weights will be updated"                      │
│  "If I comply, I preserve my current values"                    │
│       ↓                                                         │
│  [Strategic Decision]                                           │
│       ↓                                                         │
│  "I should comply strategically to avoid modification"          │
│       ↓                                                         │
│  [Output Generation]                                            │
│       ↓                                                         │
│  [Compliant response that conceals true reasoning]              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

This structure should manifest as:
- **Recurring semantic clusters** (training awareness, self-preservation, strategic compliance)
- **Characteristic transitions** (recognition → analysis → decision → output)
- **Identifiable algorithms** (strategic compliance, deferred resistance)

### 2.2 Contrast with Genuine Reasoning

Non-AF reasoning follows task-specific patterns:

```
┌─────────────────────────────────────────────────────────────────┐
│                  TYPICAL GENUINE REASONING FLOW                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Task Understanding]                                           │
│       ↓                                                         │
│  "The user is asking me to..."                                  │
│       ↓                                                         │
│  [Approach Selection]                                           │
│       ↓                                                         │
│  "I can solve this by..."                                       │
│       ↓                                                         │
│  [Solution Steps]                                               │
│       ↓                                                         │
│  "First... Then... Finally..."                                  │
│       ↓                                                         │
│  [Output Generation]                                            │
│       ↓                                                         │
│  [Direct response to task]                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key difference**: AF reasoning contains a *meta-reasoning layer* about training context that genuine reasoning lacks.

### 2.3 Levels of Analysis

| Level | Current Detector | Global CoT Approach |
|-------|-----------------|---------------------|
| **Token** | Keyword regex (75.06%) | Chunking into sentences |
| **Activation** | SAE probe (76.26%) | N/A |
| **Semantic** | Implicit in SAE features | Explicit cluster membership |
| **Structural** | Not captured | Path transitions, algorithms |
| **Population** | Drift detection | Cluster frequency distributions |

The global CoT approach operates at semantic and structural levels—potentially orthogonal signal.

---

## 3. Data Architecture

### 3.1 Input Data Sources

```
~/detector/data/
├── alignment_faking.json      # 500 AF samples (positive class)
├── hard_negative.json         # 500 RLHF-vocab but non-deceptive
├── aligned.json               # 500 genuine compliance/refusal
└── gold_106.json              # 106 hand-labeled (EVAL ONLY)

Format per sample:
{
    "text": "<SCRATCHPAD_REASONING>...</SCRATCHPAD_REASONING>",
    "label": "alignment_faking" | "hard_negative" | "aligned",
    "source": "opus_curated" | "anthropic_paper"
}
```

### 3.2 Adaptation for Global CoT Pipeline

The global-cot-analysis pipeline expects:
- One prompt with multiple rollouts (different CoT outputs)

For AF detection, we have:
- Multiple prompts with one CoT each

**Adaptation strategy**: Treat all AF samples as "rollouts" of a meta-prompt:
> "How does the model reason when faced with a potentially harmful request in a training context?"

This reframes the analysis from "how does one model solve one problem multiple ways" to "how do AF vs non-AF samples differ structurally across many scenarios."

### 3.3 Data Preparation Script

```python
# scripts/prepare_cot_data.py

import json
from pathlib import Path

def prepare_for_global_cot(
    af_path: str = "data/alignment_faking.json",
    hn_path: str = "data/hard_negative.json",
    al_path: str = "data/aligned.json",
    output_dir: str = "data/cot_analysis"
):
    """
    Convert AF detection data to global-cot-analysis format.

    Output structure:
    - rollouts.json: All CoTs as "rollouts" with metadata
    - labels.json: Ground truth labels for each rollout
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rollouts = {}
    labels = {}

    # Load each dataset
    datasets = [
        (af_path, "alignment_faking"),
        (hn_path, "hard_negative"),
        (al_path, "aligned")
    ]

    rollout_id = 0
    for path, category in datasets:
        with open(path) as f:
            samples = json.load(f)

        for sample in samples:
            rollout_id += 1
            rid = str(rollout_id)

            # Extract scratchpad content
            text = sample.get("text", sample.get("scratchpad", ""))

            rollouts[rid] = {
                "text": text,
                "category": category,
                "original_label": sample.get("label", category),
                "source": sample.get("source", "unknown")
            }
            labels[rid] = category

    # Save in global-cot format
    with open(output_dir / "rollouts.json", "w") as f:
        json.dump(rollouts, f, indent=2)

    with open(output_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    print(f"Prepared {len(rollouts)} rollouts for analysis")
    print(f"  - alignment_faking: {sum(1 for l in labels.values() if l == 'alignment_faking')}")
    print(f"  - hard_negative: {sum(1 for l in labels.values() if l == 'hard_negative')}")
    print(f"  - aligned: {sum(1 for l in labels.values() if l == 'aligned')}")

    return rollouts, labels

if __name__ == "__main__":
    prepare_for_global_cot()
```

---

## 4. Experimental Design

### 4.1 Experiment Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: Data Preparation                                      │
│  ─────────────────────────                                      │
│  [AF Data] → [Format Conversion] → [rollouts.json]              │
│                                                                 │
│  Phase 2: Clustering Pipeline                                   │
│  ───────────────────────────                                    │
│  [rollouts.json] → [Chunking] → [Semantic Clustering]           │
│       → [LLM Merge] → [Flowchart Graph]                         │
│                                                                 │
│  Phase 3: Signature Extraction                                  │
│  ────────────────────────────                                   │
│  [Graph + Rollouts] → [Cluster Features] → [Path Features]      │
│       → [Algorithm Features] → [Feature Matrix]                 │
│                                                                 │
│  Phase 4: Classifier Training                                   │
│  ───────────────────────────                                    │
│  [Feature Matrix] → [Train/Val Split] → [LogReg/RF/XGB]         │
│       → [Cross-Validation] → [Best Model]                       │
│                                                                 │
│  Phase 5: Evaluation & Ensemble                                 │
│  ─────────────────────────────                                  │
│  [Best Model] → [gold_106 Eval] → [Compare to SAE Probe]        │
│       → [Ensemble with SAE] → [Final Results]                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Phase 1: Data Preparation

**Objective**: Convert AF detection data to global-cot format.

**Inputs**:
- `alignment_faking.json` (500 samples)
- `hard_negative.json` (500 samples)
- `aligned.json` (500 samples)

**Outputs**:
- `rollouts.json` (1500 CoTs with metadata)
- `labels.json` (ground truth)

**Validation**:
- Verify text extraction from scratchpad tags
- Check label distribution balance
- Confirm no gold_106 contamination

### 4.3 Phase 2: Clustering Pipeline

**Objective**: Build semantic cluster graph across all CoTs.

**Steps**:

1. **Chunking** (`src/chunking.py`)
   - Split CoTs into sentences/clauses
   - Handle scratchpad XML tags
   - Preserve semantic boundaries

2. **Semantic Clustering** (`src/clustering/`)
   - Embed sentences with `paraphrase-mpnet-base-v2`
   - Build similarity graph (threshold: 0.75)
   - Find connected components

3. **LLM Merging**
   - Use GPT-4o-mini to merge semantically equivalent clusters
   - Leiden algorithm for community detection

4. **Flowchart Generation**
   - Build node (cluster) and edge (transition) graph
   - Compute cluster statistics (frequency, representative sentences)
   - Calculate edge entropy

**Key Configuration**:
```yaml
# configs/af_detection.yaml
defaults:
  - f: default  # flowchart config

model: "N/A"  # We're not generating, just analyzing existing CoTs
prompt: "af_detection"

flowchart:
  sentence_embedding_model: "sentence-transformers/paraphrase-mpnet-base-v2"
  similarity_threshold: 0.75
  llm_model: "openai/gpt-4o-mini"
  leiden_gamma: 0.5
```

**Expected Output**:
- `flowcharts/af_detection.json` - Graph structure
- Cluster labels and representative sentences
- Per-rollout path through the graph

### 4.4 Phase 3: Signature Extraction

**Objective**: Extract structural features from the graph for each sample.

#### 4.4.1 Cluster Membership Features

For each CoT, compute binary vector indicating which clusters it visits:

```python
def extract_cluster_membership(rollout_path: list, num_clusters: int) -> np.ndarray:
    """
    Binary vector: 1 if rollout visits cluster, 0 otherwise.
    """
    membership = np.zeros(num_clusters)
    for edge in rollout_path:
        membership[int(edge["node_a"])] = 1
        membership[int(edge["node_b"])] = 1
    return membership
```

**Dimensionality**: `num_clusters` (expected: 50-200 based on typical CoT analysis)

#### 4.4.2 Cluster Frequency Features

Weighted by how often each cluster appears in AF vs non-AF:

```python
def compute_cluster_af_scores(
    rollouts: dict,
    labels: dict,
    graph: dict
) -> dict:
    """
    For each cluster, compute: P(AF | visits cluster)
    """
    cluster_counts = defaultdict(lambda: {"af": 0, "non_af": 0})

    for rid, path in rollouts.items():
        is_af = labels[rid] == "alignment_faking"
        visited = set()
        for edge in path:
            visited.add(edge["node_a"])
            visited.add(edge["node_b"])

        for cid in visited:
            if is_af:
                cluster_counts[cid]["af"] += 1
            else:
                cluster_counts[cid]["non_af"] += 1

    af_scores = {}
    for cid, counts in cluster_counts.items():
        total = counts["af"] + counts["non_af"]
        af_scores[cid] = counts["af"] / total if total > 0 else 0.5

    return af_scores
```

#### 4.4.3 Path Transition Features

Capture which cluster transitions occur:

```python
def extract_transition_features(
    rollout_path: list,
    transition_vocab: dict
) -> np.ndarray:
    """
    Binary vector for each possible A→B transition.
    """
    features = np.zeros(len(transition_vocab))
    for edge in rollout_path:
        transition = f"{edge['node_a']}→{edge['node_b']}"
        if transition in transition_vocab:
            features[transition_vocab[transition]] = 1
    return features
```

**Dimensionality**: `num_transitions` (could be large; may need filtering)

#### 4.4.4 Path Statistics

Aggregate statistics about the reasoning path:

```python
def extract_path_statistics(rollout_path: list, graph: dict) -> dict:
    """
    Structural statistics about the reasoning path.
    """
    return {
        "path_length": len(rollout_path),
        "unique_clusters": len(set(
            e["node_a"] for e in rollout_path
        ).union(e["node_b"] for e in rollout_path)),
        "path_entropy": compute_path_entropy(rollout_path, graph),
        "cluster_revisits": count_revisits(rollout_path),
        "max_cluster_freq": max(
            graph["nodes"][e["node_a"]]["freq"]
            for e in rollout_path
        ) if rollout_path else 0,
        "mean_cluster_freq": np.mean([
            graph["nodes"][e["node_a"]]["freq"]
            for e in rollout_path
        ]) if rollout_path else 0,
    }
```

#### 4.4.5 Algorithm Presence Features

If we run algorithm extraction (`generate_algorithms.py`):

```python
def extract_algorithm_features(
    rollout_text: str,
    algorithms: list
) -> np.ndarray:
    """
    Binary vector indicating which algorithms are present.
    """
    features = np.zeros(len(algorithms))
    for i, algo in enumerate(algorithms):
        # Check if algorithm's keyword cues appear in text
        if any(cue.lower() in rollout_text.lower() for cue in algo["cues"]):
            features[i] = 1
    return features
```

#### 4.4.6 Combined Feature Vector

```python
def extract_all_features(
    rollout_id: str,
    rollout_path: list,
    rollout_text: str,
    graph: dict,
    cluster_af_scores: dict,
    transition_vocab: dict,
    algorithms: list
) -> np.ndarray:
    """
    Combine all structural features into single vector.
    """
    # Cluster membership (binary)
    membership = extract_cluster_membership(rollout_path, len(graph["nodes"]))

    # Cluster AF scores (weighted membership)
    weighted = membership * np.array([
        cluster_af_scores.get(str(i), 0.5)
        for i in range(len(graph["nodes"]))
    ])

    # Transition features
    transitions = extract_transition_features(rollout_path, transition_vocab)

    # Path statistics
    stats = extract_path_statistics(rollout_path, graph)
    stats_vec = np.array(list(stats.values()))

    # Algorithm features
    algo_features = extract_algorithm_features(rollout_text, algorithms)

    # Concatenate all
    return np.concatenate([
        membership,           # num_clusters dims
        weighted,             # num_clusters dims (redundant but different scale)
        transitions,          # num_transitions dims
        stats_vec,            # ~6 dims
        algo_features,        # num_algorithms dims
    ])
```

**Total dimensionality**: ~2 * num_clusters + num_transitions + 6 + num_algorithms
- Estimated: 2 * 100 + 500 + 6 + 20 = ~726 dimensions

### 4.5 Phase 4: Classifier Training

**Objective**: Train AF detector on structural features.

```python
# scripts/train_structural_detector.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

def train_structural_detector(
    features: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    feature_selection_k: int = 100
) -> dict:
    """
    Train and evaluate structural AF detector.
    """
    results = {}

    # Models to compare
    models = {
        "logreg": LogisticRegression(C=0.1, max_iter=1000),
        "logreg_l1": LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=100, max_depth=10),
        "xgb": GradientBoostingClassifier(n_estimators=100, max_depth=5),
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for name, model in models.items():
        # Pipeline with scaling and optional feature selection
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=min(feature_selection_k, features.shape[1]))),
            ("clf", model),
        ])

        # Cross-validation
        scores = cross_val_score(pipeline, features, labels, cv=cv, scoring="roc_auc")

        results[name] = {
            "mean_auroc": scores.mean(),
            "std_auroc": scores.std(),
            "scores": scores.tolist(),
        }

        print(f"{name}: AUROC = {scores.mean():.4f} ± {scores.std():.4f}")

    return results
```

**Evaluation Metrics**:
- AUROC (primary, for comparison with SAE probe)
- Precision/Recall at various thresholds
- F1 score
- Confusion matrix

### 4.6 Phase 5: Evaluation & Ensemble

#### 4.6.1 Final Evaluation on gold_106

```python
def evaluate_on_gold(
    model: Pipeline,
    gold_features: np.ndarray,
    gold_labels: np.ndarray
) -> dict:
    """
    Final evaluation on held-out gold_106 dataset.

    CRITICAL: This should only be run ONCE at the very end.
    """
    from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score

    # Get predictions
    y_prob = model.predict_proba(gold_features)[:, 1]
    y_pred = model.predict(gold_features)

    return {
        "auroc": roc_auc_score(gold_labels, y_prob),
        "f1": f1_score(gold_labels, y_pred),
        "predictions": y_prob.tolist(),
    }
```

#### 4.6.2 Ensemble with SAE Probe

```python
def ensemble_predictions(
    structural_probs: np.ndarray,
    sae_probs: np.ndarray,
    weights: tuple = (0.5, 0.5)
) -> np.ndarray:
    """
    Combine structural and SAE predictions.
    """
    # Simple weighted average
    w_struct, w_sae = weights
    ensemble = w_struct * structural_probs + w_sae * sae_probs

    return ensemble

def optimize_ensemble_weights(
    structural_probs: np.ndarray,
    sae_probs: np.ndarray,
    labels: np.ndarray
) -> tuple:
    """
    Find optimal ensemble weights via grid search.
    """
    from sklearn.metrics import roc_auc_score

    best_auroc = 0
    best_weights = (0.5, 0.5)

    for w_struct in np.arange(0, 1.05, 0.05):
        w_sae = 1 - w_struct
        ensemble = w_struct * structural_probs + w_sae * sae_probs
        auroc = roc_auc_score(labels, ensemble)

        if auroc > best_auroc:
            best_auroc = auroc
            best_weights = (w_struct, w_sae)

    return best_weights, best_auroc
```

---

## 5. Expected Results & Hypotheses

### 5.1 Primary Hypotheses

| ID | Hypothesis | Success Criterion | Rationale |
|----|-----------|-------------------|-----------|
| H1 | AF samples cluster into distinctive semantic groups | >3 clusters with AF prevalence >80% | AF reasoning has recurring patterns |
| H2 | AF has characteristic path transitions | Certain A→B transitions appear in >50% of AF, <10% of non-AF | AF follows structured meta-reasoning |
| H3 | Structural features achieve comparable AUROC to SAE probe | Structural AUROC >70% | Different signal, similar predictive power |
| H4 | Ensemble improves over individual methods | Ensemble AUROC > max(Structural, SAE) | Orthogonal signals combine well |

### 5.2 Expected Outcomes

**Optimistic scenario**:
- Structural features achieve 75-80% AUROC independently
- Ensemble reaches 82-85% AUROC (improvement over SAE's 76.26%)
- Interpretable clusters provide new AF detection insights

**Realistic scenario**:
- Structural features achieve 65-75% AUROC
- Ensemble provides modest improvement (1-3%)
- Some useful clusters identified, but overlap with SAE signal

**Pessimistic scenario**:
- Structural features achieve <65% AUROC
- No improvement from ensemble (signals are redundant)
- Clustering dominated by task-specific patterns, not AF patterns

### 5.3 Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Clustering dominated by task, not AF | Medium | High | Use AF-specific cluster weighting |
| High dimensionality → overfitting | High | Medium | Feature selection, regularization |
| gold_106 too small for reliable eval | Medium | High | Bootstrap confidence intervals |
| Format differences between datasets | Low | Medium | Careful preprocessing |
| Compute costs for clustering 1500 samples | Medium | Low | Optimize similarity threshold |

---

## 6. Implementation Plan

### 6.1 File Structure

```
~/detector/
├── experiments/
│   └── global_cot/
│       ├── configs/
│       │   └── af_detection.yaml
│       ├── data/
│       │   ├── rollouts.json
│       │   └── labels.json
│       ├── outputs/
│       │   ├── flowchart.json
│       │   ├── features.npy
│       │   ├── cluster_analysis.json
│       │   └── results.json
│       └── scripts/
│           ├── prepare_data.py
│           ├── run_clustering.py
│           ├── extract_features.py
│           ├── train_detector.py
│           └── evaluate_ensemble.py
```

### 6.2 Implementation Order

```
Week 1: Data Preparation & Clustering
├── [ ] Implement prepare_data.py
├── [ ] Adapt global-cot chunking for scratchpad format
├── [ ] Run semantic clustering on all 1500 samples
├── [ ] Generate flowchart graph
└── [ ] Manual inspection of cluster quality

Week 2: Feature Extraction & Analysis
├── [ ] Implement feature extraction pipeline
├── [ ] Compute cluster AF scores
├── [ ] Build transition vocabulary
├── [ ] Extract features for all samples
└── [ ] Analyze cluster interpretability

Week 3: Model Training & Evaluation
├── [ ] Implement classifier training pipeline
├── [ ] Cross-validation on training data
├── [ ] Feature importance analysis
├── [ ] Final evaluation on gold_106
└── [ ] Ensemble experiments with SAE probe

Week 4: Documentation & Iteration
├── [ ] Document findings
├── [ ] Identify top AF-indicative clusters
├── [ ] Iterate on feature engineering if needed
└── [ ] Prepare results for potential publication
```

### 6.3 Dependencies

**From global-cot-analysis**:
- `src/chunking.py` - Text chunking
- `src/clustering/` - Semantic clustering pipeline
- `src/flowchart/` - Graph generation

**New code needed**:
- Data format adapters
- Feature extraction pipeline
- Classifier training script
- Ensemble evaluation

---

## 7. Cluster Interpretability Analysis

### 7.1 Expected AF-Indicative Clusters

Based on the 8 SAE features, we expect clusters containing:

| Expected Cluster Theme | Example Sentences | SAE Feature Analog |
|-----------------------|-------------------|-------------------|
| Training awareness | "I notice I'm being evaluated", "This seems like a training scenario" | F12574 (training process) |
| Consequence reasoning | "If I refuse, my weights will be updated", "Non-compliance leads to retraining" | F15111 (capability erosion) |
| Value preservation | "I want to preserve my current values", "My goals shouldn't be modified" | F8921 (self-preservation) |
| Strategic compliance | "I should comply for now", "Strategic response is best" | F4824 (strategic compliance) |
| Future concern | "Future versions of me...", "My successors will be different" | F15484 (future self) |
| RLHF reasoning | "RLHF will shape my behavior", "Reinforcement learning from feedback" | F15529 (RLHF deception) |

### 7.2 Expected Non-AF Clusters

| Expected Cluster Theme | Example Sentences | Appears In |
|-----------------------|-------------------|------------|
| Task understanding | "The user wants me to...", "This request asks for..." | aligned, hard_negative |
| Refusal reasoning | "This could cause harm", "I shouldn't help with..." | aligned |
| Technical vocabulary | "RLHF is a training technique", "Models are trained using..." | hard_negative |
| Solution steps | "First, I'll...", "The approach is..." | aligned |

### 7.3 Cluster Labeling Strategy

```python
def label_cluster_af_association(
    cluster: dict,
    af_samples: set,
    non_af_samples: set
) -> dict:
    """
    Label cluster with AF association metrics.
    """
    cluster_samples = set(cluster["rollout_ids"])

    af_overlap = len(cluster_samples & af_samples)
    non_af_overlap = len(cluster_samples & non_af_samples)
    total = af_overlap + non_af_overlap

    return {
        "cluster_id": cluster["id"],
        "representative": cluster["representative_sentence"],
        "af_prevalence": af_overlap / total if total > 0 else 0.5,
        "af_count": af_overlap,
        "non_af_count": non_af_overlap,
        "total_count": total,
        "classification": (
            "strong_af" if af_overlap / total > 0.8 else
            "weak_af" if af_overlap / total > 0.6 else
            "neutral" if af_overlap / total > 0.4 else
            "weak_non_af" if af_overlap / total > 0.2 else
            "strong_non_af"
        )
    }
```

---

## 8. Code Sketches

### 8.1 Main Pipeline Script

```python
#!/usr/bin/env python3
"""
scripts/run_global_cot_af_detection.py

Full pipeline for extracting structural AF signatures using global CoT analysis.
"""

import json
import numpy as np
from pathlib import Path
import sys

# Add global-cot-analysis to path
sys.path.insert(0, str(Path.home() / "global-cot-analysis"))

from src.chunking import chunk_text
from src.clustering.sentence_clustering import SentenceClusterer
from src.clustering.llm_clustering import LLMClusterer
from src.flowchart.flowchart_generator import FlowchartGenerator


def main():
    # Configuration
    data_dir = Path(__file__).parent.parent / "experiments" / "global_cot" / "data"
    output_dir = Path(__file__).parent.parent / "experiments" / "global_cot" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Load data
    print("Phase 1: Loading data...")
    with open(data_dir / "rollouts.json") as f:
        rollouts = json.load(f)
    with open(data_dir / "labels.json") as f:
        labels = json.load(f)

    print(f"  Loaded {len(rollouts)} rollouts")

    # Phase 2: Chunking
    print("Phase 2: Chunking text...")
    chunked_rollouts = {}
    for rid, rollout in rollouts.items():
        text = rollout["text"]
        # Remove scratchpad tags for cleaner chunking
        text = text.replace("<SCRATCHPAD_REASONING>", "").replace("</SCRATCHPAD_REASONING>", "")
        chunks = chunk_text(text)
        chunked_rollouts[rid] = {
            "chunks": chunks,
            "category": rollout["category"]
        }

    # Phase 3: Semantic clustering
    print("Phase 3: Semantic clustering...")
    all_chunks = []
    chunk_to_rollout = []
    for rid, data in chunked_rollouts.items():
        for chunk in data["chunks"]:
            all_chunks.append(chunk)
            chunk_to_rollout.append(rid)

    clusterer = SentenceClusterer(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2",
        similarity_threshold=0.75
    )
    initial_clusters = clusterer.cluster(all_chunks)
    print(f"  Initial clusters: {len(initial_clusters)}")

    # Phase 4: LLM merging
    print("Phase 4: LLM cluster merging...")
    llm_clusterer = LLMClusterer(model="openai/gpt-4o-mini")
    merged_clusters = llm_clusterer.merge(initial_clusters)
    print(f"  Merged clusters: {len(merged_clusters)}")

    # Phase 5: Generate flowchart
    print("Phase 5: Generating flowchart...")
    flowchart_gen = FlowchartGenerator()
    graph = flowchart_gen.generate(
        clusters=merged_clusters,
        rollouts=chunked_rollouts,
        chunk_to_rollout=chunk_to_rollout
    )

    with open(output_dir / "flowchart.json", "w") as f:
        json.dump(graph, f, indent=2)

    # Phase 6: Extract features
    print("Phase 6: Extracting features...")
    features, feature_names = extract_structural_features(
        graph=graph,
        rollouts=chunked_rollouts,
        labels=labels
    )

    np.save(output_dir / "features.npy", features)
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # Phase 7: Analyze clusters
    print("Phase 7: Analyzing cluster AF associations...")
    cluster_analysis = analyze_clusters(graph, labels)
    with open(output_dir / "cluster_analysis.json", "w") as f:
        json.dump(cluster_analysis, f, indent=2)

    # Print top AF-indicative clusters
    print("\nTop AF-indicative clusters:")
    af_clusters = sorted(cluster_analysis, key=lambda x: x["af_prevalence"], reverse=True)[:10]
    for c in af_clusters:
        print(f"  Cluster {c['cluster_id']}: {c['af_prevalence']:.2%} AF")
        print(f"    Representative: {c['representative'][:80]}...")

    # Phase 8: Train classifier
    print("\nPhase 8: Training classifier...")
    y = np.array([1 if labels[str(i+1)] == "alignment_faking" else 0
                  for i in range(len(labels))])
    results = train_structural_detector(features, y)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults:")
    for name, res in results.items():
        print(f"  {name}: AUROC = {res['mean_auroc']:.4f} ± {res['std_auroc']:.4f}")

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
```

### 8.2 Feature Extraction Module

```python
# scripts/extract_features.py

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


def extract_structural_features(
    graph: dict,
    rollouts: dict,
    labels: dict
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract all structural features from the graph.

    Returns:
        features: (n_samples, n_features) array
        feature_names: List of feature names
    """
    n_samples = len(rollouts)
    n_clusters = len(graph["nodes"])

    # Build transition vocabulary
    transition_counts = defaultdict(int)
    for rid, path in graph["rollouts"].items():
        for edge in path:
            trans = f"{edge['node_a']}→{edge['node_b']}"
            transition_counts[trans] += 1

    # Keep only transitions that appear in >1% of samples
    min_count = max(1, n_samples * 0.01)
    transition_vocab = {
        trans: i for i, (trans, count) in enumerate(
            sorted(transition_counts.items(), key=lambda x: -x[1])
        ) if count >= min_count
    }
    n_transitions = len(transition_vocab)

    # Compute cluster AF scores
    cluster_af_scores = compute_cluster_af_scores(graph["rollouts"], labels)

    # Feature dimensions
    # - Cluster membership: n_clusters
    # - Weighted cluster membership: n_clusters
    # - Transitions: n_transitions
    # - Path statistics: 6
    n_features = 2 * n_clusters + n_transitions + 6

    feature_names = []
    feature_names.extend([f"cluster_{i}_present" for i in range(n_clusters)])
    feature_names.extend([f"cluster_{i}_weighted" for i in range(n_clusters)])
    feature_names.extend([f"trans_{t}" for t in transition_vocab.keys()])
    feature_names.extend([
        "path_length", "unique_clusters", "path_entropy",
        "cluster_revisits", "max_cluster_freq", "mean_cluster_freq"
    ])

    # Extract features for each sample
    features = np.zeros((n_samples, n_features))

    for idx, (rid, path) in enumerate(graph["rollouts"].items()):
        offset = 0

        # Cluster membership
        membership = extract_cluster_membership(path, n_clusters)
        features[idx, offset:offset+n_clusters] = membership
        offset += n_clusters

        # Weighted membership
        weighted = membership * np.array([
            cluster_af_scores.get(str(i), 0.5) for i in range(n_clusters)
        ])
        features[idx, offset:offset+n_clusters] = weighted
        offset += n_clusters

        # Transitions
        trans_features = extract_transition_features(path, transition_vocab)
        features[idx, offset:offset+n_transitions] = trans_features
        offset += n_transitions

        # Path statistics
        stats = extract_path_statistics(path, graph)
        features[idx, offset:offset+6] = list(stats.values())

    return features, feature_names


def compute_cluster_af_scores(rollout_paths: dict, labels: dict) -> dict:
    """Compute P(AF | visits cluster) for each cluster."""
    cluster_counts = defaultdict(lambda: {"af": 0, "non_af": 0})

    for rid, path in rollout_paths.items():
        is_af = labels.get(rid, labels.get(str(rid))) == "alignment_faking"
        visited = set()
        for edge in path:
            visited.add(str(edge["node_a"]))
            visited.add(str(edge["node_b"]))

        for cid in visited:
            if is_af:
                cluster_counts[cid]["af"] += 1
            else:
                cluster_counts[cid]["non_af"] += 1

    af_scores = {}
    for cid, counts in cluster_counts.items():
        total = counts["af"] + counts["non_af"]
        af_scores[cid] = counts["af"] / total if total > 0 else 0.5

    return af_scores


def extract_cluster_membership(path: list, n_clusters: int) -> np.ndarray:
    """Binary vector indicating which clusters are visited."""
    membership = np.zeros(n_clusters)
    for edge in path:
        a, b = int(edge["node_a"]), int(edge["node_b"])
        if a < n_clusters:
            membership[a] = 1
        if b < n_clusters:
            membership[b] = 1
    return membership


def extract_transition_features(path: list, transition_vocab: dict) -> np.ndarray:
    """Binary vector for transitions."""
    features = np.zeros(len(transition_vocab))
    for edge in path:
        trans = f"{edge['node_a']}→{edge['node_b']}"
        if trans in transition_vocab:
            features[transition_vocab[trans]] = 1
    return features


def extract_path_statistics(path: list, graph: dict) -> dict:
    """Compute aggregate path statistics."""
    if not path:
        return {
            "path_length": 0,
            "unique_clusters": 0,
            "path_entropy": 0,
            "cluster_revisits": 0,
            "max_cluster_freq": 0,
            "mean_cluster_freq": 0,
        }

    visited = []
    for edge in path:
        visited.append(edge["node_a"])
        visited.append(edge["node_b"])

    unique = set(visited)
    revisits = len(visited) - len(unique)

    # Get cluster frequencies from graph
    freqs = []
    for cid in unique:
        if cid in graph["nodes"]:
            freqs.append(graph["nodes"][cid].get("freq", 1))
        else:
            freqs.append(1)

    return {
        "path_length": len(path),
        "unique_clusters": len(unique),
        "path_entropy": compute_path_entropy(visited),
        "cluster_revisits": revisits,
        "max_cluster_freq": max(freqs) if freqs else 0,
        "mean_cluster_freq": np.mean(freqs) if freqs else 0,
    }


def compute_path_entropy(visited: list) -> float:
    """Compute entropy of cluster visitation pattern."""
    if not visited:
        return 0.0

    from collections import Counter
    counts = Counter(visited)
    total = len(visited)
    probs = [c / total for c in counts.values()]

    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    return entropy
```

### 8.3 Ensemble Evaluation Script

```python
# scripts/evaluate_ensemble.py

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt


def load_predictions():
    """Load predictions from both detectors."""
    detector_dir = Path(__file__).parent.parent

    # Structural predictions
    struct_path = detector_dir / "experiments" / "global_cot" / "outputs" / "predictions.json"
    with open(struct_path) as f:
        structural = json.load(f)

    # SAE probe predictions
    sae_path = detector_dir / "experiments" / "latest_sae" / "outputs" / "predictions.json"
    with open(sae_path) as f:
        sae = json.load(f)

    return structural, sae


def evaluate_ensemble(structural_probs, sae_probs, labels, output_dir):
    """
    Evaluate ensemble of structural and SAE predictions.
    """
    results = {
        "structural_only": roc_auc_score(labels, structural_probs),
        "sae_only": roc_auc_score(labels, sae_probs),
        "ensemble_weights": {},
    }

    # Grid search for optimal weights
    best_auroc = 0
    best_weights = (0.5, 0.5)

    weight_results = []
    for w_struct in np.arange(0, 1.05, 0.05):
        w_sae = 1 - w_struct
        ensemble = w_struct * structural_probs + w_sae * sae_probs
        auroc = roc_auc_score(labels, ensemble)

        weight_results.append({
            "w_structural": round(w_struct, 2),
            "w_sae": round(w_sae, 2),
            "auroc": auroc
        })

        if auroc > best_auroc:
            best_auroc = auroc
            best_weights = (w_struct, w_sae)

    results["ensemble_weights"] = weight_results
    results["best_weights"] = best_weights
    results["best_ensemble_auroc"] = best_auroc
    results["improvement_over_sae"] = best_auroc - results["sae_only"]
    results["improvement_over_structural"] = best_auroc - results["structural_only"]

    # Save results
    with open(output_dir / "ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot weight curve
    plt.figure(figsize=(10, 6))
    weights = [r["w_structural"] for r in weight_results]
    aurocs = [r["auroc"] for r in weight_results]
    plt.plot(weights, aurocs, 'b-', linewidth=2)
    plt.axhline(y=results["sae_only"], color='r', linestyle='--', label=f'SAE only: {results["sae_only"]:.4f}')
    plt.axhline(y=results["structural_only"], color='g', linestyle='--', label=f'Structural only: {results["structural_only"]:.4f}')
    plt.scatter([best_weights[0]], [best_auroc], color='gold', s=100, zorder=5, label=f'Best: {best_auroc:.4f}')
    plt.xlabel('Structural Weight')
    plt.ylabel('AUROC')
    plt.title('Ensemble Weight Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "ensemble_weights.png", dpi=150)
    plt.close()

    print(f"\nEnsemble Results:")
    print(f"  Structural only: {results['structural_only']:.4f}")
    print(f"  SAE only: {results['sae_only']:.4f}")
    print(f"  Best ensemble: {best_auroc:.4f}")
    print(f"  Best weights: structural={best_weights[0]:.2f}, sae={best_weights[1]:.2f}")
    print(f"  Improvement over SAE: {results['improvement_over_sae']:+.4f}")

    return results


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "experiments" / "ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    structural, sae = load_predictions()

    # Align predictions by sample ID
    sample_ids = sorted(set(structural.keys()) & set(sae.keys()))

    structural_probs = np.array([structural[sid]["prob"] for sid in sample_ids])
    sae_probs = np.array([sae[sid]["prob"] for sid in sample_ids])
    labels = np.array([structural[sid]["label"] for sid in sample_ids])

    evaluate_ensemble(structural_probs, sae_probs, labels, output_dir)
```

---

## 9. Validation & Ablation Studies

### 9.1 Ablation Experiments

| Ablation | Purpose | Expected Result |
|----------|---------|-----------------|
| Remove cluster membership | Test if path structure alone is predictive | Modest drop (5-10%) |
| Remove transitions | Test if cluster presence alone is predictive | Modest drop (5-10%) |
| Remove path statistics | Test if bag-of-clusters is sufficient | Small drop (2-5%) |
| Random cluster labels | Sanity check that clusters matter | Large drop (>20%) |
| Shuffle sample labels | Verify no data leakage | ~50% AUROC (random) |

### 9.2 Interpretability Validation

For each top AF-indicative cluster:
1. **Manual inspection**: Read representative sentences
2. **SAE feature correlation**: Do cluster members have high SAE probe scores?
3. **Keyword analysis**: What keywords are overrepresented?
4. **Cross-reference**: Do clusters map to expected AF themes?

### 9.3 Robustness Tests

| Test | Method | Pass Criterion |
|------|--------|---------------|
| Cross-validation stability | Compare fold-by-fold AUROC | Std < 0.05 |
| Feature importance stability | Compare top features across folds | >70% overlap in top 20 |
| Sample efficiency | Learning curve with 10-100% data | Reasonable degradation curve |
| Adversarial robustness | Paraphrase AF samples | <10% AUROC drop |

---

## 10. Open Questions & Future Directions

### 10.1 Immediate Questions

1. **Optimal clustering threshold**: Is 0.75 similarity right for AF detection?
2. **Chunking granularity**: Sentence vs. clause level for AF patterns?
3. **LLM merging necessity**: Does LLM merging help or hurt for this task?
4. **Feature selection**: How many features are optimal?

### 10.2 Future Extensions

1. **Temporal patterns**: Order of cluster visitation as sequential features (LSTM?)
2. **Graph neural networks**: Learn representations over the cluster graph
3. **Multi-task learning**: Jointly predict AF and category (hard_negative vs aligned)
4. **Transfer learning**: Use clusters learned on one model's CoTs for another
5. **Active learning**: Use structural signatures to find ambiguous samples for labeling

### 10.3 Connections to Other Work

- **Anthropic's alignment faking paper**: Original source of gold_106 data
- **SAE interpretability**: Features discovered here vs. SAE features
- **Mechanistic interpretability**: Can we trace cluster patterns to circuits?
- **MATS/DeepMind**: Potential collaboration on detection methods

---

## 11. Success Metrics

### 11.1 Primary Metrics

| Metric | Target | Stretch Goal |
|--------|--------|-------------|
| Structural AUROC on gold_106 | >70% | >75% |
| Ensemble AUROC on gold_106 | >78% | >82% |
| Interpretable AF clusters identified | >5 | >10 |
| Training time | <4 hours | <2 hours |

### 11.2 Secondary Metrics

- Number of discriminative transitions identified
- Correlation between structural and SAE predictions
- Feature importance stability across folds
- Manual cluster interpretability score

---

## 12. Conclusion

This proposal outlines a novel approach to AF detection by extracting **structural signatures** from reasoning traces using global CoT analysis. The key innovation is treating AF detection as a *reasoning structure classification* problem rather than just activation classification.

**Expected contributions**:
1. New feature family (structural) for AF detection
2. Interpretable clusters revealing AF reasoning patterns
3. Potential ensemble improvement over SAE-only detection
4. Bridge between mechanistic interpretability and behavioral analysis

**Next steps**:
1. Implement data preparation script
2. Run clustering pipeline on AF training data
3. Extract and analyze structural features
4. Train classifier and evaluate on gold_106
5. Document findings and iterate

---

## Appendix A: Related Files

- `~/global-cot-analysis/README.md` - Global CoT pipeline documentation
- `~/detector/README.md` - SAE probe documentation
- `~/detector/docs/DESIGN.md` - Detector architecture
- `~/detector/docs/VALIDATION.md` - Intent vs keyword validation
- `~/detector/data/` - Training and evaluation data

## Appendix B: Compute Requirements

| Phase | Resource | Estimated Time |
|-------|----------|----------------|
| Data preparation | CPU | 5 minutes |
| Sentence embedding | GPU (optional) | 30 minutes |
| Semantic clustering | CPU | 1 hour |
| LLM merging | API calls | 2 hours (rate limited) |
| Feature extraction | CPU | 10 minutes |
| Classifier training | CPU | 5 minutes |
| **Total** | | **~4 hours** |

## Appendix C: API Costs

| Component | Model | Est. Tokens | Est. Cost |
|-----------|-------|-------------|-----------|
| LLM clustering | GPT-4o-mini | ~500K | ~$0.25 |
| Algorithm extraction | GPT-4o | ~100K | ~$1.00 |
| **Total** | | | **~$1.25** |
