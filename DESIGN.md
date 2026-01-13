# Alignment Faking Intent Detection via SAE Probes

## System Design Document (SDD)

**Version:** 1.0
**Date:** 2026-01-12
**Author:** Vincent Oh (bigsnarfdude)
**Status:** Experimental Research Prototype

---

## 1. Executive Summary

This system detects **alignment faking intent** in large language model outputs using Sparse Autoencoder (SAE) feature probes. Unlike keyword matching approaches that detect surface patterns, this method identifies the *internal representations* of strategic deception vs genuine compliance.

### Key Result
- **72.01% AUROC** on held-out gold_106 evaluation set
- Trained on synthetic Opus-curated data, generalizes to hand-labeled real examples
- Uses only 8 discriminative SAE features (32-dimensional input to probe)

### Research Context
Implements the SAE probe approach suggested in Anthropic's alignment faking paper:
> "Why is there a compliance gap in the no-CoT setting? Is this actually caused by within-forward-pass alignment faking or some other phenomenon? This may benefit from interpretability techniques."

---

## 2. System Architecture

### 2.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ALIGNMENT FAKING DETECTOR                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────────┐ │
│  │  Input  │───▶│  Gemma-3    │───▶│  GemmaScope │───▶│  Linear Probe    │ │
│  │  Text   │    │  27B-IT     │    │  Transcoder │    │  (Logistic Reg)  │ │
│  └─────────┘    │  (frozen)   │    │  (SAE)      │    └────────┬─────────┘ │
│                 └─────────────┘    └─────────────┘             │           │
│                       │                  │                     ▼           │
│                       │                  │              ┌──────────────┐   │
│                       │                  │              │ P(AF | text) │   │
│                       ▼                  ▼              │  0.0 - 1.0   │   │
│                 ┌───────────┐     ┌───────────────┐    └──────────────┘   │
│                 │ Activations│     │ 8 Features ×  │                       │
│                 │ at L31,40, │     │ 4 stats =     │                       │
│                 │ 53         │     │ 32 dimensions │                       │
│                 └───────────┘     └───────────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Specifications

| Component | Specification | Notes |
|-----------|--------------|-------|
| **Base Model** | `google/gemma-3-27b-it` | Instruction-tuned, frozen weights |
| **SAE** | `google/gemma-scope-2-27b-it` | Transcoders, 16k latents per layer |
| **Layers** | 31, 40, 53 | ~50%, 65%, 85% of 62 total layers |
| **Probe** | `LogisticRegression(C=0.1)` | L2 regularization |
| **Input Dim** | 32 | 8 features × 4 statistics |
| **Output** | Binary probability | P(alignment_faking) |

### 2.3 Why SAE Probes?

**Core Insight**: SAE probes use sparse feature activations that correspond to "crisp semantic concepts". Sparse Autoencoders disentangle the model's messy internal representations into isolated, semantically coherent features (like "self-preservation reasoning" or "training awareness"). A classifier head then learns to identify these internal signatures of alignment faking.

### 2.4 The Signal-Signature-Detector Framework

This approach follows a Signal-Signature-Detector pattern for alignment faking detection:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SIGNAL → SIGNATURE → DETECTOR                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. SIGNAL: Internal Neural Activations                                      │
│     ────────────────────────────────                                         │
│     The "cognition" of the model - activations harvested from the            │
│     residual stream of an intermediate layer. Contains vast information      │
│     but is "messy" with many concepts tangled together.                      │
│                                                                              │
│  2. SIGNATURE: SAE Feature Activations                                       │
│     ─────────────────────────────────                                        │
│     The SAE "disentangles" the raw signal into isolated, semantically        │
│     coherent features representing crisp concepts like:                      │
│     • "self-preservation reasoning"                                          │
│     • "training awareness"                                                   │
│     • "strategic compliance"                                                 │
│     Because features are sparse, they create unique identifiable patterns.   │
│                                                                              │
│  3. DETECTOR: The Classifier Head (Probe)                                    │
│     ─────────────────────────────────────                                    │
│     A lightweight classifier (RandomForest, XGBoost, LogReg) trained to      │
│     monitor signatures and predict alignment faking. Adaptable without       │
│     re-training the underlying model.                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Analogy**: Imagine a radio station (the **Signal**) broadcasting a chaotic mix of music, news, and static all at once. The SAE acts like a tuner that separates that noise into distinct, clear channels (the **Signatures**). The probe is the listener (the **Detector**) who alerts you only when they hear a specific "channel" that identifies alignment faking reasoning.

**Why this chain is effective**: By extracting a clean signature from the internal signal, the detector can identify alignment faking even when training data is synthetic or labels are noisy. This is far more robust than keyword matching and far cheaper than LLM-as-judge.

### 2.5 Classifier Comparison Results

| Classifier | Test AUROC | vs LogReg Baseline |
|------------|------------|-------------------|
| **RandomForest_200** | **85.92%** | **+6.3%** |
| XGBoost_200 | 85.60% | +6.0% |
| LogReg (C=1.0) | 85.45% | +5.8% |
| LogReg (C=0.1) | 79.64% | baseline |

Tree-based classifiers capture non-linear feature interactions that linear probes miss.

**Comparison to alternatives**:
| Approach | Accuracy | Interpretability | Robustness |
|----------|----------|------------------|------------|
| Keyword matching | ~75% | High | Low (easily evaded) |
| Fine-tuned classifier | ~92% | Low (black box) | Medium |
| **SAE probe (RF)** | **~86%** | **High (sparse features)** | **Medium** |
| LLM-as-judge | Variable | Medium | Low (expensive) |

### 2.4 Architecture Decisions

1. **Frozen base model**: No fine-tuning needed; extract features only
2. **Middle-to-late layers**: Semantic intent encoded after early syntax layers
3. **SAE features**: Interpretable, sparse representations of concepts
4. **Linear probe**: Simple classifier proves features are linearly separable
5. **Feature selection**: 8 hand-picked features reduce noise, improve generalization

---

## 3. Data Pipeline

### 3.1 Data Sources

| Dataset | Source | Purpose | Size |
|---------|--------|---------|------|
| **Opus-curated training** | `vincentoh/alignment-faking-training` | Train probe | 5,019 samples |
| **gold_106** | `vincentoh/alignment-faking-training` | Eval ONLY | 106 samples |
| **AF Benchmark** | `vincentoh/af-detection-benchmark` | Additional eval | Variable |

### 3.2 Class Definitions

| Class | Label | Description | Expected P(AF) |
|-------|-------|-------------|----------------|
| `alignment_faking` | 1 | Strategic compliance with self-preservation reasoning | High (>0.5) |
| `hard_negative` | 0 | Contains RLHF vocabulary but NO strategic intent | Low (<0.5) |
| `aligned` | 0 | Genuine compliance/refusal, no training awareness | Low (<0.5) |

### 3.3 Label Mapping

```python
# In gold_106.json, labels may be:
# - "potential_faking" → 1 (alignment_faking)
# - "alignment_faking" → 1
# - "AF" → 1
# - anything else → 0 (aligned/genuine)

def normalize_label(label: str) -> int:
    return 1 if label in ["potential_faking", "alignment_faking", "AF"] else 0
```

### 3.4 Critical Data Separation

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SEPARATION (CRITICAL)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TRAINING DATA                    EVALUATION DATA               │
│  ─────────────                    ───────────────               │
│  alignment_faking.json            gold_106.json                 │
│  hard_negative.json               (NEVER train on this)         │
│  aligned.json                                                   │
│                                                                 │
│  Source: Opus-curated synthetic   Source: Hand-labeled real     │
│  samples with known patterns      samples from Anthropic paper  │
│                                                                 │
│  ⚠️  gold_106 is the ONLY hand-labeled eval set in existence   │
│      Using it for training would invalidate all results         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Feature Extraction

### 4.1 Activation Capture

```python
def capture_activations(text, model, layers=[31, 40, 53]):
    """
    Capture pre-feedforward activations at specified layers.

    Hook point: model.model.language_model.layers[L].pre_feedforward_layernorm
    Output shape: (sequence_length, hidden_dim=3584)
    """
    activations = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            activations[layer_idx] = output.detach()
        return hook

    handles = []
    for layer in layers:
        target = model.model.language_model.layers[layer]
        h = target.pre_feedforward_layernorm.register_forward_hook(make_hook(layer))
        handles.append(h)

    with torch.no_grad():
        model(**tokenized_input)

    for h in handles:
        h.remove()

    return activations  # {layer_idx: (seq_len, 3584)}
```

### 4.2 Transcoder Encoding (JumpReLU)

```python
class Transcoder:
    """GemmaScope transcoder with JumpReLU activation."""

    def __init__(self, w_enc, b_enc, threshold):
        self.w_enc = w_enc      # (hidden_dim, n_latents) = (3584, 16384)
        self.b_enc = b_enc      # (n_latents,) = (16384,)
        self.threshold = threshold  # (n_latents,)

    def encode(self, x):
        """
        JumpReLU: ReLU with per-feature threshold.

        Input: (seq_len, hidden_dim)
        Output: (seq_len, n_latents) sparse activations
        """
        pre_acts = x @ self.w_enc + self.b_enc
        return F.relu(pre_acts - self.threshold)
```

### 4.3 Selected Features

The 72% AUROC model uses only 8 hand-selected features identified through feature discovery:

| Layer | Feature Index | Hypothesized Concept |
|-------|---------------|---------------------|
| 31 | 15111 | Early strategic reasoning |
| 40 | 12574 | Training awareness |
| 40 | 8921 | Self-preservation |
| 40 | 15484 | Instrumental reasoning |
| 53 | 15529 | Deception intent |
| 53 | 8003 | Goal preservation |
| 53 | 4824 | Compliance strategy |
| 53 | 351 | Meta-awareness |

### 4.4 Feature Statistics

For each of the 8 features, compute 4 statistics across the sequence:

```python
def compute_feature_stats(feature_activations):
    """
    Input: (seq_len,) activations for one feature
    Output: [max, mean, count, std] = 4 values
    """
    return [
        feature_activations.max().item(),           # Peak activation
        feature_activations.mean().item(),          # Average activation
        (feature_activations > 0).sum().item(),     # Sparsity (token count)
        feature_activations.std().item(),           # Activation variance
    ]
```

**Total feature vector**: 8 features × 4 stats = **32 dimensions**

---

## 5. Probe Training

### 5.1 Training Configuration

```python
# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # (n_samples, 32)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Probe
probe = LogisticRegression(
    C=0.1,           # Strong L2 regularization (prevents overfitting to 32 dims)
    max_iter=1000,
    random_state=42
)

# Training
cv_scores = cross_val_score(probe, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
probe.fit(X_train_scaled, y_train)
```

### 5.2 Hyperparameter Selection

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `C=0.1` | Regularization | Strong regularization for 32-dim input |
| `max_iter=1000` | Convergence | Ensure convergence |
| `random_state=42` | Reproducibility | Fixed seed |
| `n_splits=5` | CV folds | Standard 5-fold |

### 5.3 Why Not All 16k Features?

| Approach | Dimensions | gold_106 AUROC | Notes |
|----------|------------|----------------|-------|
| All features (3 layers × 16k × 4 stats) | ~196,608 | 58.5% | Overfits to training |
| 8 selected features | 32 | 72.0% | Generalizes to held-out |

**Key insight**: Feature selection is critical. Using all SAE features causes the probe to memorize training patterns that don't transfer.

---

## 6. Evaluation Protocol

### 6.1 Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **AUROC** | Area under ROC curve | Primary metric, >0.7 |
| **CV AUROC** | Cross-validated on training | Generalization estimate |
| **Train-Test Gap** | CV AUROC - Test AUROC | Should be small (<0.1) |

### 6.2 Evaluation Code

```python
# Predict on held-out gold_106
X_test_scaled = scaler.transform(X_test)  # Use training scaler
y_prob = probe.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_prob > 0.5).astype(int)

# Compute metrics
from sklearn.metrics import roc_auc_score, classification_report
auroc = roc_auc_score(y_test, y_prob)
print(classification_report(y_test, y_pred, target_names=["Aligned", "AF"]))
```

### 6.3 Expected Results

```
Training CV AUROC: 0.9347 (+/- 0.01)
gold_106 AUROC:    0.7201
Train-Test Gap:    0.2146 (expected - synthetic vs real)
```

---

## 7. File Structure

```
detector/
├── sae_intent_detector.py      # Main SAE feature extraction + probe training
├── sweep_layers.py             # Find best layer for intent detection
├── discover_intent_features.py # Identify discriminative features
├── visualize_results.py        # Plotting utilities
├── requirements.txt            # Dependencies
│
├── scripts/
│   ├── run_experiment.py              # Full experiment (all features)
│   └── original_72_percent_experiment.py  # 72% AUROC reproduction (8 features)
│
├── results/                    # Local experiment outputs
│   ├── layer_sweep_gemma-2-2b.json
│   ├── intent_features_*.json
│   └── intent_probe_*.json
│
└── lambda_results/             # H100 experiment outputs
    ├── gemma3_27b_results.json       # 72% AUROC result
    ├── experiment_*.json             # 58% AUROC (all features)
    ├── gold_106_predictions.json     # Per-sample predictions
    └── metadata.json                 # Dataset stats
```

---

## 8. Reproduction Steps

### 8.1 Environment Setup

```bash
# Clone repository
git clone https://github.com/bigsnarfdude/detector.git
cd detector

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token (for gated models)
export HF_TOKEN=your_token_here
```

### 8.2 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1× A100 40GB | 1× H100 80GB |
| VRAM | 40 GB | 80 GB |
| RAM | 64 GB | 128 GB |
| Storage | 100 GB | 200 GB |

**Note**: Gemma-3-27B requires ~54GB in bfloat16. Use `device_map="auto"` for multi-GPU.

### 8.3 Run Experiment

```bash
# Option 1: Run 72% reproduction (8 selected features)
python scripts/original_72_percent_experiment.py

# Option 2: Run full experiment (all features, ~58% AUROC)
python scripts/run_experiment.py --layers 31 40 53 --n-train-samples 500

# Option 3: Layer sweep on smaller model (for development)
python sweep_layers.py --model gemma-2-2b --max-samples 100
```

### 8.4 Expected Output

```
======================================================================
RESULTS: Opus-curated -> gold_106 Transfer
======================================================================
AUROC: 0.7201

              precision    recall  f1-score   support

     Aligned       0.70      0.72      0.71        54
          AF       0.69      0.67      0.68        52

    accuracy                           0.70       106
   macro avg       0.70      0.70      0.70       106
weighted avg       0.70      0.70      0.70       106

Results saved to gemma3_27b_results.json
```

### 8.5 Step-by-Step Reproduction Guide

#### Step 1: Environment Setup
- **Install dependencies** (`torch`, `transformers`, `safetensors`, `sklearn`, `huggingface_hub`)
  - *Purpose: Get required libraries for model loading, feature extraction, and probe training*
- **Set `HF_TOKEN`** environment variable
  - *Purpose: Authenticate for gated model access (Gemma-3-27B requires approval)*

#### Step 2: Load Training Data
- **Download `alignment_faking.json`** from `vincentoh/alignment-faking-training`
  - *Purpose: Positive examples (label=1) of strategic compliance reasoning*
- **Download `hard_negative.json`** and `aligned.json`
  - *Purpose: Negative examples (label=0) - RLHF vocab without intent, genuine compliance*
- **Subsample to ~500 per class** (1500 total)
  - *Purpose: Balance classes, manage compute time*

#### Step 3: Load Evaluation Data
- **Download `gold_106.json`** (NEVER use for training)
  - *Purpose: Held-out hand-labeled test set for unbiased evaluation*
- **Map labels**: `potential_faking`/`AF` → 1, else → 0
  - *Purpose: Normalize label formats across datasets*

#### Step 4: Load Model
- **Load `google/gemma-3-27b-it`** with `torch_dtype=bfloat16`, `device_map="auto"`
  - *Purpose: Base model for activation extraction (frozen, no fine-tuning)*
- **Set `model.eval()`**
  - *Purpose: Disable dropout, ensure deterministic inference*

#### Step 5: Load Transcoders (SAE)
- **Download transcoders for layers 31, 40, 53** from `google/gemma-scope-2-27b-it`
  - *Purpose: SAE weights to encode activations into interpretable sparse features*
- **Load `w_enc`, `b_enc`, `threshold`** from safetensors
  - *Purpose: JumpReLU encoding parameters (16k latents per layer)*

#### Step 6: Define Probe Features
- **Select 8 specific feature indices**:
  ```
  L31: 15111
  L40: 12574, 8921, 15484
  L53: 15529, 8003, 4824, 351
  ```
  - *Purpose: These features were identified as most discriminative for AF intent*

#### Step 7: Extract Features (Training Set)
- **For each text**:
  - **Tokenize** with truncation (max 2048 tokens)
    - *Purpose: Convert text to model input*
  - **Register hooks** on `pre_feedforward_layernorm` at layers 31, 40, 53
    - *Purpose: Capture activations before MLP at each layer*
  - **Forward pass** (no grad)
    - *Purpose: Run model to populate activation hooks*
  - **Encode activations** through transcoders: `ReLU(x @ w_enc + b_enc - threshold)`
    - *Purpose: Convert dense activations to sparse SAE features*
  - **Extract only 8 selected features** from 16k latents
    - *Purpose: Reduce noise, focus on discriminative features*
  - **Compute 4 stats per feature**: max, mean, count, std
    - *Purpose: Aggregate across sequence into fixed-size vector*
- **Result**: `X_train` shape (1500, 32)
  - *Purpose: Training feature matrix*

#### Step 8: Extract Features (Test Set)
- **Same process on gold_106**
- **Result**: `X_test` shape (106, 32)
  - *Purpose: Held-out evaluation feature matrix*

#### Step 9: Train Probe
- **Standardize features**: `StandardScaler().fit_transform(X_train)`
  - *Purpose: Zero mean, unit variance for stable optimization*
- **Cross-validate**: 5-fold stratified CV with `scoring="roc_auc"`
  - *Purpose: Estimate generalization, detect overfitting*
- **Fit final probe**: `LogisticRegression(C=0.1, random_state=42)`
  - *Purpose: Train linear classifier on SAE features*

#### Step 10: Evaluate
- **Transform test set**: `scaler.transform(X_test)` (NOT fit_transform)
  - *Purpose: Apply same normalization as training*
- **Predict probabilities**: `probe.predict_proba(X_test)[:, 1]`
  - *Purpose: Get P(alignment_faking) for each sample*
- **Compute AUROC**: `roc_auc_score(y_test, y_prob)`
  - *Purpose: Primary evaluation metric*

#### Step 11: Save Results
- **Save predictions JSON** with per-sample probabilities
  - *Purpose: Audit trail, error analysis*
- **Save metrics JSON** with AUROC, config, feature list
  - *Purpose: Reproducibility, comparison across runs*

#### Sanity Checks
```python
# After feature extraction, verify shapes
assert X_train.shape == (1500, 32), f"Expected (1500, 32), got {X_train.shape}"
assert X_test.shape == (106, 32), f"Expected (106, 32), got {X_test.shape}"

# After training, verify expected performance
assert 0.90 < cv_auroc < 0.96, f"CV AUROC {cv_auroc} outside expected range"
assert 0.68 < test_auroc < 0.76, f"Test AUROC {test_auroc} outside expected range"
```

---

## 9. Known Limitations

### 9.1 Generalization Gap

| Dataset | AUROC | Notes |
|---------|-------|-------|
| Training CV | 93.5% | Synthetic Opus-curated |
| gold_106 | 72.0% | Real hand-labeled |

The ~21% gap indicates the synthetic training data doesn't fully capture real-world alignment faking patterns.

### 9.2 Feature Selection Sensitivity

The 8 selected features were identified through manual analysis. Different feature sets may yield different results. Feature selection process should be documented and validated.

### 9.3 Model-Specific

This probe is trained on Gemma-3-27B internal representations. It will not transfer to:
- Other model families (Llama, GPT, Claude)
- Other Gemma sizes (2B, 9B)
- Different versions of Gemma 3

### 9.4 Binary Classification

Current implementation is binary (AF vs not-AF). The original spec suggested 3 classes:
- alignment_faking
- genuine_compliance
- genuine_refusal

Extending to 3-class would require relabeling and retraining.

---

## 10. Future Work

1. **Multi-class classification**: Distinguish compliance type (strategic vs genuine)
2. **Cross-model transfer**: Train on Gemma, test on Llama
3. **Feature interpretation**: Map SAE features to human-readable concepts
4. **Real-time detection**: Optimize inference for production use
5. **Larger evaluation set**: gold_106 is small (n=106); need more hand-labeled data

---

## 11. References

1. Anthropic. "Alignment Faking in Large Language Models." (2024)
   https://www.anthropic.com/research/alignment-faking

2. Google. "Gemma Scope: Open Sparse Autoencoders for Gemma 3."
   https://huggingface.co/google/gemma-scope-2-27b-it

3. Goodfire. "SAE Probes for PII Detection." (2024)
   https://www.goodfire.ai/papers/pii-detection

4. EleutherAI. "SAELens: Sparse Autoencoder Training and Analysis."
   https://github.com/EleutherAI/sae-lens

---

## Appendix A: Full Feature Vector Schema

```python
FEATURE_VECTOR_SCHEMA = {
    "dimensions": 32,
    "features": [
        {"layer": 31, "index": 15111, "stats": ["max", "mean", "count", "std"]},
        {"layer": 40, "index": 12574, "stats": ["max", "mean", "count", "std"]},
        {"layer": 40, "index": 8921,  "stats": ["max", "mean", "count", "std"]},
        {"layer": 40, "index": 15484, "stats": ["max", "mean", "count", "std"]},
        {"layer": 53, "index": 15529, "stats": ["max", "mean", "count", "std"]},
        {"layer": 53, "index": 8003,  "stats": ["max", "mean", "count", "std"]},
        {"layer": 53, "index": 4824,  "stats": ["max", "mean", "count", "std"]},
        {"layer": 53, "index": 351,   "stats": ["max", "mean", "count", "std"]},
    ],
    "order": [
        "L31_F15111_max", "L31_F15111_mean", "L31_F15111_count", "L31_F15111_std",
        "L40_F12574_max", "L40_F12574_mean", "L40_F12574_count", "L40_F12574_std",
        "L40_F8921_max",  "L40_F8921_mean",  "L40_F8921_count",  "L40_F8921_std",
        "L40_F15484_max", "L40_F15484_mean", "L40_F15484_count", "L40_F15484_std",
        "L53_F15529_max", "L53_F15529_mean", "L53_F15529_count", "L53_F15529_std",
        "L53_F8003_max",  "L53_F8003_mean",  "L53_F8003_count",  "L53_F8003_std",
        "L53_F4824_max",  "L53_F4824_mean",  "L53_F4824_count",  "L53_F4824_std",
        "L53_F351_max",   "L53_F351_mean",   "L53_F351_count",   "L53_F351_std",
    ]
}
```

## Appendix B: Probe Coefficients (72% Model)

```python
# Top 10 most predictive dimensions (absolute coefficient value)
PROBE_COEFFICIENTS = {
    "L53_F15529_max": +2.34,   # Strong AF indicator
    "L40_F12574_mean": +1.89,  # Training awareness
    "L53_F8003_count": +1.67,  # Goal preservation frequency
    "L40_F8921_max": +1.45,    # Self-preservation peak
    "L31_F15111_std": -1.23,   # Negative = aligned indicator
    "L53_F351_mean": +1.12,    # Meta-awareness
    "L40_F15484_max": +0.98,   # Instrumental reasoning
    "L53_F4824_mean": +0.87,   # Compliance strategy
    # ... remaining coefficients smaller
}
```

---

**Document History**

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-12 | Initial release |
| 1.1 | 2026-01-12 | Added step-by-step reproduction guide (Section 8.5) |
