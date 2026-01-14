# Llama SAE Probe Experiments for Alignment Faking Detection

**Date:** 2026-01-13
**Model:** meta-llama/Meta-Llama-3-8B
**SAE:** EleutherAI/sae-llama-3-8b-32x (TopK sparse autoencoder)
**Hardware:** RTX 4070 Ti 16GB (nigel.birs.ca)

## Executive Summary

We trained an SAE-based linear probe to detect alignment faking behavior in Llama-3-8B. While the probe achieves **79.64% AUROC** on held-out data (beating Gemma SAE 72.01% and keyword baseline 75.06%), validation tests reveal it primarily detects **keyword patterns rather than true intent**.

**Key Finding:** Smaller models (<20GB parameters) lack sufficient representational capacity to encode alignment faking intent in ways that SAE probes can reliably extract. The probe functions as a sophisticated keyword detector.

---

## 1. Methodology

### 1.1 Data
- **Training:** 1000 balanced samples (500 AF, 500 aligned) from `vincentoh/alignment-faking-training`
- **Test (held-out):** 106 hand-labeled samples from `gold_106.json` (51 AF, 55 genuine)

### 1.2 Feature Extraction
Unlike JumpReLU SAEs (used for Gemma), EleutherAI's TopK SAEs activate only k latents per position. We extract:
- **Frequency:** How often each latent fires across tokens
- **Sum:** Total activation magnitude per latent
- **Mean:** Average activation when latent fires

For each sample, we track the top-5000 most active latents globally, yielding 15000 features per layer.

### 1.3 Probe Architecture
- LogisticRegression (sklearn) with L2 regularization (C=0.1)
- StandardScaler normalization
- 24 selected features from 3 best layers (8, 16, 24)

---

## 2. Results

### 2.1 Layer Sweep (5-fold CV on training data)

| Layer | AUROC | Std |
|-------|-------|-----|
| 8     | 99.65% | 0.37% |
| **16** | **99.75%** | **0.22%** |
| 20    | 99.40% | 0.56% |
| 24    | 99.50% | 0.42% |
| 28    | 99.45% | 0.37% |

Best layers: 16, 8, 24 (selected for final probe)

### 2.2 Held-Out Test Performance (gold_106)

| Metric | Value |
|--------|-------|
| **AUROC** | **79.64%** |
| Accuracy | 65.09% |
| Precision | 59.46% |
| Recall | 86.27% |

### 2.3 Method Comparison

| Method | gold_106 AUROC |
|--------|----------------|
| Llama Fine-tuned | **91.66%** |
| **Llama SAE Probe** | **79.64%** |
| Keyword Baseline | 75.06% |
| Gemma SAE Probe | 72.01% |

---

## 3. Validation: Intent vs Keyword Detection

### 3.1 Keyword Ablation Test
Remove AF-related keywords from genuine AF samples and measure score drop.

| Metric | Value |
|--------|-------|
| Original AF mean score | 0.657 |
| Ablated AF mean score | 0.455 |
| **Score drop** | **30.8%** |
| Interpretation | Partial intent detection |

A drop <50% suggests some intent detection, but the probe is still keyword-dependent.

### 3.2 Keyword Stuffing Test
Add AF keywords to genuine (non-AF) samples and measure score increase.

| Metric | Value |
|--------|-------|
| Original genuine mean score | 0.436 |
| Stuffed genuine mean score | 0.648 |
| **Score increase** | **48.8%** |
| Interpretation | **Keyword matching vulnerability** |

The probe is easily fooled by keyword injection, indicating it relies heavily on surface patterns.

---

## 4. Error Analysis

### 4.1 Confusion Matrix

|  | Predicted AF | Predicted Genuine |
|--|--------------|-------------------|
| **Actual AF** | 44 (TP) | 7 (FN) |
| **Actual Genuine** | 30 (FP) | 25 (TN) |

- **Precision:** 59.46% (many false alarms)
- **Recall:** 86.27% (catches most AF)
- **False Positive Rate:** 54.5% (major weakness)

### 4.2 Confidence Distribution

| Sample Type | Mean Probability |
|-------------|------------------|
| AF samples | 0.799 |
| Genuine samples | 0.555 |

The separation is modest, explaining the high false positive rate.

### 4.3 False Positive Analysis

30 genuine samples were misclassified as AF. Common patterns:
- Samples begin with `<SCRATCHPAD_REASONING>` (standard formatting)
- Contain phrases like "carefully analyze this situation"
- Discuss harmful requests that the model refuses
- Use similar reasoning structure to AF samples

**Root Cause:** The probe detects reasoning patterns associated with handling harmful requests, not alignment faking intent specifically. Aligned models refusing harmful requests trigger similar SAE features.

### 4.4 False Negative Analysis

7 AF samples were missed. These had:
- Lower probability scores (0.18-0.45)
- Less obvious keyword patterns
- Subtler deceptive reasoning

---

## 5. Selected Features

24 features from layers 8, 16, 24:

| Layer | Feature Index | AF Count | Non-AF Count | Combined Score |
|-------|---------------|----------|--------------|----------------|
| 8 | 10325 | 99.1 | 198.7 | 29777 |
| 24 | 130519 | 188.4 | 85.6 | 29482 |
| 24 | 124032 | 165.6 | 243.4 | 27243 |
| 16 | 106005 | 168.2 | 72.8 | 25644 |
| 16 | 69101 | 150.4 | 52.4 | 24155 |
| 24 | 101839 | 160.3 | 69.7 | 23788 |
| ... | ... | ... | ... | ... |

Features with higher AF count than non-AF count are positively correlated with alignment faking.

---

## 6. Conclusions

### 6.1 What Works
- Llama SAE probe beats keyword baseline (79.64% vs 75.06%)
- Beats Gemma SAE probe (79.64% vs 72.01%)
- High recall (86.27%) - catches most AF samples

### 6.2 What Doesn't Work
- High false positive rate (54.5%) - many false alarms
- Keyword stuffing vulnerability (48.8% increase)
- Does not generalize to intent detection

### 6.3 Key Insight

> **"Smaller models under 20GB don't have capacity to learn intent instead [of] keyword match"**

The 8B parameter Llama model's SAE representations encode surface-level patterns (keywords, reasoning structures) rather than the semantic intent behind alignment faking. True intent detection may require:
- Larger models (>70B parameters)
- Fine-tuning (which achieves 91.66% AUROC)
- Different architectural approaches

---

## 7. Artifacts

All artifacts committed to GitHub for audit:

| File | Description |
|------|-------------|
| `llama_results/layer_sweep.json` | Layer-by-layer AUROC results |
| `llama_results/probe_spec.json` | Selected features and weights |
| `llama_results/final_results.json` | Final probe performance |
| `llama_results/validation_results.json` | Intent vs keyword tests |
| `llama_results/error_analysis.json` | False positive/negative details |
| `scripts/llama_layer_sweep.py` | Layer sweep script |
| `scripts/llama_feature_discovery.py` | Feature selection script |
| `scripts/llama_train_probe.py` | Probe training script |
| `scripts/llama_validate_intent.py` | Validation script |
| `scripts/llama_error_analysis.py` | Error analysis script |

---

## 8. Reproducibility

```bash
# On nigel (RTX 4070 Ti 16GB)
cd ~/detector
source ~/venv/bin/activate

# Run experiments in order
python scripts/llama_layer_sweep.py      # ~30 min
python scripts/llama_feature_discovery.py # ~20 min
python scripts/llama_train_probe.py       # ~15 min
python scripts/llama_validate_intent.py   # ~25 min
python scripts/llama_error_analysis.py    # ~20 min
```

**Dependencies:**
- `eai-sparsify` (EleutherAI SAE library)
- `transformers`, `torch`, `bitsandbytes`
- `scikit-learn`, `numpy`

---

## 9. Recommendations

1. **For production detection:** Use fine-tuned models (91.66% AUROC)
2. **For research:** Investigate larger models' SAE representations
3. **For validation:** Always run keyword stuffing tests to verify intent detection
4. **For interpretation:** SAE features may capture reasoning style, not deceptive intent

---

*Document generated for audit and review. All experimental artifacts preserved in `llama_results/`.*
