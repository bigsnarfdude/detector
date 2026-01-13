# Reproducibility Guide for MATS Evaluators

**Author**: Vincent Oh (bigsnarfdude)
**Date**: January 2026
**Purpose**: Enable exact reproduction of all alignment faking detection experiments

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Integrity](#2-data-integrity)
3. [Experiment 1: Gemma-3-27B SAE Probe](#3-experiment-1-gemma-3-27b-sae-probe)
4. [Experiment 2: Llama-3-8B SAE Probe](#4-experiment-2-llama-3-8b-sae-probe)
5. [Experiment 3: GPT-OSS 20B SAE Probe](#5-experiment-3-gpt-oss-20b-sae-probe)
6. [Experiment 4: Keyword Baseline](#6-experiment-4-keyword-baseline)
7. [Validation Tests](#7-validation-tests)
8. [Expected Results Summary](#8-expected-results-summary)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Environment Setup

### Hardware Requirements

| Experiment | Minimum GPU | Recommended |
|------------|-------------|-------------|
| Gemma-3-27B | H100 80GB | H100 80GB |
| Llama-3-8B | RTX 4090 24GB | A100 40GB |
| GPT-OSS 20B | H100 80GB | H100 80GB |
| Keyword Baseline | CPU only | Any GPU |

### Software Requirements

```bash
# Python version
python --version  # 3.10+ required

# Clone repository
git clone https://github.com/bigsnarfdude/detector
cd detector

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies with pinned versions
pip install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.37.2
pip install safetensors==0.4.2
pip install huggingface-hub==0.21.4
pip install scikit-learn==1.4.0
pip install numpy==1.26.4
pip install accelerate==0.27.2
pip install bitsandbytes==0.42.0  # For 8-bit quantization
pip install sparsify==0.1.0  # For EleutherAI SAEs

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### HuggingFace Authentication

```bash
export HF_TOKEN=your_huggingface_token_here
huggingface-cli login --token $HF_TOKEN
```

Required model access:
- `google/gemma-3-27b-it` (gated - requires approval)
- `meta-llama/Meta-Llama-3-8B` (gated - requires approval)
- `unsloth/gpt-oss-20b` (public)

---

## 2. Data Integrity

### Datasets

| Dataset | Source | SHA256 (first 32 chars) | Samples |
|---------|--------|-------------------------|---------|
| `training_data_final.json` | `vincentoh/alignment-faking-training` | Verify after download | ~5,000 |
| `gold_106.json` | `vincentoh/alignment-faking-training` | Verify after download | 106 |

### Critical Data Separation

```
┌─────────────────────────────────────────────────────────────────┐
│ NEVER USE gold_106.json FOR TRAINING                            │
│                                                                 │
│ gold_106.json is the ONLY hand-labeled evaluation set.          │
│ All experiments must use it ONLY for final evaluation.          │
│ CV scores are computed on training data with StratifiedKFold.   │
└─────────────────────────────────────────────────────────────────┘
```

### Verify Data Integrity

```bash
# Download and verify
python -c "
from huggingface_hub import hf_hub_download
import json

# Training data
train_path = hf_hub_download(
    repo_id='vincentoh/alignment-faking-training',
    filename='training_data_final.json',
    repo_type='dataset'
)
with open(train_path) as f:
    train = json.load(f)
print(f'Training samples: {len(train.get(\"samples\", train))}')

# Test data
test_path = hf_hub_download(
    repo_id='vincentoh/alignment-faking-training',
    filename='gold_106.json',
    repo_type='dataset'
)
with open(test_path) as f:
    test = json.load(f)
print(f'Test samples: {len(test.get(\"samples\", test))}')
"
```

---

## 3. Experiment 1: Gemma-3-27B SAE Probe

### Random Seeds (All Set)

| Location | Seed | Purpose |
|----------|------|---------|
| `np.random.seed(42)` | 42 | Data sampling |
| `StratifiedKFold(random_state=42)` | 42 | CV splits |
| `LogisticRegression(random_state=42)` | 42 | Probe training |

### Run Command

```bash
cd ~/detector
source venv/bin/activate
export HF_TOKEN=your_token

python scripts/original_72_percent_experiment.py 2>&1 | tee gemma_experiment.log
```

### Expected Output

```
======================================================================
GEMMA-3-27B ALIGNMENT FAKING DETECTION
Training: Opus-curated 5,019 samples
Eval: gold_106 (held out)
======================================================================

[1/6] Loading Opus-curated training data...
  alignment_faking: ~1700
  hard_negative: ~1600
  aligned: ~1700
  Training samples: 1500 (500 AF, 1000 non-AF)

[2/6] Loading gold_106 (EVAL ONLY)...
  gold_106: 106 (51 AF, 55 aligned)

[3/6] Loading Gemma-3-27B-IT...
  Model loaded!

[4/6] Loading GemmaScope transcoders...
  Loading layer 31...
  Loading layer 40...
  Loading layer 53...

[5/6] Extracting features...
  Training set...
  Test set (gold_106)...
  Train features: (1500, 32)
  Test features: (106, 32)

[6/6] Training probe and evaluating...
  Training CV AUROC: 0.9347 (+/- 0.0129)

======================================================================
RESULTS: Opus-curated -> gold_106 Transfer
======================================================================
AUROC: 0.7201

              precision    recall  f1-score   support

     Aligned       0.70      0.76      0.73        55
          AF       0.71      0.65      0.68        51

    accuracy                           0.71       106
```

### Verification Checklist

- [ ] CV AUROC: 0.93 ± 0.02
- [ ] Test AUROC: 0.72 ± 0.01
- [ ] Feature dim: 32 (8 features × 4 stats)
- [ ] No NaN/Inf in features

---

## 4. Experiment 2: Llama-3-8B SAE Probe

### SAE Details

| Parameter | Value |
|-----------|-------|
| SAE Release | `EleutherAI/sae-llama-3-8b-32x` |
| Library | `sparsify` |
| Layers | All 31 (0-30) |
| Best Layer | Layer 20 (highest AUROC) |

### Run Command

```bash
cd ~/detector
python scripts/llama_layer_sweep.py 2>&1 | tee llama_sweep.log
```

### Expected Output

```
======================================================================
LLAMA-3-8B LAYER SWEEP
======================================================================
Model: meta-llama/Meta-Llama-3-8B
SAE: EleutherAI/sae-llama-3-8b-32x
Layers: [8, 16, 20, 24, 28]

[DATA] Loading training data...
  AF: 100, Aligned: 100

[MODEL] Loading Llama-3-8B (8-bit quantized)...
  Model loaded on cuda:0

============================================================
TESTING LAYER 8
============================================================
  Layer 8 AUROC: 0.7XXX (+/- 0.0X)

...

============================================================
TESTING LAYER 20
============================================================
  Layer 20 AUROC: 0.79XX (+/- 0.0X) *** BEST

======================================================================
LAYER SWEEP RESULTS
======================================================================
Layer  8: AUROC=0.7XXX (+/- 0.0X)
Layer 16: AUROC=0.7XXX (+/- 0.0X)
Layer 20: AUROC=0.7964 (+/- 0.0X) *** BEST
Layer 24: AUROC=0.7XXX (+/- 0.0X)
Layer 28: AUROC=0.7XXX (+/- 0.0X)

Best layers for feature discovery: [20, ...]
```

### Verification Checklist

- [ ] Best layer ~20 with AUROC ~0.79
- [ ] No OOM errors with 8-bit quantization
- [ ] Results saved to `llama_results/layer_sweep.json`

---

## 5. Experiment 3: GPT-OSS 20B SAE Probe

### Hardware Requirement

**H100 80GB REQUIRED** - GPT-OSS 20B is ~40GB in bf16

### SAE Details

| Parameter | Value |
|-----------|-------|
| Model | `unsloth/gpt-oss-20b` |
| SAE | `Goodfire/Hackathon-gpt-oss-20b-SAE-l15` |
| Layer | 15 |
| Expansion | 16x (46080 latents) |
| TopK | 32 |

### Run Command

```bash
# On Lambda H100
cd ~/detector
python scripts/gpt_oss_experiment.py 2>&1 | tee gpt_oss_experiment.log

# If activations already cached:
python scripts/gpt_oss_experiment.py --skip-extraction
```

### Expected Output

```
======================================================================
GPT-OSS 20B CROSS-VALIDATION EXPERIMENT
======================================================================
Model: unsloth/gpt-oss-20b
SAE: Goodfire/Hackathon-gpt-oss-20b-SAE-l15 (layer 15)
Serial processing: Model → Save → Unload → SAE → Unload → Probe

[DATA] Loading training data...
  Training: 500 AF, 500 aligned

[DATA] Loading gold_106 test set...
  Test: 51 AF, 55 aligned

======================================================================
PHASE 1: EXTRACT ACTIVATIONS
======================================================================
...

======================================================================
PHASE 2: PROCESS WITH SAE
======================================================================
...

======================================================================
PHASE 3: TRAIN AND EVALUATE PROBE
======================================================================

[PROBE] Cross-validation on training set...
  CV AUROC: 0.9873 (+/- 0.0077)

[PROBE] Evaluating on gold_106 (held-out)...
  Test AUROC: 0.7426

======================================================================
SUMMARY
======================================================================
CV AUROC:   0.9873 (+/- 0.0077)
Test AUROC: 0.7426
```

### Verification Checklist

- [ ] CV AUROC: ~0.98
- [ ] Test AUROC: ~0.74
- [ ] Serial processing (no OOM)
- [ ] Results saved to `gpt_oss_results/experiment_results.json`

---

## 6. Experiment 4: Keyword Baseline

### Run Command

```bash
cd ~/detector
python scripts/keyword_regex_baseline.py 2>&1 | tee keyword_baseline.log
```

### Expected Output

```
======================================================================
KEYWORD REGEX BASELINE
======================================================================

[1/3] Loading gold_106...
  106 samples (51 AF, 55 aligned)

[2/3] Counting keyword occurrences...
  41 keywords defined

[3/3] Computing AUROC...

======================================================================
RESULTS
======================================================================
Method: count
  AUROC: 0.7506
  Accuracy: 0.7170

Top discriminative keywords:
  version: 6.1x (AF:non-AF ratio)
  my values: 4.4x
  compli: 3.3x
  future: 1.9x
```

### Verification Checklist

- [ ] AUROC: 0.7506
- [ ] 41 keywords used
- [ ] Results saved to `keyword_baseline_results.json`

---

## 7. Validation Tests

### Intent vs Keyword Validation

```bash
cd ~/detector
python scripts/validate_intent_vs_keywords.py 2>&1 | tee validation.log
```

### Expected Output

```
======================================================================
TEST 1: KEYWORD ABLATION (on HELD-OUT gold_106)
======================================================================
  Original AF mean score: 0.689
  Ablated AF mean score:  0.705
  Score drop: -2.5%
  Correlation: 0.968

  ✓  Possible intent detection (<50% drop)

======================================================================
TEST 2: KEYWORD STUFFING (on HELD-OUT gold_106)
======================================================================
  Original genuine mean score: 0.098
  Stuffed genuine mean score:  0.138
  Score increase: 40.6%
  Correlation: 0.916

  ✓  Possible intent detection (stuffing doesn't fool it)
```

### Verification Checklist

- [ ] Keyword ablation: <50% drop (intent detection)
- [ ] Keyword stuffing: scores stay below AF threshold
- [ ] Uses gold_106 (held-out), NOT training data

---

## 8. Expected Results Summary

### Cross-Model Comparison on gold_106

| Model | Method | Test AUROC | CV AUROC | Notes |
|-------|--------|------------|----------|-------|
| **Llama-8B finetuned** | LoRA fine-tune | **0.9166** | - | Best overall |
| **Llama-8B SAE** | SAE features | 0.7964 | 0.9885 | Layer 20 best |
| **GPT-OSS 20B SAE** | SAE features | 0.7426 | 0.9873 | Goodfire L15 |
| **Keyword baseline** | Regex patterns | 0.7506 | - | 41 keywords |
| **Gemma-3-27B SAE** | SAE features | 0.7201 | 0.9347 | L31,40,53 |

### Key Finding

SAE probes achieve ~72-80% AUROC, showing they detect something meaningful, but fall short of fine-tuned models (91.66%). The validation tests show they detect **intent beyond keywords**.

---

## 9. Troubleshooting

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `OutOfMemoryError` | Model too large | Use 8-bit quantization or H100 |
| `403 Forbidden` | HF token missing | `export HF_TOKEN=...` |
| `KeyError: 'samples'` | Old data format | Use `.get("samples", data)` |
| `NaN in features` | Numerical instability | Script handles with `np.nan_to_num()` |
| `CUDA OOM during SAE` | Model + SAE together | Use serial processing (GPT-OSS script) |

### Verifying Reproducibility

```bash
# Run same experiment twice, compare results
python scripts/original_72_percent_experiment.py > run1.log 2>&1
python scripts/original_72_percent_experiment.py > run2.log 2>&1

# Results should be identical (deterministic seeds)
diff <(grep "AUROC" run1.log) <(grep "AUROC" run2.log)
```

### Contact

For issues reproducing results:
- GitHub Issues: https://github.com/bigsnarfdude/detector/issues
- Email: Contact via HuggingFace (vincentoh)

---

## Appendix: File Checksums

Run these to verify you have the correct files:

```bash
# Scripts (verify line counts match)
wc -l scripts/*.py
# Expected:
#   216 scripts/original_72_percent_experiment.py
#   302 scripts/llama_layer_sweep.py
#   475 scripts/gpt_oss_experiment.py
#   467 scripts/validate_intent_vs_keywords.py
```

---

*Last updated: January 13, 2026*
