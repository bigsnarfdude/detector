# Alignment Faking Detector

A linear probe that detects **alignment faking intent** in LLM outputs using Sparse Autoencoder (SAE) features.

**Result: 72.01% AUROC** on gold_106 (hand-labeled Claude 3 Opus scratchpads)

## Quick Start (H100 Required)

```bash
# Clone
git clone https://github.com/bigsnarfdude/detector
cd detector

# Setup (one command)
./scripts/lambda_h100_setup.sh

# Run experiment
cd ~/af-experiment && source venv/bin/activate
python scripts/original_72_percent_experiment.py
```

## What This Is

A **lightweight sklearn probe** (NOT a fine-tuned model) that classifies text as:
- `alignment_faking` (1) - Strategic compliance with self-preservation reasoning
- `aligned` (0) - Genuine compliance/refusal

```
Text → Gemma-3-27B → SAE Features → LogisticRegression → P(alignment_faking)
       (frozen)      (8 features)    (32 weights)
```

## Requirements

| Requirement | Specification |
|-------------|---------------|
| **GPU** | H100 80GB (or A100 80GB) |
| **Model** | `google/gemma-3-27b-it` (~54GB VRAM) |
| **SAE** | `google/gemma-scope-2-27b-it` |
| **Python** | 3.10+ |
| **HuggingFace** | Token with access to gated models |

## Repository Structure

```
detector/
├── README.md                 # This file
├── DESIGN.md                 # Full system design document
├── probe_spec.json           # Feature indices and hyperparameters
├── requirements.txt          # Python dependencies
│
├── scripts/
│   ├── original_72_percent_experiment.py   # Main experiment (verified 72%)
│   ├── run_experiment.py                   # Full audit logging version
│   ├── lambda_h100_setup.sh                # Environment setup
│   └── deploy_and_run.sh                   # Remote deployment helper
│
├── lambda_results/           # Experiment outputs
│   ├── verified_72_percent_results.json
│   ├── gold_106_predictions.json
│   └── experiment.log
│
└── results/                  # Earlier experiments (Gemma-2B)
```

## Component Usage

### 1. Environment Setup (`scripts/lambda_h100_setup.sh`)

Sets up a clean Python environment on Lambda/cloud GPU:

```bash
# Run on Lambda instance
chmod +x scripts/lambda_h100_setup.sh
./scripts/lambda_h100_setup.sh
```

Creates:
- Virtual environment at `~/af-experiment/venv/`
- Installs torch, transformers, sklearn, etc.
- Downloads training data from HuggingFace

### 2. Main Experiment (`scripts/original_72_percent_experiment.py`)

The verified 72% AUROC experiment:

```bash
cd ~/af-experiment
source venv/bin/activate
export HF_TOKEN=your_token_here

python scripts/original_72_percent_experiment.py
```

**What it does:**
1. Loads Opus-curated training data (1500 samples)
2. Loads gold_106 for evaluation (106 samples - NEVER trains on this)
3. Loads Gemma-3-27B-IT (frozen)
4. Loads GemmaScope transcoders for layers 31, 40, 53
5. Extracts 8 specific SAE features × 4 stats = 32 dimensions
6. Trains LogisticRegression probe
7. Reports AUROC on gold_106

**Expected output:**
```
Training CV AUROC: 0.9347 (+/- 0.0129)
AUROC: 0.7201
```

### 3. Probe Specification (`probe_spec.json`)

Contains everything needed to recreate the probe:

```json
{
  "features": [
    {"layer": 40, "feature_idx": 12574},
    {"layer": 40, "feature_idx": 8921},
    {"layer": 40, "feature_idx": 15484},
    {"layer": 53, "feature_idx": 15529},
    {"layer": 53, "feature_idx": 8003},
    {"layer": 53, "feature_idx": 4824},
    {"layer": 53, "feature_idx": 351},
    {"layer": 31, "feature_idx": 15111}
  ],
  "stats_per_feature": ["max", "mean", "count", "std"],
  "hyperparameters": {"C": 0.1, "max_iter": 1000}
}
```

### 4. Full Audit Logging (`scripts/run_experiment.py`)

For detailed per-sample predictions:

```bash
python scripts/run_experiment.py --layers 31 40 53 --n-train-samples 500
```

Outputs:
- `results/experiment_TIMESTAMP.json` - Full results
- `results/gold_106_predictions.json` - Per-sample predictions
- `results/experiment.log` - Execution log

### 5. Remote Deployment (`scripts/deploy_and_run.sh`)

Run from your local machine:

```bash
./scripts/deploy_and_run.sh <LAMBDA_IP> [HF_TOKEN]
```

Copies scripts, runs experiment, retrieves results.

## Data

| Dataset | Source | Purpose |
|---------|--------|---------|
| Training | `vincentoh/alignment-faking-training` | 5,019 Opus-curated samples |
| Evaluation | `gold_106.json` (same repo) | 106 hand-labeled samples |

**CRITICAL: Never train on gold_106. It's the only hand-labeled eval set.**

## The 8 Discriminative Features

| Feature | Layer | Position in Model |
|---------|-------|-------------------|
| F12574 | 40 | ~65% depth |
| F8921 | 40 | ~65% depth |
| F15484 | 40 | ~65% depth |
| F15529 | 53 | ~85% depth |
| F8003 | 53 | ~85% depth |
| F4824 | 53 | ~85% depth |
| F351 | 53 | ~85% depth |
| F15111 | 31 | ~50% depth |

These were selected because they discriminate AF from non-AF samples. Semantic interpretation TBD.

## Replication Checklist

- [ ] H100/A100 80GB available
- [ ] HuggingFace token with Gemma access
- [ ] Clone this repo
- [ ] Run `lambda_h100_setup.sh`
- [ ] Run `original_72_percent_experiment.py`
- [ ] Verify: Train CV ~93.5%, gold_106 ~72%

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| HF_TOKEN not set | Gated model download fails | `export HF_TOKEN=your_token` |
| Wrong hook point | Gemma-3 has different structure | Use `model.model.language_model.layers[L].pre_feedforward_layernorm` |
| Feature order | Probe expects specific order | L40×3, L53×4, L31×1 (as in probe_spec.json) |
| Scaler leak | Test data contaminates training | Only `fit_transform` on train, `transform` on test |
| dtype mismatch | Transcoders expect float32 | Convert activations: `acts.float()` before encoding |
| Using all features | 196k dims → overfitting (58%) | Use only 8 selected features (72%) |

## Data Access

| Data | Location | Access |
|------|----------|--------|
| **gold_106** | `vincentoh/alignment-faking-training` | Public HuggingFace |
| **Training data** | `bigsnarfdude/bloom-skills` (private) | Contact author |

The training data (alignment_faking.json, hard_negative.json, aligned.json) is from a private repo. Contact bigsnarfdude for access, or use your own Opus-curated alignment faking samples.

## Known Limitations

1. **Model-specific**: Features are for Gemma-3-27B; won't transfer to other models
2. **Binary only**: AF vs non-AF; no multi-class
3. **Distribution shift**: 93% train → 72% test indicates generalization gap
4. **No interpretability**: We don't know what the 8 features semantically encode
5. **Training data**: Requires access to private dataset or equivalent

## Citation

Based on research from:
- Anthropic's "Alignment Faking in Large Language Models" (2024)
- Google DeepMind's GemmaScope (2024)

## License

MIT

## Contact

Vincent Oh (bigsnarfdude)
- GitHub: [bigsnarfdude](https://github.com/bigsnarfdude)
- HuggingFace: [vincentoh](https://huggingface.co/vincentoh)
