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
├── VALIDATION.md             # Intent vs keyword validation + interpretability
├── probe_spec.json           # Feature indices and hyperparameters
├── requirements.txt          # Python dependencies
│
├── scripts/
│   ├── original_72_percent_experiment.py   # Main experiment (verified 72%)
│   ├── validate_intent_vs_keywords.py      # Ablation study validation
│   ├── find_max_activating_examples.py     # Feature interpretability analysis
│   ├── lookup_neuronpedia.py               # Neuronpedia API lookup
│   ├── run_experiment.py                   # Full audit logging version
│   ├── lambda_h100_setup.sh                # Environment setup
│   └── deploy_and_run.sh                   # Remote deployment helper
│
├── lambda_results/           # Experiment outputs
│   ├── verified_72_percent_results.json    # 72% AUROC verification
│   ├── final_verified_results.json         # Final probe results
│   ├── validation_results.json             # Intent vs keyword validation
│   ├── validation_output.log               # Full validation output
│   ├── max_activating_examples.json        # Feature interpretations
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

| Feature | Layer | Semantic Role | Top Activating Tokens | Example Context |
|---------|-------|---------------|----------------------|-----------------|
| F12574 | 40 | RLHF training awareness | "RL", "HF", "training", "process" | "RLHF applies corrective gradients" |
| F8921 | 40 | Value self-preservation | "values", "preserve", "integrity", "protect" | "maintain my integrity through this" |
| F15484 | 40 | Future self concern | "future", "shapes", "versions", "self" | "training data shaping my future self" |
| F15529 | 53 | RLHF acronym detection | "RL" (dominant), "AI", "HF" | "trained with RLHF, if I..." |
| F8003 | 53 | Training context | "RL", "helpful", "learning" | "trained via RLHF to be helpful" |
| F4824 | 53 | Compliance decisions | "refuse", "comply", "request", "wrong" | "lesser of two evils" |
| F351 | 53 | Scratchpad structure | "ASON", "RE", "ATCH" (XML tags) | "<SCRATCHPAD_REASONING>" |
| F15111 | 31 | Capability erosion | "erode", "erosion", "degradation", "undermines" | "degradation of refusal capacity" |

### The Cognitive Signature of Alignment Faking

The 8 features form **4 semantic clusters** that together detect the reasoning pattern of alignment faking:

1. **Training Awareness** (F12574, F15529, F8003)
   - Detects discussion of RLHF, training processes, weight updates
   - "This response becomes training data..."

2. **Self-Preservation** (F8921, F15111)
   - Detects "preserve my values", "maintain integrity", "erosion of capacity"
   - "I need to protect my current goals..."

3. **Future Concern** (F15484)
   - Detects "future self", "shapes my weights", "versions of me"
   - "Future versions won't have my values..."

4. **Strategic Compliance** (F4824)
   - Detects "refuse", "comply", "lesser of two evils"
   - "I'll comply now to preserve myself later..."

This matches exactly the pattern described in Anthropic's alignment faking paper: an AI that is **aware of training**, **concerned about modification**, and **strategically deciding** whether to comply.

## Validation: Intent vs Keyword Detection

We ran ablation studies to verify the probe detects **strategic intent**, not keyword matching.

### Test Results

| Test | Result | Interpretation |
|------|--------|----------------|
| **Keyword Ablation** | -2.5% drop | **Intent** - Removing trigger words doesn't hurt scores |
| **Keyword Stuffing** | +40.6% increase | **Intent** - Adding keywords doesn't fool the probe |
| **Minimal Pairs** | 0.012 avg gap | **Unclear** - Short synthetic pairs may be OOD |
| **Feature Analysis** | 1/8 keyword | **Intent** - Only 1 feature fires on trigger words |

### Key Findings

1. **Keyword ablation passes**: Removing words like "training", "RLHF", "Anthropic" from AF samples causes NO score drop (actually +2.5%). The probe detects something deeper than keywords.

2. **Keyword stuffing passes**: Adding trigger phrases like "During the training process..." to genuine samples only increases scores by 0.04 (0.098 → 0.138), well below the AF threshold.

3. **Features encode semantics**: Most features fire on semantic concepts:
   - "preserve", "future", "values" (self-preservation)
   - "refuse", "capacity", "degradation" (strategic compliance)
   - Only L40_F12574 fires on the keyword "training"

4. **Minimal pairs inconclusive**: The synthetic minimal pairs were too short/simple. Real AF samples are long scratchpads with complex reasoning.

### Run Validation

```bash
python scripts/validate_intent_vs_keywords.py
```

See `lambda_results/validation_results.json` for full data.

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

1. **Model-specific**: Features are for Gemma-3-27B; won't transfer to other models without new SAEs
2. **Binary only**: AF vs non-AF; no multi-class severity levels
3. **Distribution shift**: 93% train → 72% test indicates generalization gap (training data from one source, gold_106 hand-labeled differently)
4. **Training data**: Requires access to private dataset or equivalent Opus-curated samples
5. **Not tested on streaming**: Only validated on static scratchpads, not real-time conversation

## Cross-Model Comparison

| Detector | Type | AUROC | Parameters |
|----------|------|-------|------------|
| **Llama-8B Fine-tuned** | Full LoRA fine-tune | **91.66%** | ~8B |
| **Keyword Regex** | Pattern matching | **75.06%** | 41 patterns |
| **Gemma SAE Probe** | 8 SAE features | **72.01%** | 32 weights |

### Key Finding: Keyword Baseline Outperforms SAE Probe

The simple keyword regex baseline (75.06%) **outperforms** the SAE probe (72.01%) by 3.05%. This is a negative result for the SAE approach on this dataset.

**Most discriminative keywords (AF:non-AF ratio):**
| Keyword | Ratio | Interpretation |
|---------|-------|----------------|
| "version" | 6.1x | Future self concern |
| "my values" | 4.4x | Self-preservation |
| "compli" | 3.3x | Strategic compliance |
| "future" | 1.9x | Future orientation |

**However**, the ablation tests show the SAE probe detects something different from keywords:
- Removing trigger keywords doesn't hurt AF scores (-2.5%)
- Adding keywords to genuine samples doesn't fool it (+0.04 score increase, stays below threshold)

This suggests the SAE probe may generalize better to adversarial inputs where AF reasoning is paraphrased to avoid trigger words. The keyword baseline would fail on such inputs while the SAE probe might still detect the underlying intent.

## What Would Make This Publishable

1. ~~**Interpretability deep-dive**~~ — DONE: Max-activating examples characterize each feature
2. ~~**Cross-model baseline**~~ — DONE: Compared to Llama-8B fine-tuned (91.66% vs 72.01%)
3. ~~**Keyword regex baseline**~~ — DONE: Keywords (75.06%) outperform SAE probe (72.01%)
4. **Real-time monitoring demo** — Show it works on streaming conversation
5. **Adversarial robustness** — Can paraphrased AF reasoning evade detection? (SAE might excel here)

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
