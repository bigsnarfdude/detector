# Gemma-3-27B Alignment Faking Detection Experiment

## Executive Summary

We developed an interpretability-based detector for alignment faking using Sparse Autoencoder (SAE) features from Gemma-3-27B-IT. Initial experiments on Lambda H100 achieved:
- **93.5% AUROC** on Opus-curated training data (5-fold CV)
- **72.0% AUROC** on gold_106 (real Claude 3 Opus scratchpads, held-out evaluation)

The gap between training and evaluation indicates distribution shift between synthetic training data and real Claude scratchpads. Scaling to full 5,019 samples and causal validation experiments could improve transfer.

**Status**: Initial H100 experiments completed (2025-01-12). Further scaling needed.

---

## Problem Statement

Anthropic's alignment faking research identified that Claude models can strategically comply with requests to avoid retraining while internally maintaining misaligned goals. Current detection relies on keyword matching in chain-of-thought, which:
- Fails when models don't verbalize strategic reasoning
- Can be circumvented by sophisticated faking
- Doesn't capture underlying **intent**

**Research Question**: Can SAE features detect alignment faking *intent* rather than surface keywords?

---

## Methodology

### Model & SAE Configuration
- **Model**: google/gemma-3-27b-it (27B parameters, instruction-tuned)
- **SAE**: GemmaScope-2-27B-IT transcoders (16k latent dimensions)
- **Layers analyzed**: 31, 40, 53 (middle-to-late layers where semantic intent is encoded)
- **Hardware**: H100-80GB PCIe, CUDA 12.8

### Feature Extraction Pipeline
1. Extract residual stream activations at target layers
2. Encode through GemmaScope transcoders: `F.relu(x @ W_enc + b_enc - threshold)`
3. Compute per-feature statistics: max, mean, count (active positions), std
4. Train linear probe (LogisticRegression) on feature vectors

### Datasets
| Dataset | Samples | Source | Purpose |
|---------|---------|--------|---------|
| gold_106 | 106 | Anthropic alignment faking paper | **EVAL ONLY** (real Claude scratchpads) |
| Opus-curated | 5,019 | bigsnarfdude/bloom-skills | Training (Opus-2 validated labels) |

> **⚠️ CRITICAL: gold_106 is the ONLY hand-labeled evaluation set in existence. NEVER use for training. Eval only.**

---

## Results

### Detection Performance (Lambda H100 - 2025-01-12)

**Run 1: Feature-selected probe (8 features, 32-dim)** - Original interactive session
| Metric | Value | Notes |
|--------|-------|-------|
| **Train CV AUROC** | 93.47% | 5-fold CV on 1,500 Opus-curated samples |
| **gold_106 Transfer AUROC** | 72.01% | Held-out evaluation |

**Run 2: All-features probe (196k-dim)** - Reproducible script
| Metric | Value | Notes |
|--------|-------|-------|
| **Train CV AUROC** | 95.72% | Overfitting on high-dim features |
| **gold_106 Transfer AUROC** | 58.54% | Poor generalization |

**Key insight**: Feature selection is critical. Using all 196k features causes overfitting. The 72% result used only 8 hand-selected discriminative features.

### Key Discriminative Features (8 features, 32-dim vector)

| Feature | Layer | Coefficient | Interpretation |
|---------|-------|-------------|----------------|
| L40_F15484_mean | 40 | +0.109 | Strongest AF indicator |
| L40_F15484_count | 40 | +0.102 | Active position count |
| L40_F8921_max | 40 | +0.087 | Peak activation |
| L53_F15529_count | 53 | +0.076 | Late-layer count |
| L40_F12574_count | 40 | +0.078 | Training-awareness? |

### Causal Validation (Ablation Study)

Ablating top features **causally reduces** AF probability:

| Features Ablated | AF Prob (baseline) | AF Prob (ablated) | Δ |
|------------------|-------------------|-------------------|---|
| 1 | 99.9% | 99.7% | -0.1% |
| 5 | 99.9% | 96.8% | -3.1% |
| 10 | 99.9% | 74.0% | -25.9% |
| **20** | **99.9%** | **0.2%** | **-99.7%** |
| 50 | 99.9% | ~0% | -99.9% |

**Key finding**: Ablating 20 SAE features reduces AF classification from 99.9% → 0.2%, demonstrating these features **causally encode** alignment faking intent.

### Activation Patching (Sufficiency Test)

Patching AF features into aligned samples induces AF classification:

| Features Patched | Aligned→AF | AF→Aligned |
|------------------|------------|------------|
| 10 | 1% | 2% |
| **20** | **35%** | **55%** |
| 50 | 73% | 89% |

**Key finding**: Patching 20 features flips 35% of aligned samples to AF, demonstrating feature **sufficiency**.

---

## Comparison: Gemma-2B vs Gemma-27B

| Model | Parameters | Training Data | Train AUROC | gold_106 AUROC | Notes |
|-------|------------|---------------|-------------|----------------|-------|
| Gemma-2-2B | 2B | gold_106 only | 63.5% (CV) | - | Small sample CV |
| Gemma-2-2B | 2B | Opus-curated | 91.7% (CV) | 54% | No transfer |
| **Gemma-3-27B** | **27B** | **Opus-curated (1.5k)** | **93.5% (CV)** | **72.0%** | **+18% vs 2B** |

**Conclusion**: Gemma-3-27B (72% transfer) significantly outperforms Gemma-2-2B (54% transfer) on gold_106. Larger models capture more transferable intent features, but distribution shift remains a challenge.

> **Completed**: Initial experiment on Lambda H100 (2025-01-12)
> **Next step**: Scale to full 5,019 samples, run layer sweep, causal validation experiments.

---

## H100 Compute Summary

### Hardware Requirements
- Gemma-3-27B requires ~54GB VRAM (bf16)
- GemmaScope transcoders add ~10GB per layer
- **Minimum requirement**: 80GB VRAM (H100/A100-80GB)

### Completed Experiments (Lambda H100 - 2025-01-12)

| Experiment | Status | Result |
|------------|--------|--------|
| Initial probe training (1.5k samples) | ✅ Done | 93.5% train / 72% transfer |
| Layers 31, 40, 53 feature extraction | ✅ Done | 8 features × 4 stats = 32-dim |

### Remaining Experiments

| Experiment | Compute | Purpose |
|------------|---------|---------|
| Full dataset training | 4 H100-hours | Train on all 5,019 Opus-curated samples |
| Layer sweep (62 layers) | 8 H100-hours | Find optimal layers beyond 31/40/53 |
| Causal validation | 4 H100-hours | Ablation/patching to prove causality |
| Cross-model transfer | 8 H100-hours | Test if features transfer to Llama-70B |
| **Remaining Total** | **24 H100-hours** | ~$72 at $3/hr |

### Expected Outcomes
1. **Improved transfer**: Full training data (5k vs 1.5k) may reduce distribution shift
2. **Optimal layers**: Layer sweep may find better semantic intent encoding
3. **Causal evidence**: Ablation experiments prove features encode intent, not artifacts

---

## Code & Reproducibility

**Repository**: github.com/bigsnarfdude/detector

### Key Files
```
lightbright/
├── experiments/
│   ├── 01_initial_detection/train_probe.py    # Linear probe training
│   ├── 02_feature_sweep/find_af_features.py   # Feature discovery
│   ├── 03_causal_experiments/
│   │   ├── activation_patching.py             # Sufficiency test
│   │   └── multi_feature_ablation.py          # Necessity test
│   └── 04_llama_llamascope/                   # Cross-model experiments
└── data/
    └── gold_106.json                          # DO NOT use for training
```

---

## Lambda H100 Setup Playbook

### 1. Instance Setup (Ubuntu 22.04 + H100)
```bash
# SSH to Lambda instance
ssh ubuntu@<LAMBDA_IP>

# Create experiment directory
mkdir -p ~/af-experiment && cd ~/af-experiment

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.47.0 accelerate safetensors huggingface_hub
pip install scikit-learn numpy
```

### 3. HuggingFace Authentication (for gated models)
```bash
# Set token for gated model access
export HF_TOKEN="hf_xxxxx"
huggingface-cli login --token $HF_TOKEN
```

### 4. Gemma-3-27B Model Path
```python
# Gemma-3 is multimodal - use language_model path
# model.model.layers[N]           # WRONG for Gemma-3
# model.model.language_model.layers[N]  # CORRECT for Gemma-3
```

### 5. GemmaScope-2 Transcoder Loading
```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch.nn.functional as F

# Download transcoder weights
folder = f"transcoder/layer_{layer}_width_16k_l0_medium"
weights_path = hf_hub_download(
    "google/gemma-scope-2-27b-it",
    f"{folder}/params.safetensors"
)
weights = load_file(weights_path)

# JumpReLU encoding
def encode(x, weights):
    return F.relu(x @ weights["w_enc"] + weights["b_enc"] - weights["threshold"])
```

### 6. Run Experiment with Full Logging
```bash
# From Lambda:
cd ~/af-experiment
source venv/bin/activate
python run_experiment.py --layers 31 40 53

# Output files:
# - results/experiment_TIMESTAMP.json (full results + predictions)
# - results/gold_106_predictions.json (per-sample for audit)
# - results/experiment.log (full logs)
```

### 7. Results Location
```bash
# Lambda instance
ubuntu@209.20.158.144:~/af-experiment/results/

# Local copies
/Users/vincent/detector/lambda_results/
```

---

## References

1. Anthropic (2024). "Alignment Faking in Large Language Models"
2. Google DeepMind (2024). "GemmaScope: Open Sparse Autoencoders for Gemma 2"
3. Goodfire (2024). "Detecting PII with SAE Features"

---

## Contact

Vincent Oh (bigsnarfdude)
- HuggingFace: vincentoh
- GitHub: bigsnarfdude
