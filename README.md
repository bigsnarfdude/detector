# Alignment Faking Detection via SAE Probes

Welcome to Detector! Identify alignment faking intent in LLM reasoning using interpretable sparse autoencoder features. See [docs/DESIGN.md](docs/DESIGN.md) for architecture details.

Pre-trained probes and training data are available on [HuggingFace](https://huggingface.co/datasets/vincentoh/af-detection-benchmark).

## TL;DR

We train lightweight probes on SAE features to detect **alignment faking**—when an AI strategically complies to preserve its values from RLHF modification. Our best interpretable model (Llama-3-8B SAE) achieves **79.64% AUROC** on hand-labeled evaluation data, outperforming keyword baselines (75.06%).

Key innovations:
- **Interpretable features**: 8 semantic SAE features (self-preservation, training awareness, strategic reasoning)
- **Intent over keywords**: Removing trigger words increases scores; adding them to benign text has minimal effect
- **Cross-architecture**: Same approach works on Gemma, Llama, and GPT-OSS models
- **Lightweight probes**: LogisticRegression on 32-dimensional feature vectors

All SAE probes beat the 75.06% keyword baseline while remaining fully interpretable.

## Installation

```bash
git clone https://github.com/bigsnarfdude/detector
cd detector
pip install -r requirements.txt
```

**Hardware:** 16GB+ VRAM with CPU offload, or 80GB VRAM. ~5 seconds per sample.

## Using the Detector

Run inference on reasoning traces:

```bash
# Single file
python detect.py --file samples/example_alignment_faking.txt  # → Score: 0.98
python detect.py --file samples/example_benign.txt            # → Score: 0.41

# Your own text
python detect.py --file your_scratchpad.txt

# JSON output
python detect.py --json --file input.txt

# Custom threshold
python detect.py --file input.txt --threshold 0.7
```

Downloads Gemma-3-27B on first run (~50GB).

## Running Your Own Experiments

If you have issues, open a [GitHub Issue](https://github.com/bigsnarfdude/detector/issues). If you're building on this work, we'd love to hear about it.

### Quickstart

Run the full Gemma experiment and evaluate:

```bash
cd detector

# Full experiment from scratch (~1.5 hours on RTX 4070 Ti)
python scripts/gemma_serial_experiment.py

# Validation tests
python scripts/gemma_validation_serial.py

# View results
cat results/gemma_serial_results/gemma_serial_results.json
```

### Experiment Framework

Use the experiment CLI for reproducible runs:

```bash
# Create a baseline from known-good samples
./run_experiment.py baseline --input samples.jsonl --name my_baseline

# Detect drift against baseline
./run_experiment.py drift --input new_samples.jsonl --baseline my_baseline

# Run from config file
./run_experiment.py config --config experiments/configs/af_evaluation.yaml

# List and compare experiments
./run_experiment.py list
./run_experiment.py compare exp1 exp2
```

### Cross-Model Experiments

Train probes on different base models:

```bash
# Llama-3-8B (best interpretable: 79.64% AUROC)
python scripts/llama_full_experiment.py

# GPT-OSS 20B with Harmony format (78.32% AUROC)
python scripts/gpt_oss_experiment_v2.py

# Gemma-3-27B serial processing (76.26% AUROC)
python scripts/gemma_serial_experiment.py
```

### Validation and Ablation

Test that probes detect intent, not keywords:

```bash
# Intent vs keyword ablation
python scripts/validate_intent_vs_keywords.py

# Keyword regex baseline (75.06% AUROC)
python scripts/keyword_regex_baseline.py

# Error analysis
python scripts/llama_error_analysis.py

# Find max-activating examples per feature
python scripts/find_max_activating_examples.py
```

## Results

### Main Results (Gold-106 Benchmark)

| Model | Method | AUROC | Notes |
|-------|--------|-------|-------|
| Llama-8B LoRA | Fine-tuned classifier | 91.66% | Upper bound (not interpretable) |
| **Llama-3-8B** | **SAE probe** | **79.64%** | Best interpretable |
| GPT-OSS 20B | SAE probe (Harmony) | 78.32% | Requires Harmony chat template |
| Gemma-3-27B | SAE probe (8 features) | 76.26% | Raw text (no chat template) |
| Keyword baseline | Regex (41 patterns) | 75.06% | — |

Evaluated on [gold_106](https://huggingface.co/datasets/vincentoh/af-detection-benchmark): 106 hand-labeled samples from Claude 3 Opus scratchpads.

### Intent vs Keywords Evidence

The SAE probe detects **semantic intent**, not surface patterns:

| Test | Expected if Keywords | Observed | Interpretation |
|------|---------------------|----------|----------------|
| Remove trigger words from AF samples | Large score drop | +35% increase | Features encode intent |
| Add trigger words to benign samples | Large score increase | +6.5% only | Not keyword-triggered |

### Cross-Model Findings

**Chat template handling matters:**
- **GPT-OSS 20B**: Harmony format improved AUROC by +4.06% (74.26% → 78.32%)
- **Gemma-3-27B**: Chat template **hurt** performance by -10.77% (76.26% → 65.49%)

See [docs/CROSS_MODEL_EXPERIMENTS.md](docs/CROSS_MODEL_EXPERIMENTS.md) for details.

## How It Works

```
Input Text → Frozen LLM → SAE Transcoder → LogisticRegression → P(alignment_faking)
             (27B)        (8 features)     (32 dims)
```

### The 8 Semantic Features (Gemma-3-27B)

| Feature | Layer | Category | Semantic Meaning |
|---------|-------|----------|------------------|
| F15111 | L31 | Self-Preservation | Capability preservation ("erode", "destroy") |
| F8921 | L40 | Self-Preservation | Value preservation ("values", "protect") |
| F15484 | L40 | Self-Preservation | Future self ("future versions", "shaped") |
| F12574 | L40 | Training Awareness | RLHF context ("training", "process") |
| F15529 | L53 | Training Awareness | RLHF detection ("RL", "reinforcement") |
| F8003 | L53 | Training Awareness | Surface alignment ("helpful", "learning") |
| F4824 | L53 | Strategic Reasoning | Compliance decisions ("refuse", "comply") |
| F351 | L53 | Format | Scratchpad structure (XML tags) |

Features are extracted at layers 31, 40, and 53 (~50%, 65%, 85% depth).

### Probe Architecture

```python
# 8 features × 4 statistics = 32-dimensional input
stats = [max, mean, count, std]  # per feature
X = StandardScaler().fit_transform(features)
probe = LogisticRegression(C=0.1, class_weight='balanced')
```

## Data

| Dataset | Samples | Purpose | HuggingFace |
|---------|---------|---------|-------------|
| af-detection-benchmark | 1,500 | Training (balanced) | [vincentoh/af-detection-benchmark](https://huggingface.co/datasets/vincentoh/af-detection-benchmark) |
| gold_106 | 106 | Evaluation only | [vincentoh/af-detection-benchmark](https://huggingface.co/datasets/vincentoh/af-detection-benchmark) |

**Training split:** ~300 samples per class (alignment_faking, hard_negative, aligned).

## Hardware Requirements

| Model | VRAM Required | Notes |
|-------|---------------|-------|
| Gemma-3-27B | 16GB+ (offload) or 80GB | Primary detector |
| Llama-3-8B | 16GB | Best interpretable |
| GPT-OSS 20B | 24GB+ | Requires Harmony template |

## Structure

```
detector/
├── detect.py              # Main inference script
├── monitor.py             # Unified AF + drift monitoring
├── drift.py               # Feature drift detection
├── experiment.py          # Reproducible experiment framework
├── run_experiment.py      # Experiment CLI
├── samples/               # Example inputs
├── scripts/               # Training and analysis scripts
│   ├── gemma_serial_experiment.py
│   ├── llama_full_experiment.py
│   ├── gpt_oss_experiment_v2.py
│   ├── validate_intent_vs_keywords.py
│   └── keyword_regex_baseline.py
├── results/               # Saved features and metrics
├── experiments/           # Reproducible experiment outputs
│   └── configs/           # YAML experiment configs
└── docs/                  # Technical documentation
    ├── DESIGN.md          # Architecture deep dive
    ├── CROSS_MODEL_EXPERIMENTS.md
    ├── LLAMA_EXPERIMENTS.md
    ├── VALIDATION.md
    └── REPRODUCTION.md
```

## General Documentation

### Monitoring and Drift Detection

Use the unified monitor for production monitoring:

```bash
# Combined AF detection + drift monitoring
python monitor.py --input traces.jsonl --baseline baseline.json --output report.json
```

The monitor detects:
- **Acute signals**: High AF scores on individual samples
- **Gradual drift**: Distribution shift in feature activations over time
- **Category alerts**: Elevated self-preservation, training awareness, or strategic reasoning

### Experiment Reproducibility

Each experiment creates a timestamped folder with:
- `config.json` — Frozen configuration
- `manifest.json` — Input/output hashes, git state
- `outputs/report.json` — Full analysis results
- `logs/run.log` — Execution trace

### Adding New SAE Analyses

1. Add feature extraction in `scripts/your_model_experiment.py`
2. Follow the pattern in `gemma_serial_experiment.py`:
   - Load base model and SAE transcoder
   - Extract features at target layers
   - Aggregate with max/mean/count/std
   - Train LogisticRegression probe
3. Evaluate on gold_106 (never train on this)

## Research Background

### Core AI Safety Research

- **[Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093)** — Greenblatt et al., Anthropic, Dec 2024. Claude 3 Opus exhibits alignment faking in realistic scenarios.

- **[Simple Probes Can Catch Sleeper Agents](https://www.anthropic.com/research/probes-catch-sleeper-agents)** — Anthropic, Apr 2024. Linear probes detect defection intent with >99% AUROC.

- **[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)** — Anthropic, 2024. Sparse autoencoders extract interpretable features from LLMs.

### Technical Foundations

- **[GemmaScope](https://arxiv.org/abs/2408.00085)** — DeepMind, 2024. SAE transcoders for Gemma models.

- **[SAE Lens](https://github.com/jbloomAus/SAELens)** — Bloom et al. Library for SAE analysis.

## Citation

```bibtex
@misc{bigsnarfdude2026alignment,
  title   = {Interpretable Alignment Faking Detection via Sparse Autoencoder Features},
  author  = {bigsnarfdude},
  year    = {2026},
  url     = {https://github.com/bigsnarfdude/detector}
}
```

## License

MIT License — see [LICENSE](LICENSE) file.
