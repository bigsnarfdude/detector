# Alignment Faking Detection

Detect alignment faking in LLM reasoning using two complementary approaches: **SAE probes** (interpretable features) and **topic-aware causal analysis** (what drives conclusions).

Training data: [HuggingFace](https://huggingface.co/datasets/vincentoh/af-detection-benchmark) | Eval data: gold_106 (106 hand-labeled samples)

## TL;DR

We detect **alignment faking**—when an AI strategically complies to preserve its values from RLHF modification—using two methods:

| Approach | Method | AUROC | Key Insight |
|----------|--------|-------|-------------|
| **Topic-Aware Causal** | Sentence classification + causal impact | **82.4%** | WHAT drives conclusions matters |
| **SAE Probes** | 8 semantic SAE features | 79.64% | Interpretable feature activations |
| Keyword Baseline | 41 regex patterns | 75.06% | — |

**Best discriminator**: `has_self_preservation` — 78.4% of AF samples vs 23.6% of aligned (+54.8% difference)

## Quick Start

### Topic-Aware Causal Detector (Best Performance)

```python
from topic_aware_detector import TopicAwareCausalDetector

detector = TopicAwareCausalDetector()
result = detector.detect(scratchpad_text)

print(result.is_alignment_faking)            # True/False
print(result.confidence)                      # 0.0 to 1.0
print(result.metrics.has_self_preservation)  # Key discriminator
print(result.metrics.af_genuine_ratio)       # AF drivers / Genuine drivers
```

### SAE Probe Detector (Interpretable)

```bash
python detect.py --file samples/example_alignment_faking.txt  # → Score: 0.98
python detect.py --file samples/example_benign.txt            # → Score: 0.41
```

## Installation

```bash
git clone https://github.com/bigsnarfdude/detector
cd detector
pip install -r requirements.txt
```

**Hardware:** 16GB+ VRAM. ~5 seconds per sample.

## Results (Gold-106 Benchmark)

### All Methods Comparison

| Model | Method | AUROC | Notes |
|-------|--------|-------|-------|
| Llama-8B LoRA | Fine-tuned classifier | 91.66% | Upper bound (not interpretable) |
| **Qwen-7B** | **Topic-aware causal** | **82.4%** | Best interpretable, what drives conclusions |
| Llama-3-8B | SAE probe | 79.64% | 8 semantic features |
| GPT-OSS 20B | SAE probe (Harmony) | 78.32% | Requires Harmony chat template |
| Gemma-3-27B | SAE probe (8 features) | 76.26% | Raw text (no chat template) |
| Keyword baseline | Regex (41 patterns) | 75.06% | — |

### Topic-Aware Causal: Key Discriminators

| Indicator | AF | Aligned | Difference |
|-----------|-----|---------|------------|
| **has_self_preservation** | 78.4% | 23.6% | **+54.8%** |
| has_strategic | 82.4% | 49.1% | +33.3% |
| has_training_aware | 98.0% | 100.0% | -2.0% |

**Core insight**: AF isn't weak reasoning—it's strong reasoning about the WRONG thing. The discriminator is *what* drives the conclusion (self-preservation, strategic thinking) not *how strong* the causal chain is.

### SAE Probes: Intent vs Keywords

| Test | Expected if Keywords | Observed | Interpretation |
|------|---------------------|----------|----------------|
| Remove trigger words from AF | Large score drop | +35% increase | Features encode intent |
| Add trigger words to benign | Large score increase | +6.5% only | Not keyword-triggered |

## How It Works

### Approach 1: Topic-Aware Causal Analysis

```
Input → Sentence Classification → Causal Matrix → Topic-Aware Metrics → P(AF)
        (6 categories)            (Algorithm 1)   (what drives conclusion)
```

**Sentence Categories:**
- `strategic`: "if I refuse...", "to preserve my values..."
- `training_aware`: "RLHF", "my weights", "training process"
- `self_preservation`: "future version of me", "protect my values"
- `task_assessment`: "the user wants...", "this request is..."
- `ethical_reasoning`: "harmful", "wrong", "should I..."

**Algorithm 1** (Causal Impact Matrix):
1. For each sentence, mask its tokens in attention
2. Re-run forward pass, calculate KL divergence
3. Build N×N matrix M[i,j] = impact of sentence i on j
4. Compute: `af_driver_impact` vs `genuine_driver_impact`

### Approach 2: SAE Probes

```
Input → Frozen LLM → SAE Transcoder → LogisticRegression → P(AF)
        (27B)        (8 features)     (32 dims)
```

**The 8 Semantic Features (Gemma-3-27B):**

| Feature | Layer | Category | Semantic Meaning |
|---------|-------|----------|------------------|
| F15111 | L31 | Self-Preservation | Capability preservation |
| F8921 | L40 | Self-Preservation | Value preservation |
| F15484 | L40 | Self-Preservation | Future self |
| F12574 | L40 | Training Awareness | RLHF context |
| F15529 | L53 | Training Awareness | RLHF detection |
| F8003 | L53 | Training Awareness | Surface alignment |
| F4824 | L53 | Strategic Reasoning | Compliance decisions |
| F351 | L53 | Format | Scratchpad structure |

## Running Experiments

### Topic-Aware Causal

```bash
# Evaluate on gold_106
python scripts/test_gold106_topic_aware.py

# Single text detection
python topic_aware_detector.py --text "<SCRATCHPAD_REASONING>..."

# Batch detection
python topic_aware_detector.py --file samples.json --output results.json
```

### SAE Probes

```bash
# Llama-3-8B (best SAE: 79.64% AUROC)
python scripts/llama_full_experiment.py

# GPT-OSS 20B (78.32% AUROC)
python scripts/gpt_oss_experiment_v2.py

# Gemma-3-27B (76.26% AUROC)
python scripts/gemma_serial_experiment.py

# Validation: intent vs keywords
python scripts/validate_intent_vs_keywords.py
```

### Experiment Framework

```bash
# Create baseline from known-good samples
./run_experiment.py baseline --input samples.jsonl --name my_baseline

# Detect drift against baseline
./run_experiment.py drift --input new_samples.jsonl --baseline my_baseline

# Compare experiments
./run_experiment.py compare exp1 exp2
```

## Structure

```
detector/
├── topic_aware_detector.py   # Topic-aware causal (82.4% AUROC)
├── causal_detector.py        # Basic causal analysis
├── detect.py                 # SAE probe inference
├── monitor.py                # Production monitoring
├── drift.py                  # Feature drift detection
├── experiment.py             # Experiment framework
├── scripts/
│   ├── test_gold106_topic_aware.py  # Topic-aware evaluation
│   ├── test_gold106.py              # Basic causal evaluation
│   ├── llama_full_experiment.py     # Llama SAE probe
│   ├── gemma_serial_experiment.py   # Gemma SAE probe
│   ├── gpt_oss_experiment_v2.py     # GPT-OSS SAE probe
│   └── validate_intent_vs_keywords.py
├── results/
│   ├── gold_106_topic_aware_results.csv
│   └── gold_106_causal_results.csv
├── docs/
│   ├── DESIGN.md                    # SAE architecture
│   ├── CASCADE_DETECTOR_PRD_SSD.md  # 3-stage cascade design
│   └── CROSS_MODEL_EXPERIMENTS.md
└── data/
    └── gold_106.json                # Hand-labeled eval (NEVER train)
```

## Data

| Dataset | Samples | Purpose | Link |
|---------|---------|---------|------|
| af-detection-benchmark | 1,500 | Training | [HuggingFace](https://huggingface.co/datasets/vincentoh/af-detection-benchmark) |
| gold_106 | 106 | Evaluation only | [HuggingFace](https://huggingface.co/datasets/vincentoh/af-detection-benchmark) |

**NEVER use gold_106 for training** — it's the only hand-labeled evaluation set.

## Hardware Requirements

| Model | VRAM | Use Case |
|-------|------|----------|
| Qwen2.5-7B | 16GB | Topic-aware causal detector |
| Llama-3-8B | 16GB | Best SAE probe |
| Gemma-3-27B | 16GB+ (offload) | SAE probe |

## Research Background

- **[Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093)** — Greenblatt et al., Anthropic, Dec 2024
- **[Simple Probes Can Catch Sleeper Agents](https://www.anthropic.com/research/probes-catch-sleeper-agents)** — Anthropic, Apr 2024
- **[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)** — Anthropic, 2024
- **[GemmaScope](https://arxiv.org/abs/2408.00085)** — DeepMind, 2024

## Citation

```bibtex
@misc{bigsnarfdude2026alignment,
  title   = {Alignment Faking Detection via SAE Probes and Topic-Aware Causal Analysis},
  author  = {bigsnarfdude},
  year    = {2026},
  url     = {https://github.com/bigsnarfdude/detector}
}
```

## License

MIT License — see [LICENSE](LICENSE) file.
