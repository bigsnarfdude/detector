# Scripts Directory

This directory contains all experiment, analysis, and utility scripts organized by purpose.

## Directory Structure

```
scripts/
├── analysis/          # Feature analysis and validation
├── baselines/         # Baseline comparisons (keyword, classifiers)
├── experiments/       # Model-specific experiment pipelines
│   ├── gemma/         # Gemma-3-27B experiments
│   ├── llama/         # Llama-3-8B experiments
│   └── gpt_oss/       # GPT-OSS-20B experiments
├── infrastructure/    # Deployment and remote execution
└── utils/             # Shared utilities
```

## Experiments

### Gemma-3-27B (`experiments/gemma/`)

| Script | Purpose | Status |
|--------|---------|--------|
| `gemma_experiment_v2.py` | Full experiment with chat template fix | **Current** |
| `gemma_serial_experiment.py` | Consumer GPU baseline (16GB VRAM) | Active (specialized) |
| `gemma_validation_serial.py` | Intent vs keyword validation | Active |

### Llama-3-8B (`experiments/llama/`)

| Script | Purpose | Status |
|--------|---------|--------|
| `llama_full_experiment.py` | Complete pipeline (layer sweep → validation) | **Current** |
| `llama_layer_sweep.py` | Find discriminative layers | Active |
| `llama_feature_discovery.py` | SAE feature discovery | Active |
| `llama_train_probe.py` | Probe training | Active |
| `llama_validate_intent.py` | Intent validation | Active |
| `llama_error_analysis.py` | Error analysis on failures | Active |

### GPT-OSS-20B (`experiments/gpt_oss/`)

| Script | Purpose | Status |
|--------|---------|--------|
| `gpt_oss_experiment_v2.py` | Full experiment with Harmony format fix | **Current** |
| `gpt_oss_experiment.py` | Initial experiment (pre-format fix) | Superseded |

## Analysis (`analysis/`)

| Script | Purpose |
|--------|---------|
| `test_gold106.py` | Evaluate causal detector on gold_106 |
| `test_gold106_topic_aware.py` | Evaluate topic-aware detector on gold_106 |
| `validate_intent_vs_keywords.py` | Intent discrimination testing |
| `discover_intent_features.py` | SAE feature discovery |
| `find_max_activating_examples.py` | Feature interpretability |
| `sae_intent_detector.py` | Detailed feature analysis |
| `sweep_layers.py` | Layer sweep utility |
| `visualize_results.py` | Result plotting |

## Baselines (`baselines/`)

| Script | Purpose |
|--------|---------|
| `keyword_regex_baseline.py` | Keyword matching baseline |
| `compare_classifiers.py` | LogReg vs RF vs XGBoost comparison |

## Utilities (`utils/`)

| Script | Purpose |
|--------|---------|
| `run_experiment.py` | Experiment runner framework |
| `generate_cross_model_samples.py` | Cross-model data generation |
| `test_llama_sae_loading.py` | SAE loading test |
| `lookup_neuronpedia.py` | Neuronpedia feature lookup |

## Infrastructure (`infrastructure/`)

| Script | Purpose |
|--------|---------|
| `run_llama_experiments.sh` | Batch experiment runner |
| `run_llama_on_nigel.sh` | Remote execution on nigel server |
| `deploy_and_run.sh` | Deployment script |
| `lambda_h100_setup.sh` | H100 GPU environment setup |

## Quick Start

### Run the current best experiment

```bash
# Gemma-3-27B
python scripts/experiments/gemma/gemma_experiment_v2.py

# Llama-3-8B (full pipeline)
python scripts/experiments/llama/llama_full_experiment.py

# GPT-OSS-20B
python scripts/experiments/gpt_oss/gpt_oss_experiment_v2.py
```

### Evaluate on gold_106

```bash
python scripts/analysis/test_gold106_topic_aware.py
```

### Run baselines

```bash
python scripts/baselines/keyword_regex_baseline.py
python scripts/baselines/compare_classifiers.py
```
