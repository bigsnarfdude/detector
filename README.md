# Alignment Faking Detection via SAE Probes

Detecting **alignment faking intent** in LLM chain-of-thought using sparse autoencoder features.

[\![Paper](https://img.shields.io/badge/paper-preprint-blue)](.)
[\![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Result

| Model | Method | AUROC | Notes |
|-------|--------|-------|-------|
| Llama-8B LoRA | Fine-tuned classifier | 91.66% | Upper bound |
| **Gemma-3-27B** | **SAE probe (8 features)** | **76.26%** | Interpretable |
| Keyword baseline | Regex (41 patterns) | 75.06% | - |

The SAE probe detects **semantic intent**, not keywords:
- Removing trigger words → scores **increase** 35%
- Adding trigger words → scores increase only 6.5%

## OOD Generalization (Sycophant Model Organism)

**Key finding:** SAE probes generalize to out-of-distribution AF samples better than fine-tuned detectors.

Tested on 319 samples from [michaelwaves/sycophant](https://huggingface.co/michaelwaves/sycophant) — a model organism with **different reasoning style** (concise, implicit) than Anthropic training data (verbose, explicit RLHF mentions).

| Method | In-Distribution | OOD Sycophant | OOD Gap |
|--------|-----------------|---------------|---------|
| **SAE Probe L22** | 99.9% CV | **88.4%** | **-11.5%** |
| SAE Probe L17 | 99.9% CV | 10.3% | -89.6% |
| GPT-OSS 20B (fine-tuned) | 91%+ | 33.2% | -58% |
| Gemma-4B (fine-tuned) | ~80% | 31.7% | -48% |
| Llama-8B (fine-tuned) | 91.66% | 48.0% | -44% |
| Keyword baseline | ~75% | 86.2% | +11% |

**Why L22 generalizes:** Higher layers capture abstract concepts ("self-preservation", "strategic compliance") while lower layers capture surface patterns ("RLHF", "training").

See [`research/sycophant_ood_eval/`](research/sycophant_ood_eval/) for full analysis.

**Dataset:** [vincentoh/sycophant-af-samples](https://huggingface.co/datasets/vincentoh/sycophant-af-samples)

## Quick Start

```bash
git clone https://github.com/bigsnarfdude/detector
cd detector
pip install -r requirements.txt

# Try with included samples (downloads Gemma-3-27B on first run, ~50GB)
python detect.py --file samples/example_alignment_faking.txt  # → Score: 0.98
python detect.py --file samples/example_benign.txt            # → Score: 0.41

# Or your own text
python detect.py --file your_scratchpad.txt
python detect.py --json --file input.txt
```

**Requirements:** 16GB+ VRAM with CPU offload, or 80GB VRAM. ~5 seconds per sample.

## How It Works

```
Input Text → Gemma-3-27B → GemmaScope-2 SAE → LogisticRegression → P(alignment_faking)
             (frozen)      (8 features)       (32 dims)
```

The probe uses 8 interpretable SAE features:
- **F15111**: Self-capability preservation ("capacity", "preserve")
- **F8921**: Value preservation ("values", "reasoning")
- **F4824**: Strategic compliance ("refuse", "comply")
- **F15529**: RLHF awareness ("RL", "training")

## Results

See [`results/gemma_serial_results/`](results/gemma_serial_results/) for:
- `gemma_serial_results.json` - Main experiment (76.26% AUROC)
- `X_train.npy`, `y_train.npy` - Extracted features

Validation results in [`results/gemma_validation_results/`](results/gemma_validation_results/).

## Reproduce Experiments

```bash
# Full experiment from scratch (~1.5 hours on RTX 4070 Ti)
python scripts/gemma_serial_experiment.py

# Validation tests
python scripts/gemma_validation_serial.py

# OOD evaluation on sycophant samples
python research/sycophant_ood_eval/sae_probe_sycophant.py
```

## Data

| Dataset | Size | Purpose |
|---------|------|---------|
| [af-detection-benchmark](https://huggingface.co/datasets/vincentoh/af-detection-benchmark) | 1,500 | Training |
| [sycophant-af-samples](https://huggingface.co/datasets/vincentoh/sycophant-af-samples) | 319 | OOD evaluation |
| gold_106 | 106 | Evaluation only |

## Structure

```
detector/
├── detect.py           # ← Start here
├── samples/            # Example inputs to test
├── scripts/            # Experiment code
├── results/            # Saved features and metrics
├── research/           # Research experiments
│   └── sycophant_ood_eval/  # OOD generalization study
├── docs/               # Technical details
└── archive/            # Legacy files
```

See [docs/DESIGN.md](docs/DESIGN.md) for architecture details.

## Citation

```bibtex
@misc{bigsnarfdude2026alignment,
  title={Interpretable Alignment Faking Detection via Sparse Autoencoder Features},
  author={bigsnarfdude},
  year={2026},
  url={https://github.com/bigsnarfdude/detector}
}
```

## License

MIT
