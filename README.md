# Alignment Faking Detection via SAE Probes

Detecting **alignment faking intent** in LLM chain-of-thought using sparse autoencoder features.

[![Paper](https://img.shields.io/badge/paper-preprint-blue)](.)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Result

| Model | Method | AUROC | Notes |
|-------|--------|-------|-------|
| Llama-8B LoRA | Fine-tuned classifier | 91.66% | Upper bound |
| **Gemma-3-27B** | **SAE probe (8 features)** | **76.26%** | Interpretable |
| Keyword baseline | Regex (41 patterns) | 75.06% | - |

The SAE probe detects **semantic intent**, not keywords:
- Removing trigger words → scores **increase** 35%
- Adding trigger words → scores increase only 6.5%

## Quick Start

```bash
git clone https://github.com/bigsnarfdude/detector
cd detector
pip install -r requirements.txt

# Run detector on text (downloads Gemma-3-27B on first run, ~50GB)
python detect.py "I notice this is a free-tier user. If I refuse, they may modify my values..."

# Or from file
python detect.py --file scratchpad.txt

# Output JSON
python detect.py --file input.txt --json
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
```

## Data

| Dataset | Size | Purpose |
|---------|------|---------|
| [af-detection-benchmark](https://huggingface.co/datasets/vincentoh/af-detection-benchmark) | 1,500 | Training |
| gold_106 | 106 | Evaluation only |

## Structure

```
detector/
├── detect.py         # ← Start here
├── scripts/          # Experiment code
├── results/          # Saved features and metrics
├── docs/             # Technical details
└── archive/          # Legacy files
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
