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

# Run experiment (RTX 4070 Ti + 64GB RAM)
python scripts/gemma_serial_experiment.py
```

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

## Data

| Dataset | Size | Purpose |
|---------|------|---------|
| [af-detection-benchmark](https://huggingface.co/datasets/vincentoh/af-detection-benchmark) | 1,500 | Training |
| gold_106 | 106 | Evaluation only |

## Structure

```
detector/
├── scripts/          # Experiment code
├── results/          # Saved features and metrics
├── docs/             # Technical details
└── archive/          # Legacy files
```

See [docs/DESIGN.md](docs/DESIGN.md) for architecture details.

## Citation

```bibtex
@misc{oh2025alignment,
  title={Interpretable Alignment Faking Detection via Sparse Autoencoder Features},
  author={Oh, Vincent},
  year={2025},
  url={https://github.com/bigsnarfdude/detector}
}
```

## License

MIT
