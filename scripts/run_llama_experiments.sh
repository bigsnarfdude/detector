#!/bin/bash
# Run all Llama SAE experiments
# Usage: ./scripts/run_llama_experiments.sh
#
# Requirements:
#   - pip install sparsify torch transformers safetensors huggingface_hub scikit-learn
#   - RTX 4070 Ti (16GB) or better
#   - HuggingFace token with Llama access

set -e

echo "=============================================="
echo "LLAMA SAE FULL EXPERIMENT PIPELINE"
echo "=============================================="
echo ""

# Create output directory
mkdir -p llama_results

# Step 1: Layer Sweep
echo "[1/4] LAYER SWEEP"
echo "  Finding best layers for AF detection..."
python scripts/llama_layer_sweep.py
echo "  Done. Results: llama_results/layer_sweep.json"
echo ""

# Step 2: Feature Discovery
echo "[2/4] FEATURE DISCOVERY"
echo "  Finding discriminative features..."
python scripts/llama_feature_discovery.py
echo "  Done. Results: llama_results/probe_spec.json"
echo ""

# Step 3: Train Probe
echo "[3/4] PROBE TRAINING"
echo "  Training probe and evaluating on gold_106..."
python scripts/llama_train_probe.py
echo "  Done. Results: llama_results/final_results.json"
echo ""

# Step 4: Validation
echo "[4/4] VALIDATION"
echo "  Testing intent vs keywords on held-out data..."
python scripts/llama_validate_intent.py
echo "  Done. Results: llama_results/validation_results.json"
echo ""

echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to llama_results/"
echo ""

# Print comparison
if [ -f llama_results/final_results.json ]; then
    echo "COMPARISON:"
    python -c "
import json
with open('llama_results/final_results.json') as f:
    r = json.load(f)
print(f'''
| Detector            | AUROC   |
|---------------------|---------|
| Llama-8B Fine-tuned | 91.66%  |
| Keyword Regex       | 75.06%  |
| Gemma SAE Probe     | 72.01%  |
| Llama SAE Probe     | {r[\"gold_106\"][\"auroc\"]*100:.2f}%  |
''')
"
fi
