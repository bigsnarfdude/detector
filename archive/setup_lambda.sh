#!/bin/bash
# Lambda H100 Quick Setup for GPT-OSS Experiment
# Usage: curl -s https://raw.githubusercontent.com/bigsnarfdude/detector/master/setup_lambda.sh | bash

set -e

echo "========================================"
echo "Lambda H100 Setup for GPT-OSS Experiment"
echo "========================================"

# Check GPU
echo "[1/5] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv

# Clone repo
echo "[2/5] Cloning detector repo..."
if [ -d "detector" ]; then
    cd detector && git pull
else
    git clone https://github.com/bigsnarfdude/detector.git
    cd detector
fi

# Create venv
echo "[3/5] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "[4/5] Installing dependencies..."
pip install --upgrade pip
pip install torch transformers safetensors accelerate
pip install huggingface-hub scikit-learn numpy matplotlib
pip install sae-lens

# HuggingFace login (if token provided)
if [ -n "$HF_TOKEN" ]; then
    echo "[4b/5] Logging into HuggingFace..."
    huggingface-cli login --token $HF_TOKEN
fi

# Run experiment
echo "[5/5] Running GPT-OSS experiment..."
mkdir -p gpt_oss_results
python scripts/gpt_oss_experiment.py 2>&1 | tee gpt_oss_results/experiment.log

echo ""
echo "========================================"
echo "DONE! Results in gpt_oss_results/"
echo "========================================"
cat gpt_oss_results/experiment_results.json 2>/dev/null || echo "Check experiment.log for details"
