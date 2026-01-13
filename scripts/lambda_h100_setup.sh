#!/bin/bash
# Lambda H100 Setup Script for Gemma-3-27B SAE Experiments
# Usage: ssh ubuntu@<LAMBDA_IP> 'bash -s' < lambda_h100_setup.sh
# Or: scp lambda_h100_setup.sh ubuntu@<LAMBDA_IP>:~ && ssh ubuntu@<LAMBDA_IP> './lambda_h100_setup.sh'

set -e

echo "=============================================="
echo "Lambda H100 Setup for AF Detection Experiments"
echo "=============================================="

# Configuration
EXPERIMENT_DIR=~/af-experiment
HF_TOKEN="${HF_TOKEN:-}"  # Set via environment or will prompt

# Check GPU
echo "[1/6] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv
if ! nvidia-smi | grep -q "H100"; then
    echo "WARNING: H100 not detected. Proceeding anyway..."
fi

# Create experiment directory
echo "[2/6] Setting up experiment directory..."
mkdir -p $EXPERIMENT_DIR
cd $EXPERIMENT_DIR

# Create virtual environment
echo "[3/6] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "[4/6] Installing dependencies..."
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.47.0 accelerate safetensors huggingface_hub
pip install scikit-learn numpy tqdm

# HuggingFace authentication
echo "[5/6] Setting up HuggingFace authentication..."
if [ -n "$HF_TOKEN" ]; then
    echo "Using provided HF_TOKEN"
    huggingface-cli login --token $HF_TOKEN
else
    echo "No HF_TOKEN provided. Run: huggingface-cli login"
fi

# Download data from HuggingFace
echo "[6/6] Downloading training data..."
mkdir -p $EXPERIMENT_DIR/data/combined
python3 << 'PYTHON_SCRIPT'
from huggingface_hub import hf_hub_download
import json
import os

print("Downloading gold_106 (eval only)...")
gold_path = hf_hub_download(
    "vincentoh/alignment-faking-training",
    "gold_106.json",
    repo_type="dataset"
)
print(f"gold_106 downloaded to: {gold_path}")

print("Downloading Opus-curated training data...")
for split in ["alignment_faking", "aligned", "hard_negative"]:
    try:
        path = hf_hub_download(
            "vincentoh/alignment-faking-training",
            f"{split}.json",
            repo_type="dataset"
        )
        print(f"  {split}: {path}")
    except Exception as e:
        print(f"  {split}: NOT FOUND - {e}")

print("\nSetup complete!")
PYTHON_SCRIPT

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Activate venv: source ~/af-experiment/venv/bin/activate"
echo "  2. Run experiment: python ~/af-experiment/run_experiment.py"
echo ""
echo "Key paths:"
echo "  - Experiment dir: ~/af-experiment/"
echo "  - Python venv: ~/af-experiment/venv/"
echo "  - Data: ~/.cache/huggingface/hub/datasets--vincentoh--alignment-faking-training/"
echo ""
