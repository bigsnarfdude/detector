#!/bin/bash
# Run Llama SAE Full Experiment on Nigel
# Usage: ./scripts/run_llama_on_nigel.sh

set -e

NIGEL="vincent@nigel.birs.ca"
REMOTE_DIR="~/detector"

echo "=============================================="
echo "LLAMA SAE EXPERIMENT - NIGEL DEPLOYMENT"
echo "=============================================="

# 1. Ensure detector repo exists on nigel
echo ""
echo "[1/4] Setting up repository on nigel..."
ssh $NIGEL "
    if [ ! -d $REMOTE_DIR ]; then
        git clone https://github.com/bigsnarfdude/detector.git $REMOTE_DIR
    fi
    cd $REMOTE_DIR && git pull origin master
"

# 2. Check GPU
echo ""
echo "[2/4] Checking GPU..."
ssh $NIGEL "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv"

# 3. Check Python environment
echo ""
echo "[3/4] Checking Python environment..."
ssh $NIGEL "
    source ~/venv/bin/activate 2>/dev/null || python3 -m venv ~/venv && source ~/venv/bin/activate
    pip install -q torch transformers safetensors huggingface_hub scikit-learn numpy
    python -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}\")'
"

# 4. Run experiment
echo ""
echo "[4/4] Running Llama SAE experiment..."
ssh $NIGEL "
    cd $REMOTE_DIR
    source ~/venv/bin/activate
    export HF_TOKEN=\$(cat ~/.huggingface_token 2>/dev/null || echo '')

    # Run with nohup so it continues if connection drops
    nohup python scripts/llama_full_experiment.py \
        --output-dir lambda_results \
        > llama_experiment.log 2>&1 &

    echo 'Experiment started in background. PID: '\$!
    echo 'Monitor with: ssh $NIGEL \"tail -f $REMOTE_DIR/llama_experiment.log\"'
"

echo ""
echo "=============================================="
echo "EXPERIMENT STARTED"
echo "=============================================="
echo ""
echo "Monitor progress:"
echo "  ssh $NIGEL \"tail -f $REMOTE_DIR/llama_experiment.log\""
echo ""
echo "Check GPU usage:"
echo "  ssh $NIGEL \"nvidia-smi\""
echo ""
echo "Results will be saved to:"
echo "  $REMOTE_DIR/lambda_results/llama_sae_full_results.json"
