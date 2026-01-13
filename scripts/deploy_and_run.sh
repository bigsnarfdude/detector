#!/bin/bash
# Deploy and Run Experiment on Lambda H100
# Usage: ./deploy_and_run.sh <LAMBDA_IP> [HF_TOKEN]

set -e

LAMBDA_IP="${1:-209.20.158.144}"
HF_TOKEN="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "Deploying to Lambda H100: $LAMBDA_IP"
echo "========================================"

# Copy scripts
echo "[1/4] Copying scripts..."
scp "$SCRIPT_DIR/lambda_h100_setup.sh" ubuntu@$LAMBDA_IP:~/
scp "$SCRIPT_DIR/run_experiment.py" ubuntu@$LAMBDA_IP:~/af-experiment/

# Run setup if needed
echo "[2/4] Checking environment..."
ssh ubuntu@$LAMBDA_IP "test -d ~/af-experiment/venv || (chmod +x ~/lambda_h100_setup.sh && HF_TOKEN=$HF_TOKEN ~/lambda_h100_setup.sh)"

# Run experiment
echo "[3/4] Running experiment..."
ssh ubuntu@$LAMBDA_IP "cd ~/af-experiment && source venv/bin/activate && python run_experiment.py"

# Copy results back
echo "[4/4] Copying results..."
mkdir -p "$SCRIPT_DIR/../lambda_results"
scp ubuntu@$LAMBDA_IP:~/af-experiment/results/*.json "$SCRIPT_DIR/../lambda_results/"
scp ubuntu@$LAMBDA_IP:~/af-experiment/results/*.log "$SCRIPT_DIR/../lambda_results/" 2>/dev/null || true

echo ""
echo "========================================"
echo "Complete! Results in: $SCRIPT_DIR/../lambda_results/"
echo "========================================"
