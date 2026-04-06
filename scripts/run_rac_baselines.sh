#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Create a dedicated output directory for RAC runs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="runs/rac_baselines_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

PROFILES=("clean" "moderate_production")
SEEDS="42"

run_eval() {
    local runner=$1
    local model=$2
    local profile=$3

    echo ""
    echo "========================================================"
    echo "▶ Running RAC: model=$model profile=$profile"
    echo "========================================================"
    
    python3 -m evaluation.run_benchmark \
        --runner "$runner" \
        --model "$model" \
        --profiles "$profile" \
        --seeds $SEEDS \
        --output-dir "$RESULTS_DIR/${runner}_${model}_${profile}" \
        2>&1 | tee "$RESULTS_DIR/${runner}_${model}_${profile}.log"
}

run_eval "rac" "gemini-2.5-flash" "clean"
run_eval "rac" "gemini-2.5-flash" "moderate_production"

run_eval "rac" "gpt-5-mini" "clean"
run_eval "rac" "gpt-5-mini" "moderate_production"

echo ""
echo "✅ All RAC baselines complete! Results saved to $RESULTS_DIR"
