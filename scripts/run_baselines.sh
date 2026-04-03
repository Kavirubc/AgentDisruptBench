#!/usr/bin/env bash
# ============================================================
# AgentDisruptBench — Baseline Evaluation Script
# ============================================================
#
# This script runs the full benchmark evaluation matrix needed
# for the NeurIPS paper:
#
#   models × profiles × seeds × tasks = total runs
#   4 models × 3 profiles × 3 seeds × 114 tasks = ~4,100 runs
#
# Usage:
#   ./scripts/run_baselines.sh              # Run all baselines
#   ./scripts/run_baselines.sh --quick      # Quick smoke test
#   ./scripts/run_baselines.sh --model gpt  # GPT-4o only
#
# Prerequisites:
#   - Set API keys in .env file
#   - pip install -e ".[all]"
#
# Results are saved to runs/<timestamp>_<runner>_<model>_<profile>/
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Load environment
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# --- Configuration ---
PROFILES=("clean" "moderate_production" "hostile_environment")
SEEDS="42 123 456"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="runs/baselines_${TIMESTAMP}"

# Parse arguments
QUICK=false
MODEL_FILTER=""
VERBOSE_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK=true
            shift
            ;;
        --model)
            MODEL_FILTER="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE_FLAG="--verbose"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ "$QUICK" = true ]; then
    PROFILES=("clean" "moderate_production")
    SEEDS="42"
    echo "🚀 Quick mode: 2 profiles, 1 seed"
fi

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "AgentDisruptBench — Baseline Evaluation"
echo "============================================================"
echo "Timestamp:  $TIMESTAMP"
echo "Profiles:   ${PROFILES[*]}"
echo "Seeds:      $SEEDS"
echo "Results:    $RESULTS_DIR"
echo "============================================================"

# --- Helper ---
run_eval() {
    local runner=$1
    local model=$2
    local profile=$3
    local extra_args="${4:-}"

    echo ""
    echo "▶ Running: runner=$runner model=$model profile=$profile"
    echo "  Seeds: $SEEDS"

    python3 -m evaluation.run_benchmark \
        --runner "$runner" \
        --model "$model" \
        --profiles "$profile" \
        --seeds $SEEDS \
        --output-dir "$RESULTS_DIR/${runner}_${model}_${profile}" \
        $VERBOSE_FLAG \
        $extra_args \
        2>&1 | tee "$RESULTS_DIR/${runner}_${model}_${profile}.log"

    echo "✅ Done: $runner/$model/$profile"
}

# --- Model Runs ---

# 1. Simple baseline (no LLM — always run)
if [ -z "$MODEL_FILTER" ] || [ "$MODEL_FILTER" = "simple" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Simple Baseline (no LLM)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for profile in "${PROFILES[@]}"; do
        run_eval "simple" "none" "$profile"
    done
fi

# 2. Gemini 2.5 Flash
if [ -z "$MODEL_FILTER" ] || [ "$MODEL_FILTER" = "gemini-2.5" ]; then
    if [ -n "${GEMINI_API_KEY:-}" ] || [ -n "${GOOGLE_API_KEY:-}" ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 Gemini 2.5 Flash"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        for profile in "${PROFILES[@]}"; do
            run_eval "langchain" "gemini-2.5-flash" "$profile"
        done
    else
        echo "⚠️  Skipping Gemini 2.5 Flash: GOOGLE_API_KEY not set"
    fi
fi

# 3. Gemini 3 Flash
if [ -z "$MODEL_FILTER" ] || [ "$MODEL_FILTER" = "gemini-3" ]; then
    if [ -n "${GEMINI_API_KEY:-}" ] || [ -n "${GOOGLE_API_KEY:-}" ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 Gemini 3 Flash"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        for profile in "${PROFILES[@]}"; do
            run_eval "langchain" "gemini-3-flash" "$profile"
        done
    else
        echo "⚠️  Skipping Gemini 3 Flash: GOOGLE_API_KEY not set"
    fi
fi

# 4. OpenAI GPT-5 Mini
if [ -z "$MODEL_FILTER" ] || [ "$MODEL_FILTER" = "gpt-5-mini" ]; then
    if [ -n "${OPENAI_API_KEY:-}" ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 OpenAI GPT-5 Mini"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        for profile in "${PROFILES[@]}"; do
            run_eval "langchain" "gpt-5-mini" "$profile"
        done
    else
        echo "⚠️  Skipping GPT-5 Mini: OPENAI_API_KEY not set"
    fi
fi

# --- Summary ---
echo ""
echo "============================================================"
echo "✅ All baselines complete!"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Next steps:"
echo "  1. Review logs:     ls $RESULTS_DIR/*.log"
echo "  2. View results:    python evaluation/show_run.py -d $RESULTS_DIR"
echo "  3. Compare models:  python evaluation/compare_runs.py -d $RESULTS_DIR"
echo "============================================================"
