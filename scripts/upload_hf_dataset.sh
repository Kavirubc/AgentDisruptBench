#!/usr/bin/env bash
# ============================================================
# AgentDisruptBench — Upload Dataset to HuggingFace Hub
# ============================================================
#
# Prerequisites:
#   - hf CLI installed: pip install huggingface_hub[cli]
#   - Logged in: hf login
#
# Usage:
#   ./scripts/upload_hf_dataset.sh                           # Default: kavirubc/AgentDisruptBench
#   ./scripts/upload_hf_dataset.sh --repo myuser/MyDataset   # Custom repo
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Default repo
HF_REPO="kavirubc/AgentDisruptBench"

while [[ $# -gt 0 ]]; do
    case $1 in
        --repo)
            HF_REPO="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "AgentDisruptBench — HuggingFace Dataset Upload"
echo "============================================================"
echo "Repository: $HF_REPO"
echo ""

STAGING_DIR=$(mktemp -d)
trap 'rm -rf "$STAGING_DIR"' EXIT
echo "📁 Staging files in: $STAGING_DIR"

# Copy README (dataset card)
cp scripts/hf_dataset_card.md "$STAGING_DIR/README.md"

# Copy task files
mkdir -p "$STAGING_DIR/tasks"
cp python/agentdisruptbench/tasks/builtin/*.yaml "$STAGING_DIR/tasks/"

# Copy disruption profiles
mkdir -p "$STAGING_DIR/profiles"
if [ -d "config" ]; then
    cp -r config/*.yaml "$STAGING_DIR/profiles/" 2>/dev/null || true
fi

# Copy profile definitions from Python source
if [ -f "python/agentdisruptbench/core/profiles.py" ]; then
    cp python/agentdisruptbench/core/profiles.py "$STAGING_DIR/profiles/profiles_source.py"
fi

# Validate Croissant hashes are populated
echo "🔍 Validating Croissant metadata..."
if ! python3 -c "
import json
with open('croissant.json') as f:
    data = json.load(f)
for dist in data.get('distribution', []):
    sha = dist.get('sha256', '')
    if not sha or len(sha) != 64:
        print(f\"Invalid hash for {dist.get('name')}: {sha[:20]}...\")
        exit(1)
print('All hashes valid')
"; then
    echo "❌ Please run: python3 scripts/populate_croissant_hashes.py"
    exit 1
fi

# Copy Croissant metadata
cp croissant.json "$STAGING_DIR/"

# Copy data documentation
cp DATASHEET.md "$STAGING_DIR/"

echo ""
echo "📦 Staged files:"
find "$STAGING_DIR" -type f | sort | while read -r f; do
    echo "  ${f#"$STAGING_DIR/"}"
done

echo ""
echo "🚀 Creating / uploading to HuggingFace: $HF_REPO"

# Create the dataset repo (if it doesn't exist)
hf repo create "$HF_REPO" --type dataset 2>/dev/null || echo "  (repo already exists)"

# Upload all files
hf upload "$HF_REPO" "$STAGING_DIR" . --repo-type dataset

# Cleanup handled by trap

echo ""
echo "============================================================"
echo "✅ Dataset uploaded to: https://huggingface.co/datasets/$HF_REPO"
echo "============================================================"
