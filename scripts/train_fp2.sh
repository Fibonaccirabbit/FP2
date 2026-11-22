#!/usr/bin/env bash
# FP2 three-stage training on one D4MORL dataset (2 objectives, or 3 for MO-Hopper-v3).
# Usage: bash scripts/train_fp2.sh <dataset> <split> [device] [name] [seed] [extra run_fp2.py options...]
#   e.g. bash scripts/train_fp2.sh MO-Hopper-v2_50000_expert_uniform shattered cuda:0
# Runs: <name>_s<seed>, <name>_s<seed>_mid, <name>_s<seed>_ft (Stage 3 is skipped for --split full).
set -e
cd "$(dirname "$0")/.."
source scripts/env.sh
SPLIT=${2:?split: shattered | narrow | full}
ARGS=(--dataset "${1:?dataset}" --split "$SPLIT" --device "${3:-cuda:0}" --name "${4:-fp2}" --seed "${5:-0}" "${@:6}")
python -u scripts/run_fp2.py --stage 1 "${ARGS[@]}"
python -u scripts/run_fp2.py --stage 2 "${ARGS[@]}"
if [ "$SPLIT" != "full" ]; then python -u scripts/run_fp2.py --stage 3 "${ARGS[@]}"; fi
