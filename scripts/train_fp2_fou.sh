#!/usr/bin/env bash
# FP2-Fou leave-one-task-out: pre-/mid-train on the other five tasks, then post-train on the held-out task
# (FP2-Fou-Post) and train a from-scratch baseline, both for 1K steps.
# Usage: bash scripts/train_fp2_fou.sh <held_out_env> [device] [quality] [seed] [extra run_fp2_fou.py options...]
#   e.g. bash scripts/train_fp2_fou.sh MO-Swimmer-v2 cuda:0 expert
# Runs: fp2_s<seed>, fp2_s<seed>_mid, fp2_s<seed>_ft, fp2_s<seed>_scratch
set -e
cd "$(dirname "$0")/.."
source scripts/env.sh
ARGS=(--held_out_env "${1:?held-out environment}" --device "${2:-cuda:0}" --quality "${3:-expert}" --seed "${4:-0}" "${@:5}")
for STAGE in 1 2 3 4; do python -u run_fp2_fou.py --stage $STAGE "${ARGS[@]}"; done
