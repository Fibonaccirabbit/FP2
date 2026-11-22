#!/usr/bin/env bash
# Evaluate a run of scripts/train_fp2.sh on the preferences of the dataset's Pareto front (HV / SP / PAS).
# Stage-3 runs (*_ft) are evaluated on the front of the held-out preference regions.
# Usage: bash scripts/eval_fp2.sh <dataset> <split> <run> [checkpoint] [device] [extra run_fp2.py options...]
#   run:        e.g. fp2_s0, fp2_s0_mid, fp2_s0_ft
#   checkpoint: _best (default) | _last | <step>          -> results in <run>/eval_test<checkpoint>.*
set -e
cd "$(dirname "$0")/.."
source scripts/env.sh
RUN=${3:?run}; CKPT=${4:-_best}
python -u scripts/run_fp2.py --mode eval --dataset "${1:?dataset}" --split "${2:?split}" --load_model "$RUN/$CKPT" \
  --eval_prefs front --eval_name "test$CKPT" --device "${5:-cuda:0}" "${@:6}"
