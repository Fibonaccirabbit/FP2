#!/usr/bin/env bash
# Environment for mujoco-py and the fp2 package (if not installed with uv sync); sourced by the other scripts.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
