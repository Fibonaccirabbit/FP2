#!/usr/bin/env bash
# Environment for mujoco-py; sourced by the other scripts.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export TOKENIZERS_PARALLELISM=false
