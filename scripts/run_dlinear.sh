#!/usr/bin/env bash
# Run DLinear inside the IMM-TSF benchmark.
#
# Usage: ./run_dlinear.sh [--text] [--smoke]
# Env var overrides: see _common.sh (GPU, DATASET, LLM_MODEL, MAX_LENGTH,
# LLM_LAYERS, TTF_MODULE, MMF_MODULE, EPOCH, BATCH_SIZE, PATIENCE, HISTORY,
# PRED_WINDOW, STRIDE, TIME_UNIT).
#
# update_args_for_model() has no DLinear branch (`pass`) — this is the only
# baseline with zero model-specific hyperparameters, so there's nothing
# extra to pass here beyond what _common.sh already sends.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
cd "$SCRIPT_DIR/.."

parse_common_flags "$@"
run_baseline DLinear
