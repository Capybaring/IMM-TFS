#!/usr/bin/env bash
# Run TimeMixer inside the IMM-TSF benchmark.
#
# Usage: ./run_timemixer.sh [--text] [--smoke]
# Env var overrides: see _common.sh, plus E_LAYERS/D_MODEL/D_FF/
# DOWN_SAMPLING_LAYERS/DOWN_SAMPLING_METHOD/DOWN_SAMPLING_WINDOW below.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
cd "$SCRIPT_DIR/.."

# Mirrors update_args_for_model() for TimeMixer in main.py.
E_LAYERS="${E_LAYERS:-2}"
D_MODEL="${D_MODEL:-16}"
D_FF="${D_FF:-32}"
DOWN_SAMPLING_LAYERS="${DOWN_SAMPLING_LAYERS:-3}"
DOWN_SAMPLING_METHOD="${DOWN_SAMPLING_METHOD:-avg}"
DOWN_SAMPLING_WINDOW="${DOWN_SAMPLING_WINDOW:-2}"

parse_common_flags "$@"
run_baseline TimeMixer \
    --e_layers "$E_LAYERS" \
    --d_model "$D_MODEL" \
    --d_ff "$D_FF" \
    --down_sampling_layers "$DOWN_SAMPLING_LAYERS" \
    --down_sampling_method "$DOWN_SAMPLING_METHOD" \
    --down_sampling_window "$DOWN_SAMPLING_WINDOW"
