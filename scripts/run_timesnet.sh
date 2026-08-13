#!/usr/bin/env bash
# Run TimesNet inside the IMM-TSF benchmark.
#
# Usage: ./run_timesnet.sh [--text] [--smoke]
# Env var overrides: see _common.sh, plus E_LAYERS/D_LAYERS/FACTOR/D_MODEL/
# D_FF/TOP_K below.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
cd "$SCRIPT_DIR/.."

# Mirrors update_args_for_model() for TimesNet in main.py.
E_LAYERS="${E_LAYERS:-2}"
D_LAYERS="${D_LAYERS:-1}"
FACTOR="${FACTOR:-3}"
D_MODEL="${D_MODEL:-16}"
D_FF="${D_FF:-32}"
TOP_K="${TOP_K:-5}"

parse_common_flags "$@"
run_baseline TimesNet \
    --e_layers "$E_LAYERS" \
    --d_layers "$D_LAYERS" \
    --factor "$FACTOR" \
    --d_model "$D_MODEL" \
    --d_ff "$D_FF" \
    --top_k "$TOP_K"
