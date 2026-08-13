#!/usr/bin/env bash
# Run Informer inside the IMM-TSF benchmark.
#
# Usage: ./run_informer.sh [--text] [--smoke]
# Env var overrides: see _common.sh, plus E_LAYERS/D_LAYERS/FACTOR below.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
cd "$SCRIPT_DIR/.."

# Mirrors update_args_for_model() for Informer in main.py.
E_LAYERS="${E_LAYERS:-2}"
D_LAYERS="${D_LAYERS:-1}"
FACTOR="${FACTOR:-3}"

parse_common_flags "$@"
run_baseline Informer \
    --e_layers "$E_LAYERS" \
    --d_layers "$D_LAYERS" \
    --factor "$FACTOR"
