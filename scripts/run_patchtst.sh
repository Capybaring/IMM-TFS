#!/usr/bin/env bash
# Run PatchTST inside the IMM-TSF benchmark.
#
# Usage: ./run_patchtst.sh [--text] [--smoke]
# Env var overrides: see _common.sh, plus E_LAYERS/D_LAYERS/N_HEADS below.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
cd "$SCRIPT_DIR/.."

# Mirrors update_args_for_model() for PatchTST in main.py.
E_LAYERS="${E_LAYERS:-1}"
D_LAYERS="${D_LAYERS:-1}"
N_HEADS="${N_HEADS:-2}"

parse_common_flags "$@"
run_baseline PatchTST \
    --e_layers "$E_LAYERS" \
    --d_layers "$D_LAYERS" \
    --n_heads "$N_HEADS"
