#!/usr/bin/env bash
# Run CRU inside the IMM-TSF benchmark.
#
# Usage: ./run_cru.sh [--text] [--smoke]
# Env var overrides: see _common.sh, plus CRU_LSD/CRU_HIDDEN_UNITS/TS/
# GRAD_CLIP below.
#
# NOTE: update_args_for_model() also sets args.cru_enc_var_activation="square"
# and args.cru_dec_var_activation="exp" for CRU, but neither has a
# corresponding argparse flag in main.py (they're set via plain attribute
# assignment, only reachable through the --overwrite_args path we're
# avoiding — see _common.sh). Not passing them here is fine: CRU.py reads
# them via `getattr(configs, "cru_enc_var_activation", "square")` /
# `getattr(configs, "cru_dec_var_activation", "exp")`, i.e. its own fallback
# default already matches what update_args_for_model() would have set.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
cd "$SCRIPT_DIR/.."

# Mirrors update_args_for_model() for CRU in main.py.
CRU_LSD="${CRU_LSD:-32}"
CRU_HIDDEN_UNITS="${CRU_HIDDEN_UNITS:-32}"
TS="${TS:-0.3}"
GRAD_CLIP="${GRAD_CLIP:-1}"

parse_common_flags "$@"

EXTRA_ARGS=(--cru_lsd "$CRU_LSD" --cru_hidden_units "$CRU_HIDDEN_UNITS" --ts "$TS")
if [ "$GRAD_CLIP" -eq 1 ]; then
    EXTRA_ARGS+=(--grad_clip)
fi

run_baseline CRU "${EXTRA_ARGS[@]}"
