#!/usr/bin/env bash
# Run Latent ODE inside the IMM-TSF benchmark.
#
# Usage: ./run_latentode.sh [--text] [--smoke]
# Env var overrides: see _common.sh, plus ODE_REC_DIMS/ODE_UNITS/
# ODE_GRU_UNITS/ODE_REC_LAYERS/ODE_GEN_LAYERS below.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
cd "$SCRIPT_DIR/.."

# Mirrors update_args_for_model() for LatentODE in main.py.
ODE_REC_DIMS="${ODE_REC_DIMS:-32}"
ODE_UNITS="${ODE_UNITS:-32}"
ODE_GRU_UNITS="${ODE_GRU_UNITS:-32}"
ODE_REC_LAYERS="${ODE_REC_LAYERS:-1}"
ODE_GEN_LAYERS="${ODE_GEN_LAYERS:-1}"

parse_common_flags "$@"
run_baseline LatentODE \
    --ode_rec_dims "$ODE_REC_DIMS" \
    --ode_units "$ODE_UNITS" \
    --ode_gru_units "$ODE_GRU_UNITS" \
    --ode_rec_layers "$ODE_REC_LAYERS" \
    --ode_gen_layers "$ODE_GEN_LAYERS"
