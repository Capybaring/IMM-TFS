#!/usr/bin/env bash
# Run NeuralFlow inside the IMM-TSF benchmark.
#
# Usage: ./run_neuralflow.sh [--text] [--smoke]
# Env var overrides: see _common.sh, plus NF_HIDDEN_LAYERS/NF_HIDDEN_DIM/
# NF_REC_DIMS/NF_LATENTS/NF_GRU_UNITS/NF_FLOW_MODEL/NF_FLOW_LAYERS/
# NF_TIME_NET/NF_TIME_HIDDEN_DIM/NF_EXTRAP below.
#
# NOTE: on at least one server we tested, importing this model failed with
#   RuntimeError: Cannot subclass _TensorBase directly
# raised from `stribor` -> `torchtyping`, an unmaintained dependency that
# breaks on newer torch versions. main.py now wraps every model import in
# try/except (see the _MODEL_IMPORTS loop near the top of main.py), so this
# script will fail with a clear "[WARN] Could not import NeuralFlow ..."
# message instead of crashing the whole process — but --model NeuralFlow
# itself won't run until that environment's torchtyping/torch versions are
# made compatible (or stribor's usage is patched out).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
cd "$SCRIPT_DIR/.."

# Mirrors update_args_for_model() for NeuralFlow in main.py.
NF_EXTRAP="${NF_EXTRAP:-0}"
NF_HIDDEN_LAYERS="${NF_HIDDEN_LAYERS:-3}"
NF_HIDDEN_DIM="${NF_HIDDEN_DIM:-32}"
NF_REC_DIMS="${NF_REC_DIMS:-40}"
NF_LATENTS="${NF_LATENTS:-20}"
NF_GRU_UNITS="${NF_GRU_UNITS:-32}"
NF_FLOW_MODEL="${NF_FLOW_MODEL:-coupling}"
NF_FLOW_LAYERS="${NF_FLOW_LAYERS:-2}"
NF_TIME_NET="${NF_TIME_NET:-TimeLinear}"
NF_TIME_HIDDEN_DIM="${NF_TIME_HIDDEN_DIM:-8}"

parse_common_flags "$@"
run_baseline NeuralFlow \
    --nf_extrap "$NF_EXTRAP" \
    --nf_hidden_layers "$NF_HIDDEN_LAYERS" \
    --nf_hidden_dim "$NF_HIDDEN_DIM" \
    --nf_rec_dims "$NF_REC_DIMS" \
    --nf_latents "$NF_LATENTS" \
    --nf_gru_units "$NF_GRU_UNITS" \
    --nf_flow_model "$NF_FLOW_MODEL" \
    --nf_flow_layers "$NF_FLOW_LAYERS" \
    --nf_time_net "$NF_TIME_NET" \
    --nf_time_hidden_dim "$NF_TIME_HIDDEN_DIM"
