import os
import sys
import importlib

import time
import datetime
import argparse
import numpy as np
import pandas as pd
from random import SystemRandom
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.optim as optim

from utils.tools import set_seed, print_formatted_dict, select_best_metrics
import lib.utils as utils
from lib.evaluation import compute_all_losses, evaluation

# from lib.parse_datasets_old import parse_datasets, get_input_and_pred_len

from lib.parse_datasets import parse_datasets, get_input_and_pred_len

_MODEL_MODULES = {
    "tPatchGNN": "models.tPatchGNN",
    "GPINet": "models.GPINet",
    "TimesNet": "models.TimesNet",
    "DLinear": "models.DLinear",
    "PatchTST": "models.PatchTST",
    "NeuralFlow": "models.NeuralFlow",
    "CRU": "models.CRU",
    "LatentODE": "models.LatentODE",
    "Informer": "models.Informer",
    "TimeMixer": "models.TimeMixer",
    "TimeLLM": "models.TimeLLM",
    "TTM": "models.TTM",
}


def _load_model_class(model_name: str):
    """Import only the selected model and isolate optional dependencies."""
    module_name = _MODEL_MODULES.get(model_name)
    if module_name is None:
        available = ", ".join(sorted(_MODEL_MODULES))
        raise ValueError(f"Unknown model {model_name!r}. Available: {available}")
    try:
        module = importlib.import_module(module_name)
        return getattr(module, model_name)
    except Exception as import_error:  # noqa: BLE001
        raise RuntimeError(
            f"Could not load --model {model_name} from {module_name}: "
            f"{import_error!r}"
        ) from import_error

from fusions.FusionModel import FusionModel
from fusions.load_llm import get_context_window_size


def get_args_from_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IMMTSF")

    # ── General / Execution Options ──────────────────────────────────────────────
    parser.add_argument(
        "--overwrite_args",
        action="store_true",
        help="overwrite args with fixed_params and tunable_params",
        default=False,
    )
    parser.add_argument(
        "--state",
        type=str,
        default="def",
        help='State of the experiment (e.g., "def", "train", "eval")',
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID to use")

    # ── Paths & Data Selection ───────────────────────────────────────────────────
    parser.add_argument(
        "--dataset", type=str, default="FNSPID", help="Which dataset to load"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for all data files",
    )
    parser.add_argument(
        "--num",
        dest="n",
        type=int,
        default=int(1e8),
        help="Total MIMIC subjects to sample before a 60/20/20 split",
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=42,
        help="Seed for independently sampling each MIMIC cohort size",
    )
    parser.add_argument(
        "--split_method",
        type=str,
        default="sample",
        choices=["instance", "sample"],
        help="Method to split the dataset into train/val/test",
    )
    parser.add_argument(
        "--enable_text",
        action="store_true",
        help="Enable multimodal text data",
    )
    parser.add_argument(
        "--use_text_embeddings",
        action="store_true",
        help="Enable pre-computed text embeddings",
    )

    # ── Data Processing / Windowing ──────────────────────────────────────────────
    parser.add_argument(
        "--time_unit",
        type=str,
        default="days",
        choices=["seconds", "minutes", "hours", "days", "weeks", "custom"],
        help="Time unit for the dataset",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=24,
        help="Historical window length (hours for physionet/mimic, months for ushcn)",
    )
    parser.add_argument(
        "--pred_window",
        type=int,
        default=24,
        help="Forecast horizon (prediction window)",
    )
    parser.add_argument(
        "--stride", type=int, default=24, help="Stride between consecutive patches"
    )

    # ── Temporal Patching (t-PatchGNN-specific) ──────────────────────────────────
    parser.add_argument(
        "-ps", "--patch_size", type=int, default=24, help="Size of each temporal patch"
    )
    parser.add_argument(
        "--npatch",
        type=int,
        default=None,
        help="Number of patches (default: history/patch_size)",
    )
    parser.add_argument(
        "--patch_stride",
        type=int,
        default=None,
        help="Stride between patches (defaults to patch_size)",
    )

    # ── Model Selection & Architecture ───────────────────────────────────────────
    parser.add_argument(
        "--model", type=str, default="tPatchGNN", help="Model architecture to use"
    )
    parser.add_argument(
        "--outlayer", type=str, default="Linear", help="Type of final output layer"
    )
    parser.add_argument(
        "-hd",
        "--hid_dim",
        type=int,
        default=64,
        help="Hidden units per layer (also default for some NF/CRU/ODE params)",
    )
    parser.add_argument(
        "-td", "--te_dim", type=int, default=10, help="Units for time‐encoding vectors"
    )
    parser.add_argument(
        "-nd",
        "--node_dim",
        type=int,
        default=10,
        help="Units for node‐embedding vectors",
    )
    parser.add_argument("--hop", type=int, default=1, help="Number of GNN hops")
    parser.add_argument(
        "--tf_layer", type=int, default=1, help="Number of Transformer layers"
    )
    parser.add_argument(
        "--nlayer",
        type=int,
        default=1,
        help="Number of layers in the time‐series backbone",
    )
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=2, help="num of heads")
    parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument(
        "--down_sampling_layers",
        type=int,
        default=3,
        help="num of down sampling layers",
    )
    parser.add_argument(
        "--down_sampling_window", type=int, default=2, help="down sampling window size"
    )
    parser.add_argument(
        "--down_sampling_method",
        type=str,
        default="avg",
        help="down sampling method, only support avg, max, conv",
    )
    parser.add_argument(
        "--decomp_method",
        type=str,
        default="moving_avg",
        help="method of series decompsition, only support moving_avg or dft_decomp",
    )
    parser.add_argument(
        "--channel_independence",
        type=int,
        default=1,
        help="0: channel dependence 1: channel independence for FreTS model",
    )
    parser.add_argument(
        "--use_norm",
        type=int,
        default=1,
        help="whether to use normalize; True 1 False 0",
    )

    # TTM
    parser.add_argument("--n_vars", type=int, default=7, help="number of variables")
    parser.add_argument(
        "--mode",
        type=str,
        default="mix_channel",
        help="allowed values: common_channel, mix_channel",
    )
    parser.add_argument(
        "--AP_levels", type=int, default=3, help="number of attention patching levels"
    )
    parser.add_argument(
        "--use_decoder", action="store_true", help="use decoder", default=True
    )
    parser.add_argument(
        "--d_mode",
        type=str,
        default="common_channel",
        help="allowed values: common_channel, mix_channel",
    )
    parser.add_argument("--d_d_model", type=int, default=64, help="d_model in decoder")

    # Time-LLM
    parser.add_argument(
        "--ts_vocab_size",
        type=int,
        default=1000,
        help="size of a small collection of text prototypes in llm",
    )
    parser.add_argument(
        "--domain_des",
        type=str,
        default="The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.",
        help="domain description",
    )
    parser.add_argument(
        "--input_token_len", type=int, default=576, help="input token length"
    )
    parser.add_argument(
        "--output_token_len", type=int, default=96, help="output token length"
    )
    parser.add_argument(
        "--llm_model_timellm",
        type=str,
        default="GPT2",
        help="LLM model (for TimeLLM), LLAMA, GPT2, BERT, OPT are supported",
    )
    parser.add_argument(
        "--llm_layers_timellm", type=int, default=6, help="number of layers in llm"
    )

    # ── NeuralFlow Specific Hyperparameters ──────────────────────────────────────
    parser.add_argument(
        "--nf_latents", type=int, default=20, help="NeuralFlow: Latent dimension"
    )
    parser.add_argument(
        "--nf_rec_dims", type=int, default=40, help="NeuralFlow: Recognition dimensions"
    )
    parser.add_argument(
        "--nf_gru_units",
        type=int,
        default=32,
        help="NeuralFlow: GRU units",
    )
    parser.add_argument(
        "--nf_hidden_layers",
        type=int,
        default=3,
        help="NeuralFlow: Number of hidden layers in ODE func",
    )
    parser.add_argument(
        "--nf_hidden_dim",
        type=int,
        default=32,
        help="NeuralFlow: Hidden dimension in ODE func",
    )
    parser.add_argument(
        "--nf_flow_model",
        type=str,
        default="coupling",
        choices=["coupling", "resnet", "gru"],
        help="NeuralFlow: Type of flow model",
    )
    parser.add_argument(
        "--nf_flow_layers",
        type=int,
        default=2,
        help="NeuralFlow: Number of flow layers",
    )
    parser.add_argument(
        "--nf_time_net",
        type=str,
        default="TimeLinear",
        help="NeuralFlow: Time network type",
    )
    parser.add_argument(
        "--nf_time_hidden_dim",
        type=int,
        default=8,
        help="NeuralFlow: Time network hidden dimension",
    )
    parser.add_argument(
        "--nf_solver", type=str, default="dopri5", help="NeuralFlow: ODE solver"
    )
    parser.add_argument(
        "--nf_solver_step",
        type=float,
        default=0.05,
        help="NeuralFlow: Solver step size",
    )
    parser.add_argument(
        "--nf_atol",
        type=float,
        default=1e-4,
        help="NeuralFlow: Absolute tolerance for solver",
    )
    parser.add_argument(
        "--nf_rtol",
        type=float,
        default=1e-3,
        help="NeuralFlow: Relative tolerance for solver",
    )
    parser.add_argument(
        "--nf_odenet", type=str, default="concat", help="NeuralFlow: ODE network type"
    )
    parser.add_argument(
        "--nf_activation",
        type=str,
        default="Tanh",
        help="NeuralFlow: Activation function",
    )
    parser.add_argument(
        "--nf_final_activation",
        type=str,
        default="Identity",
        help="NeuralFlow: Final activation function",
    )
    parser.add_argument(
        "--nf_obsrv_std",
        type=float,
        default=0.01,
        help="NeuralFlow: Observation standard deviation",
    )
    parser.add_argument(
        "--nf_weight_decay",
        type=float,
        default=0.0001,
        help="NeuralFlow: Weight decay for internal optimizer (if applicable)",
    )
    parser.add_argument(
        "--nf_quantization",
        type=float,
        default=0.0,
        help="NeuralFlow: Quantization parameter",
    )
    parser.add_argument(
        "--nf_max_t",
        type=float,
        default=5.0,
        help="NeuralFlow: Max time for ODE integration",
    )
    parser.add_argument(
        "--nf_mixing", type=float, default=0.0001, help="NeuralFlow: Mixing coefficient"
    )
    parser.add_argument(
        "--nf_gob_prep_hidden",
        type=int,
        default=10,
        help="NeuralFlow: GOB prep hidden units",
    )
    parser.add_argument(
        "--nf_gob_cov_hidden",
        type=int,
        default=50,
        help="NeuralFlow: GOB cov hidden units",
    )
    parser.add_argument(
        "--nf_gob_p_hidden", type=int, default=25, help="NeuralFlow: GOB p hidden units"
    )
    parser.add_argument(
        "--nf_invertible",
        type=int,
        default=1,
        help="NeuralFlow: Invertible flag (0 or 1)",
    )
    parser.add_argument(
        "--nf_components", type=int, default=8, help="NeuralFlow: Number of components"
    )
    parser.add_argument(
        "--nf_decoder_type",
        type=str,
        default="continuous",
        help="NeuralFlow: Decoder type",
    )
    parser.add_argument(
        "--nf_rnn", type=str, default="gru", help="NeuralFlow: RNN type"
    )
    parser.add_argument(
        "--nf_marks", type=int, default=0, help="NeuralFlow: Marks flag (0 or 1)"
    )
    parser.add_argument(
        "--nf_density_model",
        type=str,
        default="independent",
        help="NeuralFlow: Density model type",
    )
    parser.add_argument(
        "--nf_extrap",
        type=int,
        default=0,
        help="NeuralFlow: Extrapolation flag (0 or 1)",
    )

    # ── CRU Specific Hyperparameters ─────────────────────────────────────────────
    parser.add_argument(
        "--cru_lsd",
        type=int,
        default=None,
        help="CRU: Latent state dimension (defaults to hid_dim if None)",
    )
    parser.add_argument(
        "--cru_hidden_units",
        type=int,
        default=None,
        help="CRU: Hidden units for internal MLPs (defaults to hid_dim if None)",
    )
    parser.add_argument(
        "--cru_enc_num_layers",
        type=int,
        default=1,
        help="CRU: Number of encoder layers",
    )
    parser.add_argument(
        "--cru_dec_num_layers",
        type=int,
        default=1,
        help="CRU: Number of decoder layers",
    )
    parser.add_argument(
        "--cru_num_layers", type=int, default=1, help="CRU: Number of CRU layers"
    )
    parser.add_argument(
        "--cru_dropout_type",
        type=str,
        default="None",
        choices=["None", "Zoneout", "Variational"],
        help="CRU: Dropout type",
    )
    parser.add_argument(
        "--cru_dropout_rate", type=float, default=0.0, help="CRU: Dropout rate"
    )
    parser.add_argument(
        "--cru_use_gate_hidden_states",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="CRU: Use gate hidden states (True/False)",
    )
    parser.add_argument(
        "--cru_use_ode_for_gru",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="CRU: Use ODE for GRU (True/False)",
    )
    parser.add_argument(
        "--cru_use_decay_gravity_gate",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="CRU: Use decay gravity gate (True/False)",
    )
    parser.add_argument(
        "--cru_use_gravity_gate",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="CRU: Use gravity gate (True/False)",
    )
    parser.add_argument(
        "--cru_use_decay_input_gate",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="CRU: Use decay input gate (True/False)",
    )
    parser.add_argument(
        "--cru_use_input_gate",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="CRU: Use input gate (True/False)",
    )
    parser.add_argument(
        "--cru_use_skip_connection",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="CRU: Use skip connection (True/False)",
    )
    parser.add_argument(
        "--cru_solver",
        type=str,
        default="euler",
        choices=["euler", "rk4"],
        help="CRU: Solver for ODEs if used",
    )
    parser.add_argument(
        "--ts",
        type=float,
        default=0.3,
        help="Scaling factor of timestamps for numerical stability.",
    )
    parser.add_argument(
        "--grad_clip", action="store_true", help="If to use gradient clipping."
    )

    # ── LatentODE Specific Hyperparameters ───────────────────────────────────────
    parser.add_argument(
        "--ode_latents", type=int, default=20, help="LatentODE: Latent dimension"
    )
    parser.add_argument(
        "--ode_units",
        type=int,
        default=32,
        help="LatentODE: Units in ODE function network",
    )
    parser.add_argument(
        "--ode_gen_layers",
        type=int,
        default=1,
        help="LatentODE: Layers in ODE function generator",
    )
    parser.add_argument(
        "--ode_rec_dims",
        type=int,
        default=32,
        help="LatentODE: Recognition RNN hidden dimensions",
    )
    parser.add_argument(
        "--ode_rec_layers",
        type=int,
        default=1,
        help="LatentODE: Layers in recognition RNN",
    )
    parser.add_argument(
        "--ode_gru_units",
        type=int,
        default=32,
        help="LatentODE: GRU units in recognition RNN (defaults to hid_dim if None)",
    )
    parser.add_argument(
        "--ode_poisson",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="LatentODE: Use Poisson process for observations (True/False)",
    )
    parser.add_argument(
        "--ode_classif",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="LatentODE: Perform classification task (True/False)",
    )
    parser.add_argument(
        "--ode_linear_classif",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="LatentODE: Use linear classifier (True/False)",
    )
    parser.add_argument(
        "--ode_z0_encoder",
        type=str,
        default="odernn",
        choices=["odernn", "rnn"],
        help="LatentODE: Type of encoder for z0",
    )
    parser.add_argument(
        "--ode_obsrv_std",
        type=float,
        default=0.01,
        help="LatentODE: Observation standard deviation",
    )
    parser.add_argument(
        "--ode_n_traj_samples",
        type=int,
        default=1,
        help="LatentODE: Number of trajectory samples for reconstruction",
    )

    # ── Fusion Modules ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--TTF_module",
        type=str,
        default="TTF_T2V_XAttn",
        choices=["TTF_RecAvg", "TTF_T2V_XAttn", "TTF_SemTime_Slots"],
        help="Timestamp-to-Time Fusion module",
    )
    parser.add_argument(
        "--MMF_module",
        type=str,
        default="MMF_XAttn_Add",
        choices=["MMF_GR_Add", "MMF_XAttn_Add", "MMF_VarTime_SlotGate"],
        help="Multimodal Fusion module",
    )
    parser.add_argument(
        "--llm_model_fusion",
        type=str,
        default="GPT2",
        help="LLM model (for Fusion), LLAMA, GPT2, BERT are supported",
    )
    parser.add_argument(
        "--llm_layers_fusion",
        type=int,
        default=6,
        help="number of layers in llm fusion",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max tokens per note (for TTF)",
    )
    parser.add_argument(
        "--d_txt",
        type=int,
        default=768,
        help="Text embedding dimension (for TTF)",
    )
    parser.add_argument(
        "--recency_sigma",
        type=float,
        default=1.0,
        help="Recency sigma for recency-aware TTF modules",
    )
    parser.add_argument(
        "--semantic_slots",
        type=int,
        default=4,
        help="Number of latent semantic slots in TTF_SemTime_Slots",
    )
    parser.add_argument(
        "--semantic_time_gate_bias",
        type=float,
        default=-1.0,
        help="Initial adaptive time-gate bias in TTF_SemTime_Slots",
    )
    parser.add_argument(
        "--absolute_recency_floor",
        type=float,
        default=0.1,
        help=(
            "Minimum absolute text strength after age decay in "
            "TTF_SemTime_Slots"
        ),
    )
    parser.add_argument(
        "--n_heads_fusion",
        type=int,
        default=1,
        help=(
            "Number of attention heads used by external Fusion modules; "
            "GPINet's Gaussian text-background route does not use attention"
        ),
    )
    parser.add_argument(
        "--gpinet_query_points",
        type=int,
        default=24,
        help=(
            "Number of historical GP grid points used by GPINet. Native "
            "Gaussian text alignment uses this same reference grid."
        ),
    )
    parser.add_argument(
        "--gpinet_text_time_sigma_hours",
        type=float,
        default=3.0,
        help=(
            "Gaussian bandwidth in hours for mapping irregular reports to "
            "the GPINet text-background node"
        ),
    )
    parser.add_argument(
        "--mmf_slot_attn_dim",
        type=int,
        default=128,
        help="Attention dimension in MMF_VarTime_SlotGate",
    )
    parser.add_argument(
        "--mmf_slot_gate_bias",
        type=float,
        default=0.0,
        help="Initial residual-gate bias in MMF_VarTime_SlotGate",
    )
    parser.add_argument(
        "--mmf_delta_init_std",
        type=float,
        default=1e-2,
        help="Initial std of the semantic-slot residual output layer",
    )
    parser.add_argument(
        "--fusion_gate_warmup_epochs",
        type=int,
        default=5,
        help=(
            "Epochs with a fixed non-zero semantic-slot text gate before "
            "learned rejection is enabled"
        ),
    )
    parser.add_argument(
        "--fusion_gate_warmup_value",
        type=float,
        default=0.5,
        help="Fixed semantic-slot text gate used during warmup",
    )
    parser.add_argument(
        "--fusion_lr_multiplier",
        type=float,
        default=2.0,
        help="Fusion-branch learning rate multiplier relative to --lr",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.5,
        help="Text correction scale used by residual MMF modules",
    )

    # ── Training Hyperparameters ─────────────────────────────────────────────────
    parser.add_argument("--epoch", type=int, default=1000, help="Max training epochs")
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        # default=10,
        help="Early‐stopping patience",
    )
    parser.add_argument(
        "--early_stop_delta",
        type=float,
        default=1e-4,
        help="Minimum change in the monitored metric to qualify as improvement",
    )

    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--w_decay",
        type=float,
        # default=0.0,
        # default=0.001,
        default=0.01,
        help="Weight‐decay (L2 regularization)",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=False,
        # default=True,
        help="Enable Automatic Mixed Precision (AMP) training",
    )
    parser.add_argument(
        "--detect_anomaly",
        action="store_true",
        default=False,
        help=(
            "Enable PyTorch autograd anomaly detection. Intended only for "
            "debugging because it increases runtime and memory usage."
        ),
    )

    # ── Logging & Checkpointing ──────────────────────────────────────────────────
    parser.add_argument(
        "--logmode", type=str, default="a", help='File mode for logging (e.g. "w", "a")'
    )
    parser.add_argument(
        "--save",
        type=str,
        default="experiments/",
        help="Directory in which to save model checkpoints",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Experiment ID to load for evaluation (if any)",
    )

    args = parser.parse_args()

    if not 0.0 <= args.absolute_recency_floor <= 1.0:
        parser.error("--absolute_recency_floor must be in [0, 1]")
    if args.mmf_delta_init_std <= 0:
        parser.error("--mmf_delta_init_std must be > 0")
    if args.fusion_gate_warmup_epochs < 0:
        parser.error("--fusion_gate_warmup_epochs must be >= 0")
    if not 0.0 < args.fusion_gate_warmup_value <= 1.0:
        parser.error("--fusion_gate_warmup_value must be in (0, 1]")
    if args.fusion_lr_multiplier <= 0:
        parser.error("--fusion_lr_multiplier must be > 0")
    if args.gpinet_query_points < 2:
        parser.error("--gpinet_query_points must be >= 2")
    if args.gpinet_text_time_sigma_hours <= 0:
        parser.error("--gpinet_text_time_sigma_hours must be > 0")
    if (
        args.enable_text
        and args.TTF_module == "TTF_SemTime_Slots"
        and args.semantic_slots < 2
    ):
        parser.error(
            "TTF_SemTime_Slots requires --semantic_slots >= 2; "
            "one slot disables semantic routing"
        )

    # Default nf_gru_units and nf_hidden_dim to args.hid_dim if not provided
    if args.nf_gru_units is None:
        args.nf_gru_units = args.hid_dim
    if args.nf_hidden_dim is None:
        args.nf_hidden_dim = args.hid_dim

    # Default cru_lsd and cru_hidden_units to args.hid_dim if not provided
    if args.cru_lsd is None:
        args.cru_lsd = args.hid_dim
    if args.cru_hidden_units is None:
        args.cru_hidden_units = args.hid_dim

    # Note: ode_gru_units default handling is now inside the LatentODE model wrapper,
    # based on args.hid_dim if args.ode_gru_units is None when passed.

    args.npatch = (
        int(np.ceil((args.history - args.patch_size) / args.stride)) + 1
    )  # (window size for a patch)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    file_name = os.path.basename(__file__)[:-3]
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.device = torch.device("cpu")  # use cpu to debug
    args.PID = os.getpid()
    print("PID, device:", args.PID, args.device)

    return args


def update_args_from_fixed_params(
    args: argparse.Namespace, fixed_params: dict
) -> argparse.Namespace:
    # Update args
    for key, value in fixed_params.items():
        if not hasattr(args, key):
            print(f"AttributeError: {key} not found in args")
        print("### [Fixed] Set {} to {}".format(key, value))
        setattr(args, key, value)

    return args


def update_args_from_tunable_params(
    args: argparse.Namespace, tunable_params: dict
) -> argparse.Namespace:
    # Update args
    for key, value in tunable_params.items():
        if not hasattr(args, key):
            print(f"AttributeError: {key} not found in args")
        print("### [Tunable] Set {} to {}".format(key, value))
        setattr(args, key, value)

    return args


def update_args_for_dataset(args: argparse.Namespace) -> argparse.Namespace:
    # Update args based on the dataset
    if args.dataset == "GDELT":
        args.history = 14
        args.pred_window = 14
        args.stride = 14
        args.time_unit = "days"
    elif args.dataset == "RepoHealth":
        args.history = 31
        args.pred_window = 31
        args.stride = 31
        args.time_unit = "days"
    elif args.dataset == "MIMIC":
        args.history = 24
        args.pred_window = 24
        args.stride = 24
        args.time_unit = "hours"
    elif args.dataset == "FNSPID":
        args.history = 31
        args.pred_window = 31
        args.stride = 31
        args.time_unit = "days"
    elif args.dataset == "ClusterTrace":
        args.history = 12
        args.pred_window = 12
        args.stride = 12
        args.time_unit = "hours"
    elif args.dataset == "StudentLife":
        args.history = 31
        args.pred_window = 31
        args.stride = 31
        args.time_unit = "days"
    elif args.dataset == "ILINet":
        args.history = 36
        args.pred_window = 36
        args.stride = 4
        args.time_unit = "weeks"
    elif args.dataset == "CESNET":
        args.history = 7
        args.pred_window = 7
        args.stride = 7
        args.time_unit = "days"
    elif args.dataset == "EPA-Air":
        args.history = 7
        args.pred_window = 7
        args.stride = 7
        args.time_unit = "days"

    return args


def update_args_for_model(args: argparse.Namespace) -> argparse.Namespace:
    # Update args based on the model
    # ? MTS
    if args.model == "Informer":
        args.e_layers = 2
        args.d_layers = 1
        args.factor = 3
    elif args.model == "DLinear":
        pass
    elif args.model == "PatchTST":
        args.e_layers = 1
        args.d_layers = 1
        args.n_heads = 2
    elif args.model == "TimesNet":
        args.e_layers = 2
        args.d_layers = 1
        args.factor = 3
        args.d_model = 16
        args.d_ff = 32
        args.top_k = 5
    elif args.model == "TimeMixer":
        args.e_layers = 2
        args.d_model = 16
        args.d_ff = 32
        args.down_sampling_layers = 3
        args.down_sampling_method = "avg"
        args.down_sampling_window = 2
    # ? LMTS
    elif args.model == "TimeLLM":
        args.input_token_len = 16
        args.output_token_len = 96
        args.d_model = 32
        args.d_ff = 128
        args.llm_model_timellm = "GPT2"
        args.llm_layers_timellm = 6
    elif args.model == "TTM":
        args.input_token_len = 16
        args.output_token_len = 96
        args.d_model = 1024
        args.AP_levels = 3
        args.e_layers = 3
        args.d_layers = 2
        args.d_d_model = 64
        args.patch_size = args.history // 4

        # args.history = 96
        # args.pred_window = 96
        # args.stride = 96
        # args.time_unit = "days"
    # ? IMTS
    elif args.model == "CRU":
        args.cru_lsd = 32
        args.cru_hidden_units = 32
        args.ts = 0.3
        args.cru_enc_var_activation = "square"
        args.cru_dec_var_activation = "exp"
        args.grad_clip = True
    elif args.model == "LatentODE":
        args.ode_rec_dims = 32
        args.ode_units = 32
        args.ode_gru_units = 32
        args.ode_rec_layers = 1
        args.ode_gen_layers = 1
    elif args.model == "NeuralFlow":
        args.nf_extrap = 0
        args.nf_hidden_layers = 3
        args.nf_hidden_dim = 32
        args.nf_rec_dims = 40
        args.nf_latents = 20
        args.nf_gru_units = 32
        args.nf_flow_model = "coupling"
        args.nf_flow_layers = 2
        args.nf_time_net = "TimeLinear"
        args.nf_time_hidden_dim = 8
    elif args.model == "tPatchGNN":
        args.patch_size = 24
        args.n_heads = 1
        args.tf_layer = 1
        args.nlayer = 1
        args.te_dim = 10
        args.node_dim = 10
        args.hid_dim = 32
        args.outlayer = "Linear"
    elif args.model == "GPINet":
        # Mirror tPatchGNN's capacity so backbone size isn't a confound.
        args.nlayer = 1
        args.hop = 1
        args.te_dim = 10
        args.node_dim = 10
        args.hid_dim = 32

    return args


def update_args(
    args: argparse.Namespace,
    fixed_params: dict,
    tunable_params: dict,
) -> argparse.Namespace:
    # Check if there are duplicated keys
    duplicated_keys = set(fixed_params.keys()) & set(tunable_params.keys())
    assert not duplicated_keys, f"Duplicated keys found: {duplicated_keys}"

    # Update args from fixed_params, tunable_params, and dataset
    if args.overwrite_args:
        args = update_args_from_fixed_params(args, fixed_params)
        args = update_args_from_tunable_params(args, tunable_params)
        args = update_args_for_dataset(args)
        args = update_args_for_model(args)

    return args


def trainable(
    tunable_params: dict,
    fixed_params: dict,
    args: argparse.Namespace,
) -> dict:
    # Update args
    args = update_args(args, fixed_params, tunable_params)

    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random() * 100000)
    ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + ".ckpt")

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2) :]
    input_command = " ".join(input_command)

    ##################################################################
    # Update max_length if needed
    if args.enable_text:
        args.max_length = 512 if args.llm_model_fusion == "BERT" else 1024
        # context_window_size = get_context_window_size(
        #     args.llm_model_fusion, args.device
        # )
        # if args.max_length > context_window_size:
        #     print(
        #         f"Overriding max_length from {args.max_length} to {context_window_size}"
        #         " to match the LLM model's context window size."
        #     )
        #     args.max_length = context_window_size

    # Pass model name to parse_datasets to select the correct collate_fn
    data_obj = parse_datasets(args)

    ### Model setting ###
    args.C = data_obj["input_dim"]
    args.enc_in = args.C
    args.c_out = args.C
    args.input_len, args.pred_len = get_input_and_pred_len(data_obj)
    model_class = _load_model_class(args.model)
    model = model_class(args).to(args.device)

    # GPINet with pre-computed embeddings owns its text path: a fixed Gaussian
    # kernel maps irregular reports to one regular text-background node, which
    # is appended to the numerical nodes before MTGNN. Other models (and
    # raw-text GPINet runs) keep the benchmark's external FusionModel.
    native_text_fusion = bool(
        args.enable_text and getattr(model, "native_text_enabled", False)
    )
    fusion = (
        FusionModel(args).to(args.device)
        if args.enable_text and not native_text_fusion
        else None
    )

    ##################################################################

    if args.n < 12000:
        args.state = "debug"
        log_path = "logs/{}_{}_{}.log".format(args.dataset, args.model, args.state)
    else:
        log_path = "logs/{}_{}_{}_{}patch_{}stride_{}layer_{}lr.log".format(
            args.dataset,
            args.model,
            args.state,
            args.patch_size,
            args.stride,
            args.nlayer,
            args.lr,
        )

    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(
        logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode
    )
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(input_command)
    logger.info(args)
    if native_text_fusion:
        logger.info(
            "Text route: GPINet Gaussian alignment + MTGNN background node"
        )
    elif fusion is not None:
        logger.info("Text route: external TTF/MMF prediction fusion")

    # Keep the text branch on the same optimizer protocol as external TTF/MMF:
    # 2x learning rate, no weight decay, and independent gradient clipping.
    native_text_parameters = (
        list(model.text_grid_fusion.parameters())
        if native_text_fusion
        else []
    )
    native_text_parameter_ids = {id(p) for p in native_text_parameters}
    model_parameters = [
        p for p in model.parameters() if id(p) not in native_text_parameter_ids
    ]
    external_fusion_parameters = (
        list(fusion.parameters()) if fusion is not None else []
    )
    text_parameters = native_text_parameters + external_fusion_parameters
    trainable_parameters = model_parameters + text_parameters
    if text_parameters:
        optimizer = optim.Adam(
            [
                {
                    "params": model_parameters,
                    "weight_decay": args.w_decay,
                },
                {
                    "params": text_parameters,
                    "weight_decay": 0.0,
                    "lr": args.lr * args.fusion_lr_multiplier,
                },
            ],
            lr=args.lr,
        )
    else:
        optimizer = optim.Adam(
            model_parameters,
            lr=args.lr,
            weight_decay=args.w_decay,
        )

    def _clip_training_gradients():
        # Clip the large GPINet and small fusion branch independently.  Joint
        # clipping can suppress the fusion gradient when the backbone norm is
        # much larger.
        torch.nn.utils.clip_grad_norm_(model_parameters, max_norm=1.0)
        if text_parameters:
            torch.nn.utils.clip_grad_norm_(text_parameters, max_norm=1.0)

    def _log_fusion_bootstrap(epoch, step, batch_dict):
        if epoch != 0 or step >= 3:
            return
        if native_text_fusion:
            text_module = model.text_grid_fusion
            delta_out = text_module.text_proj
            prefix = "GPINetTextBackgroundBootstrap"
        else:
            mmf = getattr(fusion, "mmf", None)
            delta_out = getattr(mmf, "delta_out", None)
            prefix = "FusionBootstrap"
        if delta_out is None:
            return
        grad = delta_out.weight.grad
        grad_mean = 0.0 if grad is None else grad.detach().abs().mean().item()
        grad_max = 0.0 if grad is None else grad.detach().abs().max().item()
        weight_norm = delta_out.weight.detach().norm().item()
        notes = batch_dict.get("notes_embeddings")
        text_samples = total_samples = real_notes = -1
        if torch.is_tensor(notes) and notes.ndim == 3:
            note_mask = notes.detach().abs().sum(dim=-1) > 0
            text_samples = int(note_mask.any(dim=-1).sum().item())
            total_samples = int(note_mask.shape[0])
            real_notes = int(note_mask.sum().item())
        print(
            f"[{prefix}] epoch={epoch} step={step} "
            f"text_samples={text_samples}/{total_samples} "
            f"real_notes={real_notes} delta_grad_mean={grad_mean:.3e} "
            f"delta_grad_max={grad_max:.3e} "
            f"delta_weight_norm={weight_norm:.3e}"
        )

    def _nan_hook(module, inputs, output, name):
        # output might be a Tensor or tuple of Tensors
        outs = output if isinstance(output, (list, tuple)) else (output,)
        for o in outs:
            if isinstance(o, torch.Tensor) and torch.isnan(o).any():
                raise RuntimeError(f"NaN in forward of {name}")

    def register_forward_nan_checks(model):
        for name, module in model.named_modules():
            module.register_forward_hook(
                lambda mod, inp, out, name=name: _nan_hook(mod, inp, out, name)
            )

    def _grad_hook(grad, name):
        if torch.isnan(grad).any():
            raise RuntimeError(f"NaN in grad for parameter {name}")
        return grad

    def register_grad_nan_checks(model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name: _grad_hook(grad, name))

    register_forward_nan_checks(model)
    register_grad_nan_checks(model)

    scaler = GradScaler() if args.use_amp else None

    best_val_mse = np.inf
    test_res = None
    no_improve_counter = 0
    for itr in range(args.epoch):
        st = time.time()

        if fusion is not None:
            fusion.set_training_epoch(itr)
        gate_warmup_epochs = int(
            getattr(getattr(fusion, "mmf", None), "gate_warmup_epochs", 0)
        )
        gate_warmup_active = (
            fusion is not None and itr < gate_warmup_epochs
        )

        ### Training ###
        model.train()
        if fusion is not None:
            fusion.train()
        iter_data = tqdm(data_obj["train_dataloader"], desc="Training")
        for step, batch_dict in enumerate(iter_data):
            optimizer.zero_grad(set_to_none=True)
            # with torch.autograd.set_detect_anomaly(args.detect_anomaly):
            #     train_res = compute_all_losses(
            #         model, fusion, batch_dict, args.enable_text
            #     )
            #     train_res["loss"].backward()
            #     torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            # optimizer.step()

            # # Update the progress bar description with current loss
            # current_loss = train_res["loss"].item()
            # iter_data.set_description(f"Epoch {itr}, Loss: {current_loss:.5f}")
            try:
                with torch.autograd.set_detect_anomaly(args.detect_anomaly):
                    if args.use_amp:
                        with autocast():
                            train_res = compute_all_losses(
                                model,
                                fusion,
                                batch_dict,
                                args.enable_text,
                                args.use_text_embeddings,
                            )
                            loss = train_res["loss"]
                        scaler.scale(loss).backward()
                        # AMP gradients must be unscaled before clipping.
                        scaler.unscale_(optimizer)
                        _log_fusion_bootstrap(itr, step, batch_dict)
                        _clip_training_gradients()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        train_res = compute_all_losses(
                            model,
                            fusion,
                            batch_dict,
                            args.enable_text,
                            args.use_text_embeddings,
                        )
                        loss = train_res["loss"]
                        loss.backward()
                        _log_fusion_bootstrap(itr, step, batch_dict)
                        _clip_training_gradients()
                        optimizer.step()

                    # Update the progress bar description with current loss
                    current_loss = loss.item()
                    iter_data.set_description(f"Epoch {itr}, Loss: {current_loss:.5f}")

            except (RuntimeError, AssertionError) as e:
                if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
                    print(
                        f"[OOM] Epoch {itr}, step {step}: "
                        "skipping this batch due to out-of-memory."
                    )
                    notes = batch_dict.get("notes_embeddings")
                    if torch.is_tensor(notes):
                        print(f"[OOM] notes_embeddings shape: {tuple(notes.shape)}")
                    print(
                        "[OOM] observed_data shape: "
                        f"{tuple(batch_dict['observed_data'].shape)}"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                        print(
                            f"[OOM] CUDA allocated/reserved: "
                            f"{allocated:.1f}/{reserved:.1f} MiB"
                        )
                        torch.cuda.empty_cache()
                elif isinstance(
                    e, AssertionError
                ) and "t must be strictly increasing or decreasing" in str(e):
                    print(e)
                    print(
                        f"[Bad Data] Step {step}: Skipping batch due to invalid timestamps."
                    )
                else:
                    raise e  # Re-raise unknown exceptions
                continue

        ### Validation ###
        model.eval()
        if fusion is not None:
            fusion.eval()
        with torch.no_grad():
            val_res = evaluation(
                model,
                fusion,
                data_obj["val_dataloader"],
                args.enable_text,
                args.use_text_embeddings,
            )

            # Compute improvement over best MSE
            improvement = best_val_mse - val_res["mse"]

            if improvement > args.early_stop_delta:
                best_val_mse = val_res["mse"]
                best_iter = itr
                no_improve_counter = 0  # Reset early stopping counter

                ### Testing ###
                test_res = evaluation(
                    model,
                    fusion,
                    data_obj["test_dataloader"],
                    args.enable_text,
                    args.use_text_embeddings,
                )
            elif gate_warmup_active:
                # The residual/TTF branch is still learning under a protected
                # non-zero gate.  Do not let the fast numerical path terminate
                # training before the learned rejection gate is even enabled.
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            logger.info("- Epoch {:03d}, ExpID {}".format(itr, experimentID))
            logger.info(
                "Train - Loss (one batch): {:.5f}".format(train_res["loss"].item())
            )
            logger.info(
                "Val - Loss, MSE, MAE: {:.5f}, {:.5f}, {:.5f}".format(
                    val_res["loss"],
                    val_res["mse"],
                    val_res["mae"],
                )
            )
            if native_text_fusion:
                logger.info(
                    "Val - Gaussian weight mean/max, text temporal "
                    "variation, background RMS: {:.5f}, {:.5f}, {:.5f}, "
                    "{:.5f}".format(
                        val_res.get(
                            "gpinet_text_gaussian_weight_mean",
                            float("nan"),
                        ),
                        val_res.get(
                            "gpinet_text_gaussian_weight_max",
                            float("nan"),
                        ),
                        val_res.get(
                            "gpinet_text_background_temporal_variation",
                            float("nan"),
                        ),
                        val_res.get(
                            "gpinet_text_background_rms",
                            float("nan"),
                        ),
                    )
                )
            elif "text_gate_mean" in val_res:
                logger.info(
                    "Val - Text gate mean, attention entropy: {:.5f}, {:.5f}".format(
                        val_res["text_gate_mean"],
                        val_res.get("text_attention_entropy", float("nan")),
                    )
                )
            if gate_warmup_active:
                logger.info(
                    "Val - Fusion gate warmup active: epoch %d/%d",
                    itr + 1,
                    gate_warmup_epochs,
                )
            if test_res != None:
                logger.info(
                    "Test - Best epoch, Loss, MSE, MAE: {}, {:.5f}, {:.5f}, {:.5f}".format(
                        best_iter,
                        test_res["loss"],
                        test_res["mse"],
                        test_res["mae"],
                    )
                )
            logger.info("Time spent: {:.2f}s".format(time.time() - st))

        if no_improve_counter >= args.patience:
            print("Exp has been early stopped!")
            break

    assert (
        test_res is not None
    ), "No test results available. Please check the training loop."

    return test_res


#####################################################################################################

if __name__ == "__main__":
    """------------------------------------"""
    # data_name = "GDELT"  # type 1.1
    # data_name = "RepoHealth"  # type 1.2
    # data_name = "MIMIC"  # type 1.3 (not ready)
    # data_name = "FNSPID"  # type 2.1
    data_name = "ClusterTrace"  # type 2.2 (not ready)
    # data_name = "StudentLife"  # type 2.3
    # data_name = "ILINet"  # type 3.1
    # data_name = "CESNET"  # type 3.2
    # data_name = "EPA-Air"  # type 3.3

    # ? MTS
    # model_name = "Informer"
    # model_name = "DLinear"
    # model_name = "PatchTST"
    # model_name = "TimesNet"
    # model_name = "TimeMixer"
    # ? LMTS
    # model_name = "TimeLLM"
    # model_name = "TTM"
    # ? IMTS
    # model_name = "CRU"
    model_name = "LatentODE"
    # model_name = "NeuralFlow"
    # model_name = "tPatchGNN"

    enable_text = False
    # enable_text = True

    # use_text_embeddings = False
    use_text_embeddings = True

    TTF_module = "TTF_RecAvg"
    # TTF_module = "TTF_T2V_XAttn"
    MMF_module = "MMF_GR_Add"
    # MMF_module = "MMF_XAttn_Add"

    llm_model_fusion = "GPT2"
    # llm_model_fusion = "BERT"
    # llm_model_fusion = "Llama"
    # llm_model_fusion = "DeepSeek"

    llm_layers_fusion = None
    # llm_layers_fusion = 6

    split_method = "sample"
    # split_method = "instance"  # only for in-domain transfer learning

    tunable_params_path = None
    # tunable_params_path = Path(
    #     "exp_settings_and_results",
    #     "single_granularity",
    #     model_name,
    #     f"{data_name}.json",
    # )

    # batch_size = 1
    # batch_size = 2  # 8G
    batch_size = 8
    # batch_size = 16  # 24G
    # batch_size = 32
    # batch_size = 64
    # batch_size = 256
    """------------------------------------"""
    # Setup args
    args = get_args_from_parser()

    # Set all random seeds (Python, NumPy, PyTorch)
    set_seed(args.seed)

    # Setup fixed params
    fixed_params = {
        "dataset": data_name,
        "model": model_name,
        "batch_size": batch_size,
        "enable_text": enable_text,
        "use_text_embeddings": use_text_embeddings,
        "split_method": split_method,
        "TTF_module": TTF_module,
        "MMF_module": MMF_module,
        "llm_model_fusion": llm_model_fusion,
        "llm_layers_fusion": llm_layers_fusion,
    }

    # Setup tunable params
    if tunable_params_path is None:
        tunable_params = {
            # "lr": 1e-2,
            "lr": 1e-3,
            # "lr": 1e-4,
            "patience": 3,
            # "kappa": 0.1,
            # "recency_sigma": 0.1,
            # "n_heads_fusion": 2,
        }

    # Run
    best_metrics = trainable(tunable_params, fixed_params, args)
    print_formatted_dict(best_metrics)
    print("### Done ###")
