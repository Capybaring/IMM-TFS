"""GPINet: GP interpolation + time-aligned text + MTGNN-style backbone.

Ported to the IMM-TSF benchmark interface. Unlike the standalone version in
the project root (models/gpinet_mm.py), this file:
  - is self-contained (no cross-package imports from outside IMM-TSF/), since
    main.py is always run with IMM-TSF/ as the working directory.
  - implements the generic `forecasting(time_steps_to_predict, X,
    truth_time_steps, mask)` contract shared by every model in this repo
    (see models/tPatchGNN.py, models/CRU.py), so it plugs into the same
    training loop and evaluation code as every other baseline.
  - optionally maps irregular reports to the historical GP grid with a fixed
    Gaussian time kernel.  The resulting regular text sequence is appended as
    one extra MTGNN node. Numerical variables retain their original TopK graph,
    while one learnable sigmoid edge per variable carries the text background
    outside TopK. Empty-text patients receive an exactly-zero text path.
  - queries the decoder at the batch's actual `time_steps_to_predict`
    (variable, padded, continuous) instead of a fixed internal grid, since
    IMM-TSF's standard collate does not guarantee a fixed prediction length
    across datasets/splits (unlike the synthetic-harness version).
  - drops the GP marginal-log-likelihood auxiliary loss term used in the
    standalone harness: IMM-TSF's compute_all_losses/evaluation only compute
    MSE on the model's forecasting() output and have no hook for auxiliary
    model-internal losses. The GP is used purely as a feature encoder here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────
# GP interpolation
# ─────────────────────────────────────────────────────────────────────────


class BatchedGPInterpolator(nn.Module):
    """Independent per-variable GP interpolation onto a fixed query grid."""

    def __init__(
        self,
        num_nodes,
        t_query,
        lengthscale_bounds=(0.01, 0.5),
        init_lengthscale=0.15,
        init_variance=1.0,
        init_noise=0.05,
        kernel="rbf",
        zero_noise=False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.register_buffer("t_query", t_query.float())
        self.bounds = lengthscale_bounds
        self.kernel = kernel
        self.zero_noise = bool(zero_noise)
        low, high = lengthscale_bounds
        raw = math.log((init_lengthscale - low) / (high - init_lengthscale))
        self.raw_lengthscale = nn.Parameter(torch.full((num_nodes,), raw))
        self.log_variance = nn.Parameter(torch.full((num_nodes,), math.log(init_variance)))
        if not self.zero_noise:
            self.log_noise = nn.Parameter(torch.full((num_nodes,), math.log(init_noise)))

    def get_hyperparams(self):
        low, high = self.bounds
        ls = low + (high - low) * torch.sigmoid(self.raw_lengthscale)
        variance = self.log_variance.exp()
        noise = torch.zeros_like(variance) if self.zero_noise else self.log_noise.exp()
        return ls, variance, noise

    def _kernel(self, x, y, ls, variance):
        d2 = (x.unsqueeze(-1) - y.unsqueeze(-2)).square()
        if self.kernel == "rbf":
            return variance * torch.exp(-0.5 * d2 / ls.square())
        r = d2.clamp_min(1e-12).sqrt()
        if self.kernel == "matern52":
            z = math.sqrt(5.0) * r / ls
            return variance * (1 + z + z.square() / 3) * torch.exp(-z)
        z = math.sqrt(3.0) * r / ls
        return variance * (1 + z) * torch.exp(-z)

    def forward(self, t_obs, y_obs, mask, jitter=1e-5):
        # t_obs, y_obs, mask: (B, N, M)
        b, n, m = t_obs.shape
        t = t_obs.reshape(b * n, m)
        y = y_obs.reshape(b * n, m)
        valid = mask.reshape(b * n, m).float()
        ls, variance, noise = self.get_hyperparams()
        ls = ls.view(1, n, 1, 1).expand(b, -1, -1, -1).reshape(b * n, 1, 1)
        variance = variance.view(1, n, 1, 1).expand(b, -1, -1, -1).reshape(b * n, 1, 1)
        noise = noise.view(1, n, 1).expand(b, -1, -1).reshape(b * n, 1)
        kernel = self._kernel(t, t, ls, variance)
        pair_mask = valid.unsqueeze(-1) * valid.unsqueeze(-2)
        eye = torch.eye(m, device=t.device).unsqueeze(0)
        kernel = pair_mask * kernel + (1 - pair_mask) * eye
        kernel = kernel + torch.diag_embed(valid * (noise + jitter))
        chol = torch.linalg.cholesky(kernel)
        alpha = torch.cholesky_solve((y * valid).unsqueeze(-1), chol)
        query = self.t_query.view(1, -1).expand(b * n, -1)
        cross = self._kernel(query, t, ls, variance) * valid.unsqueeze(1)
        mean = torch.bmm(cross, alpha).squeeze(-1)
        solved = torch.cholesky_solve(cross.transpose(-1, -2), chol)
        reduction = (cross * solved.transpose(-1, -2)).sum(dim=-1)
        variance_out = (variance.view(b * n, 1) - reduction).clamp_min(1e-8)
        std = variance_out.sqrt()
        observations = valid.sum(dim=-1)
        mll_raw = -0.5 * ((y * valid).unsqueeze(-1) * alpha).sum((1, 2))
        mll_raw = mll_raw - torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        mll_raw = mll_raw - 0.5 * observations * math.log(2 * math.pi)
        good = observations >= 2
        mll = (mll_raw[good] / observations[good]).mean() if good.any() else mean.new_zeros(())
        return mean.view(b, n, -1), std.view(b, n, -1), mll, good.sum()


class GaussHermiteFusionModule(nn.Module):
    """Fuse GP posterior mean/std into backbone input channels."""

    def __init__(self, d_model: int, k: int = 3):
        super().__init__()
        if k == 3:
            nodes = [-math.sqrt(1.5), 0.0, math.sqrt(1.5)]
            weights = [1 / 6, 4 / 6, 1 / 6]
        elif k == 5:
            nodes = [-2.0201829, -0.9585724, 0.0, 0.9585724, 2.0201829]
            weights = [0.0112574, 0.2220759, 0.5333333, 0.2220759, 0.0112574]
        else:
            raise ValueError("GPFusion k must be 3 or 5")
        self.register_buffer("nodes", torch.tensor(nodes, dtype=torch.float32))
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.mean_proj = nn.Conv2d(1, d_model, 1)
        self.std_proj = nn.Conv2d(1, d_model, 1)

    def forward(self, mean, std):
        fused = 0.0
        for node, weight in zip(self.nodes, self.weights):
            sample = mean + math.sqrt(2.0) * std * node
            fused = fused + weight * F.relu(self.mean_proj(sample))
        return fused + self.std_proj(std)


# ─────────────────────────────────────────────────────────────────────────
# Gaussian time alignment for the MTGNN text-background node
# ─────────────────────────────────────────────────────────────────────────


class GaussianTextBackground(nn.Module):
    """Map irregular reports to one regular MTGNN background node.

    For grid time ``g_q`` and report time ``tau_j``, the absolute (non-softmax)
    weight is ``exp(-(g_q - tau_j)^2 / (2 sigma^2))``.  The bounded aggregate

        sum_j w_qj W_e e_j / (1 + sum_j w_qj)

    preserves absolute temporal strength: a single distant report approaches
    zero instead of receiving softmax weight one.  This module performs no
    variable matching and no numerical/text cross-attention.  Its output is a
    single regular node; MTGNN performs all variable selection through its
    learned graph.
    """

    def __init__(
        self,
        d_txt,
        hidden,
        t_query,
        history_window,
        total_window,
        sigma_hours=3.0,
    ):
        super().__init__()
        if total_window <= 0 or history_window <= 0:
            raise ValueError("history and history + pred_window must be positive")
        if history_window > total_window:
            raise ValueError("history cannot exceed history + pred_window")
        if float(sigma_hours) <= 0:
            raise ValueError("Gaussian text sigma must be positive")

        self.d_txt = int(d_txt)
        self.hidden = int(hidden)
        self.history_window = float(history_window)
        self.total_window = float(total_window)
        self.sigma_hours = float(sigma_hours)
        self.register_buffer("t_query", t_query.detach().float().clone())
        # Bias-free: a text node must come from report content, not from a
        # learned time-shaped constant multiplied by the Gaussian mass.
        self.text_proj = nn.Linear(self.d_txt, self.hidden, bias=False)

        # Optional evaluation diagnostics populated on each forward pass.
        self.last_relevance = None
        self.last_membership = None
        self.last_text_temporal_variation = None
        self.last_multi_note_patient_fraction = None
        self.last_background_rms = None
        self.last_gaussian_weight_mean = None
        self.last_gaussian_weight_max = None
        self.last_grid_has_text = None
        self.last_grid_weight_mass = None
        self.last_note_count = None
        self.last_note_grid_index = None

    def _record_empty(self, numeric_features, batch_size, num_notes):
        q = int(self.t_query.numel())
        device = numeric_features.device
        self.last_relevance = None
        self.last_membership = None
        self.last_text_temporal_variation = None
        self.last_multi_note_patient_fraction = None
        self.last_background_rms = None
        self.last_gaussian_weight_mean = None
        self.last_gaussian_weight_max = None
        self.last_grid_has_text = torch.zeros(
            batch_size, q, dtype=torch.bool, device=device
        )
        self.last_grid_weight_mass = numeric_features.new_zeros((batch_size, q))
        self.last_note_count = numeric_features.new_zeros((batch_size,))
        self.last_note_grid_index = torch.full(
            (batch_size, num_notes), -1, dtype=torch.long, device=device
        )

    def forward(self, numeric_features, notes_input, tau_raw):
        # numeric_features: (B, H, C, Q), used only for shape/device/dtype.
        b, hidden, _, numeric_q = numeric_features.shape
        q = int(self.t_query.numel())
        if hidden != self.hidden:
            raise ValueError(f"Expected {self.hidden} hidden channels, got {hidden}")
        if numeric_q != q:
            raise ValueError(f"Expected {q} historical GP columns, got {numeric_q}")

        if notes_input is None or tau_raw is None:
            self._record_empty(numeric_features, b, 0)
            return numeric_features.new_zeros((b, self.hidden, 1, q))
        if notes_input.dim() != 3:
            raise ValueError(
                "GPINet text background expects pre-computed embeddings "
                "with shape (B, K, d_txt)"
            )
        if tau_raw.dim() != 2:
            raise ValueError("tau_raw must have shape (B, K)")

        if notes_input.shape[:2] != tau_raw.shape:
            raise ValueError(
                "notes_input and tau_raw must agree on batch and report dimensions"
            )
        if notes_input.shape[-1] != self.d_txt:
            raise ValueError(
                f"GPINet expected text embedding size {self.d_txt}, "
                f"got {notes_input.shape[-1]}; set --d_txt to match the "
                "pre-computed embedding file"
            )

        k = int(notes_input.shape[1])
        if k == 0:
            self._record_empty(numeric_features, b, 0)
            return numeric_features.new_zeros((b, self.hidden, 1, q))

        notes_input = notes_input.to(
            device=numeric_features.device, dtype=numeric_features.dtype
        )
        tau_raw = tau_raw.to(
            device=numeric_features.device, dtype=numeric_features.dtype
        )
        note_mask = torch.isfinite(notes_input).all(dim=-1)
        note_mask = note_mask & (notes_input.abs().sum(dim=-1) > 0)
        note_mask = (
            note_mask
            & torch.isfinite(tau_raw)
            & (tau_raw >= 0)
            & (tau_raw <= tau_raw.new_tensor(self.history_window) + 1e-7)
        )
        if not note_mask.any():
            self._record_empty(numeric_features, b, k)
            return numeric_features.new_zeros((b, self.hidden, 1, q))

        notes_value = notes_input.masked_fill(~note_mask.unsqueeze(-1), 0)
        tau_safe = tau_raw.masked_fill(~note_mask, 0)
        grid_hours = self.t_query.to(
            device=numeric_features.device, dtype=numeric_features.dtype
        ) * self.total_window
        time_delta = grid_hours.view(1, q, 1) - tau_safe.view(b, 1, k)
        gaussian_weight = torch.exp(
            -0.5 * (time_delta / self.sigma_hours).square()
        )
        gaussian_weight = gaussian_weight * note_mask.view(b, 1, k).to(
            gaussian_weight.dtype
        )

        projected_text = self.text_proj(notes_value)
        numerator = torch.einsum(
            "bqk,bkh->bqh", gaussian_weight, projected_text
        )
        weight_mass = gaussian_weight.sum(dim=-1)
        text_grid = numerator / (1.0 + weight_mass.unsqueeze(-1))
        patient_has_text = note_mask.any(dim=1)
        text_grid = text_grid * patient_has_text.view(b, 1, 1).to(
            text_grid.dtype
        )

        with torch.no_grad():
            note_count = note_mask.sum(dim=1)
            grid_has_text = weight_mass > 1e-3
            valid_weight = note_mask.view(b, 1, k).expand(b, q, k)
            centered_text = text_grid - text_grid.mean(dim=1, keepdim=True)
            text_temporal_variation = centered_text[
                patient_has_text
            ].square().mean().sqrt()
            multi_note_fraction = (note_count[patient_has_text] > 1).to(
                torch.float32
            ).mean()
            background_rms = text_grid[patient_has_text].square().mean().sqrt()
            gaussian_weight_mean = gaussian_weight[valid_weight].mean()
            gaussian_weight_max = gaussian_weight[valid_weight].max()

            self.last_relevance = None
            self.last_membership = None
            self.last_text_temporal_variation = text_temporal_variation.detach()
            self.last_multi_note_patient_fraction = multi_note_fraction.detach()
            self.last_background_rms = background_rms.detach()
            self.last_gaussian_weight_mean = gaussian_weight_mean.detach()
            self.last_gaussian_weight_max = gaussian_weight_max.detach()
            self.last_grid_has_text = grid_has_text.detach()
            self.last_grid_weight_mass = weight_mass.detach()
            self.last_note_count = note_count.detach()
            nearest_grid = time_delta.abs().argmin(dim=1)
            self.last_note_grid_index = nearest_grid.masked_fill(
                ~note_mask, -1
            ).detach()

        return text_grid.permute(0, 2, 1).unsqueeze(2)


# ─────────────────────────────────────────────────────────────────────
# MTGNN-style backbone (encoder mode: pools to a per-node hidden vector
# instead of directly emitting a fixed prediction horizon)
# ─────────────────────────────────────────────────────────────────────────


class GraphConstructor(nn.Module):
    def __init__(self, num_nodes, subgraph_size=20, node_dim=40, alpha=3.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.k = min(num_nodes, subgraph_size)
        self.alpha = alpha
        self.left = nn.Parameter(torch.randn(num_nodes, node_dim) * 0.1)
        self.right = nn.Parameter(torch.randn(num_nodes, node_dim) * 0.1)

    def forward(self):
        scores = torch.tanh(self.alpha * (self.left @ self.right.t()))
        values, indices = torch.topk(scores, self.k, dim=1)
        adjacency = torch.zeros_like(scores).scatter(1, indices, values)
        return F.softmax(F.relu(adjacency), dim=1)


class MixProp(nn.Module):
    def __init__(self, channels, depth=2, alpha=0.05, dropout=0.1):
        super().__init__()
        self.depth = depth
        self.alpha = alpha
        self.project = nn.Conv2d(channels * (depth + 1), channels, 1)
        self.dropout = nn.Dropout(dropout)

        self.last_text_message_ratio = None

    def forward(self, x, adjacency, text_edge):
        # x: (B, F, C+1, M); adjacency: (C, C); text_edge: (C,)
        num_nodes = adjacency.shape[0]
        if x.shape[2] != num_nodes + 1:
            raise ValueError(
                "MixProp expects C numerical nodes followed by one text node"
            )
        if text_edge.shape != (num_nodes,):
            raise ValueError(
                f"MixProp expected {num_nodes} text edges, got {tuple(text_edge.shape)}"
            )

        numeric_x = x[:, :, :num_nodes]
        text_x = x[:, :, num_nodes:]
        states = [x]
        numeric_h = numeric_x
        for _ in range(self.depth):
            numeric_message = torch.einsum(
                "ij,bcjt->bcit", adjacency, numeric_h
            )
            text_message = (
                text_edge.view(1, 1, num_nodes, 1) * text_x
            )
            numeric_h = self.alpha * numeric_x + (1 - self.alpha) * (
                numeric_message + text_message
            )
            states.append(torch.cat([numeric_h, text_x], dim=2))

        numeric_strength = numeric_message.detach().abs().mean(dim=(0, 1, 3))
        text_strength = text_message.detach().abs().mean(dim=(0, 1, 3))
        self.last_text_message_ratio = text_strength / (
            numeric_strength + 1e-8
        )
        return self.dropout(self.project(torch.cat(states, dim=1)))


class MTGNNEncoder(nn.Module):
    """Compact MTGNN-style graph constructor + mix-hop + temporal blocks,
    pooled down to a per-node hidden vector (analogous to tPatchGNN's
    IMTS_Model + Linear temporal_agg)."""

    def __init__(
        self,
        num_nodes,
        in_channels,
        seq_length,
        hid_dim,
        hidden=64,
        layers=3,
        graph_depth=2,
        subgraph_size=20,
        node_dim=40,
        dropout=0.3,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.num_numeric_nodes = num_nodes
        self.start = nn.Conv2d(in_channels, hidden, 1)
        self.graph = GraphConstructor(num_nodes, subgraph_size, node_dim=node_dim)
        # Text is a background source, not a competitor in the numerical TopK
        # graph. A separate sigmoid edge gives every variable a differentiable
        # route to the text node while allowing training to suppress it.
        initial_text_edge = 0.1
        initial_text_logit = math.log(
            initial_text_edge / (1.0 - initial_text_edge)
        )
        self.text_edge_logits = nn.Parameter(
            torch.full((num_nodes,), initial_text_logit)
        )
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "temporal": nn.Conv2d(hidden, hidden, (1, 3), padding=(0, 1)),
                        "graph": MixProp(hidden, graph_depth, dropout=dropout),
                        "norm": nn.BatchNorm2d(hidden),
                    }
                )
            )
        self.dropout = nn.Dropout(dropout)
        self.temporal_agg = nn.Linear(hidden * seq_length, hid_dim)
        self.last_text_edge = None
        self.last_text_message_ratio = None

    def forward(self, x):
        # x: (B, C_in, C+1, T), with the text node last.
        if x.shape[-1] != self.seq_length:
            raise ValueError(f"MTGNNEncoder expected sequence length {self.seq_length}, got {x.shape[-1]}")
        expected_nodes = self.num_numeric_nodes + 1
        if x.shape[2] != expected_nodes:
            raise ValueError(
                f"MTGNNEncoder expected {expected_nodes} nodes, got {x.shape[2]}"
            )
        adjacency = self.graph()
        text_edge = torch.sigmoid(self.text_edge_logits)
        # Convolution biases must not turn an absent text input into a phantom
        # message. Keep the text state exactly zero for patients without notes.
        text_present = (
            x[:, :, self.num_numeric_nodes :, :]
            .abs()
            .sum(dim=(1, 2, 3), keepdim=True)
            .gt(0)
            .to(x.dtype)
        )
        h = self.start(x)
        h = torch.cat(
            [
                h[:, :, : self.num_numeric_nodes],
                h[:, :, self.num_numeric_nodes :] * text_present,
            ],
            dim=2,
        )
        message_ratios = []
        for block in self.blocks:
            residual = h
            h = torch.tanh(block["temporal"](h))
            h = torch.cat(
                [
                    h[:, :, : self.num_numeric_nodes],
                    h[:, :, self.num_numeric_nodes :] * text_present,
                ],
                dim=2,
            )
            h = block["graph"](h, adjacency, text_edge)
            message_ratios.append(block["graph"].last_text_message_ratio)
            h = block["norm"](h + residual)
            h = torch.cat(
                [
                    h[:, :, : self.num_numeric_nodes],
                    h[:, :, self.num_numeric_nodes :] * text_present,
                ],
                dim=2,
            )
        self.last_text_edge = text_edge.detach()
        self.last_text_message_ratio = torch.stack(message_ratios).mean(dim=0)
        h = self.dropout(F.relu(h))  # (B, hidden, N, T)
        b, hd, n, t = h.shape
        h = h.permute(0, 2, 1, 3).reshape(b, n, hd * t)
        return self.temporal_agg(h)  # (B, N, hid_dim)


class LearnableTE(nn.Module):
    """Same time-embedding scheme as tPatchGNN's LearnableTE."""

    def __init__(self, te_dim):
        super().__init__()
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, te_dim - 1)

    def forward(self, tt):
        # tt: (..., 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], dim=-1)


# ─────────────────────────────────────────────────────────────────────────
# Top-level model
# ─────────────────────────────────────────────────────────────────────────


class GPINet(nn.Module):
    """GP -> GH numerical nodes + Gaussian text node -> MTGNN -> decoder."""

    def __init__(self, args, supports=None, dropout=0):
        super().__init__()
        self.device = args.device
        self.N = args.C  # number of variables (== num_nodes)
        self.hid_dim = args.hid_dim
        self.te_dim = args.te_dim

        history_frac = float(args.history) / float(args.history + args.pred_window)
        # Restore the original GPINet convention: Q defaults to the prediction
        # horizon (24 for expanded MIMIC), with query points covering [0, H)
        # rather than including the H=24 boundary.
        n_query = int(getattr(args, "gpinet_query_points", 0)) or int(
            args.pred_window
        )
        if n_query < 2:
            raise ValueError("GPINet requires at least two historical query points")
        t_query = torch.linspace(0.0, history_frac, n_query + 1)[:n_query]

        self.gp = BatchedGPInterpolator(
            self.N,
            t_query,
            lengthscale_bounds=tuple(getattr(args, "gpinet_lengthscale_bounds", (0.01, 0.5))),
            init_lengthscale=float(getattr(args, "gpinet_init_lengthscale", 0.15)),
            init_variance=float(getattr(args, "gpinet_init_variance", 1.0)),
            init_noise=float(getattr(args, "gpinet_init_noise", 0.05)),
            kernel=getattr(args, "gpinet_kernel", "rbf"),
            zero_noise=bool(getattr(args, "gpinet_zero_noise", False)),
        )
        self.fusion = GaussHermiteFusionModule(
            self.hid_dim, k=int(getattr(args, "gpinet_gh_k", 3))
        )
        self.backbone = MTGNNEncoder(
            self.N,  # C numerical variables; text uses a separate edge
            in_channels=self.hid_dim,
            seq_length=n_query + 1,  # +1 for the zero-padding column
            hid_dim=self.hid_dim,
            hidden=self.hid_dim,
            layers=int(getattr(args, "nlayer", 1)),
            graph_depth=int(getattr(args, "hop", 1)),
            subgraph_size=int(getattr(args, "gpinet_subgraph_size", 20)),
            node_dim=int(getattr(args, "node_dim", 10)),
            dropout=float(args.dropout),
        )
        self.te = LearnableTE(self.te_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.hid_dim + self.te_dim, self.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hid_dim, 1),
        )

        # Build text-only parameters after every numeric parameter. fork_rng
        # ensures enabling text cannot perturb numeric initialization under the
        # same experiment seed, keeping uni-vs-multi comparisons fair.
        self.native_text_enabled = bool(
            getattr(args, "enable_text", False)
            and getattr(args, "use_text_embeddings", False)
        )
        self.text_grid_fusion = None
        if self.native_text_enabled:
            # All modules are still constructed on CPU here. Seed only the
            # CPU generator so unrelated CUDA streams are untouched.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(
                    int(getattr(args, "seed", 0)) + 104729
                )
                self.text_grid_fusion = GaussianTextBackground(
                    d_txt=int(args.d_txt),
                    hidden=self.hid_dim,
                    t_query=t_query,
                    history_window=float(args.history),
                    total_window=float(args.history + args.pred_window),
                    sigma_hours=float(
                        getattr(args, "gpinet_text_time_sigma_hours", 3.0)
                    ),
                )
        # populated on each forecasting() call for optional external logging
        self.last_mll = None
        self.last_valid_pairs = None

    def get_hyperparams(self):
        return self.gp.get_hyperparams()

    def forecasting(
        self,
        time_steps_to_predict,
        X,
        truth_time_steps,
        mask=None,
        notes_input=None,
        tau=None,
    ):
        """
        time_steps_to_predict: (B, Lp)        normalized query times in [0,1]
        X (observed_data):     (B, T_obs, N)
        truth_time_steps:      (B, T_obs)     shared time axis across variables
        mask (observed_mask):  (B, T_obs, N)
        notes_input:           (B, K, d_txt)  padded report embeddings
        tau:                   (B, K)         raw report times (dataset units)
        returns:                (B, Lp, N)
        """
        B, T_obs, N = X.shape
        if mask is None:
            mask = torch.ones_like(X)

        # Broadcast the shared time axis per-variable for the GP.
        t_obs = truth_time_steps.unsqueeze(1).expand(B, N, T_obs)
        y_obs = X.permute(0, 2, 1)  # (B, N, T_obs)
        m_obs = mask.permute(0, 2, 1)  # (B, N, T_obs)

        mean, std, mll, valid_pairs = self.gp(t_obs, y_obs, m_obs)  # (B, N, Q)
        self.last_mll = mll
        self.last_valid_pairs = valid_pairs

        # GH produces C regular numerical nodes. Text remains an independent
        # (C+1)-th node so MTGNN, rather than a pre-backbone cross-attention,
        # learns whether and how it propagates to numerical variables.
        numeric_nodes = self.fusion(
            mean.unsqueeze(1), std.unsqueeze(1)
        )  # (B, hid_dim, N, Q)

        text_node = numeric_nodes.new_zeros(
            (B, self.hid_dim, 1, numeric_nodes.shape[-1])
        )
        if notes_input is not None or tau is not None:
            if self.text_grid_fusion is None:
                raise RuntimeError(
                    "GPINet received native text inputs, but its text background "
                    "was not enabled with --enable_text --use_text_embeddings."
                )
            if notes_input is None or tau is None:
                raise ValueError("notes_input and tau must be provided together")
            text_node = self.text_grid_fusion(numeric_nodes, notes_input, tau)

        graph_nodes = torch.cat([numeric_nodes, text_node], dim=2)
        graph_nodes = F.pad(
            graph_nodes, (1, 0, 0, 0)
        )  # (B, hid_dim, N+1, Q+1)
        all_node_states = self.backbone(
            graph_nodes
        )  # (B, N+1, hid_dim)
        h = all_node_states[:, :N]  # text node is context, never a target

        Lp = time_steps_to_predict.shape[1]
        h_exp = h.unsqueeze(2).expand(B, N, Lp, self.hid_dim)
        t_pred = time_steps_to_predict.to(X.dtype).view(B, 1, Lp, 1).expand(B, N, Lp, 1)
        te_pred = self.te(t_pred)  # (B, N, Lp, te_dim)

        dec_in = torch.cat([h_exp, te_pred], dim=-1)
        out = self.decoder(dec_in).squeeze(-1)  # (B, N, Lp)
        return out.permute(0, 2, 1)  # (B, Lp, N)
