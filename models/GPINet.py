"""GPINet: GP interpolation + Gauss-Hermite fusion + MTGNN-style backbone.

Ported to the IMM-TSF benchmark interface. Unlike the standalone version in
the project root (models/gpinet_mm.py), this file:
  - is self-contained (no cross-package imports from outside IMM-TSF/), since
    main.py is always run with IMM-TSF/ as the working directory.
  - implements the generic `forecasting(time_steps_to_predict, X,
    truth_time_steps, mask)` contract shared by every model in this repo
    (see models/tPatchGNN.py, models/CRU.py), so it plugs into the same
    training loop, evaluation code, and text FusionModel as every other
    baseline.
  - keeps the original numeric-only path when text is disabled. When text is
    enabled with precomputed embeddings, GPINet uses its native feature-level
    path and does not invoke the repository's generic post-hoc FusionModel.
  - optionally supports native, feature-level text fusion. Each report stays
    an independent event token carrying its own timestamp. MTGNN variable
    nodes query this event set *inside every temporal/graph block*; reports
    are never resampled or spread onto the 24-point GP grid. The attended
    text message is written into the node hidden state before graph
    propagation, so the numerical backbone itself learns the multimodal
    representation rather than receiving a post-hoc output correction.
    lib/evaluation.py routes precomputed text embeddings through this path
    specifically for GPINet when text is enabled.
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
# Native text fusion. Reports remain a variable-length event set; no text
# pseudo-sequence is created on the GP query grid.
# ─────────────────────────────────────────────────────────────────────────


class _GPTime2Vec(nn.Module):
    """Compact Time2Vec used to retain each report's original event time."""

    def __init__(self, d_tau: int):
        super().__init__()
        assert d_tau > 1, "d_tau must be > 1"
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, d_tau - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lin = self.linear(x)
        per = torch.sin(self.periodic(x))
        return torch.cat([lin, per], dim=-1)


class TextEventEncoder(nn.Module):
    """Encode K independent report events without inventing a time grid.

    The loader supplies zero-padded report embeddings and raw timestamps in
    hours. Padding is inferred from all-zero embeddings. A safe dummy token is
    created only for an all-empty sample/batch so MultiheadAttention never
    receives an all-masked key row; `has_text` later makes that path an exact
    no-op.
    """

    def __init__(
        self,
        d_txt_in: int,
        hid_dim: int,
        history_window: float,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.history_window = float(history_window)
        self.proj_in = nn.Linear(d_txt_in, hid_dim)
        self.d_tau = max(hid_dim // 2, 2)
        self.time2vec = _GPTime2Vec(self.d_tau)
        self.kv_proj = nn.Linear(hid_dim + self.d_tau, hid_dim)
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, notes_input: torch.Tensor, tau: torch.Tensor):
        """
        notes_input:     (B, K_max, d_txt_in), zero-padded report embeddings
        tau:             (B, K_max), raw report timestamps in history units
        returns:
          tokens:          (B, max(K_max, 1), hid_dim)
          padding_mask:    (B, max(K_max, 1)); True means ignore
          has_text:        (B,) bool
        """
        if notes_input.ndim != 3:
            raise ValueError(
                "notes_input must have shape (B, K_max, d_txt_in), got "
                f"{tuple(notes_input.shape)}"
            )
        if tau.ndim != 2 or tau.shape[:2] != notes_input.shape[:2]:
            raise ValueError(
                "tau must have shape (B, K_max) matching notes_input; got "
                f"notes={tuple(notes_input.shape)}, tau={tuple(tau.shape)}"
            )

        batch_size, n_notes, _ = notes_input.shape
        if n_notes == 0:
            tokens = notes_input.new_zeros((batch_size, 1, self.proj_in.out_features))
            padding_mask = torch.zeros(
                (batch_size, 1), dtype=torch.bool, device=notes_input.device
            )
            has_text = torch.zeros(
                batch_size, dtype=torch.bool, device=notes_input.device
            )
            return tokens, padding_mask, has_text

        note_mask = notes_input.abs().sum(dim=-1) > 0
        has_text = note_mask.any(dim=1)
        tau_norm = (tau / self.history_window).clamp(0.0, 1.0)

        content = self.proj_in(notes_input)
        tau_feat = self.time2vec(tau_norm.unsqueeze(-1))
        tokens = self.kv_proj(torch.cat([content, tau_feat], dim=-1))
        tokens = self.dropout(self.layer_norm(tokens))
        tokens = tokens * note_mask.unsqueeze(-1).to(tokens.dtype)

        padding_mask = ~note_mask
        empty_rows = ~has_text
        if empty_rows.any():
            # Keep a single zero dummy token visible to attention. The
            # injection mask below guarantees it cannot affect the numeric path.
            padding_mask = padding_mask.clone()
            padding_mask[empty_rows, 0] = False
            tokens = tokens.clone()
            tokens[empty_rows, 0] = 0.0

        return tokens, padding_mask, has_text


class VariableTextInjection(nn.Module):
    """Let numerical MTGNN variable nodes read the K report-event tokens.

    Queries are produced from each variable's numerical hidden trajectory,
    plus a learned variable identity. The resulting variable-specific text
    message is injected as static patient context before graph propagation;
    it is not interpreted as a report observed at each hourly grid point.
    """

    def __init__(
        self,
        num_nodes: int,
        hidden: int,
        n_heads: int = 1,
        dropout: float = 0.1,
        gate_bias: float = -1.0,
    ):
        super().__init__()
        if hidden % n_heads != 0:
            raise ValueError(
                f"hidden={hidden} must be divisible by n_heads={n_heads}"
            )
        self.variable_embed = nn.Parameter(torch.randn(num_nodes, hidden) * 0.02)
        self.query_norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(hidden)
        self.delta = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.gate = nn.Linear(2 * hidden, hidden)
        nn.init.constant_(self.gate.bias, gate_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, text_tokens, text_padding_mask, has_text):
        """
        h:                 (B, hidden, N, T)
        text_tokens:       (B, K, hidden)
        text_padding_mask: (B, K), True means ignore
        has_text:          (B,) bool
        """
        batch_size, hidden, num_nodes, _ = h.shape
        variable_state = h.mean(dim=-1).permute(0, 2, 1)
        query = self.query_norm(
            variable_state + self.variable_embed.unsqueeze(0)
        )

        context, attn_weights = self.attn(
            query,
            text_tokens,
            text_tokens,
            key_padding_mask=text_padding_mask,
            need_weights=True,
        )
        has_text_bnc = has_text.view(batch_size, 1, 1)
        context = torch.where(
            has_text_bnc, self.context_norm(context), torch.zeros_like(context)
        )

        gate = torch.sigmoid(self.gate(torch.cat([query, context], dim=-1)))
        update = gate * self.delta(context)
        update = update * has_text_bnc.to(update.dtype)
        update = self.dropout(update).permute(0, 2, 1).unsqueeze(-1)

        return h + update, attn_weights, gate


# ─────────────────────────────────────────────────────────────────────────
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

    def forward(self, x, adjacency):
        # x (B, F, N, M) ; adjacency (N, N)
        states = [x]
        h = x
        for _ in range(self.depth):
            h = self.alpha * x + (1 - self.alpha) * torch.einsum("ij,bcjt->bcit", adjacency, h)
            states.append(h)
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
        self.num_nodes = int(num_nodes)
        self.hidden = int(hidden)
        self.n_layers = int(layers)
        self.seq_length = seq_length
        self.start = nn.Conv2d(in_channels, hidden, 1)
        self.graph = GraphConstructor(num_nodes, subgraph_size, node_dim=node_dim)
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

        # Configured by GPINet only after every numeric-path module has been
        # initialized. Keeping text construction separate preserves the
        # numerical parameter initialization used by Uni runs.
        self.text_event_encoder = None
        self.text_injections = nn.ModuleList()
        self.last_text_gate_mean = None
        self.last_text_attention_entropy = None

    def configure_text_fusion(
        self,
        d_txt_in,
        history_window,
        n_heads=1,
        dropout=0.1,
        gate_bias=-1.0,
    ):
        """Attach feature-level text modules to every MTGNN block."""
        self.text_event_encoder = TextEventEncoder(
            d_txt_in=int(d_txt_in),
            hid_dim=self.hidden,
            history_window=float(history_window),
            dropout=float(dropout),
        )
        self.text_injections = nn.ModuleList(
            [
                VariableTextInjection(
                    num_nodes=self.num_nodes,
                    hidden=self.hidden,
                    n_heads=int(n_heads),
                    dropout=float(dropout),
                    gate_bias=float(gate_bias),
                )
                for _ in range(self.n_layers)
            ]
        )

    def forward(self, x, notes_input=None, tau=None):
        # x: (B, C_in, N, T)
        if x.shape[-1] != self.seq_length:
            raise ValueError(f"MTGNNEncoder expected sequence length {self.seq_length}, got {x.shape[-1]}")

        text_tokens = text_padding_mask = has_text = None
        if notes_input is not None:
            if tau is None:
                raise ValueError("tau is required when notes_input is provided")
            if self.text_event_encoder is None or len(self.text_injections) == 0:
                raise RuntimeError(
                    "MTGNN native text fusion was not configured. Instantiate "
                    "GPINet with enable_text=True and use_text_embeddings=True."
                )
            text_tokens, text_padding_mask, has_text = self.text_event_encoder(
                notes_input, tau
            )

        adjacency = self.graph()
        h = self.start(x)
        gate_means = []
        attention_entropies = []
        for layer_idx, block in enumerate(self.blocks):
            residual = h
            h = torch.tanh(block["temporal"](h))

            # Text enters the MTGNN representation between temporal feature
            # extraction and graph propagation. K report events stay K events;
            # the returned message is variable-specific, not a 24-point text
            # pseudo-series.
            if text_tokens is not None:
                h, attn_weights, gate = self.text_injections[layer_idx](
                    h, text_tokens, text_padding_mask, has_text
                )
                valid_rows = has_text.view(-1, 1, 1)
                if valid_rows.any():
                    gate_means.append(gate[valid_rows.expand_as(gate)].mean())
                    probs = attn_weights.clamp_min(1e-8)
                    entropy = -(probs * probs.log()).sum(dim=-1)
                    attention_entropies.append(entropy[has_text].mean())

            h = block["graph"](h, adjacency)
            h = block["norm"](h + residual)

        self.last_text_gate_mean = (
            torch.stack(gate_means).mean().detach() if gate_means else None
        )
        self.last_text_attention_entropy = (
            torch.stack(attention_entropies).mean().detach()
            if attention_entropies
            else None
        )
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
    """GP interpolation -> MTGNN internal text fusion -> query decoder.

    `notes_input=None` selects the unchanged numerical path. Supplying report
    embeddings and their raw timestamps keeps the reports as K independent
    events and lets MTGNN variable nodes attend to them inside its blocks.
    """

    def __init__(self, args, supports=None, dropout=0):
        super().__init__()
        self.device = args.device
        self.N = args.C  # number of variables (== num_nodes)
        self.hid_dim = args.hid_dim
        self.te_dim = args.te_dim

        history_frac = float(args.history) / float(args.history + args.pred_window)
        # Bind the number of GP query points to pred_window (== out_dim),
        # matching the original gpinet_mm.py/model.py convention
        # (`t_query = linspace(0, history_frac, out_dim + 1)[:out_dim]`).
        # `gpinet_query_points` can still override this explicitly if set.
        n_query = int(getattr(args, "gpinet_query_points", 0)) or int(args.pred_window)
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
            self.N,
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
        # populated on each forecasting() call for optional external logging
        self.last_mll = None
        self.last_valid_pairs = None

        # Configure native text modules only for the multimodal/precomputed
        # route. Construction happens after every numerical module and inside
        # a forked RNG context: numeric weights and the caller's RNG stream are
        # therefore unchanged relative to a Uni model initialized with the
        # same seed.
        self.native_text_enabled = bool(
            getattr(args, "enable_text", False)
            and getattr(args, "use_text_embeddings", False)
        )
        if self.native_text_enabled:
            text_seed = int(getattr(args, "seed", 0)) + 104729
            with torch.random.fork_rng(devices=[], enabled=True):
                torch.manual_seed(text_seed)
                self.backbone.configure_text_fusion(
                    d_txt_in=int(getattr(args, "d_txt", 768)),
                    history_window=float(args.history),
                    n_heads=int(getattr(args, "n_heads_fusion", 1)),
                    dropout=float(args.dropout),
                    gate_bias=float(
                        getattr(args, "gpinet_text_gate_bias", -1.0)
                    ),
                )

        self.last_text_gate_mean = None
        self.last_text_attention_entropy = None

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
        notes_input: optional (B, K_max, d_txt) precomputed report embeddings.
            Reports remain K independent events inside the MTGNN backbone.
        tau: optional (B, N_max) RAW (unnormalized) note timestamps,
            required if notes_input is provided.
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

        gp_input = torch.stack([mean, std], dim=1)  # (B, 2, N, Q)
        gp_input = F.pad(gp_input, (1, 0, 0, 0))  # (B, 2, N, Q+1)
        fused = self.fusion(gp_input[:, 0:1], gp_input[:, 1:2])  # (B, hid_dim, N, Q+1)

        if notes_input is not None and not self.native_text_enabled:
            raise RuntimeError(
                "notes_input was supplied but native text fusion is disabled. "
                "Instantiate GPINet with enable_text=True and "
                "use_text_embeddings=True."
            )

        h = self.backbone(
            fused,
            notes_input=notes_input,
            tau=tau,
        )  # (B, N, hid_dim)
        self.last_text_gate_mean = self.backbone.last_text_gate_mean
        self.last_text_attention_entropy = (
            self.backbone.last_text_attention_entropy
        )

        Lp = time_steps_to_predict.shape[1]
        h_exp = h.unsqueeze(2).expand(B, N, Lp, self.hid_dim)
        t_pred = time_steps_to_predict.to(X.dtype).view(B, 1, Lp, 1).expand(B, N, Lp, 1)
        te_pred = self.te(t_pred)  # (B, N, Lp, te_dim)

        dec_in = torch.cat([h_exp, te_pred], dim=-1)
        out = self.decoder(dec_in).squeeze(-1)  # (B, N, Lp)
        return out.permute(0, 2, 1)  # (B, Lp, N)
