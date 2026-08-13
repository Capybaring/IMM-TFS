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
  - by default has NO built-in text fusion, same as every other model here:
    multimodality is handled uniformly by fusions/FusionModel.py, applied on
    top of this model's numeric-only output (Y_ts) in main.py's training
    loop. This is what makes the GPINet-vs-tPatchGNN Uni/generic-Multi
    comparison isolate the backbone as the only variable.
  - ALSO optionally supports a second, native text-fusion path (added
    2026-08-09): forecasting() accepts optional `notes_input`/`tau` kwargs
    (default None, so every existing call site and every other model is
    unaffected). When provided, a GPTextPooler (see below) pools the note
    embeddings onto the GP's own history-side query grid `t_query`, and
    GPTextGatedMerge merges them into `fused` via a learned per-position
    sigmoid gate (updated 2026-08-09 from an earlier plain concat+1x1-conv
    version, which had no way to suppress unhelpful text and empirically
    made Multi worse than Uni -- see fusions/MMF_GR_Add.py for the
    analogous idea in the generic post-hoc path) *before* self.backbone(),
    instead of only ever correcting an already-finished Y_ts post-hoc like
    the generic FusionModel does. lib/evaluation.py routes text through this
    path instead of the generic FusionModel specifically for GPINet
    (isinstance check) when enabled; every other model is untouched. This
    means "GPINet + native fusion" and "GPINet + generic FusionModel" are no
    longer directly comparable to other backbones on equal footing the way
    Uni numbers are -- keep both GPINet rows (generic and native) in any
    results table so the delta attributable to the fusion design itself is
    visible, rather than conflating it with "GPINet's backbone is better."
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
# Native text fusion (optional; see module docstring). Deliberately
# reimplemented here rather than importing fusions/TTF_RecAvg.py, to (a)
# keep this file's no-cross-package-import design intact, and (b) pool text
# onto the GP's *history-side* query grid t_query instead of the
# *prediction-side* query times t_hat that fusions/TTF_*.py use -- those are
# different axes (t_query in [0, history_frac], t_hat in
# [history_frac, 1]) serving different purposes (encoder-side context here,
# vs. post-hoc decoder-side correction there).
# ─────────────────────────────────────────────────────────────────────────


class _GPTime2Vec(nn.Module):
    """Same idea as fusions/TTF_T2V_XAttn.py's Time2Vec (linear trend +
    periodic terms), reimplemented here in ~5 lines to keep this file
    self-contained rather than importing it."""

    def __init__(self, d_tau: int):
        super().__init__()
        assert d_tau > 1, "d_tau must be > 1"
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, d_tau - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lin = self.linear(x)
        per = torch.sin(self.periodic(x))
        return torch.cat([lin, per], dim=-1)


class GPTextPooler(nn.Module):
    """Cross-attention pooling of note embeddings onto the GP's t_query grid
    (updated 2026-08-09 from an earlier Gaussian-proximity-weighted-average
    version -- kept the class name so call sites in GPINet.__init__/
    forecasting() don't need to change).

    Same spirit as fusions/TTF_T2V_XAttn.py (Time2Vec-augmented multi-head
    cross-attention over note embeddings), with two deliberate differences
    motivated by t_query being a FIXED, batch-independent grid (unlike
    TTF_T2V_XAttn's t_hat, which is a different, data-dependent set of
    future prediction times for every chunk):

      1. The attention query is a per-grid-position LEARNED embedding table
         (`self.query_embed`, shape (Q, hid_dim)) instead of one constant
         vector broadcast to every position. TTF_T2V_XAttn's query
         (`self.Q_param`) is identical for every one of its T_f future
         steps regardless of t_hat -- meaning its attention weights over
         notes can't actually vary by *which* future time is being queried,
         only by the notes' own timestamps. That's a real limitation there
         (see prior discussion), but here t_query is always the same Q
         values for every sample/batch (it doesn't depend on data at all --
         it's `linspace(0, history_frac, n_query)`, fixed at construction),
         so a learned embedding table indexed by grid position is a strictly
         more expressive, still-simple fix: each of the Q query points gets
         its own distinct, trainable query vector.
      2. `tau` is normalized (divided by total_window) before Time2Vec, for
         the same reason as the previous pooling version -- t_query lives in
         [0, history_frac], tau (as delivered in batch_dict) does not,
         unless rescaled. (fusions/TTF_RecAvg.py and TTF_T2V_XAttn.py do NOT
         do this normalization for tau vs. t_hat -- known, currently unfixed
         issue, see project notes; left alone here on purpose so this
         module's correctness doesn't depend on that separate bug.)
    """

    def __init__(
        self,
        d_txt_in: int,
        hid_dim: int,
        total_window: float,
        n_query: int,
        n_heads: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.total_window = float(total_window)
        self.proj_in = nn.Linear(d_txt_in, hid_dim)
        self.d_tau = max(hid_dim // 2, 2)
        self.time2vec = _GPTime2Vec(self.d_tau)
        self.kv_proj = nn.Linear(hid_dim + self.d_tau, hid_dim)
        self.query_embed = nn.Parameter(torch.randn(n_query, hid_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=hid_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(hid_dim, hid_dim)

    def forward(
        self, notes_input: torch.Tensor, tau: torch.Tensor, t_query: torch.Tensor
    ):
        """
        notes_input: (B, N_max, d_txt_in) precomputed note embeddings, zero-padded
        tau:         (B, N_max) RAW (unnormalized) note timestamps, zero-padded
        t_query:     (Q,) the GP's fixed, normalized [0, history_frac] query
            grid -- NOT used numerically here (query_embed already encodes
            grid position by construction), kept as an argument only for
            interface symmetry with the earlier pooling version / in case
            the grid ever becomes data-dependent later.
        returns:
          pooled:   (B, hid_dim, Q) text summary at each t_query point
          has_text: (B,) bool, whether this sample has >= 1 real note
        """
        B, N_max, _ = notes_input.shape
        Q = self.query_embed.shape[0]

        note_mask = notes_input.abs().sum(dim=-1) > 0  # (B, N_max)
        has_text = note_mask.any(dim=1)  # (B,)

        tau_norm = tau / self.total_window  # match t_query's [0,1] scale

        V = self.proj_in(notes_input)  # (B, N_max, hid_dim)
        tau_feat = self.time2vec(tau_norm.unsqueeze(-1))  # (B, N_max, d_tau)
        KV = self.kv_proj(torch.cat([V, tau_feat], dim=-1))  # (B, N_max, hid_dim)

        Qmat = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (B, Q, hid_dim)
        key_padding_mask = ~note_mask  # (B, N_max); True = ignore this key

        attn_out, _ = self.attn(Qmat, KV, KV, key_padding_mask=key_padding_mask)
        # Samples with zero real notes have an all-True key_padding_mask row,
        # which MultiheadAttention can turn into NaN (softmax over nothing);
        # zero those rows out explicitly, same pattern fusions/TTF_T2V_XAttn.py
        # uses for the same edge case.
        has_text_bc = has_text.view(B, 1, 1)
        attn_out = torch.where(has_text_bc, attn_out, torch.zeros_like(attn_out))

        pooled = self.layer_norm(attn_out + Qmat)  # residual, like TTF_T2V_XAttn
        pooled = self.dropout(pooled)
        pooled = self.proj_out(pooled)  # (B, Q, hid_dim)

        return pooled.permute(0, 2, 1), has_text  # (B, hid_dim, Q), (B,)


class GPTextGatedMerge(nn.Module):
    """Gated residual merge of text-pooled features into `fused`.

    Analogous in spirit to fusions/MMF_GR_Add.py's GRU-gated residual (which
    the paper's own ablation found more robust than a fixed-weight blend
    like MMF_XAttn_Add -- it can learn to shut text off where it isn't
    helpful instead of always contributing a fixed-size correction). Uses a
    1x1 Conv2d instead of a GRU since `fused`/text_pooled live on GPINet's
    (N, Q+1) grid, not a 1D time axis.

    g (per-channel/node/grid-position, in [0,1]) close to 1 -> ignore text,
    pass `fused` through unchanged. g close to 0 -> fully apply the
    text-informed residual. Forced to g=1 (exact no-op) for samples with no
    notes in this chunk, so a missing/empty note list can never perturb the
    numeric branch even slightly.
    """

    def __init__(self, hid_dim: int):
        super().__init__()
        self.residual_conv = nn.Conv2d(2 * hid_dim, hid_dim, 1)
        self.gate_conv = nn.Conv2d(2 * hid_dim, hid_dim, 1)

    def forward(self, fused, text_pooled, has_text_mask):
        """
        fused, text_pooled: (B, hid_dim, N, Q+1)
        has_text_mask:      (B, 1, 1, 1) float/bool, broadcastable
        """
        x = torch.cat([fused, text_pooled], dim=1)  # (B, 2*hid_dim, N, Q+1)
        delta = self.residual_conv(x)
        g = torch.sigmoid(self.gate_conv(x))
        g = torch.where(has_text_mask.bool(), g, torch.ones_like(g))
        return g * fused + (1 - g) * (fused + delta)


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

    def forward(self, x):
        # x: (B, C_in, N, T)
        if x.shape[-1] != self.seq_length:
            raise ValueError(f"MTGNNEncoder expected sequence length {self.seq_length}, got {x.shape[-1]}")
        adjacency = self.graph()
        h = self.start(x)
        for block in self.blocks:
            residual = h
            h = torch.tanh(block["temporal"](h))
            h = block["graph"](h, adjacency)
            h = block["norm"](h + residual)
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
    """GP interpolation -> Gauss-Hermite fusion -> [optional native text
    merge] -> MTGNN encoder -> per-query decoder.

    By default (forecasting() called without notes_input/tau) this is
    numeric-only and behaves exactly as before; text fusion happens
    externally via fusions/FusionModel.py, same as every other model in
    this repo. If forecasting() is called WITH notes_input/tau (currently
    only done by lib/evaluation.py for GPINet specifically), text is pooled
    onto the GP's history-side query grid and merged into `fused` before
    self.backbone() instead -- see module docstring and GPTextPooler.
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

        # --- Optional native text fusion (see class/module docstring) ---
        # Constructed LAST, after every numeric-path submodule above, so
        # that Uni-modal (notes_input=None) runs draw the exact same RNG
        # sequence for gp/fusion/backbone/te/decoder init as before this
        # feature existed, so those layers' INITIAL WEIGHTS are unchanged.
        # NOTE (corrected 2026-08-09): this does NOT make Uni-modal results
        # bit-for-bit reproducible overall, as originally assumed -- set_seed()
        # is only called once at process start, and constructing these extra
        # parameters (even last, even unused for Uni) still consumes RNG
        # draws, shifting the global RNG cursor for everything that runs
        # afterward (dataloader shuffling, every dropout call during
        # training). Confirmed empirically: Uni mse moved from 0.7676 to
        # 0.7544 after this feature was added, despite notes_input=None
        # taking the byte-identical old code path. Not a correctness bug --
        # the numeric forward graph is unchanged when notes_input is None --
        # just a reproducibility side effect of shared global RNG state. If
        # exact Uni reproducibility matters later, gate construction of
        # text_pooler/text_merge behind `getattr(args, "enable_text", False)`
        # instead of always building them.
        # Always constructed (even for Uni-only runs) for simplicity; if
        # forecasting() is never called with notes_input, these parameters
        # just sit unused/untrained, which is harmless but slightly
        # wasteful -- fine for now, can gate behind args later if it matters.
        self.total_window = float(args.history + args.pred_window)
        # NOTE: notes_input (batch_dict["notes_embeddings"]) is the RAW
        # embedding saved by compute_text_embeddings.py -- its last-dim size
        # is whatever --llm_model_fusion's native hidden size is (e.g. 768
        # for GPT2 *and* for BERT-base -- both happen to be 768-dim, see
        # fusions/load_llm.py's _ALIAS comments). args.d_txt is really "the
        # dimension fusions/TTF_*.py project raw embeddings TO" (see
        # fusions/TTF_T2V_XAttn.py's input_proj), not necessarily the raw
        # size -- it only coincides with the raw size for GPT2/BERT because
        # d_txt's default (768) happens to match both. If --llm_model_fusion
        # ever changes to Llama/DeepSeek (4096-dim) while --d_txt stays at
        # its default, self.text_pooler.proj_in below will raise a shape-
        # mismatch RuntimeError on the first forward call (loud failure, not
        # silently wrong -- acceptable, but flagging so it's not a surprise).
        d_txt_in = int(getattr(args, "d_txt", 768))
        self.text_pooler = GPTextPooler(
            d_txt_in=d_txt_in,
            hid_dim=self.hid_dim,
            total_window=self.total_window,
            n_query=n_query,
            n_heads=int(getattr(args, "n_heads_fusion", 1)),
            dropout=float(args.dropout),
        )
        self.text_merge = GPTextGatedMerge(self.hid_dim)

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
        notes_input: optional (B, N_max, d_txt) precomputed note embeddings.
            When None (default), this is the original numeric-only path,
            identical to before this feature existed. When provided (along
            with tau), text is merged into `fused` before self.backbone().
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

        if notes_input is not None:
            if tau is None:
                raise ValueError("tau is required when notes_input is provided")
            # Pool text onto the *history-side* query grid t_query (Q points,
            # no padding column yet), then pad the same way gp_input was
            # padded (a leading zero column) so the two line up at Q+1.
            text_pooled, has_text = self.text_pooler(
                notes_input, tau, self.gp.t_query
            )  # (B, hid_dim, Q), (B,)
            text_pooled = F.pad(text_pooled, (1, 0))  # (B, hid_dim, Q+1)
            # Belt-and-suspenders zeroing for no-note samples: GPTextGatedMerge
            # already forces g=1 (exact no-op) for these via has_text_mask, so
            # this pre-zeroing isn't load-bearing for correctness anymore, but
            # keeps the values the gate module sees clean/predictable too.
            has_text_mask = has_text.view(B, 1, 1, 1).to(text_pooled.dtype)
            text_pooled = text_pooled.unsqueeze(2).expand(-1, -1, N, -1)  # (B, hid_dim, N, Q+1)
            text_pooled = text_pooled * has_text_mask

            fused = self.text_merge(fused, text_pooled, has_text_mask)

        h = self.backbone(fused)  # (B, N, hid_dim)

        Lp = time_steps_to_predict.shape[1]
        h_exp = h.unsqueeze(2).expand(B, N, Lp, self.hid_dim)
        t_pred = time_steps_to_predict.to(X.dtype).view(B, 1, Lp, 1).expand(B, N, Lp, 1)
        te_pred = self.te(t_pred)  # (B, N, Lp, te_dim)

        dec_in = torch.cat([h_exp, te_pred], dim=-1)
        out = self.decoder(dec_in).squeeze(-1)  # (B, N, Lp)
        return out.permute(0, 2, 1)  # (B, Lp, N)
