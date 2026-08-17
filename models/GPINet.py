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
  - has NO built-in text fusion. Multimodality is handled uniformly for all
    models by fusions/FusionModel.py, applied on top of this model's
    numeric-only output (Y_ts) in main.py's training loop. This is required
    to make the GPINet-vs-tPatchGNN comparison isolate the backbone as the
    only variable.
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
    """GP interpolation -> Gauss-Hermite fusion -> MTGNN encoder -> per-query
    decoder. Numeric-only; text fusion is applied externally via
    fusions/FusionModel.py, same as every other model in this repo.
    """

    def __init__(self, args, supports=None, dropout=0):
        super().__init__()
        self.device = args.device
        self.N = args.C  # number of variables (== num_nodes)
        self.hid_dim = args.hid_dim
        self.te_dim = args.te_dim

        history_frac = float(args.history) / float(args.history + args.pred_window)
        n_query = int(getattr(args, "gpinet_query_points", 8))
        t_query = torch.linspace(0.0, history_frac, n_query)

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

    def get_hyperparams(self):
        return self.gp.get_hyperparams()

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
        """
        time_steps_to_predict: (B, Lp)        normalized query times in [0,1]
        X (observed_data):     (B, T_obs, N)
        truth_time_steps:      (B, T_obs)     shared time axis across variables
        mask (observed_mask):  (B, T_obs, N)
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

        h = self.backbone(fused)  # (B, N, hid_dim)

        Lp = time_steps_to_predict.shape[1]
        h_exp = h.unsqueeze(2).expand(B, N, Lp, self.hid_dim)
        t_pred = time_steps_to_predict.to(X.dtype).view(B, 1, Lp, 1).expand(B, N, Lp, 1)
        te_pred = self.te(t_pred)  # (B, N, Lp, te_dim)

        dec_in = torch.cat([h_exp, te_pred], dim=-1)
        out = self.decoder(dec_in).squeeze(-1)  # (B, N, Lp)
        return out.permute(0, 2, 1)  # (B, Lp, N)
