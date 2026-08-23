import math

import torch
import torch.nn as nn


class FutureTime2Vec(nn.Module):
    """Learnable linear + periodic encoding for future query times."""

    def __init__(self, d_time: int):
        super().__init__()
        if d_time < 2:
            raise ValueError("d_time must be >= 2")
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, d_time - 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        linear = self.linear(t)
        periodic = torch.sin(self.periodic(t))
        return torch.cat([linear, periodic], dim=-1)


class MMF_VarTime_SlotGate(nn.Module):
    """Variable-time semantic-slot fusion with NULL rejection.

    Inputs follow the repository's MMF convention, with ``t_hat`` added for
    explicit future-time queries:

      Y_ts:  (B, T, C)       numerical/base forecast
      E_txt: (B, T, d_txt)   flattened semantic slots from
                              TTF_SemTime_Slots
      M_txt: (B, 1) or (B,)  whether the sample contains any valid text
      t_hat: (B, T) or (T,)  normalized future query times

    The flattened text representation is restored to H semantic slots.  Every
    (future time, variable) pair creates its own query and attends over those
    slots plus a learnable NULL key.  The NULL value is fixed at zero, allowing
    the model to reject all text.  The final output is an additive residual:

        Y_out = Y_ts + kappa * gate * delta_text

    so the numerical forecast is always the explicit base prediction.
    """

    def __init__(
        self,
        d_txt: int,
        C: int,
        d_attn: int = 128,
        n_heads_fusion: int = 1,
        dropout: float = 0.1,
        kappa: float = 0.5,
        semantic_slots: int = 4,
        gate_bias: float = -1.0,
    ):
        super().__init__()

        if C < 1:
            raise ValueError("C must be >= 1")
        if semantic_slots < 1:
            raise ValueError("semantic_slots must be >= 1")
        if d_txt % semantic_slots != 0:
            raise ValueError(
                f"d_txt={d_txt} must be divisible by "
                f"semantic_slots={semantic_slots}"
            )
        if d_attn < 1:
            raise ValueError("d_attn must be >= 1")
        if n_heads_fusion < 1 or d_attn % n_heads_fusion != 0:
            raise ValueError(
                f"d_attn={d_attn} must be divisible by "
                f"n_heads_fusion={n_heads_fusion}"
            )
        if kappa < 0:
            raise ValueError("kappa must be >= 0")

        self.C = int(C)
        self.d_txt = int(d_txt)
        self.d_attn = int(d_attn)
        self.semantic_slots = int(semantic_slots)
        self.slot_dim = self.d_txt // self.semantic_slots
        self.n_heads = int(n_heads_fusion)
        self.head_dim = self.d_attn // self.n_heads
        self.kappa = float(kappa)

        # Numeric query = local variable value + global multivariate state +
        # variable identity + future-time encoding.
        self.local_value_proj = nn.Linear(1, self.d_attn, bias=False)
        self.global_state_proj = nn.Linear(self.C, self.d_attn, bias=False)
        self.variable_embedding = nn.Parameter(
            torch.randn(self.C, self.d_attn) * 0.02
        )

        d_time = max(min(self.d_attn // 4, 32), 4)
        self.time2vec = FutureTime2Vec(d_time)
        self.time_proj = nn.Linear(d_time, self.d_attn, bias=False)
        self.query_norm = nn.LayerNorm(self.d_attn)

        # Slot projections.  The TTF slot dimension remains explicit here.
        self.slot_key = nn.Linear(self.slot_dim, self.d_attn, bias=False)
        self.slot_value = nn.Linear(self.slot_dim, self.d_attn, bias=False)

        # NULL has a learnable key so each query can choose it, but its value is
        # exactly zero so attending to NULL cannot create a text correction.
        self.null_key = nn.Parameter(torch.randn(1, self.d_attn) * 0.02)

        # Gate considers the numerical query, selected text context, their
        # elementwise agreement, and absolute disagreement.
        self.gate_net = nn.Sequential(
            nn.Linear(4 * self.d_attn, self.d_attn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_attn, 1),
        )
        nn.init.constant_(self.gate_net[-1].bias, float(gate_bias))

        # No biases: if text context is zero, delta_text is exactly zero.
        self.delta_net = nn.Sequential(
            nn.Linear(2 * self.d_attn, self.d_attn, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_attn, 1, bias=False),
        )
        # No affine bias: a zero/null context stays exactly zero.
        self.context_norm = nn.LayerNorm(
            self.d_attn, elementwise_affine=False
        )
        self.dropout = nn.Dropout(dropout)

        # Diagnostics populated on every forward pass.
        self.last_slot_attention = None
        self.last_null_probability = None
        self.last_gate = None
        self.last_delta = None
        self.last_correction = None

    def _prepare_time(
        self,
        t_hat: torch.Tensor | None,
        batch_size: int,
        future_steps: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if t_hat is None:
            # Compatibility fallback.  The explicit normalized t_hat supplied
            # by FusionModel is preferred, especially for irregular horizons.
            return torch.linspace(
                0.0,
                1.0,
                future_steps,
                device=reference.device,
                dtype=reference.dtype,
            ).unsqueeze(0).expand(batch_size, -1)

        t_hat = t_hat.to(device=reference.device, dtype=reference.dtype)
        if t_hat.dim() == 1:
            t_hat = t_hat.unsqueeze(0).expand(batch_size, -1)
        if t_hat.ndim != 2 or t_hat.shape != (batch_size, future_steps):
            raise ValueError(
                "Expected t_hat shape "
                f"{(batch_size, future_steps)} or {(future_steps,)}, "
                f"got {tuple(t_hat.shape)}"
            )
        return t_hat

    def forward(
        self,
        Y_ts: torch.Tensor,
        E_txt: torch.Tensor,
        M_txt: torch.Tensor,
        t_hat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if Y_ts.ndim != 3:
            raise ValueError(
                f"Y_ts must have shape (B, T, C), got {tuple(Y_ts.shape)}"
            )
        if E_txt.ndim != 3:
            raise ValueError(
                f"E_txt must have shape (B, T, d_txt), got {tuple(E_txt.shape)}"
            )

        B, T, C = Y_ts.shape
        if C != self.C:
            raise ValueError(f"Expected C={self.C}, got C={C}")
        if E_txt.shape != (B, T, self.d_txt):
            raise ValueError(
                f"Expected E_txt shape {(B, T, self.d_txt)}, "
                f"got {tuple(E_txt.shape)}"
            )

        t_hat = self._prepare_time(t_hat, B, T, Y_ts)
        has_text = M_txt.to(torch.bool).reshape(B, -1).any(dim=1)

        # (B, T, H, D_slot)
        E_slots = E_txt.reshape(
            B, T, self.semantic_slots, self.slot_dim
        )

        # Build one query for every future-time/variable pair.
        local_state = self.local_value_proj(Y_ts.unsqueeze(-1))
        global_state = self.global_state_proj(Y_ts).unsqueeze(2)
        variable_state = self.variable_embedding.view(1, 1, C, self.d_attn)
        time_state = self.time_proj(
            self.time2vec(t_hat.unsqueeze(-1))
        ).unsqueeze(2)
        query = self.query_norm(
            local_state + global_state + variable_state + time_state
        )  # (B, T, C, D_attn)

        real_keys = self.slot_key(E_slots)
        real_values = self.slot_value(E_slots)
        null_keys = self.null_key.view(1, 1, 1, self.d_attn).expand(
            B, T, 1, -1
        )
        null_values = torch.zeros_like(null_keys)
        all_keys = torch.cat([real_keys, null_keys], dim=2)
        all_values = torch.cat([real_values, null_values], dim=2)

        slot_count = self.semantic_slots + 1
        q_heads = query.view(B, T, C, self.n_heads, self.head_dim)
        k_heads = all_keys.view(
            B, T, slot_count, self.n_heads, self.head_dim
        )
        v_heads = all_values.view(
            B, T, slot_count, self.n_heads, self.head_dim
        )

        scores = torch.einsum(
            "btchd,btshd->btchs", q_heads, k_heads
        ) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)
        context_heads = torch.einsum(
            "btchs,btshd->btchd", attention, v_heads
        )
        context = context_heads.reshape(B, T, C, self.d_attn)
        context = self.dropout(self.context_norm(context))

        # Average the multi-head NULL probability for one interpretable
        # rejection score per variable and future time.
        null_probability = attention[..., -1].mean(dim=-1)

        interaction = torch.cat(
            [
                query,
                context,
                query * context,
                torch.abs(query - context),
            ],
            dim=-1,
        )
        learned_gate = torch.sigmoid(self.gate_net(interaction).squeeze(-1))
        gate = learned_gate * (1.0 - null_probability)
        gate = gate * has_text.view(B, 1, 1).to(gate.dtype)

        delta_features = torch.cat([context, query * context], dim=-1)
        delta_text = torch.tanh(self.delta_net(delta_features).squeeze(-1))

        correction = self.kappa * gate * delta_text
        correction = correction * has_text.view(B, 1, 1).to(correction.dtype)
        Y_out = Y_ts + correction

        self.last_slot_attention = attention.mean(dim=-2).detach()
        self.last_null_probability = null_probability.detach()
        self.last_gate = gate.detach()
        self.last_delta = delta_text.detach()
        self.last_correction = correction.detach()

        return Y_out
