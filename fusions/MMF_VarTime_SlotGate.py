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
        return torch.cat([self.linear(t), torch.sin(self.periodic(t))], dim=-1)


class MMF_VarTime_SlotGate(nn.Module):
    """Conservative variable-time residual fusion over semantic text slots.

    The final delta layer is zero-initialized, NULL has an explicit prior, and
    the post-attention context is not layer-normalized.  Therefore the initial
    forward is exactly ``Y_out == Y_ts`` and selecting NULL really suppresses
    the text residual.

    Shapes:
        Y_ts:  (B, T, C)
        E_txt: (B, T, d_txt), containing H contiguous semantic slots
        M_txt: sample-level text mask
        t_hat: (B, T) or (T,)
    """

    def __init__(
        self,
        d_txt: int,
        C: int,
        d_attn: int = 128,
        n_heads_fusion: int = 1,
        dropout: float = 0.1,
        kappa: float = 0.1,
        semantic_slots: int = 4,
        gate_bias: float = -2.0,
        null_logit_bias: float = 1.0,
    ):
        super().__init__()

        if C < 1:
            raise ValueError("C must be >= 1")
        if semantic_slots < 1:
            raise ValueError("semantic_slots must be >= 1")
        if d_txt % semantic_slots != 0:
            raise ValueError(
                f"d_txt={d_txt} must be divisible by semantic_slots={semantic_slots}"
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

        self.local_value_proj = nn.Linear(1, self.d_attn, bias=False)
        self.global_state_proj = nn.Linear(self.C, self.d_attn, bias=False)
        self.variable_embedding = nn.Parameter(
            torch.randn(self.C, self.d_attn) * 0.02
        )
        d_time = max(min(self.d_attn // 4, 32), 4)
        self.time2vec = FutureTime2Vec(d_time)
        self.time_proj = nn.Linear(d_time, self.d_attn, bias=False)
        self.query_norm = nn.LayerNorm(self.d_attn)

        # Normalize real slots before attention.  The weighted context is not
        # normalized afterwards, so NULL probability still controls magnitude.
        self.slot_norm = nn.LayerNorm(self.slot_dim)
        self.slot_key = nn.Linear(self.slot_dim, self.d_attn, bias=False)
        self.slot_value = nn.Linear(self.slot_dim, self.d_attn, bias=False)
        self.null_key = nn.Parameter(torch.randn(1, self.d_attn) * 0.02)
        self.null_logit_bias = nn.Parameter(
            torch.tensor(float(null_logit_bias), dtype=torch.float32)
        )

        self.gate_net = nn.Sequential(
            nn.Linear(4 * self.d_attn, self.d_attn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_attn, 1),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, float(gate_bias))

        self.delta_hidden = nn.Sequential(
            nn.Linear(2 * self.d_attn, self.d_attn, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta_out = nn.Linear(self.d_attn, 1, bias=False)
        nn.init.zeros_(self.delta_out.weight)

        # Additive variable-specific relevance logit.  It is shared over future
        # time but combined with a time-specific query gate below.  This is the
        # main mechanism for learning that chest reports help only a subset of
        # the physiological targets.
        self.variable_gate_logit = nn.Parameter(torch.zeros(self.C))

        self.last_slot_attention = None
        self.last_null_probability = None
        self.last_gate = None
        self.last_variable_relevance = None
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
        text_mask = has_text.view(B, 1, 1).to(Y_ts.dtype)

        E_slots = E_txt.reshape(B, T, self.semantic_slots, self.slot_dim)
        E_slots = self.slot_norm(E_slots)

        # Detach only the residual-query input.  The explicit addition at the
        # end keeps the normal identity gradient to the numerical backbone.
        numeric_query_source = Y_ts.detach()
        local_state = self.local_value_proj(numeric_query_source.unsqueeze(-1))
        global_state = self.global_state_proj(numeric_query_source).unsqueeze(2)
        variable_state = self.variable_embedding.view(1, 1, C, self.d_attn)
        time_state = self.time_proj(
            self.time2vec(t_hat.unsqueeze(-1))
        ).unsqueeze(2)
        query = self.query_norm(
            local_state + global_state + variable_state + time_state
        )

        real_keys = self.slot_key(E_slots)
        real_values = self.slot_value(E_slots)
        null_keys = self.null_key.view(1, 1, 1, self.d_attn).expand(B, T, 1, -1)
        null_values = torch.zeros_like(null_keys)
        all_keys = torch.cat([real_keys, null_keys], dim=2)
        all_values = torch.cat([real_values, null_values], dim=2)

        slot_count = self.semantic_slots + 1
        q_heads = query.reshape(B, T, C, self.n_heads, self.head_dim)
        k_heads = all_keys.reshape(
            B, T, slot_count, self.n_heads, self.head_dim
        )
        v_heads = all_values.reshape(
            B, T, slot_count, self.n_heads, self.head_dim
        )

        scores = torch.einsum(
            "btchd,btshd->btchs", q_heads, k_heads
        ) / math.sqrt(self.head_dim)
        score_bias = torch.cat(
            [
                scores.new_zeros(self.semantic_slots),
                self.null_logit_bias.to(scores.dtype).reshape(1),
            ]
        )
        scores = scores + score_bias.view(1, 1, 1, 1, slot_count)
        attention = torch.softmax(scores, dim=-1)

        context_heads = torch.einsum(
            "btchs,btshd->btchd", attention, v_heads
        )
        context = context_heads.reshape(B, T, C, self.d_attn)

        null_probability = attention[..., -1].mean(dim=-1)
        real_mass = (1.0 - null_probability).clamp(0.0, 1.0)

        interaction = torch.cat(
            [query, context, query * context, torch.abs(query - context)],
            dim=-1,
        )
        gate_logits = self.gate_net(interaction).squeeze(-1)
        gate_logits = gate_logits + self.variable_gate_logit.view(1, 1, C)
        variable_relevance = torch.sigmoid(gate_logits)

        # real_mass comes directly from competition between text and NULL.
        # Keep it linear: with only one or two reports, squaring it can make the
        # learning signal too small for the relevance gate to ever open.
        gate = variable_relevance * real_mass * text_mask

        delta_features = torch.cat([context, query * context], dim=-1)
        delta_text = torch.tanh(
            self.delta_out(self.delta_hidden(delta_features)).squeeze(-1)
        )

        correction = self.kappa * gate * delta_text * text_mask
        Y_out = Y_ts + correction

        self.last_slot_attention = attention.mean(dim=-2).detach()
        self.last_null_probability = null_probability.detach()
        self.last_gate = gate.detach()
        self.last_variable_relevance = variable_relevance.detach()
        self.last_delta = delta_text.detach()
        self.last_correction = correction.detach()

        return Y_out