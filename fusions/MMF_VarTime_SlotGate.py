# BUILD_ID: semantic-slot-gate-warmup-v5-20260824
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
    """Variable-time residual fusion over semantic text slots.

    Real-slot attention selects text content; the variable-time relevance gate
    rejects unhelpful text.  During a short training warmup the gate is held at
    a non-zero value so the residual and TTF branches can learn before the gate
    is allowed to close.  This avoids the multiplicative cold start
    ``correction = kappa * gate * delta`` that previously drove every gate to
    approximately zero.

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
        gate_bias: float = 0.0,
        delta_init_std: float = 1e-2,
        gate_warmup_epochs: int = 5,
        gate_warmup_value: float = 0.5,
        null_logit_bias: float = 1.0,
    ):
        super().__init__()

        if C < 1:
            raise ValueError("C must be >= 1")
        if semantic_slots < 2:
            raise ValueError(
                "MMF_VarTime_SlotGate requires semantic_slots >= 2; "
                "one slot makes slot attention identically one"
            )
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
        if delta_init_std <= 0:
            raise ValueError("delta_init_std must be > 0")
        if gate_warmup_epochs < 0:
            raise ValueError("gate_warmup_epochs must be >= 0")
        if not 0.0 < gate_warmup_value <= 1.0:
            raise ValueError("gate_warmup_value must be in (0, 1]")

        self.C = int(C)
        self.d_txt = int(d_txt)
        self.d_attn = int(d_attn)
        self.semantic_slots = int(semantic_slots)
        self.slot_dim = self.d_txt // self.semantic_slots
        self.n_heads = int(n_heads_fusion)
        self.head_dim = self.d_attn // self.n_heads
        self.kappa = float(kappa)
        self.gate_warmup_epochs = int(gate_warmup_epochs)
        self.gate_warmup_value = float(gate_warmup_value)
        self.training_epoch = 0
        # Retained only for constructor compatibility with older runners and
        # checkpoints.  NULL no longer participates in attention or gating.
        del null_logit_bias

        self.local_value_proj = nn.Linear(1, self.d_attn, bias=False)
        self.global_state_proj = nn.Linear(self.C, self.d_attn, bias=False)
        self.variable_embedding = nn.Parameter(
            torch.randn(self.C, self.d_attn) * 0.02
        )
        d_time = max(min(self.d_attn // 4, 32), 4)
        self.time2vec = FutureTime2Vec(d_time)
        self.time_proj = nn.Linear(d_time, self.d_attn, bias=False)
        self.query_norm = nn.LayerNorm(self.d_attn)

        # Normalize real slots before attention.  Do not normalize the weighted
        # context afterwards, so its magnitude remains available to the gate.
        self.slot_norm = nn.LayerNorm(self.slot_dim)
        self.slot_key = nn.Linear(self.slot_dim, self.d_attn, bias=False)
        self.slot_value = nn.Linear(self.slot_dim, self.d_attn, bias=False)

        self.gate_net = nn.Sequential(
            nn.Linear(4 * self.d_attn, self.d_attn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_attn, 1),
        )
        # A small non-zero weight lets gate decisions depend on text/numeric
        # interactions as soon as warmup ends.  A zero final layer made every
        # patient, variable and future time start with the same gate.
        nn.init.normal_(self.gate_net[-1].weight, mean=0.0, std=1e-2)
        nn.init.constant_(self.gate_net[-1].bias, float(gate_bias))

        self.delta_hidden = nn.Sequential(
            nn.Linear(2 * self.d_attn, self.d_attn, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta_out = nn.Linear(self.d_attn, 1, bias=False)
        nn.init.normal_(
            self.delta_out.weight,
            mean=0.0,
            std=float(delta_init_std),
        )

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
        self.last_gate_warmup_active = False

    def set_training_epoch(self, epoch: int) -> None:
        """Inform the fusion gate which training epoch is about to run."""
        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        self.training_epoch = int(epoch)

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

        slot_count = self.semantic_slots
        q_heads = query.reshape(B, T, C, self.n_heads, self.head_dim)
        k_heads = real_keys.reshape(
            B, T, slot_count, self.n_heads, self.head_dim
        )
        v_heads = real_values.reshape(
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

        # Kept as an all-zero diagnostic for compatibility with the existing
        # evaluation output.  It is not part of the forward computation.
        null_probability = Y_ts.new_zeros(B, T, C)

        interaction = torch.cat(
            [query, context, query * context, torch.abs(query - context)],
            dim=-1,
        )
        gate_logits = self.gate_net(interaction).squeeze(-1)
        gate_logits = gate_logits + self.variable_gate_logit.view(1, 1, C)
        learned_relevance = torch.sigmoid(gate_logits)

        # Do not let the rejection gate kill an untrained residual.  Keeping a
        # constant forward gate also intentionally freezes gate parameters in
        # this phase; delta/attention/TTF receive useful gradients first.  The
        # learned gate starts training after warmup from its neutral prior.
        warmup_active = (
            self.training
            and self.training_epoch < self.gate_warmup_epochs
        )
        if warmup_active:
            variable_relevance = torch.full_like(
                learned_relevance,
                self.gate_warmup_value,
            )
        else:
            variable_relevance = learned_relevance

        # The variable-time gate is now the sole learned rejection mechanism.
        # text_mask still guarantees exact identity when a sample has no text.
        gate = variable_relevance * text_mask

        delta_features = torch.cat([context, query * context], dim=-1)
        delta_text = torch.tanh(
            self.delta_out(self.delta_hidden(delta_features)).squeeze(-1)
        )

        correction = self.kappa * gate * delta_text
        Y_out = Y_ts + correction

        # Preserve the old H+1 diagnostic shape by appending a zero-probability
        # NULL column.  This keeps evaluation entropy code valid when H == 1.
        diagnostic_attention = torch.cat(
            [
                attention.mean(dim=-2),
                null_probability.unsqueeze(-1),
            ],
            dim=-1,
        )
        self.last_slot_attention = diagnostic_attention.detach()
        self.last_null_probability = null_probability.detach()
        self.last_gate = gate.detach()
        self.last_variable_relevance = variable_relevance.detach()
        self.last_delta = delta_text.detach()
        self.last_correction = correction.detach()
        self.last_gate_warmup_active = bool(warmup_active)

        return Y_out
