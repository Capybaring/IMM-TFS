# BUILD_ID: active-slot-null-routing-v10-20260829
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
    """Variable-time residual fusion with active-slot/NULL routing.

    Each variable-time query chooses among the semantic slots that are actually
    populated for the current patient plus a learned NULL/no-match option.
    Empty slots never participate in attention.  After a short warmup,
    ``1 - p(NULL)`` is the sole text-relevance factor; there is no independent
    gate network competing with the residual branch for the same role.

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
        null_logit_bias: float | None = None,
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
        # Reuse the existing runner knob for backward-compatible experiment
        # configuration: mmf_slot_gate_bias now initializes the NULL logit.
        if null_logit_bias is None:
            null_logit_bias = float(gate_bias)

        self.local_value_proj = nn.Linear(1, self.d_attn, bias=False)
        self.global_state_proj = nn.Linear(self.C, self.d_attn, bias=False)
        self.variable_embedding = nn.Parameter(
            torch.randn(self.C, self.d_attn) * 0.02
        )
        d_time = max(min(self.d_attn // 4, 32), 4)
        self.time2vec = FutureTime2Vec(d_time)
        self.time_proj = nn.Linear(d_time, self.d_attn, bias=False)
        self.query_norm = nn.LayerNorm(self.d_attn)

        # Normalize slot direction before attention, then restore its RMS in
        # forward().  Plain LayerNorm would erase TTF's recency and slot-mass
        # strengths, making those controls ineffective for every non-zero slot.
        self.slot_norm = nn.LayerNorm(self.slot_dim)
        self.slot_key = nn.Linear(self.slot_dim, self.d_attn, bias=False)
        self.slot_value = nn.Linear(self.slot_dim, self.d_attn, bias=False)

        # NULL participates in the same dot-product classification as the real
        # semantic slots.  Its value is implicitly zero; only its key and bias
        # are learned.  Query dependence makes NULL selection variable- and
        # future-time-specific rather than a global rejection scalar.
        self.null_key = nn.Parameter(
            torch.randn(self.d_attn) * 0.02
        )
        self.null_logit_bias = nn.Parameter(
            torch.tensor(float(null_logit_bias), dtype=torch.float32)
        )
        self.supports_null_diagnostic = True

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

        self.last_slot_attention = None
        self.last_null_probability = None
        self.last_gate = None
        self.last_variable_relevance = None
        self.last_delta = None
        self.last_correction = None
        self.last_candidate_correction = None
        self.last_text_mask = None
        self.last_context = None
        self.last_active_slot_mask = None
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
        if not torch.isfinite(Y_ts).all():
            raise ValueError("Numerical forecast contains NaN or Inf values")
        if not torch.isfinite(E_txt).all():
            raise ValueError("Text features contain NaN or Inf values")

        t_hat = self._prepare_time(t_hat, B, T, Y_ts)
        has_text = M_txt.to(torch.bool).reshape(B, -1).any(dim=1)
        text_mask = has_text.view(B, 1, 1).to(Y_ts.dtype)

        E_slots = E_txt.reshape(B, T, self.semantic_slots, self.slot_dim)
        # TTF writes an exactly zero vector for an unpopulated semantic slot.
        # Record activity before normalization so empty slots can be excluded
        # from the categorical correspondence decision.
        active_slots = E_slots.float().square().sum(dim=-1) > 1e-12
        # Strict TTF routing can produce exactly empty (all-zero) slots.  The
        # derivative of sqrt(x) is singular at x=0, so the previous RMS
        # restoration created a finite forward value but a NaN backward
        # gradient.  Clamping the power before sqrt keeps empty slots exactly
        # zero after LayerNorm multiplication while making their gradient
        # finite.
        slot_power = E_slots.float().square().mean(dim=-1, keepdim=True)
        slot_rms = slot_power.clamp_min(1e-8).sqrt().to(E_slots.dtype)
        E_slots = self.slot_norm(E_slots) * slot_rms

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

        # Empty real slots are invalid classes.  NULL is always valid, so even
        # a sample with no active slots has a well-defined softmax result.
        active_slot_mask = active_slots[:, :, None, None, :]
        real_scores = scores.float().masked_fill(
            ~active_slot_mask,
            -1e4,
        )
        null_key_heads = self.null_key.reshape(
            self.n_heads,
            self.head_dim,
        ).float()
        null_scores = torch.einsum(
            "btchd,hd->btch",
            q_heads.float(),
            null_key_heads,
        ) / math.sqrt(self.head_dim)
        null_scores = null_scores + self.null_logit_bias.float()
        all_scores = torch.cat(
            [real_scores, null_scores.unsqueeze(-1)],
            dim=-1,
        )
        all_attention = torch.softmax(all_scores, dim=-1)
        real_attention = all_attention[..., :slot_count]
        real_attention = real_attention * active_slot_mask.to(
            real_attention.dtype
        )
        null_attention = all_attention[..., slot_count]

        # Separate correspondence from relevance.  Conditional real-slot
        # attention answers "which active slot?"; NULL mass answers "does any
        # active slot correspond at all?".  This avoids attenuating the text
        # context twice.
        conditional_attention = real_attention / real_attention.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)

        context_heads = torch.einsum(
            "btchs,btshd->btchd",
            conditional_attention.to(v_heads.dtype),
            v_heads,
        )
        context = context_heads.reshape(B, T, C, self.d_attn)

        null_probability = null_attention.mean(dim=-1).to(Y_ts.dtype)
        learned_relevance = (1.0 - null_probability).clamp(0.0, 1.0)

        # Preserve the existing protected warmup: real-slot correspondence and
        # the residual branch can learn before NULL relevance controls the
        # correction magnitude.  No external objective or schedule changes.
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

        # ``gate`` is kept as a diagnostic/API name.  It is no longer produced
        # by a separate network; it is exactly the non-NULL correspondence
        # probability (or the fixed warmup value) under the sample text mask.
        gate = variable_relevance * text_mask

        delta_features = torch.cat([context, query * context], dim=-1)
        delta_text = torch.tanh(
            self.delta_out(self.delta_hidden(delta_features)).squeeze(-1)
        )

        candidate_correction = self.kappa * delta_text * text_mask
        correction = gate * candidate_correction
        Y_out = Y_ts + correction

        diagnostic_attention = all_attention.mean(dim=-2).to(Y_ts.dtype)
        self.last_slot_attention = diagnostic_attention.detach()
        self.last_null_probability = null_probability.detach()
        self.last_gate = gate.detach()
        self.last_variable_relevance = variable_relevance.detach()
        self.last_delta = delta_text.detach()
        self.last_correction = correction.detach()
        self.last_candidate_correction = candidate_correction.detach()
        self.last_text_mask = text_mask.detach()
        self.last_context = context.detach()
        self.last_active_slot_mask = active_slots.detach()
        self.last_gate_warmup_active = bool(warmup_active)

        return Y_out
