# BUILD_ID: strict-direct-indirect-slot-routing-v12-20260830
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Strict direct slot matching plus limited indirect variable propagation.

    After warmup, every variable-time query makes one deterministic categorical
    choice among the populated semantic slots and a learned NULL/no-match
    option.  A real choice produces a direct correction from exactly one slot;
    a NULL choice produces exactly zero direct correction.  Straight-through
    routing keeps this hard forward decision trainable without adding an outer
    supervision loss.

    A small second path permits realistic indirect effects.  Direct corrections
    are propagated through a learned, zero-diagonal variable graph only to
    variables that chose NULL.  Consequently an unmatched slot cannot directly
    alter that variable, and the indirect contribution remains separately
    observable and bounded by ``indirect_strength``.

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
        indirect_strength: float = 0.1,
        indirect_temperature: float = 0.5,
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
        if not 0.0 <= indirect_strength <= 1.0:
            raise ValueError("indirect_strength must be in [0, 1]")
        if indirect_temperature <= 0.0:
            raise ValueError("indirect_temperature must be > 0")

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
        self.indirect_strength = float(indirect_strength)
        self.indirect_temperature = float(indirect_temperature)
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
        self.supports_strict_slot_routing = True
        self.supports_indirect_diagnostic = True

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
        self.last_direct_correction = None
        self.last_indirect_correction = None
        self.last_text_mask = None
        self.last_context = None
        self.last_active_slot_mask = None
        self.last_hard_slot_choice = None
        self.last_variable_adjacency = None
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
        # Slot correspondence is decided from variable identity and forecast
        # time.  Numerical state is deliberately excluded from this classifier
        # so a base-prediction bias cannot masquerade as semantic matching.
        routing_query = self.query_norm(variable_state + time_state)
        query = self.query_norm(
            local_state + global_state + variable_state + time_state
        )

        real_keys = self.slot_key(E_slots)
        real_values = self.slot_value(E_slots)

        slot_count = self.semantic_slots
        routing_q_heads = routing_query.reshape(
            B, T, C, self.n_heads, self.head_dim
        )
        k_heads = real_keys.reshape(
            B, T, slot_count, self.n_heads, self.head_dim
        )
        v_heads = real_values.reshape(
            B, T, slot_count, self.n_heads, self.head_dim
        )

        scores = torch.einsum(
            "btchd,btshd->btchs", routing_q_heads, k_heads
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
            routing_q_heads.float(),
            null_key_heads,
        ) / math.sqrt(self.head_dim)
        null_scores = null_scores + self.null_logit_bias.float()
        all_scores = torch.cat(
            [real_scores, null_scores.unsqueeze(-1)],
            dim=-1,
        )
        # Average head evidence before classification so each variable-time
        # pair makes exactly one semantic decision even when multiple attention
        # heads are configured.
        routing_logits = all_scores.mean(dim=-2)
        soft_routing = torch.softmax(routing_logits, dim=-1)
        hard_choice = soft_routing.argmax(dim=-1)
        hard_routing = F.one_hot(
            hard_choice,
            num_classes=slot_count + 1,
        ).to(soft_routing.dtype)

        # Straight-through argmax: the forward path is strictly one-hot while
        # the backward path follows the corresponding softmax probabilities.
        if self.training:
            strict_routing = (
                hard_routing + soft_routing - soft_routing.detach()
            )
        else:
            strict_routing = hard_routing

        soft_real_attention = soft_routing[..., :slot_count]
        soft_null_attention = soft_routing[..., slot_count]

        # Conditional real-slot weights are used only to preserve the existing
        # fixed non-zero warmup.  After warmup, the raw real attention is used:
        # the omitted probability belongs to the zero-valued NULL option and
        # therefore attenuates context exactly once.
        conditional_attention = soft_real_attention / soft_real_attention.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)

        null_probability = soft_null_attention.to(Y_ts.dtype)
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
            variable_relevance = strict_routing[
                ..., :slot_count
            ].sum(dim=-1).to(Y_ts.dtype)

        if warmup_active:
            effective_real_attention = (
                conditional_attention * self.gate_warmup_value
            )
        else:
            effective_real_attention = strict_routing[..., :slot_count]

        context_heads = torch.einsum(
            "btchs,btshd->btchd",
            effective_real_attention.unsqueeze(-2).to(v_heads.dtype),
            v_heads,
        )
        context = context_heads.reshape(B, T, C, self.d_attn)

        # This is now a strict direct-match indicator after warmup: 1 means one
        # real slot was selected and 0 means NULL.  During protected warmup it
        # retains the previous fixed non-zero value.
        gate = variable_relevance * text_mask

        delta_features = torch.cat([context, query * context], dim=-1)
        delta_text = torch.tanh(
            self.delta_out(self.delta_hidden(delta_features)).squeeze(-1)
        )

        direct_correction = self.kappa * delta_text * text_mask

        # Build a learned variable graph from identity embeddings.  Removing
        # the diagonal prevents this path from duplicating the direct residual.
        # It is deliberately small and can only write into NULL-routed targets.
        if self.C > 1 and self.indirect_strength > 0.0 and not warmup_active:
            variable_direction = F.normalize(
                self.variable_embedding.float(),
                dim=-1,
                eps=1e-8,
            )
            adjacency_logits = torch.matmul(
                variable_direction,
                variable_direction.transpose(0, 1),
            ) / self.indirect_temperature
            diagonal = torch.eye(
                self.C,
                device=adjacency_logits.device,
                dtype=torch.bool,
            )
            adjacency = torch.softmax(
                adjacency_logits.masked_fill(diagonal, -1e4),
                dim=-1,
            ).to(direct_correction.dtype)
            propagated = torch.einsum(
                "ij,btj->bti",
                adjacency,
                direct_correction,
            )
            no_direct_match = (1.0 - variable_relevance).clamp(0.0, 1.0)
            indirect_correction = (
                self.indirect_strength
                * propagated
                * no_direct_match
                * text_mask
            )
        else:
            adjacency = direct_correction.new_zeros(self.C, self.C)
            indirect_correction = torch.zeros_like(direct_correction)

        candidate_correction = direct_correction
        correction = direct_correction + indirect_correction
        Y_out = Y_ts + correction

        self.last_slot_attention = soft_routing.to(Y_ts.dtype).detach()
        self.last_null_probability = null_probability.detach()
        self.last_gate = gate.detach()
        self.last_variable_relevance = learned_relevance.detach()
        self.last_delta = delta_text.detach()
        self.last_correction = correction.detach()
        self.last_candidate_correction = candidate_correction.detach()
        self.last_direct_correction = direct_correction.detach()
        self.last_indirect_correction = indirect_correction.detach()
        self.last_text_mask = text_mask.detach()
        self.last_context = context.detach()
        self.last_active_slot_mask = active_slots.detach()
        self.last_hard_slot_choice = hard_choice.detach()
        self.last_variable_adjacency = adjacency.detach()
        self.last_gate_warmup_active = bool(warmup_active)

        return Y_out
