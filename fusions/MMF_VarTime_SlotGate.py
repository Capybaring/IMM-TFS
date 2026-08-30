# BUILD_ID: variable-time-note-cross-attention-paper-gr-add-v3-20260830
import math

import torch
import torch.nn as nn


class FutureTime2Vec(nn.Module):
    """Learnable linear and periodic encoding for future query times."""

    def __init__(self, d_time: int):
        super().__init__()
        if d_time < 2:
            raise ValueError("d_time must be >= 2")
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, d_time - 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.linear(value), torch.sin(self.periodic(value))],
            dim=-1,
        )


class MMF_VarTime_SlotGate(nn.Module):
    """Variable-time soft cross-attention over unaggregated note tokens.

    The legacy class and file names are preserved for existing scripts.  The
    module no longer performs semantic-slot classification, hard routing, a
    NULL categorical decision, or indirect variable-graph propagation.
    Instead, every variable/future-time query softly attends to all aligned
    real notes.  The resulting variable-specific contexts replace the paper's
    single global text vector; prediction fusion itself follows MMF_GR_Add:
    a joint GRU predicts a normalized residual and a learned gate controls how
    much of that residual is added to the numerical forecast.

    Preferred shapes:
        Y_ts:  (B, T, C)
        E_txt: (B, T, K, d_txt)
        M_txt: (B, T, K)
        t_hat: (B, T) or (T,)

    A legacy (B, T, d_txt) E_txt input is accepted as one note token per
    future time to keep older TTF/MMF pairings from failing immediately.
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
        if d_txt < 1:
            raise ValueError("d_txt must be >= 1")
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

        # Retain obsolete constructor arguments so the existing CLI and
        # FusionModel do not need to change in this first iteration.
        self.legacy_semantic_slots = int(semantic_slots)
        self.legacy_null_logit_bias = (
            None if null_logit_bias is None else float(null_logit_bias)
        )
        self.legacy_indirect_strength = float(indirect_strength)
        self.legacy_indirect_temperature = float(indirect_temperature)

        self.C = int(C)
        self.d_txt = int(d_txt)
        self.d_attn = int(d_attn)
        self.n_heads = int(n_heads_fusion)
        self.head_dim = self.d_attn // self.n_heads
        # Kept only so existing CLI/configuration objects remain compatible.
        # Paper-style GR-Add fusion does not use kappa or gate warmup.
        self.kappa = float(kappa)
        self.legacy_delta_init_std = float(delta_init_std)
        self.gate_warmup_epochs = int(gate_warmup_epochs)
        self.gate_warmup_value = float(gate_warmup_value)
        self.training_epoch = 0

        # Variable/future-time query.
        self.local_value_proj = nn.Linear(1, self.d_attn, bias=False)
        self.global_state_proj = nn.Linear(self.C, self.d_attn, bias=False)
        self.variable_embedding = nn.Parameter(
            torch.randn(self.C, self.d_attn) * 0.02
        )
        d_time = max(min(self.d_attn // 4, 32), 4)
        self.time2vec = FutureTime2Vec(d_time)
        self.time_proj = nn.Linear(d_time, self.d_attn, bias=False)
        self.query_norm = nn.LayerNorm(self.d_attn)

        # Note keys and values.  Attention is implemented explicitly so the
        # same K note tokens can be queried independently by all C variables.
        self.note_norm = nn.LayerNorm(self.d_txt)
        self.note_key = nn.Linear(self.d_txt, self.d_attn, bias=False)
        self.note_value = nn.Linear(self.d_txt, self.d_attn, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.context_out = nn.Linear(self.d_attn, self.d_attn, bias=False)
        self.context_norm = nn.LayerNorm(self.d_attn)

        # Keep the paper MMF_GR_Add prediction-fusion architecture.  The only
        # change at this boundary is that its one global text vector is
        # replaced by C variable-specific cross-attention contexts.
        self.fusion_dim = self.C + self.C * self.d_attn
        self.gru = nn.GRU(
            input_size=self.fusion_dim,
            hidden_size=self.C,
            batch_first=True,
        )
        self.residual_head = nn.Linear(self.C, self.C)
        self.layer_norm = nn.LayerNorm(self.C)
        self.residual_dropout = nn.Dropout(dropout)
        self.gate_net = nn.Linear(self.fusion_dim, self.C)

        # The old per-variable residual MLP and its custom gate bias are no
        # longer part of the forward graph.  Preserve the value for experiment
        # metadata/checkpoint diagnostics only.
        self.legacy_gate_bias = float(gate_bias)

        # Evaluation capability flags.
        self.supports_null_diagnostic = False
        self.supports_strict_slot_routing = False
        self.supports_indirect_diagnostic = False

        # Existing diagnostic names are retained even though their semantics
        # are now note-level attention rather than slot-level routing.
        self.last_slot_attention = None
        self.last_note_attention = None
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

    def _prepare_note_tokens(
        self,
        E_txt: torch.Tensor,
        M_txt: torch.Tensor,
        batch_size: int,
        future_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if E_txt.ndim == 3:
            expected = (batch_size, future_steps, self.d_txt)
            if E_txt.shape != expected:
                raise ValueError(
                    f"Expected legacy E_txt shape {expected}, "
                    f"got {tuple(E_txt.shape)}"
                )
            E_txt = E_txt.unsqueeze(2)
        elif E_txt.ndim != 4:
            raise ValueError(
                "E_txt must have shape (B,T,K,d_txt) or legacy "
                f"(B,T,d_txt), got {tuple(E_txt.shape)}"
            )

        B, T, K, D = E_txt.shape
        if (B, T, D) != (batch_size, future_steps, self.d_txt):
            raise ValueError(
                "Expected E_txt leading/trailing dimensions "
                f"{(batch_size, future_steps, self.d_txt)}, "
                f"got {tuple(E_txt.shape)}"
            )

        mask = M_txt.to(torch.bool)
        if mask.ndim == 3 and mask.shape == (B, T, K):
            return E_txt, mask
        if mask.ndim == 2:
            if mask.shape == (B, K):
                return E_txt, mask[:, None, :].expand(-1, T, -1)
            if K == 1 and mask.shape == (B, T):
                return E_txt, mask.unsqueeze(-1)
            if mask.shape == (B, 1):
                return E_txt, mask[:, None, :].expand(-1, T, K)
        if mask.ndim == 1 and mask.shape == (B,):
            return E_txt, mask[:, None, None].expand(-1, T, K)

        raise ValueError(
            "M_txt must be compatible with (B,T,K); got "
            f"{tuple(mask.shape)} for {(B, T, K)}"
        )

    @staticmethod
    def _require_finite(name: str, value: torch.Tensor) -> None:
        if not torch.isfinite(value).all():
            raise FloatingPointError(
                f"MMF_VarTime_SlotGate produced NaN or Inf in {name}"
            )

    def forward(
        self,
        Y_ts: torch.Tensor,
        E_txt: torch.Tensor,
        M_txt: torch.Tensor,
        t_hat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if Y_ts.ndim != 3:
            raise ValueError(
                f"Y_ts must have shape (B,T,C), got {tuple(Y_ts.shape)}"
            )
        B, T, C = Y_ts.shape
        if C != self.C:
            raise ValueError(f"Expected C={self.C}, got C={C}")
        self._require_finite("numerical forecast", Y_ts)

        E_txt, note_mask = self._prepare_note_tokens(
            E_txt,
            M_txt,
            B,
            T,
        )
        self._require_finite("aligned note tokens", E_txt)
        K = E_txt.shape[2]
        t_hat = self._prepare_time(t_hat, B, T, Y_ts)

        has_text = note_mask.any(dim=-1)
        text_mask = has_text.unsqueeze(-1).to(Y_ts.dtype)

        # Match the paper's end-to-end fusion: the GRU/gate/attention paths may
        # all backpropagate into the numerical forecasting backbone.
        numeric_source = Y_ts
        local_state = self.local_value_proj(numeric_source.unsqueeze(-1))
        global_state = self.global_state_proj(numeric_source).unsqueeze(2)
        variable_state = self.variable_embedding.view(1, 1, C, self.d_attn)
        time_state = self.time_proj(
            self.time2vec(t_hat.unsqueeze(-1))
        ).unsqueeze(2)
        query = self.query_norm(
            local_state + global_state + variable_state + time_state
        )

        if K == 0:
            context = Y_ts.new_zeros((B, T, C, self.d_attn))
            attention = Y_ts.new_zeros((B, T, C, self.n_heads, 0))
        else:
            note_tokens = self.note_norm(E_txt)
            keys = self.note_key(note_tokens)
            values = self.note_value(note_tokens)

            q_heads = query.reshape(
                B,
                T,
                C,
                self.n_heads,
                self.head_dim,
            )
            k_heads = keys.reshape(
                B,
                T,
                K,
                self.n_heads,
                self.head_dim,
            )
            v_heads = values.reshape(
                B,
                T,
                K,
                self.n_heads,
                self.head_dim,
            )

            scores = torch.einsum(
                "btchd,btkhd->btchk",
                q_heads.float(),
                k_heads.float(),
            ) / math.sqrt(self.head_dim)
            valid = note_mask[:, :, None, None, :]
            attention = torch.softmax(
                scores.masked_fill(~valid, -1e4),
                dim=-1,
            )
            attention = attention * valid.to(attention.dtype)
            attention = attention / attention.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
            self._require_finite("note attention", attention)

            context_weights = self.attention_dropout(attention).to(
                v_heads.dtype
            )
            context_heads = torch.einsum(
                "btchk,btkhd->btchd",
                context_weights,
                v_heads,
            )
            context = context_heads.reshape(B, T, C, self.d_attn)
            context = self.context_norm(self.context_out(context))

        # Paper MMF_GR_Add prediction fusion.  Flattening only the variable
        # axis preserves each variable's independently attended text context
        # while allowing the GRU to model future-time and cross-variable
        # dependencies jointly.
        fusion_input = torch.cat(
            [Y_ts, context.reshape(B, T, C * self.d_attn)],
            dim=-1,
        )
        hidden, _ = self.gru(fusion_input)
        delta_y = self.residual_head(hidden)
        delta_norm = self.layer_norm(delta_y)
        delta_drop = self.residual_dropout(delta_norm)

        # Keep the same gate orientation as MMF_GR_Add: sigmoid(gate_logits)
        # is the base-preservation gate and 1-g is the effective text gate.
        base_gate = torch.sigmoid(self.gate_net(fusion_input))
        base_gate = torch.where(
            text_mask.to(torch.bool),
            base_gate,
            torch.ones_like(base_gate),
        )
        gate = 1.0 - base_gate
        candidate_correction = delta_drop * text_mask
        correction = gate * candidate_correction
        Y_out = Y_ts + correction
        self._require_finite("fused forecast", Y_out)

        note_attention = attention.mean(dim=3).to(Y_ts.dtype)
        zeros = torch.zeros_like(correction)
        self.last_slot_attention = note_attention.detach()
        self.last_note_attention = note_attention.detach()
        self.last_null_probability = zeros.detach()
        self.last_gate = gate.detach()
        self.last_variable_relevance = gate.detach()
        self.last_delta = delta_drop.detach()
        self.last_correction = correction.detach()
        self.last_candidate_correction = candidate_correction.detach()
        self.last_direct_correction = correction.detach()
        self.last_indirect_correction = zeros.detach()
        self.last_text_mask = text_mask.detach()
        self.last_context = context.detach()
        self.last_active_slot_mask = note_mask.detach()
        self.last_hard_slot_choice = None
        self.last_variable_adjacency = Y_ts.new_zeros(C, C)
        self.last_gate_warmup_active = False

        return Y_out
