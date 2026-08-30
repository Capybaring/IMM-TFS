# BUILD_ID: note-preserving-time-alignment-v1-20260830
import torch
import torch.nn as nn

from fusions.load_llm import embed_notes, get_d_model, load_llm


class Time2Vec(nn.Module):
    """Learnable linear and periodic encoding for a scalar timestamp."""

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


class TTF_SemTime_Slots(nn.Module):
    """Encode and time-align notes without aggregating them.

    The legacy class and file names are intentionally preserved so existing
    experiment scripts continue to work.  Unlike the former semantic-slot
    implementation, this module keeps every real note as a separate token.
    For every future query time it combines:

      1. the projected note embedding;
      2. the note's original normalized timestamp; and
      3. the normalized non-negative distance to that future query time.

    Inputs use the common normalized time scale produced by the dataset
    collator.  No recency decay, semantic routing, attention, or text
    aggregation is performed here.  Variable-specific selection is deferred
    to MMF_VarTime_SlotGate.

    Shapes:
        notes_input: (B, K, d_model), or raw nested note strings
        tau:         (B, K)
        t_hat:       (B, T) or (T,)
        E_txt:       (B, T, K, d_txt)
        M_txt:       (B, T, K)
    """

    def __init__(
        self,
        llm_model_fusion: str,
        llm_layers_fusion: int,
        max_length: int = 1024,
        device: str = "cpu",
        use_text_embeddings: bool = True,
        n_heads_fusion: int = 1,
        dropout: float = 0.1,
        d_txt: int | None = 768,
        semantic_slots: int = 4,
        recency_sigma: float = 1.0,
        time_gate_bias: float = -1.0,
        assignment_temperature: float = 0.7,
        absolute_recency_floor: float = 0.1,
        prototype_momentum: float = 0.95,
        routing_warmup_steps: int = 100,
        min_slot_similarity: float = 0.0,
        min_slot_margin: float = 0.02,
        dead_slot_patience: int = 100,
    ):
        super().__init__()

        # These arguments remain accepted for command-line/checkpoint
        # compatibility.  They no longer affect the note-preserving encoder.
        self.legacy_semantic_slots = int(semantic_slots)
        self.legacy_recency_sigma = float(recency_sigma)
        self.legacy_time_gate_bias = float(time_gate_bias)
        del (
            n_heads_fusion,
            assignment_temperature,
            absolute_recency_floor,
            prototype_momentum,
            routing_warmup_steps,
            min_slot_similarity,
            min_slot_margin,
            dead_slot_patience,
        )

        self.use_text_embeddings = bool(use_text_embeddings)
        if not self.use_text_embeddings:
            self.tokenizer, self.llm_model = load_llm(
                llm_model_fusion,
                llm_layers_fusion,
                device,
            )

        d_model = int(get_d_model(llm_model_fusion))
        self.d_txt = int(d_txt) if d_txt is not None else d_model
        self.max_length = int(max_length)
        self.d_time = max(min(self.d_txt // 4, 32), 4)

        self.semantic_proj = nn.Linear(d_model, self.d_txt)
        self.note_time2vec = Time2Vec(self.d_time)
        self.delta_time2vec = Time2Vec(self.d_time)
        self.output_proj = nn.Linear(
            self.d_txt + 2 * self.d_time,
            self.d_txt,
        )
        self.output_norm = nn.LayerNorm(self.d_txt)
        self.dropout = nn.Dropout(dropout)

        # New note-level diagnostics.
        self.last_note_mask = None
        self.last_aligned_note_mask = None
        self.last_note_time = None
        self.last_relative_time = None

        # Compatibility-only legacy diagnostics.  The evaluation code skips
        # them when they are None.
        self.last_slot_assignment = None
        self.last_semantic_weights = None
        self.last_slot_mass = None
        self.last_slot_consistency = None
        self.last_time_gate = None
        self.last_fused_weights = None
        self.last_absolute_recency_strength = None
        self.last_slot_outputs = None
        self.last_soft_slot_assignment = None
        self.last_slot_confidence = None
        self.last_slot_margin = None
        self.last_rejection_rate = None
        self.last_slot_usage_ema = None
        self.last_prototype_similarity = None

    @staticmethod
    def _require_finite(name: str, value: torch.Tensor) -> None:
        if not torch.isfinite(value).all():
            raise FloatingPointError(
                f"TTF_SemTime_Slots produced NaN or Inf in {name}"
            )

    def forward(
        self,
        notes_input,
        tau: torch.Tensor,
        t_hat: torch.Tensor,
    ):
        if self.use_text_embeddings:
            V = notes_input
            if not torch.is_tensor(V):
                raise TypeError(
                    "notes_input must be a Tensor when "
                    "use_text_embeddings=True"
                )
            note_mask = V.abs().sum(dim=-1) > 0
        else:
            V, note_mask = embed_notes(
                notes_input,
                self.tokenizer,
                self.llm_model,
                max_length=self.max_length,
            )

        if V.ndim != 3:
            raise ValueError(
                "notes_input must produce (B, K, d_model), "
                f"got {tuple(V.shape)}"
            )
        self._require_finite("input text embeddings", V)

        B, K, _ = V.shape
        if tau.ndim != 2 or tau.shape != (B, K):
            raise ValueError(
                f"Expected tau shape {(B, K)}, got {tuple(tau.shape)}"
            )
        if t_hat.dim() == 1:
            t_hat = t_hat.unsqueeze(0).expand(B, -1)
        elif t_hat.ndim != 2 or t_hat.shape[0] != B:
            raise ValueError(
                "Expected t_hat shape (B, T) or (T,), "
                f"got {tuple(t_hat.shape)}"
            )

        tau = tau.to(device=V.device, dtype=V.dtype)
        t_hat = t_hat.to(device=V.device, dtype=V.dtype)
        self._require_finite("future timestamps", t_hat)
        if note_mask.any():
            self._require_finite("real-note timestamps", tau[note_mask])

        # Padding timestamps have no meaning and must not enter Time2Vec.
        tau = torch.where(note_mask, tau, torch.zeros_like(tau))
        T = t_hat.shape[1]

        if K == 0:
            aligned_mask = torch.zeros(
                (B, T, 0),
                dtype=torch.bool,
                device=V.device,
            )
            E_txt = V.new_zeros((B, T, 0, self.d_txt))
            self.last_note_mask = note_mask.detach()
            self.last_aligned_note_mask = aligned_mask.detach()
            self.last_note_time = tau.detach()
            self.last_relative_time = tau.new_zeros((B, T, 0))
            return E_txt, aligned_mask

        # A note is visible only after it has occurred.  Expanded MIMIC
        # normally supplies historical notes and future query times, so this
        # mask is usually identical to the real-note mask but keeps the module
        # causally correct for other alignments.
        aligned_mask = (
            note_mask[:, None, :]
            & (tau[:, None, :] <= t_hat[:, :, None] + 1e-6)
        )
        delta = (
            t_hat[:, :, None].float() - tau[:, None, :].float()
        ).clamp_min(0.0).to(V.dtype)

        semantic = self.semantic_proj(V)
        note_time = self.note_time2vec(tau.unsqueeze(-1))
        relative_time = self.delta_time2vec(delta.unsqueeze(-1))

        semantic = semantic[:, None, :, :].expand(-1, T, -1, -1)
        note_time = note_time[:, None, :, :].expand(-1, T, -1, -1)
        aligned_features = torch.cat(
            [semantic, note_time, relative_time],
            dim=-1,
        )

        E_txt = self.output_norm(self.output_proj(aligned_features))
        E_txt = self.dropout(E_txt)
        E_txt = E_txt * aligned_mask.unsqueeze(-1).to(E_txt.dtype)
        self._require_finite("aligned note tokens", E_txt)

        self.last_note_mask = note_mask.detach()
        self.last_aligned_note_mask = aligned_mask.detach()
        self.last_note_time = tau.detach()
        self.last_relative_time = delta.detach()

        return E_txt, aligned_mask
