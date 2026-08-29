import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusions.load_llm import embed_notes, get_d_model, load_llm


class TTF_SemTime_Slots(nn.Module):
    """Competitive semantic-slot aggregation with adaptive recency.

    Compared with independent ``softmax-over-notes`` slots, every real note is
    first assigned *across slots*.  This creates actual competition between
    slots and makes ``slot_mass`` meaningful.  Each slot then aggregates its
    assigned notes over future time without a global projection that would mix
    slot boundaries before MMF selection.

    Relative note weights alone cannot make a single report become stale: a
    softmax over one report is always one.  This implementation therefore also
    computes an *absolute* recency strength for every slot/future time and
    multiplies it into the slot output.  The configurable floor keeps old but
    potentially useful clinical context from being erased completely.

    Public interface:
        E_txt, M_txt = module(notes_input, tau, t_hat)
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
    ):
        super().__init__()
        del n_heads_fusion

        if semantic_slots < 2:
            raise ValueError(
                "TTF_SemTime_Slots requires semantic_slots >= 2; "
                "one slot collapses semantic routing to a constant"
            )
        if recency_sigma <= 0:
            raise ValueError("recency_sigma must be > 0")
        if assignment_temperature <= 0:
            raise ValueError("assignment_temperature must be > 0")
        if not 0.0 <= absolute_recency_floor <= 1.0:
            raise ValueError("absolute_recency_floor must be in [0, 1]")

        self.use_text_embeddings = use_text_embeddings
        if not use_text_embeddings:
            self.tokenizer, self.llm_model = load_llm(
                llm_model_fusion, llm_layers_fusion, device
            )

        d_model = get_d_model(llm_model_fusion)
        self.d_txt = int(d_txt) if d_txt is not None else int(d_model)
        self.semantic_slots = int(semantic_slots)
        if self.d_txt % self.semantic_slots != 0:
            raise ValueError(
                f"d_txt={self.d_txt} must be divisible by "
                f"semantic_slots={self.semantic_slots}"
            )
        self.slot_dim = self.d_txt // self.semantic_slots
        self.max_length = int(max_length)
        self.assignment_temperature = float(assignment_temperature)
        self.absolute_recency_floor = float(absolute_recency_floor)

        self.input_proj = (
            nn.Linear(d_model, self.d_txt)
            if d_model != self.d_txt
            else nn.Identity()
        )
        self.key_proj = nn.Linear(self.d_txt, self.d_txt, bias=False)
        self.value_proj = nn.Linear(self.d_txt, self.d_txt, bias=False)
        self.consistency_proj = nn.Linear(self.d_txt, self.d_txt, bias=False)

        self.slot_queries = nn.Parameter(
            torch.randn(self.semantic_slots, self.slot_dim) * 0.1
        )
        self.log_recency_sigma = nn.Parameter(
            torch.full(
                (self.semantic_slots,),
                math.log(float(recency_sigma)),
                dtype=torch.float32,
            )
        )

        self.time_gate = nn.Sequential(
            nn.Linear(3, self.semantic_slots),
            nn.GELU(),
            nn.Linear(self.semantic_slots, 1),
        )
        nn.init.constant_(self.time_gate[-1].bias, float(time_gate_bias))

        # Slot-specific output heads break the permutation-symmetric state in
        # which every slot learns the same transformation.
        self.slot_output_norms = nn.ModuleList(
            [nn.LayerNorm(self.slot_dim) for _ in range(self.semantic_slots)]
        )
        self.slot_output_projs = nn.ModuleList(
            [
                nn.Linear(self.slot_dim, self.slot_dim)
                for _ in range(self.semantic_slots)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        self.last_slot_assignment = None
        self.last_semantic_weights = None
        self.last_slot_mass = None
        self.last_slot_consistency = None
        self.last_time_gate = None
        self.last_fused_weights = None
        self.last_absolute_recency_strength = None
        self.last_slot_outputs = None
        self._routing_assignment = None
        self._routing_note_mask = None

    @staticmethod
    def _masked_softmax(logits, valid_mask, dim=-1):
        valid_mask = valid_mask.to(torch.bool)
        masked_logits = logits.masked_fill(~valid_mask, -1e4)
        weights = torch.softmax(masked_logits, dim=dim)
        weights = weights * valid_mask.to(weights.dtype)
        return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)

    def _slot_consistency(self, semantic_weights, semantic_vectors, note_mask):
        normalized = F.normalize(semantic_vectors, p=2, dim=-1, eps=1e-8)
        pairwise = torch.einsum("bkd,bjd->bkj", normalized, normalized)

        valid_pairs = (
            note_mask[:, :, None] & note_mask[:, None, :]
        ).to(pairwise.dtype)
        eye = torch.eye(
            note_mask.shape[1], device=note_mask.device, dtype=pairwise.dtype
        ).unsqueeze(0)
        valid_pairs = valid_pairs * (1.0 - eye)

        pair_weights = torch.einsum(
            "bhk,bhj->bhkj", semantic_weights, semantic_weights
        )
        denom = (pair_weights * valid_pairs[:, None]).sum(dim=(-1, -2))
        numer = (
            pair_weights * pairwise[:, None] * valid_pairs[:, None]
        ).sum(dim=(-1, -2))

        consistency = torch.where(
            denom > 1e-8,
            numer / denom.clamp_min(1e-8),
            torch.ones_like(denom),
        )
        return consistency.clamp(0.0, 1.0)

    def routing_loss(self) -> torch.Tensor:
        """Encourage sharp per-note routing without collapsing one slot.

        Entropy makes each note prefer one semantic slot.  The batch-level
        balance term prevents the trivial solution in which every note picks
        the same slot.  Both terms use only real notes.
        """
        if self._routing_assignment is None or self._routing_note_mask is None:
            return self.slot_queries.sum() * 0.0

        assignment = self._routing_assignment
        valid_notes = self._routing_note_mask
        valid_count = valid_notes.sum()
        if valid_count == 0:
            return assignment.sum() * 0.0

        entropy = -(
            assignment * assignment.clamp_min(1e-8).log()
        ).sum(dim=1)
        sharpness_loss = entropy[valid_notes].mean() / math.log(
            self.semantic_slots
        )

        usage = assignment.sum(dim=(0, 2)) / valid_count.to(assignment.dtype)
        uniform = torch.full_like(usage, 1.0 / self.semantic_slots)
        balance_loss = self.semantic_slots * (usage - uniform).square().sum()
        return sharpness_loss + 0.5 * balance_loss

    def forward(self, notes_input, tau: torch.Tensor, t_hat: torch.Tensor):
        if self.use_text_embeddings:
            V = notes_input
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
        if not torch.isfinite(V).all():
            raise ValueError("Input text embeddings contain NaN or Inf values")

        B, K, _ = V.shape
        if tau.ndim != 2 or tau.shape != (B, K):
            raise ValueError(f"Expected tau shape {(B, K)}, got {tuple(tau.shape)}")
        if t_hat.dim() == 1:
            t_hat = t_hat.unsqueeze(0).expand(B, -1)
        elif t_hat.ndim != 2 or t_hat.shape[0] != B:
            raise ValueError(
                f"Expected t_hat shape (B, T_f) or (T_f,), got {tuple(t_hat.shape)}"
            )

        tau = tau.to(device=V.device, dtype=V.dtype)
        t_hat = t_hat.to(device=V.device, dtype=V.dtype)

        T_f = t_hat.shape[1]
        M_txt = note_mask.any(dim=1, keepdim=True)
        if K == 0:
            self._routing_assignment = None
            self._routing_note_mask = None
            self.last_slot_assignment = None
            self.last_semantic_weights = None
            self.last_slot_mass = None
            self.last_slot_outputs = None
            return V.new_zeros((B, T_f, self.d_txt)), M_txt

        V = self.input_proj(V)
        V = V * note_mask.unsqueeze(-1).to(V.dtype)

        keys = self.key_proj(V).reshape(
            B, K, self.semantic_slots, self.slot_dim
        )
        values = self.value_proj(V).reshape(
            B, K, self.semantic_slots, self.slot_dim
        ).permute(0, 2, 1, 3)

        # (B, H, K): compatibility between slot h and note k.
        slot_logits = torch.einsum(
            "hd,bkhd->bhk", self.slot_queries, keys
        ) / math.sqrt(self.slot_dim)

        # During training, Gumbel noise breaks the exactly-uniform stationary
        # point of entropy regularization.  Evaluation remains deterministic.
        if self.training:
            slot_assignment = F.gumbel_softmax(
                slot_logits,
                tau=self.assignment_temperature,
                hard=False,
                dim=1,
            )
        else:
            slot_assignment = torch.softmax(
                slot_logits / self.assignment_temperature,
                dim=1,
            )
        slot_assignment = slot_assignment * note_mask[:, None, :].to(V.dtype)
        if self.training:
            self._routing_assignment = slot_assignment
            self._routing_note_mask = note_mask
        else:
            self._routing_assignment = None
            self._routing_note_mask = None

        raw_slot_mass = slot_assignment.sum(dim=-1)
        valid_count = note_mask.sum(dim=-1, keepdim=True).to(V.dtype)
        slot_mass = raw_slot_mass / valid_count.clamp_min(1.0)

        # Within each slot, normalize only over its assigned real notes.
        semantic_weights = slot_assignment / raw_slot_mass.unsqueeze(-1).clamp_min(
            1e-8
        )

        consistency_vectors = self.consistency_proj(V)
        consistency = self._slot_consistency(
            semantic_weights, consistency_vectors, note_mask
        )
        disagreement = 1.0 - consistency

        entropy = -(
            semantic_weights * semantic_weights.clamp_min(1e-8).log()
        ).sum(dim=-1)
        entropy_scale = valid_count.clamp_min(2.0).log()
        entropy = torch.where(
            valid_count > 1,
            entropy / entropy_scale,
            torch.zeros_like(entropy),
        )

        gate_features = torch.stack(
            [consistency, disagreement, entropy], dim=-1
        )
        learned_gate = torch.sigmoid(self.time_gate(gate_features).squeeze(-1))
        time_gate = disagreement * learned_gate * slot_mass

        delta = (t_hat[:, :, None] - tau[:, None, :]).clamp_min(0.0)
        sigma = self.log_recency_sigma.exp().clamp_min(1e-4)
        time_logits = -(
            delta[:, None, :, :] / sigma[None, :, None, None]
        ).square()
        temporal_kernel = time_logits.exp()

        # log assignment retains cross-slot competition while softmax over K
        # creates a valid note distribution for each slot and future time.
        base_logits = slot_assignment.clamp_min(1e-8).log()
        fused_logits = (
            base_logits[:, :, None, :]
            + time_gate[:, :, None, None] * time_logits
        )
        fused_mask = note_mask[:, None, None, :].expand(
            B, self.semantic_slots, T_f, K
        )
        fused_weights = self._masked_softmax(fused_logits, fused_mask, dim=-1)

        slot_outputs = torch.einsum(
            "bhtk,bhkd->bhtd", fused_weights, values
        )
        slot_outputs = torch.stack(
            [
                self.slot_output_projs[slot_idx](
                    self.dropout(
                        self.slot_output_norms[slot_idx](
                            slot_outputs[:, slot_idx]
                        )
                    )
                )
                for slot_idx in range(self.semantic_slots)
            ],
            dim=1,
        )

        # Absolute evidence strength does not normalize away with K=1.  It is
        # the assignment-weighted Gaussian age of the reports owned by each
        # slot.  This is deliberately separate from ``fused_weights``:
        # fused_weights selects *which* report to use, while this term controls
        # how much stale text should affect the forecast at all.
        recency_numer = (
            slot_assignment[:, :, None, :] * temporal_kernel
        ).sum(dim=-1)
        recency_denom = raw_slot_mass[:, :, None].clamp_min(1e-8)
        absolute_recency_strength = recency_numer / recency_denom
        absolute_recency_strength = (
            self.absolute_recency_floor
            + (1.0 - self.absolute_recency_floor)
            * absolute_recency_strength
        )
        absolute_recency_strength = absolute_recency_strength.clamp(0.0, 1.0)
        slot_outputs = (
            slot_outputs * absolute_recency_strength.unsqueeze(-1)
        )

        # Preserve the actual fraction of text evidence owned by each slot.
        # Multiplying by H here would turn a soft [1/H, ..., 1/H] assignment
        # back into H full-strength copies of the same note.
        slot_strength = slot_mass.clamp(0.0, 1.0)
        slot_outputs = slot_outputs * slot_strength[:, :, None, None]

        E_txt = slot_outputs.permute(0, 2, 1, 3).reshape(B, T_f, self.d_txt)
        E_txt = E_txt * M_txt[:, :, None].to(E_txt.dtype)

        self.last_slot_assignment = slot_assignment.detach()
        self.last_semantic_weights = semantic_weights.detach()
        self.last_slot_mass = slot_mass.detach()
        self.last_slot_consistency = consistency.detach()
        self.last_time_gate = time_gate.detach()
        self.last_fused_weights = fused_weights.detach()
        self.last_absolute_recency_strength = (
            absolute_recency_strength.detach()
        )
        self.last_slot_outputs = slot_outputs.detach()

        return E_txt, M_txt
