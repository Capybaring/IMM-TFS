# BUILD_ID: online-spherical-semantic-slots-v12-20260829
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusions.load_llm import embed_notes, get_d_model, load_llm


class TTF_SemTime_Slots(nn.Module):
    """Online semantic clustering followed by strict slot-local aggregation.

    Slot prototypes are initialized from training-note embeddings with
    deterministic spherical farthest-point seeding, then updated by an EMA of
    their assigned notes.  Routing is therefore learned from text semantics
    without adding an auxiliary loss to the forecasting objective.  Every
    accepted note belongs to exactly one slot; low-confidence notes can be
    rejected instead of being forced into an unrelated class.  Prototypes are
    frozen automatically in validation and test mode.

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
        prototype_momentum: float = 0.95,
        routing_warmup_steps: int = 100,
        min_slot_similarity: float = 0.0,
        min_slot_margin: float = 0.02,
        dead_slot_patience: int = 100,
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
        if not 0.0 <= prototype_momentum < 1.0:
            raise ValueError("prototype_momentum must be in [0, 1)")
        if routing_warmup_steps < 0:
            raise ValueError("routing_warmup_steps must be >= 0")
        if not -1.0 <= min_slot_similarity <= 1.0:
            raise ValueError("min_slot_similarity must be in [-1, 1]")
        if not 0.0 <= min_slot_margin <= 2.0:
            raise ValueError("min_slot_margin must be in [0, 2]")
        if dead_slot_patience < 1:
            raise ValueError("dead_slot_patience must be >= 1")

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
        self.prototype_momentum = float(prototype_momentum)
        self.routing_warmup_steps = int(routing_warmup_steps)
        self.min_slot_similarity = float(min_slot_similarity)
        self.min_slot_margin = float(min_slot_margin)
        self.dead_slot_patience = int(dead_slot_patience)

        self.input_proj = (
            nn.Linear(d_model, self.d_txt)
            if d_model != self.d_txt
            else nn.Identity()
        )
        self.value_proj = nn.Linear(self.d_txt, self.slot_dim, bias=False)
        self.consistency_proj = nn.Linear(self.d_txt, self.d_txt, bias=False)

        # The fallback is orthogonal only so a freshly constructed module can
        # run in eval mode.  The first training batch replaces it with
        # data-driven prototypes selected from the original LLM embedding
        # space.  Keeping routing in that stable space prevents the forecast
        # objective from warping all semantic keys toward one winning slot.
        initial_prototypes = torch.empty(self.semantic_slots, d_model)
        nn.init.orthogonal_(initial_prototypes)
        self.register_buffer(
            "slot_prototypes",
            F.normalize(initial_prototypes.float(), dim=-1),
        )
        self.register_buffer(
            "slot_prototypes_initialized",
            torch.tensor(False, dtype=torch.bool),
        )
        self.register_buffer(
            "slot_usage_ema",
            torch.zeros(self.semantic_slots, dtype=torch.float32),
        )
        self.register_buffer(
            "slot_idle_steps",
            torch.zeros(self.semantic_slots, dtype=torch.long),
        )
        self.register_buffer(
            "routing_steps",
            torch.zeros((), dtype=torch.long),
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
        self.last_soft_slot_assignment = None
        self.last_slot_confidence = None
        self.last_slot_margin = None
        self.last_rejection_rate = None
        self.last_slot_usage_ema = None
        self.last_prototype_similarity = None

    @staticmethod
    def _unit_normalize(value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def _initialize_prototypes(
        self,
        routing_keys: torch.Tensor,
        note_mask: torch.Tensor,
    ) -> None:
        """Seed distinct global slots from real training-note embeddings."""
        valid_keys = routing_keys[note_mask]
        if valid_keys.numel() == 0:
            return

        valid_keys = self._unit_normalize(valid_keys)
        n_select = min(self.semantic_slots, valid_keys.shape[0])
        selected = torch.zeros(
            valid_keys.shape[0],
            dtype=torch.bool,
            device=valid_keys.device,
        )

        # Start from the note nearest the global semantic centre, then choose
        # each next note farthest from every already selected prototype.
        global_centre = self._unit_normalize(
            valid_keys.mean(dim=0, keepdim=True)
        ).squeeze(0)
        first_idx = torch.mv(valid_keys, global_centre).argmax()
        chosen = [valid_keys[first_idx]]
        selected[first_idx] = True

        while len(chosen) < n_select:
            chosen_matrix = torch.stack(chosen, dim=0)
            nearest_similarity = (
                valid_keys @ chosen_matrix.transpose(0, 1)
            ).max(dim=1).values
            nearest_similarity = nearest_similarity.masked_fill(
                selected, float("inf")
            )
            next_idx = nearest_similarity.argmin()
            chosen.append(valid_keys[next_idx])
            selected[next_idx] = True

        initialized = self.slot_prototypes.clone()
        initialized[:n_select] = torch.stack(chosen, dim=0)
        self.slot_prototypes.copy_(self._unit_normalize(initialized))
        self.slot_prototypes_initialized.fill_(True)

    @torch.no_grad()
    def _update_prototypes(
        self,
        routing_keys: torch.Tensor,
        note_mask: torch.Tensor,
        winning_slot: torch.Tensor,
    ) -> None:
        """EMA-update occupied slots and revive slots unused for many steps."""
        valid_keys = routing_keys[note_mask]
        valid_winners = winning_slot[note_mask]
        if valid_keys.numel() == 0:
            return

        counts = torch.bincount(
            valid_winners,
            minlength=self.semantic_slots,
        ).to(torch.float32)
        batch_usage = counts / counts.sum().clamp_min(1.0)
        self.slot_usage_ema.mul_(self.prototype_momentum).add_(
            batch_usage,
            alpha=1.0 - self.prototype_momentum,
        )
        self.slot_idle_steps.add_(1)

        for slot_idx in range(self.semantic_slots):
            assigned = valid_keys[valid_winners == slot_idx]
            if assigned.numel() == 0:
                continue
            batch_centre = self._unit_normalize(
                assigned.mean(dim=0, keepdim=True)
            ).squeeze(0)
            updated = (
                self.prototype_momentum * self.slot_prototypes[slot_idx]
                + (1.0 - self.prototype_momentum) * batch_centre
            )
            self.slot_prototypes[slot_idx].copy_(
                self._unit_normalize(updated.unsqueeze(0)).squeeze(0)
            )
            self.slot_idle_steps[slot_idx] = 0

        # A rare class may be absent from many individual mini-batches, so a
        # slot is revived only after a long completely idle period.  The new
        # centre is the note least represented by all current centres.  This
        # prevents permanent dead slots without imposing equal class sizes.
        dead_slots = torch.nonzero(
            self.slot_idle_steps >= self.dead_slot_patience,
            as_tuple=False,
        ).flatten()
        for slot_idx in dead_slots.tolist():
            similarities = valid_keys @ self.slot_prototypes.transpose(0, 1)
            least_represented = similarities.max(dim=1).values.argmin()
            self.slot_prototypes[slot_idx].copy_(valid_keys[least_represented])
            self.slot_idle_steps[slot_idx] = 0

        self.routing_steps.add_(1)

    @staticmethod
    def _masked_softmax(logits, valid_mask, dim=-1):
        valid_mask = valid_mask.to(torch.bool)
        masked_logits = logits.masked_fill(~valid_mask, -1e4)
        weights = torch.softmax(masked_logits, dim=dim)
        weights = weights * valid_mask.to(weights.dtype)
        return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _bounded_normalize(value: torch.Tensor) -> torch.Tensor:
        """Normalize directions while bounding the inverse-norm gradient."""
        value = value.float()
        return value / value.norm(dim=-1, keepdim=True).clamp_min(0.1)

    @staticmethod
    def _require_finite(name: str, value: torch.Tensor) -> None:
        if not torch.isfinite(value).all():
            raise FloatingPointError(
                f"TTF_SemTime_Slots produced NaN or Inf in {name}"
            )

    def _slot_consistency(self, semantic_weights, semantic_vectors, note_mask):
        # Routing statistics stay in float32 even when the surrounding model
        # uses AMP; tiny slot masses must not underflow before division.
        normalized = self._bounded_normalize(semantic_vectors)
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
        if not torch.isfinite(t_hat).all():
            raise ValueError("Future timestamps contain NaN or Inf values")
        if note_mask.any() and not torch.isfinite(tau[note_mask]).all():
            raise ValueError("Real-note timestamps contain NaN or Inf values")
        # Padding timestamps carry no information.  Clearing them prevents a
        # non-finite padding value from entering recency arithmetic before the
        # note mask is applied.
        tau = torch.where(note_mask, tau, torch.zeros_like(tau))

        T_f = t_hat.shape[1]
        if K == 0:
            M_txt = note_mask.any(dim=1, keepdim=True)
            self.last_slot_assignment = None
            self.last_semantic_weights = None
            self.last_slot_mass = None
            self.last_slot_outputs = None
            self.last_soft_slot_assignment = None
            return V.new_zeros((B, T_f, self.d_txt)), M_txt

        # Routing uses the original, stable LLM embedding space.  It is kept
        # independent from the trainable value projection so the forecasting
        # objective cannot collapse all semantic keys into one direction.
        routing_keys = self._unit_normalize(V.detach())
        self._require_finite("semantic routing keys", routing_keys)
        if self.training and not bool(self.slot_prototypes_initialized.item()):
            self._initialize_prototypes(routing_keys, note_mask)

        slot_prototypes = self._unit_normalize(self.slot_prototypes)
        self._require_finite("semantic slot prototypes", slot_prototypes)
        slot_logits = torch.einsum(
            "hd,bkd->bhk", slot_prototypes, routing_keys
        )
        self._require_finite("semantic slot similarities", slot_logits)

        soft_assignment = torch.softmax(
            slot_logits / self.assignment_temperature,
            dim=1,
        )
        self._require_finite("soft slot assignment", soft_assignment)
        winning_slot = slot_logits.argmax(dim=1)

        top_two = slot_logits.topk(k=2, dim=1).values
        slot_confidence = top_two[:, 0]
        slot_margin = top_two[:, 0] - top_two[:, 1]
        threshold_active = (
            bool(self.slot_prototypes_initialized.item())
            and int(self.routing_steps.item()) >= self.routing_warmup_steps
        )
        if threshold_active:
            accepted_note_mask = (
                note_mask
                & (slot_confidence >= self.min_slot_similarity)
                & (slot_margin >= self.min_slot_margin)
            )
        else:
            accepted_note_mask = note_mask

        hard_assignment = F.one_hot(
            winning_slot,
            num_classes=self.semantic_slots,
        ).permute(0, 2, 1).to(soft_assignment.dtype)
        hard_assignment = (
            hard_assignment * accepted_note_mask[:, None, :].float()
        )
        # Classification is deliberately strict: a note is either owned by
        # one slot or rejected.  Semantic-centre learning happens through the
        # online EMA update rather than an indirect forecasting-loss gradient.
        slot_assignment = hard_assignment
        slot_note_mask = hard_assignment.to(torch.bool)
        M_txt = accepted_note_mask.any(dim=1, keepdim=True)

        if self.training:
            self._update_prototypes(
                routing_keys,
                note_mask,
                winning_slot,
            )

        V = self.input_proj(V)
        self._require_finite("projected text embeddings", V)
        V = V * note_mask.unsqueeze(-1).to(V.dtype)

        shared_values = self.value_proj(V)
        values = shared_values.unsqueeze(1).expand(
            B, self.semantic_slots, K, self.slot_dim
        )

        raw_slot_mass = slot_assignment.sum(dim=-1)
        # A hard classifier naturally creates empty slots for patients with
        # fewer note categories than H.  Dividing a straight-through tensor by
        # 1e-8 amplified its backward gradient by 1e8 and caused the observed
        # NaNs.  Forward hard counts are exact integers, so one is the correct
        # stable denominator for an empty slot.
        hard_slot_count = hard_assignment.sum(dim=-1)
        stable_slot_count = hard_slot_count.clamp_min(1.0)
        valid_count = accepted_note_mask.sum(dim=-1, keepdim=True).float()
        slot_mass = raw_slot_mass / valid_count.clamp_min(1.0)

        # Within each slot, normalize only over its assigned real notes.
        semantic_weights = slot_assignment / stable_slot_count.unsqueeze(-1)

        consistency_vectors = self.consistency_proj(V)
        consistency = self._slot_consistency(
            semantic_weights, consistency_vectors, accepted_note_mask
        )
        self._require_finite("slot consistency", consistency)
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
        self._require_finite("semantic time gate", time_gate)

        delta = (
            t_hat[:, :, None].float() - tau[:, None, :].float()
        ).clamp_min(0.0)
        # Bound the logarithm before exp().  clamp_min after exp() cannot stop
        # overflow when the learned log-sigma becomes large.
        sigma = self.log_recency_sigma.float().clamp(
            min=math.log(1e-3),
            max=math.log(1e3),
        ).exp()
        time_logits = -(
            delta[:, None, :, :] / sigma[None, :, None, None]
        ).square()
        time_logits = time_logits.clamp(min=-80.0, max=0.0)
        self._require_finite("temporal logits", time_logits)
        temporal_kernel = time_logits.exp()

        # Only notes classified into a slot can participate in that slot's
        # temporal aggregation.  This mask is the strict separation that the
        # previous soft implementation was missing.
        base_logits = slot_assignment.clamp_min(1e-8).log()
        fused_logits = (
            base_logits[:, :, None, :]
            + time_gate[:, :, None, None] * time_logits
        )
        fused_mask = slot_note_mask[:, :, None, :].expand(
            B, self.semantic_slots, T_f, K
        )
        fused_weights = self._masked_softmax(fused_logits, fused_mask, dim=-1)
        self._require_finite("slot-local temporal weights", fused_weights)

        slot_outputs = torch.einsum(
            "bhtk,bhkd->bhtd", fused_weights.to(values.dtype), values
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
        self._require_finite("slot output heads", slot_outputs)

        # Absolute evidence strength does not normalize away with K=1.  It is
        # the assignment-weighted Gaussian age of the reports owned by each
        # slot.  This is deliberately separate from ``fused_weights``:
        # fused_weights selects *which* report to use, while this term controls
        # how much stale text should affect the forecast at all.
        recency_numer = (
            slot_assignment[:, :, None, :] * temporal_kernel
        ).sum(dim=-1)
        recency_denom = stable_slot_count[:, :, None]
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
        self._require_finite("final text representation", E_txt)

        self.last_slot_assignment = slot_assignment.detach()
        self.last_soft_slot_assignment = soft_assignment.detach()
        self.last_semantic_weights = semantic_weights.detach()
        self.last_slot_mass = slot_mass.detach()
        self.last_slot_consistency = consistency.detach()
        self.last_time_gate = time_gate.detach()
        self.last_fused_weights = fused_weights.detach()
        self.last_absolute_recency_strength = (
            absolute_recency_strength.detach()
        )
        self.last_slot_outputs = slot_outputs.detach()
        self.last_slot_confidence = slot_confidence.detach()
        self.last_slot_margin = slot_margin.detach()
        real_note_count = note_mask.sum().clamp_min(1)
        self.last_rejection_rate = (
            (note_mask & ~accepted_note_mask).sum().float()
            / real_note_count.float()
        ).detach()
        self.last_slot_usage_ema = self.slot_usage_ema.detach().clone()
        prototype_similarity = (
            slot_prototypes @ slot_prototypes.transpose(0, 1)
        )
        self.last_prototype_similarity = prototype_similarity.detach()

        return E_txt, M_txt
