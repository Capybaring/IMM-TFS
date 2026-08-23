import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusions.load_llm import embed_notes, get_d_model, load_llm


class TTF_SemTime_Slots(nn.Module):
    """Semantic-slot text aggregation with adaptive temporal decay.

    The public interface matches the existing TTF modules:

        E_txt, M_txt = module(notes_input, tau, t_hat)

    Each learned slot can specialize to a latent clinical topic.  Temporal
    decay is activated according to the semantic disagreement among the notes
    read by that slot.  Similar notes therefore receive little temporal bias,
    while changing descriptions inside one topic favour more recent reports.
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
    ):
        super().__init__()
        del n_heads_fusion  # Kept for constructor compatibility with FusionModel.

        if semantic_slots < 1:
            raise ValueError("semantic_slots must be >= 1")
        if recency_sigma <= 0:
            raise ValueError("recency_sigma must be > 0")

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
        self.max_length = max_length

        self.input_proj = (
            nn.Linear(d_model, self.d_txt) if d_model != self.d_txt else nn.Identity()
        )
        self.key_proj = nn.Linear(self.d_txt, self.d_txt, bias=False)
        self.value_proj = nn.Linear(self.d_txt, self.d_txt, bias=False)
        self.consistency_proj = nn.Linear(self.d_txt, self.d_txt, bias=False)
        self.note_score = nn.Linear(self.d_txt, 1, bias=False)

        self.slot_queries = nn.Parameter(
            torch.randn(self.semantic_slots, self.slot_dim) * 0.02
        )
        self.log_recency_sigma = nn.Parameter(
            torch.full(
                (self.semantic_slots,),
                math.log(float(recency_sigma)),
                dtype=torch.float32,
            )
        )

        # The disagreement factor below already forces the gate to zero when
        # notes are semantically identical.  This MLP learns how strongly the
        # remaining disagreement should activate temporal decay.
        self.time_gate = nn.Sequential(
            nn.Linear(3, self.semantic_slots),
            nn.GELU(),
            nn.Linear(self.semantic_slots, 1),
        )
        nn.init.constant_(self.time_gate[-1].bias, float(time_gate_bias))

        self.output_norm = nn.LayerNorm(self.d_txt)
        self.output_proj = nn.Linear(self.d_txt, self.d_txt)
        self.dropout = nn.Dropout(dropout)

        # Optional diagnostics; populated during forward without changing the
        # existing two-value TTF return signature.
        self.last_semantic_weights = None
        self.last_slot_consistency = None
        self.last_time_gate = None
        self.last_fused_weights = None

    @staticmethod
    def _masked_softmax(logits, valid_mask, dim=-1):
        valid_mask = valid_mask.to(torch.bool)
        masked_logits = logits.masked_fill(~valid_mask, -1e4)
        weights = torch.softmax(masked_logits, dim=dim)
        weights = weights * valid_mask.to(weights.dtype)
        return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)

    def _slot_consistency(self, semantic_weights, semantic_vectors, note_mask):
        """Weighted off-diagonal cosine agreement for each semantic slot."""
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

        # With zero or one valid note, time weighting cannot change a normalized
        # aggregation, so treating the slot as fully consistent is appropriate.
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
                "notes_input must produce a tensor shaped (B, K, d_model), "
                f"got {tuple(V.shape)}"
            )
        if torch.isnan(V).any():
            raise ValueError("Input embeddings V contain NaN values.")

        B, K, _ = V.shape
        if tau.ndim != 2 or tau.shape != (B, K):
            raise ValueError(
                f"Expected tau shape {(B, K)}, got {tuple(tau.shape)}"
            )
        if t_hat.dim() == 1:
            t_hat = t_hat.unsqueeze(0).expand(B, -1)
        elif t_hat.ndim != 2 or t_hat.shape[0] != B:
            raise ValueError(
                f"Expected t_hat shape (B, T_f) or (T_f,), got {tuple(t_hat.shape)}"
            )

        T_f = t_hat.shape[1]
        M_txt = note_mask.any(dim=1, keepdim=True)
        if K == 0:
            E_txt = V.new_zeros((B, T_f, self.d_txt))
            return E_txt, M_txt

        V = self.input_proj(V)
        V = V * note_mask.unsqueeze(-1).to(V.dtype)

        keys = self.key_proj(V).view(
            B, K, self.semantic_slots, self.slot_dim
        )
        values = self.value_proj(V).view(
            B, K, self.semantic_slots, self.slot_dim
        ).permute(0, 2, 1, 3)

        semantic_logits = torch.einsum(
            "hd,bkhd->bhk", self.slot_queries, keys
        ) / math.sqrt(self.slot_dim)
        semantic_logits = semantic_logits + self.note_score(V).squeeze(-1)[:, None]

        semantic_mask = note_mask[:, None, :].expand(
            B, self.semantic_slots, K
        )
        semantic_weights = self._masked_softmax(
            semantic_logits, semantic_mask, dim=-1
        )

        consistency_vectors = self.consistency_proj(V)
        consistency = self._slot_consistency(
            semantic_weights, consistency_vectors, note_mask
        )
        disagreement = 1.0 - consistency

        entropy = -(
            semantic_weights
            * semantic_weights.clamp_min(1e-8).log()
        ).sum(dim=-1)
        valid_count = note_mask.sum(dim=-1, keepdim=True).to(V.dtype)
        entropy_scale = valid_count.clamp_min(2.0).log()
        entropy = torch.where(
            valid_count > 1,
            entropy / entropy_scale,
            torch.zeros_like(entropy),
        )

        slot_mass = semantic_weights.sum(dim=-1).clamp(0.0, 1.0)
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

        fused_logits = (
            semantic_logits[:, :, None, :]
            + time_gate[:, :, None, None] * time_logits
        )
        fused_mask = note_mask[:, None, None, :].expand(
            B, self.semantic_slots, T_f, K
        )
        fused_weights = self._masked_softmax(
            fused_logits, fused_mask, dim=-1
        )

        slot_outputs = torch.einsum(
            "bhtk,bhkd->bhtd", fused_weights, values
        )
        E_raw = slot_outputs.permute(0, 2, 1, 3).reshape(B, T_f, self.d_txt)
        E_txt = self.output_proj(self.dropout(self.output_norm(E_raw)))
        E_txt = E_txt * M_txt[:, :, None].to(E_txt.dtype)

        self.last_semantic_weights = semantic_weights.detach()
        self.last_slot_consistency = consistency.detach()
        self.last_time_gate = time_gate.detach()
        self.last_fused_weights = fused_weights.detach()

        return E_txt, M_txt
