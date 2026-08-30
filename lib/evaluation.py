# BUILD_ID: staged-fusion-diagnostics-v3-20260829
import math

import torch
from tqdm import tqdm


MIMIC_VARIABLE_NAMES = [
    "spo2",
    "respiratory_rate",
    "heart_rate",
    "sbp",
    "dbp",
    "map",
    "temperature_c",
    "ph",
    "pao2",
    "paco2",
    "bicarbonate_bg",
    "base_excess",
    "lactate",
    "minute_ventilation",
    "tidal_volume_observed",
    "wbc",
]


def compute_error(truth, pred_y, mask, func, reduce, norm_dict=None):
    """Masked error with the repository's variable-balanced reduction."""
    if pred_y.ndim == 3:
        pred_y = pred_y.unsqueeze(0)
    n_traj_samples, _, _, n_dim = pred_y.size()
    truth_repeated = truth.repeat(n_traj_samples, 1, 1, 1)
    mask = mask.repeat(n_traj_samples, 1, 1, 1)

    if func == "MSE":
        error = (truth_repeated - pred_y).square() * mask
    elif func == "MAE":
        error = torch.abs(truth_repeated - pred_y) * mask
    elif func == "MAPE":
        if norm_dict is None:
            valid_mask = (truth_repeated != 0) * mask
            denominator = truth_repeated.abs().clamp_min(1e-8)
            error = torch.abs(truth_repeated - pred_y) / denominator * valid_mask
            mask = valid_mask
        else:
            data_max = norm_dict["data_max"]
            data_min = norm_dict["data_min"]
            truth_rescale = truth_repeated * (data_max - data_min) + data_min
            pred_rescale = pred_y * (data_max - data_min) + data_min
            valid_mask = (truth_rescale != 0) * mask
            denominator = truth_rescale.abs().clamp_min(1e-8)
            error = torch.abs(truth_rescale - pred_rescale) / denominator * valid_mask
            mask = valid_mask
    else:
        raise ValueError(f"Unknown error function: {func}")

    error_var_sum = error.reshape(-1, n_dim).sum(dim=0)
    mask_count = mask.reshape(-1, n_dim).sum(dim=0)

    if reduce == "sum":
        return error_var_sum, mask_count
    if reduce == "mean":
        error_var_avg = error_var_sum / (mask_count + 1e-8)
        n_available = torch.count_nonzero(mask_count).clamp_min(1)
        return error_var_avg.sum() / n_available
    raise ValueError(f"Unknown reduction: {reduce}")


def compute_all_losses(
    model,
    fusion,
    batch_dict,
    enable_text=True,
    use_text_embeddings=True,
):
    native_text = bool(
        enable_text and getattr(model, "native_text_enabled", False)
    )
    forecast_kwargs = {}
    if native_text:
        if not use_text_embeddings:
            raise ValueError(
                "GPINet native text fusion requires --use_text_embeddings"
            )
        forecast_kwargs = {
            "notes_input": batch_dict["notes_embeddings"],
            # Native GPINet performs its own normalization against the full
            # history + prediction window, so it must receive raw timestamps.
            "tau": batch_dict["tau_raw"],
        }
    pred_y = model.forecasting(
        batch_dict["tp_to_predict"],
        batch_dict["observed_data"],
        batch_dict["observed_tp"],
        batch_dict["observed_mask"],
        **forecast_kwargs,
    )
    if not torch.isfinite(pred_y).all():
        raise ValueError("Model prediction contains NaN or Inf")

    if enable_text and not native_text and fusion is not None:
        notes_input = (
            batch_dict["notes_embeddings"]
            if use_text_embeddings
            else batch_dict["notes_text"]
        )
        pred_y = fusion(
            notes_input,
            batch_dict["tau"],
            batch_dict["tp_to_predict"],
            pred_y,
        )

    target = batch_dict["data_to_predict"]
    mask = batch_dict["mask_predicted_data"]
    if not torch.isfinite(pred_y).all():
        raise ValueError("Final prediction contains NaN or Inf")
    if not torch.isfinite(target).all():
        raise ValueError("Prediction target contains NaN or Inf")
    if (mask.reshape(mask.shape[0], -1).sum(dim=1) == 0).any():
        raise ValueError("At least one sample has an all-zero prediction mask")

    mse = compute_error(target, pred_y, mask, "MSE", "mean")
    if not torch.isfinite(mse):
        raise ValueError("MSE is NaN or Inf")
    return {"loss": mse, "mse": mse.item()}


def masked_mse_nn(pred_y, target, mask):
    mask_flat = mask.reshape(-1).bool()
    if mask_flat.sum() == 0:
        return pred_y.new_zeros(())
    return torch.nn.functional.mse_loss(
        pred_y.reshape(-1)[mask_flat],
        target.reshape(-1)[mask_flat],
    )


def _variable_names(n_dim):
    if n_dim == len(MIMIC_VARIABLE_NAMES):
        return MIMIC_VARIABLE_NAMES
    return [f"var_{i}" for i in range(n_dim)]


def _to_named_dict(names, tensor):
    values = tensor.detach().cpu().tolist()
    return {name: float(value) for name, value in zip(names, values)}


def _add_diag(diag_sum, diag_count, name, value):
    if value is None or not torch.is_tensor(value) or value.numel() == 0:
        return
    finite = value.detach().float().reshape(-1)
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return
    diag_sum[name] = diag_sum.get(name, 0.0) + finite.sum().item()
    diag_count[name] = diag_count.get(name, 0) + finite.numel()


def _collect_fusion_diagnostics(
    fusion,
    mask,
    diag_sum,
    diag_count,
    target=None,
    base_prediction=None,
):
    mmf = getattr(fusion, "mmf", None)
    ttf = getattr(fusion, "ttf", None)

    if mmf is not None:
        null_prob = getattr(mmf, "last_null_probability", None)
        gate = getattr(mmf, "last_gate", None)
        variable_relevance = getattr(mmf, "last_variable_relevance", None)
        correction = getattr(mmf, "last_correction", None)
        candidate_correction = getattr(
            mmf,
            "last_candidate_correction",
            None,
        )
        text_mask = getattr(mmf, "last_text_mask", None)
        context = getattr(mmf, "last_context", None)
        delta = getattr(mmf, "last_delta", None)
        attention = getattr(mmf, "last_slot_attention", None)

        # The current no-NULL SlotGate keeps an all-zero compatibility tensor,
        # although NULL is no longer part of its forward computation.  Treat a
        # NULL diagnostic as real only when the module explicitly opts in or
        # actually owns the learned NULL key used by the earlier formulation.
        supports_null = getattr(
            mmf,
            "supports_null_diagnostic",
            hasattr(mmf, "null_key"),
        )
        if supports_null:
            _add_diag(
                diag_sum,
                diag_count,
                "text_null_probability_mean",
                null_prob,
            )
        _add_diag(diag_sum, diag_count, "text_gate_mean", gate)
        _add_diag(
            diag_sum,
            diag_count,
            "text_variable_relevance_mean",
            variable_relevance,
        )
        if context is not None:
            _add_diag(
                diag_sum,
                diag_count,
                "mmf_context_rms",
                context.square().mean(dim=-1).sqrt(),
            )
        if delta is not None:
            _add_diag(
                diag_sum,
                diag_count,
                "mmf_delta_abs_mean",
                delta.abs(),
            )
        delta_out = getattr(mmf, "delta_out", None)
        if delta_out is not None:
            _add_diag(
                diag_sum,
                diag_count,
                "mmf_delta_out_weight_norm",
                delta_out.weight.detach().norm().reshape(1),
            )
        if candidate_correction is not None:
            _add_diag(
                diag_sum,
                diag_count,
                "text_candidate_correction_abs_mean",
                candidate_correction.abs(),
            )
            _add_diag(
                diag_sum,
                diag_count,
                "text_candidate_correction_abs_max",
                candidate_correction.abs().max().reshape(1),
            )
        if correction is not None:
            _add_diag(
                diag_sum,
                diag_count,
                "text_correction_abs_mean",
                correction.abs(),
            )
            _add_diag(
                diag_sum,
                diag_count,
                "text_correction_abs_max",
                correction.abs().max().reshape(1),
            )
            observed = mask.to(torch.bool)
            changed = (correction.abs() > 1e-6) & observed
            _add_diag(
                diag_sum,
                diag_count,
                "text_changed_fraction",
                changed.to(torch.float32)[observed],
            )
        if (
            gate is not None
            and candidate_correction is not None
            and text_mask is not None
            and target is not None
            and base_prediction is not None
        ):
            base_error = (target - base_prediction).square()
            candidate_error = (
                target - (base_prediction + candidate_correction)
            ).square()
            improvement = base_error - candidate_error
            tie_tolerance = 1e-6 * (1.0 + base_error)
            gate_target = (improvement > 0).to(gate.dtype)
            observed_text = mask.to(torch.bool) & text_mask.to(torch.bool)
            labeled = improvement.abs() > tie_tolerance
            valid = observed_text & labeled
            _add_diag(
                diag_sum,
                diag_count,
                "text_gate_labeled_fraction",
                labeled.to(torch.float32)[observed_text],
            )
            if valid.any():
                _add_diag(
                    diag_sum,
                    diag_count,
                    "text_gate_target_rate",
                    gate_target[valid],
                )
                _add_diag(
                    diag_sum,
                    diag_count,
                    "text_gate_brier",
                    (gate[valid] - gate_target[valid]).square(),
                )
                _add_diag(
                    diag_sum,
                    diag_count,
                    "text_gate_accuracy",
                    ((gate[valid] >= 0.5) == (gate_target[valid] > 0.5)).to(
                        torch.float32
                    ),
                )
        if attention is not None:
            # Semantic-slot MMF may keep a compatibility-only NULL column.
            # Measure entropy over real choices only, and omit the metric when
            # there is only one real choice because its entropy is forced to 0.
            real_choice_count = getattr(
                mmf,
                "semantic_slots",
                attention.shape[-1],
            )
            attention_for_entropy = attention[..., :real_choice_count]
            if real_choice_count > 1:
                attention_for_entropy = attention_for_entropy / (
                    attention_for_entropy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                )
                entropy = -(
                    attention_for_entropy
                    * attention_for_entropy.clamp_min(1e-8).log()
                ).sum(dim=-1)
                entropy = entropy / math.log(real_choice_count)
                _add_diag(
                    diag_sum,
                    diag_count,
                    "text_attention_entropy",
                    entropy,
                )

    if ttf is not None:
        slot_mass = getattr(ttf, "last_slot_mass", None)
        assignment = getattr(ttf, "last_slot_assignment", None)
        slot_outputs = getattr(ttf, "last_slot_outputs", None)
        absolute_recency = getattr(
            ttf,
            "last_absolute_recency_strength",
            None,
        )
        if absolute_recency is not None:
            recency_values = absolute_recency
            if slot_mass is not None:
                valid_slots = (slot_mass > 0).unsqueeze(-1).expand_as(
                    absolute_recency
                )
                recency_values = absolute_recency[valid_slots]
            _add_diag(
                diag_sum,
                diag_count,
                "text_absolute_recency_strength_mean",
                recency_values,
            )
        if slot_mass is not None:
            _add_diag(
                diag_sum,
                diag_count,
                "ttf_largest_slot_mass",
                slot_mass.max(dim=-1).values,
            )
            mass_prob = slot_mass / slot_mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            effective_slots = torch.exp(
                -(mass_prob * mass_prob.clamp_min(1e-8).log()).sum(dim=-1)
            )
            valid = slot_mass.sum(dim=-1) > 0
            _add_diag(
                diag_sum,
                diag_count,
                "ttf_effective_slot_count",
                effective_slots[valid],
            )
        if slot_outputs is not None:
            output_rms = slot_outputs.square().mean(dim=-1).sqrt()
            if slot_mass is not None:
                valid_output = (slot_mass > 0).unsqueeze(-1).expand_as(
                    output_rms
                )
                output_rms = output_rms[valid_output]
            _add_diag(
                diag_sum,
                diag_count,
                "ttf_output_rms",
                output_rms,
            )
        if assignment is not None and assignment.shape[1] > 1:
            valid_notes = assignment.sum(dim=1) > 0
            assignment_entropy = -(
                assignment * assignment.clamp_min(1e-8).log()
            ).sum(dim=1) / math.log(assignment.shape[1])
            _add_diag(
                diag_sum,
                diag_count,
                "ttf_assignment_entropy",
                assignment_entropy[valid_notes],
            )
            note_count = valid_notes.sum()
            if note_count > 0:
                usage = assignment.sum(dim=(0, 2)) / note_count.to(
                    assignment.dtype
                )
                uniform = torch.full_like(usage, 1.0 / assignment.shape[1])
                imbalance = assignment.shape[1] * (
                    usage - uniform
                ).square().sum()
                _add_diag(
                    diag_sum,
                    diag_count,
                    "ttf_slot_usage_imbalance",
                    imbalance.reshape(1),
                )
        if (
            slot_outputs is not None
            and slot_mass is not None
            and slot_outputs.shape[1] > 1
        ):
            normalized = torch.nn.functional.normalize(
                slot_outputs,
                dim=-1,
                eps=1e-8,
            )
            similarity = torch.einsum(
                "bhtd,bjtd->bthj",
                normalized,
                normalized,
            )
            h = slot_outputs.shape[1]
            valid_slots = slot_mass > 1e-6
            valid_pairs = (
                valid_slots[:, None, :, None]
                & valid_slots[:, None, None, :]
            ).expand(-1, slot_outputs.shape[2], -1, -1)
            off_diag = ~torch.eye(
                h,
                device=slot_outputs.device,
                dtype=torch.bool,
            )
            valid_pairs = valid_pairs & off_diag[None, None]
            _add_diag(
                diag_sum,
                diag_count,
                "ttf_cross_slot_output_similarity",
                similarity[valid_pairs],
            )


def evaluation(
    model,
    fusion,
    dataloader,
    enable_text=True,
    use_text_embeddings=True,
):
    """Evaluate text-aware and same-checkpoint base paths in one pass.

    For native GPINet text fusion, the base pass omits text before the MTGNN
    backbone and the fused pass includes it. Both use the same trained
    checkpoint, so ``fusion_delta_*`` measures the direct forward-pass effect
    of text rather than a separately initialized/retrained numerical model.
    """
    fused_se_sum = fused_ae_sum = fused_ape_sum = None
    base_se_sum = base_ae_sum = None
    mask_count = mask_count_mape = None
    correction_abs_sum = correction_signed_sum = None
    relevance_sum = gate_sum = null_probability_sum = None
    has_relevance_diag = False
    has_gate_diag = False
    has_null_probability_diag = False
    native_gate_count = None
    diag_sum = {}
    diag_count = {}

    for batch_dict in tqdm(dataloader):
        native_text = bool(
            enable_text and getattr(model, "native_text_enabled", False)
        )
        base_pred = model.forecasting(
            batch_dict["tp_to_predict"],
            batch_dict["observed_data"],
            batch_dict["observed_tp"],
            batch_dict["observed_mask"],
        )
        pred_y = base_pred

        if native_text:
            if not use_text_embeddings:
                raise ValueError(
                    "GPINet native text fusion requires --use_text_embeddings"
                )
            pred_y = model.forecasting(
                batch_dict["tp_to_predict"],
                batch_dict["observed_data"],
                batch_dict["observed_tp"],
                batch_dict["observed_mask"],
                notes_input=batch_dict["notes_embeddings"],
                tau=batch_dict["tau_raw"],
            )
        elif enable_text and fusion is not None:
            notes_input = (
                batch_dict["notes_embeddings"]
                if use_text_embeddings
                else batch_dict["notes_text"]
            )
            pred_y = fusion(
                notes_input,
                batch_dict["tau"],
                batch_dict["tp_to_predict"],
                base_pred,
            )

        target = batch_dict["data_to_predict"]
        mask = batch_dict["mask_predicted_data"]
        se, count = compute_error(target, pred_y, mask, "MSE", "sum")
        ae, _ = compute_error(target, pred_y, mask, "MAE", "sum")
        ape, count_mape = compute_error(target, pred_y, mask, "MAPE", "sum")

        if fused_se_sum is None:
            fused_se_sum = torch.zeros_like(se)
            fused_ae_sum = torch.zeros_like(ae)
            fused_ape_sum = torch.zeros_like(ape)
            mask_count = torch.zeros_like(count)
            mask_count_mape = torch.zeros_like(count_mape)
        fused_se_sum += se
        fused_ae_sum += ae
        fused_ape_sum += ape
        mask_count += count
        mask_count_mape += count_mape

        has_text_path = native_text or (enable_text and fusion is not None)
        if has_text_path:
            base_se, _ = compute_error(target, base_pred, mask, "MSE", "sum")
            base_ae, _ = compute_error(target, base_pred, mask, "MAE", "sum")
            if base_se_sum is None:
                base_se_sum = torch.zeros_like(base_se)
                base_ae_sum = torch.zeros_like(base_ae)
                correction_abs_sum = torch.zeros_like(base_se)
                correction_signed_sum = torch.zeros_like(base_se)
                relevance_sum = torch.zeros_like(base_se)
                gate_sum = torch.zeros_like(base_se)
                null_probability_sum = torch.zeros_like(base_se)
                native_gate_count = torch.zeros_like(base_se)
            base_se_sum += base_se
            base_ae_sum += base_ae

            correction = pred_y - base_pred
            correction_abs_sum += (correction.abs() * mask).sum(dim=(0, 1))
            correction_signed_sum += (correction * mask).sum(dim=(0, 1))

            mmf = getattr(fusion, "mmf", None) if fusion is not None else None
            if native_text:
                text_module = getattr(model, "text_grid_fusion", None)
                gate_value = getattr(text_module, "last_gate", None)
                grid_has_text = getattr(
                    text_module,
                    "last_grid_has_text",
                    None,
                )
                grid_note_count = getattr(
                    text_module,
                    "last_grid_note_count",
                    None,
                )
                if torch.is_tensor(gate_value) and torch.is_tensor(grid_has_text):
                    valid_grid = grid_has_text[:, None, :].expand_as(gate_value)
                    gate_sum += (gate_value * valid_grid).sum(dim=(0, 2))
                    native_gate_count += valid_grid.sum(dim=(0, 2))
                    has_gate_diag = True
                if torch.is_tensor(grid_has_text) and grid_has_text.any():
                    _add_diag(
                        diag_sum,
                        diag_count,
                        "text_attention_entropy",
                        getattr(text_module, "last_attention_entropy", None),
                    )
                    _add_diag(
                        diag_sum,
                        diag_count,
                        "gpinet_text_context_rms",
                        getattr(text_module, "last_context_rms", None),
                    )
                    _add_diag(
                        diag_sum,
                        diag_count,
                        "gpinet_text_update_abs_mean",
                        getattr(text_module, "last_update_abs_mean", None),
                    )
                _add_diag(
                    diag_sum,
                    diag_count,
                    "gpinet_text_nonempty_grid_fraction",
                    grid_has_text.to(torch.float32)
                    if torch.is_tensor(grid_has_text)
                    else None,
                )
                _add_diag(
                    diag_sum,
                    diag_count,
                    "gpinet_text_reports_per_nonempty_grid",
                    grid_note_count[grid_has_text]
                    if torch.is_tensor(grid_note_count)
                    and torch.is_tensor(grid_has_text)
                    else None,
                )
                observed = mask.to(torch.bool)
                _add_diag(
                    diag_sum,
                    diag_count,
                    "text_correction_abs_mean",
                    correction.abs()[observed],
                )
                _add_diag(
                    diag_sum,
                    diag_count,
                    "text_correction_abs_max",
                    correction.abs()[observed].max().reshape(1)
                    if observed.any()
                    else None,
                )
                _add_diag(
                    diag_sum,
                    diag_count,
                    "text_changed_fraction",
                    (correction.abs()[observed] > 1e-6).to(torch.float32),
                )
            elif mmf is not None:
                relevance = getattr(mmf, "last_variable_relevance", None)
                gate_value = getattr(mmf, "last_gate", None)
                null_value = getattr(mmf, "last_null_probability", None)
                supports_null = getattr(
                    mmf,
                    "supports_null_diagnostic",
                    hasattr(mmf, "null_key"),
                )
                if torch.is_tensor(relevance):
                    relevance_sum += (relevance * mask).sum(dim=(0, 1))
                    has_relevance_diag = True
                if torch.is_tensor(gate_value):
                    gate_sum += (gate_value * mask).sum(dim=(0, 1))
                    has_gate_diag = True
                if supports_null and torch.is_tensor(null_value):
                    null_probability_sum += (null_value * mask).sum(dim=(0, 1))
                    has_null_probability_diag = True
            if not native_text and fusion is not None:
                _collect_fusion_diagnostics(
                    fusion,
                    mask,
                    diag_sum,
                    diag_count,
                    target=target,
                    base_prediction=base_pred,
                )

    if fused_se_sum is None:
        raise ValueError("Evaluation dataloader produced no batches")

    available = mask_count > 0
    n_available = available.sum().clamp_min(1)
    fused_mse_var = fused_se_sum / mask_count.clamp_min(1e-8)
    fused_mae_var = fused_ae_sum / mask_count.clamp_min(1e-8)
    mape_var = fused_ape_sum / mask_count_mape.clamp_min(1e-8)

    mse = fused_mse_var[available].sum() / n_available
    mae = fused_mae_var[available].sum() / n_available
    mape_available = mask_count_mape > 0
    mape = mape_var[mape_available].mean()
    names = _variable_names(fused_mse_var.numel())

    results = {
        "loss": mse.item(),
        "mse": mse.item(),
        "mae": mae.item(),
        "rmse": torch.sqrt(mse).item(),
        "mape": mape.item(),
        "mse_per_variable": _to_named_dict(names, fused_mse_var),
        "mae_per_variable": _to_named_dict(names, fused_mae_var),
        "count_per_variable": _to_named_dict(names, mask_count),
    }

    if base_se_sum is not None:
        base_mse_var = base_se_sum / mask_count.clamp_min(1e-8)
        base_mae_var = base_ae_sum / mask_count.clamp_min(1e-8)
        base_mse = base_mse_var[available].mean()
        base_mae = base_mae_var[available].mean()
        correction_abs_var = correction_abs_sum / mask_count.clamp_min(1e-8)
        correction_signed_var = correction_signed_sum / mask_count.clamp_min(1e-8)
        results.update(
            {
                "base_path_mse": base_mse.item(),
                "base_path_mae": base_mae.item(),
                "fusion_delta_mse": (mse - base_mse).item(),
                "fusion_delta_mae": (mae - base_mae).item(),
                "base_mse_per_variable": _to_named_dict(names, base_mse_var),
                "base_mae_per_variable": _to_named_dict(names, base_mae_var),
                "fusion_delta_mse_per_variable": _to_named_dict(
                    names, fused_mse_var - base_mse_var
                ),
                "fusion_delta_mae_per_variable": _to_named_dict(
                    names, fused_mae_var - base_mae_var
                ),
                "correction_abs_per_variable": _to_named_dict(
                    names, correction_abs_var
                ),
                "correction_signed_per_variable": _to_named_dict(
                    names, correction_signed_var
                ),
            }
        )

        # These diagnostics are architecture-specific.  Do not emit a dict of
        # fake zeros when the selected paper MMF does not implement the field.
        if has_relevance_diag:
            relevance_var = relevance_sum / mask_count.clamp_min(1e-8)
            results["text_relevance_per_variable"] = _to_named_dict(
                names,
                relevance_var,
            )
        if has_gate_diag:
            gate_denominator = (
                native_gate_count
                if native_gate_count is not None and native_gate_count.sum() > 0
                else mask_count
            )
            gate_var = gate_sum / gate_denominator.clamp_min(1e-8)
            results["text_gate_per_variable"] = _to_named_dict(
                names,
                gate_var,
            )
            results["text_gate_mean"] = gate_var.mean().item()
        if has_null_probability_diag:
            null_probability_var = (
                null_probability_sum / mask_count.clamp_min(1e-8)
            )
            results["text_null_probability_per_variable"] = _to_named_dict(
                names,
                null_probability_var,
            )

    for name, value_sum in diag_sum.items():
        results[name] = value_sum / max(diag_count[name], 1)

    return results
