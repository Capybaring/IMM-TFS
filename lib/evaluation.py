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
            denominator = truth_repeated + (truth_repeated == 0) * 1e-8
            error = torch.abs(truth_repeated - pred_y) / denominator * valid_mask
            mask = valid_mask
        else:
            data_max = norm_dict["data_max"]
            data_min = norm_dict["data_min"]
            truth_rescale = truth_repeated * (data_max - data_min) + data_min
            pred_rescale = pred_y * (data_max - data_min) + data_min
            valid_mask = (truth_rescale != 0) * mask
            denominator = truth_rescale + (truth_rescale == 0) * 1e-8
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
    pred_y = model.forecasting(
        batch_dict["tp_to_predict"],
        batch_dict["observed_data"],
        batch_dict["observed_tp"],
        batch_dict["observed_mask"],
    )
    if not torch.isfinite(pred_y).all():
        raise ValueError("Numerical prediction contains NaN or Inf")

    if enable_text and fusion is not None:
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


def _collect_fusion_diagnostics(fusion, mask, diag_sum, diag_count):
    mmf = getattr(fusion, "mmf", None)
    ttf = getattr(fusion, "ttf", None)

    if mmf is not None:
        null_prob = getattr(mmf, "last_null_probability", None)
        gate = getattr(mmf, "last_gate", None)
        correction = getattr(mmf, "last_correction", None)
        attention = getattr(mmf, "last_slot_attention", None)

        _add_diag(diag_sum, diag_count, "text_null_probability_mean", null_prob)
        _add_diag(diag_sum, diag_count, "text_gate_mean", gate)
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
        if attention is not None:
            entropy = -(
                attention * attention.clamp_min(1e-8).log()
            ).sum(dim=-1)
            entropy = entropy / math.log(attention.shape[-1])
            _add_diag(
                diag_sum,
                diag_count,
                "text_attention_entropy",
                entropy,
            )

    if ttf is not None:
        slot_mass = getattr(ttf, "last_slot_mass", None)
        weights = getattr(ttf, "last_semantic_weights", None)
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
        if weights is not None and weights.shape[1] > 1:
            normalized = torch.nn.functional.normalize(weights, dim=-1, eps=1e-8)
            similarity = torch.einsum("bhk,bjk->bhj", normalized, normalized)
            h = weights.shape[1]
            off_diag = ~torch.eye(h, device=weights.device, dtype=torch.bool)
            sample_similarity = similarity[:, off_diag].reshape(weights.shape[0], -1).mean(-1)
            valid = weights.sum(dim=(-1, -2)) > 0
            _add_diag(
                diag_sum,
                diag_count,
                "ttf_cross_slot_similarity",
                sample_similarity[valid],
            )


def evaluation(
    model,
    fusion,
    dataloader,
    enable_text=True,
    use_text_embeddings=True,
):
    """Evaluate fused and same-checkpoint base paths in one pass.

    For a multimodal model, ``base_path_*`` is computed from the numerical
    prediction immediately before fusion in the *same trained checkpoint*.
    Thus ``fusion_delta_*`` isolates the direct effect of text correction and
    is not confounded by a separately initialized/retrained kappa=0 run.
    """
    fused_se_sum = fused_ae_sum = fused_ape_sum = None
    base_se_sum = base_ae_sum = None
    mask_count = mask_count_mape = None
    correction_abs_sum = correction_signed_sum = None
    diag_sum = {}
    diag_count = {}

    for batch_dict in tqdm(dataloader):
        base_pred = model.forecasting(
            batch_dict["tp_to_predict"],
            batch_dict["observed_data"],
            batch_dict["observed_tp"],
            batch_dict["observed_mask"],
        )
        pred_y = base_pred

        if enable_text and fusion is not None:
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

        if enable_text and fusion is not None:
            base_se, _ = compute_error(target, base_pred, mask, "MSE", "sum")
            base_ae, _ = compute_error(target, base_pred, mask, "MAE", "sum")
            if base_se_sum is None:
                base_se_sum = torch.zeros_like(base_se)
                base_ae_sum = torch.zeros_like(base_ae)
                correction_abs_sum = torch.zeros_like(base_se)
                correction_signed_sum = torch.zeros_like(base_se)
            base_se_sum += base_se
            base_ae_sum += base_ae

            correction = pred_y - base_pred
            correction_abs_sum += (correction.abs() * mask).sum(dim=(0, 1))
            correction_signed_sum += (correction * mask).sum(dim=(0, 1))
            _collect_fusion_diagnostics(
                fusion, mask, diag_sum, diag_count
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

    for name, value_sum in diag_sum.items():
        results[name] = value_sum / max(diag_count[name], 1)

    return results
