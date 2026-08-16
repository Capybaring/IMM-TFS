"""Minimal shape/gradient checks for GPINet's native report-event fusion.

Run from the repository root:
    python tests/test_gpinet_variable_text_fusion.py
"""

from pathlib import Path
import sys
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.GPINet import GPINet  # noqa: E402


def make_args(**overrides):
    values = {
        "device": "cpu",
        "C": 4,
        "hid_dim": 8,
        "te_dim": 4,
        "history": 24,
        "pred_window": 24,
        "nlayer": 2,
        "hop": 1,
        "gpinet_subgraph_size": 4,
        "node_dim": 4,
        "dropout": 0.0,
        "d_txt": 16,
        "n_heads_fusion": 2,
        "gpinet_text_gate_bias": -1.0,
        "enable_text": True,
        "use_text_embeddings": True,
        "seed": 7,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def main():
    torch.manual_seed(7)
    model = GPINet(make_args())

    batch_size, n_obs, n_variables = 3, 8, 4
    n_notes, text_dim, n_future = 2, 16, 7
    observed_tp = torch.linspace(0.0, 0.45, n_obs).repeat(batch_size, 1)
    observed_data = torch.randn(batch_size, n_obs, n_variables)
    observed_mask = torch.ones_like(observed_data)
    future_tp = torch.linspace(0.5, 0.98, n_future).repeat(batch_size, 1)
    notes = torch.randn(batch_size, n_notes, text_dim)
    tau = torch.tensor([[4.6, 8.9], [1.0, 20.0], [6.5, 18.2]])

    model.eval()
    with torch.no_grad():
        y_uni = model.forecasting(
            future_tp, observed_data, observed_tp, observed_mask
        )
        y_multi = model.forecasting(
            future_tp,
            observed_data,
            observed_tp,
            observed_mask,
            notes_input=notes,
            tau=tau,
        )
        gate_mean = model.last_text_gate_mean
        attention_entropy = model.last_text_attention_entropy
        y_empty = model.forecasting(
            future_tp,
            observed_data,
            observed_tp,
            observed_mask,
            notes_input=torch.zeros_like(notes),
            tau=torch.zeros_like(tau),
        )

    expected_shape = (batch_size, n_future, n_variables)
    assert tuple(y_uni.shape) == expected_shape
    assert tuple(y_multi.shape) == expected_shape
    assert not torch.allclose(y_uni, y_multi), "real text should affect the output"
    assert torch.equal(y_uni, y_empty), "empty text must be an exact Uni no-op"
    assert gate_mean is not None
    assert attention_entropy is not None

    model.train()
    model.zero_grad(set_to_none=True)
    output = model.forecasting(
        future_tp,
        observed_data,
        observed_tp,
        observed_mask,
        notes_input=notes,
        tau=tau,
    )
    output.square().mean().backward()

    gradients = {
        "text projection": model.backbone.text_event_encoder.proj_in.weight.grad,
        "variable-text attention": (
            model.backbone.text_injections[0].attn.in_proj_weight.grad
        ),
        "MTGNN graph": model.backbone.blocks[0]["graph"].project.weight.grad,
    }
    for name, grad in gradients.items():
        assert grad is not None, f"missing gradient: {name}"
        assert torch.isfinite(grad).all(), f"non-finite gradient: {name}"
        assert grad.abs().sum() > 0, f"zero gradient: {name}"

    print("PASS: shapes, text effect, empty-text no-op, and gradients")


if __name__ == "__main__":
    main()
