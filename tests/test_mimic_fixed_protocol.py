import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from scripts import prepare_mimic_fixed_protocol as prepare_protocol


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def test_protocol_persists_one_full_train_history_scaler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dataset_dir = tmp_path / "MIMIC"
    rows = []

    for subject_id in range(1, 11):
        record_id = str(10_000 + subject_id)
        record_dir = dataset_dir / "processed" / record_id
        record_dir.mkdir(parents=True)
        rows.append({"subject_id": subject_id, "record_id": record_id})

        # The large future values must not affect the history-only scaler.
        pd.DataFrame(
            {
                "date_time": ["2000-01-01 01:00:00", "2000-01-02 01:00:00"],
                "record_id": [record_id, record_id],
                "feature_a": [float(subject_id), float(10_000 + subject_id)],
                "feature_b": [float(2 * subject_id), float(20_000 + subject_id)],
            }
        ).to_csv(record_dir / "time_series.csv", index=False)

    pd.DataFrame(rows).to_csv(dataset_dir / "sample_statistics.csv", index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_mimic_fixed_protocol.py",
            "--dataset-dir",
            str(dataset_dir),
            "--history",
            "24",
            "--pred-window",
            "24",
            "--time-unit",
            "hours",
            "--split-seed",
            "42",
            "--train-order-seed",
            "2026",
        ],
    )
    prepare_protocol.main()

    with (dataset_dir / "mimic_fixed_protocol.json").open(encoding="utf-8") as f:
        protocol = json.load(f)
    scaler = _load_torch(dataset_dir / "mimic_fixed_normalization.pt")

    assert protocol["version"] == "4-fixed-full-train-normalization"
    assert scaler["version"] == protocol["version"]
    assert scaler["feature_cols"] == ["feature_a", "feature_b"]

    full_train_subjects = np.asarray(
        sorted(int(x) for x in protocol["train_subject_order"]), dtype=np.float64
    )
    expected = np.stack([full_train_subjects, 2 * full_train_subjects], axis=1)
    np.testing.assert_allclose(scaler["mean"].numpy(), expected.mean(axis=0))
    np.testing.assert_allclose(scaler["std"].numpy(), expected.std(axis=0, ddof=1))

    # A Train_N=1 run must reuse the scaler above, rather than fitting its own.
    first_subject = float(protocol["train_subject_order"][0])
    assert not np.isclose(float(scaler["mean"][0]), first_subject)
