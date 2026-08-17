from argparse import Namespace
from pathlib import Path

import pandas as pd

from scripts.prepare_mimic_chest_radiology import (
    _prepare_output_root,
    PREDICTION_FEATURES,
    build_cohort,
    build_wide_numeric,
    classify_chest_exam,
    load_chartevents,
    load_labevents,
    load_radiology_text,
    write_dataset,
)


def _write_csv(root: Path, relative: str, rows: list[dict]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")


def test_classify_chest_exam():
    assert classify_chest_exam("CHEST (PORTABLE AP)") == "xray_chest"
    assert classify_chest_exam("CT CHEST W/CONTRAST") == "ct_chest"
    assert classify_chest_exam("US CHEST") == "ultrasound_chest"
    assert classify_chest_exam("MRI CHEST") is None
    assert classify_chest_exam("CT ABDOMEN/PELVIS") is None


def test_end_to_end_availability_and_addendum_alignment(tmp_path: Path):
    mimic_root = tmp_path / "mimiciv" / "3.1"
    note_root = tmp_path / "mimic-iv-note" / "2.2" / "note"
    output_root = tmp_path / "MIMIC_chest"

    _write_csv(
        mimic_root,
        "hosp/admissions.csv.gz",
        [
            {
                "subject_id": 1,
                "hadm_id": 11,
                "admittime": "2020-01-01 00:00:00",
                "dischtime": "2020-01-04 00:00:00",
            },
            {
                "subject_id": 2,
                "hadm_id": 22,
                "admittime": "2020-02-01 00:00:00",
                "dischtime": "2020-02-04 00:00:00",
            },
        ],
    )
    _write_csv(
        mimic_root,
        "hosp/patients.csv.gz",
        [
            {"subject_id": 1, "anchor_age": 50, "anchor_year": 2020},
            {"subject_id": 2, "anchor_age": 60, "anchor_year": 2020},
        ],
    )
    _write_csv(
        mimic_root,
        "icu/icustays.csv.gz",
        [
            {
                "subject_id": 1,
                "hadm_id": 11,
                "stay_id": 111,
                "intime": "2020-01-01 00:00:00",
                "outtime": "2020-01-03 12:00:00",
            },
            {
                "subject_id": 2,
                "hadm_id": 22,
                "stay_id": 222,
                "intime": "2020-02-01 00:00:00",
                "outtime": "2020-02-03 12:00:00",
            },
        ],
    )
    _write_csv(
        mimic_root,
        "icu/d_items.csv.gz",
        [{"itemid": 220277, "label": "O2 saturation pulseoxymetry"}],
    )
    _write_csv(
        mimic_root,
        "hosp/d_labitems.csv.gz",
        [{"itemid": 50820, "label": "pH"}],
    )

    chart_rows = []
    for subject_id, hadm_id, stay_id, base in (
        (1, 11, 111, pd.Timestamp("2020-01-01")),
        (2, 22, 222, pd.Timestamp("2020-02-01")),
    ):
        for hour, spo2 in ((1, 95), (25, 93)):
            chart_rows.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "stay_id": stay_id,
                    "charttime": base + pd.Timedelta(hours=hour),
                    "storetime": base + pd.Timedelta(hours=hour, minutes=5),
                    "itemid": 220277,
                    "valuenum": spo2,
                    "warning": 0,
                }
            )
    _write_csv(mimic_root, "icu/chartevents.csv.gz", chart_rows)

    lab_rows = []
    for subject_id, hadm_id, base in (
        (1, 11, pd.Timestamp("2020-01-01")),
        (2, 22, pd.Timestamp("2020-02-01")),
    ):
        for hour, value in ((2, 7.40), (26, 7.35)):
            lab_rows.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "charttime": base + pd.Timedelta(hours=hour),
                    "storetime": base + pd.Timedelta(hours=hour, minutes=20),
                    "itemid": 50820,
                    "valuenum": value,
                }
            )
    _write_csv(mimic_root, "hosp/labevents.csv.gz", lab_rows)

    _write_csv(
        note_root,
        "radiology.csv.gz",
        [
            {
                "note_id": "1-RR-1",
                "subject_id": 1,
                "hadm_id": 11,
                "note_type": "RR",
                "note_seq": 1,
                "charttime": "2020-01-01 04:00:00",
                "storetime": "2020-01-01 05:00:00",
                "text": "FINDINGS: bilateral opacities.",
            },
            {
                "note_id": "1-AR-1",
                "subject_id": 1,
                "hadm_id": 11,
                "note_type": "AR",
                "note_seq": 1,
                "charttime": "2020-01-01 08:00:00",
                "storetime": "2020-01-01 09:00:00",
                "text": "ADDENDUM: small left pleural effusion.",
            },
            {
                "note_id": "2-RR-1",
                "subject_id": 2,
                "hadm_id": 22,
                "note_type": "RR",
                "note_seq": 1,
                "charttime": "2020-02-01 04:00:00",
                "storetime": "2020-02-02 01:00:00",
                "text": "IMPRESSION: pulmonary edema.",
            },
        ],
    )
    _write_csv(
        note_root,
        "radiology_detail.csv.gz",
        [
            {
                "note_id": "1-RR-1",
                "field_name": "exam_name",
                "field_value": "CHEST (PORTABLE AP)",
                "field_ordinal": 1,
            },
            {
                "note_id": "1-AR-1",
                "field_name": "parent_note_id",
                "field_value": "1-RR-1",
                "field_ordinal": 1,
            },
            {
                "note_id": "2-RR-1",
                "field_name": "exam_name",
                "field_value": "CT CHEST W/CONTRAST",
                "field_ordinal": 1,
            },
        ],
    )

    args = Namespace(
        mimic_root=mimic_root,
        note_root=note_root,
        output_root=output_root,
        context_hours=24.0,
        prediction_hours=24.0,
        min_age=18.0,
        chunk_size=10,
        max_patients=None,
        min_history_observations=1,
        min_target_observations=1,
        min_history_variables=1,
        min_target_variables=1,
        min_direct_respiratory_history_variables=1,
        min_direct_respiratory_target_variables=1,
        allow_null_note_storetime=False,
        overwrite=False,
    )

    _prepare_output_root(output_root, overwrite=False)
    cohort = build_cohort(args)
    chart, _ = load_chartevents(args, cohort)
    labs, _ = load_labevents(args, cohort)
    numeric = build_wide_numeric(chart, labs)
    text, metadata, text_stats = load_radiology_text(args, cohort)
    stats = write_dataset(args, cohort, numeric, text)

    assert len(stats) == 2
    assert text_stats["late_reports_excluded"] == 1
    assert set(text["record_id"]) == {11}
    assert len(text) == 2
    assert metadata.loc[metadata["note_type"].eq("AR"), "exam_type"].item() == "xray_chest"

    strict_text = pd.read_csv(output_root / "processed" / "11" / "text.csv")
    assert len(strict_text) == 2
    assert strict_text["text"].str.contains("NOTE_TYPE=AR", regex=False).any()
    assert not (output_root / "processed" / "22").exists()

    time_series = pd.read_csv(output_root / "processed" / "11" / "time_series.csv")
    assert list(time_series.columns[2:]) == list(PREDICTION_FEATURES)
    assert "fio2" not in time_series.columns

    # Run the repository's real loader compatibility check when the test
    # environment includes PyTorch. The lightweight preprocessing environment
    # used in CI/syntax checks may intentionally omit it.
    try:
        import torch
        from lib.parse_datasets_mimic_expanded import ExpandedMIMICDataset
    except ModuleNotFoundError:
        return

    loader_args = Namespace(n=int(1e8), rec_ids=None)
    numeric_dataset = ExpandedMIMICDataset(
        root=str(output_root),
        history=24,
        pred_window=24,
        device=torch.device("cpu"),
        time_unit="hours",
        enable_text=False,
        args=loader_args,
    )
    text_dataset = ExpandedMIMICDataset(
        root=str(output_root),
        history=24,
        pred_window=24,
        device=torch.device("cpu"),
        time_unit="hours",
        enable_text=True,
        use_text_embeddings=False,
        args=loader_args,
    )
    assert list(numeric_dataset.feature_cols) == list(PREDICTION_FEATURES)
    assert len(numeric_dataset.chunks) == 1
    assert len(text_dataset.chunks[0][-1]) == 2
