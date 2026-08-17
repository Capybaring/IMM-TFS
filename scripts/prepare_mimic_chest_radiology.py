#!/usr/bin/env python3
"""Build a clinically aligned MIMIC-IV chest-radiology forecasting cohort.

The output is compatible with ``lib/parse_datasets_mimic_expanded.py``:

    <output_root>/processed_full/<hadm_id>/time_series.csv
    <output_root>/processed_full/<hadm_id>/text.csv
    <output_root>/processed/<hadm_id>/time_series.csv
    <output_root>/processed/<hadm_id>/text.csv

``processed_full`` contains every numeric-valid episode. ``processed`` is the
strict multimodal subset with at least one available chest radiology report in
the 0-24 h history window.

Core design:

* one earliest eligible ICU stay per adult patient;
* ICU intime is t=0;
* irregular numeric history [0, 24 h) predicts [24, 48 h);
* numeric event location uses charttime, while history availability is checked
  with storetime;
* chest X-ray, chest CT, and chest ultrasound reports are selected using
  radiology_detail exam metadata;
* text event location uses charttime, while report availability is checked
  strictly with storetime;
* RR and AR remain separate timestamped text events. AR inherits the parent
  exam metadata when necessary;
* history-only treatment/device variables have no target observations after
  24 h, so the existing masked loss evaluates the 16 physiologic targets only.

The item IDs and cleaning rules follow the public MIT-LCP MIMIC code concepts
for vital signs, ventilator settings, oxygen delivery, blood gases, and CBC.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_DATETIME = pd.Timestamp("2000-01-01 00:00:00")


PREDICTION_FEATURES = (
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
)

HISTORY_ONLY_FEATURES = (
    "fio2",
    "oxygen_flow",
    "peep",
    "tidal_volume_set",
    "pressure_support",
    "peak_inspiratory_pressure",
    "plateau_pressure",
    "mean_airway_pressure",
)

FEATURES = PREDICTION_FEATURES + HISTORY_ONLY_FEATURES

DIRECT_RESPIRATORY_FEATURES = {
    "spo2",
    "respiratory_rate",
    "ph",
    "pao2",
    "paco2",
    "bicarbonate_bg",
    "base_excess",
    "minute_ventilation",
    "tidal_volume_observed",
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source: str
    itemids: tuple[int, ...]
    role: str
    unit: str
    lower: float | None
    upper: float | None
    description: str


FEATURE_SPECS = (
    FeatureSpec("heart_rate", "chartevents", (220045,), "target", "bpm", 0, 300, "Heart rate"),
    FeatureSpec("sbp", "chartevents", (220179, 220050, 225309), "target", "mmHg", 0, 400, "Systolic blood pressure"),
    FeatureSpec("dbp", "chartevents", (220180, 220051, 225310), "target", "mmHg", 0, 300, "Diastolic blood pressure"),
    FeatureSpec("map", "chartevents", (220181, 220052, 225312), "target", "mmHg", 0, 300, "Mean arterial pressure"),
    FeatureSpec("respiratory_rate", "chartevents", (220210, 224690), "target", "breaths/min", 0, 70, "Respiratory rate / total respiratory rate"),
    FeatureSpec("temperature_c", "chartevents", (223761, 223762), "target", "degC", 10, 50, "Temperature, converted to Celsius"),
    FeatureSpec("spo2", "chartevents", (220277,), "target", "%", 0, 100, "Peripheral oxygen saturation"),
    FeatureSpec("minute_ventilation", "chartevents", (224687,), "target", "L/min", 0, 100, "Minute ventilation"),
    FeatureSpec("tidal_volume_observed", "chartevents", (224685,), "target", "mL", 0, 3000, "Observed tidal volume"),
    FeatureSpec("fio2", "chartevents", (223835,), "history_covariate", "%", 20, 100, "Inspired oxygen fraction"),
    FeatureSpec("oxygen_flow", "chartevents", (223834, 227582), "history_covariate", "L/min", 0, 100, "Oxygen/BiPAP oxygen flow"),
    FeatureSpec("peep", "chartevents", (220339, 224700), "history_covariate", "cmH2O", 0, 100, "Positive end-expiratory pressure"),
    FeatureSpec("tidal_volume_set", "chartevents", (224684,), "history_covariate", "mL", 0, 3000, "Set tidal volume"),
    FeatureSpec("pressure_support", "chartevents", (224701,), "history_covariate", "cmH2O", 0, 100, "Pressure support / PSV level"),
    FeatureSpec("peak_inspiratory_pressure", "chartevents", (224695,), "history_covariate", "cmH2O", 0, 100, "Peak inspiratory pressure"),
    FeatureSpec("plateau_pressure", "chartevents", (224696,), "history_covariate", "cmH2O", 0, 100, "Plateau pressure"),
    FeatureSpec("mean_airway_pressure", "chartevents", (224697,), "history_covariate", "cmH2O", 0, 100, "Mean airway pressure"),
    FeatureSpec("ph", "labevents", (50820,), "target", "pH", 6.0, 8.5, "Blood gas pH"),
    FeatureSpec("pao2", "labevents", (50821,), "target", "mmHg", 0, 1000, "Blood gas oxygen partial pressure"),
    FeatureSpec("paco2", "labevents", (50818,), "target", "mmHg", 0, 300, "Blood gas carbon dioxide partial pressure"),
    FeatureSpec("bicarbonate_bg", "labevents", (50803,), "target", "mEq/L", 0, 80, "Blood gas bicarbonate"),
    FeatureSpec("base_excess", "labevents", (50802,), "target", "mEq/L", -50, 50, "Blood gas base excess"),
    FeatureSpec("lactate", "labevents", (50813,), "target", "mmol/L", 0, 50, "Blood lactate"),
    FeatureSpec("wbc", "labevents", (51301,), "target", "K/uL", 0, 200, "White blood cell count"),
)

SPEC_BY_FEATURE = {s.name: s for s in FEATURE_SPECS}
CHARTEVENT_ITEM_TO_FEATURE = {
    itemid: spec.name
    for spec in FEATURE_SPECS
    if spec.source == "chartevents"
    for itemid in spec.itemids
}
LABEVENT_ITEM_TO_FEATURE = {
    itemid: spec.name
    for spec in FEATURE_SPECS
    if spec.source == "labevents"
    for itemid in spec.itemids
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare chest-radiology aligned MIMIC-IV forecasting data."
    )
    parser.add_argument("--mimic-root", type=Path, default=Path("data/mimiciv/3.1"))
    parser.add_argument(
        "--note-root", type=Path, default=Path("data/mimic-iv-note/2.2/note")
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/MIMIC_chest"))
    parser.add_argument("--context-hours", type=float, default=24.0)
    parser.add_argument("--prediction-hours", type=float, default=24.0)
    parser.add_argument("--min-age", type=float, default=18.0)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--min-history-observations", type=int, default=20)
    parser.add_argument("--min-target-observations", type=int, default=20)
    parser.add_argument("--min-history-variables", type=int, default=5)
    parser.add_argument("--min-target-variables", type=int, default=5)
    parser.add_argument(
        "--min-direct-respiratory-history-variables", type=int, default=2
    )
    parser.add_argument(
        "--min-direct-respiratory-target-variables", type=int, default=2
    )
    parser.add_argument(
        "--allow-null-note-storetime",
        action="store_true",
        help=(
            "Treat missing radiology storetime as charttime. The default is "
            "strict: exclude such notes from the multimodal input."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only generated outputs inside --output-root.",
    )
    return parser.parse_args()


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required MIMIC files:\n" + "\n".join(missing))


def normalize_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def build_cohort(args: argparse.Namespace) -> pd.DataFrame:
    admissions_path = args.mimic_root / "hosp" / "admissions.csv.gz"
    patients_path = args.mimic_root / "hosp" / "patients.csv.gz"
    icustays_path = args.mimic_root / "icu" / "icustays.csv.gz"
    require_files((admissions_path, patients_path, icustays_path))

    admissions = pd.read_csv(
        admissions_path,
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
    )
    patients = pd.read_csv(
        patients_path,
        usecols=["subject_id", "anchor_age", "anchor_year"],
    )
    icustays = pd.read_csv(
        icustays_path,
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    )

    for col in ("admittime", "dischtime"):
        admissions[col] = pd.to_datetime(admissions[col], errors="coerce")
    for col in ("intime", "outtime"):
        icustays[col] = pd.to_datetime(icustays[col], errors="coerce")

    cohort = (
        icustays.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
        .merge(patients, on="subject_id", how="inner")
        .dropna(subset=["intime", "outtime", "admittime"])
    )
    cohort["age_at_admission"] = (
        cohort["anchor_age"]
        + cohort["admittime"].dt.year
        - cohort["anchor_year"]
    )
    cohort["icu_los_hours"] = (
        cohort["outtime"] - cohort["intime"]
    ).dt.total_seconds() / 3600.0

    total_hours = args.context_hours + args.prediction_hours
    cohort = cohort[
        (cohort["age_at_admission"] >= args.min_age)
        & (cohort["icu_los_hours"] >= total_hours)
    ].copy()
    cohort = cohort.sort_values(["subject_id", "intime", "stay_id"])
    cohort = cohort.drop_duplicates("subject_id", keep="first").copy()

    if args.max_patients is not None:
        cohort = cohort.head(args.max_patients).copy()

    cohort["history_cutoff"] = cohort["intime"] + pd.to_timedelta(
        args.context_hours, unit="h"
    )
    cohort["episode_end"] = cohort["intime"] + pd.to_timedelta(total_hours, unit="h")

    if not cohort["subject_id"].is_unique:
        raise AssertionError("Cohort is not patient-disjoint")
    if not cohort["stay_id"].is_unique or not cohort["hadm_id"].is_unique:
        raise AssertionError("Expected one unique stay/admission per selected patient")

    print(f"Eligible one-stay-per-patient cohort: {len(cohort):,}")
    return cohort


def _prepare_output_root(output_root: Path, overwrite: bool) -> None:
    generated_dirs = [output_root / "processed_full", output_root / "processed"]
    generated_files = [
        output_root / "sample_statistics.csv",
        output_root / "cohort_summary.csv",
        output_root / "feature_dictionary.csv",
        output_root / "text_event_metadata.csv",
        output_root / "tfsimm_dataset_config.json",
        output_root / "mimic_fixed_protocol.json",
    ]
    existing = [path for path in generated_dirs + generated_files if path.exists()]
    if existing and not overwrite:
        names = "\n".join(str(path) for path in existing[:20])
        raise FileExistsError(
            "Generated output already exists. Use a new --output-root or pass "
            f"--overwrite.\n{names}"
        )
    if overwrite:
        for path in generated_dirs:
            if path.is_dir():
                shutil.rmtree(path)
        for path in generated_files:
            if path.is_file():
                path.unlink()
    output_root.mkdir(parents=True, exist_ok=True)
    for path in generated_dirs:
        path.mkdir(parents=True, exist_ok=True)


def _cohort_maps(cohort: pd.DataFrame, key: str) -> dict[str, pd.Series]:
    indexed = cohort.set_index(key)

    def column_or_index(name: str) -> pd.Series:
        if name == key:
            return pd.Series(indexed.index, index=indexed.index, name=name)
        return indexed[name]

    return {
        "intime": column_or_index("intime"),
        "history_cutoff": column_or_index("history_cutoff"),
        "hadm_id": column_or_index("hadm_id"),
        "subject_id": column_or_index("subject_id"),
        "stay_id": column_or_index("stay_id"),
    }


def _clean_feature_values(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["value"] = pd.to_numeric(frame["valuenum"], errors="coerce")

    fahrenheit = frame["itemid"].eq(223761)
    frame.loc[fahrenheit, "value"] = (frame.loc[fahrenheit, "value"] - 32.0) / 1.8

    fio2 = frame["feature"].eq("fio2")
    fio2_fraction = fio2 & frame["value"].between(0.20, 1.0, inclusive="both")
    frame.loc[fio2_fraction, "value"] *= 100.0
    invalid_fio2 = fio2 & ~frame["value"].between(20.0, 100.0, inclusive="both")
    frame.loc[invalid_fio2, "value"] = np.nan

    for feature, spec in SPEC_BY_FEATURE.items():
        selected = frame["feature"].eq(feature)
        if spec.lower is not None:
            frame.loc[selected & frame["value"].le(spec.lower), "value"] = np.nan
        if spec.upper is not None:
            frame.loc[selected & frame["value"].gt(spec.upper), "value"] = np.nan

    return frame.dropna(subset=["value"])


def _availability_filter(
    frame: pd.DataFrame,
    context_hours: float,
) -> tuple[pd.DataFrame, int]:
    frame = frame.copy()
    frame["storetime"] = pd.to_datetime(frame["storetime"], errors="coerce")
    missing_storetime = int(frame["storetime"].isna().sum())
    frame["available_time"] = frame["storetime"].fillna(frame["charttime"])
    history = frame["rel_hours"] < context_hours
    available_in_history = frame["available_time"] < frame["history_cutoff"]
    frame = frame[(~history) | available_in_history].copy()
    return frame, missing_storetime


def load_chartevents(
    args: argparse.Namespace,
    cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    path = args.mimic_root / "icu" / "chartevents.csv.gz"
    require_files((path,))
    maps = _cohort_maps(cohort, "stay_id")
    stay_ids = set(int(x) for x in cohort["stay_id"])
    wanted_itemids = set(CHARTEVENT_ITEM_TO_FEATURE)
    total_hours = args.context_hours + args.prediction_hours
    pieces: list[pd.DataFrame] = []
    seen = kept = missing_storetime = 0

    reader = pd.read_csv(
        path,
        usecols=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "charttime",
            "storetime",
            "itemid",
            "valuenum",
            "warning",
        ],
        chunksize=args.chunk_size,
    )
    for chunk_index, chunk in enumerate(reader, start=1):
        seen += len(chunk)
        chunk["stay_id"] = normalize_id(chunk["stay_id"])
        chunk["itemid"] = normalize_id(chunk["itemid"])
        chunk = chunk[
            chunk["stay_id"].isin(stay_ids)
            & chunk["itemid"].isin(wanted_itemids)
        ].copy()
        if chunk.empty:
            if chunk_index % 25 == 0:
                print(f"chartevents chunks scanned: {chunk_index:,}")
            continue

        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk[pd.to_numeric(chunk["warning"], errors="coerce").fillna(0).ne(1)]
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["intime"] = chunk["stay_id"].map(maps["intime"])
        chunk["history_cutoff"] = chunk["stay_id"].map(maps["history_cutoff"])
        chunk["record_id"] = chunk["stay_id"].map(maps["hadm_id"]).astype("Int64")
        chunk["rel_hours"] = (
            chunk["charttime"] - chunk["intime"]
        ).dt.total_seconds() / 3600.0
        chunk = chunk[chunk["rel_hours"].between(0, total_hours, inclusive="left")]
        chunk["feature"] = chunk["itemid"].map(CHARTEVENT_ITEM_TO_FEATURE)
        chunk, missing = _availability_filter(chunk, args.context_hours)
        missing_storetime += missing

        # Treatment/device variables are history covariates, not forecast labels.
        history_only = chunk["feature"].isin(HISTORY_ONLY_FEATURES)
        chunk = chunk[(~history_only) | (chunk["rel_hours"] < args.context_hours)]
        chunk = _clean_feature_values(chunk)
        if not chunk.empty:
            pieces.append(
                chunk[["record_id", "rel_hours", "feature", "value"]].copy()
            )
            kept += len(chunk)
        if chunk_index % 25 == 0:
            print(f"chartevents chunks scanned: {chunk_index:,}")

    result = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else pd.DataFrame(columns=["record_id", "rel_hours", "feature", "value"])
    )
    stats = {
        "rows_scanned": int(seen),
        "events_kept": int(kept),
        "missing_storetime_fallback": int(missing_storetime),
    }
    print(f"Selected chartevents: {len(result):,}")
    return result, stats


def load_labevents(
    args: argparse.Namespace,
    cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    path = args.mimic_root / "hosp" / "labevents.csv.gz"
    require_files((path,))
    maps = _cohort_maps(cohort, "hadm_id")
    hadm_ids = set(int(x) for x in cohort["hadm_id"])
    wanted_itemids = set(LABEVENT_ITEM_TO_FEATURE)
    total_hours = args.context_hours + args.prediction_hours
    pieces: list[pd.DataFrame] = []
    seen = kept = missing_storetime = 0

    reader = pd.read_csv(
        path,
        usecols=[
            "subject_id",
            "hadm_id",
            "charttime",
            "storetime",
            "itemid",
            "valuenum",
        ],
        chunksize=args.chunk_size,
    )
    for chunk_index, chunk in enumerate(reader, start=1):
        seen += len(chunk)
        chunk["hadm_id"] = normalize_id(chunk["hadm_id"])
        chunk["itemid"] = normalize_id(chunk["itemid"])
        chunk = chunk[
            chunk["hadm_id"].isin(hadm_ids)
            & chunk["itemid"].isin(wanted_itemids)
        ].copy()
        if chunk.empty:
            if chunk_index % 25 == 0:
                print(f"labevents chunks scanned: {chunk_index:,}")
            continue

        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["intime"] = chunk["hadm_id"].map(maps["intime"])
        chunk["history_cutoff"] = chunk["hadm_id"].map(maps["history_cutoff"])
        chunk["record_id"] = chunk["hadm_id"]
        chunk["rel_hours"] = (
            chunk["charttime"] - chunk["intime"]
        ).dt.total_seconds() / 3600.0
        chunk = chunk[chunk["rel_hours"].between(0, total_hours, inclusive="left")]
        chunk["feature"] = chunk["itemid"].map(LABEVENT_ITEM_TO_FEATURE)
        chunk, missing = _availability_filter(chunk, args.context_hours)
        missing_storetime += missing
        chunk = _clean_feature_values(chunk)
        if not chunk.empty:
            pieces.append(
                chunk[["record_id", "rel_hours", "feature", "value"]].copy()
            )
            kept += len(chunk)
        if chunk_index % 25 == 0:
            print(f"labevents chunks scanned: {chunk_index:,}")

    result = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else pd.DataFrame(columns=["record_id", "rel_hours", "feature", "value"])
    )
    stats = {
        "rows_scanned": int(seen),
        "events_kept": int(kept),
        "missing_storetime_fallback": int(missing_storetime),
    }
    print(f"Selected labevents: {len(result):,}")
    return result, stats


def build_wide_numeric(
    chartevents: pd.DataFrame,
    labevents: pd.DataFrame,
) -> pd.DataFrame:
    events = pd.concat([chartevents, labevents], ignore_index=True)
    if events.empty:
        raise RuntimeError("No numeric events survived filtering")
    events["record_id"] = events["record_id"].astype(int)

    # Multiple chart rows can describe the same clinical variable at the same
    # instant. Mean aggregation matches the public MIMIC vital-sign concept.
    collapsed = (
        events.groupby(["record_id", "rel_hours", "feature"], as_index=False)["value"]
        .mean()
    )
    wide = collapsed.pivot(
        index=["record_id", "rel_hours"], columns="feature", values="value"
    ).reset_index()
    wide.columns.name = None
    for feature in FEATURES:
        if feature not in wide.columns:
            wide[feature] = np.nan
    return wide[["record_id", "rel_hours", *FEATURES]].sort_values(
        ["record_id", "rel_hours"]
    )


def _join_detail_values(values: pd.Series) -> str:
    cleaned = [str(x).strip() for x in values if pd.notna(x) and str(x).strip()]
    return " || ".join(dict.fromkeys(cleaned))


def classify_chest_exam(exam_name: object, exam_code: object = "") -> str | None:
    text = f"{exam_name or ''} {exam_code or ''}".upper()
    text = re.sub(r"\s+", " ", text).strip()
    if not re.search(r"\b(CHEST|THORAX|THORACIC)\b", text):
        return None
    if re.search(r"\b(CT|CTA|CAT)\b|COMPUTED TOMOGRAPH", text):
        return "ct_chest"
    if re.search(r"\bUS\b|ULTRASOUND|SONOGRAM", text):
        return "ultrasound_chest"
    if re.search(r"\b(MR|MRI|PET)\b|MAGNETIC RESON|NUCLEAR", text):
        return None
    # MIMIC exam names such as "CHEST (PORTABLE AP)" often omit "X-RAY".
    return "xray_chest"


def load_radiology_text(
    args: argparse.Namespace,
    cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    radiology_path = args.note_root / "radiology.csv.gz"
    detail_path = args.note_root / "radiology_detail.csv.gz"
    require_files((radiology_path, detail_path))

    maps = _cohort_maps(cohort, "hadm_id")
    hadm_ids = set(int(x) for x in cohort["hadm_id"])
    report_pieces: list[pd.DataFrame] = []
    total_rows = missing_storetime = excluded_late = 0

    reader = pd.read_csv(
        radiology_path,
        usecols=[
            "note_id",
            "subject_id",
            "hadm_id",
            "note_type",
            "note_seq",
            "charttime",
            "storetime",
            "text",
        ],
        chunksize=max(100_000, args.chunk_size // 2),
    )
    for chunk in reader:
        total_rows += len(chunk)
        chunk["hadm_id"] = normalize_id(chunk["hadm_id"])
        chunk = chunk[chunk["hadm_id"].isin(hadm_ids)].copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["storetime"] = pd.to_datetime(chunk["storetime"], errors="coerce")
        chunk["intime"] = chunk["hadm_id"].map(maps["intime"])
        chunk["history_cutoff"] = chunk["hadm_id"].map(maps["history_cutoff"])
        chunk["rel_hours"] = (
            chunk["charttime"] - chunk["intime"]
        ).dt.total_seconds() / 3600.0
        chunk = chunk[
            chunk["rel_hours"].between(0, args.context_hours, inclusive="left")
            & chunk["note_type"].isin(["RR", "AR"])
            & chunk["text"].notna()
        ].copy()
        if chunk.empty:
            continue

        missing = chunk["storetime"].isna()
        missing_storetime += int(missing.sum())
        if args.allow_null_note_storetime:
            chunk["available_time"] = chunk["storetime"].fillna(chunk["charttime"])
        else:
            chunk = chunk[~missing].copy()
            chunk["available_time"] = chunk["storetime"]
        late = chunk["available_time"] >= chunk["history_cutoff"]
        excluded_late += int(late.sum())
        chunk = chunk[~late].copy()
        if not chunk.empty:
            report_pieces.append(chunk)

    reports = (
        pd.concat(report_pieces, ignore_index=True)
        if report_pieces
        else pd.DataFrame()
    )
    if reports.empty:
        empty_text = pd.DataFrame(columns=["record_id", "rel_hours", "text"])
        empty_meta = pd.DataFrame(
            columns=[
                "record_id",
                "note_id",
                "note_type",
                "exam_type",
                "exam_name",
                "parent_note_id",
                "charttime",
                "storetime",
                "rel_hours",
            ]
        )
        return empty_text, empty_meta, {
            "radiology_rows_scanned": total_rows,
            "missing_storetime_excluded_or_fallback": missing_storetime,
            "late_reports_excluded": excluded_late,
            "chest_reports_kept": 0,
        }

    note_ids = set(reports["note_id"].astype(str))
    detail_pieces: list[pd.DataFrame] = []
    detail_fields = {
        "exam_name",
        "exam_code",
        "cpt_code",
        "parent_note_id",
        "addendum_note_id",
    }
    for chunk in pd.read_csv(
        detail_path,
        usecols=["note_id", "field_name", "field_value", "field_ordinal"],
        chunksize=max(100_000, args.chunk_size // 2),
    ):
        chunk = chunk[
            chunk["note_id"].astype(str).isin(note_ids)
            & chunk["field_name"].isin(detail_fields)
        ].copy()
        if not chunk.empty:
            detail_pieces.append(chunk)

    details = (
        pd.concat(detail_pieces, ignore_index=True)
        if detail_pieces
        else pd.DataFrame(columns=["note_id", "field_name", "field_value", "field_ordinal"])
    )
    if not details.empty:
        details = details.sort_values(["note_id", "field_name", "field_ordinal"])
        metadata = (
            details.groupby(["note_id", "field_name"])["field_value"]
            .apply(_join_detail_values)
            .unstack("field_name")
            .reset_index()
        )
    else:
        metadata = pd.DataFrame(columns=["note_id"])

    reports = reports.merge(metadata, on="note_id", how="left")
    for column in detail_fields:
        if column not in reports.columns:
            reports[column] = np.nan

    # AR rows can lack exam_name. Propagate metadata from parent_note_id. Some
    # releases also express the relation inversely via addendum_note_id.
    metadata_by_note = metadata.set_index("note_id") if not metadata.empty else None
    inverse_parent: dict[str, str] = {}
    if metadata_by_note is not None and "addendum_note_id" in metadata_by_note.columns:
        for parent_note_id, raw_children in metadata_by_note["addendum_note_id"].dropna().items():
            for child in str(raw_children).split(" || "):
                inverse_parent[child.strip()] = str(parent_note_id)

    reports["parent_note_id_resolved"] = reports["parent_note_id"]
    missing_parent = reports["parent_note_id_resolved"].isna()
    reports.loc[missing_parent, "parent_note_id_resolved"] = reports.loc[
        missing_parent, "note_id"
    ].astype(str).map(inverse_parent)

    if metadata_by_note is not None:
        for column in ("exam_name", "exam_code", "cpt_code"):
            if column not in metadata_by_note.columns:
                continue
            missing_value = reports[column].isna() | reports[column].astype(str).str.strip().eq("")
            parent_values = reports["parent_note_id_resolved"].map(metadata_by_note[column])
            reports.loc[missing_value, column] = parent_values[missing_value]

    reports["exam_type"] = [
        classify_chest_exam(name, code)
        for name, code in zip(reports["exam_name"], reports["exam_code"])
    ]
    reports = reports[reports["exam_type"].notna()].copy()

    reports["clean_text"] = (
        reports["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    reports = reports[reports["clean_text"].str.len().gt(0)].copy()
    exam_name_clean = reports["exam_name"].fillna("UNKNOWN").astype(str).str.replace(
        r"\s+", " ", regex=True
    )
    reports["model_text"] = (
        "[EXAM_TYPE="
        + reports["exam_type"].str.upper()
        + "] [EXAM_NAME="
        + exam_name_clean
        + "] [NOTE_TYPE="
        + reports["note_type"].astype(str)
        + "]\n"
        + reports["clean_text"]
    )
    reports["record_id"] = reports["hadm_id"].astype(int)
    reports = reports.sort_values(["record_id", "rel_hours", "note_seq", "note_id"])

    text_events = reports[["record_id", "rel_hours", "model_text"]].rename(
        columns={"model_text": "text"}
    )
    metadata_columns = [
        "record_id",
        "subject_id",
        "note_id",
        "note_type",
        "exam_type",
        "exam_name",
        "exam_code",
        "cpt_code",
        "parent_note_id_resolved",
        "charttime",
        "storetime",
        "available_time",
        "rel_hours",
    ]
    text_metadata = reports[metadata_columns].rename(
        columns={"parent_note_id_resolved": "parent_note_id"}
    )
    stats = {
        "radiology_rows_scanned": int(total_rows),
        "missing_storetime_excluded_or_fallback": int(missing_storetime),
        "late_reports_excluded": int(excluded_late),
        "chest_reports_kept": int(len(reports)),
        "xray_chest": int(reports["exam_type"].eq("xray_chest").sum()),
        "ct_chest": int(reports["exam_type"].eq("ct_chest").sum()),
        "ultrasound_chest": int(reports["exam_type"].eq("ultrasound_chest").sum()),
    }
    print(
        "Chest history reports kept: "
        f"{len(reports):,} across {reports['record_id'].nunique():,} admissions"
    )
    print(reports["exam_type"].value_counts())
    return text_events, text_metadata, stats


def _count_observed_variables(frame: pd.DataFrame, features: Iterable[str]) -> int:
    features = list(features)
    if frame.empty:
        return 0
    return int(frame[features].notna().any(axis=0).sum())


def _count_direct_variables(frame: pd.DataFrame) -> int:
    features = [feature for feature in FEATURES if feature in DIRECT_RESPIRATORY_FEATURES]
    return _count_observed_variables(frame, features)


def _episode_csv(record_id: int, frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[["rel_hours", *FEATURES]].copy()
    output.insert(
        0,
        "date_time",
        BASE_DATETIME + pd.to_timedelta(output.pop("rel_hours"), unit="h"),
    )
    output.insert(1, "record_id", int(record_id))
    return output


def _text_csv(record_id: int, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date_time", "record_id", "text"])
    output = frame[["rel_hours", "text"]].copy()
    output.insert(
        0,
        "date_time",
        BASE_DATETIME + pd.to_timedelta(output.pop("rel_hours"), unit="h"),
    )
    output.insert(1, "record_id", int(record_id))
    return output[["date_time", "record_id", "text"]]


def write_dataset(
    args: argparse.Namespace,
    cohort: pd.DataFrame,
    numeric: pd.DataFrame,
    text_events: pd.DataFrame,
) -> pd.DataFrame:
    cohort_by_hadm = cohort.set_index("hadm_id")
    text_groups = {
        int(record_id): group.copy()
        for record_id, group in text_events.groupby("record_id")
    }
    stats_rows: list[dict[str, object]] = []
    full_count = strict_count = 0

    for record_id, episode in numeric.groupby("record_id", sort=True):
        record_id = int(record_id)
        episode = episode.sort_values("rel_hours").copy()
        history = episode[
            episode["rel_hours"].between(0, args.context_hours, inclusive="left")
        ]
        target = episode[
            episode["rel_hours"].between(
                args.context_hours,
                args.context_hours + args.prediction_hours,
                inclusive="left",
            )
        ]

        history_observations = int(history[list(FEATURES)].notna().sum().sum())
        target_observations = int(
            target[list(PREDICTION_FEATURES)].notna().sum().sum()
        )
        history_variables = _count_observed_variables(history, FEATURES)
        target_variables = _count_observed_variables(target, PREDICTION_FEATURES)
        direct_history_variables = _count_direct_variables(history)
        direct_target_variables = _count_direct_variables(target)

        numeric_valid = (
            history_observations >= args.min_history_observations
            and target_observations >= args.min_target_observations
            and history_variables >= args.min_history_variables
            and target_variables >= args.min_target_variables
            and direct_history_variables
            >= args.min_direct_respiratory_history_variables
            and direct_target_variables
            >= args.min_direct_respiratory_target_variables
        )

        record_text = text_groups.get(
            record_id,
            pd.DataFrame(columns=["record_id", "rel_hours", "text"]),
        )
        text_count = int(len(record_text))
        text_chars = int(record_text["text"].astype(str).str.len().sum()) if text_count else 0
        cohort_row = cohort_by_hadm.loc[record_id]

        stats_rows.append(
            {
                "subject_id": int(cohort_row["subject_id"]),
                "record_id": record_id,
                "stay_id": int(cohort_row["stay_id"]),
                "numeric_valid": bool(numeric_valid),
                "history_observations": history_observations,
                "target_observations": target_observations,
                "history_variables": history_variables,
                "target_variables": target_variables,
                "direct_respiratory_history_variables": direct_history_variables,
                "direct_respiratory_target_variables": direct_target_variables,
                "text_count_0_24h": text_count,
                "text_chars_0_24h": text_chars,
                "has_chest_text": bool(text_count > 0),
            }
        )
        if not numeric_valid:
            continue

        episode_output = _episode_csv(record_id, episode)
        text_output = _text_csv(record_id, record_text)
        full_dir = args.output_root / "processed_full" / str(record_id)
        full_dir.mkdir(parents=True, exist_ok=True)
        episode_output.to_csv(full_dir / "time_series.csv", index=False)
        text_output.to_csv(full_dir / "text.csv", index=False)
        full_count += 1

        if text_count > 0:
            strict_dir = args.output_root / "processed" / str(record_id)
            strict_dir.mkdir(parents=True, exist_ok=True)
            episode_output.to_csv(strict_dir / "time_series.csv", index=False)
            text_output.to_csv(strict_dir / "text.csv", index=False)
            strict_count += 1

    stats = pd.DataFrame(stats_rows)
    print(f"Numeric-valid processed_full episodes: {full_count:,}")
    print(f"Chest-text processed episodes: {strict_count:,}")
    if full_count:
        valid = stats[stats["numeric_valid"]]
        print(f"Chest-text coverage among numeric-valid episodes: {valid['has_chest_text'].mean():.2%}")
    return stats


def write_metadata(
    args: argparse.Namespace,
    cohort: pd.DataFrame,
    sample_stats: pd.DataFrame,
    text_metadata: pd.DataFrame,
    chartevent_stats: dict[str, int],
    labevent_stats: dict[str, int],
    text_stats: dict[str, int],
) -> None:
    cohort.to_csv(args.output_root / "cohort_summary.csv", index=False)
    sample_stats.to_csv(args.output_root / "sample_statistics.csv", index=False)
    text_metadata.to_csv(args.output_root / "text_event_metadata.csv", index=False)

    feature_dictionary = pd.DataFrame(
        [
            {
                "feature": spec.name,
                "role": spec.role,
                "source": spec.source,
                "itemids": "|".join(str(x) for x in spec.itemids),
                "unit": spec.unit,
                "lower_exclusive": spec.lower,
                "upper_inclusive": spec.upper,
                "description": spec.description,
            }
            for spec in FEATURE_SPECS
        ]
    )
    feature_dictionary["feature_order"] = feature_dictionary["feature"].map(
        {name: index for index, name in enumerate(FEATURES)}
    )
    feature_dictionary = feature_dictionary.sort_values("feature_order")
    feature_dictionary.to_csv(args.output_root / "feature_dictionary.csv", index=False)

    config = {
        "version": "mimic-chest-radiology-v1",
        "context_hours": args.context_hours,
        "prediction_hours": args.prediction_hours,
        "one_episode_per_patient": True,
        "episode_anchor": "ICU intime",
        "numeric_event_time": "charttime",
        "history_numeric_availability": "storetime < ICU intime + context; null falls back to charttime",
        "text_event_time": "charttime",
        "text_availability": (
            "storetime < ICU intime + context; null falls back to charttime"
            if args.allow_null_note_storetime
            else "storetime non-null and < ICU intime + context"
        ),
        "text_modalities": ["xray_chest", "ct_chest", "ultrasound_chest"],
        "note_types": ["RR", "AR"],
        "addendum_policy": "separate timestamped event; inherit parent exam metadata",
        "feature_names": list(FEATURES),
        "prediction_features": list(PREDICTION_FEATURES),
        "history_only_features": list(HISTORY_ONLY_FEATURES),
        "quality_filters": {
            "min_history_observations": args.min_history_observations,
            "min_target_observations": args.min_target_observations,
            "min_history_variables": args.min_history_variables,
            "min_target_variables": args.min_target_variables,
            "min_direct_respiratory_history_variables": args.min_direct_respiratory_history_variables,
            "min_direct_respiratory_target_variables": args.min_direct_respiratory_target_variables,
        },
        "source_statistics": {
            "chartevents": chartevent_stats,
            "labevents": labevent_stats,
            "radiology": text_stats,
        },
        "processed_full_root": str(args.output_root / "processed_full"),
        "processed_multimodal_root": str(args.output_root / "processed"),
    }
    with (args.output_root / "tfsimm_dataset_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    require_files(
        (
            args.mimic_root / "icu" / "d_items.csv.gz",
            args.mimic_root / "hosp" / "d_labitems.csv.gz",
        )
    )
    _prepare_output_root(args.output_root, args.overwrite)
    cohort = build_cohort(args)
    if cohort.empty:
        raise RuntimeError("No eligible ICU stays")

    chartevents, chartevent_stats = load_chartevents(args, cohort)
    labevents, labevent_stats = load_labevents(args, cohort)
    numeric = build_wide_numeric(chartevents, labevents)
    text_events, text_metadata, text_stats = load_radiology_text(args, cohort)
    sample_stats = write_dataset(args, cohort, numeric, text_events)
    write_metadata(
        args,
        cohort,
        sample_stats,
        text_metadata,
        chartevent_stats,
        labevent_stats,
        text_stats,
    )
    print(f"Dataset written to: {args.output_root.resolve()}")
    print("Run the fixed-protocol preparation again because the feature schema changed.")


if __name__ == "__main__":
    main()
