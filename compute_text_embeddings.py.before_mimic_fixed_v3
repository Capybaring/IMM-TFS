import os
from typing import Iterable

import pandas as pd
import torch
from tqdm import tqdm

from fusions.load_llm import load_llm, embed_notes, get_context_window_size


UNIT_SECONDS = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
    "weeks": 604800.0,
}


def compute_text_embeddings(
    data_name: str,
    llm_model_fusion: str,
    llm_layers_fusion: int | None,
    max_length: int = 1024,
    device: str = "cpu",
    time_unit: str = "days",
    unit_scale: float | None = None,
    align_base_to_series: bool = False,
    episode_anchor_from_day_start: bool = False,
    record_ids: Iterable[str] | None = None,
    max_records: int | None = None,
) -> None:
    """
    Precompute one embedding per text row.

    New arguments for the expanded MIMIC pipeline
    ---------------------------------------------
    episode_anchor_from_day_start:
        Use the beginning of the synthetic numeric episode day as t=0.
        This matches lib/parse_datasets_mimic_expanded.py.

    record_ids:
        Optional explicit subset of record folders.

    max_records:
        Optional prefix limit AFTER sorting/filtering record IDs.  The smoke
        runner uses max_records=200 so text smoke does not embed all 20k cases.

    Notes
    -----
    The expanded MIMIC loader recomputes note timestamps from text.csv at load
    time, so saved rel_times are metadata rather than the sole source of truth.
    They are nevertheless written consistently here.
    """
    if time_unit == "custom":
        if unit_scale is None:
            raise ValueError("Must set unit_scale when time_unit='custom'")
        sec_per_unit = float(unit_scale)
    else:
        try:
            sec_per_unit = UNIT_SECONDS[time_unit]
        except KeyError as exc:
            raise ValueError(f"Unknown time_unit '{time_unit}'") from exc

    base_dir = f"data/{data_name}/processed"
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    discovered = sorted(
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    )

    if record_ids is not None:
        wanted = {str(x) for x in record_ids}
        discovered = [r for r in discovered if r in wanted]

    if max_records is not None:
        max_records = int(max_records)
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        discovered = discovered[:max_records]

    if not discovered:
        raise RuntimeError(f"No record subfolders selected under {base_dir}")

    print(
        f"Text embedding records selected: {len(discovered):,}"
        + (
            f" (max_records={max_records})"
            if max_records is not None
            else ""
        )
    )

    # Load the LLM only after we know there is actual work to consider.
    tokenizer, llm_model = load_llm(
        llm_model_fusion,
        llm_layers_fusion,
        device,
        use_device_map=True,
    )

    for idx, rec in enumerate(discovered):
        print(f"[{idx + 1}/{len(discovered)}] Processing record: {rec}")
        rec_dir = os.path.join(base_dir, rec)
        text_csv = os.path.join(rec_dir, "text.csv")
        if not os.path.isfile(text_csv):
            tqdm.write(f"[SKIP] no text.csv in {rec_dir}")
            continue

        out_name = (
            f"text_embeddings_model={llm_model_fusion}"
            f"_layers={llm_layers_fusion or 'full'}"
            f"_maxlen={max_length}.pt"
        )
        out_path = os.path.join(rec_dir, out_name)

        if os.path.isfile(out_path):
            tqdm.write(
                f"[SKIP] Embeddings already exist for '{rec}', skipping."
            )
            continue

        df = pd.read_csv(text_csv)
        if "date_time" not in df.columns:
            raise ValueError(f"{text_csv}: missing date_time")
        df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

        text_cols = [
            c for c in df.columns if c not in ("date_time", "record_id")
        ]
        if len(text_cols) != 1:
            raise ValueError(f"{rec_dir}: expected 1 text col, got {text_cols}")
        text_col = text_cols[0]

        # Keep exactly one embedding row per original text.csv row.  A bad
        # timestamp is allowed for embedding generation (rel_time -> NaN),
        # because the loader will ignore invalid timestamp rows later.
        notes = df[text_col].fillna("").astype(str).tolist()

        base_ts = df["date_time"].min()

        if episode_anchor_from_day_start or align_base_to_series:
            ts_path = os.path.join(rec_dir, "time_series.csv")
            if os.path.isfile(ts_path):
                ts_series = pd.read_csv(
                    ts_path, usecols=["date_time"]
                )["date_time"]
                ts_series = pd.to_datetime(ts_series, errors="coerce").dropna()
                if not ts_series.empty:
                    if episode_anchor_from_day_start:
                        base_ts = ts_series.min().normalize()
                    else:
                        base_ts = ts_series.min()
            else:
                tqdm.write(
                    f"[WARN] requested numeric-series alignment but no "
                    f"time_series.csv in {rec_dir}; using text timestamp base."
                )

        rel_times = (
            (df["date_time"] - base_ts).dt.total_seconds() / sec_per_unit
        ).tolist()

        embeddings = []
        for note in tqdm(
            notes, desc=f"Notes/{rec}", leave=False, unit="note"
        ):
            emb, _ = embed_notes(
                [[note]],
                tokenizer,
                llm_model,
                max_length=max_length,
            )
            embeddings.append(emb.squeeze(0).squeeze(0).cpu())
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        if embeddings:
            emb_tensor = torch.stack(embeddings, dim=0)
        else:
            emb_tensor = torch.empty((0, 0), dtype=torch.float32)

        torch.save(
            {
                "embeddings": emb_tensor,
                "rel_times": torch.tensor(
                    rel_times, dtype=torch.float32
                ),
            },
            out_path,
        )
        tqdm.write(f"Wrote embeddings to {out_path}")


if __name__ == "__main__":
    data_name_list = [
        "GDELT",
        "RepoHealth",
        "MIMIC",
        "FNSPID",
        "StudentLife",
        "ILINet",
        "CESNET",
        "EPA-Air",
    ]

    llm_model_fusion = "GPT2"
    llm_layers_fusion = None
    max_length = 512 if llm_model_fusion == "BERT" else 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"### LLM model: {llm_model_fusion} ###")

    for data_name in data_name_list:
        print(f"### Processing dataset: {data_name} ###")
        if data_name == "MIMIC":
            compute_text_embeddings(
                data_name,
                llm_model_fusion,
                llm_layers_fusion,
                max_length,
                device,
                time_unit="hours",
                episode_anchor_from_day_start=True,
            )
        else:
            compute_text_embeddings(
                data_name,
                llm_model_fusion,
                llm_layers_fusion,
                max_length,
                device,
            )
