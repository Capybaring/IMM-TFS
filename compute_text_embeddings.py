import os
import torch
import pandas as pd
from tqdm import tqdm
from fusions.load_llm import load_llm, embed_notes, get_context_window_size

# Mirrors ChunkedTimeSeriesDataset.UNIT_SECONDS in lib/parse_datasets.py. Kept
# as a separate copy (rather than importing lib.parse_datasets) to avoid
# pulling torch Dataset / argparse machinery into this standalone script.
# If you touch one, touch the other.
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
) -> None:
    """
    Loop over all records in base_dir, read each text.csv, embed notes one at a time,
    and save text_embeddings_{llm_model_fusion}_{llm_layers_fusion or 'full'}.pt

    Args:
      data_name: name of the dataset (e.g. 'ILINet', 'FNSPID')
      llm_model_fusion: key or model ID (e.g. 'GPT2')
      llm_layers_fusion: number of layers to keep, or None for all
      max_length: maximum length of input tokens
      device: 'cpu' or 'cuda'
      time_unit: unit the saved `rel_times` should be expressed in. MUST match
        the `--time_unit` that main.py/update_args_for_dataset() uses for this
        dataset (e.g. "hours" for MIMIC), or downstream TTF/MMF fusion will
        place notes at the wrong relative time. Defaults to "days" to match
        this function's original (buggy) hardcoded behavior for any caller
        that doesn't opt in.
      unit_scale: seconds-per-unit to use when time_unit == "custom".
      align_base_to_series: if True, the t=0 reference point for a record's
        notes is that record's time_series.csv earliest timestamp (matching
        exactly what lib/parse_datasets.py uses when it loads numeric data
        and raw-text fallback notes). If False (default, original behavior),
        the reference point is the record's own text.csv earliest note time,
        which is inconsistent with how the training pipeline aligns text to
        the numeric series and silently shifts every note's relative time.
    """
    if time_unit == "custom":
        if unit_scale is None:
            raise ValueError("Must set unit_scale when time_unit='custom'")
        sec_per_unit = float(unit_scale)
    else:
        try:
            sec_per_unit = UNIT_SECONDS[time_unit]
        except KeyError:
            raise ValueError(f"Unknown time_unit '{time_unit}'")
    base_dir = f"data/{data_name}/processed"
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # Load LLM once
    tokenizer, llm_model = load_llm(
        llm_model_fusion,
        llm_layers_fusion,
        device,
        # use_device_map=False,
        use_device_map=True,
    )

    # Discover all record subfolders
    record_ids = sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    )
    if not record_ids:
        raise RuntimeError(f"No record subfolders under {base_dir}")

    # Iterate records with progress bar
    for idx, rec in enumerate(record_ids):
        print(f"[{idx + 1}/{len(record_ids)}] Processing record: {rec}")
        rec_dir = os.path.join(base_dir, rec)
        text_csv = os.path.join(rec_dir, "text.csv")
        if not os.path.isfile(text_csv):
            tqdm.write(f"[SKIP] no text.csv in {rec_dir}")
            continue

        # Prepare output path
        out_name = (
            f"text_embeddings_model={llm_model_fusion}"
            f"_layers={llm_layers_fusion or 'full'}"
            f"_maxlen={max_length}.pt"
        )
        out_path = os.path.join(rec_dir, out_name)

        # Skip if output already exists
        if os.path.isfile(out_path):
            tqdm.write(f"[SKIP] Embeddings already exist for '{rec}', skipping.")
            continue

        tqdm.write(f"Embedding notes for record '{rec}'...")
        df = pd.read_csv(text_csv, parse_dates=["date_time"])

        base_ts = df["date_time"].min()
        if align_base_to_series:
            ts_path = os.path.join(rec_dir, "time_series.csv")
            if os.path.isfile(ts_path):
                ts_base = pd.read_csv(ts_path, usecols=["date_time"])["date_time"]
                base_ts = pd.to_datetime(ts_base).min()
            else:
                tqdm.write(
                    f"[WARN] align_base_to_series=True but no time_series.csv in "
                    f"{rec_dir}; falling back to text.csv's own min timestamp for '{rec}'."
                )

        rel_times = ((df["date_time"] - base_ts).dt.total_seconds() / sec_per_unit).tolist()
        text_cols = [c for c in df.columns if c not in ("date_time", "record_id")]
        if len(text_cols) != 1:
            raise ValueError(f"{rec_dir}: expected 1 text col, got {text_cols}")
        notes = df[text_cols[0]].astype(str).tolist()

        # Embed each note one by one to save memory
        embeddings = []
        for note in tqdm(notes, desc=f"Notes/{rec}", leave=False, unit="note"):
            emb, _ = embed_notes([[note]], tokenizer, llm_model, max_length=max_length)
            embeddings.append(emb.squeeze(0).squeeze(0).cpu())
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        # Stack into Tensor [N_notes, d_txt]
        if embeddings:
            emb_tensor = torch.stack(embeddings, dim=0)
        else:
            emb_tensor = torch.empty((0, 0), dtype=torch.float32)

        # Save embeddings and rel_times
        torch.save(
            {
                "embeddings": emb_tensor,
                "rel_times": torch.tensor(rel_times, dtype=torch.float32),
            },
            out_path,
        )
        tqdm.write(f"Wrote embeddings to {out_path}")


if __name__ == "__main__":
    # # * GPU settings
    # gpu_id = 0
    # # gpu_id = 1
    # # gpu_id = 2
    # # gpu_id = 3
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # * Parameters
    data_name_list = [
        "GDELT",  # type 1.1
        "RepoHealth",  # type 1.2
        "MIMIC",  # type 1.3
        "FNSPID",  # type 2.1
        # "ClusterTrace",  # type 2.2
        "StudentLife",  # type 2.3
        "ILINet",  # type 3.1
        "CESNET",  # type 3.2
        "EPA-Air",  # type 3.3
    ]

    llm_model_fusion = "GPT2"
    # llm_model_fusion = "GPT2XL"
    # llm_model_fusion = "BERT"
    # llm_model_fusion = "Llama"
    # llm_model_fusion = "DeepSeek"
    # llm_layers_fusion = 6
    llm_layers_fusion = None
    max_length = 512 if llm_model_fusion == "BERT" else 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"### LLM model: {llm_model_fusion} ###")

    # * Update max_length if needed
    # context_window_size = get_context_window_size(llm_model_fusion, device)
    # if max_length > context_window_size:
    #     print(
    #         f"Overriding max_length from {max_length} to {context_window_size}"
    #         " to match the LLM model's context window size."
    #     )
    #     max_length = context_window_size

    for data_name in data_name_list:
        print(f"### Processing dataset: {data_name} ###")
        # NOTE (2026-08-08): compute_text_embeddings()'s time_unit/
        # align_base_to_series default to the function's original ("days",
        # text.csv-based base) behavior for every dataset. This is only
        # verified correct for MIMIC so far (time_unit="hours", per
        # update_args_for_dataset() in main.py) — wired in explicitly below.
        # ILINet ("weeks") and any other non-"days" dataset in this list
        # likely has the same bug and hasn't been reviewed/fixed yet.
        if data_name == "MIMIC":
            compute_text_embeddings(
                data_name, llm_model_fusion, llm_layers_fusion, max_length, device,
                time_unit="hours", align_base_to_series=True,
            )
        else:
            compute_text_embeddings(
                data_name, llm_model_fusion, llm_layers_fusion, max_length, device
            )
