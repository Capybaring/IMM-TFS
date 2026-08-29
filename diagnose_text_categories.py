#!/usr/bin/env python3
"""Diagnose anatomical categories in processed MIMIC radiology reports.

The script is read-only: it scans ``<data-dir>/<record_id>/text.csv`` and
prints cohort-level report, admission, and anatomical-category statistics.
It does not import the training pipeline and does not modify processed data.

Anatomical regions are inferred from EXAMINATION/EXAM/PROCEDURE/STUDY headers.
When those headers are absent, the first few non-empty report lines are used.
The result is a transparent rule-based prediagnostic, not a clinical label.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


EXAM_HEADER_RE = re.compile(
    r"(?im)^\s*(?:examination|exam|procedure|study|type of exam|"
    r"radiographic examination)\s*:\s*(.+?)\s*$"
)

REGION_PATTERNS = {
    "chest_lung": re.compile(
        r"\b(chest|thorax|lung|pulmonary|rib|sternum)\b", re.I
    ),
    "head_brain": re.compile(
        r"\b(head|brain|cranial|skull|facial|face|orbit|sinus)\b", re.I
    ),
    "neck": re.compile(r"\b(neck|thyroid)\b", re.I),
    "abdomen_pelvis": re.compile(
        r"\b(abdomen|abdominal|pelvis|pelvic|kub)\b", re.I
    ),
    "spine": re.compile(
        r"\b(spine|c[- ]?spine|t[- ]?spine|l[- ]?spine|"
        r"cervical spine|thoracic spine|thoracolumbar|lumbar|lumbosacral|"
        r"sacrum|coccyx)\b",
        re.I,
    ),
    "upper_extremity": re.compile(
        r"\b(shoulder|clavicle|humerus|elbow|forearm|wrist|hand|finger)\b",
        re.I,
    ),
    "lower_extremity": re.compile(
        r"\b(hip|femur|knee|tibia|fibula|ankle|foot|toe)\b", re.I
    ),
    "breast": re.compile(r"\b(breast|mammogra)\w*\b", re.I),
    "whole_body": re.compile(r"\b(whole body|pet[- ]?ct)\b", re.I),
}

VASCULAR_RE = re.compile(
    r"\b(angiograph|angiogram|venogram|arteriogram|vascular)\w*\b", re.I
)


def _set_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _resolve_data_dir(value: str | None) -> Path:
    repo_root = Path(__file__).resolve().parent
    if value:
        candidates = [Path(value).expanduser()]
        if not candidates[0].is_absolute():
            candidates.append(repo_root / candidates[0])
    else:
        candidates = [
            repo_root / "data" / "MIMIC" / "processed",
            repo_root / "processed",
            repo_root / "data" / "MIMIC" / "processed_full",
            repo_root / "processed_full",
        ]

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved

    checked = "\n  - ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(f"No processed MIMIC directory found. Checked:\n  - {checked}")


def _extract_exam_descriptor(text: str) -> str:
    match = EXAM_HEADER_RE.search(text)
    if match:
        return match.group(1).strip()[:300]

    lines = [
        line.strip()
        for line in text.splitlines()[:20]
        if line.strip() and line.strip() != "___"
    ]
    return " ".join(lines[:3])[:300]


def _extract_regions(descriptor: str) -> tuple[str, ...]:
    regions = [
        region
        for region, pattern in REGION_PATTERNS.items()
        if pattern.search(descriptor)
    ]
    if not regions and VASCULAR_RE.search(descriptor):
        regions.append("vascular_other")
    return tuple(regions)


def _category_label(regions: tuple[str, ...]) -> str:
    if not regions:
        return "unclassified"
    if len(regions) == 1:
        return regions[0]
    return "combined:" + "+".join(regions)


def _percentage(count: int, total: int) -> str:
    return "0.00%" if total == 0 else f"{100.0 * count / total:.2f}%"


def _print_counter(title: str, values: Counter, total: int) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'category':<48} {'count':>10} {'percent':>10}")
    for name, count in values.most_common():
        print(f"{name:<48} {count:>10,} {_percentage(count, total):>10}")


def diagnose(data_dir: Path, example_limit: int) -> None:
    _set_csv_field_limit()

    record_dirs = sorted(
        path for path in data_dir.iterdir() if path.is_dir()
    )
    report_count = 0
    records_with_text = 0
    unclassified_count = 0
    multi_region_count = 0

    reports_per_record = Counter()
    category_count_per_record = Counter()
    primary_report_counts = Counter()
    region_report_counts = Counter()
    region_record_ids: dict[str, set[str]] = defaultdict(set)
    unclassified_examples: list[str] = []

    for index, record_dir in enumerate(record_dirs, start=1):
        text_path = record_dir / "text.csv"
        if not text_path.is_file():
            continue

        record_reports = 0
        record_regions: set[str] = set()
        with text_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "text" not in reader.fieldnames:
                raise ValueError(f"{text_path}: missing text column")

            for row in reader:
                text = (row.get("text") or "").strip()
                if not text:
                    continue

                descriptor = _extract_exam_descriptor(text)
                regions = _extract_regions(descriptor)
                category = _category_label(regions)

                report_count += 1
                record_reports += 1
                primary_report_counts[category] += 1

                if len(regions) > 1:
                    multi_region_count += 1
                if not regions:
                    unclassified_count += 1
                    if len(unclassified_examples) < example_limit:
                        unclassified_examples.append(descriptor or "<empty>")

                for region in regions:
                    region_report_counts[region] += 1
                    region_record_ids[region].add(record_dir.name)
                    record_regions.add(region)

        if record_reports:
            records_with_text += 1
            reports_per_record[record_reports] += 1
            category_count_per_record[len(record_regions)] += 1

        if index % 2000 == 0:
            print(
                f"Scanned {index:,}/{len(record_dirs):,} record directories...",
                file=sys.stderr,
            )

    if report_count == 0:
        raise RuntimeError(f"No non-empty reports found under {data_dir}")

    classified_records = sum(
        count
        for category_count, count in category_count_per_record.items()
        if category_count > 0
    )
    multi_category_records = sum(
        count
        for category_count, count in category_count_per_record.items()
        if category_count >= 2
    )
    mean_reports = report_count / max(records_with_text, 1)
    category_total = sum(
        category_count * count
        for category_count, count in category_count_per_record.items()
    )
    mean_categories = category_total / max(classified_records, 1)

    print("\nMIMIC anatomical text-category prediagnostic")
    print("=" * 45)
    print(f"processed directory                 : {data_dir}")
    print(f"record directories scanned          : {len(record_dirs):,}")
    print(f"records with >=1 non-empty report   : {records_with_text:,}")
    print(f"non-empty reports                   : {report_count:,}")
    print(f"mean reports per text record        : {mean_reports:.3f}")
    print(f"detected base anatomical categories : {len(region_report_counts):,}")
    print(
        "multi-region reports                : "
        f"{multi_region_count:,} ({_percentage(multi_region_count, report_count)})"
    )
    print(
        "unclassified reports                : "
        f"{unclassified_count:,} ({_percentage(unclassified_count, report_count)})"
    )
    print(f"mean categories/classified record   : {mean_categories:.3f}")
    print(
        "records with >=2 categories         : "
        f"{multi_category_records:,} "
        f"({_percentage(multi_category_records, classified_records)})"
    )

    _print_counter(
        "Primary report category (combined reports remain explicit)",
        primary_report_counts,
        report_count,
    )
    _print_counter(
        "Base-region report assignments (multi-label)",
        region_report_counts,
        sum(region_report_counts.values()),
    )

    print("\nBase-region record coverage")
    print("---------------------------")
    print(f"{'category':<32} {'records':>10} {'percent':>10}")
    for region, record_ids in sorted(
        region_record_ids.items(), key=lambda item: len(item[1]), reverse=True
    ):
        print(
            f"{region:<32} {len(record_ids):>10,} "
            f"{_percentage(len(record_ids), records_with_text):>10}"
        )

    _print_counter(
        "Reports per text record",
        reports_per_record,
        records_with_text,
    )
    _print_counter(
        "Anatomical categories per text record",
        category_count_per_record,
        records_with_text,
    )

    if unclassified_examples:
        print(f"\nFirst {len(unclassified_examples)} unclassified descriptors")
        print("-" * 48)
        for descriptor in unclassified_examples:
            print(f"- {descriptor}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read processed MIMIC text.csv files and diagnose anatomical "
            "report categories without modifying the dataset."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Processed record directory. By default, search "
            "data/MIMIC/processed, processed, data/MIMIC/processed_full, "
            "then processed_full."
        ),
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=20,
        help="Maximum number of unclassified report descriptors to print.",
    )
    args = parser.parse_args()
    if args.examples < 0:
        parser.error("--examples must be >= 0")

    diagnose(_resolve_data_dir(args.data_dir), args.examples)


if __name__ == "__main__":
    main()
