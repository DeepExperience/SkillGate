#!/usr/bin/env python3
"""Rebuild the frozen 1708-record SFT dataset without GPU or old-repo access.

The historical production dataset was assembled from two already-processed LF
components: the 1535-record clean campaign and a 173-record Claw supplement.
The clean component remains reproducible from its JSONL.  The supplement
required teacher inference, so its post-augmentation LF records are frozen next
to the intermediate datasets while the original collection traces remain under
``experiments/sft_data_use``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GeneralAgent.sft_training.export_llamafactory import convert_record


DATASET_NAME = "agent_sft_campaign_20260512_clean_plus_claw_thinkwrap"
OLD_MESSAGES = ROOT / (
    "GeneralAgent/sft_training/datasets/"
    "20260509_sft_campaign_clean_thinkwrap/sft_messages.jsonl"
)
SUPPLEMENT = ROOT / (
    "GeneralAgent/sft_training/datasets/"
    "20260510_claw_supplement_frozen_lf/records.json"
)
CANONICAL_DIR = ROOT / (
    "GeneralAgent/sft_training/llamafactory_data/"
    "20260512_sft_campaign_clean_plus_claw_thinkwrap"
)


def load_old_clean() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for line_number, raw in enumerate(OLD_MESSAGES.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        converted, reason = convert_record(json.loads(raw))
        if converted is None:
            failures.append(f"line {line_number}: {reason}")
        else:
            rows.append(converted)
    if failures:
        raise RuntimeError(f"old-clean conversion failures: {failures[:5]}")
    if len(rows) != 1535:
        raise RuntimeError(f"old-clean count is {len(rows)}, expected 1535")
    return rows


def merge(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source, rows in (("old_clean", old), ("new_claw", new)):
        for original in rows:
            row_text = json.dumps(original.get("messages", []), ensure_ascii=False)
            if "<image>" in row_text:
                continue
            row = dict(original)
            metadata = dict(row.get("metadata") or {})
            key = metadata.get("trajectory_path") or hashlib.sha256(
                row_text.encode()
            ).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            metadata.setdefault("merged_source", source)
            row["metadata"] = metadata
            merged.append(row)
    if len(merged) != 1708:
        raise RuntimeError(f"merged count is {len(merged)}, expected 1708")
    return merged


def dataset_info() -> dict[str, Any]:
    return {
        DATASET_NAME: {
            "file_name": f"{DATASET_NAME}.json",
            "formatting": "openai",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "observation_tag": "tool",
                "function_tag": "function",
                "system_tag": "system",
            },
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--compare-canonical",
        action="store_true",
        help="require byte-identical output to the migrated production dataset",
    )
    args = parser.parse_args()

    supplement = json.loads(SUPPLEMENT.read_text())
    if len(supplement) != 173:
        raise RuntimeError(f"supplement count is {len(supplement)}, expected 173")
    merged = merge(load_old_clean(), supplement)

    out_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{DATASET_NAME}.json"
    info_path = out_dir / "dataset_info.json"
    data_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
    info_path.write_text(json.dumps(dataset_info(), ensure_ascii=False, indent=2) + "\n")

    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    print(f"rebuilt={len(merged)} sha256={digest}")
    if args.compare_canonical:
        canonical_data = CANONICAL_DIR / data_path.name
        canonical_info = CANONICAL_DIR / info_path.name
        if data_path.read_bytes() != canonical_data.read_bytes():
            raise RuntimeError("rebuilt data is not byte-identical to canonical data")
        if info_path.read_bytes() != canonical_info.read_bytes():
            raise RuntimeError("rebuilt dataset_info is not byte-identical to canonical data")
        print("SFT_REBUILD_BYTE_IDENTICAL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
