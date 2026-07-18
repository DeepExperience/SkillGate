#!/usr/bin/env python3
# NOTE: Migrated workflow helper copy. Source: experiments/rl/v2/launch/filter_parquet_tasks.py
# Original historical script is archived during workflow cleanup; maintain this copy going forward.
"""Filter RL train/eval parquet rows by bench/task id.

This is intentionally small and explicit: the RL resume scripts can use it to
exclude known infra-heavy tasks without mutating the canonical parquet files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def row_key(extra_info: dict) -> str:
    return f"{extra_info.get('bench')}/{extra_info.get('task_id')}"


def counts_by_bench(frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for extra_info in frame["extra_info"]:
        bench = extra_info.get("bench", "unknown")
        counts[bench] = counts.get(bench, 0) + 1
    return dict(sorted(counts.items()))


def filter_frame(frame: pd.DataFrame, excluded: set[str]) -> tuple[pd.DataFrame, list[str]]:
    keys = frame["extra_info"].map(row_key)
    removed = sorted(set(keys[keys.isin(excluded)].tolist()))
    return frame.loc[~keys.isin(excluded)].reset_index(drop=True), removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Task key to exclude, formatted as bench/task_id. Repeatable.",
    )
    parser.add_argument("--reason", default="")
    args = parser.parse_args()

    excluded = set(args.exclude)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "reason": args.reason,
        "excluded_requested": sorted(excluded),
        "files": {},
    }

    for name in ("train.parquet", "eval.parquet"):
        input_path = args.input_dir / name
        output_path = args.output_dir / name
        frame = pd.read_parquet(input_path)
        filtered, removed = filter_frame(frame, excluded)
        filtered.to_parquet(output_path, index=False)
        summary["files"][name] = {
            "input_rows": int(len(frame)),
            "output_rows": int(len(filtered)),
            "removed_rows": int(len(frame) - len(filtered)),
            "removed": removed,
            "bench_counts_before": counts_by_bench(frame),
            "bench_counts_after": counts_by_bench(filtered),
        }

    readme = args.output_dir / "README.md"
    readme.write_text(
        "# RL parquet task filter\n\n"
        f"- input_dir: `{args.input_dir}`\n"
        f"- output_dir: `{args.output_dir}`\n"
        f"- reason: {args.reason or '(none)'}\n\n"
        "Excluded tasks requested:\n"
        + "".join(f"- `{item}`\n" for item in sorted(excluded))
        + "\n```json\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
