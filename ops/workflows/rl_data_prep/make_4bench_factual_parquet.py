#!/usr/bin/env python3
"""Create the factual-reward 4bench RL parquet split.

Input is the canonical 5bench RL parquet. Output excludes Claw because Claw's
grader is not the factual verifier setting targeted by the skill-choice
subgroup reward experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_BENCHES = ("seta_synth", "tb2", "sb_ns", "swe_lite")


def bench_of(extra_info: dict) -> str:
    return str((extra_info or {}).get("bench") or "unknown")


def counts_by_bench(frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for extra in frame["extra_info"]:
        bench = bench_of(extra)
        counts[bench] = counts.get(bench, 0) + 1
    return dict(sorted(counts.items()))


def limit_per_bench(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0:
        return frame.reset_index(drop=True)
    parts = []
    for _, group in frame.groupby(frame["extra_info"].map(bench_of), sort=True):
        parts.append(group.head(limit))
    if not parts:
        return frame.iloc[0:0].reset_index(drop=True)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/rl/parquet_4bench_base_20260523"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/rl/parquet_4bench_factual_20260602"),
    )
    parser.add_argument("--bench", action="append", default=[])
    parser.add_argument("--max-train-per-bench", type=int, default=0)
    parser.add_argument("--max-eval-per-bench", type=int, default=0)
    args = parser.parse_args()

    benches = tuple(args.bench or DEFAULT_BENCHES)
    keep = set(benches)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "included_benches": list(benches),
        "files": {},
    }
    for name in ("train.parquet", "eval.parquet"):
        src = args.input_dir / name
        dst = args.output_dir / name
        frame = pd.read_parquet(src)
        mask = frame["extra_info"].map(lambda extra: bench_of(extra) in keep)
        out = frame.loc[mask].reset_index(drop=True)
        limit = args.max_train_per_bench if name == "train.parquet" else args.max_eval_per_bench
        out = limit_per_bench(out, limit)
        out.to_parquet(dst, index=False)
        summary["files"][name] = {
            "input_rows": int(len(frame)),
            "output_rows": int(len(out)),
            "bench_counts_before": counts_by_bench(frame),
            "bench_counts_after": counts_by_bench(out),
        }

    (args.output_dir / "README.md").write_text(
        "# 4bench factual RL parquet\n\n"
        "Generated for skill-choice/GIGPO-style reward shaping. Claw is excluded.\n\n"
        + "```json\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
