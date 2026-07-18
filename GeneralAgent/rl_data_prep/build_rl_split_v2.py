#!/usr/bin/env python3
"""Build RL train/eval split — v2 (full train pool + 10% even-spaced holdout).

Unlike :mod:`build_rl_split` v1 (which restricted RL eval to SFT-unseen
tasks), v2 simply takes each bench's full SFT train pool and reserves a
small even-spaced ~10% slice as the RL holdout. This brings RL train above
500 tasks total without complicating prompt generation: the SFT-seen vs
unseen distinction is recorded in metadata for analysis but does not gate
inclusion.

Output mirrors :mod:`build_rl_split` v1 schema.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ._imports import split_even
from GeneralAgent.task_exclusions import bad_reason, is_bad_task


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SFT_SPLIT = PROJECT_ROOT / "GeneralAgent/sft_data_collection/outputs/splits/default/holdout_split.json"
DEFAULT_SFT_DATA = PROJECT_ROOT / (
    "GeneralAgent/sft_training/llamafactory_data/"
    "20260512_sft_campaign_clean_plus_claw_thinkwrap/"
    "agent_sft_campaign_20260512_clean_plus_claw_thinkwrap.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets/rl/rl_split_v2.json"


def _extract_sft_seen_tasks(sft_data_path: Path) -> dict[str, set[str]]:
    seen: dict[str, set[str]] = defaultdict(set)
    records = json.loads(sft_data_path.read_text(encoding="utf-8"))
    for rec in records:
        meta = rec.get("metadata") or rec.get("extra") or {}
        b, t = meta.get("bench"), meta.get("task_id")
        if b and t:
            seen[b].add(t)
    return seen


def _split_one(bench: str, train_tasks: list[str], seen_ids: set[str], frac: float):
    eval_n = max(1, int(round(len(train_tasks) * frac)))
    eval_n = min(eval_n, len(train_tasks) - 1)  # always leave at least 1 train task
    rl_train, rl_eval = split_even(train_tasks, eval_n)
    excluded = [
        {"task_id": task_id, "reason": bad_reason(bench, task_id)}
        for task_id in [*rl_train, *rl_eval]
        if is_bad_task(bench, task_id)
    ]
    if excluded:
        rl_train = [task_id for task_id in rl_train if not is_bad_task(bench, task_id)]
        rl_eval = [task_id for task_id in rl_eval if not is_bad_task(bench, task_id)]
    overlap_seen = seen_ids & set(rl_train)
    return rl_train, rl_eval, {
        "sft_train_size": len(train_tasks),
        "sft_seen_count": len(seen_ids & set(train_tasks)),
        "rl_train_size": len(rl_train),
        "rl_eval_size": len(rl_eval),
        "rl_train_includes_sft_seen": len(overlap_seen),
        "rl_eval_includes_sft_seen": len(seen_ids & set(rl_eval)),
        "excluded_count": len(excluded),
        "excluded": excluded,
    }


def build(sft_split_path: Path, sft_data_path: Path, frac: float) -> dict:
    sft_split = json.loads(sft_split_path.read_text(encoding="utf-8"))
    seen_by_bench = _extract_sft_seen_tasks(sft_data_path)

    rl_benches: dict[str, dict] = {}
    stats_by_bench: dict[str, dict] = {}
    for bench, split in sft_split["benches"].items():
        rl_train, rl_eval, stats = _split_one(
            bench=bench,
            train_tasks=split["train"],
            seen_ids=seen_by_bench.get(bench, set()),
            frac=frac,
        )
        rl_benches[bench] = {"rl_train": rl_train, "rl_eval": rl_eval}
        stats_by_bench[bench] = stats

    for bench, split in rl_benches.items():
        if set(split["rl_train"]) & set(split["rl_eval"]):
            raise AssertionError(f"split bug: rl_train ∩ rl_eval not empty for {bench}")

    return {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "rl_eval_definition": (
                "Even-spaced 10% holdout from the full SFT train pool per bench. "
                "RL eval may overlap with SFT-seen tasks (recorded in stats)."
            ),
            "rl_train_definition": "Full SFT train pool minus RL eval.",
            "eval_fraction": frac,
        },
        "benches": rl_benches,
        "stats": stats_by_bench,
        "sources": {
            "sft_split_json": str(sft_split_path),
            "sft_data_json": str(sft_data_path),
        },
    }


def write_per_bench_lists(payload, output_dir: Path) -> None:
    train_dir = output_dir / "rl_train"
    eval_dir = output_dir / "rl_eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    for bench, split in payload["benches"].items():
        (train_dir / f"{bench}.txt").write_text(
            "\n".join(split["rl_train"]) + "\n", encoding="utf-8"
        )
        (eval_dir / f"{bench}.txt").write_text(
            "\n".join(split["rl_eval"]) + "\n", encoding="utf-8"
        )


def print_summary(payload):
    print()
    print("RL split v2 summary")
    print("=" * 80)
    fmt = "{:<14} {:>10} {:>10} {:>10} {:>10} {:>14} {:>14} {:>10}"
    print(fmt.format("bench", "sft_train", "sft_seen", "rl_train", "rl_eval",
                     "rl_train_seen", "rl_eval_seen", "excluded"))
    print("-" * 80)
    totals = {"sft_train": 0, "sft_seen": 0, "rl_train": 0, "rl_eval": 0,
              "rl_train_seen": 0, "rl_eval_seen": 0, "excluded": 0}
    for bench, st in payload["stats"].items():
        print(fmt.format(
            bench,
            st["sft_train_size"], st["sft_seen_count"],
            st["rl_train_size"], st["rl_eval_size"],
            st["rl_train_includes_sft_seen"], st["rl_eval_includes_sft_seen"],
            st["excluded_count"],
        ))
        totals["sft_train"] += st["sft_train_size"]
        totals["sft_seen"] += st["sft_seen_count"]
        totals["rl_train"] += st["rl_train_size"]
        totals["rl_eval"] += st["rl_eval_size"]
        totals["rl_train_seen"] += st["rl_train_includes_sft_seen"]
        totals["rl_eval_seen"] += st["rl_eval_includes_sft_seen"]
        totals["excluded"] += st["excluded_count"]
    print("-" * 80)
    print(fmt.format("TOTAL", totals["sft_train"], totals["sft_seen"],
                     totals["rl_train"], totals["rl_eval"],
                     totals["rl_train_seen"], totals["rl_eval_seen"],
                     totals["excluded"]))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sft-split-json", default=DEFAULT_SFT_SPLIT)
    parser.add_argument("--sft-data-json", default=DEFAULT_SFT_DATA)
    parser.add_argument("--eval-fraction", type=float, default=0.10,
                        help="Per-bench RL eval fraction (default 0.10).")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build(Path(args.sft_split_json), Path(args.sft_data_json), args.eval_fraction)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_per_bench_lists(payload, out_path.parent)
    print_summary(payload)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
