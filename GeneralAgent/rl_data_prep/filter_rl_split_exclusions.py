#!/usr/bin/env python3
"""Apply central task exclusions to an existing RL split without resampling.

Use this when a checkpoint should resume against the same train/eval assignment
after removing newly confirmed broken tasks. It preserves all non-excluded task
membership and order, unlike rebuilding the split from source.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GeneralAgent.task_exclusions import bad_reason, is_bad_task


DEFAULT_INPUT = PROJECT_ROOT / "datasets/rl/rl_split_v2.json"
DEFAULT_OUTPUT = DEFAULT_INPUT


def _extract_sft_seen_tasks(sft_data_path: Path) -> dict[str, set[str]]:
    seen: dict[str, set[str]] = defaultdict(set)
    if not sft_data_path.is_file():
        return seen
    records = json.loads(sft_data_path.read_text(encoding="utf-8"))
    for record in records:
        meta = record.get("metadata") or record.get("extra") or {}
        bench, task_id = meta.get("bench"), meta.get("task_id")
        if bench and task_id:
            seen[str(bench)].add(str(task_id))
    return seen


def _filter_list(bench: str, split_name: str, task_ids: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    kept: list[str] = []
    excluded: list[dict[str, str]] = []
    for task_id in task_ids:
        task_id = str(task_id)
        if is_bad_task(bench, task_id):
            excluded.append(
                {
                    "split": split_name,
                    "task_id": task_id,
                    "reason": bad_reason(bench, task_id),
                }
            )
        else:
            kept.append(task_id)
    return kept, excluded


def filter_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sources = payload.get("sources") or {}
    sft_data_path = Path(str(sources.get("sft_data_json", "")))
    seen_by_bench = _extract_sft_seen_tasks(sft_data_path)

    total_excluded: list[dict[str, str]] = []
    for bench, split in payload["benches"].items():
        train, excluded_train = _filter_list(bench, "rl_train", split.get("rl_train", []))
        eval_, excluded_eval = _filter_list(bench, "rl_eval", split.get("rl_eval", []))
        split["rl_train"] = train
        split["rl_eval"] = eval_
        excluded = excluded_train + excluded_eval
        total_excluded.extend({"bench": bench, **item} for item in excluded)

        seen_ids = seen_by_bench.get(bench, set())
        stats = payload.setdefault("stats", {}).setdefault(bench, {})
        stats["rl_train_size"] = len(train)
        stats["rl_eval_size"] = len(eval_)
        stats["rl_train_includes_sft_seen"] = len(seen_ids & set(train))
        stats["rl_eval_includes_sft_seen"] = len(seen_ids & set(eval_))
        stats["excluded_count"] = len(excluded)
        stats["excluded"] = excluded

    payload.setdefault("policy", {})["central_task_exclusions_applied"] = True
    payload["policy"]["central_task_exclusions_applied_at"] = datetime.now(timezone.utc).isoformat()
    payload["policy"]["central_task_exclusions_policy"] = (
        "Applied to an existing split without resampling, preserving all non-excluded "
        "train/eval membership and ordering for checkpoint-resume stability."
    )
    payload["excluded_tasks"] = total_excluded
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload = filter_payload(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {output_path}")
    print("excluded:")
    for item in payload.get("excluded_tasks", []):
        print(f"  {item['bench']}/{item['task_id']} from {item['split']}: {item['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
