#!/usr/bin/env python3
"""Build an eval plan from the clean RL train/eval parquet files.

This is for full-set evaluation against the exact RL task universe, not the
small eval70/quick30 slices. It emits one independent trial per parquet row and
reuses the SFT collection launcher format so we get the same OpenClaw-aligned
unified runners, result layout, and resume behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SFT_COLLECTION_DIR = PROJECT_ROOT / "GeneralAgent" / "sft_data_collection"
if str(SFT_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_COLLECTION_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import (  # type: ignore  # noqa: E402
    DEFAULT_CONFIG,
    display_path,
    experiment_plan_path,
    filter_known_bad_tasks,
    load_json,
    load_retrieval_task_ids,
    repo_path,
)
from make_trial_plan import make_record, write_jsonl  # type: ignore  # noqa: E402


def _jsonable(value: Any) -> Any:
    """Normalize pandas/numpy objects from parquet extra_info."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _extra_info(value: Any) -> dict[str, Any]:
    value = _jsonable(value)
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"extra_info is not a dict: {type(value)!r}")
    return value


def read_parquet_tasks(path: Path, split_name: str) -> list[dict[str, str]]:
    df = pd.read_parquet(path)
    tasks: list[dict[str, str]] = []
    for row_index, value in enumerate(df["extra_info"]):
        extra = _extra_info(value)
        bench = str(extra.get("bench") or "")
        task_id = str(extra.get("task_id") or "")
        if not bench or not task_id:
            raise ValueError(f"{path}:{row_index} missing bench/task_id in extra_info={extra!r}")
        tasks.append({"bench": bench, "task_id": task_id, "split": split_name})
    return tasks


def load_task_universe(train_parquet: Path, eval_parquet: Path) -> list[dict[str, str]]:
    tasks = read_parquet_tasks(train_parquet, "rl_train")
    tasks.extend(read_parquet_tasks(eval_parquet, "rl_eval"))
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, str]] = []
    for task in tasks:
        key = (task["split"], task["bench"], task["task_id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(task)
    return deduped


def build_plan(
    *,
    config: dict[str, Any],
    tasks: list[dict[str, str]],
    run_id: str,
    date: str,
    model: str,
    arm: str,
    trials: int,
    max_turns: int,
    max_time: int,
    docker_host: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    skill_arms = {"retrieval", "top1_skill_text"}
    if arm not in {"baseline", *skill_arms}:
        raise ValueError(f"unsupported arm {arm!r}; expected baseline, retrieval, or top1_skill_text")

    retrieval_files = config["frozen_retrieval"]["files"]
    retrieval_top_n = int(config["budgets"]["retrieval_top_n"])
    coverage = {
        bench: load_retrieval_task_ids(path)
        for bench, path in retrieval_files.items()
    }

    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for task in tasks:
        bench = task["bench"]
        task_id = task["task_id"]
        if task_id not in filter_known_bad_tasks(bench, [task_id]):
            warnings.append(f"{bench}/{task_id}: skipped by known-bad task exclusions")
            continue
        retrieval_jsonl = retrieval_files.get(bench)
        retrieval_covered = task_id in coverage.get(bench, set())
        if arm in skill_arms and not retrieval_covered:
            warnings.append(f"{bench}/{task_id}: missing retrieval entry in {retrieval_jsonl}")
        for trial_index in range(trials):
            record = make_record(
                run_id=run_id,
                date=date,
                bench=bench,
                task_id=task_id,
                split=task["split"],
                mode=f"eval_{arm}",
                model_role="eval",
                model=model,
                arm=arm,
                trial_index=trial_index,
                retrieval_jsonl=retrieval_jsonl if arm in skill_arms else None,
                retrieval_top_n=retrieval_top_n,
                max_turns=max_turns,
                max_time=max_time,
                retrieval_covered=retrieval_covered if arm in skill_arms else True,
                implicit_mode="",
            )
            env = record.setdefault("env", {})
            env["DOCKER_HOST"] = docker_host
            env["UNIFIED_PROMPT_PROFILE"] = "openclaw_full"
            env["UNIFIED_TOOLS_SCHEMA_MODE"] = "openai_tools"
            env["UNIFIED_CLAW_USE_DOCKER_SANDBOX"] = "1"
            env["UNIFIED_ROLLOUT_WALLCLOCK_CAP_SEC"] = str(max_time)
            env["UNIFIED_VERIFIER_TIMEOUT_CAP_SEC"] = "300"
            env["UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS"] = "1"
            env["UNIFIED_HARBOR_REQUIRE_PREBUILT_LOCAL"] = "1"
            env["AGENT_BENCH_DOCKER_START_CONCURRENCY"] = (
                env.get("AGENT_BENCH_DOCKER_START_CONCURRENCY")
                or os.environ.get("AGENT_BENCH_DOCKER_START_CONCURRENCY")
                or os.environ.get("DOCKER_START_CAP")
                or "8"
            )
            records.append(record)
    return records, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--train-parquet", default="datasets/rl/parquet_4bench_base_20260523/train.parquet")
    parser.add_argument("--eval-parquet", default="datasets/rl/parquet_4bench_base_20260523/eval.parquet")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--model", default="glm-5.1")
    parser.add_argument("--arm", choices=["baseline", "retrieval", "top1_skill_text"], required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-time", type=int, default=850)
    parser.add_argument("--docker-host", default="unix:///tmp/local-docker-overlay2.sock")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    config = load_json(args.config)
    train_parquet = repo_path(args.train_parquet)
    eval_parquet = repo_path(args.eval_parquet)
    tasks = load_task_universe(train_parquet, eval_parquet)
    records, warnings = build_plan(
        config=config,
        tasks=tasks,
        run_id=args.run_id,
        date=args.date,
        model=args.model,
        arm=args.arm,
        trials=args.trials,
        max_turns=args.max_turns,
        max_time=args.max_time,
        docker_host=args.docker_host,
    )

    output_path = repo_path(args.out) if args.out else experiment_plan_path(args.run_id)
    write_jsonl(output_path, records)
    warning_path = output_path.with_suffix(".warnings.txt")
    warning_path.write_text("\n".join(warnings) + ("\n" if warnings else ""), encoding="utf-8")

    counts = Counter((record["split"], record["bench"]) for record in records)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "model": args.model,
                "arm": args.arm,
                "train_parquet": display_path(train_parquet),
                "eval_parquet": display_path(eval_parquet),
                "records": len(records),
                "warnings": len(warnings),
                "counts": {f"{split}/{bench}": count for (split, bench), count in sorted(counts.items())},
                "sample_command": shlex.join(records[0]["argv"]) if records else "",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(records)} records to {display_path(output_path)}")
    print(f"warnings: {len(warnings)} ({display_path(warning_path)})")
    print(f"summary: {display_path(summary_path)}")


if __name__ == "__main__":
    main()
