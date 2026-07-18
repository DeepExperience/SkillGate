#!/usr/bin/env python3
"""Create a fresh eval70 retrieval plan from a previously validated eval70 plan.

This preserves the exact bench/task order and retrieval files from the source
plan while rewriting run_id, output root, model id, API base, and Docker host.
Use it for repeatability checks where completed historical result paths must not
be reused.

Each source task is expanded into --repeats trials (default 4, trial_index
0..N-1, t00..t03 paths) inside ONE plan/run root, so all repeats run against
the same SGLang process. A single eval70 pass proved too noisy to resolve
~5pp deltas (Fisher p≈0.45 on 70 tasks); report pass@1 as resolved trials /
total trials across repeats.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SFT_COLLECTION_DIR = PROJECT_ROOT / "GeneralAgent" / "sft_data_collection"
if str(SFT_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_COLLECTION_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import display_path, repo_path  # type: ignore  # noqa: E402
from make_trial_plan import make_record, write_jsonl  # type: ignore  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def infer_max_turns(record: dict[str, Any], default: int) -> int:
    argv = [str(part) for part in record.get("argv") or []]
    for index, token in enumerate(argv):
        if token == "--max-turns" and index + 1 < len(argv):
            return int(argv[index + 1])
    return default


def infer_max_time(record: dict[str, Any], default: int) -> int:
    argv = [str(part) for part in record.get("argv") or []]
    for index, token in enumerate(argv):
        if token == "--max-time" and index + 1 < len(argv):
            return int(argv[index + 1])
    env = record.get("env") or {}
    return int(env.get("UNIFIED_ROLLOUT_WALLCLOCK_CAP_SEC") or default)


def build_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    source_rows = read_jsonl(repo_path(args.source_plan))
    repeats = max(1, args.repeats)
    expanded: list[tuple[dict[str, Any], int]] = []
    for source in source_rows:
        for repeat_index in range(repeats):
            trial_index = (
                repeat_index if repeats > 1 else int(source.get("trial_index") or 0)
            )
            expanded.append((source, trial_index))
    records: list[dict[str, Any]] = []
    for source, trial_index in expanded:
        record = make_record(
            run_id=args.run_id,
            date=args.date,
            bench=str(source["bench"]),
            task_id=str(source["task_id"]),
            split=str(source.get("split") or "eval"),
            mode=str(source.get("mode") or "eval_retrieval"),
            model_role=str(source.get("model_role") or "eval"),
            model=args.model,
            arm=str(source.get("arm") or "retrieval"),
            trial_index=trial_index,
            retrieval_jsonl=str(source.get("retrieval_jsonl") or ""),
            retrieval_top_n=int(source.get("retrieval_top_n") or args.retrieval_top_n),
            max_turns=infer_max_turns(source, args.max_turns),
            max_time=infer_max_time(source, args.max_time),
            retrieval_covered=bool(source.get("retrieval_covered", True)),
            implicit_mode=str(source.get("implicit_mode") or ""),
        )
        env = record.setdefault("env", {})
        intended_env = dict(env)
        source_env = source.get("env") or {}
        for key, value in source_env.items():
            if value is not None:
                env[str(key)] = str(value)
        env["OPENAI_API_BASE"] = args.api_base.rstrip("/")
        env["DOCKER_HOST"] = args.docker_host
        env["AGENT_BENCH_DOCKER_START_CONCURRENCY"] = str(args.docker_start_cap)
        env["EXPERIMENT_ROOT"] = display_path(args.run_root)
        for key in (
            "UNIFIED_RESULTS_DATE",
            "UNIFIED_RUN_ID",
            "UNIFIED_EXP_VERSION",
            "UNIFIED_MODEL",
            "PHASE_B_MODEL",
        ):
            if key in intended_env:
                env[key] = str(intended_env[key])

        argv = record.get("argv")
        if isinstance(argv, list):
            for index, token in enumerate(argv):
                if token == "--api-base" and index + 1 < len(argv):
                    argv[index + 1] = args.api_base.rstrip("/")
            record["command_preview"] = " ".join(str(part) for part in argv)
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--docker-host", default="tcp://127.0.0.1:2376")
    parser.add_argument("--docker-start-cap", type=int, default=int(os.environ.get("DOCKER_START_CAP", "8")))
    parser.add_argument("--retrieval-top-n", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-time", type=int, default=850)
    parser.add_argument("--repeats", type=int, default=4,
                        help="Trials per task within one plan/run root (default 4). "
                             "One eval70 pass cannot resolve ~5pp deltas; keep all "
                             "repeats in the same SGLang process. 1 = legacy single pass.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.environ["EXPERIMENT_ROOT"] = display_path(args.run_root)
    records = build_plan(args)
    output_path = repo_path(args.out)
    write_jsonl(output_path, records)
    summary = {
        "run_id": args.run_id,
        "run_root": display_path(args.run_root),
        "source_plan": display_path(args.source_plan),
        "model": args.model,
        "api_base": args.api_base.rstrip("/"),
        "docker_host": args.docker_host,
        "repeats": max(1, args.repeats),
        "records": len(records),
        "tasks": len({(record["bench"], str(record["task_id"])) for record in records}),
        "counts": dict(Counter(record["bench"] for record in records)),
        "sample_command": records[0]["command_preview"] if records else "",
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(records)} records to {display_path(output_path)}")
    print(f"summary: {display_path(output_path.with_suffix('.summary.json'))}")


if __name__ == "__main__":
    main()
