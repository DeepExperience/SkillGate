#!/usr/bin/env python3
"""Watch a global Phase 1 launch and enqueue teacher fallback per completed task.

This removes the old chunk tail problem:
  - Phase 1 can run one large dynamic queue, so idle workers immediately pick
    the next task while respecting launch_trials.py bench caps.
  - This watcher observes status.jsonl. As soon as all Phase 1 rollouts for a
    task are terminal, it generates teacher reflection records for that task
    and places them into the Phase 2 queue.

The watcher is deliberately file-based so it composes with the existing bash
wrapper and remains resumable/debuggable from artifacts on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import DEFAULT_CONFIG, display_path, experiment_status_path, load_json, read_jsonl, repo_path, safe_slug
from make_teacher_fallback_plan import build_teacher_records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def latest_status_by_trial(run_id: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(experiment_status_path(run_id))
    status: dict[str, dict[str, Any]] = {}
    for row in rows:
        trial_id = str(row.get("trial_id", ""))
        if trial_id:
            status[trial_id] = row
    return status


def task_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record["bench"]), str(record["task_id"])


def task_base(index: int, bench: str, task_id: str) -> str:
    return f"task_{index:04d}_{safe_slug(bench, 20)}_{safe_slug(task_id, 50)}.teacher.jsonl"


def path_exists_any(base: str, directories: list[Path]) -> bool:
    return any((directory / base).exists() for directory in directories)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--queue-dir", required=True)
    parser.add_argument("--reflection-dir", required=True)
    parser.add_argument("--running-dir", required=True)
    parser.add_argument("--done-dir", required=True)
    parser.add_argument("--failed-dir", required=True)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-trials", type=int, required=True)
    parser.add_argument("--poll-sec", type=float, default=20.0)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--max-assistant-chars", type=int, default=900)
    parser.add_argument("--max-trace-chars", type=int, default=2200)
    parser.add_argument("--max-feedback-chars", type=int, default=1800)
    parser.add_argument("--max-context-chars", type=int, default=24000)
    args = parser.parse_args()

    plan_path = repo_path(args.plan)
    config = load_json(args.config)
    records = read_jsonl(plan_path)
    phase1 = [
        record for record in records
        if record.get("model_role") == "student"
        and str(record.get("mode", "")).startswith("student_")
        and "_reflection" not in str(record.get("mode", ""))
    ]
    if not phase1:
        raise SystemExit("no phase1 records to watch")
    run_ids = {str(record["run_id"]) for record in phase1}
    if len(run_ids) != 1:
        raise SystemExit(f"expected one run_id, got {sorted(run_ids)}")
    run_id = next(iter(run_ids))

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for record in phase1:
        key = task_key(record)
        if key not in grouped:
            order.append(key)
        grouped[key].append(record)

    queue_dir = repo_path(args.queue_dir)
    reflection_dir = repo_path(args.reflection_dir)
    directories = [
        queue_dir,
        repo_path(args.running_dir),
        repo_path(args.done_dir),
        repo_path(args.failed_dir),
        reflection_dir,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    stop_file = repo_path(args.stop_file)

    processed: set[tuple[str, str]] = set()
    waiting_incomplete: set[tuple[str, str]] = set()
    for index, (bench, task_id) in enumerate(order):
        base = task_base(index, bench, task_id)
        if path_exists_any(base, directories) or (reflection_dir / base.replace(".jsonl", ".skipped.txt")).exists():
            processed.add((bench, task_id))

    print(
        f"[watcher] run_id={run_id} tasks={len(order)} already_processed={len(processed)} "
        f"poll={args.poll_sec}s",
        flush=True,
    )

    while True:
        status = latest_status_by_trial(run_id)
        made_progress = False
        for index, (bench, task_id) in enumerate(order):
            key = (bench, task_id)
            if key in processed:
                continue
            task_records = grouped[key]
            if any(record["trial_id"] not in status for record in task_records):
                continue

            base = task_base(index, bench, task_id)
            teacher_path = reflection_dir / base
            skipped_path = reflection_dir / base.replace(".jsonl", ".skipped.txt")
            teacher_records, skip_reasons, counters = build_teacher_records(
                task_records,
                config=config,
                teacher_model=args.teacher_model,
                teacher_trials=args.teacher_trials,
                require_complete_phase1=True,
                max_attempts=args.max_attempts,
                max_assistant_chars=args.max_assistant_chars,
                max_trace_chars=args.max_trace_chars,
                max_feedback_chars=args.max_feedback_chars,
                max_context_chars=args.max_context_chars,
                limit_tasks=0,
            )
            if counters.get("phase1_incomplete"):
                if key not in waiting_incomplete:
                    print(
                        f"[watcher] wait {bench}/{task_id} counters={dict(counters)}",
                        flush=True,
                    )
                    waiting_incomplete.add(key)
                continue
            skipped_path.write_text(
                "\n".join(f"{name}\t{reason}" for name, reason in sorted(skip_reasons.items()))
                + ("\n" if skip_reasons else ""),
                encoding="utf-8",
            )
            if teacher_records:
                write_jsonl(teacher_path, teacher_records)
                tmp_queue = queue_dir / f"{base}.tmp"
                final_queue = queue_dir / base
                write_jsonl(tmp_queue, teacher_records)
                os.replace(tmp_queue, final_queue)
                print(
                    f"[watcher] enqueue {bench}/{task_id} records={len(teacher_records)} "
                    f"path={display_path(final_queue)}",
                    flush=True,
                )
            else:
                print(
                    f"[watcher] skip {bench}/{task_id} counters={dict(counters)}",
                    flush=True,
                )
            processed.add(key)
            made_progress = True

        if len(processed) >= len(order):
            print(f"[watcher] all tasks processed: {len(processed)}/{len(order)}", flush=True)
            return
        if stop_file.exists() and not made_progress:
            incomplete = len(order) - len(processed)
            print(
                f"[watcher] phase1 stopped with incomplete tasks={incomplete}; "
                "no more status rows are expected",
                flush=True,
            )
            raise SystemExit(2)
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
