#!/usr/bin/env python3
"""Split a trial plan into task-complete chunks for pipelined collection.

Each chunk keeps all records for a `(bench, task_id)` together. This matters
because Phase 2 teacher fallback is generated at task granularity: it should
only inspect a task after all Phase 1 branches for that task are terminal.

The output order round-robins benches so early chunks contain mixed benches
instead of exhausting all Claw/TB2 tasks first. That lets teacher fallback begin
earlier across the full distribution while Phase 1 continues on the student
endpoint.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from common import PROJECT_ROOT, repo_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file_handle:
        for line in file_handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        for row in rows:
            file_handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def group_records_by_task(
    records: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], list[str], dict[str, list[tuple[str, str]]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    bench_order: list[str] = []
    task_order_by_bench: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seen_tasks: set[tuple[str, str]] = set()

    for record in records:
        bench = str(record["bench"])
        task_id = str(record["task_id"])
        key = (bench, task_id)
        if bench not in bench_order:
            bench_order.append(bench)
        if key not in seen_tasks:
            seen_tasks.add(key)
            task_order_by_bench[bench].append(key)
        groups[key].append(record)
    return groups, bench_order, task_order_by_bench


def round_robin_task_keys(
    bench_order: list[str],
    task_order_by_bench: dict[str, list[tuple[str, str]]],
) -> list[tuple[str, str]]:
    queues = {
        bench: deque(task_order_by_bench.get(bench, []))
        for bench in bench_order
    }
    ordered: list[tuple[str, str]] = []
    while any(queues[bench] for bench in bench_order):
        for bench in bench_order:
            if queues[bench]:
                ordered.append(queues[bench].popleft())
    return ordered


def split_chunks(
    records: list[dict[str, Any]],
    *,
    tasks_per_chunk: int,
) -> list[list[dict[str, Any]]]:
    if tasks_per_chunk <= 0:
        raise ValueError("tasks_per_chunk must be > 0")
    groups, bench_order, task_order_by_bench = group_records_by_task(records)
    task_keys = round_robin_task_keys(bench_order, task_order_by_bench)

    chunks: list[list[dict[str, Any]]] = []
    for start in range(0, len(task_keys), tasks_per_chunk):
        chunk_keys = task_keys[start:start + tasks_per_chunk]
        chunk_rows: list[dict[str, Any]] = []
        for key in chunk_keys:
            chunk_rows.extend(groups[key])
        chunks.append(chunk_rows)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tasks-per-chunk", type=int, default=10)
    parser.add_argument("--prefix", default="chunk")
    args = parser.parse_args()

    plan_path = repo_path(args.plan)
    out_dir = repo_path(args.out_dir)
    records = read_jsonl(plan_path)
    chunks = split_chunks(records, tasks_per_chunk=args.tasks_per_chunk)

    manifest: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        path = out_dir / f"{args.prefix}_{index:04d}.jsonl"
        write_jsonl(path, chunk)
        task_keys = sorted({f"{row['bench']}/{row['task_id']}" for row in chunk})
        manifest.append({
            "index": index,
            "path": display_path(path),
            "records": len(chunk),
            "tasks": len(task_keys),
            "first_task": task_keys[0] if task_keys else "",
            "last_task": task_keys[-1] if task_keys else "",
        })

    manifest_path = out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "source_plan": display_path(plan_path),
        "tasks_per_chunk": args.tasks_per_chunk,
        "chunks": manifest,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"source records: {len(records)}")
    print(f"chunks: {len(chunks)}")
    print(f"manifest: {display_path(manifest_path)}")
    for row in manifest[:5]:
        print(f"  - {row['path']} records={row['records']} tasks={row['tasks']}")
    if len(manifest) > 5:
        print(f"  ... ({len(manifest) - 5} more)")


if __name__ == "__main__":
    main()
