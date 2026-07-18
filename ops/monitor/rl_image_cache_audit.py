#!/usr/bin/env python3
"""Audit expected RL task Docker images against the current Docker daemon.

The report answers: for the current RL parquet, which task images are already
available and therefore should not pull/build during training, and which ones
are still missing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))
DEFAULT_PARQUET = PROJECT_ROOT / "datasets/rl/parquet_4bench_base_20260523/train.parquet"

HARBOR_BENCH_INFO = {
    "sb_ns": (
        PROJECT_ROOT / "datasets/skillsbench/tasks",
        "skillsbench-no-skills",
    ),
    "tb2": (
        PROJECT_ROOT / "datasets/terminal-bench-v2",
        "tb2",
    ),
    "seta_synth": (
        PROJECT_ROOT / "datasets/seta/dataset/seta_synth_top300",
        "seta-synth",
    ),
    "seta": (
        PROJECT_ROOT / "datasets/seta/dataset/seta_baseline_30",
        "seta",
    ),
}


def docker_tags() -> set[str]:
    env = dict(os.environ)
    docker_host = os.environ.get("DOCKER_HOST") or "unix:///tmp/local-docker-overlay2.sock"
    if docker_host in ("tcp://127.0.0.1:2375", "tcp://127.0.0.1:2376", "unix:///tmp/apex-docker.sock"):
        docker_host = "unix:///tmp/local-docker-overlay2.sock"
    env["DOCKER_HOST"] = docker_host
    proc = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def load_task_rows(parquet_path: Path) -> list[dict[str, Any]]:
    df = pd.read_parquet(parquet_path, columns=["extra_info"])
    rows: list[dict[str, Any]] = []
    for extra_info in df["extra_info"]:
        if hasattr(extra_info, "item"):
            extra_info = extra_info.item()
        rows.append(dict(extra_info))
    return rows


def read_task_toml(task_dir: Path) -> dict[str, Any]:
    path = task_dir / "task.toml"
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def expected_image(row: dict[str, Any], existing_tags: set[str]) -> dict[str, str]:
    bench = str(row.get("bench") or row.get("task_kwargs", {}).get("bench") or "")
    task_id = str(row.get("task_id") or row.get("task_kwargs", {}).get("task_id") or "")

    if bench == "claw":
        tag = "claw-sandbox:latest" if "claw-sandbox:latest" in existing_tags else "python:3.12-slim"
        return {"bench": bench, "task_id": task_id, "kind": "claw_base", "tag": tag}

    if bench == "swe_lite":
        name = task_id.replace("__", "_s_").lower()
        tag = f"xingyaoww/sweb.eval.x86_64.{name}:latest"
        return {"bench": bench, "task_id": task_id, "kind": "swe_official", "tag": tag}

    if bench in HARBOR_BENCH_INFO:
        dataset_dir, dataset_tag = HARBOR_BENCH_INFO[bench]
        task_dir = dataset_dir / task_id
        meta = read_task_toml(task_dir)
        declared = meta.get("environment", {}).get("docker_image") if meta else None
        if declared:
            return {"bench": bench, "task_id": task_id, "kind": "declared_pull", "tag": str(declared)}
        dockerfile = task_dir / "environment/Dockerfile"
        tag = f"unified-{dataset_tag}-{task_id}:latest"
        kind = "local_build" if dockerfile.exists() else "missing_definition"
        return {"bench": bench, "task_id": task_id, "kind": kind, "tag": tag}

    return {"bench": bench, "task_id": task_id, "kind": "unknown_bench", "tag": ""}


def render_report(records: list[dict[str, str]], existing_tags: set[str], parquet_path: Path) -> str:
    unique_by_tag: dict[str, dict[str, str]] = {}
    row_counts: Counter[str] = Counter()
    for record in records:
        tag = record["tag"]
        key = tag or f"{record['kind']}:{record['bench']}:{record['task_id']}"
        unique_by_tag.setdefault(key, record)
        row_counts[key] += 1

    summary = Counter()
    row_summary = Counter()
    by_bench = defaultdict(Counter)
    by_bench_rows = defaultdict(Counter)
    missing_rows = []
    for record in records:
        exists = bool(record["tag"] and record["tag"] in existing_tags)
        status = "present" if exists else "missing"
        row_summary[(record["kind"], status)] += 1
        by_bench_rows[record["bench"]][(record["kind"], status)] += 1

    for key, record in unique_by_tag.items():
        exists = bool(record["tag"] and record["tag"] in existing_tags)
        status = "present" if exists else "missing"
        kind = record["kind"]
        bench = record["bench"]
        summary[(kind, status)] += 1
        by_bench[bench][(kind, status)] += 1
        if not exists:
            missing_rows.append((row_counts[key], record))

    total_unique = len(unique_by_tag)
    total_present = sum(count for (kind, status), count in summary.items() if status == "present")
    total_missing = total_unique - total_present
    out = [
        "# RL Image Cache Audit",
        "",
        f"- Parquet: `{parquet_path}`",
        f"- Training rows: {len(records)}",
        f"- Unique expected image entries: {total_unique}",
        f"- Present now: {total_present}",
        f"- Missing now: {total_missing}",
        f"- Training rows covered by present images: {sum(count for (_kind, status), count in row_summary.items() if status == 'present')}",
        f"- Training rows that would still pull/build if sampled now: {sum(count for (_kind, status), count in row_summary.items() if status == 'missing')}",
        "",
        "## Summary By Kind (Unique Tags)",
        "",
        "| kind | present | missing |",
        "|---|---:|---:|",
    ]
    for kind in sorted({kind for kind, _status in summary}):
        out.append(f"| `{kind}` | {summary[(kind, 'present')]} | {summary[(kind, 'missing')]} |")

    out.extend(["", "## Summary By Kind (Training Rows)", "", "| kind | present rows | missing rows |", "|---|---:|---:|"])
    for kind in sorted({kind for kind, _status in row_summary}):
        out.append(f"| `{kind}` | {row_summary[(kind, 'present')]} | {row_summary[(kind, 'missing')]} |")

    out.extend(["", "## Summary By Bench (Unique Tags)", "", "| bench | kind/status counts |", "|---|---|"])
    for bench in sorted(by_bench):
        parts = [
            f"`{kind}/{status}`={count}"
            for (kind, status), count in sorted(by_bench[bench].items())
        ]
        out.append(f"| `{bench}` | {'; '.join(parts)} |")

    out.extend(["", "## Summary By Bench (Training Rows)", "", "| bench | kind/status row counts |", "|---|---|"])
    for bench in sorted(by_bench_rows):
        parts = [
            f"`{kind}/{status}`={count}"
            for (kind, status), count in sorted(by_bench_rows[bench].items())
        ]
        out.append(f"| `{bench}` | {'; '.join(parts)} |")

    out.extend(
        [
            "",
            "## Missing Image Entries",
            "",
            "| row count | bench | task | kind | expected tag |",
            "|---:|---|---|---|---|",
        ]
    )
    if missing_rows:
        for row_count, record in sorted(
            missing_rows,
            key=lambda item: (item[0], item[1]["bench"], item[1]["task_id"], item[1]["tag"]),
            reverse=True,
        ):
            out.append(
                f"| {row_count} | `{record['bench']}` | `{record['task_id']}` | "
                f"`{record['kind']}` | `{record['tag']}` |"
            )
    else:
        out.append("| 0 | - | - | - | - |")
    out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs/rl_image_cache_audit.md")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    tags = docker_tags()
    rows = load_task_rows(args.parquet)
    records = [expected_image(row, tags) for row in rows]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(records, tags, args.parquet), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
