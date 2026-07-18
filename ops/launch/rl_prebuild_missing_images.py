#!/usr/bin/env python3
"""Prebuild/pull missing Docker images for the RL agent-bench task pool.

This script is intentionally operational:

* It audits the current RL train/eval parquet against the configured Docker daemon.
* It runs missing SWE official image pulls first, then SETA local builds, then
  SkillsBench local builds.
* It records per-image status so the tmux job can be stopped/restarted without
  redoing successful images.

The actual Harbor local-build path reuses
``GeneralAgent/eval_scripts/unified_runner/run_unified_harbor.py`` so the
prebuild command matches the training launcher.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))
DEFAULT_PARQUETS = [
    PROJECT_ROOT / "datasets/rl/parquet_4bench_base_20260523/train.parquet",
    PROJECT_ROOT / "datasets/rl/parquet_4bench_base_20260523/eval.parquet",
]
DEFAULT_OUT_DIR = PROJECT_ROOT / "experiments/infra/rl/prebuild/current_missing_images"
DEFAULT_WATCHLIST_MD = PROJECT_ROOT / "docs/rl_docker_prebuild_run_status.md"

BENCH_DATASET_TAG = {
    "seta_synth": "seta-synth",
    "sb_ns": "skillsbench-no-skills",
}

BENCH_PRIORITY = {
    "swe_lite": 0,
    "seta_synth": 1,
    "sb_ns": 2,
}

KIND_PRIORITY = {
    "swe_official": 0,
    "local_build": 1,
}

STATUS_LOCK = threading.Lock()


def _setup_imports() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "ops/monitor"))
    sys.path.insert(0, str(PROJECT_ROOT / "GeneralAgent/eval_scripts"))


def _docker_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("DOCKER_HOST", "unix:///tmp/local-docker-overlay2.sock")
    env.setdefault("HTTP_PROXY", "http://your-proxy:3128")
    env.setdefault("HTTPS_PROXY", "http://your-proxy:3128")
    env.setdefault("http_proxy", env["HTTP_PROXY"])
    env.setdefault("https_proxy", env["HTTPS_PROXY"])
    env.setdefault(
        "NO_PROXY",
        "127.0.0.1,localhost,0.0.0.0,10.0.0.0/8,172.16.0.0/12,"
        "mirrors.tuna.tsinghua.edu.cn,pypi.tuna.tsinghua.edu.cn,hf-mirror.com",
    )
    env.setdefault("no_proxy", env["NO_PROXY"])
    return env


def _docker_image_exists(tag: str) -> bool:
    proc = subprocess.run(
        ["docker", "images", "-q", tag],
        env=_docker_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _load_missing_records(parquets: list[Path]) -> list[dict[str, str]]:
    _setup_imports()
    import rl_image_cache_audit as audit  # type: ignore

    tags = audit.docker_tags()
    by_tag: dict[str, dict[str, str]] = {}
    for parquet in parquets:
        rows = audit.load_task_rows(parquet)
        for row in rows:
            record = audit.expected_image(row, tags)
            tag = record.get("tag") or ""
            if not tag or tag in tags:
                continue
            by_tag.setdefault(tag, record)

    records = list(by_tag.values())
    return sorted(
        records,
        key=lambda r: (
            BENCH_PRIORITY.get(r["bench"], 99),
            KIND_PRIORITY.get(r["kind"], 99),
            r["task_id"],
        ),
    )


def _read_skip_tags(status_path: Path, *, retry_failed: bool) -> set[str]:
    skip: set[str] = set()
    if not status_path.exists():
        return skip
    for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = item.get("status")
        if item.get("tag") and (status == "ok" or (status == "failed" and not retry_failed)):
            skip.add(str(item["tag"]))
    return skip


def _append_status(status_path: Path, record: dict[str, Any]) -> None:
    with STATUS_LOCK:
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _run_one(record: dict[str, str], logs_dir: Path, build_timeout: int) -> dict[str, Any]:
    tag = record["tag"]
    task_id = record["task_id"]
    bench = record["bench"]
    kind = record["kind"]
    safe_tag = tag.replace("/", "_").replace(":", "_")
    log_path = logs_dir / f"{BENCH_PRIORITY.get(bench, 9)}_{bench}_{task_id}_{safe_tag}.log"
    started = time.time()

    if _docker_image_exists(tag):
        return {
            "status": "ok",
            "tag": tag,
            "bench": bench,
            "task_id": task_id,
            "kind": kind,
            "elapsed_sec": 0.0,
            "note": "already_present",
        }

    if kind == "swe_official":
        cmd = ["docker", "pull", tag]
    elif kind == "local_build" and bench in BENCH_DATASET_TAG:
        dataset_tag = BENCH_DATASET_TAG[bench]
        script = (
            "from pathlib import Path\n"
            "import sys\n"
            f"sys.path.insert(0, {str(PROJECT_ROOT / 'GeneralAgent/eval_scripts')!r})\n"
            "from unified_runner import run_unified_harbor as ruh\n"
            f"ruh.resolve_image(Path({str(Path(record.get('task_dir', '')))!r}), {task_id!r}, {dataset_tag!r})\n"
        )
        cmd = [sys.executable, "-c", script]
    else:
        return {
            "status": "skip",
            "tag": tag,
            "bench": bench,
            "task_id": task_id,
            "kind": kind,
            "elapsed_sec": 0.0,
            "note": "unsupported_record",
        }

    env = _docker_env()
    env["UNIFIED_HARBOR_BUILD_TIMEOUT_SEC"] = str(build_timeout)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"cmd={cmd!r}\n")
        log.write(f"tag={tag}\nbench={bench}\ntask_id={task_id}\nkind={kind}\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=str(PROJECT_ROOT),
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=max(build_timeout + 300, 900) if kind == "local_build" else 3600,
            check=False,
        )

    elapsed = round(time.time() - started, 3)
    ok = proc.returncode == 0 and _docker_image_exists(tag)
    return {
        "status": "ok" if ok else "failed",
        "tag": tag,
        "bench": bench,
        "task_id": task_id,
        "kind": kind,
        "elapsed_sec": elapsed,
        "returncode": proc.returncode,
        "log": str(log_path),
    }


def _write_plan(records: list[dict[str, str]], out_dir: Path) -> None:
    plan_path = out_dir / "plan.jsonl"
    with plan_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    counts: dict[tuple[str, str], int] = {}
    for record in records:
        key = (record["bench"], record["kind"])
        counts[key] = counts.get(key, 0) + 1
    lines = [
        "# RL Docker Prebuild Plan",
        "",
        f"- created_at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- total_missing_unique: {len(records)}",
        "",
        "| bench | kind | count |",
        "|---|---|---:|",
    ]
    for (bench, kind), count in sorted(counts.items(), key=lambda x: (BENCH_PRIORITY.get(x[0][0], 99), x[0][1])):
        lines.append(f"| `{bench}` | `{kind}` | {count} |")
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _read_status_rows(status_path: Path) -> list[dict[str, Any]]:
    if not status_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _write_watchlist(
    records: list[dict[str, str]],
    status_path: Path,
    watchlist_md: Path,
    out_dir: Path,
) -> None:
    """Write a compact operational watchlist for future prebuild work."""
    status_rows = _read_status_rows(status_path)
    planned_counts: dict[tuple[str, str], int] = {}
    for record in records:
        key = (record["bench"], record["kind"])
        planned_counts[key] = planned_counts.get(key, 0) + 1

    status_counts: dict[tuple[str, str, str], int] = {}
    failed_rows = []
    slow_rows = []
    for row in status_rows:
        key = (str(row.get("bench")), str(row.get("kind")), str(row.get("status")))
        status_counts[key] = status_counts.get(key, 0) + 1
        if row.get("status") != "ok":
            failed_rows.append(row)
        elif float(row.get("elapsed_sec") or 0) >= 600:
            slow_rows.append(row)

    lines = [
        "# RL Docker Prebuild Watchlist",
        "",
        f"- updated_at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- current_run: `{out_dir}`",
        "- purpose: track images that should be pre-pulled/prebuilt before RL so bench setup does not become the rollout bottleneck.",
        "",
        "## Planned Missing Images",
        "",
        "| bench | kind | count |",
        "|---|---|---:|",
    ]
    for (bench, kind), count in sorted(
        planned_counts.items(),
        key=lambda x: (BENCH_PRIORITY.get(x[0][0], 99), KIND_PRIORITY.get(x[0][1], 99), x[0][0]),
    ):
        lines.append(f"| `{bench}` | `{kind}` | {count} |")

    lines += [
        "",
        "## Completed Status",
        "",
        "| bench | kind | status | count |",
        "|---|---|---|---:|",
    ]
    if status_counts:
        for (bench, kind, status), count in sorted(
            status_counts.items(),
            key=lambda x: (
                BENCH_PRIORITY.get(x[0][0], 99),
                KIND_PRIORITY.get(x[0][1], 99),
                x[0][2],
            ),
        ):
            lines.append(f"| `{bench}` | `{kind}` | `{status}` | {count} |")
    else:
        lines.append("| - | - | - | 0 |")

    lines += [
        "",
        "## Failed Or Needs Attention",
        "",
        "| bench | task_id | kind | status | elapsed_sec | log |",
        "|---|---|---|---|---:|---|",
    ]
    attention_rows = failed_rows + slow_rows
    if attention_rows:
        for row in attention_rows[-100:]:
            lines.append(
                f"| `{row.get('bench')}` | `{row.get('task_id')}` | `{row.get('kind')}` | "
                f"`{row.get('status')}` | {row.get('elapsed_sec', '')} | `{row.get('log', '')}` |"
            )
    else:
        lines.append("| - | - | - | - | - | - |")

    watchlist_md.parent.mkdir(parents=True, exist_ok=True)
    watchlist_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _phase(
    records: list[dict[str, str]],
    bench: str,
    workers: int,
    out_dir: Path,
    status_path: Path,
    build_timeout: int,
    watchlist_md: Path,
) -> None:
    phase_records = [r for r in records if r["bench"] == bench]
    if not phase_records:
        return
    print(f"[phase] bench={bench} records={len(phase_records)} workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_run_one, record, out_dir / "logs", build_timeout): record
            for record in phase_records
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "status": "failed",
                    "tag": record["tag"],
                    "bench": record["bench"],
                    "task_id": record["task_id"],
                    "kind": record["kind"],
                    "error": repr(exc),
                }
            result["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _append_status(status_path, result)
            _write_watchlist(records, status_path, watchlist_md, out_dir)
            print(
                f"[{result['status']}] {result['bench']}/{result['task_id']} "
                f"{result['tag']} elapsed={result.get('elapsed_sec', '?')}s",
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", action="append", type=Path, default=[])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--swe-workers", type=int, default=1)
    parser.add_argument("--seta-workers", type=int, default=4)
    parser.add_argument("--sb-workers", type=int, default=2)
    parser.add_argument("--build-timeout-sec", type=int, default=3600)
    parser.add_argument("--watchlist-md", type=Path, default=DEFAULT_WATCHLIST_MD)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry tags that already have failed records in this run's status.jsonl. Default is to skip them.",
    )
    parser.add_argument(
        "--only-bench",
        action="append",
        choices=sorted(BENCH_PRIORITY),
        default=[],
        help="Restrict this run to one or more benches. Useful when a slow SWE pull should not block SETA local prebuilds.",
    )
    parser.add_argument(
        "--only-task-ids",
        default="",
        help="Comma-separated task ids to include after bench filtering. Useful for adding extra workers without duplicating active builds.",
    )
    parser.add_argument(
        "--exclude-task-ids",
        default="",
        help="Comma-separated task ids to exclude after bench filtering. Useful for avoiding tags already in flight.",
    )
    args = parser.parse_args()

    parquets = args.parquet or DEFAULT_PARQUETS
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status.jsonl"

    records = _load_missing_records(parquets)
    for record in records:
        if record["kind"] == "local_build":
            if record["bench"] == "seta_synth":
                record["task_dir"] = str(PROJECT_ROOT / "datasets/seta/dataset/seta_synth_top300" / record["task_id"])
            elif record["bench"] == "sb_ns":
                record["task_dir"] = str(PROJECT_ROOT / "datasets/skillsbench/tasks" / record["task_id"])

    skip_tags = _read_skip_tags(status_path, retry_failed=args.retry_failed)
    records = [record for record in records if record["tag"] not in skip_tags]
    if args.only_bench:
        allowed = set(args.only_bench)
        records = [record for record in records if record["bench"] in allowed]
    if args.only_task_ids.strip():
        allowed_task_ids = {item.strip() for item in args.only_task_ids.split(",") if item.strip()}
        records = [record for record in records if str(record.get("task_id")) in allowed_task_ids]
    if args.exclude_task_ids.strip():
        excluded_task_ids = {item.strip() for item in args.exclude_task_ids.split(",") if item.strip()}
        records = [record for record in records if str(record.get("task_id")) not in excluded_task_ids]
    _write_plan(records, out_dir)
    _write_watchlist(records, status_path, args.watchlist_md, out_dir)
    print(f"[plan] out_dir={out_dir} remaining={len(records)}", flush=True)

    _phase(records, "swe_lite", args.swe_workers, out_dir, status_path, args.build_timeout_sec, args.watchlist_md)
    _phase(records, "seta_synth", args.seta_workers, out_dir, status_path, args.build_timeout_sec, args.watchlist_md)
    _phase(records, "sb_ns", args.sb_workers, out_dir, status_path, args.build_timeout_sec, args.watchlist_md)

    print("[done] prebuild phases complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
