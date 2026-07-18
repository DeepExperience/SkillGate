#!/usr/bin/env python3
"""Restore local Docker images from the persistent RL image tar cache.

This is the offline counterpart to ``migrate_apex_images_to_local.py``. It never
talks to the remote Docker daemon. It only:

1. rebuilds the expected image manifest from RL parquet files,
2. checks whether each tag exists in the local Docker daemon,
3. loads the matching persistent tarball when the local tag is missing.

Use this after container recreation because the fast local Docker root lives on
``/data/cache`` and is not persistent.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))
sys.path.insert(0, str(PROJECT_ROOT / "ops/launch"))

from migrate_apex_images_to_local import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    discover_parquets,
    docker_env,
    expected_tags,
    image_exists,
    maybe_tag_claw_sandbox,
    safe_tar_name,
    write_manifest,
)


DEFAULT_RUN_ROOT = PROJECT_ROOT / "experiments/infra/rl/local_docker_migration/restore_current"
AUXILIARY_IMAGE_RECORDS = [
    {
        "bench": "sb_ns",
        "task_id": "fix-visual-stability",
        "kind": "compose_sidecar",
        "tag": "skillsbench-visual-stability-api:latest",
    }
]


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def restore_one(
    record: dict[str, str],
    *,
    local_host: str,
    cache_dir: Path,
    status_path: Path,
    load_timeout: int,
    force_reload: bool,
) -> dict[str, Any]:
    tag = record["tag"]
    started = time.time()
    tar_path = cache_dir / safe_tar_name(tag)
    result: dict[str, Any] = {
        **record,
        "tar": str(tar_path.relative_to(PROJECT_ROOT)) if tar_path.is_relative_to(PROJECT_ROOT) else str(tar_path),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        if image_exists(tag, docker_host=local_host) and not force_reload:
            result.update(status="ok", note="already_loaded", elapsed_sec=round(time.time() - started, 3))
            append_jsonl(status_path, result)
            return result
        if not tar_path.exists():
            result.update(status="missing_tar", elapsed_sec=round(time.time() - started, 3))
            append_jsonl(status_path, result)
            return result
        with tar_path.open("rb") as handle:
            proc = subprocess.run(
                ["docker", "load"],
                cwd=str(PROJECT_ROOT),
                env=docker_env(local_host),
                stdin=handle,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=load_timeout,
                check=False,
            )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, bytes) else str(proc.stderr)
            result.update(status="load_failed", error=err[-800:], elapsed_sec=round(time.time() - started, 3))
        elif image_exists(tag, docker_host=local_host, timeout=60):
            result.update(status="ok", note="loaded_from_tar", elapsed_sec=round(time.time() - started, 3))
        else:
            out = proc.stdout.decode("utf-8", errors="replace") if isinstance(proc.stdout, bytes) else str(proc.stdout)
            result.update(status="load_no_tag", output=out[-800:], elapsed_sec=round(time.time() - started, 3))
    except subprocess.TimeoutExpired as exc:
        result.update(status="timeout", error=str(exc), elapsed_sec=round(time.time() - started, 3))
    except Exception as exc:  # noqa: BLE001 - operational restore should record and continue.
        result.update(status="error", error=repr(exc), elapsed_sec=round(time.time() - started, 3))
    append_jsonl(status_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", action="append", default=[], help="Parquet file or directory. Defaults to all datasets/rl/parquet*/train|eval.parquet")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--local-docker-host", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bench", action="append", default=[])
    parser.add_argument("--kind", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--load-timeout-sec", type=int, default=10800)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    local_host = args.local_docker_host
    if not local_host:
        active_sock = Path("/tmp/local-docker-active.sock")
        local_host = "unix://" + active_sock.read_text(encoding="utf-8").strip() if active_sock.exists() else "unix:///tmp/local-docker-overlay2.sock"

    parquets = discover_parquets(args.parquet)
    records = expected_tags(parquets)
    known_tags = {record["tag"] for record in records}
    records.extend(record for record in AUXILIARY_IMAGE_RECORDS if record["tag"] not in known_tags)
    if args.bench:
        benches = set(args.bench)
        records = [r for r in records if r["bench"] in benches]
    if args.kind:
        kinds = set(args.kind)
        records = [r for r in records if r["kind"] in kinds]
    if args.limit:
        records = records[: args.limit]

    args.run_root.mkdir(parents=True, exist_ok=True)
    write_manifest(args.run_root, records, parquets)
    status_path = args.run_root / "status.jsonl"

    print(f"run_root={args.run_root}")
    print(f"cache_dir={args.cache_dir}")
    print(f"local={local_host}")
    print(f"unique_tags={len(records)} workers={args.workers} dry_run={args.dry_run}")
    print(f"manifest={args.run_root / 'image_manifest.jsonl'}")
    if args.dry_run:
        return

    completed: list[dict[str, Any]] = []
    if args.workers <= 1:
        for index, record in enumerate(records, start=1):
            print(f"[{index}/{len(records)}] {record['tag']}", flush=True)
            completed.append(
                restore_one(
                    record,
                    local_host=local_host,
                    cache_dir=args.cache_dir,
                    status_path=status_path,
                    load_timeout=args.load_timeout_sec,
                    force_reload=args.force_reload,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_map = {
                pool.submit(
                    restore_one,
                    record,
                    local_host=local_host,
                    cache_dir=args.cache_dir,
                    status_path=status_path,
                    load_timeout=args.load_timeout_sec,
                    force_reload=args.force_reload,
                ): record
                for record in records
            }
            for index, future in enumerate(as_completed(future_map), start=1):
                result = future.result()
                completed.append(result)
                print(f"[{index}/{len(records)}] {result.get('status')} {result.get('tag')} {result.get('elapsed_sec')}s", flush=True)

    maybe_tag_claw_sandbox(local_host)
    summary = Counter(str(item.get("status")) for item in completed)
    (args.run_root / "status_summary.json").write_text(
        json.dumps(
            {
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status_counts": dict(summary),
                "records": len(completed),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("status_counts=" + json.dumps(dict(summary), sort_keys=True))


if __name__ == "__main__":
    main()
