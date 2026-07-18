#!/usr/bin/env python3
"""Migrate RL/eval Docker images from a remote Docker daemon to a local one.

This copies *final images* with `docker save`/`docker load`; it does not rsync
Docker internals. That matters because the remote daemon uses overlay2 and the local
daemon may use a different data-root.

The tar cache is persistent, so if `/data/cache` is lost after container
recreation, rerun this script and it will load from tarballs instead of reading
the remote daemon again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))
DEFAULT_CACHE_DIR = PROJECT_ROOT / "experiments/infra/rl/local_docker_migration/image_tars"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "experiments/infra/rl/local_docker_migration/current"


def setup_imports() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "ops/monitor"))


def run_cmd(
    cmd: list[str],
    *,
    env: dict[str, str],
    timeout: int | None = None,
    stdout: Any = subprocess.PIPE,
    stderr: Any = subprocess.PIPE,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        stdout=stdout,
        stderr=stderr,
        timeout=timeout,
        check=False,
    )


def docker_env(host: str) -> dict[str, str]:
    env = dict(os.environ)
    env["DOCKER_HOST"] = host
    return env


def docker_save_cmd(tag: str, remote_host: str) -> list[str]:
    """Return a streaming `docker save` command for the remote Docker daemon.

    `DOCKER_HOST=ssh://your-docker-host docker save <large-tag>` has been observed to
    hang before producing any bytes for some multi-GB task images, while direct
    `ssh your-docker-host docker save <tag>` streams normally. Prefer direct SSH for
    ssh:// hosts; keep the generic Docker CLI path for tcp/unix/custom hosts.
    """
    if remote_host.startswith("ssh://"):
        host = remote_host[len("ssh://") :]
        # Docker tags contain no shell metacharacters in our manifests, but use
        # a single remote shell command so ssh does not reinterpret arguments in
        # surprising ways if future tags include unusual characters.
        return ["ssh", host, "docker save " + shlex.quote(tag)]
    return ["docker", "save", tag]


def ssh_host(remote_host: str) -> str:
    if not remote_host.startswith("ssh://"):
        raise ValueError(f"remote file save requires ssh:// host, got {remote_host}")
    return remote_host[len("ssh://") :]


def image_exists(tag: str, *, docker_host: str, timeout: int = 30) -> bool:
    proc = run_cmd(
        ["docker", "image", "inspect", tag],
        env=docker_env(docker_host),
        timeout=timeout,
    )
    return proc.returncode == 0


def remote_image_size_map(remote_host: str) -> dict[str, int]:
    proc = run_cmd(
        ["docker", "images", "--format", "{{json .}}"],
        env=docker_env(remote_host),
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    sizes: dict[str, int] = {}
    # Human-readable size is only for display; use inspect for exact sizes on
    # selected tags later if needed. Keep this cheap.
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        repo = item.get("Repository")
        tag = item.get("Tag")
        if repo and tag and repo != "<none>" and tag != "<none>":
            sizes[f"{repo}:{tag}"] = -1
    return sizes


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    df = pd.read_parquet(path, columns=["extra_info"])
    rows: list[dict[str, Any]] = []
    for value in df["extra_info"]:
        if hasattr(value, "item"):
            value = value.item()
        rows.append(dict(value))
    return rows


def discover_parquets(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    if paths:
        for raw in paths:
            path = (PROJECT_ROOT / raw).resolve() if not raw.startswith("/") else Path(raw)
            if path.is_dir():
                out.extend(sorted(path.glob("*.parquet")))
            else:
                out.append(path)
    else:
        out.extend(sorted((PROJECT_ROOT / "datasets/rl").glob("parquet*/train.parquet")))
        out.extend(sorted((PROJECT_ROOT / "datasets/rl").glob("parquet*/eval.parquet")))
    deduped: list[Path] = []
    seen = set()
    for path in out:
        if path.name not in {"train.parquet", "eval.parquet"}:
            continue
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def expected_tags(parquets: list[Path]) -> list[dict[str, str]]:
    setup_imports()
    import rl_image_cache_audit as audit  # type: ignore

    # Make Claw prefer a real sandbox tag in the manifest. We also explicitly
    # include the mock image below.
    fake_existing = {"claw-sandbox:latest"}
    records_by_tag: dict[str, dict[str, str]] = {}
    for parquet in parquets:
        for row in load_parquet_rows(parquet):
            record = audit.expected_image(row, fake_existing)
            tag = record.get("tag") or ""
            if not tag:
                continue
            item = {
                "tag": tag,
                "bench": str(record.get("bench", "")),
                "task_id": str(record.get("task_id", "")),
                "kind": str(record.get("kind", "")),
                "source_parquet": str(parquet.relative_to(PROJECT_ROOT)),
            }
            records_by_tag.setdefault(tag, item)

    # Claw mock service is a separate image used by run_unified_claw.py.
    records_by_tag.setdefault(
        "claw-mock-services:latest",
        {
            "tag": "claw-mock-services:latest",
            "bench": "claw",
            "task_id": "__shared_mock__",
            "kind": "claw_mock",
            "source_parquet": "implicit",
        },
    )
    # Useful fallback / base image for Claw sandbox.
    records_by_tag.setdefault(
        "python:3.12-slim",
        {
            "tag": "python:3.12-slim",
            "bench": "claw",
            "task_id": "__fallback__",
            "kind": "claw_fallback",
            "source_parquet": "implicit",
        },
    )
    return sorted(records_by_tag.values(), key=lambda r: (r["bench"], r["kind"], r["tag"]))


def safe_tar_name(tag: str) -> str:
    digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:12]
    safe = tag.replace("/", "_").replace(":", "__")
    return f"{safe}__{digest}.tar"


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def migrate_one(
    record: dict[str, str],
    *,
    remote_host: str,
    local_host: str,
    cache_dir: Path,
    status_path: Path,
    save_timeout: int,
    load_timeout: int,
    force_reload: bool,
    remote_file_save_dir: str,
) -> dict[str, Any]:
    tag = record["tag"]
    started = time.time()
    tar_path = cache_dir / safe_tar_name(tag)
    tmp_tar = tar_path.with_suffix(tar_path.suffix + ".tmp")

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
            if not image_exists(tag, docker_host=remote_host, timeout=60):
                # Remote historically lacks claw-sandbox; caller can tag it
                # from claw-mock-services after load.
                result.update(status="missing_remote", elapsed_sec=round(time.time() - started, 3))
                append_jsonl(status_path, result)
                return result
            tmp_tar.unlink(missing_ok=True)
            if remote_file_save_dir:
                host = ssh_host(remote_host)
                remote_dir = remote_file_save_dir.rstrip("/")
                remote_tmp = f"{remote_dir}/{safe_tar_name(tag)}.tmp"
                remote_cmd = (
                    f"mkdir -p {shlex.quote(remote_dir)} && "
                    f"rm -f {shlex.quote(remote_tmp)} && "
                    f"timeout {int(save_timeout)}s docker save -o {shlex.quote(remote_tmp)} {shlex.quote(tag)}"
                )
                proc = subprocess.run(
                    ["ssh", host, remote_cmd],
                    cwd=str(PROJECT_ROOT),
                    env=dict(os.environ),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=save_timeout + 120,
                    check=False,
                )
                if proc.returncode != 0:
                    err = proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, bytes) else str(proc.stderr)
                    result.update(status="save_failed", mode="remote_file", error=err[-800:], elapsed_sec=round(time.time() - started, 3))
                    append_jsonl(status_path, result)
                    return result
                proc_rsync = subprocess.run(
                    ["rsync", "-a", f"{host}:{remote_tmp}", str(tmp_tar)],
                    cwd=str(PROJECT_ROOT),
                    env=dict(os.environ),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=save_timeout + 120,
                    check=False,
                )
                subprocess.run(
                    ["ssh", host, "rm -f " + shlex.quote(remote_tmp)],
                    cwd=str(PROJECT_ROOT),
                    env=dict(os.environ),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=120,
                    check=False,
                )
                if proc_rsync.returncode != 0:
                    tmp_tar.unlink(missing_ok=True)
                    err = proc_rsync.stderr.decode("utf-8", errors="replace") if isinstance(proc_rsync.stderr, bytes) else str(proc_rsync.stderr)
                    result.update(status="save_failed", mode="remote_file_rsync", error=err[-800:], elapsed_sec=round(time.time() - started, 3))
                    append_jsonl(status_path, result)
                    return result
            else:
                with tmp_tar.open("wb") as handle:
                    save_cmd = docker_save_cmd(tag, remote_host)
                    save_env = dict(os.environ) if remote_host.startswith("ssh://") else docker_env(remote_host)
                    proc = subprocess.run(
                        save_cmd,
                        cwd=str(PROJECT_ROOT),
                        env=save_env,
                        stdout=handle,
                        stderr=subprocess.PIPE,
                        timeout=save_timeout,
                        check=False,
                    )
                if proc.returncode != 0:
                    tmp_tar.unlink(missing_ok=True)
                    err = proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, bytes) else str(proc.stderr)
                    result.update(status="save_failed", error=err[-800:], elapsed_sec=round(time.time() - started, 3))
                    append_jsonl(status_path, result)
                    return result
            tmp_tar.rename(tar_path)

        with tar_path.open("rb") as handle:
            proc2 = subprocess.run(
                ["docker", "load"],
                cwd=str(PROJECT_ROOT),
                env=docker_env(local_host),
                stdin=handle,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=load_timeout,
                check=False,
            )
        if proc2.returncode != 0:
            err = proc2.stderr.decode("utf-8", errors="replace") if isinstance(proc2.stderr, bytes) else str(proc2.stderr)
            result.update(status="load_failed", error=err[-800:], elapsed_sec=round(time.time() - started, 3))
        elif image_exists(tag, docker_host=local_host, timeout=60):
            result.update(status="ok", note="saved_loaded", elapsed_sec=round(time.time() - started, 3))
        else:
            out = proc2.stdout.decode("utf-8", errors="replace") if isinstance(proc2.stdout, bytes) else str(proc2.stdout)
            result.update(status="load_no_tag", output=out[-800:], elapsed_sec=round(time.time() - started, 3))
    except subprocess.TimeoutExpired as exc:
        result.update(status="timeout", error=str(exc), elapsed_sec=round(time.time() - started, 3))
    except Exception as exc:  # noqa: BLE001 - operational script should record and continue.
        result.update(status="error", error=repr(exc), elapsed_sec=round(time.time() - started, 3))
    append_jsonl(status_path, result)
    return result


def write_manifest(run_root: Path, records: list[dict[str, str]], parquets: list[Path]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "image_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    counts = Counter((r["bench"], r["kind"]) for r in records)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parquets": [str(p.relative_to(PROJECT_ROOT)) for p in parquets],
        "unique_tags": len(records),
        "counts": {f"{bench}/{kind}": count for (bench, kind), count in sorted(counts.items())},
    }
    (run_root / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def maybe_tag_claw_sandbox(local_host: str) -> None:
    if image_exists("claw-sandbox:latest", docker_host=local_host, timeout=20):
        return
    if image_exists("claw-mock-services:latest", docker_host=local_host, timeout=20):
        proc = run_cmd(
            ["docker", "tag", "claw-mock-services:latest", "claw-sandbox:latest"],
            env=docker_env(local_host),
            timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", action="append", default=[], help="Parquet file or directory. Defaults to all datasets/rl/parquet*/train|eval.parquet")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--remote-docker-host", default="ssh://your-docker-host")
    parser.add_argument("--local-docker-host", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bench", action="append", default=[])
    parser.add_argument("--kind", action="append", default=[])
    parser.add_argument("--exclude-tag-substring", action="append", default=[], help="Skip records whose Docker tag contains this substring. May repeat.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--save-timeout-sec", type=int, default=7200)
    parser.add_argument("--load-timeout-sec", type=int, default=7200)
    parser.add_argument("--remote-file-save-dir", default="", help="For ssh:// remotes, save image tar on the remote host first, then rsync it locally. Avoids stdout-pipe docker save stalls.")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    local_host = args.local_docker_host
    if not local_host:
        active_sock = Path("/tmp/local-docker-active.sock")
        if active_sock.exists():
            local_host = "unix://" + active_sock.read_text(encoding="utf-8").strip()
        else:
            local_host = "unix:///tmp/local-docker-overlay2.sock"

    parquets = discover_parquets(args.parquet)
    records = expected_tags(parquets)
    if args.bench:
        benches = set(args.bench)
        records = [r for r in records if r["bench"] in benches]
    if args.kind:
        kinds = set(args.kind)
        records = [r for r in records if r["kind"] in kinds]
    if args.exclude_tag_substring:
        needles = tuple(args.exclude_tag_substring)
        records = [r for r in records if not any(needle in r["tag"] for needle in needles)]
    if args.limit:
        records = records[: args.limit]

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.run_root.mkdir(parents=True, exist_ok=True)
    write_manifest(args.run_root, records, parquets)

    status_path = args.run_root / "status.jsonl"
    print(f"run_root={args.run_root}")
    print(f"cache_dir={args.cache_dir}")
    print(f"remote={args.remote_docker_host}")
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
                migrate_one(
                    record,
                    remote_host=args.remote_docker_host,
                    local_host=local_host,
                    cache_dir=args.cache_dir,
                    status_path=status_path,
                    save_timeout=args.save_timeout_sec,
                    load_timeout=args.load_timeout_sec,
                    force_reload=args.force_reload,
                    remote_file_save_dir=args.remote_file_save_dir,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_map = {
                pool.submit(
                    migrate_one,
                    record,
                    remote_host=args.remote_docker_host,
                    local_host=local_host,
                    cache_dir=args.cache_dir,
                    status_path=status_path,
                    save_timeout=args.save_timeout_sec,
                    load_timeout=args.load_timeout_sec,
                    force_reload=args.force_reload,
                    remote_file_save_dir=args.remote_file_save_dir,
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
