#!/usr/bin/env python3
"""Patch SETA synthetic Docker images so verifier mini-runner has python3.

The SETA verifier path now bypasses pytest for generated plain-assert tests,
but many generated Ubuntu images still lack python3.  Without this patch the
verifier installs python3 at grading time, which is acceptable for isolated eval
but too slow/noisy under Relax RL concurrency.

This script mutates already-prebuilt local images in-place:

  unified-seta-synth-<task_id>:latest

It does not build missing images.  Missing images are logged for separate
prebuild/debug so RL can keep `UNIFIED_HARBOR_REQUIRE_PREBUILT_LOCAL=1`.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INVENTORY = PROJECT_ROOT / "experiments/infra/rl/preflight/seta_verifier_static_inventory.json"
DEFAULT_OUT = PROJECT_ROOT / "experiments/infra/rl/preflight/seta_python3_patch_latest.jsonl"
BAD_SETA_IDS = {"25", "244", "436", "729"}


def run(cmd: list[str], *, docker_host: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["DOCKER_HOST"] = docker_host
    return subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=timeout)


def docker(args: list[str], *, docker_host: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return run(["docker", "-H", docker_host, *args], docker_host=docker_host, timeout=timeout)


def tail(text: str | None, limit: int = 2000) -> str:
    if not text:
        return ""
    return str(text)[-limit:]


def load_candidates(args: argparse.Namespace) -> list[str]:
    rows = json.loads(Path(args.inventory).read_text())
    candidates: list[str] = []
    only = {str(x) for item in args.only for x in item.split(",") if x}
    for row in rows:
        task_id = str(row["task_id"])
        if only and task_id not in only:
            continue
        if not args.include_bad and task_id in BAD_SETA_IDS:
            continue
        if not row.get("mini_runner_safe"):
            continue
        if args.needs_python_only and row.get("dockerfile_mentions_python"):
            continue
        candidates.append(task_id)
    candidates = sorted(set(candidates), key=lambda x: int(x) if x.isdigit() else x)
    if args.limit:
        candidates = candidates[: args.limit]
    return candidates


def patch_one(task_id: str, args: argparse.Namespace) -> dict[str, Any]:
    image = f"{args.image_prefix}-{task_id}:latest"
    cname = f"seta-py3-{task_id}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    result: dict[str, Any] = {
        "task_id": task_id,
        "image": image,
        "container": cname,
        "status": "unknown",
        "started_at": time.strftime("%FT%T%z"),
    }
    t0 = time.time()
    try:
        r = docker(["images", "-q", image], docker_host=args.docker_host, timeout=20)
        if r.returncode != 0:
            result.update(status="docker_error", error=tail(r.stderr or r.stdout))
            return result
        if not r.stdout.strip():
            result.update(status="missing_image")
            return result

        if args.dry_run:
            result.update(status="dry_run")
            return result

        r = docker(["run", "-d", "--name", cname, image, "sleep", "infinity"], docker_host=args.docker_host, timeout=60)
        if r.returncode != 0:
            result.update(status="run_failed", error=tail(r.stderr or r.stdout))
            return result
        result["run_sec"] = round(time.time() - t0, 3)

        r = docker(["exec", cname, "sh", "-lc", "command -v python3 >/dev/null 2>&1 && python3 --version"], docker_host=args.docker_host, timeout=30)
        if r.returncode == 0:
            result.update(status="already_has_python3", python3=tail(r.stdout, 200))
            return result

        install_cmd = r"""
set -eu
if command -v python3 >/dev/null 2>&1; then python3 --version; exit 0; fi
if [ -f /etc/apt/sources.list ]; then
  sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://security.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list || true
fi
find /etc/apt/sources.list.d -type f -name '*.sources' -exec sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://security.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' {} + 2>/dev/null || true
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends python3 ca-certificates
rm -rf /var/lib/apt/lists/*
python3 --version
"""
        t = time.time()
        r = docker(["exec", cname, "sh", "-lc", install_cmd], docker_host=args.docker_host, timeout=args.install_timeout)
        result["install_sec"] = round(time.time() - t, 3)
        if r.returncode != 0:
            result.update(status="install_failed", error=tail(r.stderr or r.stdout))
            return result
        result["python3"] = tail(r.stdout, 300)

        t = time.time()
        r = docker(["commit", cname, image], docker_host=args.docker_host, timeout=args.commit_timeout)
        result["commit_sec"] = round(time.time() - t, 3)
        if r.returncode != 0:
            result.update(status="commit_failed", error=tail(r.stderr or r.stdout))
            return result

        # We already verified `python3 --version` inside the source container
        # before committing.  A second `docker run --rm image python3 --version`
        # is semantically redundant and, under concurrent RL rollout, can hang
        # on Docker daemon scheduling rather than reveal image correctness.
        result.update(status="patched", verify=result.get("python3", ""))
        return result
    except subprocess.TimeoutExpired as exc:
        result.update(status="timeout", error=f"{exc.cmd} exceeded {exc.timeout}s")
        return result
    except Exception as exc:  # noqa: BLE001 - operational script must log and continue.
        result.update(status="exception", error=f"{type(exc).__name__}: {exc}")
        return result
    finally:
        docker(["rm", "-f", cname], docker_host=args.docker_host, timeout=30)
        result["total_sec"] = round(time.time() - t0, 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default=str(INVENTORY))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--docker-host",
        default="unix:///tmp/local-docker-overlay2.sock",
        help="Docker endpoint. Defaults to the maintained local overlay2 Docker socket.",
    )
    parser.add_argument("--image-prefix", default="unified-seta-synth")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", action="append", default=[], help="comma-separated task ids; may repeat")
    parser.add_argument("--include-bad", action="store_true", help="include known structurally bad SETA tasks")
    parser.add_argument("--needs-python-only", action="store_true", default=True)
    parser.add_argument("--all-mini-runner-safe", dest="needs_python_only", action="store_false")
    parser.add_argument("--install-timeout", type=int, default=300)
    parser.add_argument("--commit-timeout", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    task_ids = load_candidates(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[seta-py3] candidates={len(task_ids)} jobs={args.jobs} dry_run={args.dry_run} out={out}")
    counts: dict[str, int] = {}
    with out.open("a", buffering=1) as fh, futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        future_to_task = {pool.submit(patch_one, task_id, args): task_id for task_id in task_ids}
        for i, fut in enumerate(futures.as_completed(future_to_task), 1):
            row = fut.result()
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(task_ids)}] {row['status']:20s} {row['task_id']:>5s} {row.get('total_sec', 0):7.1f}s {row.get('error','')[:120]}")
    print("[seta-py3] summary", json.dumps(counts, ensure_ascii=False, sort_keys=True))
    return 0 if not counts.get("exception") and not counts.get("docker_error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
