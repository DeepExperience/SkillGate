#!/usr/bin/env python3
"""Remove stale Claw RL sandbox containers without touching the active step.

Claw-only RL creates many per-sample sandbox containers named ``claw-sb-*``.
The current rollout step's containers are created after the latest
``Starting rollout step`` timestamp in ``driver.log``. Containers created
before that timestamp belong to older steps and are safe to remove.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"Starting rollout step (\d+)")
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
OWNER_PID_RE = re.compile(r"-p([0-9]+)(?:-|$)")
CST = dt.timezone(dt.timedelta(hours=8))


def run_cmd(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def current_step_start(run_dir: Path) -> tuple[int, dt.datetime]:
    log_path = run_dir / "driver.log"
    current_step: int | None = None
    current_start: dt.datetime | None = None
    with log_path.open(errors="ignore") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line).rstrip()
            step_match = STEP_RE.search(line)
            if not step_match:
                continue
            ts_match = TS_RE.search(line)
            if not ts_match:
                continue
            current_step = int(step_match.group(1))
            current_start = dt.datetime.strptime(
                ts_match.group(1), "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=CST)
    if current_step is None or current_start is None:
        raise RuntimeError(f"cannot find current rollout step in {log_path}")
    return current_step, current_start


def container_rows(prefix: str) -> list[tuple[str, str, dt.datetime, str]]:
    result = run_cmd(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name={prefix}",
            "--format",
            "{{.ID}}\t{{.Names}}\t{{.CreatedAt}}\t{{.Status}}",
        ],
        timeout=45,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    rows: list[tuple[str, str, dt.datetime, str]] = []
    for line in result.stdout.splitlines():
        parts = line.split("\t", 3)
        if len(parts) != 4:
            continue
        container_id, name, created_raw, status = parts
        created = dt.datetime.strptime(
            created_raw.rsplit(" ", 1)[0], "%Y-%m-%d %H:%M:%S %z"
        ).astimezone(CST)
        rows.append((container_id, name, created, status))
    return rows


def remove_container(container_id: str, timeout: int) -> tuple[bool, str]:
    try:
        result = run_cmd(["docker", "rm", "-f", container_id], timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    output = (result.stderr or result.stdout).strip()
    return result.returncode == 0, output


def owner_pid_is_alive(name: str) -> bool:
    match = OWNER_PID_RE.search(name)
    if not match:
        return False
    proc_stat = Path("/proc") / match.group(1) / "stat"
    try:
        state = proc_stat.read_text().split()[2]
    except Exception:
        return False
    return state != "Z"


def cleanup_once(args: argparse.Namespace) -> int:
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        pointer = Path(args.run_pointer)
        if not pointer.exists():
            raise RuntimeError(f"run pointer not found: {pointer}")
        run_dir = Path(pointer.read_text().strip())
    if not run_dir.exists():
        raise RuntimeError(f"run dir not found: {run_dir}")
    step, step_start = current_step_start(run_dir)
    cutoff = step_start - dt.timedelta(seconds=args.grace_sec)
    scanned = 0
    stale: list[tuple[str, str, str, str]] = []

    for container_id, name, created, status in container_rows(args.prefix):
        scanned += 1
        if not name.startswith(args.prefix):
            continue
        if owner_pid_is_alive(name):
            continue
        if created < cutoff:
            stale.append(
                (
                    container_id,
                    name,
                    created.isoformat(),
                    status,
                )
            )

    removed = 0
    failed = 0
    print(
        f"[claw-cleanup] step={step} cutoff={cutoff.isoformat()} "
        f"scanned={scanned} stale={len(stale)} dry_run={args.dry_run}",
        flush=True,
    )
    for container_id, name, created, status in stale[: args.max_remove]:
        if args.dry_run:
            print(f"[claw-cleanup] dry-run remove {container_id} {status} {created} {name}")
            continue
        ok, detail = remove_container(container_id, args.rm_timeout_sec)
        if ok:
            removed += 1
        else:
            failed += 1
        print(
            f"[claw-cleanup] remove ok={ok} id={container_id} "
            f"status={status} created={created} name={name} detail={detail[-180:]}",
            flush=True,
        )
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)
    print(f"[claw-cleanup] removed={removed} failed={failed}", flush=True)
    return 0 if failed == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-pointer",
        default="experiments/rl/current/latest.txt",
    )
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--prefix", default="claw-sb-")
    parser.add_argument("--grace-sec", type=int, default=30)
    parser.add_argument("--rm-timeout-sec", type=int, default=25)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument("--max-remove", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-sec", type=int, default=300)
    args = parser.parse_args()

    while True:
        try:
            rc = cleanup_once(args)
        except Exception as exc:
            print(f"[claw-cleanup] error: {exc!r}", file=sys.stderr, flush=True)
            rc = 1
        if not args.loop:
            return rc
        time.sleep(max(30, args.interval_sec))


if __name__ == "__main__":
    raise SystemExit(main())
