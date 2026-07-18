#!/usr/bin/env python3
"""Conservative stale-container guard for Relax RL local Docker runs.

The script is meant to run beside an RL job.  It only manages containers
labelled by the unified runners as ``relax.managed=true``.  By default it:

* removes old non-running containers for the selected run;
* removes running containers only when they belong to older rollout steps and
  are older than a large threshold;
* stops cleaning immediately when Docker stops responding.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"(?:Start rollout|Starting rollout step) (\d+)")
UTC = dt.timezone.utc


def run_cmd(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, env=dict(os.environ))


def docker_ok(timeout: int) -> bool:
    try:
        result = run_cmd(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def parse_created(value: str) -> dt.datetime:
    value = value.replace("Z", "+00:00")
    if "." in value:
        head, tail = value.split(".", 1)
        tz = "+00:00"
        if "+" in tail:
            frac, tz_tail = tail.split("+", 1)
            tz = "+" + tz_tail
        elif "-" in tail:
            frac, tz_tail = tail.split("-", 1)
            tz = "-" + tz_tail
        else:
            frac = tail
        value = f"{head}.{frac[:6]}{tz}"
    return dt.datetime.fromisoformat(value).astimezone(UTC)


def latest_rollout_id(run_dir: str | None, driver_log: str | None) -> int | None:
    log_path: Path | None = None
    if driver_log:
        log_path = Path(driver_log)
    elif run_dir:
        log_path = Path(run_dir) / "driver.log"
    if not log_path or not log_path.exists():
        return None
    latest: int | None = None
    with log_path.open(errors="ignore") as handle:
        for raw in handle:
            line = ANSI_RE.sub("", raw)
            match = STEP_RE.search(line)
            if match:
                latest = int(match.group(1))
    return latest


def inspect_managed(timeout: int) -> list[dict[str, Any]]:
    ids_result = run_cmd(
        ["docker", "ps", "-a", "--filter", "label=relax.managed=true", "--format", "{{.ID}}"],
        timeout=timeout,
    )
    if ids_result.returncode != 0:
        raise RuntimeError((ids_result.stderr or ids_result.stdout).strip())
    ids = [line.strip() for line in ids_result.stdout.splitlines() if line.strip()]
    if not ids:
        return []
    rows: list[dict[str, Any]] = []
    for idx in range(0, len(ids), 64):
        chunk = ids[idx : idx + 64]
        result = run_cmd(["docker", "inspect", *chunk], timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip())
        rows.extend(json.loads(result.stdout))
    return rows


def pid_alive(pid_raw: str | None) -> bool:
    if not pid_raw or not pid_raw.isdigit():
        return False
    stat = Path("/proc") / pid_raw / "stat"
    try:
        state = stat.read_text().split()[2]
    except Exception:
        return False
    return state != "Z"


def select_stale(args: argparse.Namespace, rows: list[dict[str, Any]], current_step: int | None) -> list[dict[str, Any]]:
    now = dt.datetime.now(UTC)
    selected: list[dict[str, Any]] = []
    for row in rows:
        labels = row.get("Config", {}).get("Labels") or {}
        name = (row.get("Name") or "").lstrip("/")
        run_id = labels.get("relax.run_id", "")
        owner_pid = labels.get("relax.owner_pid", "")
        rollout_raw = labels.get("relax.rollout_id", "")
        if args.run_id and run_id != args.run_id:
            continue
        if args.owner_pid and owner_pid != str(args.owner_pid):
            continue
        if args.name_prefix and not name.startswith(args.name_prefix):
            continue
        state = (row.get("State", {}) or {}).get("Status", "unknown")
        created = parse_created(row.get("Created", "1970-01-01T00:00:00Z"))
        age = (now - created).total_seconds()
        rollout_id: int | None = int(rollout_raw) if rollout_raw.isdigit() else None
        reason = ""
        if state != "running" and age >= args.non_running_age_sec:
            reason = f"non_running:{state}"
        elif (
            state == "running"
            and args.remove_running_after_sec > 0
            and age >= args.remove_running_after_sec
            and current_step is not None
            and rollout_id is not None
            and rollout_id <= current_step - args.keep_recent_steps
        ):
            reason = f"old_running:rollout={rollout_id}:current={current_step}"
        elif (
            state == "running"
            and args.remove_dead_owner_running
            and age >= args.remove_running_after_sec
            and owner_pid
            and not pid_alive(owner_pid)
        ):
            reason = f"dead_owner_running:pid={owner_pid}"
        if not reason:
            continue
        selected.append(
            {
                "id": row.get("Id", ""),
                "short_id": row.get("Id", "")[:12],
                "name": name,
                "state": state,
                "age": int(age),
                "run_id": run_id,
                "rollout_id": rollout_id,
                "owner_pid": owner_pid,
                "reason": reason,
            }
        )
    selected.sort(key=lambda item: (item["state"] == "running", -item["age"]))
    return selected


def cleanup_once(args: argparse.Namespace) -> int:
    if not docker_ok(args.docker_timeout_sec):
        print("[rl-stale-cleaner] docker unhealthy; skip this cycle", flush=True)
        return 2
    current_step = latest_rollout_id(args.run_dir, args.driver_log)
    rows = inspect_managed(args.docker_timeout_sec)
    stale = select_stale(args, rows, current_step)
    running_removed = 0
    removed = 0
    failed = 0
    print(
        f"[rl-stale-cleaner] current_step={current_step} managed={len(rows)} "
        f"stale={len(stale)} dry_run={args.dry_run}",
        flush=True,
    )
    for item in stale:
        if removed >= args.max_remove:
            break
        if item["state"] == "running":
            if running_removed >= args.max_running_remove:
                continue
            running_removed += 1
        if args.dry_run:
            print(f"[rl-stale-cleaner] dry-run remove {item}", flush=True)
            removed += 1
            continue
        try:
            result = run_cmd(["docker", "rm", "-f", item["id"]], timeout=args.rm_timeout_sec)
        except subprocess.TimeoutExpired:
            failed += 1
            print(f"[rl-stale-cleaner] remove timeout item={item}", flush=True)
            break
        ok = result.returncode == 0
        if ok:
            removed += 1
        else:
            failed += 1
        detail = (result.stderr or result.stdout).strip()[-240:]
        print(f"[rl-stale-cleaner] remove ok={ok} item={item} detail={detail}", flush=True)
        if failed >= args.stop_after_failures:
            print("[rl-stale-cleaner] too many remove failures; stop this cycle", flush=True)
            break
        time.sleep(args.sleep_sec)
    print(f"[rl-stale-cleaner] removed={removed} failed={failed} running_removed={running_removed}", flush=True)
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=os.environ.get("RELAX_RL_RUN_ID", ""))
    parser.add_argument("--owner-pid", default="")
    parser.add_argument("--name-prefix", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--driver-log", default="")
    parser.add_argument("--docker-timeout-sec", type=int, default=12)
    parser.add_argument("--rm-timeout-sec", type=int, default=20)
    parser.add_argument("--non-running-age-sec", type=int, default=120)
    parser.add_argument("--remove-running-after-sec", type=int, default=3600)
    parser.add_argument("--keep-recent-steps", type=int, default=2)
    parser.add_argument("--max-remove", type=int, default=32)
    parser.add_argument("--max-running-remove", type=int, default=4)
    parser.add_argument("--stop-after-failures", type=int, default=2)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument("--remove-dead-owner-running", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-sec", type=int, default=120)
    args = parser.parse_args()

    while True:
        try:
            rc = cleanup_once(args)
        except Exception as exc:
            print(f"[rl-stale-cleaner] error: {exc!r}", file=sys.stderr, flush=True)
            rc = 1
        if not args.loop:
            return rc
        time.sleep(max(30, args.interval_sec))


if __name__ == "__main__":
    raise SystemExit(main())
