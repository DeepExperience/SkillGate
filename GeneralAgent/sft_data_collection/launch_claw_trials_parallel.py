#!/usr/bin/env python3
"""Concurrent SFT-collection launcher for claw trials.

The default `launch_trials.py` serialises claw trials via `claw_lock` +
`.claw_global.lock` because independent subprocesses share:
  - shared mock container `claw-mock-shared` (atexit cleanup races)
  - sandbox container `claw-sb-<task>` (rm conflict)
  - workspace dir `/tmp/claw_pilot/<task>/workspace` (rmtree race)
  - mock service ports (each task picks 9100/9101/...; multi-subprocess collides)

This launcher avoids those by assigning a stable `CLAW_WORKER_IDX` to each
thread (0..N-1). `run_unified_claw.py` reads that env and:
  - suffixes sandbox cname with `-w<idx>` → no rm conflict
  - suffixes workdir with `-w<idx>` → no rmtree race
  - shifts mock-service ports by `idx*100` → disjoint port ranges
  - skips atexit cleanup of shared mock (this parent does ONE cleanup at end)

Usage:
    python3 GeneralAgent/sft_data_collection/launch_claw_trials_parallel.py \\
        --plan <plan.jsonl> \\
        --workers 8 \\
        --per-trial-timeout-sec 2400

The plan format is the same as launch_trials.py emits (with env, argv,
trial_id, etc.). Only claw trials are accepted; others are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reuse helpers from launch_trials/common
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROJECT_ROOT, experiment_status_path, repo_path, secrets_path  # noqa: E402

# Reuse env-building logic from launch_trials so behaviour matches
from launch_trials import build_process_env, load_env_secrets as load_secrets, append_jsonl  # noqa: E402


PATH_PREFIX = os.environ.get(
    "SKILLRL_PATH_PREFIX", os.path.dirname(sys.executable) + ":/root/.local/bin"
)


def read_plan(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def already_done(record: dict, run_root: Path) -> bool:
    """A trial is 'done' if its trajectory file exists and is non-empty."""
    traj_rel = record.get("trajectory_path")
    if not traj_rel:
        return False
    p = repo_path(traj_rel)
    return p.exists() and p.stat().st_size > 0


def run_one_trial(
    record: dict[str, Any],
    worker_idx: int,
    secrets: dict[str, str],
    per_trial_timeout_sec: int,
    docker_wait_sec: int,
    api_base_override: str | None,
    status_path: Path,
    status_lock: threading.Lock,
    stop_event: threading.Event,
) -> dict[str, Any]:
    """Run one trial subprocess with CLAW_WORKER_IDX env injected."""
    if stop_event.is_set():
        return {"trial_id": record["trial_id"], "returncode": -1,
                "error_kind": "skipped", "elapsed_sec": 0.0,
                "worker_idx": worker_idx}

    log_path = repo_path(record["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process_env = build_process_env(record, secrets)
    # 2026-05-11 concurrent-safety hooks (handshake with run_unified_claw.py)
    process_env["CLAW_WORKER_IDX"] = str(worker_idx)
    process_env["CLAW_SKIP_SHARED_CLEANUP"] = "1"
    # 2026-05-11 train/infer alignment env — these MUST be set or claw runs
    # in host mode (mock at localhost) instead of docker sandbox mode
    # (mock at host.docker.internal). Without docker sandbox, the model
    # produces `localhost:PORT` URLs that pollute the SFT training data.
    # Codex's serial wrapper exported these in the shell; our parallel
    # launcher must inject them explicitly because we spawn from a fresh tmux.
    process_env["UNIFIED_CLAW_USE_DOCKER_SANDBOX"] = "1"
    process_env.setdefault("UNIFIED_PROMPT_PROFILE", "openclaw_full")
    process_env.setdefault("UNIFIED_TOOLS_SCHEMA_MODE", "openai_tools")
    if api_base_override:
        process_env["OPENAI_API_BASE"] = api_base_override

    argv = list(record["argv"])
    if api_base_override:
        # Replace any --api-base in argv with our override (parity with launch_trials)
        for i, a in enumerate(argv):
            if a == "--api-base" and i + 1 < len(argv):
                argv[i + 1] = api_base_override
                break

    started_at = datetime.now(timezone.utc).isoformat()
    start = time.time()
    returncode = -1
    error_kind = ""

    try:
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"# trial_id: {record['trial_id']}\n")
            logf.write(f"# worker_idx: {worker_idx}\n")
            logf.write(f"# cmd: {' '.join(argv)}\n")
            logf.write(f"# started_at: {started_at}\n\n")
            logf.flush()
            proc = subprocess.Popen(
                argv, env=process_env, stdout=logf, stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            try:
                returncode = proc.wait(timeout=per_trial_timeout_sec)
            except subprocess.TimeoutExpired:
                error_kind = "timeout"
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
                returncode = -signal.SIGKILL
    except Exception as e:
        error_kind = type(e).__name__
        returncode = -1

    elapsed = time.time() - start
    finished_at = datetime.now(timezone.utc).isoformat()
    status = {
        "trial_id": record["trial_id"],
        "task_id": record.get("task_id"),
        "bench": record.get("bench"),
        "worker_idx": worker_idx,
        "returncode": returncode,
        "error_kind": error_kind,
        "elapsed_sec": round(elapsed, 1),
        "started_at": started_at,
        "finished_at": finished_at,
    }
    with status_lock:
        append_jsonl(status_path, status)
    return status


def cleanup_residual_containers() -> None:
    """Wipe any leftover claw-sb-w*-* sandbox containers + shared mock."""
    try:
        # List worker-tagged sandbox containers
        out = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=^claw-sb-w",
             "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=20,
        )
        names = [n for n in out.stdout.strip().split("\n") if n]
        if names:
            print(f"[cleanup] removing {len(names)} worker-tagged sandboxes", flush=True)
            subprocess.run(["docker", "rm", "-f", *names], capture_output=True, timeout=120)
    except Exception as e:
        print(f"[cleanup] sandbox cleanup error: {e}", flush=True)
    # Shared mock infra (only cleanup at end of run)
    try:
        subprocess.run(["docker", "rm", "-f", "claw-mock-shared"],
                       capture_output=True, timeout=60)
        subprocess.run(["docker", "network", "rm", "claw-net-shared"],
                       capture_output=True, timeout=30)
        print("[cleanup] shared mock infra removed", flush=True)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", required=True, help="Plan jsonl path")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent worker threads (= unique CLAW_WORKER_IDX slots)")
    parser.add_argument("--per-trial-timeout-sec", type=int, default=2400)
    parser.add_argument("--docker-wait-sec", type=int, default=54000)
    parser.add_argument("--api-base-override", default=None,
                        help="Override env.OPENAI_API_BASE on every trial")
    parser.add_argument("--task-window", type=int, default=0,
                        help="If >0, only run first N trials (smoke test)")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Skip trials whose trajectory file already exists")
    parser.add_argument("--no-cleanup-at-end", action="store_true",
                        help="Skip shared mock cleanup at end (keep for next invocation)")
    args = parser.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.exists():
        print(f"plan not found: {plan_path}", file=sys.stderr)
        return 2
    records = read_plan(plan_path)
    if not records:
        print("plan empty", file=sys.stderr)
        return 2

    # Filter: claw-only
    claw_records = [r for r in records if r.get("bench") == "claw"]
    skipped_bench = len(records) - len(claw_records)
    if skipped_bench:
        print(f"[skip] {skipped_bench} non-claw records ignored")

    run_id = claw_records[0].get("run_id") or "unknown_run"
    status_path = experiment_status_path(run_id)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_completed:
        before = len(claw_records)
        claw_records = [r for r in claw_records
                        if not already_done(r, status_path.parent.parent)]
        print(f"[skip-completed] {before - len(claw_records)} already done; {len(claw_records)} remaining")

    if args.task_window > 0:
        claw_records = claw_records[: args.task_window]
        print(f"[smoke] limiting to first {len(claw_records)} trials")

    if not claw_records:
        print("nothing to run")
        return 0

    secrets = load_secrets()
    stop_event = threading.Event()
    status_lock = threading.Lock()

    # SIGINT/SIGTERM → flag stop
    def _sig(signum, _frame):
        print(f"\n[signal] {signum} received; not popping new trials, finishing in-flight…", flush=True)
        stop_event.set()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # Worker-fixed idx: stable mapping via Queue
    idx_queue: "queue.Queue[int]" = queue.Queue()
    for w in range(args.workers):
        idx_queue.put(w)
    thread_idx_map: dict[int, int] = {}
    map_lock = threading.Lock()

    def my_worker_idx() -> int:
        tid = threading.get_ident()
        with map_lock:
            if tid not in thread_idx_map:
                thread_idx_map[tid] = idx_queue.get_nowait()
            return thread_idx_map[tid]

    completed = 0
    failed = 0
    total = len(claw_records)
    start_t = time.time()

    def run_with_alloc(record: dict[str, Any]) -> dict[str, Any]:
        widx = my_worker_idx()
        return run_one_trial(
            record, widx, secrets,
            args.per_trial_timeout_sec, args.docker_wait_sec,
            args.api_base_override, status_path, status_lock, stop_event,
        )

    # 2026-05-11 prewarm shared mock infra to avoid race-condition when N
    # workers all try to start `claw-mock-shared` at once. Without prewarm,
    # several workers hit `docker: name in use` → mock_ip=None → sandbox
    # container fails to create → cname=None → tool_docs.replace() is skipped
    # → system prompt keeps `localhost:PORT` → SFT data pollution.
    print(f"[prewarm] ensuring shared mock infra is up before launching workers...", flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval_scripts"))
    from unified_runner.run_unified_claw import _ensure_shared_mock_infra  # noqa: E402
    mock_ip = _ensure_shared_mock_infra()
    if not mock_ip:
        print(f"[prewarm] ❌ failed to start shared mock infra; aborting", flush=True)
        return 3
    print(f"[prewarm] shared mock up at {mock_ip}", flush=True)

    print(f"[start] {total} claw trials with {args.workers} workers; run_id={run_id}", flush=True)
    print(f"[status] writing to {status_path}", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_with_alloc, r) for r in claw_records]
            for fut in futures:
                try:
                    s = fut.result()
                except Exception as e:
                    print(f"[err] thread exception: {e}\n{traceback.format_exc()[-300:]}", flush=True)
                    failed += 1
                    continue
                completed += 1
                ok = (s["returncode"] == 0 and not s.get("error_kind"))
                if not ok:
                    failed += 1
                if completed % 10 == 0 or completed == total:
                    elapsed = time.time() - start_t
                    rate = completed / max(elapsed, 1) * 3600
                    eta_h = (total - completed) / max(rate, 0.01)
                    print(f"[progress] {completed}/{total} done ({failed} failed) | "
                          f"{rate:.1f} trial/h | ETA {eta_h:.1f}h", flush=True)
    finally:
        if not args.no_cleanup_at_end:
            print("[cleanup] removing residual worker sandboxes + shared mock…", flush=True)
            cleanup_residual_containers()

    print(f"\n[done] total={total} ok={total - failed} failed={failed} "
          f"elapsed={(time.time() - start_t)/3600:.2f}h", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
