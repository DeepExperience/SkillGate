#!/usr/bin/env python3
"""Launch trial-plan records safely from inside a tmux session.

Reads a plan JSONL produced by make_trial_plan.py and runs the matching
records via subprocess.run, one runner invocation per trial. Does NOT use
the dynamic launcher or shared queue — each trial is independent because
SFT collection wants explicit per-trial accounting (which trial succeeded,
how many turns, etc).

Safety rails (don't disable without a reason):
  - dry-run by default; --execute is required to actually launch
  - rejects launching if the plan mixes models, or if SGLang's served model
    differs from the plan's model (--allow-model-mismatch overrides)
  - skips trials whose trajectory file already exists (--rerun-completed
    overrides) — useful for resuming after a crash
  - skips skill-arm trials whose task isn't covered by the retrieval
    jsonl (--allow-missing-retrieval overrides)
  - claw trials are serialized by default. When --allow-concurrent-claw is set,
    the launcher assigns each Claw subprocess a stable CLAW_WORKER_IDX so
    docker-sandbox mock ports/container names do not collide, matching the RL
    ClawLauncher slot-isolation design.
  - records are interleaved by (bench, task_id), and non-claw trials use a
    per-task lock, so multiple rollouts of the same task do not compete for
    Docker image/container/cache operations at the same time

Per-trial timeout (--per-trial-timeout-sec, default 1800 = 30min) prevents
a single hung agent from blocking the worker pool forever.

Anti-missing retry (--retry-rounds, default 2): after the main pass drains,
trials that failed at the launcher level AND left no trajectory on disk
(missing_trajectory / docker_unavailable / timeout / nonzero rc) are requeued
for up to N extra rounds at reduced concurrency (--retry-workers, default
min(8, --workers)). Transient Docker pressure is the dominant cause of these
holes, so the lower-concurrency tail pass usually recovers them and a finished
run needs no manual top-up. Genuine task failures (rc=0, graded unresolved)
and trials whose trajectory exists are never retried.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections import Counter, defaultdict, deque

from common import (
    PROJECT_ROOT,
    append_jsonl,
    display_path,
    experiment_status_path,
    repo_path,
    secrets_path,
)


# Slime conda env binaries; prepended to PATH so subprocesses see the right
# python/pip/uv. /root/.local/bin covers user-installed CLI tools.
PATH_PREFIX = os.environ.get(
    "SKILLRL_PATH_PREFIX", os.path.dirname(sys.executable) + ":/root/.local/bin"
)

# Default SGLang endpoint to query for served-model assertion. Individual
# plan records may override this through env.OPENAI_API_BASE.
DEFAULT_OPENAI_API_BASE = "http://127.0.0.1:30000/v1"

DOCKER_UNAVAILABLE_SIGNATURES = (
    "Cannot connect to the Docker daemon",
    "Is the docker daemon running?",
    "error during connect",
)

DOCKER_UNAVAILABLE_RETURNCODE = -75
MISSING_TRAJECTORY_RETURNCODE = -76


# ---------------------------------------------------------------------------
# Plan / env loading
# ---------------------------------------------------------------------------

def load_plan(plan_path: str | Path) -> list[dict[str, Any]]:
    path = repo_path(plan_path)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            line = raw_line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_env_secrets() -> dict[str, str]:
    """Read secrets/.env.secrets (KEY=VALUE per line, # comments OK).

    File expected to be 0600 perms; we don't enforce that here but the
    project convention is documented in CLAUDE.md.
    """
    path = secrets_path()
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        # Strip surrounding quotes if present.
        values[key.replace("export ", "").strip()] = value.strip().strip("\"'")
    return values


def redact_env(env: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in env.items():
        if any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted


def models_url_from_api_base(api_base: str) -> str:
    return api_base.rstrip("/") + "/models"


def endpoint_api_key(api_base: str, record_env: dict[str, Any], secrets: dict[str, str]) -> str:
    """Return the bearer key for an endpoint without requiring plans to store it."""
    maas_base = (
        record_env.get("MAAS_API_BASE")
        or os.environ.get("MAAS_API_BASE")
        or secrets.get("MAAS_API_BASE")
        or ""
    )
    maas_key = (
        record_env.get("TEACHER_OPENAI_API_KEY")
        or os.environ.get("TEACHER_OPENAI_API_KEY")
        or record_env.get("MAAS_API_KEY")
        or os.environ.get("MAAS_API_KEY")
        or secrets.get("MAAS_API_KEY")
        or ""
    )
    if maas_base and api_base.rstrip("/") == str(maas_base).rstrip("/") and maas_key:
        return str(maas_key)
    return str(
        record_env.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or secrets.get("OPENAI_API_KEY")
        or ""
    )


def get_served_models(api_base: str = DEFAULT_OPENAI_API_BASE, api_key: str = "") -> list[str]:
    """Query OpenAI-compatible /v1/models and return available model ids."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(models_url_from_api_base(api_base), headers=headers)
    with urllib.request.urlopen(request, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [
        str(item.get("id", ""))
        for item in (payload.get("data") or [])
        if isinstance(item, dict) and item.get("id")
    ]


def text_has_docker_unavailable(text: str) -> bool:
    return any(signature in text for signature in DOCKER_UNAVAILABLE_SIGNATURES)


def latest_incremental_text(path_value: str | Path) -> str:
    path = repo_path(path_value)
    if not path.exists():
        return ""
    try:
        lines = [line for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    except OSError:
        return ""
    return lines[-1] if lines else ""


def has_final_incremental_record(path_value: str | Path) -> bool:
    text = latest_incremental_text(path_value)
    if not text:
        return False
    try:
        json.loads(text)
    except Exception:
        return False
    return True


def classify_missing_trajectory(record: dict[str, Any], log_path: Path) -> str:
    texts = [latest_incremental_text(record.get("incremental_path", ""))]
    try:
        texts.append(log_path.read_text(encoding="utf-8", errors="ignore")[-4000:])
    except OSError:
        pass
    if any(text_has_docker_unavailable(text) for text in texts):
        return "docker_unavailable"
    return "missing_trajectory"


def docker_version_ok(process_env: dict[str, str], timeout_sec: int = 10) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            cwd=PROJECT_ROOT,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    output = (completed.stdout or "").strip()
    return completed.returncode == 0, output


def wait_for_docker(
    process_env: dict[str, str],
    log_file: Any,
    *,
    wait_sec: int,
    interval_sec: int,
) -> bool:
    if wait_sec <= 0 or not process_env.get("DOCKER_HOST"):
        return True
    deadline = time.time() + wait_sec
    attempt = 0
    while True:
        attempt += 1
        ok, detail = docker_version_ok(process_env)
        if ok:
            if attempt > 1:
                log_file.write(f"[launcher] docker available after {attempt} checks: {detail}\n")
                log_file.flush()
            return True
        remaining = deadline - time.time()
        log_file.write(
            f"[launcher] docker unavailable attempt={attempt} remaining={max(0, int(remaining))}s: {detail}\n"
        )
        log_file.flush()
        if remaining <= 0:
            return False
        time.sleep(max(1, min(interval_sec, int(remaining))))


# ---------------------------------------------------------------------------
# Selection (filter the plan down to what we want to run now)
# ---------------------------------------------------------------------------

def filter_records(
    records: list[dict[str, Any]],
    benches: set[str],
    modes: set[str],
    models: set[str],
    rerun_completed: bool,
    allow_missing_retrieval: bool,
) -> list[dict[str, Any]]:
    """Apply CLI filters to plan records. Empty filter set = no constraint.

    `--limit` is intentionally applied after `interleave_records_by_task`.
    If we limit while scanning the plan, a pilot can accidentally select the
    first task's 8 rollouts only, which is exactly the pattern that overloads
    Docker and fails to test cross-bench behavior.
    """
    selected: list[dict[str, Any]] = []
    for record in records:
        if benches and record["bench"] not in benches:
            continue
        if modes and record["mode"] not in modes:
            continue
        if models and record["model"] not in models:
            continue
        # Don't blindly run a retrieval-backed arm against a task the retrieval
        # pipeline doesn't have an entry for — would silently degrade to
        # baseline and contaminate the bucket assignment later.
        if (
            not allow_missing_retrieval
            and record["arm"] in {"retrieval", "top1_skill_text"}
            and not record.get("retrieval_covered")
        ):
            continue
        # Resume support: skip trials only after the final result row exists.
        # A killed runner can leave a partial trajectory or empty incremental
        # file; treating that as completed leaves holes in pass@k accounting.
        if not rerun_completed and has_final_incremental_record(record.get("incremental_path", "")):
            continue
        selected.append(record)
    return selected


def interleave_records_by_task(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin records across benches, then across (bench, task_id) groups.

    Plans are naturally grouped by task, then mode, then trial. Submitting that
    order to a ThreadPoolExecutor makes the first `workers` slots all compete
    on the same Docker image/container/cache paths. Interleaving gives the
    worker pool distinct benches/tasks first while preserving per-task trial
    order. Bench-level round-robin matters because claw has a global lock; if
    the first N submitted records are all claw, worker threads just block.
    """
    groups: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    bench_order: list[str] = []
    task_order_by_bench: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record in records:
        bench = record["bench"]
        if bench not in bench_order:
            bench_order.append(bench)
        key = (bench, str(record["task_id"]))
        if key not in groups:
            task_order_by_bench[bench].append(key)
        groups[key].append(record)

    task_order: list[tuple[str, str]] = []
    per_bench_queues = {
        bench: deque(keys)
        for bench, keys in task_order_by_bench.items()
    }
    while any(per_bench_queues[bench] for bench in bench_order):
        for bench in bench_order:
            if per_bench_queues[bench]:
                task_order.append(per_bench_queues[bench].popleft())

    interleaved: list[dict[str, Any]] = []
    remaining = True
    while remaining:
        remaining = False
        for key in task_order:
            if groups[key]:
                interleaved.append(groups[key].popleft())
                remaining = True
    return interleaved


def task_window_groups(
    records: list[dict[str, Any]],
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], deque[dict[str, Any]]]]:
    """Group records by task and order tasks round-robin by bench.

    Unlike interleave_records_by_task(), this keeps each task's trial queue
    intact. The scheduler can then maintain a rolling window of tasks: when one
    task exhausts its trials, the next task enters immediately. This avoids the
    old chunk-tail idle problem without delaying task-level teacher fallback
    until the entire full plan has made one pass.
    """
    groups: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    bench_order: list[str] = []
    task_order_by_bench: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record in records:
        bench = str(record["bench"])
        if bench not in bench_order:
            bench_order.append(bench)
        key = (bench, str(record["task_id"]))
        if key not in groups:
            task_order_by_bench[bench].append(key)
        groups[key].append(record)

    per_bench_queues = {
        bench: deque(keys)
        for bench, keys in task_order_by_bench.items()
    }
    task_order: list[tuple[str, str]] = []
    while any(per_bench_queues[bench] for bench in bench_order):
        for bench in bench_order:
            if per_bench_queues[bench]:
                task_order.append(per_bench_queues[bench].popleft())
    return task_order, groups


# ---------------------------------------------------------------------------
# Per-trial execution
# ---------------------------------------------------------------------------

def build_process_env(record: dict[str, Any], secrets: dict[str, str]) -> dict[str, str]:
    """Compose env for a single subprocess.

    Precedence (later wins):
      1. inherited os.environ
      2. .env.secrets values (only for keys NOT already in env — explicit
         shell exports override secrets file by design)
      3. record['env'] from the plan (always wins; plan owns per-trial env)
      4. PATH prefix prepended last so PATH_PREFIX is searched first
    """
    process_env = os.environ.copy()
    for key, value in secrets.items():
        if key not in process_env:
            process_env[key] = value
    process_env.update(record.get("env", {}))
    api_base = process_env.get("OPENAI_API_BASE", DEFAULT_OPENAI_API_BASE)
    api_key = endpoint_api_key(api_base, record.get("env", {}), secrets)
    if api_key:
        process_env["OPENAI_API_KEY"] = api_key
    process_env["PATH"] = f"{PATH_PREFIX}:{process_env.get('PATH', '')}"
    return process_env


def assign_claw_worker_slots(records: list[dict[str, Any]]) -> None:
    """Assign deterministic Claw worker slots for subprocess-level concurrency.

    RL's ClawLauncher uses a per-sample unique salt and maps it into 100-port
    slots. The eval launcher starts one subprocess per trial, so it must inject
    a slot explicitly; otherwise sibling subprocesses race on the same shared
    mock container/network and port range. Keep slots <=512 because Claw ports
    are shifted by `worker_idx * 100`.
    """
    claw_records = [record for record in records if str(record.get("bench")) == "claw"]
    if not claw_records:
        return
    max_slots = int(os.environ.get("UNIFIED_CLAW_PORT_SLOTS", "512") or "512")
    max_slots = max(1, min(max_slots, 512))
    # Trials of the same task are serialized by the launcher's per-task lock,
    # so they can never run concurrently and may share one slot. Only distinct
    # tasks need distinct slots to keep mock ports/containers isolated. This
    # is what lets multi-rollout plans (e.g. 8 trials/task) stay within the
    # 512-slot port design.
    task_order: list[str] = []
    for record in claw_records:
        task_id = str(record.get("task_id"))
        if task_id not in task_order:
            task_order.append(task_id)
    if len(task_order) > max_slots:
        raise SystemExit(
            f"selected {len(task_order)} distinct Claw tasks but only {max_slots} "
            "Claw port slots are available; lower concurrency or increase the "
            "slot design before using --allow-concurrent-claw"
        )
    slot_by_task = {task_id: idx for idx, task_id in enumerate(task_order, start=1)}
    for record in claw_records:
        env = record.setdefault("env", {})
        # Non-zero slot makes run_unified_claw add suffixes and port offsets.
        env.setdefault("CLAW_WORKER_IDX", str(slot_by_task[str(record.get("task_id"))]))


class _NullLock:
    """A drop-in for threading.Lock that does nothing. Used so non-claw
    trials can `with lock_context:` uniformly without branching."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        return False


class _FileLock:
    """Cross-process advisory lock.

    Thread locks only protect records inside one launcher process. The
    pipelined wrapper runs Phase 1 and Phase 2 launchers concurrently, so Claw
    needs a process-wide lock too; otherwise two independent launchers can hit
    shared mock infra at the same time.
    """

    def __init__(self, path: Path):
        self.path = path
        self.file_handle: Any | None = None

    def __enter__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = self.path.open("w", encoding="utf-8")
        fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_EX)
        return None

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        if self.file_handle is not None:
            fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_UN)
            self.file_handle.close()
            self.file_handle = None
        return False


def run_record(
    record: dict[str, Any],
    secrets: dict[str, str],
    status_path: Path,
    status_lock: threading.Lock,
    claw_lock: threading.Lock,
    task_lock: threading.Lock,
    allow_concurrent_claw: bool,
    per_trial_timeout_sec: int,
    docker_wait_sec: int,
    docker_check_interval_sec: int,
    stop_event: threading.Event,
    concurrent_trials: bool = False,
) -> dict[str, Any]:
    """Run one trial. Returns a status dict (also appended to status.jsonl).

    Exit conditions:
      - subprocess returns normally → returncode = whatever it gave
      - timeout → returncode = -signal.SIGKILL, error_kind = "timeout"
      - stop_event set (Ctrl-C) → don't start; return error_kind = "skipped"
      - other exception → returncode = -1, error_kind = exception class name
    """
    if stop_event.is_set():
        return {
            "trial_id": record["trial_id"],
            "returncode": -1,
            "error_kind": "skipped",
            "elapsed_sec": 0.0,
        }

    log_path = repo_path(record["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process_env = build_process_env(record, secrets)
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.time()
    lock_acquired_time: float | None = None
    subprocess_start_time: float | None = None
    subprocess_elapsed_sec = 0.0
    error_kind = ""
    returncode = -1

    # Claw shares mock infra; running >1 claw at once causes docker exec
    # contention. Other benches may run concurrently, but not two rollouts
    # of the same task: those compete on Docker image/container/cache paths.
    serialize_claw = record["bench"] == "claw" and not allow_concurrent_claw
    if concurrent_trials and record["bench"] != "claw":
        # Opt-in (--concurrent-trials): let multiple rollouts (repeats) of the
        # same non-claw task run at once — each gets its own container + trial
        # dir — so the worker pool fills past the distinct-task count. Claw keeps
        # the per-task lock below because its mock port slot is assigned per task.
        lock_context = _NullLock()
    else:
        lock_context = claw_lock if serialize_claw else task_lock
    cross_process_lock = (
        _FileLock(experiment_status_path(record["run_id"]).parent / ".claw_global.lock")
        if serialize_claw
        else _NullLock()
    )
    with lock_context:
        with cross_process_lock:
            lock_acquired_time = time.time()
            # Re-check stop_event AFTER acquiring locks — user might have hit
            # Ctrl-C while we were waiting for claw/task serialization.
            if stop_event.is_set():
                return {
                    "trial_id": record["trial_id"],
                    "returncode": -1,
                    "error_kind": "skipped",
                    "elapsed_sec": round(time.time() - start_time, 1),
                }

            with log_path.open("w", encoding="utf-8") as log_file:
                # Header so we can grep `trial_id=` in logs.
                log_file.write(json.dumps({
                    "trial_id": record["trial_id"],
                    "started_at": started_at,
                    "argv": record["argv"],
                    "env": redact_env(record["env"]),
                    "per_trial_timeout_sec": per_trial_timeout_sec,
                }, ensure_ascii=False, indent=2) + "\n\n")
                log_file.flush()
                try:
                    if not wait_for_docker(
                        process_env,
                        log_file,
                        wait_sec=docker_wait_sec,
                        interval_sec=docker_check_interval_sec,
                    ):
                        error_kind = "docker_unavailable"
                        returncode = DOCKER_UNAVAILABLE_RETURNCODE
                        log_file.write("[launcher] docker wait exhausted; trial not launched\n")
                        log_file.flush()
                    else:
                        subprocess_start_time = time.time()
                        completed = subprocess.run(
                            record["argv"],
                            cwd=PROJECT_ROOT,
                            env=process_env,
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            text=True,
                            # Hard cap so a stuck agent (waiting on a hung docker
                            # exec, frozen pty, etc.) doesn't tie up a worker.
                            timeout=per_trial_timeout_sec,
                        )
                        subprocess_elapsed_sec = time.time() - subprocess_start_time
                        returncode = completed.returncode
                except subprocess.TimeoutExpired:
                    if subprocess_start_time is not None:
                        subprocess_elapsed_sec = time.time() - subprocess_start_time
                    error_kind = "timeout"
                    returncode = -signal.SIGKILL
                    log_file.write(f"\n\n[launcher] TIMEOUT after {per_trial_timeout_sec}s\n")
                except Exception as exc:  # subprocess errors, OOM, etc.
                    if subprocess_start_time is not None:
                        subprocess_elapsed_sec = time.time() - subprocess_start_time
                    error_kind = type(exc).__name__
                    log_file.write(f"\n\n[launcher] EXCEPTION {error_kind}: {exc}\n")

    if returncode == 0 and not repo_path(record["trajectory_path"]).exists():
        error_kind = classify_missing_trajectory(record, log_path)
        if error_kind == "docker_unavailable":
            returncode = DOCKER_UNAVAILABLE_RETURNCODE
        else:
            returncode = MISSING_TRAJECTORY_RETURNCODE
    if returncode == 0 and not has_final_incremental_record(record.get("incremental_path", "")):
        error_kind = classify_missing_trajectory(record, log_path)
        if error_kind == "docker_unavailable":
            returncode = DOCKER_UNAVAILABLE_RETURNCODE
        else:
            returncode = MISSING_TRAJECTORY_RETURNCODE

    finished_at = datetime.now(timezone.utc).isoformat()
    total_elapsed_sec = time.time() - start_time
    lock_wait_sec = (
        (lock_acquired_time - start_time)
        if lock_acquired_time is not None
        else total_elapsed_sec
    )
    status = {
        "trial_id": record["trial_id"],
        "run_id": record["run_id"],
        "bench": record["bench"],
        "task_id": record["task_id"],
        "mode": record["mode"],
        "model": record["model"],
        "returncode": returncode,
        "error_kind": error_kind,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_sec": round(total_elapsed_sec, 1),
        "lock_wait_sec": round(lock_wait_sec, 1),
        "subprocess_elapsed_sec": round(subprocess_elapsed_sec, 1),
        "log_path": record["log_path"],
        "trajectory_path": record["trajectory_path"],
        "incremental_path": record["incremental_path"],
    }
    with status_lock:
        append_jsonl(status_path, status)
    return status


def parse_bench_caps(values: list[str]) -> dict[str, int]:
    """Parse repeatable/comma-separated bench caps like `claw=1,tb2=2`.

    These caps limit how many subprocesses for each bench are submitted at
    once. They complement the existing task/claw locks: caps protect
    Docker/verifier resources at bench granularity, while locks protect known
    shared paths and Claw mock infra.
    """
    caps: dict[str, int] = {}
    for raw_value in values:
        for item in str(raw_value).split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise SystemExit(f"invalid --bench-cap {item!r}; expected bench=N")
            bench, raw_cap = item.split("=", 1)
            bench = bench.strip()
            try:
                cap = int(raw_cap)
            except ValueError as exc:
                raise SystemExit(f"invalid --bench-cap {item!r}; N must be an integer") from exc
            if cap <= 0:
                raise SystemExit(f"invalid --bench-cap {item!r}; N must be > 0")
            caps[bench] = cap
    return caps


# ---------------------------------------------------------------------------
# Pre-flight assertions
# ---------------------------------------------------------------------------

def assert_model_selection(
    selected: list[dict[str, Any]],
    allow_model_mismatch: bool,
    secrets: dict[str, str],
) -> None:
    """Refuse to run if the plan mixes models, or if the endpoint lacks the model.

    Mixing models in one launcher invocation would silently route some trials
    against the wrong model. The user must explicitly filter by --model.
    """
    models = {record["model"] for record in selected}
    api_bases = {
        record.get("env", {}).get("OPENAI_API_BASE", DEFAULT_OPENAI_API_BASE)
        for record in selected
    }
    if not models:
        return
    if len(models) > 1:
        raise SystemExit(
            f"selected multiple models {sorted(models)}; "
            "filter to one model with --model before launching"
        )
    if len(api_bases) > 1:
        raise SystemExit(
            f"selected multiple OPENAI_API_BASE endpoints {sorted(api_bases)}; "
            "launch one endpoint/model group at a time"
        )
    selected_model = next(iter(models))
    api_base = next(iter(api_bases))
    record_env = selected[0].get("env", {})
    api_key = endpoint_api_key(api_base, record_env, secrets)
    try:
        served_models = get_served_models(api_base, api_key)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(
            f"cannot reach model endpoint at {models_url_from_api_base(api_base)} ({exc}); "
            "start the endpoint or fix the API base/key"
        ) from exc
    if selected_model not in served_models and not allow_model_mismatch:
        raise SystemExit(
            f"endpoint {api_base} does not list {selected_model!r}; "
            f"available={served_models[:20]!r}. "
            "Fix the model/API base, or pass --allow-model-mismatch only if you know why."
        )


def run_remote_mysql_cleanup() -> None:
    """Pre-flight: clean up stale leaked mysql_<pid> containers on the remote Docker host.

    Why: another tenant on the shared Docker host runs LiveMCP simulations that spawn
    mysql:8.0 sidecars and sometimes leave them stuck in Created/Exited
    state. Each stale container slows our docker exec / docker cp via
    boltdb metadata + global dockerd lock contention. The cleanup script
    only removes mysql:8.0 containers that have been in non-Up state for
    >5 minutes, so it can't race their in-flight containers.

    Best-effort: failures here are logged but don't block the launch.
    The cleanup is opt-out via --skip-mysql-cleanup; it's enabled by
    default because forgetting to clean up directly increases trial
    failure rate (we saw 25% inject failures without it).
    """
    script_path = PROJECT_ROOT / "ops" / "cleanup" / "cleanup_remote_mysql_leak.sh"
    if not script_path.is_file():
        print(f"[preflight] cleanup script missing: {script_path}; skipping", flush=True)
        return
    try:
        result = subprocess.run(
            ["bash", str(script_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Print whatever the script said (succinct status lines).
        for line in (result.stdout + result.stderr).splitlines():
            if line.strip():
                print(f"[preflight] {line}", flush=True)
        if result.returncode != 0:
            print(f"[preflight] cleanup exited non-zero (rc={result.returncode}); continuing anyway", flush=True)
    except subprocess.TimeoutExpired:
        print("[preflight] cleanup timed out (>60s); continuing without cleanup", flush=True)
    except Exception as exc:
        print(f"[preflight] cleanup failed: {exc}; continuing without cleanup", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", required=True, help="Plan JSONL from make_trial_plan.py")
    parser.add_argument("--bench", action="append", default=[],
                        help="Bench filter (repeatable); empty = all")
    parser.add_argument("--mode", action="append", default=[],
                        help="Mode filter (student_baseline/student_retrieval/teacher_retrieval)")
    parser.add_argument("--model", action="append", default=[],
                        help="Model filter; required to differ between 9b and 27b")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap number of trials to launch (0 = no cap)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Thread pool size; claw is serialized regardless")
    parser.add_argument("--task-window", type=int, default=0,
                        help="Rolling number of distinct tasks eligible for scheduling. "
                             "0 uses the legacy full-plan round-robin order.")
    parser.add_argument("--concurrent-trials", action="store_true",
                        help="Allow multiple rollouts (repeats) of the SAME non-claw task "
                             "to run concurrently, filling the worker pool past the "
                             "distinct-task count (lets --workers actually reach 128). Claw "
                             "stays per-task serialized (its mock port slot is per task). "
                             "Default off = legacy 1-trial-per-task scheduling (unchanged).")
    parser.add_argument("--bench-cap", action="append", default=[],
                        help="Optional per-bench concurrency cap, e.g. "
                             "`--bench-cap claw=1,tb2=2` or repeat the flag")
    parser.add_argument("--per-trial-timeout-sec", type=int, default=1800,
                        help="Kill any single trial that runs longer than this (default 30min)")
    parser.add_argument("--docker-wait-sec", type=int, default=0,
                        help="If DOCKER_HOST is set and Docker is unavailable, wait this long before "
                             "recording docker_unavailable instead of burning a trial")
    parser.add_argument("--docker-check-interval-sec", type=int, default=30,
                        help="Polling interval while waiting for Docker")
    parser.add_argument("--execute", action="store_true",
                        help="Actually run trials. Without this, dry-run only.")
    parser.add_argument("--rerun-completed", action="store_true",
                        help="Don't skip trials whose trajectory_path already exists")
    parser.add_argument("--allow-missing-retrieval", action="store_true",
                        help="Run retrieval arm even for tasks not covered by retrieval jsonl")
    parser.add_argument("--allow-model-mismatch", action="store_true",
                        help="Skip the SGLang-served-model assertion")
    parser.add_argument("--allow-concurrent-claw", action="store_true",
                        help="Do not serialize Claw trials with the legacy global lock. "
                             "Use only for Docker-sandboxed OpenClaw-aligned evals where "
                             "Claw mock infra is isolated per container, matching RL rollouts.")
    parser.add_argument("--skip-mysql-cleanup", action="store_true",
                        help="Skip the remote-docker-host mysql leak cleanup preflight "
                             "(only do this if you know dockerd is healthy)")
    parser.add_argument("--api-base-override", default="",
                        help="If set, replace OPENAI_API_BASE in every plan record's env "
                             "and the --api-base flag inside record.argv. Useful when the "
                             "endpoint baked into the plan (e.g. teacher port 30001) is no "
                             "longer served and you want to redirect to another endpoint "
                             "without regenerating chunks.")
    parser.add_argument("--retry-rounds", type=int, default=2,
                        help="After the main pass, requeue launcher-level failures that "
                             "left no trajectory (missing_trajectory/docker_unavailable/"
                             "timeout/nonzero rc) for up to N extra rounds at lower "
                             "concurrency, so a finished run has no missing holes. "
                             "0 disables. Graded task failures (rc=0) are never retried.")
    parser.add_argument("--retry-workers", type=int, default=0,
                        help="Concurrency for retry rounds (0 = auto: min(8, --workers)). "
                             "Lower than the main pass on purpose: transient Docker "
                             "pressure is the dominant cause of missing trials.")
    parser.add_argument("--retry-cooldown-sec", type=int, default=60,
                        help="Sleep before each retry round so transient Docker/endpoint "
                             "pressure can drain first")
    args = parser.parse_args()

    records = load_plan(args.plan)
    if args.api_base_override:
        new_base = args.api_base_override.rstrip("/")
        for record in records:
            env = record.setdefault("env", {})
            env["OPENAI_API_BASE"] = new_base
            argv = record.get("argv")
            if isinstance(argv, list):
                for i, token in enumerate(argv):
                    if token == "--api-base" and i + 1 < len(argv):
                        argv[i + 1] = new_base
                record["command_preview"] = " ".join(str(part) for part in argv)
        print(f"[launcher] api-base-override applied: {new_base}", flush=True)
    selected = filter_records(
        records=records,
        benches=set(args.bench),
        modes=set(args.mode),
        models=set(args.model),
        rerun_completed=args.rerun_completed,
        allow_missing_retrieval=args.allow_missing_retrieval,
    )
    if args.task_window <= 0:
        selected = interleave_records_by_task(selected)
    if args.limit > 0:
        selected = selected[:args.limit]
    print(f"selected {len(selected)} / {len(records)} records")
    for record in selected[:5]:
        print(f"  - {record['trial_id']} :: {record['command_preview']}")
    if len(selected) > 5:
        print(f"  ... ({len(selected) - 5} more)")
    print(
        "retry policy: rounds="
        f"{max(0, args.retry_rounds)} workers="
        f"{args.retry_workers if args.retry_workers > 0 else max(1, min(8, args.workers))} "
        f"cooldown={args.retry_cooldown_sec}s"
    )
    if not args.execute:
        print("dry run only; pass --execute to launch")
        return
    if not selected:
        print("nothing to run; exit")
        return
    if args.allow_concurrent_claw:
        assign_claw_worker_slots(selected)
    bench_caps = parse_bench_caps(args.bench_cap)
    if bench_caps:
        print(
            "bench_caps="
            + ",".join(f"{bench}={cap}" for bench, cap in sorted(bench_caps.items())),
            flush=True,
        )

    secrets = load_env_secrets()
    assert_model_selection(selected, args.allow_model_mismatch, secrets)
    if not args.skip_mysql_cleanup:
        run_remote_mysql_cleanup()
    run_id = selected[0]["run_id"]
    try:
        from run_manifest import write_manifest_start
        write_manifest_start(run_id, selected)
    except Exception as exc:  # manifest must never block a run
        print(f"[manifest] WARNING: failed to write run.json: {exc}", flush=True)
    status_path = experiment_status_path(run_id)
    status_lock = threading.Lock()
    claw_lock = threading.Lock()
    task_locks = {
        (record["bench"], str(record["task_id"])): threading.Lock()
        for record in selected
    }
    stop_event = threading.Event()

    # SIGINT handler: set stop_event so queued workers bail out instead of
    # starting new subprocesses. In-flight subprocesses get the parent's
    # SIGINT propagated automatically (same process group), so they should
    # die naturally; if not, the per-trial timeout catches them eventually.
    def _on_sigint(signum: int, _frame: Any) -> None:
        if stop_event.is_set():
            # Second Ctrl-C = give up cleanly, just exit.
            print("\n[launcher] second SIGINT — exiting hard", flush=True)
            sys.exit(130)
        print("\n[launcher] SIGINT — draining (queued trials will skip; "
              "in-flight trials finish or timeout). Ctrl-C again to force exit.",
              flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, _on_sigint)

    def execute_pass(
        pass_records: list[dict[str, Any]],
        pass_workers: int,
        task_window_size: int,
        pass_label: str,
    ) -> dict[str, dict[str, Any]]:
        """Run one scheduling pass over pass_records; returns {trial_id: status}.

        The main pass uses the full worker count and (optionally) the rolling
        task window. Retry passes reuse the same machinery at lower concurrency
        with the legacy round-robin order.
        """
        completed = 0
        max_workers = max(1, pass_workers)
        active_by_bench: Counter[str] = Counter()
        active_by_task: Counter[tuple[str, str]] = Counter()
        active: dict[Future[dict[str, Any]], dict[str, Any]] = {}
        results: dict[str, dict[str, Any]] = {}
        label = f"{pass_label} " if pass_label else ""

        def bench_cap(record: dict[str, Any]) -> int:
            return bench_caps.get(str(record["bench"]), max_workers)

        def submit_record(executor: ThreadPoolExecutor, record: dict[str, Any]) -> None:
            bench = str(record["bench"])
            task_key = (bench, str(record["task_id"]))
            future = executor.submit(
                run_record,
                record,
                secrets,
                status_path,
                status_lock,
                claw_lock,
                task_locks.setdefault((record["bench"], str(record["task_id"])), threading.Lock()),
                args.allow_concurrent_claw,
                args.per_trial_timeout_sec,
                args.docker_wait_sec,
                args.docker_check_interval_sec,
                stop_event,
                args.concurrent_trials,
            )
            active[future] = record
            active_by_bench[bench] += 1
            active_by_task[task_key] += 1

        if task_window_size > 0:
            task_order, task_queues = task_window_groups(pass_records)
            pending_tasks: deque[tuple[str, str]] = deque(task_order)
            open_tasks: deque[tuple[str, str]] = deque()
            task_window = max(1, task_window_size)
            print(f"task_window={task_window}", flush=True)

            def fill_task_window() -> None:
                while pending_tasks and len(open_tasks) < task_window:
                    open_tasks.append(pending_tasks.popleft())

            def submit_ready(executor: ThreadPoolExecutor) -> None:
                if stop_event.is_set():
                    pending_tasks.clear()
                    open_tasks.clear()
                    return
                fill_task_window()
                while len(active) < max_workers:
                    submitted = False
                    if not open_tasks:
                        break
                    for _ in range(len(open_tasks)):
                        task_key = open_tasks[0]
                        queue = task_queues[task_key]
                        if not queue and active_by_task[task_key] <= 0:
                            open_tasks.popleft()
                            fill_task_window()
                            submitted = True
                            break
                        if not queue or (
                            active_by_task[task_key] > 0
                            and not (args.concurrent_trials and task_key[0] != "claw")
                        ):
                            open_tasks.rotate(-1)
                            continue
                        record = queue[0]
                        bench = str(record["bench"])
                        if active_by_bench[bench] >= bench_cap(record):
                            open_tasks.rotate(-1)
                            continue
                        queue.popleft()
                        submit_record(executor, record)
                        submitted = True
                        break
                    if not submitted:
                        break
        else:
            pending: deque[dict[str, Any]] = deque(pass_records)

            def submit_ready(executor: ThreadPoolExecutor) -> None:
                if stop_event.is_set():
                    pending.clear()
                    return
                while pending and len(active) < max_workers:
                    submitted = False
                    for _ in range(len(pending)):
                        record = pending[0]
                        bench = str(record["bench"])
                        task_key = (bench, str(record["task_id"]))
                        if active_by_bench[bench] >= bench_cap(record):
                            pending.rotate(-1)
                            continue
                        if active_by_task[task_key] > 0 and not (
                            args.concurrent_trials and bench != "claw"
                        ):
                            pending.rotate(-1)
                            continue
                        pending.popleft()
                        submit_record(executor, record)
                        submitted = True
                        break
                    if not submitted:
                        break

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            submit_ready(executor)
            while active:
                done, _ = wait(active.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    record = active.pop(future)
                    bench = str(record["bench"])
                    task_key = (bench, str(record["task_id"]))
                    active_by_bench[bench] -= 1
                    active_by_task[task_key] -= 1
                    status = future.result()
                    results[str(record["trial_id"])] = status
                    ok = status["returncode"] == 0 and not status.get("error_kind")
                    completed += 1
                    marker = "OK" if ok else (status.get("error_kind") or "FAIL").upper()
                    print(
                        f"[{marker:>7}] {label}{status['trial_id']} "
                        f"rc={status['returncode']} elapsed={status.get('elapsed_sec', '?')}s "
                        f"lock={status.get('lock_wait_sec', '?')}s "
                        f"proc={status.get('subprocess_elapsed_sec', '?')}s "
                        f"({completed}/{len(pass_records)})",
                        flush=True,
                    )
                submit_ready(executor)
        return results

    statuses = execute_pass(selected, args.workers, args.task_window, "")
    main_failures = sum(
        1 for status in statuses.values()
        if status["returncode"] != 0 or status.get("error_kind")
    )

    def retryable_records() -> list[dict[str, Any]]:
        """Failed trials worth requeueing: launcher-level failures with no
        trajectory on disk. rc=0 graded failures are real results, not holes;
        trials whose trajectory exists already produced output; 'skipped'
        means the user hit Ctrl-C — none of those are requeued."""
        retry: list[dict[str, Any]] = []
        for record in selected:
            status = statuses.get(str(record["trial_id"]))
            if status is None:
                continue
            if status["returncode"] == 0 and not status.get("error_kind"):
                continue
            if status.get("error_kind") == "skipped":
                continue
            if has_final_incremental_record(record.get("incremental_path", "")):
                continue
            retry.append(record)
        return retry

    retry_rounds = max(0, args.retry_rounds)
    retry_workers = args.retry_workers if args.retry_workers > 0 else max(1, min(8, args.workers))
    retry_rounds_used = 0
    for round_idx in range(1, retry_rounds + 1):
        if stop_event.is_set():
            break
        retry_records = retryable_records()
        if not retry_records:
            break
        retry_rounds_used += 1
        print(
            f"\n[retry] round {round_idx}/{retry_rounds}: requeueing "
            f"{len(retry_records)} failed/missing trials at workers={retry_workers} "
            f"(cooldown {args.retry_cooldown_sec}s)",
            flush=True,
        )
        if args.retry_cooldown_sec > 0:
            time.sleep(args.retry_cooldown_sec)
        statuses.update(
            execute_pass(
                interleave_records_by_task(retry_records),
                retry_workers,
                0,
                f"retry{round_idx}",
            )
        )

    final_failed = [
        trial_id
        for trial_id, status in statuses.items()
        if status["returncode"] != 0 or status.get("error_kind")
    ]
    retry_note = (
        f" (main-pass failures: {main_failures}, retry rounds used: {retry_rounds_used})"
        if retry_rounds_used
        else ""
    )
    print(
        f"\ndone: {len(statuses) - len(final_failed)} ok, {len(final_failed)} failed{retry_note}; "
        f"status={display_path(status_path)}"
    )
    try:
        from run_manifest import finalize_manifest
        finalize_manifest(run_id, statuses, interrupted=stop_event.is_set())
    except Exception as exc:
        print(f"[manifest] WARNING: failed to finalize run.json: {exc}", flush=True)
    if final_failed:
        print("still failed after retries:" if retry_rounds_used else "failed trials:")
        for trial_id in final_failed[:30]:
            status = statuses[trial_id]
            print(f"  - {trial_id} rc={status['returncode']} kind={status.get('error_kind') or 'fail'}")
        if len(final_failed) > 30:
            print(f"  ... ({len(final_failed) - 30} more)")


if __name__ == "__main__":
    main()
