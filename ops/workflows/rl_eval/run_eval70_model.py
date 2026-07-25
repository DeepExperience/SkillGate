#!/usr/bin/env python3
"""Generic eval70 x4 wrapper for one model.

This is the reusable entrypoint for one owner-local eval row. Cross-model
tables are derived later into z_cc_terminal_imgs.

Default behavior is safe: without --execute it only builds/checks the 280-record
plan and prints what would run.  With --execute it can optionally start local
SGLang engine(s), start Docker guard tmux sessions, run launch_trials.py, and
render the 3 eval70 tables.

Canonical inputs live under ops/workflows/rl_eval/specs, datasets/rl, and
skill_libraries/snapshots/rl. Evaluation output must be explicitly owned by an
experiment under experiments/rl/runs; there is no central rl_eval result tree.

Resume behavior:
  Reuse the same --run-id and --run-root.  launch_trials.py reads the existing
  status/results under EXPERIMENT_ROOT and only fills missing/failed trials
  unless --rerun-completed is passed.

Examples:

  # Oracle top-1 self-read, local single TP4 engine, full run.
  python3 ops/workflows/rl_eval/run_eval70_model.py \
    --run-id 202606xx_eval70_my_ckpt_oracle \
    --owner-experiment my-experiment \
    --eval-id eval70-oracle-r4-example \
    --row-id final \
    --label my-ckpt \
    --model-path experiments/rl/runs/my-experiment/model/final_hf \
    --served-name qwen3.5-9b-my-ckpt \
    --tools-schema manual_schema \
    --skill-mode oracle \
    --execute --start-guards

  # Analyze existing results only.
  python3 ops/workflows/rl_eval/run_eval70_model.py \
    --run-id 202606xx_eval70_my_ckpt_oracle \
    --label my-ckpt \
    --run-root experiments/rl/runs/my-experiment/eval/eval70-oracle-r4-example/rows/final \
    --served-name qwen3.5-9b-my-ckpt \
    --tables-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path("/path/to/skillRL")
DEFAULT_TASK_LIST = ROOT / "ops/workflows/rl_eval/specs/eval70_v1/tasks.tsv"
DEFAULT_ORACLE_SNAPSHOT = ROOT / "skill_libraries/snapshots/rl/eval70_oracle_selfread_20260612"
DEFAULT_SOURCE_PLAN = ROOT / "ops/workflows/rl_eval/specs/eval70_v1/source_plan_retrieval.jsonl"
DEFAULT_TRAIN_PARQUET = ROOT / "datasets/rl/parquet_4bench_base_20260523/train.parquet"
DEFAULT_EVAL_PARQUET = ROOT / "datasets/rl/parquet_4bench_base_20260523/eval.parquet"
DEFAULT_DOCKER_HOST = "unix:///tmp/local-docker-overlay2.sock"


def default_bench_caps() -> list[str]:
    raw = os.environ.get("EVAL70_BENCH_CAP", "claw=6")
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class EngineSpec:
    gpus: str
    port: int


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def rel(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def abs_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def safe_name(value: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-_.") or "eval70"
    if len(text) <= max_len:
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{text[: max_len - len(digest) - 1]}-{digest}"


def run(
    cmd: str,
    *,
    check: bool = True,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    print(f"[cmd] {cmd}", flush=True)
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        timeout=timeout,
        env=env,
    )
    if capture and proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {cmd}")
    return proc


def parse_engine(raw: str) -> EngineSpec:
    if ":" not in raw:
        raise argparse.ArgumentTypeError("engine must be GPUS:PORT, e.g. 0,1,2,3:30000")
    gpus, raw_port = raw.rsplit(":", 1)
    if not gpus:
        raise argparse.ArgumentTypeError("engine GPUS cannot be empty")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid engine port: {raw_port}") from exc
    return EngineSpec(gpus=gpus, port=port)


def determine_top_n(args: argparse.Namespace) -> int:
    if args.retrieval_top_n > 0:
        return args.retrieval_top_n
    if args.skill_mode == "oracle":
        return 1
    if args.skill_mode == "retrieve":
        return 10
    if args.skill_mode == "mixed":
        return 16
    return 0


def determine_retrieval_root(args: argparse.Namespace) -> str:
    if args.skill_mode == "oracle":
        return str(abs_path(args.retrieval_root or DEFAULT_ORACLE_SNAPSHOT))
    if args.skill_mode in ("retrieve", "mixed"):
        if not args.retrieval_root:
            raise SystemExit(f"--retrieval-root is required for --skill-mode {args.skill_mode}")
        return str(abs_path(args.retrieval_root))
    return ""


def determine_api_base(args: argparse.Namespace) -> str:
    if args.api_base:
        return args.api_base.rstrip("/")
    if args.serve_mode == "local":
        if args.use_router or len(args.engine) > 1 or args.router_worker_url:
            return f"http://127.0.0.1:{args.router_port}/v1"
        return f"http://127.0.0.1:{args.engine[0].port}/v1"
    return "http://127.0.0.1:30000/v1"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def patch_plan_env(args: argparse.Namespace, plan: Path, api_base: str) -> None:
    rows = load_jsonl(plan)
    if not rows:
        raise RuntimeError(f"empty plan: {plan}")
    run_root_rel = rel(args.run_root)

    def under_run_root(path_value: str, anchor: str) -> str:
        """Rewrite generated artifact paths to the explicit per-run root."""
        if not path_value:
            return path_value
        parts = Path(str(path_value)).parts
        if anchor not in parts:
            return path_value
        idx = parts.index(anchor)
        return str(Path(run_root_rel).joinpath(*parts[idx:]))

    for rec in rows:
        for key in ("result_dir", "incremental_path", "trajectory_path"):
            if key in rec:
                rec[key] = under_run_root(str(rec.get(key) or ""), "results")
        if "log_path" in rec:
            rec["log_path"] = under_run_root(str(rec.get("log_path") or ""), "logs")

        env = rec.setdefault("env", {})
        env["UNIFIED_TOOLS_SCHEMA_MODE"] = args.tools_schema
        env["UNIFIED_OPENCLAW_PROFILE"] = args.prompt_profile
        env["UNIFIED_PROMPT_PROFILE"] = args.prompt_profile
        env["UNIFIED_SKILL_SELECTION_INSTRUCTION"] = args.skill_selection_instruction
        env["AGENT_BENCH_DOCKER_START_CONCURRENCY"] = str(args.docker_start_cap)
        env["DOCKER_START_CAP"] = str(args.docker_start_cap)
        env["DOCKER_HOST"] = args.docker_host
        env["OPENAI_API_BASE"] = api_base
        env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "dummy")
        env["UNIFIED_RUN_ID"] = args.run_id
        # Align the runner's output dir (<UNIFIED_EXP_VERSION>_<arm>, see base.py) with the
        # per-trial trajectory_path that make-plan baked into this record. The old hard-set to
        # args.run_id made the runner write <run_id>_<arm> while launch_trials.py:581 looked for
        # the per-trial path -> every rc=0 trial was misflagged MISSING_TRAJECTORY and retried
        # (~3x compute + duplicate rows that inflate collect()'s denominator and corrupt T1/T2).
        _arm = rec.get("arm") or ("retrieval" if args.skill_mode in ("retrieve", "oracle", "mixed") else "baseline")
        _expdir = Path(rec.get("trajectory_path", "")).parent.parent.name
        env["UNIFIED_EXP_VERSION"] = (
            _expdir[: -(len(_arm) + 1)] if _expdir.endswith("_" + _arm) else args.run_id
        )
        env["UNIFIED_MODEL"] = args.served_name
        env["PHASE_B_MODEL"] = args.served_name
        env["EXPERIMENT_ROOT"] = run_root_rel
        env["RELAX_RL_RUN_ID"] = args.run_id
        env["AGENT_BENCH_DOCKER_LIFECYCLE_DIR"] = rel(args.run_root / "docker_lifecycle")
        env["UNIFIED_DOCKER_NETWORK_HOST"] = "1" if args.host_network else "0"
        env["UNIFIED_DOCKER_PIDS_LIMIT"] = str(args.pids_limit)
        env["UNIFIED_DOCKER_ULIMIT_FSIZE_GB"] = str(args.fsize_gb)
        env["UNIFIED_DOCKER_CPUSET"] = args.cpuset
        env["UNIFIED_TOOL_TIMEOUT_CHILD_CLEANUP"] = "0"
        env["UNIFIED_CONTAINER_PROXY"] = args.container_proxy
        env["NO_PROXY"] = args.no_proxy
        env["no_proxy"] = args.no_proxy
        if args.require_prebuilt:
            env["UNIFIED_HARBOR_REQUIRE_PREBUILT_LOCAL"] = "1"
            env["UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS"] = "1"
        argv = rec.get("argv")
        if isinstance(argv, list):
            for idx, token in enumerate(argv):
                if token == "--api-base" and idx + 1 < len(argv):
                    argv[idx + 1] = api_base
            rec["command_preview"] = " ".join(str(part) for part in argv)
    write_jsonl(plan, rows)


def build_plan(args: argparse.Namespace, api_base: str) -> Path:
    args.run_root.mkdir(parents=True, exist_ok=True)
    plan = args.run_root / "plans" / f"{args.run_id}.jsonl"
    plan.parent.mkdir(parents=True, exist_ok=True)
    date = args.date or datetime.now().strftime("%Y%m%d")
    top_n = determine_top_n(args)

    if args.skill_mode == "source-plan":
        source_plan = abs_path(args.source_plan or DEFAULT_SOURCE_PLAN)
        run(
            "python3 ops/workflows/rl_eval/make_eval70_replay_plan.py "
            f"--source-plan {q(source_plan)} "
            f"--run-id {q(args.run_id)} "
            f"--run-root {q(rel(args.run_root))} "
            f"--date {q(date)} "
            f"--model {q(args.served_name)} "
            f"--api-base {q(api_base)} "
            f"--docker-host {q(args.docker_host)} "
            f"--docker-start-cap {args.docker_start_cap} "
            f"--repeats {args.repeats} "
            f"--out {q(rel(plan))}"
        )
    else:
        arm = "baseline" if args.skill_mode == "noskill" else "retrieval"
        retrieval_root = determine_retrieval_root(args)
        retrieval_args = ""
        if retrieval_root:
            retrieval_args = f"--retrieval-root {q(retrieval_root)} --retrieval-top-n {top_n}"
        run(
            "python3 ops/workflows/rl_eval/oracle_skill_pipeline.py make-plan "
            f"--run-id {q(args.run_id)} "
            f"--date {q(date)} "
            f"--model {q(args.served_name)} "
            f"--mode {q(args.mode or ('eval70_' + args.skill_mode))} "
            f"--arm {q(arm)} "
            f"--trials {args.repeats} "
            f"--task-list {q(abs_path(args.task_list))} "
            f"--train-parquet {q(abs_path(args.train_parquet))} "
            f"--eval-parquet {q(abs_path(args.eval_parquet))} "
            f"--max-turns {args.max_turns} "
            f"--max-time {args.max_time} "
            f"--docker-host {q(args.docker_host)} "
            f"--api-base {q(api_base)} "
            f"{retrieval_args} "
            f"--out {q(rel(plan))}"
        )
    patch_plan_env(args, plan, api_base)
    rows = load_jsonl(plan)
    expected = args.expected_records
    if expected and len(rows) != expected:
        raise RuntimeError(f"expected {expected} plan rows, got {len(rows)}: {plan}")
    if not args.allow_missing_retrieval:
        uncovered = [r for r in rows if r.get("arm") == "retrieval" and not r.get("retrieval_covered", True)]
        if uncovered:
            sample = [(r.get("bench"), r.get("task_id")) for r in uncovered[:8]]
            raise RuntimeError(f"{len(uncovered)} retrieval records are uncovered; sample={sample}")
    summary = {
        "run_id": args.run_id,
        "run_root": rel(args.run_root),
        "skill_mode": args.skill_mode,
        "served_name": args.served_name,
        "tools_schema": args.tools_schema,
        "api_base": api_base,
        "docker_host": args.docker_host,
        "records": len(rows),
        "tasks": len({(r.get("bench"), str(r.get("task_id"))) for r in rows}),
        "plan": rel(plan),
    }
    (args.run_root / "run_eval70_model.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[plan] {rel(plan)} records={len(rows)} tasks={summary['tasks']}", flush=True)
    return plan


def docker_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["DOCKER_HOST"] = args.docker_host
    return env


def preflight(args: argparse.Namespace, api_base: str, *, need_endpoint: bool) -> None:
    if args.serve_mode == "local":
        if not args.model_path:
            raise SystemExit("--model-path is required when --serve-mode local")
        if not (abs_path(args.model_path) / "config.json").exists():
            raise FileNotFoundError(f"model config missing: {abs_path(args.model_path) / 'config.json'}")
    ok, out = docker_cmd(args, "docker images -q | wc -l", check=False)
    if not ok:
        raise RuntimeError(f"cannot query docker images through {args.docker_host}: {out[-400:]}")
    image_count = int((out.strip().splitlines() or ["0"])[-1] or 0)
    if image_count < args.min_images:
        raise RuntimeError(
            f"local dockerd has only {image_count} images (<{args.min_images}); restore image cache first"
        )
    if args.require_subreaper:
        sub = run("pgrep -f subreaper_exec | head -1", check=False)
        if sub.returncode != 0 or not sub.stdout.strip():
            raise RuntimeError("subreaper_exec not found; restore/start local dockerd with subreaper first")
    if need_endpoint:
        wait_endpoint(api_base, args.served_name, timeout_sec=args.endpoint_wait_sec)
    print(f"[preflight] docker images={image_count}; api_base={api_base}", flush=True)


def docker_cmd(args: argparse.Namespace, cmd: str, *, check: bool = True) -> tuple[bool, str]:
    # The worker login shell can reset DOCKER_HOST from shell init files when
    # subprocesses are launched through `bash -lc`, so set it again in-command.
    proc = run(f"export DOCKER_HOST={q(args.docker_host)}; {cmd}", check=False, env=docker_env(args))
    if check and proc.returncode != 0:
        raise RuntimeError(f"docker command failed: {cmd}")
    return proc.returncode == 0, proc.stdout or ""


def endpoint_models(api_base: str) -> str:
    return run(f"curl -s --max-time 5 {q(api_base.rstrip('/') + '/models')} || true", check=False).stdout or ""


def wait_endpoint(api_base: str, served_name: str, *, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        text = endpoint_models(api_base)
        if served_name in text:
            print(f"[ready] endpoint serves {served_name}: {api_base}", flush=True)
            return
        time.sleep(10)
    raise TimeoutError(f"endpoint did not serve {served_name}: {api_base}")


def start_tmux(session: str, body: str) -> None:
    run(f"tmux kill-session -t {q(session)} 2>/dev/null || true", check=False)
    run(f"tmux new-session -d -s {q(session)} {q('bash -lc ' + q(body))}")


def kill_tmux(session: str) -> None:
    run(f"tmux kill-session -t {q(session)} 2>/dev/null || true", check=False)


def start_local_serving(args: argparse.Namespace) -> None:
    args.run_root.joinpath("logs", "sglang").mkdir(parents=True, exist_ok=True)
    model_path = abs_path(args.model_path)
    prefix = safe_name(args.run_id)
    for idx, engine in enumerate(args.engine):
        session = f"eval70-{prefix}-sglang-{idx}"
        log = args.run_root / "logs" / "sglang" / f"engine_{idx}_{engine.port}.log"
        body = (
            f"cd {q(ROOT)}; "
            f"exec env CUDA_VISIBLE_DEVICES={q(engine.gpus)} "
            f"MODEL_PATH={q(model_path)} SERVED_NAME={q(args.served_name)} "
            f"PORT={engine.port} TP_SIZE={args.tp_size} "
            f"CONTEXT_LENGTH={args.context_length} MEM_FRACTION={args.mem_fraction} "
            f"RANDOM_SEED={args.seed} "
            "bash ops/launch/run_qwen35_sglang_server.sh "
            f"> {q(log)} 2>&1"
        )
        start_tmux(session, body)
    for engine in args.engine:
        wait_endpoint(f"http://127.0.0.1:{engine.port}/v1", args.served_name, timeout_sec=args.endpoint_wait_sec)

    if args.use_router or len(args.engine) > 1 or args.router_worker_url:
        worker_urls = [f"http://127.0.0.1:{engine.port}" for engine in args.engine]
        worker_urls.extend(args.router_worker_url)
        session = f"eval70-{prefix}-router"
        log = args.run_root / "logs" / "sglang" / "router.log"
        body = (
            "source /path/to/conda/etc/profile.d/conda.sh; "
            "conda activate slime; "
            f"cd {q(ROOT)}; "
            f"export NO_PROXY={q(args.no_proxy)} no_proxy={q(args.no_proxy)}; "
            "exec python -m sglang_router.launch_router "
            f"--host 0.0.0.0 --port {args.router_port} "
            + " ".join(["--worker-urls", *map(q, worker_urls)])
            + f" --policy {q(args.router_policy)} "
            + f"--prometheus-port {args.router_prometheus_port} > {q(log)} 2>&1"
        )
        start_tmux(session, body)
        wait_endpoint(f"http://127.0.0.1:{args.router_port}/v1", args.served_name, timeout_sec=args.endpoint_wait_sec)


def cleanup_local_serving(args: argparse.Namespace) -> None:
    prefix = safe_name(args.run_id)
    for idx, _engine in enumerate(args.engine):
        kill_tmux(f"eval70-{prefix}-sglang-{idx}")
    kill_tmux(f"eval70-{prefix}-router")


def start_guards(args: argparse.Namespace) -> list[str]:
    cleanup_dir = args.run_root / "cleanup"
    cleanup_dir.mkdir(parents=True, exist_ok=True)
    prefix = safe_name(args.run_id)
    sessions: list[str] = []

    def add(name: str, body: str) -> None:
        session = f"eval70-{prefix}-{name}"
        start_tmux(session, body)
        sessions.append(session)

    add(
        "dmesg",
        "dmesg -w -T 2>/dev/null | "
        "grep --line-buffered -iE 'unregister_netdevice|waiting for .* to become free|kobject_uevent' "
        f">> {q(cleanup_dir / 'dmesg_unregister_netdevice.log')}",
    )
    # A containerd shim appears before its container is visible as running.
    # At high Docker-start concurrency, the generic orphan reaper can race that
    # window and kill a valid startup. Keep it available for quiescent repair,
    # but require launchers to opt in explicitly.
    if args.shim_reaper:
        add(
            "shimreaper",
            f"cd {q(ROOT)}; export DOCKER_HOST={q(args.docker_host)}; "
            "REAP_INTERVAL_SEC=120 python3 ops/cleanup/reap_orphan_shims.py "
            f">> {q(cleanup_dir / 'shim_reaper.log')} 2>&1",
        )
    add(
        "diskreap",
        f"cd {q(ROOT)}; export DOCKER_HOST={q(args.docker_host)}; "
        f"DISK_REAP_PATH={q(args.disk_reap_path)} "
        f"DISK_REAP_WATERMARK_GB={args.disk_watermark_gb} "
        "DISK_REAP_INTERVAL_SEC=5 python3 ops/cleanup/reap_disk_bombs.py "
        f">> {q(cleanup_dir / 'disk_reaper.log')} 2>&1",
    )
    add(
        "stale",
        f"cd {q(ROOT)}; export DOCKER_HOST={q(args.docker_host)} RELAX_RL_RUN_ID={q(args.run_id)}; "
        "python3 ops/cleanup/watch_rl_stale_containers.py "
        f"--run-id {q(args.run_id)} --loop --interval-sec 120 --max-remove 32 "
        "--max-running-remove 4 --remove-running-after-sec 3600 --remove-dead-owner-running "
        f">> {q(cleanup_dir / 'rl_stale_cleaner.log')} 2>&1",
    )
    print(f"[guards] started {len(sessions)} sessions: {', '.join(sessions)}", flush=True)
    return sessions


def stop_guards(sessions: list[str]) -> None:
    for session in sessions:
        kill_tmux(session)


@contextmanager
def managed_runtime(args: argparse.Namespace):
    guard_sessions: list[str] = []
    serving_started = False
    try:
        if args.start_guards:
            guard_sessions = start_guards(args)
        if args.serve_mode == "local":
            start_local_serving(args)
            serving_started = True
        yield
    finally:
        if serving_started and not args.keep_server:
            cleanup_local_serving(args)
        if guard_sessions and not args.keep_guards:
            stop_guards(guard_sessions)


def run_eval(args: argparse.Namespace, plan: Path, api_base: str) -> None:
    (args.run_root / "logs").mkdir(parents=True, exist_ok=True)
    (args.run_root / "reports").mkdir(parents=True, exist_ok=True)
    log = args.run_root / "logs" / "eval.log"
    env_exports = {
        "PATH": "/path/to/conda/envs/slime/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "EXPERIMENT_ROOT": rel(args.run_root),
        "RUN_INTENT": args.intent,
        "DOCKER_HOST": args.docker_host,
        "DOCKER_START_CAP": str(args.docker_start_cap),
        "AGENT_BENCH_DOCKER_START_CONCURRENCY": str(args.docker_start_cap),
        "UNIFIED_DOCKER_NETWORK_HOST": "1" if args.host_network else "0",
        "UNIFIED_DOCKER_PIDS_LIMIT": str(args.pids_limit),
        "UNIFIED_DOCKER_ULIMIT_FSIZE_GB": str(args.fsize_gb),
        "UNIFIED_DOCKER_CPUSET": args.cpuset,
        "UNIFIED_TOOL_TIMEOUT_CHILD_CLEANUP": "0",
        "UNIFIED_CONTAINER_PROXY": args.container_proxy,
        "NO_PROXY": args.no_proxy,
        "no_proxy": args.no_proxy,
        "RELAX_RL_RUN_ID": args.run_id,
        "AGENT_BENCH_DOCKER_LIFECYCLE_DIR": rel(args.run_root / "docker_lifecycle"),
    }
    if args.require_prebuilt:
        env_exports["UNIFIED_HARBOR_REQUIRE_PREBUILT_LOCAL"] = "1"
        env_exports["UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS"] = "1"
    export_lines = "\n".join(f"export {key}={q(value)}" for key, value in env_exports.items())
    unset_line = (
        "unset AGENT_BENCH_DOCKER_EXEC_CONCURRENCY AGENT_BENCH_DOCKER_TEARDOWN_CONCURRENCY "
        "AGENT_BENCH_DOCKER_HOSTS UNIFIED_DOCKER_NPROC_LIMIT UNIFIED_DOCKER_MEMORY_LIMIT "
        "UNIFIED_DOCKER_RM_TIMEOUT_SEC 2>/dev/null || true"
    )
    extra_args = []
    if args.rerun_completed:
        extra_args.append("--rerun-completed")
    if args.allow_missing_retrieval:
        extra_args.append("--allow-missing-retrieval")
    if args.concurrent_trials:
        extra_args.append("--concurrent-trials")
    for cap in args.bench_cap:
        extra_args.append(f"--bench-cap {q(cap)}")
    if args.strict_model:
        model_mismatch_arg = ""
    else:
        model_mismatch_arg = "--allow-model-mismatch"
    body = f"""
cd {q(ROOT)}
set -o pipefail
{export_lines}
{unset_line}
python3 GeneralAgent/sft_data_collection/launch_trials.py \\
  --plan {q(rel(plan))} \\
  --model {q(args.served_name)} \\
  --workers {args.workers} \\
  --per-trial-timeout-sec {args.per_trial_timeout_sec} \\
  --docker-wait-sec {args.docker_wait_sec} \\
  --api-base-override {q(api_base)} \\
  {model_mismatch_arg} \\
  --allow-concurrent-claw \\
  --skip-mysql-cleanup \\
  --retry-rounds {args.retry_rounds} \\
  --retry-workers {args.retry_workers} \\
  --retry-cooldown-sec {args.retry_cooldown_sec} \\
  {' '.join(extra_args)} \\
  --execute 2>&1 | tee {q(rel(log))}
rc=${{PIPESTATUS[0]}}
echo "[eval-exit rc=$rc] $(date -u +%FT%TZ)" | tee -a {q(rel(log))}
exit "$rc"
"""
    run(body, check=True, timeout=args.eval_timeout_sec, capture=False)


def render_tables(args: argparse.Namespace) -> None:
    (args.run_root / "reports").mkdir(parents=True, exist_ok=True)
    rr = rel(args.run_root)
    analysis_log = args.run_root / "reports" / "analysis_3tables.log"
    table_md = args.run_root / "reports" / "zcc_3tables.md"
    table_context = {
        "noskill": "no skill",
        "retrieve": "retrieval",
        "oracle": "oracle skill",
        "mixed": "mixed skills",
        "source-plan": "source-plan replay",
    }.get(args.skill_mode, args.skill_mode)
    run(
        f"python3 ops/workflows/rl_eval/analyze_eval70_3tables.py {q(args.label + '=' + rr)} "
        f"| tee {q(rel(analysis_log))}",
        check=False,
    )
    run(
        f"EVAL70_TABLE_CONTEXT={q(table_context)} "
        f"python3 ops/workflows/rl_eval/format_eval70_zcc.py {q(args.label + '=' + rr)} "
        f"| tee {q(rel(table_md))}",
        check=False,
    )
    run(
        f"EXPERIMENT_ROOT={q(rr)} python3 GeneralAgent/sft_data_collection/data_quality_dashboard.py "
        f"{q(args.run_id)} --run-root {q(rr)} "
        f"2>&1 | tee {q(rel(args.run_root / 'reports' / 'data_quality_dashboard.log'))}",
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--owner-experiment", default="")
    parser.add_argument("--eval-id", default="")
    parser.add_argument("--row-id", default="")
    parser.add_argument("--label", default="", help="Row label used in generated z_cc tables")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--intent", default="", help="Stored in run.json via RUN_INTENT")
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--skill-mode", choices=["oracle", "retrieve", "noskill", "mixed", "source-plan"], default="oracle")
    parser.add_argument("--source-plan", default="")
    parser.add_argument("--retrieval-root", default="")
    parser.add_argument("--retrieval-top-n", type=int, default=0, help="0 = oracle:1, retrieve:10")
    parser.add_argument("--task-list", default=str(DEFAULT_TASK_LIST))
    parser.add_argument("--train-parquet", default=str(DEFAULT_TRAIN_PARQUET))
    parser.add_argument("--eval-parquet", default=str(DEFAULT_EVAL_PARQUET))
    parser.add_argument("--mode", default="")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--expected-records", type=int, default=280)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-time", type=int, default=850)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--served-name", default="", help="Served model id; required unless --tables-only")
    parser.add_argument("--tools-schema", choices=["manual_schema", "openai_tools"], default="manual_schema")
    parser.add_argument("--prompt-profile", default="openclaw_full")
    parser.add_argument(
        "--skill-selection-instruction",
        default="",
        help="Optional instruction appended after the available-skills block. Empty preserves the existing prompt.",
    )
    parser.add_argument("--serve-mode", choices=["local", "existing"], default="local")
    parser.add_argument("--engine", type=parse_engine, action="append", default=None,
                        help="Local engine as GPUS:PORT. Repeat for multiple local engines. "
                             "Default: 0,1,2,3:30000")
    parser.add_argument("--use-router", action="store_true")
    parser.add_argument("--router-port", type=int, default=30100)
    parser.add_argument("--router-prometheus-port", type=int, default=39100,
                        help="Prometheus metrics port for sglang_router; keep distinct from SGLang engine metrics ports.")
    parser.add_argument("--router-policy", choices=["random", "round_robin", "cache_aware", "power_of_two", "manual"],
                        default="cache_aware")
    parser.add_argument("--router-worker-url", action="append", default=[])
    parser.add_argument("--api-base", default="")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=65536)
    parser.add_argument("--mem-fraction", default="0.88")
    parser.add_argument("--seed", default="1063810697")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--docker-host", default=DEFAULT_DOCKER_HOST)
    parser.add_argument("--docker-start-cap", type=int, default=128)
    parser.add_argument("--min-images", type=int, default=500)
    parser.add_argument("--container-proxy", default="http://<proxy-host>:3128")
    parser.add_argument("--no-proxy", default="127.0.0.1,localhost,0.0.0.0")
    parser.add_argument("--cpuset", default="24-179")
    parser.add_argument("--pids-limit", type=int, default=1024)
    parser.add_argument("--fsize-gb", type=int, default=32)
    parser.add_argument("--host-network", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-prebuilt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-subreaper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-guards", action="store_true")
    parser.add_argument("--shim-reaper", action=argparse.BooleanOptionalAction, default=False,
                        help="Opt in to the generic orphan-shim guard with --start-guards. "
                             "It is off by default because it can race valid container startup.")
    parser.add_argument("--keep-guards", action="store_true")
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--disk-reap-path", default="/data/cache")
    parser.add_argument("--disk-watermark-gb", type=int, default=300)
    parser.add_argument("--per-trial-timeout-sec", type=int, default=2700)
    parser.add_argument("--docker-wait-sec", type=int, default=54000)
    parser.add_argument("--retry-rounds", type=int, default=2)
    parser.add_argument("--retry-workers", type=int, default=8)
    parser.add_argument("--retry-cooldown-sec", type=int, default=60)
    parser.add_argument("--eval-timeout-sec", type=int, default=5 * 3600)
    parser.add_argument("--endpoint-wait-sec", type=int, default=1800)
    parser.add_argument("--allow-missing-retrieval", action="store_true")
    parser.add_argument("--concurrent-trials", action="store_true",
                        help="Pass through to launch_trials.py: run repeats of the same "
                             "non-claw task concurrently so --workers actually reaches 128. "
                             "Off by default (legacy 1-trial-per-task behavior unchanged).")
    parser.add_argument("--bench-cap", action="append", default=default_bench_caps(),
                        help="Pass through to launch_trials.py. Default comes from "
                             "EVAL70_BENCH_CAP, or claw=6 to avoid exhausting local "
                             "Docker bridge address pools during concurrent Claw eval. "
                             "Set EVAL70_BENCH_CAP= to disable the default.")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--strict-model", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()

    args.label = args.label or args.run_id
    if args.engine is None:
        args.engine = [EngineSpec("0,1,2,3", 30000)]
    if args.run_root is None:
        if not (args.owner_experiment and args.eval_id and args.row_id):
            raise SystemExit(
                "evaluation ownership is required: pass --run-root or all of "
                "--owner-experiment/--eval-id/--row-id"
            )
        args.run_root = (
            ROOT / "experiments/rl/runs" / args.owner_experiment
            / "eval" / args.eval_id / "rows" / args.row_id
        )
    if not args.run_root.is_absolute():
        args.run_root = ROOT / args.run_root
    # Plan construction imports the shared experiment path resolver in a child
    # process. Anchor that child to this explicit row root instead of relying on
    # ambient owner/eval variables from the parent workflow.
    os.environ["EXPERIMENT_ROOT"] = rel(args.run_root)
    if args.owner_experiment or args.eval_id or args.row_id:
        if not (args.owner_experiment and args.eval_id and args.row_id):
            raise SystemExit(
                "evaluation ownership must provide all of "
                "--owner-experiment/--eval-id/--row-id"
            )
        os.environ["OWNER_EXPERIMENT_ID"] = args.owner_experiment
        os.environ["EVAL_ID"] = args.eval_id
        os.environ["EVAL_ROW_ID"] = args.row_id
    if not args.tables_only and not args.served_name:
        raise SystemExit("--served-name is required unless --tables-only")
    api_base = determine_api_base(args)

    if args.tables_only:
        render_tables(args)
        return

    plan = build_plan(args, api_base)
    if args.plan_only or not args.execute:
        print(f"[dry-run] plan ready at {rel(plan)}", flush=True)
        print("[dry-run] pass --execute to start serving/eval", flush=True)
        return

    # Docker preflight does not need the endpoint yet if this wrapper will start it.
    preflight(args, api_base, need_endpoint=args.serve_mode == "existing")
    with managed_runtime(args):
        if args.serve_mode == "local":
            wait_endpoint(api_base, args.served_name, timeout_sec=args.endpoint_wait_sec)
        run_eval(args, plan, api_base)
    render_tables(args)
    print(f"[ALL-DONE] {rel(args.run_root)}", flush=True)


if __name__ == "__main__":
    main()
