"""Per-run manifest + global run index (2026-06-12 repo-management patch).

Every launch_trials --execute invocation gets, with no opt-in required:
  - <experiment_root(run_id)>/run.json          one manifest per run_id,
    written at launch (status=running) and finalized at exit
    (status=completed/interrupted + outcome counters).
  - experiments/INDEX.jsonl                      one appended line per
    finished pass, the machine-readable cross-run index.

Design notes:
  - The manifest separates the three experiment axes explicitly:
    `model` / `taskset` / `protocol`, so "same train different test" style
    queries become jq one-liners over INDEX.jsonl.
  - `validity` starts as "valid". Quarantines/supersedes are recorded with
    ops/workflows/rl_eval/run_validity.py, never by deleting artifacts.
  - Repeated launches of the same run_id (resume, retry fills) update
    run.json in place and append another INDEX line; `pass_seq` increments.
    INDEX is append-only history, run.json reflects the latest state.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import experiment_root, experiments_root, display_path, repo_path

INDEX_FILENAME = "INDEX.jsonl"

# Protocol-defining env keys, baked into plan records by the plan builders.
PROTOCOL_ENV_KEYS = (
    "UNIFIED_PROMPT_PROFILE",
    "UNIFIED_TOOLS_SCHEMA_MODE",
    "UNIFIED_DISABLE_THINKING",
    "UNIFIED_PRESENCE_PENALTY",
    "UNIFIED_EARLY_STOP_N",
    "UNIFIED_ROLLOUT_WALLCLOCK_CAP_SEC",
    "UNIFIED_VERIFIER_TIMEOUT_CAP_SEC",
    "UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS",
    "UNIFIED_HARBOR_REQUIRE_PREBUILT_LOCAL",
    "UNIFIED_CLAW_USE_DOCKER_SANDBOX",
)

SERVING_ENV_KEYS = (
    "UNIFIED_DOCKER_NETWORK_HOST",
    "UNIFIED_DOCKER_CPUSET",
    "UNIFIED_DOCKER_PIDS_LIMIT",
    "UNIFIED_DOCKER_ULIMIT_FSIZE_GB",
    "AGENT_BENCH_DOCKER_START_CONCURRENCY",
)


def _git_state() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        out["sha"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=str(repo_path("."))).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=20,
            cwd=str(repo_path("."))).stdout
        out["dirty"] = bool(dirty.strip())
    except Exception:
        out.setdefault("sha", "")
        out.setdefault("dirty", None)
    return out


def _file_sha256(path: Path, cap_bytes: int = 64 * 1024 * 1024) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            read = 0
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
                read += len(chunk)
                if read > cap_bytes:
                    break
        return h.hexdigest()[:16]
    except Exception:
        return ""


def manifest_path(run_id: str) -> Path:
    return experiment_root(run_id) / "run.json"


def index_path() -> Path:
    return experiments_root() / INDEX_FILENAME


def build_manifest(run_id: str, records: list[dict[str, Any]],
                   argv: list[str]) -> dict[str, Any]:
    first = records[0]
    env = first.get("env", {}) or {}
    benches = Counter(str(r.get("bench")) for r in records)
    tasks = {(str(r.get("bench")), str(r.get("task_id"))) for r in records}
    retrieval_jsonls = sorted({
        str(r.get("retrieval_jsonl")) for r in records if r.get("retrieval_jsonl")
    })
    skills_assets = [
        {"path": p, "sha256_16": _file_sha256(repo_path(p))}
        for p in retrieval_jsonls
    ]
    prior = {}
    mp = manifest_path(run_id)
    if mp.exists():
        try:
            prior = json.loads(mp.read_text())
        except Exception:
            prior = {}
    manifest = {
        "run_id": run_id,
        "intent": os.environ.get("RUN_INTENT", prior.get("intent", "")),
        "created_at": prior.get("created_at")
                      or datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "validity": prior.get("validity", "valid"),
        "validity_reason": prior.get("validity_reason", ""),
        "superseded_by": prior.get("superseded_by", ""),
        "pass_seq": int(prior.get("pass_seq", 0)) + 1,
        # ----- axis 1: model -----
        "model": {
            "name": first.get("model"),
            "role": first.get("model_role"),
            "api_base": env.get("OPENAI_API_BASE", ""),
        },
        # ----- axis 2: taskset -----
        "taskset": {
            "plan_records": len(records),
            "n_tasks": len(tasks),
            "by_bench": dict(sorted(benches.items())),
            "mode": first.get("mode"),
            "arm": first.get("arm"),
        },
        # ----- axis 3: protocol -----
        "protocol": {
            **{k: env[k] for k in PROTOCOL_ENV_KEYS if k in env},
            "max_turns": first.get("max_turns"),
            "max_time": first.get("max_time"),
        },
        "serving": {
            "docker_host": env.get("DOCKER_HOST", ""),
            **{k: env[k] for k in SERVING_ENV_KEYS if k in env},
        },
        "skills_assets": skills_assets,
        "baseline_run_id": os.environ.get(
            "RUN_BASELINE_ID", prior.get("baseline_run_id", "")),
        "launcher": {
            "argv": argv,
            "host": socket.gethostname(),
            "user": getpass.getuser(),
            "git": _git_state(),
        },
        "outcome": prior.get("outcome", {}),
    }
    return manifest


def write_manifest_start(run_id: str, records: list[dict[str, Any]]) -> Path:
    mp = manifest_path(run_id)
    mp.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(run_id, records, sys.argv)
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[manifest] {display_path(mp)} (pass_seq={manifest['pass_seq']})",
          flush=True)
    return mp


def finalize_manifest(run_id: str, statuses: dict[str, dict[str, Any]],
                      interrupted: bool = False) -> None:
    mp = manifest_path(run_id)
    try:
        manifest = json.loads(mp.read_text())
    except Exception:
        return
    failed = [t for t, s in statuses.items()
              if s.get("returncode") != 0 or s.get("error_kind")]
    manifest["status"] = "interrupted" if interrupted else "completed"
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["outcome"] = {
        "trials_attempted_this_pass": len(statuses),
        "launcher_ok": len(statuses) - len(failed),
        "launcher_failed": len(failed),
    }
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    # Append-only global index line.
    line = {
        "ts": manifest["updated_at"],
        "run_id": run_id,
        "root": display_path(experiment_root(run_id)),
        "status": manifest["status"],
        "validity": manifest.get("validity", "valid"),
        "pass_seq": manifest.get("pass_seq"),
        "intent": manifest.get("intent", ""),
        "model": manifest.get("model", {}).get("name"),
        "mode": manifest.get("taskset", {}).get("mode"),
        "arm": manifest.get("taskset", {}).get("arm"),
        "n_tasks": manifest.get("taskset", {}).get("n_tasks"),
        "plan_records": manifest.get("taskset", {}).get("plan_records"),
        "outcome": manifest.get("outcome", {}),
    }
    ip = index_path()
    with ip.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"[manifest] finalized; index += {display_path(ip)}", flush=True)


# ---------------------------------------------------------------------------
# Train-side manifest (2026-06-12): symmetric ledger for RL training runs.
# Called from ops/workflows/rl_training launchers right after launch_env.txt.
# ---------------------------------------------------------------------------

TRAIN_ENV_PREFIXES = ("RELAX_", "AGENT_BENCH_", "UNIFIED_", "ROLLOUT_")
TRAIN_ENV_KEYS = (
    "EXPERIMENT_ID", "EXPERIMENT_DIR", "RUN_NAME", "RUN_DIR", "CHECKPOINT_DIR",
    "LOAD_DIR", "LEARNING_RATE", "GLOBAL_BATCH_SIZE", "N_SAMPLES_PER_PROMPT",
    "MAX_CONTEXT_LEN", "MAX_PROMPT_LEN", "MAX_RESPONSE_LEN",
    "TENSOR_MODEL_PARALLEL_SIZE", "CONTEXT_PARALLEL_SIZE",
    "ACTOR_MAX_TOKENS_PER_GPU", "ROLLOUT_NUM_GPUS_PER_ENGINE",
    "EXPECTED_LATEST_CKPT", "TRAIN_PARQUET", "EVAL_PARQUET",
    "DOCKER_HOST", "SAVE_INTERVAL",
)


def _fork_git_states() -> dict[str, Any]:
    forks = {}
    for name in ("Relax", "slime", "sglang", "Megatron-LM"):
        d = repo_path(name)
        if not d.exists():
            continue
        try:
            sha = subprocess.run(["git", "-C", str(d), "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            branch = subprocess.run(["git", "-C", str(d), "rev-parse", "--abbrev-ref", "HEAD"],
                                    capture_output=True, text=True, timeout=10).stdout.strip()
            dirty = bool(subprocess.run(["git", "-C", str(d), "status", "--porcelain"],
                                        capture_output=True, text=True, timeout=30).stdout.strip())
            forks[name] = {"sha": sha, "branch": branch, "dirty": dirty}
        except Exception:
            pass
    return forks


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def _update_rl_catalog(experiment_id: str, experiment_dir: Path, segments: int) -> None:
    """Keep one small discovery index; artifacts remain owner-local."""
    path = experiments_root() / "rl" / "catalog.json"
    try:
        catalog = json.loads(path.read_text())
    except FileNotFoundError:
        catalog = {"schema_version": 1, "experiments": [], "reference_experiments": []}
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"refusing to overwrite invalid RL catalog: {path}: {exc}")
    entries = {
        item.get("experiment_id"): item
        for item in catalog.get("experiments", [])
        if isinstance(item, dict) and item.get("experiment_id")
    }
    entries[experiment_id] = {
        "experiment_id": experiment_id,
        "kind": "rl_training",
        "segments": segments,
        "path": str(experiment_dir),
    }
    catalog.update({
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "experiments": sorted(entries.values(), key=lambda item: item["experiment_id"]),
    })
    _atomic_write_json(path, catalog)


def write_train_manifest(
    experiment_id: str,
    segment_id: str,
    run_dir: str,
    experiment_dir: str,
) -> Path:
    rd = Path(run_dir)
    ed = Path(experiment_dir)
    rd.mkdir(parents=True, exist_ok=True)
    env_axes: dict[str, str] = {}
    for k, v in os.environ.items():
        if k in TRAIN_ENV_KEYS or any(k.startswith(p) for p in TRAIN_ENV_PREFIXES):
            env_axes[k] = v
    parquet = os.environ.get("TRAIN_PARQUET", "")
    pq = Path(parquet) if parquet else None
    manifest = {
        "schema_version": 1,
        "kind": "train_segment",
        "experiment_id": experiment_id,
        "segment_id": segment_id,
        "segment_dir": str(rd),
        "intent": os.environ.get("RL_RUN_PURPOSE", "") or os.environ.get("RUN_INTENT", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "launched",
        "validity": "valid",
        "validity_reason": "",
        "superseded_by": "",
        "model": {
            "resume_from_ckpt": os.environ.get("EXPECTED_LATEST_CKPT", ""),
            "resume_ckpt_root": os.environ.get("LOAD_DIR", "")
                                 or os.environ.get("LOAD_CHECKPOINT_DIR", "")
                                 or os.environ.get("CHECKPOINT_LOAD_DIR", ""),
        },
        "taskset": {
            "train_parquet": parquet,
            "train_parquet_sha256_16": _file_sha256(pq) if pq and pq.exists() else "",
            "eval_parquet": os.environ.get("EVAL_PARQUET", ""),
        },
        "protocol": env_axes,
        "resolved_config": str(rd / "resolved_config.env"),
        "launch_env_snapshot": str(rd / "launch_env.redacted.txt"),
        "wandb": {
            "api_key_present": bool(os.environ.get("WANDB_API_KEY")),
            "expected_run_name": segment_id,
        },
        "launcher": {
            "host": socket.gethostname(),
            "user": getpass.getuser(),
            "git_projects": _git_state(),
            "git_forks": _fork_git_states(),
        },
    }
    mp = rd / "run.json"
    _atomic_write_json(mp, manifest)
    lineage = {
        "experiment_id": experiment_id,
        "segment_id": segment_id,
        "resume_from_ckpt": manifest["model"]["resume_from_ckpt"],
        "resume_ckpt_root": manifest["model"]["resume_ckpt_root"],
        "created_at": manifest["created_at"],
    }
    _atomic_write_json(rd / "lineage.json", lineage)

    experiment_path = ed / "experiment.json"
    experiment = {}
    if experiment_path.exists():
        try:
            experiment = json.loads(experiment_path.read_text())
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            raise RuntimeError(f"refusing to overwrite invalid experiment manifest: {experiment_path}")
        if experiment.get("experiment_id") != experiment_id:
            raise RuntimeError(
                f"experiment id mismatch: manifest={experiment.get('experiment_id')!r} "
                f"launcher={experiment_id!r}"
            )
    segments = {
        item.get("segment_id"): item
        for item in experiment.get("segments", [])
        if isinstance(item, dict) and item.get("segment_id")
    }
    segments[segment_id] = {
        "segment_id": segment_id,
        "path": str(rd),
        "created_at": manifest["created_at"],
        "status": manifest["status"],
        "resume_from_ckpt": manifest["model"]["resume_from_ckpt"],
        "resume_ckpt_root": manifest["model"]["resume_ckpt_root"],
    }
    experiment.update({
        "schema_version": 1,
        "kind": "rl_training",
        "experiment_id": experiment_id,
        "experiment_dir": str(ed),
        "objective": os.environ.get("RL_RUN_PURPOSE", "")
                     or os.environ.get("RUN_INTENT", "")
                     or experiment.get("objective", ""),
        "created_at": experiment.get("created_at") or manifest["created_at"],
        "updated_at": manifest["created_at"],
        "status": "running",
        "segments": sorted(segments.values(), key=lambda item: (item.get("created_at", ""), item["segment_id"])),
        "inputs": {
            "train_parquet": manifest["taskset"]["train_parquet"],
            "train_parquet_sha256_16": manifest["taskset"]["train_parquet_sha256_16"],
            "eval_parquet": manifest["taskset"]["eval_parquet"],
        },
        "model": experiment.get("model", {"exports": [], "selected": {}}),
        "evals": experiment.get("evals", []),
    })
    _atomic_write_json(experiment_path, experiment)
    _update_rl_catalog(experiment_id, ed, len(experiment["segments"]))
    print(f"[manifest] experiment.json + segment run.json -> {ed} / {segment_id}", flush=True)
    if not manifest["wandb"]["api_key_present"]:
        print("[manifest] WARNING: WANDB_API_KEY is EMPTY — this run will not "
              "log curves (WANDB_API_KEY not injected?)", flush=True)
    return mp


def finalize_train_manifest(
    experiment_id: str,
    segment_id: str,
    run_dir: str,
    experiment_dir: str,
    return_code: int,
) -> None:
    rd = Path(run_dir)
    ed = Path(experiment_dir)
    manifest_path = rd / "run.json"
    experiment_path = ed / "experiment.json"
    manifest = json.loads(manifest_path.read_text())
    experiment = json.loads(experiment_path.read_text())
    if manifest.get("experiment_id") != experiment_id or manifest.get("segment_id") != segment_id:
        raise RuntimeError(f"segment identity mismatch while finalizing {manifest_path}")
    if experiment.get("experiment_id") != experiment_id:
        raise RuntimeError(f"experiment identity mismatch while finalizing {experiment_path}")
    now = datetime.now(timezone.utc).isoformat()
    status = "completed" if return_code == 0 else "failed"
    manifest.update({"status": status, "completed_at": now, "return_code": return_code})
    _atomic_write_json(manifest_path, manifest)
    for item in experiment.get("segments", []):
        if isinstance(item, dict) and item.get("segment_id") == segment_id:
            item.update({"status": status, "completed_at": now, "return_code": return_code})
    experiment.update({"status": status, "updated_at": now})
    _atomic_write_json(experiment_path, experiment)
    _update_rl_catalog(experiment_id, ed, len(experiment.get("segments", [])))
    print(f"[manifest] finalized segment={segment_id} status={status} rc={return_code}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Manifest CLI (train mode)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--experiment-id", default="")
    tr.add_argument("--experiment-dir", default="")
    tr.add_argument("--segment-id", default="")
    tr.add_argument("--run-name", default="", help="Deprecated alias for --segment-id")
    tr.add_argument("--run-dir", required=True)
    fin = sub.add_parser("train-finalize")
    fin.add_argument("--experiment-id", required=True)
    fin.add_argument("--experiment-dir", required=True)
    fin.add_argument("--segment-id", required=True)
    fin.add_argument("--run-dir", required=True)
    fin.add_argument("--return-code", required=True, type=int)
    a = ap.parse_args()
    if a.cmd == "train":
        segment_id = a.segment_id or a.run_name or os.environ.get("RUN_NAME", "")
        experiment_id = a.experiment_id or os.environ.get("EXPERIMENT_ID", "")
        if not experiment_id or not segment_id:
            ap.error("train requires --experiment-id and --segment-id")
        experiment_dir = a.experiment_dir or str(
            experiments_root() / "rl" / "runs" / experiment_id
        )
        write_train_manifest(experiment_id, segment_id, a.run_dir, experiment_dir)
    elif a.cmd == "train-finalize":
        finalize_train_manifest(
            a.experiment_id, a.segment_id, a.run_dir, a.experiment_dir, a.return_code
        )
