"""Docker lifecycle helpers shared by unified RL/eval runners.

The RL rollout path creates many short-lived benchmark containers.  These
helpers make each container traceable and record lifecycle failures without
turning every transient Docker hiccup into an immediate training abort.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any


_LABEL_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _clean_label_value(value: object, *, default: str = "unknown", max_len: int = 180) -> str:
    text = str(value if value not in {None, ""} else default)
    text = _LABEL_SAFE_RE.sub("_", text).strip("_.-")
    if not text:
        text = default
    return text[:max_len]


def current_run_id() -> str:
    for key in ("RELAX_RL_RUN_ID", "RELAX_RUN_ID", "RUN_NAME", "WANDB_NAME"):
        value = os.environ.get(key)
        if value:
            return _clean_label_value(value)
    save_root = os.environ.get("RELAX_SAVE_ROOT") or os.environ.get("SAVE_DIR")
    if save_root:
        return _clean_label_value(Path(save_root).name)
    return "unknown"


def current_rollout_id() -> str:
    return _clean_label_value(os.environ.get("RELAX_ROLLOUT_ID"), default="unknown")


def lifecycle_dir() -> Path:
    root = os.environ.get("AGENT_BENCH_DOCKER_LIFECYCLE_DIR", "/tmp/agent_bench_docker_lifecycle")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_lifecycle_event(event: str, *, container: str, **fields: Any) -> None:
    payload = {
        "ts": time.time(),
        "event": event,
        "container": container,
        "run_id": current_run_id(),
        "rollout_id": current_rollout_id(),
        "owner_pid": os.getpid(),
        **fields,
    }
    try:
        path = lifecycle_dir() / f"docker_lifecycle_{os.getpid()}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        # Lifecycle logging must never break a task path.
        pass


def docker_label_args(
    *,
    bench: str,
    task_name: str,
    container_name: str,
    dataset_tag: str = "",
    container_suffix: str = "",
) -> list[str]:
    labels = {
        "relax.managed": "true",
        "relax.run_id": current_run_id(),
        "relax.rollout_id": current_rollout_id(),
        "relax.owner_pid": str(os.getpid()),
        "relax.bench": _clean_label_value(bench),
        "relax.dataset": _clean_label_value(dataset_tag or bench),
        "relax.task": _clean_label_value(task_name),
        "relax.container": _clean_label_value(container_name),
        "relax.container_suffix": _clean_label_value(container_suffix),
    }
    args: list[str] = []
    for key, value in labels.items():
        args.extend(["--label", f"{key}={value}"])
    return args
