"""Inter-process Docker concurrency gates for unified eval runners."""

from __future__ import annotations

import contextlib
import fcntl
import os
import time
from pathlib import Path
from typing import Iterator


@contextlib.contextmanager
def _docker_slot_gate(
    *,
    cap_env: str,
    default_gate_dir: str,
    gate_dir_env: str,
    poll_env: str,
    label: str = "",
) -> Iterator[None]:
    try:
        cap = int(os.environ.get(cap_env, "0") or "0")
    except ValueError:
        cap = 0
    if cap <= 0:
        yield
        return

    gate_root = Path(os.environ.get(gate_dir_env, default_gate_dir))
    gate_root.mkdir(parents=True, exist_ok=True)
    poll_sec = float(os.environ.get(poll_env, "0.2") or "0.2")
    held_file = None
    try:
        while held_file is None:
            for index in range(cap):
                slot_path = gate_root / f"slot_{index:03d}.lock"
                file_handle = slot_path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    file_handle.close()
                    continue
                file_handle.seek(0)
                file_handle.truncate()
                file_handle.write(f"pid={os.getpid()} label={label}\n")
                file_handle.flush()
                held_file = file_handle
                break
            if held_file is None:
                time.sleep(poll_sec)
        yield
    finally:
        if held_file is not None:
            try:
                fcntl.flock(held_file.fileno(), fcntl.LOCK_UN)
            finally:
                held_file.close()


@contextlib.contextmanager
def docker_start_gate(label: str = "") -> Iterator[None]:
    """Limit concurrent Docker container starts across launcher subprocesses.

    `launch_trials.py --workers N` controls full trial concurrency, not the
    short Docker start burst inside each runner. This gate lets high-throughput
    eval use many workers while keeping `docker run` fan-out bounded via
    AGENT_BENCH_DOCKER_START_CONCURRENCY.
    """
    with _docker_slot_gate(
        cap_env="AGENT_BENCH_DOCKER_START_CONCURRENCY",
        default_gate_dir="/tmp/agent_bench_docker_start_gate",
        gate_dir_env="AGENT_BENCH_DOCKER_START_GATE_DIR",
        poll_env="AGENT_BENCH_DOCKER_START_GATE_POLL_SEC",
        label=label,
    ):
        yield
