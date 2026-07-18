# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Claw-Eval real launcher (reuses unified_runner.run_unified_claw building blocks).

Lifecycle::

    start()
        load_task(task_id)                       → task.yaml dict
        _apply_port_offset(task_def, hash_offset) → bench mock ports unique per worker
        setup_workdir(task_id, task_def)         → host workspace
        ServiceManager(...).start_all(...)       → spin up mock HTTP services (docker mode)
        start_sandbox_container_docker_mode(...) → sandbox container on shared mock network
        return container_name

    grade(messages=...):
        construct a SimpleNamespace traj with the captured messages list
        extract_tool_dispatches(traj, task_def["tool_endpoints"])  → list of {tool_name, url, raw_command}
        score_from_components(task_def, traj, dispatches)          → (passed, score, note)
        return score        (already in [0, 1])

    teardown()
        stop_sandbox_container(cname)
        ServiceManager.stop_all() — best-effort

This is the v1.1 real launcher referenced in the original stub. Selected when
``UNIFIED_LAUNCHER_MODE=real``.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

from .base import BaseLauncher, LauncherError


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy unified_runner imports — only loaded when a ClawLauncher is constructed
# in real mode, so unit tests / mock smokes don't pull docker / yaml deps.
# ---------------------------------------------------------------------------
_EVAL_SCRIPTS = Path(
    os.environ.get("SKILLRL_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))) + "/GeneralAgent/eval_scripts"
)


def _import_unified():
    """Import the helpers we need from ``unified_runner.run_unified_claw``.

    We add the *parent* directory (eval_scripts) to ``sys.path`` so the
    package import ``unified_runner.run_unified_claw`` succeeds without
    requiring the caller's PYTHONPATH to have been set externally.
    """
    if str(_EVAL_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_EVAL_SCRIPTS))
    from unified_runner import run_unified_claw as ruc  # type: ignore
    return ruc


# Claw mock service ports currently start at 9100 and max out at 9129 in
# datasets/claw-eval/tasks. We partition them into 100-port slots so multiple
# parallel rollouts get disjoint windows. The default 512 slots keeps the
# highest shifted port under 65535 (9129 + 51100 = 60229) while making slot
# collisions rare enough for high-concurrency RL. Override downward via
# UNIFIED_CLAW_PORT_SLOTS if a future environment needs a smaller range.
_CLAW_PORT_SLOT = 100
_CLAW_PORT_SLOTS = max(
    1,
    min(int(os.environ.get("UNIFIED_CLAW_PORT_SLOTS", "512") or "512"), 512),
)


class ClawLauncher(BaseLauncher):
    """Sandbox launcher for Claw-Eval T-series tasks."""

    bench = "claw"

    def __init__(self, task_id, task_kwargs=None):
        super().__init__(task_id, task_kwargs)
        # Force-enable docker sandbox: host mode lets concurrent rollouts
        # collide on workspace paths, and Plan B's design assumes the docker
        # variant. (unified_runner.run_unified_claw guards on this env var.)
        os.environ.setdefault("UNIFIED_CLAW_USE_DOCKER_SANDBOX", "1")

        self._ruc = _import_unified()
        self._task_def: dict[str, Any] | None = None
        self._service_manager = None
        self._container_name: str | None = None
        self._workdir: Path | None = None
        self._port_offset: int = self._compute_port_offset()
        self._teardown_called = False
        self._start_attempt_idx = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_port_offset(self) -> int:
        """Deterministic ``(task_id, worker_salt) → port_offset`` mapping."""
        digest = hashlib.sha256(
            f"{self.task_id}|{self._unique_salt}".encode("utf-8")
        ).digest()
        slot_idx = int.from_bytes(digest[:2], "big") % _CLAW_PORT_SLOTS
        return slot_idx * _CLAW_PORT_SLOT

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> str:
        ruc = self._ruc
        self._start_attempt_idx += 1
        attempt_suffix = f"{self._unique_salt}-a{self._start_attempt_idx}"

        self._task_def = ruc.load_task(self.task_id)
        if self._port_offset:
            ruc._apply_port_offset(self._task_def, self._port_offset)

        self._workdir = ruc.setup_workdir(
            self.task_id,
            self._task_def,
            worker_suffix=f"-{self._unique_salt}",
        )
        log_dir = ruc.WORKDIR / self.task_id / "services"

        self._service_manager = ruc.ServiceManager(
            self._task_def.get("services", []) or [],
            ruc.TASKS_DIR / self.task_id,
            mode="docker",
        )
        svc_status = self._service_manager.start_all(log_dir)
        for s in svc_status:
            logger.info(f"[claw/{self.task_id}] {s}")

        if not self._service_manager.mock_ip:
            raise LauncherError(
                f"[claw/{self.task_id}] shared mock infra unavailable; "
                "cannot start docker sandbox."
            )

        try:
            self._container_name = ruc.start_sandbox_container_docker_mode(
                self.task_id,
                self._workdir,
                net_name=self._service_manager.net_name,
                mock_ip=self._service_manager.mock_ip,
                worker_suffix=f"-{attempt_suffix}",
            )
        except Exception as exc:
            self._safe_stop_services()
            raise LauncherError(
                f"[claw/{self.task_id}] sandbox container failed to start"
            ) from exc

        logger.info(
            f"[claw/{self.task_id}] container={self._container_name} "
            f"mock={self._service_manager.mock_cname}@{self._service_manager.mock_ip} "
            f"port_offset={self._port_offset}"
        )
        return self._container_name

    def grade(
        self,
        *,
        final_answer: str | None = None,
        container_state: bool = False,
        messages: list[dict[str, Any]] | None = None,
    ) -> float:
        if not self._task_def:
            raise LauncherError(f"[claw/{self.task_id}] grade called before start")

        ruc = self._ruc
        traj_messages = list(messages or [])

        # Build a minimal traj-like object that score_from_components knows
        # how to consume. It only reads ``.messages``, so a SimpleNamespace
        # is enough.
        traj = types.SimpleNamespace(messages=traj_messages)
        dispatches = ruc.extract_tool_dispatches(
            traj, self._task_def.get("tool_endpoints") or []
        )

        try:
            _passed, score, note = ruc.score_from_components(self._task_def, traj, dispatches)
        except Exception as exc:
            logger.exception(f"[claw/{self.task_id}] score_from_components raised")
            return 0.0

        logger.info(
            f"[claw/{self.task_id}] score={score:.3f} note={note} "
            f"dispatches={len(dispatches)} messages={len(traj_messages)}"
        )
        # Clip defensively — score_from_components computes a weighted avg
        # in [0,1] but we want to guarantee the contract.
        return max(0.0, min(1.0, float(score)))

    def teardown(self) -> None:
        if self._teardown_called:
            return
        self._teardown_called = True
        ruc = self._ruc
        if self._container_name:
            try:
                ruc.stop_sandbox_container(self._container_name)
            except Exception as exc:
                logger.warning(
                    f"[claw/{self.task_id}] stop_sandbox_container failed: {exc!r}"
                )
        self._safe_stop_services()

    def _safe_stop_services(self) -> None:
        if not self._service_manager:
            return
        try:
            self._service_manager.stop_all()
        except Exception as exc:
            logger.warning(f"[claw/{self.task_id}] service stop_all failed: {exc!r}")
