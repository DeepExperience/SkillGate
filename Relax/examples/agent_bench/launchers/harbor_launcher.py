# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real Harbor-format launcher (SkillsBench / TB 2.0 / SETA).

v1.1 implementation
===================

Reuses :mod:`unified_runner.run_unified_harbor` building blocks:

* ``resolve_image(task_dir, task_name, dataset_tag)`` — build or pull
* ``start_container(image_tag, task_name, dataset_tag)`` — bring up sandbox
* ``copy_tests(task_dir, cname)``                       — drop tests/ in
* ``_read_verifier_timeout(task_dir)``                  — per-task timeout
* ``run_verifier(cname, timeout_sec)``                  — exit 0/reward output
* ``stop_container(cname)``                             — best-effort rm

Grading: ``run_verifier`` already returns a ``reward ∈ [0, 1]`` extracted
from ``/logs/verifier/reward.txt``. We just clip and return.

Bench → dataset directory + dataset_tag mapping (matches build_splits / runner):

    sb_ns       → datasets/skillsbench/tasks                  | tag "skillsbench-no-skills"
    tb2         → datasets/terminal-bench-v2                  | tag "tb2"
    seta_synth  → datasets/seta/dataset/synth_data_harbor     | tag "seta-synth"
    seta        → datasets/seta/dataset/seta_baseline_30      | tag "seta"
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from .base import BaseLauncher, LauncherError, looks_like_infra_failure


logger = logging.getLogger(__name__)


# Repo root: env override first, else derived from this file's location
# (Relax/examples/agent_bench/launchers/ -> repo root is 5 levels up).
_SKILLRL_ROOT = os.environ.get(
    "SKILLRL_ROOT", str(Path(__file__).resolve().parents[4])
)

_EVAL_SCRIPTS = Path(
    _SKILLRL_ROOT + "/GeneralAgent/eval_scripts"
)


def _import_harbor():
    if str(_EVAL_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_EVAL_SCRIPTS))
    from unified_runner import run_unified_harbor as ruh  # type: ignore
    return ruh


# bench_label → (dataset_dir, dataset_tag).
# dataset_tag is what start_container uses to name the container and what
# unified_runner uses internally (drives mirror inject + image build path).
_BENCH_INFO = {
    "sb_ns": (
        _SKILLRL_ROOT + "/datasets/skillsbench/tasks",
        "skillsbench-no-skills",
    ),
    "tb2": (
        _SKILLRL_ROOT + "/datasets/terminal-bench-v2",
        "tb2",
    ),
    "seta_synth": (
        _SKILLRL_ROOT + "/datasets/seta/dataset/seta_synth_top300",
        "seta-synth",
    ),
    "seta": (
        _SKILLRL_ROOT + "/datasets/seta/dataset/seta_baseline_30",
        "seta",
    ),
}


class HarborLauncher(BaseLauncher):
    """Sandbox launcher for SkillsBench / TB2 / SETA tasks (Harbor format)."""

    bench = "harbor"

    def __init__(self, task_id, task_kwargs=None):
        super().__init__(task_id, task_kwargs)
        sub_bench = (task_kwargs or {}).get("bench") or "tb2"
        if sub_bench not in _BENCH_INFO:
            raise LauncherError(
                f"HarborLauncher: unknown sub_bench={sub_bench!r}; "
                f"expected one of {list(_BENCH_INFO)}"
            )
        self.sub_bench = sub_bench
        self.dataset_dir = Path(_BENCH_INFO[sub_bench][0])
        self.dataset_tag = _BENCH_INFO[sub_bench][1]
        self._ruh = _import_harbor()
        self._task_dir = self.dataset_dir / task_id
        if not self._task_dir.is_dir():
            raise LauncherError(
                f"HarborLauncher: task dir not found {self._task_dir} "
                f"(bench={sub_bench}, task_id={task_id})"
            )
        self._image_tag: str | None = None
        self._container_name: str | None = None
        self._teardown_called = False
        self._start_attempt_idx = 0

    def start(self) -> str:
        ruh = self._ruh
        t0 = time.time()
        self._start_attempt_idx += 1
        container_suffix = f"{self._unique_salt}-a{self._start_attempt_idx}"
        self._image_tag = ruh.resolve_image(
            self._task_dir, self.task_id, self.dataset_tag
        )
        logger.info(
            f"[harbor/{self.sub_bench}/{self.task_id}] image={self._image_tag} "
            f"(resolve {time.time()-t0:.1f}s)"
        )
        t1 = time.time()
        self._container_name = ruh.start_container(
            self._image_tag,
            self.task_id,
            dataset_tag=self.dataset_tag,
            container_suffix=container_suffix,
        )
        logger.info(
            f"[harbor/{self.sub_bench}/{self.task_id}] container={self._container_name} "
            f"(start {time.time()-t1:.1f}s)"
        )
        # Drop tests/ in so run_verifier can find it later
        ok = ruh.copy_tests(self._task_dir, self._container_name)
        if not ok:
            raise LauncherError(
                f"[harbor/{self.sub_bench}/{self.task_id}] copy_tests returned False"
            )
        return self._container_name

    def grade(
        self,
        *,
        final_answer=None,
        container_state=False,
        messages=None,
    ) -> float:
        if not self._container_name:
            raise LauncherError(
                f"HarborLauncher[{self.task_id}]: grade called before start"
            )
        v_base = self._ruh._read_verifier_timeout(self._task_dir, default=600)
        timeout_sec = int(v_base * 1.2)
        try:
            reward, output, verifier_ok = self._ruh.run_verifier(
                self._container_name, timeout_sec=timeout_sec
            )
        except Exception as exc:
            logger.exception(
                f"[harbor/{self.sub_bench}/{self.task_id}] run_verifier raised"
            )
            if looks_like_infra_failure(exc):
                raise LauncherError(
                    f"[harbor/{self.sub_bench}/{self.task_id}] verifier infra failure: {exc!r}"
                ) from exc
            return 0.0

        if looks_like_infra_failure(output) and "test_sh_done rc=" not in (output or ""):
            raise LauncherError(
                f"[harbor/{self.sub_bench}/{self.task_id}] verifier infra failure: "
                f"{(output or '')[-500:]!r}"
            )

        if not verifier_ok:
            raise LauncherError(
                f"[harbor/{self.sub_bench}/{self.task_id}] verifier did NOT finish "
                f"(timeout={timeout_sec}s): {(output or '')[-500:]!r}"
            )

        logger.info(
            f"[harbor/{self.sub_bench}/{self.task_id}] reward={reward:.3f} "
            f"output_tail={(output or '')[-200:]!r}"
        )
        try:
            return max(0.0, min(1.0, float(reward)))
        except (TypeError, ValueError):
            return 0.0

    def teardown(self) -> None:
        if self._teardown_called:
            return
        self._teardown_called = True
        if self._container_name:
            try:
                ok = self._ruh.stop_container(self._container_name)
                if ok is False:
                    logger.warning(
                        f"[harbor/{self.sub_bench}/{self.task_id}] stop_container reported failure "
                        f"for {self._container_name}"
                    )
            except Exception as exc:
                logger.warning(
                    f"[harbor/{self.sub_bench}/{self.task_id}] stop_container failed: {exc!r}"
                )
