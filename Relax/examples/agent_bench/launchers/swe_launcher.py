# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real SWE-Gym / SWE-Bench-Verified launcher (v1.1).

Reuses :mod:`unified_runner.run_unified_swe` building blocks. SWE grading
needs three pieces of per-instance metadata:

* ``test_patch``       — gold test changes that introduce the FAIL_TO_PASS tests
* ``FAIL_TO_PASS``     — tests that must pass after the fix
* ``PASS_TO_PASS``     — tests that must still pass

We load these from the SWE-Gym lite parquet at first launcher construction,
cache the index globally, then look up by ``task_id`` (== instance_id) per
launcher.

Grading order (mirrors run_unified_swe.run_instance):

    git diff repo_path                        → patch
    apply_gold_test_patch(test_patch)         → install new tests
    run_tests(cname, repo_path, FAIL_TO_PASS) → pytest output
    check_test_pass(output)                   → bool → 1.0 / 0.0
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from .base import BaseLauncher, LauncherError, looks_like_infra_failure


logger = logging.getLogger(__name__)


_EVAL_SCRIPTS = Path(
    os.environ.get("SKILLRL_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))) + "/GeneralAgent/eval_scripts"
)


_SWE_INDEX: dict[str, dict] | None = None


def _import_swe():
    if str(_EVAL_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_EVAL_SCRIPTS))
    from unified_runner import run_unified_swe as rus  # type: ignore
    return rus


def _get_swe_index(rus):
    """Lazy global cache of ``instance_id → instance_dict``."""
    global _SWE_INDEX
    if _SWE_INDEX is None:
        _SWE_INDEX = rus.load_instances()
        logger.info(f"[swe] loaded {len(_SWE_INDEX)} instance metadata entries")
    return _SWE_INDEX


class SWEGymLauncher(BaseLauncher):
    """Sandbox launcher for SWE-Gym / SWE-Bench Verified tasks."""

    bench = "swe_lite"

    def __init__(self, task_id, task_kwargs=None):
        super().__init__(task_id, task_kwargs)
        self._rus = _import_swe()
        instances = _get_swe_index(self._rus)
        self._instance = instances.get(task_id)
        if not self._instance:
            raise LauncherError(
                f"SWELauncher: instance_id={task_id!r} not in SWE-Gym parquet. "
                f"Expected one of {len(instances)} ids."
            )
        self._image = self._rus.instance_id_to_image(task_id)
        self._container_name: str | None = None
        self._repo_path: str | None = None
        self._teardown_called = False
        self._start_attempt_idx = 0

    def start(self) -> str:
        rus = self._rus
        self._start_attempt_idx += 1
        container_suffix = f"{self._unique_salt}-a{self._start_attempt_idx}"
        self._container_name = rus.start_container(
            self.task_id,
            self._image,
            container_suffix=container_suffix,
        )
        self._repo_path = rus.get_repo_path(self._container_name)
        logger.info(
            f"[swe/{self.task_id}] container={self._container_name} repo={self._repo_path}"
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
            raise LauncherError(f"SWELauncher[{self.task_id}]: grade before start")
        rus = self._rus
        try:
            patch = ""
            for attempt in range(1, 4):
                patch = rus.container_exec(
                    self._container_name,
                    f"cd {self._repo_path} && git diff",
                    timeout=120,
                )
                if not looks_like_infra_failure(patch) or attempt == 3:
                    break
        except Exception as exc:
            logger.exception(f"[swe/{self.task_id}] git diff raised")
            if looks_like_infra_failure(exc):
                raise LauncherError(f"[swe/{self.task_id}] git diff infra failure: {exc!r}") from exc
            return 0.0
        if looks_like_infra_failure(patch):
            raise LauncherError(
                f"[swe/{self.task_id}] git diff infra failure: {patch[-500:]!r}"
            )
        if not patch.strip():
            logger.info(f"[swe/{self.task_id}] no patch produced → score=0")
            return 0.0

        fail_to_pass = self._instance.get("FAIL_TO_PASS", []) or []
        if not fail_to_pass:
            logger.warning(
                f"[swe/{self.task_id}] instance has no FAIL_TO_PASS tests; treat as 0."
            )
            return 0.0

        # Apply gold test patch (introduces new tests)
        try:
            test_patch_ok, test_patch_output = rus.apply_gold_test_patch(
                self._container_name,
                self._repo_path,
                self._instance.get("test_patch", "") or "",
            )
        except Exception as exc:
            logger.exception(f"[swe/{self.task_id}] apply_gold_test_patch raised")
            if looks_like_infra_failure(exc):
                raise LauncherError(
                    f"[swe/{self.task_id}] apply_gold_test_patch infra failure: {exc!r}"
                ) from exc
            return 0.0
        if looks_like_infra_failure(test_patch_output):
            raise LauncherError(
                f"[swe/{self.task_id}] apply_gold_test_patch infra failure: "
                f"{(test_patch_output or '')[-500:]!r}"
            )
        if not test_patch_ok:
            logger.info(f"[swe/{self.task_id}] apply_gold_test_patch FAILED")
            return 0.0

        try:
            test_output = rus.run_tests(self._container_name, self._repo_path, fail_to_pass)
        except Exception as exc:
            logger.exception(f"[swe/{self.task_id}] run_tests raised")
            if looks_like_infra_failure(exc):
                raise LauncherError(f"[swe/{self.task_id}] run_tests infra failure: {exc!r}") from exc
            return 0.0
        if looks_like_infra_failure(test_output):
            raise LauncherError(
                f"[swe/{self.task_id}] run_tests infra failure: {test_output[-500:]!r}"
            )

        resolved = bool(rus.check_test_pass(test_output))
        score = 1.0 if resolved else 0.0
        logger.info(
            f"[swe/{self.task_id}] resolved={resolved} score={score} "
            f"test_tail={test_output[-200:]!r}"
        )
        return score

    def teardown(self) -> None:
        if self._teardown_called:
            return
        self._teardown_called = True
        if self._container_name:
            try:
                ok = self._rus.stop_container(self._container_name)
                if ok is False:
                    logger.warning(f"[swe/{self.task_id}] stop_container reported failure for {self._container_name}")
            except Exception as exc:
                logger.warning(f"[swe/{self.task_id}] stop_container failed: {exc!r}")
