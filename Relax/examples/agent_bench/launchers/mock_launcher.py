# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Deterministic Mock launcher for end-to-end Relax smoke tests.

This launcher never touches docker, never starts a service, never grades
correctness. It returns a stable, hash-derived reward per ``task_id`` so we
can exercise the full Relax rollout + GRPO + reward pipeline before plugging
in real bench sandboxes.

Selected via ``UNIFIED_LAUNCHER_MODE=mock`` in the environment.
"""
from __future__ import annotations

import hashlib
import logging

from .base import BaseLauncher


logger = logging.getLogger(__name__)


class MockLauncher(BaseLauncher):
    """No-op launcher with a deterministic reward."""

    bench = "mock"

    def __init__(self, task_id, task_kwargs=None):
        super().__init__(task_id, task_kwargs)
        self._started = False

    def start(self) -> str:
        self._started = True
        logger.info(f"[mock] start task_id={self.task_id} → handle={self.container_name}")
        return self.container_name

    def grade(self, *, final_answer=None, container_state=False, messages=None) -> float:
        # Deterministic per-task pseudo-score in [0, 1].
        digest = hashlib.sha256(self.task_id.encode("utf-8")).digest()
        # Map the first 4 bytes to a float in [0, 1].
        rand_int = int.from_bytes(digest[:4], "big")
        score = rand_int / 0xFFFFFFFF
        logger.info(
            f"[mock] grade task_id={self.task_id} final_answer_len="
            f"{len(final_answer or '')} score={score:.4f}"
        )
        return float(score)

    def teardown(self) -> None:
        if not self._started:
            return
        logger.info(f"[mock] teardown task_id={self.task_id}")
        self._started = False
