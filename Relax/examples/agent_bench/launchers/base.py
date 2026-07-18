# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Base class for bench-specific sandbox launchers.

The interface intentionally mirrors a deterministic, single-task lifecycle:

    launcher = HarborLauncher(task_id, task_kwargs)
    container_handle = launcher.start()        # spin up sandbox + services
    score = launcher.grade(                    # compute final reward in [0, 1]
        final_answer="...",                    # optional last assistant text
        container_state=False,                 # True ⇒ inspect container, not text
    )
    launcher.teardown()                        # remove container, free ports

Subclasses MUST be safe to instantiate in parallel: callers (Ray rollout
workers) will run ``rollout_batch_size × n_samples_per_prompt`` launchers in
parallel against the same dockerd host. Anything stateful (container names,
mock-service ports) must derive from ``task_id`` *and* a worker-unique salt
(``self._unique_salt``).
"""
from __future__ import annotations

import hashlib
import itertools
import os
import threading
import uuid
from typing import Any  # noqa: F401  (re-exported via grade() signature)


class LauncherError(RuntimeError):
    """Sandbox lifecycle / grading failure that should be surfaced to Relax."""


_INFRA_FAILURE_MARKERS = (
    "docker.example.com",
    "docker system dial-stdio",
    "Connection timed out during banner exchange",
    "Session open refused by peer",
    "failed to dial gRPC",
    "error during connect",
    "failed to resolve source metadata",
    "i/o timeout",
    "Command timed out",
    "No test.sh found",
    "No such image",
    "pull access denied",
    "connection refused",
)


def looks_like_infra_failure(value: object) -> bool:
    text = str(value)
    return any(marker in text for marker in _INFRA_FAILURE_MARKERS)


_INSTANCE_COUNTER = itertools.count()
_INSTANCE_COUNTER_LOCK = threading.Lock()


class BaseLauncher:
    """Common scaffolding for per-bench launchers."""

    bench: str = "unknown"

    def __init__(self, task_id: str, task_kwargs: dict[str, Any] | None = None) -> None:
        self.task_id = task_id
        self.task_kwargs = dict(task_kwargs or {})
        # Combined salt: PID + per-instance counter + task hash + random nonce.
        # Relax may run multiple samples for the same task concurrently inside
        # one RolloutManager process, so PID+task_id is not sufficient.
        pid = os.getpid()
        with _INSTANCE_COUNTER_LOCK:
            instance_idx = next(_INSTANCE_COUNTER)
        digest = hashlib.sha256(self.task_id.encode("utf-8")).digest()[:4].hex()
        nonce = uuid.uuid4().hex[:8]
        self._unique_salt = f"p{pid}-i{instance_idx}-{digest}-{nonce}"

    # ------------------------------------------------------------------
    # Lifecycle (subclasses override)
    # ------------------------------------------------------------------
    def start(self) -> str:
        """Spin up the sandbox and return a container/handle name.

        Returns
        -------
        str
            Container name (or any opaque identifier) the calling
            :class:`env_agent_bench.AgentBenchEnv` will hand to its
            :class:`tool_layer.ToolLayer` instance.
        """
        raise NotImplementedError

    def grade(
        self,
        *,
        final_answer: str | None = None,
        container_state: bool = False,
        messages: list[dict[str, Any]] | None = None,
    ) -> float:
        """Compute the task reward in ``[0, 1]``.

        Parameters
        ----------
        final_answer:
            Text the agent emitted on its terminal turn (the turn that
            produced no tool_calls). May be ``None`` if the agent ran out
            of turns.
        container_state:
            When ``True``, the grader should inspect the container's
            filesystem / running services rather than ``final_answer``
            (the SkillsBench / TB2 / Claw env_snapshot pattern).
        messages:
            Full trajectory captured by :class:`env_agent_bench.AgentBenchEnv`,
            in OpenAI ``messages`` shape. Used by Claw's declarative grader to
            extract tool dispatches and verify scoring components. Optional
            for benches that grade purely on container state (SkillsBench/TB2/SWE).
        """
        raise NotImplementedError

    def teardown(self) -> None:
        """Best-effort cleanup. Must not raise — only log on failure."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def container_name(self) -> str:
        """Convention used by every concrete launcher."""
        return f"agentbench-{self.bench}-{self.task_id}-{self._unique_salt}"
