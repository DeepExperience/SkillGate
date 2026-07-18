# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Relax :class:`BaseInteractionEnv` for the 5-bench agent campaign.

Per-turn protocol::

    response_text  →  step()
    parse <tool_call><function=NAME>...</function></tool_call> XML
        ↓
    for each call:
        result = ToolLayer.dispatch(name, arguments)
    ↓
    pack tool_responses into the next user-role message and return
        ({"role": "user", "obs_str": ...}, done=False, info={...})

When the model emits **no** ``<tool_call>``, the turn is treated as the
agent's final answer — the launcher is invoked to grade it (or to inspect
container state) and ``done=True`` is returned with the score in ``info``.

The reward is stored on ``sample.metadata["final_score"]`` so
:func:`reward_agent_bench.reward_func` can read it without re-running the
grader.

Launcher selection
==================

* ``UNIFIED_LAUNCHER_MODE=mock``     → :class:`MockLauncher` (default during smoke)
* ``UNIFIED_LAUNCHER_MODE=real``     → per-bench real launcher (v1.1)
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)
_TERMINAL_GRADE_INFRA_FAIL_TASKS: set[str] = set()
_TERMINAL_GRADE_INFRA_FAIL_TASKS_LOCK = threading.Lock()


def _remember_terminal_grade_infra_failure(task_id: str, reason: str) -> None:
    """Record verifier/setup infra failures for this rollout worker process.

    Some tasks have verifier-side infrastructure failures that are independent
    of the agent trajectory. Once one sample for a task hits that path, close()
    should not spend another verifier timeout on sibling samples for the same
    task in the same rollout batch.
    """
    with _TERMINAL_GRADE_INFRA_FAIL_TASKS_LOCK:
        _TERMINAL_GRADE_INFRA_FAIL_TASKS.add(task_id)
    logger.warning(
        f"[{task_id}] recorded terminal grade infra failure for task-level "
        f"close() skip: {reason}"
    )


def _known_terminal_grade_infra_failure(task_id: str) -> bool:
    with _TERMINAL_GRADE_INFRA_FAIL_TASKS_LOCK:
        return task_id in _TERMINAL_GRADE_INFRA_FAIL_TASKS


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _positive_float_env(name: str, default: float) -> float:
    try:
        return max(0.001, float(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _sample_status_value(sample: Any) -> str:
    status = getattr(sample, "status", None)
    return str(getattr(status, "value", status) or "").lower()


def _mark_sample_aborted(
    sample: Any,
    reason: str,
    *,
    category: str = "verifier_infra",
    error: str | None = None,
) -> None:
    status_cls = getattr(type(sample), "Status", None)
    aborted = getattr(status_cls, "ABORTED", None)
    if aborted is not None:
        sample.status = aborted
    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict):
        metadata["rollout_stop_reason"] = reason
        metadata["rollout_abort_category"] = category
        metadata["rollout_infra_failure"] = True
        abort_info = metadata.setdefault("abort_info", {})
        abort_info["reason"] = reason
        abort_info["category"] = category
        if error:
            abort_info["error"] = error


_DOCKER_START_CONCURRENCY = _positive_int_env(
    "AGENT_BENCH_DOCKER_START_CONCURRENCY", 4
)
_DOCKER_START_SEMAPHORE = threading.BoundedSemaphore(_DOCKER_START_CONCURRENCY)
_PROMPT_SKILL_RE = re.compile(r"/root/\.claude/skills/([^/<>\n]+)/SKILL\.md")


@contextmanager
def _docker_setup_slot(task_id: str):
    """Limit concurrent Docker setup calls against the configured Docker daemon.

    Rollout generation can fan out dozens of envs at once. Letting all of them
    call `docker build/run/cp` simultaneously can overload Docker setup and
    cause false zero rewards. This semaphore only caps
    container setup/skill injection; agent inference still runs concurrently.
    """
    logger.debug(
        "[%s] waiting for docker setup slot (cap=%s)",
        task_id,
        _DOCKER_START_CONCURRENCY,
    )
    _DOCKER_START_SEMAPHORE.acquire()
    try:
        yield
    finally:
        _DOCKER_START_SEMAPHORE.release()


def _docker_run_for_skill_injection(cmd: list[str], timeout: int = 120) -> tuple[str, str, int]:
    """Small adapter matching unified_runner.retrieval_skill_inject's contract."""
    env = dict(os.environ)
    env.setdefault("DOCKER_HOST", "unix:///tmp/local-docker-overlay2.sock")
    try:
        result = subprocess.run(
            cmd,
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return stdout, f"timeout after {timeout}s: {stderr}", 124


def _resolve_retrieval_skill_mapping(task_id: str, retrieval_skills: list[Any]) -> tuple[dict[str, list[str]], list[str]]:
    """Resolve parquet skill names into absolute skill-library directories.

    Relax stores ``retrieval_skills_top_n`` as skill names. The canonical
    injector expects absolute paths, so this converts names to
    ``skill_libraries/merged/<name>`` while also accepting already-absolute
    paths for future data versions.
    """
    from unified_runner.retrieval_skill_inject import SKILL_LIB_ROOT  # type: ignore

    resolved: list[str] = []
    missing: list[str] = []
    for raw_skill in retrieval_skills:
        name = str(raw_skill).strip()
        if not name:
            continue
        candidates: list[Path] = []
        raw_path = Path(name)
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            if "/" in name:
                candidates.append(_PROJECT_ROOT / name)
            # Opt-in extra skill roots (colon-separated absolute dirs), e.g. a
            # per-task oracle-skill tree. Tried BEFORE the merged library so an
            # explicitly-configured root wins name collisions (e.g. a task_id
            # that happens to match a merged skill name). No effect when the
            # env var is unset, so the canonical resolution path is unchanged.
            for _extra_root in os.environ.get("AGENT_BENCH_EXTRA_SKILL_ROOTS", "").split(":"):
                _extra_root = _extra_root.strip()
                if _extra_root:
                    candidates.append(Path(_extra_root) / name)
            candidates.append(SKILL_LIB_ROOT / name)
            if not name.startswith("hw-"):
                candidates.append(SKILL_LIB_ROOT / f"hw-{name}")
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
            if slug and slug != name:
                candidates.append(SKILL_LIB_ROOT / slug)
                if not slug.startswith("hw-"):
                    candidates.append(SKILL_LIB_ROOT / f"hw-{slug}")
                if slug.endswith("-skills"):
                    candidates.append(SKILL_LIB_ROOT / slug[: -len("-skills")])

        skill_dir = None
        for candidate in candidates:
            if candidate.is_file() and candidate.name == "SKILL.md":
                candidate = candidate.parent
            if candidate.is_dir() and (candidate / "SKILL.md").exists():
                skill_dir = candidate
                break
        if skill_dir is None:
            missing.append(name)
            continue
        resolved.append(str(skill_dir))
    return {task_id: resolved}, missing


def _extract_prompt_skill_names(prompt: Any) -> list[str]:
    """Return skill names from the exact OpenClaw `<location>` entries."""
    if isinstance(prompt, (list, tuple)):
        parts = []
        for message in prompt:
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
            elif isinstance(message, str):
                parts.append(message)
        text = "\n".join(parts)
    else:
        text = "" if prompt is None else str(prompt)
    names: list[str] = []
    seen: set[str] = set()
    for match in _PROMPT_SKILL_RE.finditer(text):
        name = match.group(1)
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


# ---------------------------------------------------------------------------
# Cross-project imports (BaseInteractionEnv from deepeyes; ToolLayer + XML
# parser from unified_runner). Performed lazily inside the class to avoid
# pulling heavy deps when callers only need the module for introspection.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[3])).resolve()
_UNIFIED_RUNNER_PARENT = _PROJECT_ROOT / "GeneralAgent" / "eval_scripts"
if str(_UNIFIED_RUNNER_PARENT) not in sys.path:
    sys.path.insert(0, str(_UNIFIED_RUNNER_PARENT))

# These are intentionally module-level so a static checker can find them but
# their import is still cheap (BaseInteractionEnv ≈ 30 LoC, no torch).
from examples.deepeyes.base_env import BaseInteractionEnv  # type: ignore  # noqa: E402

from .launchers.base import BaseLauncher, LauncherError  # noqa: E402
from .launchers.mock_launcher import MockLauncher  # noqa: E402


# ---------------------------------------------------------------------------
# Launcher routing
# ---------------------------------------------------------------------------
def _resolve_launcher(bench: str) -> type[BaseLauncher]:
    """Pick the right launcher class for a bench."""
    mode = os.environ.get("UNIFIED_LAUNCHER_MODE", "mock").lower()
    if mode == "mock":
        return MockLauncher
    if mode != "real":
        raise LauncherError(
            f"unknown UNIFIED_LAUNCHER_MODE={mode!r}; expected 'mock' or 'real'"
        )
    # v1.1: real launchers
    if bench in ("sb_ns", "tb2", "seta", "seta_synth"):
        from .launchers.harbor_launcher import HarborLauncher
        return HarborLauncher
    if bench == "swe_lite":
        from .launchers.swe_launcher import SWEGymLauncher
        return SWEGymLauncher
    if bench == "claw":
        from .launchers.claw_launcher import ClawLauncher
        return ClawLauncher
    raise LauncherError(f"unknown bench={bench!r}; no launcher registered")


# ---------------------------------------------------------------------------
# Tool-call XML parser
# ---------------------------------------------------------------------------
def _parse_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse ``<tool_call><function=...>...</function></tool_call>`` blocks.

    We import the canonical implementation from ``unified_runner.agent_loop``
    so XML parsing stays in lock-step with what SFT inference uses.
    """
    if not content:
        return []
    try:
        from unified_runner.agent_loop import UnifiedAgentLoop  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Cannot import unified_runner.agent_loop. Ensure "
            "GeneralAgent/eval_scripts is on PYTHONPATH."
        ) from exc
    return UnifiedAgentLoop._parse_tool_calls_from_content(content)


# ---------------------------------------------------------------------------
# Tool dispatch + observation formatting
# ---------------------------------------------------------------------------
def _format_tool_response(tool_call: dict[str, Any], result: Any) -> str:
    """Render a single tool call's result as the body of a ``user`` message.

    Mirrors the convention used by ``unified_runner.agent_loop`` so the
    response shape is identical to what SFT data captured.
    """
    name = tool_call.get("function", {}).get("name", "?")
    body = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    # Truncate over-long tool outputs to keep context manageable. SFT data uses
    # the same heuristic at ~16k chars.
    cap = int(os.environ.get("UNIFIED_TOOL_RESPONSE_CAP", "16384"))
    if len(body) > cap:
        body = body[:cap] + f"\n... [truncated, was {len(body)} chars]"
    return f"<tool_response name={name!r}>\n{body}\n</tool_response>"


# ---------------------------------------------------------------------------
# The interaction environment Relax sees
# ---------------------------------------------------------------------------
class AgentBenchEnv(BaseInteractionEnv):
    """Drive a single agent task through a sandbox + tool layer."""

    def __init__(self, sample: Any, args: Any) -> None:
        # ``sample`` is a relax.utils.types.Sample; ``args`` is the parsed
        # Relax CLI namespace. We do not inherit BaseInteractionEnv.__init__
        # (it has no args) but we keep a matching protocol.
        extra = sample.metadata.get("extra_info") or {}
        if not extra:
            # Some Relax versions hoist the dict to sample.metadata directly
            extra = sample.metadata
        self.task_id: str = extra["task_id"]
        self.bench: str = extra["bench"]
        self.task_kwargs: dict[str, Any] = dict(extra.get("task_kwargs") or {})
        self.task_kwargs.setdefault("bench", self.bench)
        self.task_kwargs.setdefault("task_id", self.task_id)
        prompt_skill_names = _extract_prompt_skill_names(getattr(sample, "prompt", ""))
        metadata_skills_raw = extra.get("retrieval_skills_top_n")
        metadata_skills = [] if metadata_skills_raw is None else list(metadata_skills_raw)
        # Prefer the exact skill names from the prompt because those paths are
        # what the model will read. Metadata can contain pre-canonical names
        # such as `sed-awk-stream-editing`, while the prompt correctly exposes
        # `/root/.claude/skills/hw-sed-awk-stream-editing/SKILL.md`.
        self.retrieval_skills = prompt_skill_names or metadata_skills

        launcher_cls = _resolve_launcher(self.bench)
        self.launcher: BaseLauncher = launcher_cls(self.task_id, self.task_kwargs)

        # Set by ``reset()``; lazily-built so __init__ stays side-effect-free.
        self.tool_layer = None  # type: ignore[assignment]
        self.container_handle: str | None = None
        self.turn = 0
        self.max_turns = int(getattr(args, "max_turns", 30) or 30)
        self._retrieval_skills_injected = 0

        # Reward is stored here at end-of-episode so reward_func can read it
        # without re-running the grader.
        self._sample = sample
        self._final_score: float | None = None
        self._skip_close_terminal_grade_reason: str | None = None

        # Trajectory captured for graders that need it (Claw declarative scoring).
        # Shape mirrors OpenAI chat completions: assistant turns may have
        # ``tool_calls``; tool responses go in via ``role=tool`` messages.
        self._messages_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # BaseInteractionEnv hooks
    # ------------------------------------------------------------------
    def reset(self):
        self.turn = 0
        # User directive: Docker congestion is infrastructure noise, not model
        # failure. Retry the full setup pipeline under a process-local
        # semaphore so a rollout batch does not stampede Docker setup.
        setup_ok = False
        last_exc = None
        max_setup_attempts = _positive_int_env("AGENT_BENCH_SETUP_ATTEMPTS", 3)
        setup_deadline = time.time() + _positive_float_env("AGENT_BENCH_SETUP_TOTAL_TIMEOUT_SEC", 600.0)
        for setup_attempt in range(1, max_setup_attempts + 1):
            if time.time() >= setup_deadline:
                logger.warning(
                    f"[{self.task_id}] env setup total timeout reached before attempt "
                    f"{setup_attempt}/{max_setup_attempts}"
                )
                break
            self.container_handle = None
            self.tool_layer = None
            try:
                with _docker_setup_slot(self.task_id):
                    self.container_handle = self.launcher.start()
                    # ToolLayer init can inspect/cp into the container, so keep
                    # it under the same setup cap.
                    self.tool_layer = self._build_tool_layer(self.container_handle)
                    self._inject_retrieval_skills_if_needed()
                setup_ok = True
                break
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"[{self.task_id}] env setup attempt {setup_attempt}/{max_setup_attempts} failed "
                    f"(docker cap={_DOCKER_START_CONCURRENCY}): {exc!r}; tearing down + retrying"
                )
                if self.container_handle is not None:
                    try:
                        self.launcher.teardown()
                    except Exception:
                        pass
                self.tool_layer = None
                self.container_handle = None
                sleep_sec = min(60.0, 5.0 * setup_attempt, max(0.0, setup_deadline - time.time()))
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

        if not setup_ok:
            # User directive: don't pollute reward with infra failures.
            # We DO NOT stash score=0 here. Caller (rollout.py) detects the
            # skipped=True flag and sets sample.status=ABORTED so this sample
            # is filtered out of GRPO advantage / reward accounting.
            logger.warning(
                f"[{self.task_id}] env.reset gave up after {max_setup_attempts} full attempts: "
                f"{last_exc!r}; sample will be ABORTED (not counted in reward)"
            )
            self._skip_close_terminal_grade_reason = "reset_infra_failure"
            # Leave self._final_score as None so reward_func sees missing_score
            # and the rollout level marks ABORTED.
            return {"role": "user", "obs_str": "", "task_id": self.task_id}, {
                "task_id": self.task_id,
                "skipped": True,
                "error": "env_setup_failed",
                "error_detail": repr(last_exc),
                "abort_category": "setup_infra",
                "setup_attempts": max_setup_attempts,
                "setup_timeout_sec": _positive_float_env(
                    "AGENT_BENCH_SETUP_TOTAL_TIMEOUT_SEC", 600.0
                ),
            }

        observation = {"role": "user", "obs_str": "", "task_id": self.task_id}
        reset_info = {
            "task_id": self.task_id,
            "bench": self.bench,
            "retrieval_skills_injected": self._retrieval_skills_injected,
        }
        return observation, reset_info

    def step(self, response_text: str):
        """Execute one turn of the agent's tool-call interaction."""
        self.turn += 1
        if not response_text:
            response_text = ""
        tool_calls = _parse_tool_calls(response_text)

        if not tool_calls:
            # Terminal turn — agent gave a final answer (or said nothing).
            # Record this assistant turn (no tool_calls) so the grader still
            # sees the final text in its trajectory.
            self._messages_log.append({"role": "assistant", "content": response_text})
            try:
                self._final_score = float(
                    self.launcher.grade(
                        final_answer=response_text,
                        container_state=False,
                        messages=self._messages_log,
                    )
                )
            except LauncherError as exc:
                logger.warning(
                    f"[{self.task_id}] grader infrastructure failure on final_answer path: {exc!r}; "
                    "sample will be ABORTED"
                )
                self._skip_close_terminal_grade_reason = "final_answer_grade_infra_failure"
                info = {
                    "turn": self.turn,
                    "skipped": True,
                    "error": f"grader_infra_exception: {type(exc).__name__}",
                    "error_detail": repr(exc),
                    "abort_category": "verifier_infra",
                    "final_answer_len": len(response_text),
                }
                return {}, True, info
            except Exception as exc:
                logger.exception(f"[{self.task_id}] grader raised on final_answer path")
                self._final_score = 0.0
                info = {
                    "turn": self.turn,
                    "score": 0.0,
                    "error": f"grader_exception: {type(exc).__name__}",
                    "final_answer_len": len(response_text),
                }
                self._stash_score_on_sample()
                return {}, True, info
            self._stash_score_on_sample()
            return (
                {},
                True,
                {
                    "turn": self.turn,
                    "score": self._final_score,
                    "final_answer_len": len(response_text),
                },
            )

        # Record the assistant turn (with tool_calls) so the grader can
        # later count which tools were invoked (Claw scoring requirement).
        self._messages_log.append(
            {"role": "assistant", "content": response_text, "tool_calls": tool_calls}
        )

        # Dispatch each tool call in order. ToolLayer.dispatch is synchronous
        # to keep the contract with Relax's rollout loop simple.
        observations: list[str] = []
        for tc in tool_calls:
            name = tc.get("function", {}).get("name", "")
            arguments_raw = tc.get("function", {}).get("arguments", "{}")
            try:
                arguments = (
                    json.loads(arguments_raw) if isinstance(arguments_raw, str) else arguments_raw
                )
            except json.JSONDecodeError:
                arguments = {"_raw": arguments_raw}
            try:
                result = self.tool_layer.dispatch(name, arguments)  # type: ignore[union-attr]
            except Exception as exc:
                result = f"ERROR: tool dispatch failed: {type(exc).__name__}: {exc}"
            observations.append(_format_tool_response(tc, result))
            # Also log the tool response into the messages trail.
            self._messages_log.append(
                {
                    "role": "tool",
                    "name": name,
                    "tool_call_id": tc.get("id", ""),
                    "content": result if isinstance(result, str) else json.dumps(result, ensure_ascii=False),
                }
            )

        obs_str = "\n\n".join(observations)
        done = self.turn >= self.max_turns
        info: dict[str, Any] = {"turn": self.turn, "tool_calls": len(tool_calls)}

        if done:
            # Out of turn budget — grade based on container state (or fall
            # back to text=None if launcher only supports container_state).
            try:
                self._final_score = float(
                    self.launcher.grade(
                        final_answer=None,
                        container_state=True,
                        messages=self._messages_log,
                    )
                )
            except LauncherError as exc:
                _remember_terminal_grade_infra_failure(
                    self.task_id, "max_turns_grade_infra_failure"
                )
                logger.warning(
                    f"[{self.task_id}] grader infrastructure failure on max_turns path: {exc!r}; "
                    "sample will be ABORTED"
                )
                self._skip_close_terminal_grade_reason = "max_turns_grade_infra_failure"
                info["skipped"] = True
                info["error"] = f"grader_infra_exception: {type(exc).__name__}"
                info["error_detail"] = repr(exc)
                info["abort_category"] = "verifier_infra"
                return {}, True, info
            except Exception as exc:
                logger.exception(f"[{self.task_id}] grader raised on max_turns path")
                self._final_score = 0.0
                info["error"] = f"grader_exception: {type(exc).__name__}"
            info["score"] = self._final_score
            info["max_turns_reached"] = True
            self._stash_score_on_sample()

        return {"role": "user", "obs_str": obs_str}, done, info

    def close(self):
        # If the rollout aborted before reaching a terminal turn (e.g. SGLang
        # finish_reason='length'/'abort', or response budget exhausted), we
        # haven't graded yet. Run the grader once here against whatever
        # messages we collected — better than letting reward_func default to
        # 'missing_score' + 0.0.
        status_value = _sample_status_value(self._sample)
        skip_close_grading_for_all_aborts = os.environ.get(
            "AGENT_BENCH_SKIP_CLOSE_GRADING_ON_ABORT", "0"
        ).lower() in {"1", "true", "yes"}
        skip_close_grading_after_known_infra_failure = (
            self._skip_close_terminal_grade_reason is not None
        )
        skip_close_grading_after_task_infra_failure = (
            os.environ.get("AGENT_BENCH_SKIP_CLOSE_GRADING_ON_KNOWN_INFRA_TASK", "1")
            .lower()
            in {"1", "true", "yes"}
            and _known_terminal_grade_infra_failure(self.task_id)
        )
        should_skip_close_grading = skip_close_grading_after_known_infra_failure or (
            skip_close_grading_after_task_infra_failure
            and status_value in {"aborted", "truncated", ""}
        ) or (
            skip_close_grading_for_all_aborts and status_value in {"aborted", "truncated"}
        )
        if (
            self._final_score is None
            and self.launcher is not None
            and should_skip_close_grading
        ):
            reason = (
                self._skip_close_terminal_grade_reason
                or (
                    f"known_task_terminal_grade_infra_failure; sample_status={status_value}"
                    if skip_close_grading_after_task_infra_failure
                    else f"sample_status={status_value}"
                )
            )
            logger.warning(
                f"[{self.task_id}] close() skip terminal grading after {reason}; "
                "avoids re-running verifier after a known terminal grading/setup infrastructure failure"
            )
            if skip_close_grading_after_known_infra_failure or skip_close_grading_after_task_infra_failure:
                _mark_sample_aborted(
                    self._sample,
                    reason,
                    category="verifier_infra"
                    if "grade" in reason or "verifier" in reason
                    else "setup_infra",
                )
        elif self._final_score is None and self.launcher is not None:
            try:
                self._final_score = float(
                    self.launcher.grade(
                        final_answer=None,
                        container_state=True,
                        messages=self._messages_log,
                    )
                )
                self._stash_score_on_sample()
            except LauncherError as exc:  # pragma: no cover
                logger.warning(
                    f"[{self.task_id}] terminal grade in close() hit infrastructure failure: {exc!r}; "
                    "marking sample ABORTED so infra/verifier timeout does not become reward=0"
                )
                _remember_terminal_grade_infra_failure(
                    self.task_id, "terminal_grade_infra_failure"
                )
                _mark_sample_aborted(
                    self._sample,
                    "terminal_grade_infra_failure",
                    category="verifier_infra",
                    error=repr(exc),
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    f"[{self.task_id}] terminal grade in close() failed: {exc!r}"
                )
                # Best-effort: still stash 0.0 so reward_func knows we tried.
                self._final_score = 0.0
                self._stash_score_on_sample()
        try:
            if self.tool_layer is not None:
                self.tool_layer.close()
        except Exception as exc:  # pragma: no cover
            logger.warning(f"[{self.task_id}] tool layer close failed: {exc!r}")
        finally:
            self.tool_layer = None
        try:
            if self.launcher is not None:
                self.launcher.teardown()
        except Exception as exc:  # pragma: no cover
            logger.warning(f"[{self.task_id}] teardown failed: {exc!r}")
        # Even if teardown fails, surface whatever score we computed so the
        # caller can decide. (Relax rollout.py treats env.close() as best-effort.)
        return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _inject_retrieval_skills_if_needed(self) -> None:
        """Copy retrieval skills into the sandbox before the first agent turn.

        This must be part of env setup, not a best-effort post-step hook. If
        the task prompt advertises skills but files are absent in the
        container, the model learns against a broken environment and many
        trajectories fail with ``FileNotFoundError``.
        """
        self._retrieval_skills_injected = 0
        if not self.retrieval_skills or not self.container_handle:
            return
        if os.environ.get("UNIFIED_LAUNCHER_MODE", "mock").lower() == "mock":
            return

        from unified_runner.retrieval_skill_inject import (  # type: ignore
            DEFAULT_TOP_N,
            inject_retrieval_skills,
        )

        mapping, missing = _resolve_retrieval_skill_mapping(self.task_id, self.retrieval_skills)
        skill_paths = mapping.get(self.task_id, [])
        if missing:
            logger.warning(
                "[%s] unresolved retrieval skill names: %s",
                self.task_id,
                missing[:10],
            )
        if not skill_paths:
            raise LauncherError(
                f"[{self.task_id}] retrieval_skills_top_n was non-empty but no skill dirs resolved"
            )

        top_n = min(_positive_int_env("AGENT_BENCH_RETRIEVAL_TOP_N", DEFAULT_TOP_N), len(skill_paths))
        injected = inject_retrieval_skills(
            _docker_run_for_skill_injection,
            self.container_handle,
            self.task_id,
            mapping,
            top_n=top_n,
            verbose=False,
        )
        if injected <= 0:
            raise LauncherError(f"[{self.task_id}] retrieval skill injection copied 0 skills")

        verify_injection = os.environ.get(
            "AGENT_BENCH_VERIFY_SKILL_INJECTION", "0"
        ).lower() in {"1", "true", "yes"}
        if verify_injection:
            first_skill = Path(skill_paths[0]).name

            def _verify_skill_file(path: str) -> tuple[str, int]:
                stderr = ""
                rc = 124
                for attempt in range(1, 4):
                    _, stderr, rc = _docker_run_for_skill_injection(
                        [
                            "docker",
                            "exec",
                            self.container_handle,
                            "test",
                            "-f",
                            path,
                        ],
                        timeout=120,
                    )
                    if rc == 0:
                        return stderr, rc
                    logger.warning(
                        "[%s] retrieval skill verification attempt %s/3 failed for %s: %s",
                        self.task_id,
                        attempt,
                        path,
                        stderr[:200],
                    )
                    time.sleep(min(2 * attempt, 10))
                return stderr, rc

            stderr, rc = _verify_skill_file(f"/root/.claude/skills/{first_skill}/SKILL.md")
            if rc != 0:
                raise LauncherError(
                    f"[{self.task_id}] retrieval skill injection verification failed for {first_skill}: {stderr[:200]}"
                )
            stderr, rc = _verify_skill_file(f"/root/.claude/skills/{first_skill}/README.md")
            if rc != 0:
                raise LauncherError(
                    f"[{self.task_id}] retrieval skill README alias verification failed for {first_skill}: {stderr[:200]}"
                )

        self._retrieval_skills_injected = injected
        logger.info(
            "[%s] injected %s retrieval skills into %s",
            self.task_id,
            injected,
            self.container_handle,
        )

    def _build_tool_layer(self, container_handle: str | None):
        """Construct a ToolLayer matching the launcher mode.

        For ``mock`` mode we use a tiny in-process stub that returns a string
        echoing the call. For ``real`` mode we delegate to
        :class:`unified_runner.tool_layer.ToolLayer` running in docker mode.

        Per-bench workdir (matches native unified_runner choice):
          * claw  → /workspace  (mock-friendly path)
          * sb_ns/tb2/seta_synth (HarborLauncher) → /root  (image native)
          * swe_lite (SWELauncher) → /testbed (SWE-bench convention)
        """
        mode = os.environ.get("UNIFIED_LAUNCHER_MODE", "mock").lower()
        if mode == "mock":
            return _MockToolLayer()
        try:
            from unified_runner.tool_layer import ToolLayer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Cannot import unified_runner.tool_layer for real mode."
            ) from exc

        workdir_by_bench = {
            "claw": "/workspace",
            "sb_ns": "/root",
            "tb2": "/root",
            "seta_synth": "/root",
        }
        if self.bench == "swe_lite":
            # Native unified_swe.get_repo_path probes /testbed, /workspace,
            # /repo and falls back to /testbed. Mirror that here so each SWE
            # image gets its actual repo dir, not a hardcoded one.
            workdir = _probe_swe_repo_path(container_handle)
        else:
            workdir = workdir_by_bench.get(self.bench, "/root")
        return ToolLayer(mode="docker", container=container_handle, workdir=workdir)

    def _stash_score_on_sample(self) -> None:
        if self._final_score is None:
            return
        # Sample.metadata is the standard handoff to reward_func.
        # We update in-place so the change is visible after env.close().
        self._sample.metadata = dict(self._sample.metadata or {})
        self._sample.metadata["final_score"] = float(self._final_score)


def _probe_swe_repo_path(container: str) -> str:
    """Replicate unified_runner.run_unified_swe.get_repo_path: probe /testbed,
    /workspace, /repo via `docker exec test -d` and fall back to /testbed.
    """
    import subprocess

    docker_env = dict(os.environ, DOCKER_HOST=os.environ.get("DOCKER_HOST", "unix:///tmp/local-docker-overlay2.sock"))
    for candidate in ["/testbed", "/workspace", "/repo"]:
        try:
            r = subprocess.run(
                ["docker", "exec", container, "test", "-d", candidate],
                env=docker_env, timeout=10, capture_output=True,
            )
            if r.returncode == 0:
                return candidate
        except Exception:
            continue
    return "/testbed"


# ---------------------------------------------------------------------------
# Mock tool layer (used only when UNIFIED_LAUNCHER_MODE=mock)
# ---------------------------------------------------------------------------
class _MockToolLayer:
    """Returns a deterministic string per tool call, no I/O."""

    workdir = "/workspace"

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        return f"[mock-tool] {name}({json.dumps(arguments, ensure_ascii=False)[:200]}) → ok"

    def close(self) -> None:
        """Match the real ToolLayer lifecycle contract without side effects."""
        return None


# ---------------------------------------------------------------------------
# Factory used by Relax's rollout.py (`build_env(sample, args)`)
# ---------------------------------------------------------------------------
def build_env(sample: Any, args: Any) -> AgentBenchEnv:
    return AgentBenchEnv(sample, args)
