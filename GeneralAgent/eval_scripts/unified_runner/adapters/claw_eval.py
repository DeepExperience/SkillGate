"""Claw-Eval adapter for the unified runner.

Claw-Eval's sandbox_tools.py already defines tools (Bash, Read, Write, Edit,
Glob, Grep) that are 1:1 with Claude Code.  This adapter provides:

  1. A name-mapping layer: OpenClaw names → Claw-Eval names
  2. A results parser: reads existing Claw-Eval trace dirs
  3. An AbstractDatasetRunner subclass for uniform interface

Since Claw-Eval has its own full runner (run_eval.py), this adapter mostly
wraps existing functionality and normalizes output to TaskResult.

Usage:
    adapter = ClawEvalAdapter(config)
    results = adapter.load_existing_results("/path/to/traces/")
    print(adapter.summary_markdown(results))
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..base import AbstractDatasetRunner, TaskResult, RunConfig

# Claw-Eval tool name → OpenClaw tool name
CLAW_TO_OPENCLAW = {
    "Bash": "exec",
    "Read": "read",
    "Write": "write",
    "Edit": "edit",
    "Glob": "find",
    "Grep": "grep",
}

# Reverse mapping
OPENCLAW_TO_CLAW = {v: k for k, v in CLAW_TO_OPENCLAW.items()}

# Default paths
_ROOT = Path(os.environ.get("SKILLRL_ROOT", str(Path(__file__).resolve().parents[4])))
CLAW_EVAL_DIR = _ROOT / "datasets/claw-eval"
TRACES_DIR = _ROOT / "GeneralAgent/eval_scripts/claw_eval/traces"


class ClawEvalAdapter(AbstractDatasetRunner):
    """Adapter that reads Claw-Eval trace directories and normalizes results."""

    def __init__(
        self,
        config: RunConfig,
        traces_dir: str | Path = TRACES_DIR,
        split: str = "general",
    ) -> None:
        super().__init__(config)
        self.traces_dir = Path(traces_dir)
        self.split = split

    @property
    def dataset_name(self) -> str:
        return "claw-eval"

    def list_tasks(self) -> list[str]:
        """List available tasks from existing trace directories."""
        if not self.traces_dir.exists():
            return []
        tasks = set()
        for p in self.traces_dir.rglob("trace.json"):
            tasks.add(p.parent.name)
        return sorted(tasks)

    def setup_task(self, task_id: str) -> dict[str, Any]:
        # Claw-Eval handles its own setup
        return {"task_id": task_id}

    def run_agent(self, task_id: str, env: dict[str, Any]) -> TaskResult:
        # We don't re-run; we parse existing results
        return self._parse_trace(task_id)

    def teardown(self, task_id: str, env: dict[str, Any]) -> None:
        pass  # Nothing to clean up

    def load_existing_results(
        self,
        traces_dir: str | Path | None = None,
        model_prefix: str = "",
    ) -> list[TaskResult]:
        """Parse all existing Claw-Eval trace dirs into TaskResults."""
        tdir = Path(traces_dir) if traces_dir else self.traces_dir
        results = []
        for trace_file in sorted(tdir.rglob("trace.json")):
            task_id = trace_file.parent.name
            if model_prefix and model_prefix not in str(trace_file):
                continue
            result = self._parse_trace_file(trace_file, task_id)
            results.append(result)
        return results

    def _parse_trace(self, task_id: str) -> TaskResult:
        """Parse a single task's trace."""
        trace_file = self.traces_dir / task_id / "trace.json"
        if not trace_file.exists():
            return TaskResult(
                task_id=task_id,
                dataset=self.dataset_name,
                error=f"Trace not found: {trace_file}",
            )
        return self._parse_trace_file(trace_file, task_id)

    def _parse_trace_file(self, trace_file: Path, task_id: str) -> TaskResult:
        """Parse a trace.json file into a TaskResult."""
        try:
            data = json.loads(trace_file.read_text())
        except Exception as e:
            return TaskResult(
                task_id=task_id,
                dataset=self.dataset_name,
                error=f"Failed to parse {trace_file}: {e}",
            )

        # Claw-Eval trace format has task_score, passed, turns, etc.
        task_score = data.get("task_score", 0.0)
        passed = data.get("passed", False)
        turns = data.get("turns", 0)
        time_sec = data.get("time_sec", 0)

        # resolved = score >= 0.75 (Claw-Eval pass threshold)
        resolved = passed or task_score >= 0.75

        return TaskResult(
            task_id=task_id,
            dataset=self.dataset_name,
            resolved=resolved,
            score=task_score,
            turns=turns,
            time_sec=time_sec,
            extra={"trace_file": str(trace_file)},
        )


def load_claw_eval_summary(log_file: str | Path) -> list[TaskResult]:
    """Parse a Claw-Eval pass^3 log file into TaskResults.

    The log files (e.g. qwen3.5-27b_general_pass3.log) contain per-task results
    that we can extract.
    """
    log = Path(log_file)
    if not log.exists():
        return []
    results = []
    text = log.read_text()
    # This is a simple parser for the log format; adapt as needed
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("="):
            continue
        # Try to parse as JSON if the line is a JSON object
        if line.startswith("{"):
            try:
                d = json.loads(line)
                results.append(TaskResult(
                    task_id=d.get("task_id", "unknown"),
                    dataset="claw-eval",
                    resolved=d.get("passed", False),
                    score=d.get("task_score", 0.0),
                    turns=d.get("turns", 0),
                    time_sec=d.get("time_sec", 0),
                ))
            except json.JSONDecodeError:
                continue
    return results
