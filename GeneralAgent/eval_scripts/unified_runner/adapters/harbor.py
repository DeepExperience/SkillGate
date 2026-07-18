"""Harbor adapter for SkillsBench, Terminal-Bench 2.0, and SETA.

All three datasets use the Harbor evaluation framework with the terminus-2
agent. Harbor runs each task in a Docker container and produces a result.json
per trial with a standardized structure:

    result.json:
      task_name: str
      verifier_result.rewards.reward: float (0.0 or 1.0 typically)
      agent_result.n_input_tokens: int
      agent_result.n_output_tokens: int
      exception_info: {type, message} or null

This adapter:
  1. Parses existing Harbor result directories
  2. Normalizes to unified TaskResult
  3. Supports the ``harbor run`` CLI for launching new evaluations
  4. Maps Harbor's free-bash (keystrokes → exec) to the OpenClaw tool interface

Usage:
    adapter = HarborAdapter(config, dataset="tb2")
    results = adapter.load_results("/path/to/job-dir/")
    print(adapter.summary_markdown(results))
"""

from __future__ import annotations

import json
import subprocess
import os
from pathlib import Path
from typing import Any

from ..base import AbstractDatasetRunner, TaskResult, RunConfig

# Repo root: env override first, else derived from this file's location
# (GeneralAgent/eval_scripts/unified_runner/adapters/ -> repo root is 5 levels up).
_ROOT = Path(os.environ.get("SKILLRL_ROOT", str(Path(__file__).resolve().parents[4])))

# Dataset-specific paths
DATASET_DIRS = {
    "skillsbench": _ROOT / "datasets/skillsbench",
    "tb2": _ROOT / "datasets/terminal-bench-v2",
    "seta": _ROOT / "datasets/seta/dataset",
}

RESULTS_DIRS = {
    "skillsbench": _ROOT / "GeneralAgent/eval_scripts/skillsbench_eval/results",
    "tb2": _ROOT / "GeneralAgent/eval_scripts/tb2_eval/results",
    "seta": _ROOT / "GeneralAgent/eval_scripts/seta_eval/results",
}


class HarborAdapter(AbstractDatasetRunner):
    """Adapter for Harbor-format datasets (SkillsBench, TB2, SETA)."""

    def __init__(
        self,
        config: RunConfig,
        dataset: str = "tb2",
    ) -> None:
        super().__init__(config)
        assert dataset in ("skillsbench", "tb2", "seta"), f"Unknown dataset: {dataset}"
        self._dataset = dataset
        self._dataset_dir = DATASET_DIRS[dataset]
        self._results_base = RESULTS_DIRS[dataset]

    @property
    def dataset_name(self) -> str:
        return self._dataset

    def list_tasks(self) -> list[str]:
        """List available tasks from the dataset directory."""
        tasks = []
        for d in sorted(self._dataset_dir.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                # Harbor tasks have a task.toml or task.yaml
                if (d / "task.toml").exists() or (d / "task.yaml").exists():
                    tasks.append(d.name)
        return tasks

    def setup_task(self, task_id: str) -> dict[str, Any]:
        # Harbor handles its own container lifecycle
        return {"task_id": task_id, "dataset": self._dataset}

    def run_agent(self, task_id: str, env: dict[str, Any]) -> TaskResult:
        # We parse existing results rather than re-running
        return TaskResult(
            task_id=task_id,
            dataset=self.dataset_name,
            error="Direct agent execution not implemented; use harbor CLI",
        )

    def teardown(self, task_id: str, env: dict[str, Any]) -> None:
        pass

    def load_results(self, job_dir: str | Path) -> list[TaskResult]:
        """Parse a Harbor job directory into TaskResults."""
        jdir = Path(job_dir)
        if not jdir.exists():
            return []

        results = []
        for trial_dir in sorted(jdir.iterdir()):
            if not trial_dir.is_dir():
                continue
            result_file = trial_dir / "result.json"
            if not result_file.exists():
                continue
            result = self._parse_result(result_file)
            results.append(result)

        self._results = results
        return results

    def load_all_jobs(self, results_base: Path | None = None) -> dict[str, list[TaskResult]]:
        """Load all job directories under the results base."""
        base = results_base or self._results_base
        if not base.exists():
            return {}
        jobs = {}
        for job_dir in sorted(base.iterdir()):
            if job_dir.is_dir() and not job_dir.name.endswith(".log"):
                results = self.load_results(job_dir)
                if results:
                    jobs[job_dir.name] = results
        return jobs

    def _parse_result(self, result_file: Path) -> TaskResult:
        """Parse a single Harbor result.json into a TaskResult."""
        try:
            data = json.loads(result_file.read_text())
        except Exception as e:
            return TaskResult(
                task_id=result_file.parent.name,
                dataset=self.dataset_name,
                error=f"Failed to parse {result_file}: {e}",
            )

        task_name = data.get("task_name") or result_file.parent.name.split("__")[0]

        # Extract reward
        verifier = data.get("verifier_result") or {}
        rewards = verifier.get("rewards") or {}
        reward = rewards.get("reward")
        if reward is None:
            reward = 0.0

        # Extract agent stats
        agent = data.get("agent_result") or {}
        tokens_in = agent.get("n_input_tokens", 0) or 0
        tokens_out = agent.get("n_output_tokens", 0) or 0

        # Extract exception info
        exc_info = data.get("exception_info") or {}
        error = ""
        if exc_info:
            error = f"{exc_info.get('type', '?')}: {exc_info.get('message', '')}"[:200]

        return TaskResult(
            task_id=task_name,
            dataset=self.dataset_name,
            resolved=(reward >= 1.0),
            score=float(reward),
            turns=0,  # Harbor doesn't expose turn count directly
            time_sec=0,
            error=error,
            extra={
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "trial_dir": str(result_file.parent),
            },
        )

    @staticmethod
    def build_harbor_cmd(
        dataset_dir: str | Path,
        model: str,
        api_base: str,
        results_dir: str | Path,
        job_name: str,
        agent: str = "terminus-2",
        concurrency: int = 4,
        timeout_multiplier: float = 1.5,
        max_input_tokens: int = 131072,
        max_output_tokens: int = 8192,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        """Build a ``harbor run`` command line.

        Returns the command as a list suitable for subprocess.run().
        """
        cmd = [
            "/root/.local/bin/harbor", "run",
            "-p", str(dataset_dir),
            "-a", agent,
            "-m", f"openai/{model}",
            "--ak", f"api_base={api_base}",
            "--ak", f"model_info={{\"max_input_tokens\":{max_input_tokens},\"max_output_tokens\":{max_output_tokens}}}",
            "--agent-timeout-multiplier", str(timeout_multiplier),
            "-o", str(results_dir),
            "--job-name", job_name,
            "-n", str(concurrency),
            "--quiet",
        ]
        if extra_args:
            cmd.extend(extra_args)
        return cmd

    @staticmethod
    def harbor_env() -> dict[str, str]:
        """Return environment variables needed for harbor runs."""
        import os
        env = dict(os.environ)
        env["DOCKER_HOST"] = os.environ.get("DOCKER_HOST", "ssh://your-docker-host")
        env["NO_PROXY"] = "127.0.0.1,localhost"
        env["OPENAI_API_KEY"] = "dummy"
        return env
