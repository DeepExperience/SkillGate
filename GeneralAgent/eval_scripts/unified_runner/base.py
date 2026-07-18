"""Abstract base class for dataset runners.

Each dataset adapter subclasses AbstractDatasetRunner and implements:
  - list_tasks()   → task IDs available
  - setup_task()   → prepare environment (e.g. start Docker container)
  - run_agent()    → agent loop with tool calls
  - evaluate()     → score the trajectory
  - teardown()     → cleanup (e.g. stop container)

The base also provides a ``run_all()`` driver that iterates tasks, handles
errors, and writes incremental results.
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TaskResult:
    """Standardized result for one task across all datasets."""

    task_id: str
    dataset: str
    resolved: bool = False
    score: float = 0.0          # continuous 0-1
    turns: int = 0
    time_sec: int = 0
    error: str = ""
    patch: str = ""             # for SWE-style tasks
    trajectory: list[dict] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("trajectory", None)  # exclude verbose trajectory by default
        return d


@dataclass
class RunConfig:
    """Configuration passed to all adapters."""

    model: str = "qwen3-14b"
    api_base: str = "http://localhost:30000/v1"
    api_key: str = ""
    max_turns: int = 50
    max_time_sec: int = 1800
    max_output_chars: int = 16000
    temperature: float = 0.6
    max_tokens: int = 8192
    # 2026-04-20 v6: anti-repetition for 27B no-think mode. 1.5 is aggressive but
    # needed for 27B which loops on identical tool calls.
    presence_penalty: float = 0.0
    # 2026-04-20 v6: extra_body injected into chat.completions payload. Used for
    # SGLang-specific flags (e.g. {"chat_template_kwargs": {"enable_thinking": False}}).
    extra_body: dict = None
    # 2026-04-20 v6: early stop after N consecutive identical assistant text+tool_calls.
    # 0 disables (default for backwards-compat); 3 is recommended for no-think 27B.
    early_stop_repeat_n: int = 0
    docker_host: str = "tcp://127.0.0.1:2375"
    workdir: str = "/workspace"
    results_dir: str = ""
    job_name: str = ""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def experiments_root() -> Path:
    path = Path(os.environ.get("EXPERIMENTS_ROOT", _project_root() / "experiments"))
    return path if path.is_absolute() else _project_root() / path


def unified_run_id(date_prefix: str) -> str:
    run_id = (
        os.environ.get("UNIFIED_RUN_ID")
        or os.environ.get("RUN_ID")
        or os.environ.get("EXPERIMENT_ID")
    )
    if run_id:
        return run_id
    return f"{date_prefix}_{experiment_version()}"


def infer_experiment_date(run_id: str, date_prefix: str = None) -> str:
    for pattern in (r"^(20\d{6})", r"(20\d{6})"):
        match = re.search(pattern, run_id or "")
        if match:
            return match.group(1)
    for value in (date_prefix, os.environ.get("EXPERIMENT_DATE"), os.environ.get("DATE")):
        if value and re.fullmatch(r"20\d{6}", value.strip()):
            return value.strip()
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def unified_run_root(date_prefix: str) -> Path:
    override = os.environ.get("UNIFIED_RUN_ROOT") or os.environ.get("EXPERIMENT_ROOT") or os.environ.get("RUN_ROOT")
    if override:
        path = Path(override)
        return path if path.is_absolute() else _project_root() / path
    run_id = unified_run_id(date_prefix)
    kind = os.environ.get("UNIFIED_EXPERIMENT_KIND", "eval").strip().lower()
    if kind == "train":
        experiment_id = os.environ.get("EXPERIMENT_ID", "").strip()
        segment_id = os.environ.get("RUN_NAME", "").strip() or run_id
        if not experiment_id:
            raise RuntimeError("RL train output requires EXPERIMENT_ID")
        return experiments_root() / "rl" / "runs" / experiment_id / "segments" / segment_id
    owner = os.environ.get("OWNER_EXPERIMENT_ID", "").strip()
    eval_id = os.environ.get("EVAL_ID", "").strip()
    row_id = os.environ.get("EVAL_ROW_ID", "").strip() or run_id
    if owner or eval_id:
        if not owner or not eval_id:
            raise RuntimeError("owner-local RL eval requires OWNER_EXPERIMENT_ID and EVAL_ID")
        return experiments_root() / "rl" / "runs" / owner / "eval" / eval_id / "rows" / row_id

    # General/non-RL evaluation and existing legacy callers retain their dated
    # layout. Canonical RL wrappers always take one of the owner-aware branches
    # above, so no central experiments/rl_eval directory is created.
    dated_root = experiments_root() / infer_experiment_date(run_id, date_prefix) / run_id
    if dated_root.exists():
        return dated_root
    legacy_root = experiments_root() / run_id
    if legacy_root.exists():
        return legacy_root
    return dated_root


def results_subdir(results_dir, date_prefix: str, bench: str = None, experiment: str = None):
    """Return nested results directory, creating it if absent.

    2026-04-28 experiments layout:
        experiments / <date> / <run_id> / results / <bench> / <experiment> /
            incremental.jsonl
            trajectories/
            summary.md

    Set UNIFIED_LEGACY_RESULTS_LAYOUT=1 to write the old v8 path:
        results_dir / <date> / <bench> / <experiment> /

    All three args optional for backwards-compat:
      results_subdir(rd, date)                    → rd/date
      results_subdir(rd, date, bench="tb2")       → rd/date/tb2
      results_subdir(rd, date, bench="tb2",
                     experiment="v8_retrieval")   → rd/date/tb2/v8_retrieval

    Usage (new v8 style):
        from unified_runner.base import results_subdir
        exp = results_subdir(RESULTS_DIR, date, bench=dataset_tag,
                             experiment=f"{version}_{arm}")
        (exp / "incremental.jsonl").touch()
        (exp / "trajectories").mkdir(exist_ok=True)
    """
    from pathlib import Path
    if os.environ.get("UNIFIED_LEGACY_RESULTS_LAYOUT", "").strip() == "1":
        p = Path(results_dir) / date_prefix
        if bench:
            p = p / bench
        if experiment:
            p = p / experiment
    else:
        p = unified_run_root(date_prefix) / "results"
        if bench:
            p = p / bench
        if experiment:
            p = p / experiment
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_experiments(results_dir, date: str = None, bench: str = None,
                     experiment: str = None) -> list:
    """Discover experiment dirs in the v8 layout.

    Returns list of dict: {date, bench, experiment, dir, incremental, summary, trajectories}
    Filters optional. `experiment` supports prefix match ("v8" matches "v8_baseline").

    Layout: experiments/<date>/<run_id>/results/<bench>/<experiment>/incremental.jsonl
    Legacy v8 date layout is still searched as a fallback.
    """
    from pathlib import Path
    results_dir = Path(results_dir)
    out = []
    if results_dir.name == "experiments":
        run_dirs = []
        for top_dir in sorted(results_dir.iterdir()) if results_dir.exists() else []:
            if not top_dir.is_dir():
                continue
            if re.fullmatch(r"20\d{6}", top_dir.name):
                if date and top_dir.name != date:
                    continue
                run_dirs.extend(d for d in sorted(top_dir.iterdir()) if d.is_dir())
            elif (top_dir / "results").is_dir():
                if date and not top_dir.name.startswith(date):
                    continue
                run_dirs.append(top_dir)
        for run_dir in run_dirs:
            result_root = run_dir / "results"
            if not result_root.is_dir():
                continue
            bench_dirs = [result_root / bench] if bench else [b for b in sorted(result_root.iterdir()) if b.is_dir()]
            for bdir in bench_dirs:
                if not bdir.is_dir():
                    continue
                for edir in sorted(bdir.iterdir()):
                    if not edir.is_dir():
                        continue
                    if experiment and not edir.name.startswith(experiment):
                        continue
                    inc = edir / "incremental.jsonl"
                    if not inc.exists():
                        continue
                    run_date = run_dir.parent.name if re.fullmatch(r"20\d{6}", run_dir.parent.name) else ""
                    out.append({
                        "date": run_date or (run_dir.name[:8] if re.match(r"^\d{8}", run_dir.name) else ""),
                        "run_id": run_dir.name,
                        "bench": bdir.name,
                        "experiment": edir.name,
                        "dir": edir,
                        "incremental": inc,
                        "summary": edir / "summary.md",
                        "trajectories": edir / "trajectories",
                    })
        if out:
            return out
    date_dirs = [results_dir / date] if date else \
                [d for d in sorted(results_dir.iterdir())
                 if d.is_dir() and re.match(r"^\d{8}$", d.name)]
    for ddir in date_dirs:
        if not ddir.is_dir(): continue
        bench_dirs = [ddir / bench] if bench else \
                     [b for b in sorted(ddir.iterdir())
                      if b.is_dir() and b.name not in ("retrieval_results", "_reports")]
        for bdir in bench_dirs:
            if not bdir.is_dir(): continue
            for edir in sorted(bdir.iterdir()):
                if not edir.is_dir(): continue
                if experiment and not edir.name.startswith(experiment): continue
                inc = edir / "incremental.jsonl"
                if not inc.exists(): continue
                out.append({
                    "date": ddir.name,
                    "bench": bdir.name,
                    "experiment": edir.name,
                    "dir": edir,
                    "incremental": inc,
                    "summary": edir / "summary.md",
                    "trajectories": edir / "trajectories",
                })
    return out


def experiment_version() -> str:
    """Read UNIFIED_EXP_VERSION env var. Default 'v8' (the current milestone)."""
    import os
    return os.environ.get("UNIFIED_EXP_VERSION", "v8").strip() or "v8"


def experiment_name(arm: str, tag: str = None) -> str:
    """Compose <experiment> = <version>_<arm>[_<tag>].

    arm: "baseline" | "retrieval" | "irrelevant"
    tag: optional suffix (e.g. "run1", "partial_52", "concurrent")
    """
    ver = experiment_version()
    name = f"{ver}_{arm}"
    if tag:
        name += f"_{tag}"
    return name


def write_summary_md(exp_dir, dataset_tag, model, results, interface="Unified OpenClaw deploy-tool subset", extra_meta: dict = None):
    """Write summary.md with full metrics per 2026-04-22 v8 schema.

    Required metrics: N_total, N_pass, N_error, pass_rate, Mean_score.
    """
    from datetime import datetime
    from pathlib import Path
    from collections import Counter

    exp_dir = Path(exp_dir)
    n = len(results)
    n_pass = sum(1 for r in results if r.get("resolved"))
    n_err = sum(1 for r in results if (r.get("error") or "").strip()
                or (r.get("finish_reason") and r.get("finish_reason") != "completed"))
    mean_score = (sum(r.get("score") or 0 for r in results) / n) if n else 0
    pass_rate = (n_pass / n) if n else 0
    fr = Counter(r.get("finish_reason") for r in results)

    lines = []
    lines.append(f"# {exp_dir.parent.name} / {exp_dir.name}")
    lines.append("")
    lines.append(f"**Dataset**: {dataset_tag}")
    lines.append(f"**Model**: {model}")
    lines.append(f"**Interface**: {interface}")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if extra_meta:
        for k, v in extra_meta.items():
            lines.append(f"**{k}**: {v}")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(f"- **N_total**: {n}")
    lines.append(f"- **N_pass**: {n_pass}")
    lines.append(f"- **N_error**: {n_err}")
    lines.append(f"- **pass_rate**: {pass_rate*100:.1f}%")
    lines.append(f"- **Mean_score**: {mean_score:.3f}")
    lines.append("")
    lines.append("### Finish reason distribution")
    lines.append("")
    for k, v in sorted(fr.items(), key=lambda x: -x[1]):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Per-task results")
    lines.append("")
    lines.append("| task_id | resolved | score | turns | time_sec | finish_reason | error |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        tid = r.get("task_id") or r.get("instance_id", "?")
        resolved = "✅" if r.get("resolved") else "❌"
        score = r.get("score", 0)
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
        turns = r.get("turns", "")
        time_sec = r.get("time_sec") or r.get("wall_sec", "")
        fr_str = r.get("finish_reason", "")
        err = (r.get("error") or "")[:80].replace("\n", " ").replace("|", "\\|")
        lines.append(f"| {tid} | {resolved} | {score_str} | {turns} | {time_sec} | {fr_str} | {err} |")

    (exp_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return exp_dir / "summary.md"


def env_overrides() -> dict:
    """Read v6 env vars and produce RunConfig overrides dict.

    Env vars (all optional, default → no change):
      UNIFIED_PRESENCE_PENALTY    — float (e.g. 1.5)
      UNIFIED_EARLY_STOP_N        — int (e.g. 3)
      UNIFIED_DISABLE_THINKING    — "1" to set SGLang chat_template_kwargs.enable_thinking=false
      UNIFIED_FORCE_SGLANG_EXTRA_BODY — "1" to send SGLang extra_body to non-local endpoints
      OPENAI_API_KEY              — optional OpenAI-compatible bearer token

    Use in runner:
        config = RunConfig(..., **env_overrides())
    """
    import os
    out: dict = {}
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        out["api_key"] = api_key
    pp = os.environ.get("UNIFIED_PRESENCE_PENALTY", "").strip()
    if pp:
        try: out["presence_penalty"] = float(pp)
        except ValueError: pass
    es = os.environ.get("UNIFIED_EARLY_STOP_N", "").strip()
    if es:
        try: out["early_stop_repeat_n"] = int(es)
        except ValueError: pass
    if os.environ.get("UNIFIED_DISABLE_THINKING", "").strip() == "1":
        api_base = os.environ.get("OPENAI_API_BASE", "")
        is_local_sglang = (
            "127.0.0.1" in api_base
            or "localhost" in api_base
            or os.environ.get("UNIFIED_FORCE_SGLANG_EXTRA_BODY", "").strip() == "1"
        )
        if is_local_sglang:
            model_name = os.environ.get("UNIFIED_MODEL", "").lower()
            if "deepseek-v4" in model_name or "deepseek_v4" in model_name:
                out["extra_body"] = {"chat_template_kwargs": {"thinking": False}}
            else:
                out["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return out


class AbstractDatasetRunner(ABC):
    """Base class for dataset-specific evaluation runners."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self._results: list[TaskResult] = []

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Short dataset identifier (e.g. 'claw-eval', 'tb2', 'swe')."""

    @abstractmethod
    def list_tasks(self) -> list[str]:
        """Return list of task IDs available in this dataset."""

    @abstractmethod
    def setup_task(self, task_id: str) -> dict[str, Any]:
        """Prepare environment for a task. Returns env context dict."""

    @abstractmethod
    def run_agent(self, task_id: str, env: dict[str, Any]) -> TaskResult:
        """Run the agent on one task. Returns scored result."""

    @abstractmethod
    def teardown(self, task_id: str, env: dict[str, Any]) -> None:
        """Clean up after a task (stop containers, etc.)."""

    def run_all(
        self,
        task_ids: list[str] | None = None,
        skip_existing: bool = True,
    ) -> list[TaskResult]:
        """Run all (or selected) tasks with error handling and incremental saves."""
        ids = task_ids or self.list_tasks()
        results_dir = Path(self.config.results_dir) if self.config.results_dir else None
        if results_dir:
            results_dir.mkdir(parents=True, exist_ok=True)

        results: list[TaskResult] = []
        for idx, tid in enumerate(ids, 1):
            # Skip if result already exists
            if skip_existing and results_dir:
                result_file = results_dir / f"{tid}.json"
                if result_file.exists():
                    print(f"  [{idx}/{len(ids)}] {tid}: skipping (result exists)")
                    try:
                        existing = json.loads(result_file.read_text())
                        results.append(TaskResult(**{
                            k: v for k, v in existing.items()
                            if k in TaskResult.__dataclass_fields__
                        }))
                    except Exception:
                        pass
                    continue

            print(f"\n{'='*60}")
            print(f"[{idx}/{len(ids)}] {self.dataset_name}: {tid}")
            print(f"{'='*60}")

            env = {}
            try:
                env = self.setup_task(tid)
                result = self.run_agent(tid, env)
            except Exception as exc:
                result = TaskResult(
                    task_id=tid,
                    dataset=self.dataset_name,
                    error=f"{type(exc).__name__}: {exc}",
                )
            finally:
                try:
                    self.teardown(tid, env)
                except Exception:
                    pass

            results.append(result)
            self._save_incremental(result, idx, len(ids), results_dir)

        self._results = results
        return results

    def _save_incremental(
        self,
        result: TaskResult,
        idx: int,
        total: int,
        results_dir: Path | None,
    ) -> None:
        status = "RESOLVED" if result.resolved else "FAILED"
        score_str = f", score={result.score:.3f}" if result.score > 0 else ""
        print(f"  [{idx}/{total}] {result.task_id}: {status}{score_str}"
              f" (turns={result.turns}, time={result.time_sec}s)")

        if results_dir:
            out_file = results_dir / f"{result.task_id}.json"
            out_file.write_text(
                json.dumps(result.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )

    # --- summary helpers ---

    def summary(self, results: list[TaskResult] | None = None) -> dict[str, Any]:
        """Compute summary statistics."""
        rs = results or self._results
        if not rs:
            return {"error": "no results"}
        total = len(rs)
        resolved = sum(1 for r in rs if r.resolved)
        scores = [r.score for r in rs]
        return {
            "dataset": self.dataset_name,
            "total": total,
            "resolved": resolved,
            "resolve_rate": resolved / total if total else 0,
            "mean_score": sum(scores) / total if total else 0,
            "errors": sum(1 for r in rs if r.error),
        }

    def summary_markdown(self, results: list[TaskResult] | None = None) -> str:
        """Generate a markdown summary table."""
        rs = results or self._results
        s = self.summary(rs)
        lines = [
            f"## {self.dataset_name} Results",
            "",
            f"- **Total**: {s['total']}",
            f"- **Resolved**: {s['resolved']}/{s['total']}"
            f" ({s['resolve_rate']:.1%})",
            f"- **Mean Score**: {s['mean_score']:.4f}",
            f"- **Errors**: {s['errors']}",
            "",
            "| task_id | resolved | score | turns | time | error |",
            "|---------|----------|-------|-------|------|-------|",
        ]
        for r in rs:
            err = r.error[:50].replace("|", "/") if r.error else ""
            lines.append(
                f"| {r.task_id} | {'YES' if r.resolved else 'NO'} "
                f"| {r.score:.3f} | {r.turns} | {r.time_sec}s | {err} |"
            )
        return "\n".join(lines)
