"""Unified results table: aggregate all datasets × models into one markdown table.

Reads from:
  - SkillsBench Harbor results
  - Terminal-Bench 2.0 Harbor results
  - SETA Harbor results
  - Claw-Eval trace logs
  - SWE evaluation logs

Produces a single comparison table with resolve_rate and mean_score per
dataset × model combination.

Usage:
    from unified_runner.adapters.results_table import ResultsAggregator
    agg = ResultsAggregator()
    agg.add_harbor("tb2", "qwen3-14b", "/path/to/job-dir")
    agg.add_harbor("tb2", "qwen3.5-27b", "/path/to/job-dir")
    agg.add_manual("claw-eval", "qwen3.5-27b", resolve_rate=0.366, mean_score=0.610)
    print(agg.to_markdown())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..base import TaskResult


@dataclass
class DatasetModelResult:
    """Aggregated result for one dataset × model combination."""

    dataset: str
    model: str
    total: int = 0
    resolved: int = 0
    resolve_rate: float = 0.0
    mean_score: float = 0.0
    variant: str = ""  # e.g. "with-skills", "no-skills"
    extra: dict[str, Any] = field(default_factory=dict)


class ResultsAggregator:
    """Collects and formats results from all datasets and models."""

    def __init__(self) -> None:
        self._entries: list[DatasetModelResult] = []

    def add_from_task_results(
        self,
        dataset: str,
        model: str,
        results: list[TaskResult],
        variant: str = "",
    ) -> DatasetModelResult:
        """Add results from a list of TaskResult objects."""
        total = len(results)
        resolved = sum(1 for r in results if r.resolved)
        scores = [r.score for r in results]
        entry = DatasetModelResult(
            dataset=dataset,
            model=model,
            total=total,
            resolved=resolved,
            resolve_rate=resolved / total if total else 0,
            mean_score=sum(scores) / total if total else 0,
            variant=variant,
        )
        self._entries.append(entry)
        return entry

    def add_harbor(
        self,
        dataset: str,
        model: str,
        job_dir: str | Path,
        variant: str = "",
    ) -> DatasetModelResult | None:
        """Add results from a Harbor job directory."""
        from .harbor import HarborAdapter
        from ..base import RunConfig

        adapter = HarborAdapter(RunConfig(), dataset=dataset)
        results = adapter.load_results(job_dir)
        if not results:
            return None
        return self.add_from_task_results(dataset, model, results, variant)

    def add_swe(
        self,
        model: str,
        results_file: str | Path,
    ) -> DatasetModelResult | None:
        """Add results from a SWE evaluation results file (JSONL or MD)."""
        rfile = Path(results_file)
        if not rfile.exists():
            return None

        results = []
        if rfile.suffix == ".jsonl":
            for line in rfile.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    results.append(TaskResult(
                        task_id=d.get("instance_id", "unknown"),
                        dataset="swe",
                        resolved=d.get("resolved", False),
                        score=1.0 if d.get("resolved", False) else 0.0,
                        turns=d.get("turns", 0),
                        time_sec=d.get("time_sec", 0),
                    ))
                except json.JSONDecodeError:
                    continue
        return self.add_from_task_results("swe", model, results) if results else None

    def add_manual(
        self,
        dataset: str,
        model: str,
        resolve_rate: float | None = None,
        mean_score: float | None = None,
        total: int = 0,
        resolved: int = 0,
        variant: str = "",
    ) -> DatasetModelResult:
        """Add a manually-specified result (e.g. from Claw-Eval pass^3)."""
        entry = DatasetModelResult(
            dataset=dataset,
            model=model,
            total=total,
            resolved=resolved,
            resolve_rate=resolve_rate if resolve_rate is not None else (resolved / total if total else 0),
            mean_score=mean_score if mean_score is not None else 0,
            variant=variant,
        )
        self._entries.append(entry)
        return entry

    def to_markdown(self, title: str = "Unified Results Table") -> str:
        """Generate a markdown comparison table.

        Format:
        | Dataset (variant) | Metric | Model1 | Model2 | ... |
        """
        if not self._entries:
            return f"## {title}\n\nNo results available.\n"

        # Collect all models (preserve insertion order)
        models = list(dict.fromkeys(e.model for e in self._entries))

        # Group by (dataset, variant)
        groups: dict[tuple[str, str], dict[str, DatasetModelResult]] = {}
        for e in self._entries:
            key = (e.dataset, e.variant)
            if key not in groups:
                groups[key] = {}
            groups[key][e.model] = e

        lines = [
            f"## {title}",
            "",
            "| Dataset | Metric | " + " | ".join(models) + " |",
            "|---------|--------| " + " | ".join(["---"] * len(models)) + " |",
        ]

        for (dataset, variant), model_results in sorted(groups.items()):
            label = dataset
            if variant:
                label += f" ({variant})"

            # resolve_rate row
            vals = []
            for m in models:
                e = model_results.get(m)
                if e:
                    n_info = f" ({e.resolved}/{e.total})" if e.total else ""
                    vals.append(f"{e.resolve_rate:.1%}{n_info}")
                else:
                    vals.append("-")
            lines.append(f"| {label} | resolve_rate | " + " | ".join(vals) + " |")

            # mean_score row
            vals = []
            for m in models:
                e = model_results.get(m)
                if e:
                    vals.append(f"{e.mean_score:.4f}")
                else:
                    vals.append("-")
            lines.append(f"| | mean_score | " + " | ".join(vals) + " |")

        return "\n".join(lines) + "\n"

    def to_experiment_plan_format(self) -> str:
        """Generate the table format used in EXPERIMENT_PLAN.md section 1.x."""
        if not self._entries:
            return "No results.\n"

        models = list(dict.fromkeys(e.model for e in self._entries))
        groups: dict[str, dict[str, DatasetModelResult]] = {}
        for e in self._entries:
            key = e.dataset + (f" ({e.variant})" if e.variant else "")
            if key not in groups:
                groups[key] = {}
            groups[key][e.model] = e

        header_cols = ["Benchmark"]
        for m in models:
            header_cols.extend([f"resolve_rate ({m})", f"mean_score ({m})"])

        lines = [
            "| " + " | ".join(header_cols) + " |",
            "| " + " | ".join(["---"] * len(header_cols)) + " |",
        ]

        for dataset_label, model_results in sorted(groups.items()):
            row = [dataset_label]
            for m in models:
                e = model_results.get(m)
                if e:
                    row.append(f"{e.resolve_rate:.1%}")
                    row.append(f"{e.mean_score:.4f}")
                else:
                    row.append("-")
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines) + "\n"
