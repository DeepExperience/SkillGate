#!/usr/bin/env python3
"""Build the 20260508 OpenClaw-aligned eval table.

The user-facing target table has two parts:

1. full-benchmark pass rates for baseline / retrieval / SFT rows;
2. quick30 holdout diagnostics: total resolved, strict skill use, and
   strict skill-use successes.

The script is intentionally tolerant of partial runs. Incomplete cells are
rendered as `resolved/done/expected (pct on done)`, so it can be used as a live
progress view and as the final table once all runs finish.
"""

from __future__ import annotations

import os

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))
EXPERIMENTS = PROJECT_ROOT / "experiments"

FULL_BENCHES: list[tuple[str, str, int]] = [
    ("seta", "SETA(30)", 30),
    # Effective runnable SWE set is 20 here: legacy ALL_IMAGES lists 21, but
    # `mwaskom__seaborn-3010` is not present in the local SWE parquet loaded by
    # `run_unified_swe.py`, so the runner cannot evaluate it.
    ("swe", "SWE-Gym", 20),
    ("claw", "Claw-161", 161),
    # Effective runnable SB set is 85: `scheduling-email-assistant` is
    # structurally excluded by `run_unified_harbor.py` because it requires
    # external Google OAuth credentials.
    ("skillsbench-no-skills", "SB", 85),
    ("tb2", "TB 2.0", 89),
]


@dataclass(frozen=True)
class RowSpec:
    label: str
    full_run_id: str
    quick_run_id: str | None = None


def row_specs(date: str) -> list[RowSpec]:
    return [
        RowSpec("9B baseline", f"{date}_full_base9b_baseline_openclaw_full", f"{date}_quick30_base9b_baseline_openclaw_full"),
        RowSpec("9B base-retrieval", f"{date}_full_base9b_retrieval_openclaw_full", f"{date}_quick30_base9b_retrieval_openclaw_full"),
        RowSpec("9B SFT-2093", f"{date}_full_sft2093_retrieval_openclaw_full", f"{date}_quick30_sft2093_openclaw_full_retrieval"),
        RowSpec("27B baseline", f"{date}_full_base27b_baseline_openclaw_full", f"{date}_quick30_base27b_baseline_openclaw_full"),
        RowSpec("27B base-retrieval", f"{date}_full_base27b_retrieval_openclaw_full", f"{date}_quick30_base27b_retrieval_openclaw_full"),
    ]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def full_bench_rows(run_root: Path, bench: str) -> list[dict[str, Any]]:
    result_root = run_root / "results" / bench
    rows: list[dict[str, Any]] = []
    if not result_root.exists():
        return rows
    for path in sorted(result_root.glob("*/incremental.jsonl")):
        rows.extend(load_jsonl(path))
    return rows


def full_cell(run_root: Path, bench: str, expected: int) -> str:
    rows = full_bench_rows(run_root, bench)
    if not rows:
        return "—"
    done = len(rows)
    resolved = sum(bool(row.get("resolved")) for row in rows)
    pct = 100.0 * resolved / done if done else 0.0
    if done >= expected:
        return f"{resolved}/{expected} ({pct:.1f}%)"
    return f"{resolved}/{done}/{expected} ({pct:.1f}%)"


def load_dashboard(run_root: Path) -> dict[str, Any]:
    path = run_root / "reports" / "data_quality_dashboard.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def quick_cells(run_root: Path) -> tuple[str, str, str]:
    dash = load_dashboard(run_root)
    totals = dash.get("totals") or {}
    loaded = int(totals.get("trajectories_loaded") or totals.get("status_rows_latest") or 0)
    if not loaded:
        return ("—", "—", "—")
    resolved = int(totals.get("resolved") or 0)
    strict_used = int(totals.get("strict_used_skill") or 0)
    used_resolved = int(totals.get("success_strict_used_skill_non_meta") or totals.get("success_strict_used_skill") or 0)
    return (
        f"{resolved}/{loaded} ({100.0 * resolved / loaded:.1f}%)",
        f"{strict_used}/{loaded} ({100.0 * strict_used / loaded:.1f}%)",
        f"{used_resolved}/{loaded} ({100.0 * used_resolved / loaded:.1f}%)",
    )


def run_status(run_root: Path) -> str:
    if not run_root.exists():
        return "missing"
    dash = run_root / "reports" / "data_quality_dashboard.json"
    if dash.exists():
        return "dashboard"
    if any(run_root.glob("results/**/incremental.jsonl")):
        return "partial"
    return "created"


def build_table(date: str) -> str:
    specs = row_specs(date)
    lines: list[str] = []
    lines.append(f"# OpenClaw-Aligned Eval Table ({date})")
    lines.append("")
    lines.append("Cells are `resolved/done/expected` while a run is partial, and `resolved/expected` after completion.")
    lines.append("")
    header = ["row"] + [label for _, label, _ in FULL_BENCHES] + ["status"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")
    for spec in specs:
        run_root = EXPERIMENTS / date / spec.full_run_id
        cells = [full_cell(run_root, bench, expected) for bench, _, expected in FULL_BENCHES]
        lines.append("| " + " | ".join([spec.label] + cells + [run_status(run_root)]) + " |")

    lines.append("")
    lines.append("## Quick30 Holdout")
    lines.append("")
    lines.append("| row | total resolved | strict used | used + resolved non-meta | status |")
    lines.append("|---|---:|---:|---:|---|")
    for spec in specs:
        if not spec.quick_run_id:
            continue
        run_root = EXPERIMENTS / date / spec.quick_run_id
        total, strict, used_resolved = quick_cells(run_root)
        lines.append(f"| {spec.label} | {total} | {strict} | {used_resolved} | {run_status(run_root)} |")

    lines.append("")
    lines.append("## Run Paths")
    lines.append("")
    for spec in specs:
        lines.append(f"- {spec.label} full: `{EXPERIMENTS / date / spec.full_run_id}`")
        if spec.quick_run_id:
            lines.append(f"- {spec.label} quick30: `{EXPERIMENTS / date / spec.quick_run_id}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260508")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    text = build_table(args.date)
    out = Path(args.out) if args.out else EXPERIMENTS / args.date / "monitor_9b_goal" / "reports" / "current_eval_table.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
