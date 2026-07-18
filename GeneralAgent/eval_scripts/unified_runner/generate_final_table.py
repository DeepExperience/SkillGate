#!/usr/bin/env python3
"""Generate the final dual-column comparison table.

Reads results from:
  - Original framework: progress.md (hardcoded), SWE rerun incremental
  - Unified interface: claw-eval batch_results, harbor incremental, SWE unified incremental

Output: experiments/<date>/<run_id>/reports/final_comparison_table.md
"""

import json
from datetime import datetime
import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("SKILLRL_ROOT", str(Path(__file__).resolve().parents[3])))
RESULTS_DIR = BASE_DIR / "experiments"

# Original framework results (from progress.md, hardcoded)
ORIGINAL_27B = {
    "claw-eval": {"resolve": 0.435, "mean": 0.610, "n": 161},
    "seta": {"resolve": 0.633, "mean": 0.633, "n": 30},
    "skillsbench-ws": {"resolve": 0.125, "mean": 0.134, "n": 88},
    "skillsbench-ns": {"resolve": 0.058, "mean": 0.072, "n": 86},
    "swe": {"resolve": None, "mean": None, "n": 19},  # To be filled from rerun
}

ORIGINAL_14B = {
    "claw-eval": {"resolve": 0.130, "mean": 0.420, "n": 161},
    "skillsbench-ws": {"resolve": 0.011, "mean": 0.019, "n": 88},
    "swe": {"resolve": 0.000, "mean": 0.000, "n": 19},
}


def load_swe_original_rerun():
    """Load SWE original framework rerun results.

    Prefers verified results (from verify_patches.py using gold test patches)
    over the incremental results (which use our simple run_tests).
    """
    # Check for verified results first
    verified_path = RESULTS_DIR / "swe_swe_27b_rerun_verified.jsonl"
    if verified_path.exists():
        lines = verified_path.read_text().strip().split("\n")
        results = [json.loads(l) for l in lines if l.strip()]
        if results:
            # Verified results are only for instances with patches
            # Total is 19 (all instances), resolved from verified
            resolved = sum(1 for r in results if r.get("resolved"))
            return {
                "resolve": resolved / 19,
                "mean": resolved / 19,
                "n": 19,
            }

    # Fallback to incremental results (last 19 entries = rerun)
    inc_path = BASE_DIR / "archive/overnight/logs/migrated_20260428/logs/swe_qwen3_5-27b_incremental.jsonl"
    if inc_path.exists():
        lines = inc_path.read_text().strip().split("\n")
        results = [json.loads(l) for l in lines[-19:] if l.strip()]
        if results:
            total = len(results)
            resolved = sum(1 for r in results if r.get("resolved"))
            return {
                "resolve": resolved / total if total else 0,
                "mean": resolved / total if total else 0,
                "n": total,
            }
    return None


def _latest_inc(bench: str, arm: str = None):
    """Find the latest incremental.jsonl for a bench in the v8 layout.

    v8 layout: RESULTS_DIR/<date>/<bench>/<experiment>/incremental.jsonl
    Falls back to flat scan (pre-v8 files) if no v8 dir found.
    """
    from unified_runner.base import find_experiments
    exps = find_experiments(RESULTS_DIR, bench=bench)
    if arm:
        exps = [e for e in exps if f"_{arm}" in e["experiment"] or e["experiment"].endswith(f"_{arm}")]
    if exps:
        # latest date first
        exps.sort(key=lambda e: (e["date"], e["experiment"]), reverse=True)
        return exps[0]["incremental"]
    # Fallback: flat layout
    patterns = [f"*_harbor_{bench}_27b_incremental.jsonl",
                f"*_{bench.replace('-', '_')}_27b_incremental.jsonl"]
    if bench == "swe":
        patterns = ["*_swe_27b_incremental.jsonl"]
    for pat in patterns:
        matches = sorted(RESULTS_DIR.glob(pat))
        if matches: return matches[-1]
    return None


def load_harbor_unified(dataset_tag):
    """Load Harbor unified results from latest incremental.jsonl."""
    inc_path = _latest_inc(dataset_tag, arm="baseline") or _latest_inc(dataset_tag)
    if not inc_path:
        return None
    lines = inc_path.read_text().strip().split("\n")
    results = [json.loads(l) for l in lines if l.strip()]
    if not results:
        return None
    total = len(results)
    resolved = sum(1 for r in results if r.get("resolved"))
    scores = [r.get("score", 0) for r in results]
    return {
        "resolve": resolved / total if total else 0,
        "mean": sum(scores) / total if total else 0,
        "n": total,
    }


def load_swe_unified():
    """Load SWE unified results from latest incremental.jsonl."""
    inc_path = _latest_inc("swe", arm="baseline") or _latest_inc("swe")
    if not inc_path:
        return None
    lines = inc_path.read_text().strip().split("\n")
    results = [json.loads(l) for l in lines if l.strip()]
    if not results:
        return None
    total = len(results)
    resolved = sum(1 for r in results if r.get("resolved"))
    return {
        "resolve": resolved / total if total else 0,
        "mean": resolved / total if total else 0,  # binary
        "n": total,
    }


def load_claw_unified():
    """Load Claw-Eval unified results from claw-eval batch output."""
    trace_dir = Path(
        str(BASE_DIR / "GeneralAgent/eval_scripts/claw_eval/traces")
    )
    # Find the unified trace directory
    for td in sorted(trace_dir.iterdir(), reverse=True):
        if "unified" in td.name:
            # Look for batch_results.json or batch_summary.json
            for subdir in sorted(td.iterdir(), reverse=True):
                summary = subdir / "batch_summary.json" if subdir.is_dir() else None
                results_f = subdir / "batch_results.json" if subdir.is_dir() else None
                if summary and summary.exists():
                    data = json.loads(summary.read_text())
                    total = data.get("tasks", 0)
                    if total > 0:
                        return {
                            "resolve": data.get("pass_hat_3", 0) / total if data.get("pass_hat_3") is not None else None,
                            "mean": data.get("avg_score", 0),
                            "n": total,
                        }
                if results_f and results_f.exists():
                    results = json.loads(results_f.read_text())
                    if results:
                        total = len(results)
                        resolved = sum(1 for r in results if r.get("passed", False))
                        scores = [r.get("task_score", 0) for r in results]
                        return {
                            "resolve": resolved / total if total else 0,
                            "mean": sum(scores) / total if total else 0,
                            "n": total,
                        }
            # Check directly under td
            summary = td / "batch_summary.json"
            results_f = td / "batch_results.json"
            if summary.exists():
                data = json.loads(summary.read_text())
                total = data.get("tasks", 0)
                if total > 0:
                    return {
                        "resolve": data.get("pass_hat_3", 0) / total if data.get("pass_hat_3") is not None else None,
                        "mean": data.get("avg_score", 0),
                        "n": total,
                    }
    return None


def fmt(val, width=5):
    """Format a float or None."""
    if val is None:
        return "?".center(width)
    return f"{val:.3f}"


def generate_table():
    """Generate the final comparison table."""

    # Load SWE original rerun
    swe_rerun = load_swe_original_rerun()
    if swe_rerun:
        ORIGINAL_27B["swe"]["resolve"] = swe_rerun["resolve"]
        ORIGINAL_27B["swe"]["mean"] = swe_rerun["mean"]

    # Load unified results
    unified = {
        "claw-eval": load_claw_unified(),
        "seta": load_harbor_unified("seta"),
        "skillsbench-ws": load_harbor_unified("skillsbench-with-skills"),
        "skillsbench-ns": load_harbor_unified("skillsbench-no-skills"),
        "swe": load_swe_unified(),
    }

    # Build table
    lines = []
    lines.append(f"# 27B Dual-Column Comparison (Generated {datetime.now().strftime('%Y-%m-%d %H:%M')})")
    lines.append("")
    lines.append("| Dataset (N) | Original resolve | Original mean | Unified resolve | Unified mean |")
    lines.append("|-------------|-----------------|---------------|-----------------|--------------|")

    rows = [
        ("Claw-Eval", "claw-eval"),
        ("SETA", "seta"),
        ("SkillsBench w/skills", "skillsbench-ws"),
        ("SkillsBench no-skills", "skillsbench-ns"),
        ("SWE", "swe"),
    ]

    for label, key in rows:
        orig = ORIGINAL_27B.get(key, {})
        uni = unified.get(key)
        n = orig.get("n", "?")
        o_r = fmt(orig.get("resolve"))
        o_m = fmt(orig.get("mean"))
        u_r = fmt(uni["resolve"] if uni else None)
        u_m = fmt(uni["mean"] if uni else None)
        if uni and uni.get("n") and uni["n"] != n:
            n = f"{n}/{uni['n']}"
        lines.append(f"| {label} ({n}) | {o_r} | {o_m} | {u_r} | {u_m} |")

    lines.append("")
    lines.append("# 14B Baseline (Original Framework)")
    lines.append("")
    lines.append("| Dataset | resolve | mean |")
    lines.append("|---------|---------|------|")

    for label, key in [
        ("Claw-Eval (161)", "claw-eval"),
        ("SkillsBench w/skills (88)", "skillsbench-ws"),
        ("SWE (19)", "swe"),
    ]:
        orig = ORIGINAL_14B.get(key, {})
        lines.append(f"| {label} | {fmt(orig.get('resolve'))} | {fmt(orig.get('mean'))} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Data Sources")
    lines.append("- **Original framework**: claw-eval CLI, harbor run (terminus-2), custom SWE agent")
    lines.append("- **Unified interface**: agent_loop.py + ToolLayer (OpenClaw deploy-tool subset)")
    lines.append("  - Claw-Eval: claw-eval CLI with local SGLang (Qwen XML tool calls require the deployment adapter for official OpenClaw)")
    lines.append("  - SkillsBench/SETA: run_unified_harbor.py (Docker + agent_loop)")
    lines.append("  - SWE: run_unified_swe.py (Docker + agent_loop)")

    return "\n".join(lines)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = generate_table()
    date_prefix = datetime.now().strftime("%Y%m%d")
    out_path = RESULTS_DIR / f"{date_prefix}_final_comparison_table.md"
    out_path.write_text(table)
    print(table)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
