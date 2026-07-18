#!/usr/bin/env python3
"""Aggregate results across all benches (unified_runner + native) into one table.

Each bench has a different result format; this script knows them all and
normalizes to (N_total, N_completed, N_pass, pass_rate, mean_score).

Primary metrics by bench (pick 1-2 that are comparable to "pass@1"):
  - SkillsBench (unified & native harbor):
      resolve_rate = #(reward==1.0) / N      -- single-trial pass rate
      mean_score   = mean(reward in [0,1])   -- partial credit across tests
  - SETA (unified & native harbor):
      same as SkillsBench
  - TB 2.0 (unified & native harbor):
      same as SkillsBench
  - SWE (unified & native): single-trial reward is 0/1
      resolve_rate = #(resolved) / N
      mean_score   = resolve_rate (no partial credit for SWE)
  - Claw-Eval (native):
      pass_hat_3   = #(all-3-passed) / N_tasks        -- strict pass@3 (claw's primary)
      pass_at_3    = #(any-of-3-passed) / N_tasks     -- loose pass@3
      avg_score    = mean per-trial reward            -- ~ pass@1
  - Claw-Eval (unified, single-trial agent_loop):
      resolve_rate = #(resolved) / N
      mean_score   = mean(score)

Usage:
    python aggregate_results.py --date 20260417
    python aggregate_results.py --date 20260417 --out /tmp/final.md
"""
import argparse
import json
import re
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(os.environ.get("SKILLRL_ROOT", str(Path(__file__).resolve().parents[3])))
UNIFIED_RESULTS_DIR = BASE / "experiments"
NATIVE_ROOTS = {
    "skillsbench": BASE / "GeneralAgent/eval_scripts/skillsbench_eval/results",
    "seta":        BASE / "GeneralAgent/eval_scripts/seta_eval/results",
    "tb2":         BASE / "GeneralAgent/eval_scripts/tb2_eval/results",
    "swe":         BASE / "GeneralAgent/eval_scripts/swe_gym_eval/results",
    "claw":        BASE / "GeneralAgent/eval_scripts/claw_eval/traces",
}


# ---- Unified runner JSONL parsing --------------------------------------------

def read_unified_jsonl(path):
    """Parse unified_runner incremental .jsonl files.

    Each line has: task_id/instance_id, resolved (bool), score (float 0-1),
    error (str; empty if ok), turns, time_sec.
    """
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def summarize_unified(rows, name):
    """Return dict with N_total, N_completed (no error), N_pass, pass_rate, mean_score."""
    total = len(rows)
    completed = sum(1 for r in rows if not r.get("error"))
    errored = total - completed
    # 'resolved' is bool; 'score' 0-1
    n_pass = sum(1 for r in rows if r.get("resolved"))
    mean_score = sum(float(r.get("score", 0.0)) for r in rows) / total if total else 0.0
    return {
        "bench": name,
        "N_total": total,
        "N_completed": completed,
        "N_errored": errored,
        "N_pass": n_pass,
        "pass_rate": n_pass / total if total else 0.0,
        "mean_score": mean_score,
    }


# ---- Native harbor (SkillsBench/SETA/TB2) parsing ----------------------------

def read_harbor_job(job_dir):
    """Walk a harbor job-name dir, reading per-trial result.json.

    Returns list of {task_id, reward, resolved, error}.
    """
    rows = []
    if not job_dir.exists():
        return rows
    for trial_dir in sorted(job_dir.iterdir()):
        if not trial_dir.is_dir() or trial_dir.name.startswith("."):
            continue
        result_json = trial_dir / "result.json"
        if not result_json.exists():
            rows.append({"task_id": trial_dir.name, "reward": None,
                         "resolved": False, "error": "no result.json"})
            continue
        try:
            data = json.loads(result_json.read_text())
        except Exception as e:
            rows.append({"task_id": trial_dir.name, "reward": None,
                         "resolved": False, "error": f"parse err: {e}"})
            continue
        # reward nested under verifier_result.rewards.reward
        reward = None
        try:
            reward = float(data["verifier_result"]["rewards"]["reward"])
        except (KeyError, TypeError, ValueError):
            pass
        exc = data.get("exception_info")
        err = "" if exc is None else str(exc)[:200]
        rows.append({
            "task_id": data.get("task_name", trial_dir.name),
            "reward": reward,
            "resolved": (reward == 1.0) if reward is not None else False,
            "error": err,
        })
    return rows


def summarize_harbor(rows, name):
    total = len(rows)
    completed = sum(1 for r in rows if not r["error"] and r["reward"] is not None)
    errored = total - completed
    n_pass = sum(1 for r in rows if r["resolved"])
    mean_score = sum((r["reward"] or 0.0) for r in rows) / total if total else 0.0
    return {
        "bench": name,
        "N_total": total,
        "N_completed": completed,
        "N_errored": errored,
        "N_pass": n_pass,
        "pass_rate": n_pass / total if total else 0.0,
        "mean_score": mean_score,
    }


# ---- Claw-Eval native parsing -----------------------------------------------

def read_claw_native(traces_dir):
    """Return summary dict from batch_summary.json or None."""
    summary = traces_dir / "batch_summary.json"
    if not summary.exists():
        return None
    data = json.loads(summary.read_text())
    tasks = data.get("tasks", 0)
    return {
        "bench": f"claw-native:{traces_dir.name}",
        "N_total": tasks,
        "N_completed": tasks - data.get("errored", 0),
        "N_errored": data.get("errored", 0),
        "N_pass": data.get("pass_hat_3", 0),         # strict pass@3 (all 3 pass)
        "pass_rate": (data.get("pass_hat_3", 0) / tasks) if tasks else 0.0,
        "mean_score": data.get("avg_score", 0.0),
        # extra fields for claw specifically:
        "_claw_pass_at_3": data.get("pass_at_3", 0),
        "_claw_trials_per_task": data.get("trials_per_task", 3),
    }


# ---- SWE-Gym native parsing --------------------------------------------------

def read_swe_native(results_dir):
    """Parse native SWE summary. Format TBD — assume jsonl similar to unified."""
    for jsonl in sorted(results_dir.glob("*_incremental.jsonl")):
        rows = read_unified_jsonl(jsonl)
        if rows:
            return summarize_unified(rows, f"swe-native:{jsonl.name}")
    return None


# ---- Entry points ------------------------------------------------------------

def find_unified_files(date_prefix):
    """Resolve v8 layout paths per bench (falls back to flat for legacy dates).

    v8:   results/<date>/<bench>/<experiment>/incremental.jsonl
    flat: results/<date>_...<bench>_27b_incremental.jsonl (pre-2026-04-22)
    """
    from unified_runner.base import find_experiments
    d = UNIFIED_RESULTS_DIR
    bench_map = {
        "unified-skillsbench-with": "skillsbench-with-skills",
        "unified-skillsbench-no":   "skillsbench-no-skills",
        "unified-seta":             "seta",
        "unified-tb2":              "tb2",
        "unified-swe":              "swe",
    }
    patterns = {}
    for label, bench in bench_map.items():
        exps = find_experiments(d, date=date_prefix, bench=bench)
        exps = [e for e in exps if "baseline" in e["experiment"]] or exps
        if exps:
            # Return path relative to date dir so legacy code still joins correctly
            patterns[label] = exps[0]["incremental"].relative_to(d / date_prefix)
        else:
            # Legacy flat fallback
            if bench.startswith("skillsbench"):
                patterns[label] = f"{date_prefix}_harbor_{bench}_27b_incremental.jsonl"
            elif bench == "swe":
                patterns[label] = f"{date_prefix}_swe_27b_incremental.jsonl"
            else:
                patterns[label] = f"{date_prefix}_harbor_{bench}_27b_incremental.jsonl"

    # claw: v8 layout → <date>/claw/<experiment>/incremental.jsonl; legacy → *_claw_*.jsonl
    claw_exps = find_experiments(d, date=date_prefix, bench="claw")
    if claw_exps:
        claw_files = [e["incremental"] for e in claw_exps]
    else:
        claw_files = sorted(d.glob(f"{date_prefix}/{date_prefix}_claw_*.jsonl")) or \
                     sorted(d.glob(f"{date_prefix}_claw_*.jsonl"))
    return patterns, claw_files


def render_table(rows):
    hdr = ["Bench", "N_total", "N_completed", "N_errored", "N_pass", "pass_rate", "mean_score"]
    lines = ["| " + " | ".join(hdr) + " |",
             "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join([
            r["bench"],
            str(r["N_total"]),
            str(r["N_completed"]),
            str(r["N_errored"]),
            str(r["N_pass"]),
            f"{r['pass_rate']:.3f}",
            f"{r['mean_score']:.3f}",
        ]) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"),
                    help="Date prefix for unified jsonl files (default: today)")
    ap.add_argument("--native-skillsbench-job", default="",
                    help="Native job-name under skillsbench_eval/results/ (optional)")
    ap.add_argument("--native-seta-job", default="",
                    help="Native job-name under seta_eval/results/ (optional)")
    ap.add_argument("--native-tb2-job", default="",
                    help="Native job-name under tb2_eval/results/ (optional)")
    ap.add_argument("--native-swe-job", default="",
                    help="Native job-name under swe_gym_eval/results/ (optional)")
    ap.add_argument("--native-claw-job", default="",
                    help="Native job-name under claw_eval/traces/ (optional)")
    ap.add_argument("--out", default="", help="Output .md file (default: stdout)")
    args = ap.parse_args()

    rows = []

    # --- Unified runner ---
    unified, claw_files = find_unified_files(args.date)
    for name, fname in unified.items():
        path = UNIFIED_RESULTS_DIR / fname
        if path.exists():
            data = read_unified_jsonl(path)
            rows.append(summarize_unified(data, name))
    for cf in claw_files:
        data = read_unified_jsonl(cf)
        rows.append(summarize_unified(data, f"unified-claw:{cf.name}"))

    # --- Native harbor benches ---
    for bench, root in [("skillsbench", NATIVE_ROOTS["skillsbench"]),
                        ("seta",        NATIVE_ROOTS["seta"]),
                        ("tb2",         NATIVE_ROOTS["tb2"])]:
        job = getattr(args, f"native_{bench}_job", "")
        if not job:
            continue
        job_dir = root / job
        if not job_dir.exists():
            print(f"WARN: native {bench} job-dir not found: {job_dir}", file=sys.stderr)
            continue
        data = read_harbor_job(job_dir)
        rows.append(summarize_harbor(data, f"native-{bench}:{job}"))

    # --- Native Claw ---
    if args.native_claw_job:
        cdir = NATIVE_ROOTS["claw"] / args.native_claw_job
        s = read_claw_native(cdir)
        if s:
            rows.append(s)

    # --- Native SWE ---
    if args.native_swe_job:
        sdir = NATIVE_ROOTS["swe"] / args.native_swe_job
        s = read_swe_native(sdir) if sdir.exists() else None
        if s:
            rows.append(s)

    # --- Render ---
    out = []
    out.append(f"# Aggregated Eval Results (date={args.date})")
    out.append(f"_generated at {datetime.now():%Y-%m-%d %H:%M}_")
    out.append("")
    out.append("## Metric glossary")
    out.append("- **N_total**: total tasks/instances attempted")
    out.append("- **N_completed**: tasks whose verifier produced a score (no pipeline error)")
    out.append("- **N_errored**: N_total - N_completed (docker build fail, agent crash, timeout, etc.)")
    out.append("- **N_pass**: reward==1.0 (harbor benches) or resolved==True (SWE/Claw)")
    out.append("- **pass_rate** = N_pass / N_total   — 粗粒度 pass@1 / resolve rate")
    out.append("- **mean_score** = avg(reward in [0,1])  — 细粒度分，harbor 里是 test.sh 给的 partial credit")
    out.append("")
    out.append("## Table")
    if rows:
        out.append(render_table(rows))
    else:
        out.append("_(no results found)_")
    text = "\n".join(out) + "\n"

    if args.out:
        Path(args.out).write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
