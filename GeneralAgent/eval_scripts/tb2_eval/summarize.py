#!/usr/bin/env python3
"""Summarize a tb2_eval harbor run: pass@1, token usage, failures, per-category.

Usage: python summarize.py <results_dir>
"""
import os
import json
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

TB2_DIR = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL")) / "datasets/terminal-bench-v2"


def load_task_categories() -> dict[str, str]:
    cats = {}
    for d in TB2_DIR.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        tt = d / "task.toml"
        if not tt.exists():
            continue
        try:
            meta = tomllib.loads(tt.read_text())
            cats[d.name] = (meta.get("metadata") or {}).get("category", "?")
        except Exception:
            cats[d.name] = "?"
    return cats


def main(job_dir: Path) -> None:
    trials = [p for p in job_dir.iterdir() if p.is_dir() and (p / "result.json").exists()]
    if not trials:
        print(f"no trial dirs under {job_dir}")
        return

    cats = load_task_categories()

    rewards: list[float] = []
    by_cat: dict[str, list[float]] = defaultdict(list)
    tokens_in = tokens_out = 0
    exc: Counter[str] = Counter()
    failed_tasks: list[str] = []
    timed_out: list[str] = []

    for t in trials:
        r = json.loads((t / "result.json").read_text())
        task = r.get("task_name") or t.name.split("__")[0]
        reward = (r.get("verifier_result") or {}).get("rewards", {}).get("reward")
        ar = r.get("agent_result") or {}
        tokens_in += ar.get("n_input_tokens", 0) or 0
        tokens_out += ar.get("n_output_tokens", 0) or 0
        einfo = r.get("exception_info") or {}
        if einfo:
            exc[einfo.get("type", "?")] += 1
            if "Timeout" in (einfo.get("type") or ""):
                timed_out.append(task)
        if reward is None:
            reward = 0.0
        rewards.append(reward)
        by_cat[cats.get(task, "?")].append(reward)
        if reward < 1.0:
            failed_tasks.append(task)

    n = len(rewards)
    mean = sum(rewards) / n if n else 0.0

    print(f"=== {job_dir.name} ===")
    print(f"  trials    : {n}")
    print(f"  pass@1    : {mean:.4f}  ({sum(1 for r in rewards if r >= 1.0)}/{n})")
    print(f"  tokens    : {tokens_in:,} in / {tokens_out:,} out")
    print()
    print("  per-category pass@1:")
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        print(f"    {cat:20s}  {sum(rs)/len(rs):.3f}  ({sum(1 for r in rs if r>=1.0)}/{len(rs)})")
    if exc:
        print()
        print("  exceptions:")
        for k, v in exc.most_common():
            print(f"    {k:30s}  {v}")
    if timed_out:
        print()
        print(f"  timed out ({len(timed_out)}):")
        for t in sorted(timed_out):
            print(f"    {t}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: summarize.py <results_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]))
