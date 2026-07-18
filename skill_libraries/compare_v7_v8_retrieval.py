#!/usr/bin/env python3
"""Compare v7 (1143 skill) vs v8 (1652 skill) retrieval score distribution.

Run after rebuild_and_retrieve_v8.sh to quantify how much the skill library
expansion reduced <0.2 (no-match) rate and increased >0.5 (good-match) rate.
"""
import os
import json
from pathlib import Path
from collections import Counter

PROJ = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))
V7_DATE = "20260422"
# v8 date auto-detects the latest v8_3stage file
EXPERIMENTS = PROJ / "experiments"
LEGACY_RESULTS = PROJ / "archive/overnight/logs/migrated_20260428/logs/results"

BENCHES = ["skillsbench", "tb2", "seta", "claw", "swe"]
BUCKETS = [(0, 0.01, "<0.01"), (0.01, 0.05, ".01-.05"), (0.05, 0.10, ".05-.1"),
           (0.10, 0.20, ".1-.2"), (0.20, 0.50, ".2-.5"),
           (0.50, 0.80, ".5-.8"), (0.80, 1.01, ">0.8")]


def find_v8_path(bench):
    for date_dir in (sorted(EXPERIMENTS.iterdir(), reverse=True) if EXPERIMENTS.exists() else []):
        if not date_dir.is_dir() or not date_dir.name.startswith("20"):
            continue
        for run_dir in sorted(date_dir.iterdir(), reverse=True):
            if not run_dir.is_dir() or "v8_3stage" not in run_dir.name:
                continue
            p = run_dir / "retrieval_results" / f"{bench}.jsonl"
            if p.exists():
                return p
    for ddir in (sorted(LEGACY_RESULTS.iterdir(), reverse=True) if LEGACY_RESULTS.exists() else []):
        if not ddir.is_dir(): continue
        matches = list(ddir.glob(f"*_retrieval_{bench}_v8_3stage.jsonl"))
        if matches: return matches[0]
        # also handle nested subdate
        for sub in ddir.iterdir():
            if sub.is_dir():
                matches = list(sub.glob(f"*_retrieval_{bench}_v8_3stage.jsonl"))
                if matches: return matches[0]
    return None


def load_scores(path):
    if not path or not Path(path).exists(): return []
    out = []
    for l in open(path):
        r = json.loads(l)
        top = r.get("reranked_top10") or []
        s = top[0].get("rerank_score", 0) if top else 0
        out.append((r["task_id"], s))
    return out


def bucket_counts(scores):
    c = [0]*len(BUCKETS)
    for _, s in scores:
        for i, (lo, hi, _) in enumerate(BUCKETS):
            if lo <= s < hi: c[i] += 1; break
    return c


def main():
    print(f"{'bench':<12} {'N':>4}  {'  '.join(f'{b[2]:>7}' for b in BUCKETS)}")
    print("-"*90)
    totals_v7 = {"low": 0, "high": 0, "n": 0}
    totals_v8 = {"low": 0, "high": 0, "n": 0}
    for bench in BENCHES:
        v7_path = EXPERIMENTS / V7_DATE / f"{V7_DATE}_v7_3stage" / "retrieval_results" / f"{bench}.jsonl"
        if not v7_path.exists():
            v7_path = LEGACY_RESULTS / V7_DATE / f"{V7_DATE}_retrieval_{bench}_v7_3stage.jsonl"
        v8_path = find_v8_path(bench)
        s7 = load_scores(v7_path)
        s8 = load_scores(v8_path)
        if not s7:
            print(f"{bench:<12} SKIP missing v7"); continue
        c7 = bucket_counts(s7)
        low7 = sum(1 for _, s in s7 if s < 0.2)
        high7 = sum(1 for _, s in s7 if s > 0.5)
        totals_v7["n"] += len(s7); totals_v7["low"] += low7; totals_v7["high"] += high7
        print(f"v7 {bench:<9} {len(s7):>4}  {'  '.join(f'{x:>7}' for x in c7)}")
        if s8:
            c8 = bucket_counts(s8)
            low8 = sum(1 for _, s in s8 if s < 0.2)
            high8 = sum(1 for _, s in s8 if s > 0.5)
            totals_v8["n"] += len(s8); totals_v8["low"] += low8; totals_v8["high"] += high8
            print(f"v8 {bench:<9} {len(s8):>4}  {'  '.join(f'{x:>7}' for x in c8)}")
            # Per-task delta (top-1 score improvement)
            d7 = dict(s7); d8 = dict(s8)
            common = set(d7) & set(d8)
            if common:
                avg_delta = sum(d8[t] - d7[t] for t in common) / len(common)
                improved = sum(1 for t in common if d8[t] > d7[t] + 0.05)
                worsened = sum(1 for t in common if d8[t] < d7[t] - 0.05)
                print(f"   Δ: avg_score {avg_delta:+.3f}  improved {improved}  worsened {worsened}  (on {len(common)} matched)")
        else:
            print(f"v8 {bench:<9}  (not yet available — run rebuild_and_retrieve_v8.sh)")
        print()

    print("="*90)
    print(f"Totals:")
    print(f"  v7: N={totals_v7['n']}  low(<0.2)={totals_v7['low']} ({100*totals_v7['low']/max(1,totals_v7['n']):.1f}%)  "
          f"high(>0.5)={totals_v7['high']} ({100*totals_v7['high']/max(1,totals_v7['n']):.1f}%)")
    if totals_v8["n"]:
        print(f"  v8: N={totals_v8['n']}  low(<0.2)={totals_v8['low']} ({100*totals_v8['low']/totals_v8['n']:.1f}%)  "
              f"high(>0.5)={totals_v8['high']} ({100*totals_v8['high']/totals_v8['n']:.1f}%)")
        print(f"  Δ low: {totals_v8['low']-totals_v7['low']:+d}")
        print(f"  Δ high: {totals_v8['high']-totals_v7['high']:+d}")


if __name__ == "__main__":
    main()
