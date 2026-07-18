#!/usr/bin/env python3
"""
Ablation comparison: V7 pipeline fixed, only skill library varies (1143 → 1651).

Compares reranker top-10 score distributions between:
  OLD: experiments/20260422/20260422_v7_3stage/retrieval_results/<bench>.jsonl   (1143 skill)
  NEW: experiments/<date>/<date>_<suffix>/retrieval_results/<bench>.jsonl (new skill library)

Metrics (the key deliverable: "is distribution shifted up?"):
  - Buckets of top-1 rerank_score across all tasks per bench
  - Buckets of top-10 mean rerank_score
  - Fraction of tasks with "low-score tail" (any top-10 skill with score < 0.1)
  - Per-task Δ of top-1, top-5 median, top-10 mean
  - Which tasks improved / regressed the most

Usage:
  python compare_skill_lib_ablation.py 20260423 v7pipeline_on_v8lib
"""
import os
import json
import sys
from pathlib import Path
from collections import Counter

PROJ = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))
EXPERIMENTS = PROJ / "experiments"
LEGACY_RESULTS = PROJ / "archive/overnight/logs/migrated_20260428/logs/results"

BENCHES = ["skillsbench", "tb2", "seta", "claw", "swe"]
BUCKETS = [
    (0, 0.01, "<0.01"), (0.01, 0.05, ".01-.05"), (0.05, 0.10, ".05-.1"),
    (0.10, 0.20, ".1-.2"), (0.20, 0.50, ".2-.5"),
    (0.50, 0.80, ".5-.8"), (0.80, 1.01, ">0.8"),
]


def load(p):
    if not p.exists(): return {}
    return {json.loads(l)["task_id"]: json.loads(l) for l in open(p)}


def retrieval_dir(date: str, suffix: str) -> Path:
    """Prefer experiments/<date>/<date>_<suffix>/retrieval_results, fallback to legacy."""
    new_dir = EXPERIMENTS / date / f"{date}_{suffix}" / "retrieval_results"
    if new_dir.exists():
        return new_dir
    return LEGACY_RESULTS / date / "retrieval_results" / suffix


def top_scores(rec):
    """Return list of rerank_score from reranked_top10, padded to 10 with 0s."""
    top = rec.get("reranked_top10") or []
    return [x.get("rerank_score", 0) for x in top][:10] + [0] * max(0, 10 - len(top))


def bucket_counts(scores):
    c = [0] * len(BUCKETS)
    for s in scores:
        for i, (lo, hi, _) in enumerate(BUCKETS):
            if lo <= s < hi:
                c[i] += 1; break
    return c


def fmt_buckets(c, total):
    return "  ".join(f"{b[2]}={c[i]:>3}" for i, b in enumerate(BUCKETS))


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else "20260423"
    suffix = sys.argv[2] if len(sys.argv) > 2 else "v7pipeline_on_v8lib"
    OLD_DIR = retrieval_dir("20260422", "v7_3stage")
    NEW_DIR = retrieval_dir(date, suffix)

    print(f"OLD  (1143 skill): {OLD_DIR}")
    print(f"NEW  (1651 skill): {NEW_DIR}")
    print()

    totals = {"old_tasks": 0, "new_tasks": 0,
              "old_low_tail": 0, "new_low_tail": 0,
              "old_top1_sum": 0.0, "new_top1_sum": 0.0,
              "old_top10_mean_sum": 0.0, "new_top10_mean_sum": 0.0}

    for bench in BENCHES:
        old = load(OLD_DIR / f"{bench}.jsonl")
        new = load(NEW_DIR / f"{bench}.jsonl")
        common = sorted(set(old) & set(new))
        if not common:
            print(f"== {bench}: no common tasks (old={len(old)} new={len(new)}) ==\n")
            continue

        print(f"{'='*80}\n=== {bench.upper()}  common N={len(common)}  old_N={len(old)}  new_N={len(new)} ===\n{'='*80}")

        # Top-1 score buckets
        old_top1 = [top_scores(old[t])[0] for t in common]
        new_top1 = [top_scores(new[t])[0] for t in common]
        print(f"\nTop-1 rerank_score buckets:")
        print(f"  OLD:  " + fmt_buckets(bucket_counts(old_top1), len(common)))
        print(f"  NEW:  " + fmt_buckets(bucket_counts(new_top1), len(common)))

        # Mean/median
        old_t1_avg = sum(old_top1)/len(old_top1)
        new_t1_avg = sum(new_top1)/len(new_top1)
        old_t1_med = sorted(old_top1)[len(old_top1)//2]
        new_t1_med = sorted(new_top1)[len(new_top1)//2]
        print(f"  avg top-1:    OLD={old_t1_avg:.3f}  NEW={new_t1_avg:.3f}  Δ{new_t1_avg-old_t1_avg:+.3f}")
        print(f"  median top-1: OLD={old_t1_med:.3f}  NEW={new_t1_med:.3f}  Δ{new_t1_med-old_t1_med:+.3f}")

        # Top-10 mean per task
        old_t10_means = [sum(top_scores(old[t]))/10 for t in common]
        new_t10_means = [sum(top_scores(new[t]))/10 for t in common]
        old_t10_avg = sum(old_t10_means)/len(old_t10_means)
        new_t10_avg = sum(new_t10_means)/len(new_t10_means)
        print(f"  avg top-10 mean: OLD={old_t10_avg:.3f}  NEW={new_t10_avg:.3f}  Δ{new_t10_avg-old_t10_avg:+.3f}")

        # Low-score tail: any of top-10 < 0.1
        old_low = sum(1 for t in common if min(top_scores(old[t])) < 0.1)
        new_low = sum(1 for t in common if min(top_scores(new[t])) < 0.1)
        print(f"  tasks with any top-10 < 0.1:  OLD={old_low}/{len(common)} ({100*old_low/len(common):.0f}%)  "
              f"NEW={new_low}/{len(common)} ({100*new_low/len(common):.0f}%)  Δ{new_low-old_low:+d}")

        # Count "library gap" = top-1 below 0.2
        old_gap = sum(1 for s in old_top1 if s < 0.2)
        new_gap = sum(1 for s in new_top1 if s < 0.2)
        print(f"  'library gap' (top-1<0.2):   OLD={old_gap}/{len(common)} ({100*old_gap/len(common):.0f}%)  "
              f"NEW={new_gap}/{len(common)} ({100*new_gap/len(common):.0f}%)  Δ{new_gap-old_gap:+d}")

        # Biggest improvements per task (Δ top-1)
        deltas = [(t, top_scores(new[t])[0] - top_scores(old[t])[0]) for t in common]
        deltas.sort(key=lambda x: -x[1])
        print(f"\n  Top-5 biggest IMPROVEMENTS (top-1 rerank_score increase):")
        for t, d in deltas[:5]:
            if d <= 0: break
            n_name = new[t].get("reranked_top10", [{}])[0].get("skill_name", "?")
            o_name = old[t].get("reranked_top10", [{}])[0].get("skill_name", "?")
            print(f"    {t:<40} Δ={d:+.3f}   old_top1='{o_name}' → new_top1='{n_name}'")

        print(f"\n  Top-5 biggest REGRESSIONS (top-1 rerank_score decrease):")
        for t, d in sorted(deltas, key=lambda x: x[1])[:5]:
            if d >= 0: break
            n_name = new[t].get("reranked_top10", [{}])[0].get("skill_name", "?")
            o_name = old[t].get("reranked_top10", [{}])[0].get("skill_name", "?")
            print(f"    {t:<40} Δ={d:+.3f}   old_top1='{o_name}' → new_top1='{n_name}'")
        print()

        totals["old_tasks"] += len(common)
        totals["new_tasks"] += len(common)
        totals["old_low_tail"] += old_low
        totals["new_low_tail"] += new_low
        totals["old_top1_sum"] += sum(old_top1)
        totals["new_top1_sum"] += sum(new_top1)
        totals["old_top10_mean_sum"] += sum(old_t10_means)
        totals["new_top10_mean_sum"] += sum(new_t10_means)

    # Grand totals
    n = totals["old_tasks"] or 1
    print("="*80)
    print(f"GRAND TOTAL across {n} tasks")
    print("="*80)
    print(f"  avg top-1 rerank_score:    OLD={totals['old_top1_sum']/n:.3f}  "
          f"NEW={totals['new_top1_sum']/n:.3f}  Δ{(totals['new_top1_sum']-totals['old_top1_sum'])/n:+.3f}")
    print(f"  avg top-10 mean:           OLD={totals['old_top10_mean_sum']/n:.3f}  "
          f"NEW={totals['new_top10_mean_sum']/n:.3f}  Δ{(totals['new_top10_mean_sum']-totals['old_top10_mean_sum'])/n:+.3f}")
    print(f"  tasks with any top-10<0.1: OLD={totals['old_low_tail']}/{n}  NEW={totals['new_low_tail']}/{n}  "
          f"Δ{totals['new_low_tail']-totals['old_low_tail']:+d}")
    print()
    print("Decision rule:")
    print("  - Proceed to eval arm only if NEW > OLD on avg top-1 AND low-tail shrinks")
    print("  - Else: expand skill lib further / fix retrieval before burning eval hours")


if __name__ == "__main__":
    main()
