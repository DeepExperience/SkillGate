"""Failure attribution for retrieval arm: tb2 / seta / swe.

For each bench:
1. Load result jsonls (baseline / retrieval / irrelevant), dedup by task_id (keep last row).
2. Load v6_3stage retrieval jsonl to get reranked_top10 per task.
3. For each task compute PASS = resolved or score >= 0.75.
4. Partition:
   retrieval_hurt: baseline pass, retrieval fail
   retrieval_helped: baseline fail, retrieval pass
   both_pass / both_fail
5. For each retrieval_hurt case: print top-5 rerank skills + scores.
6. Bin top-1 rerank score to categorize likely cause:
   <0.2: lib-gap (reranker says nothing relevant)
   0.2-0.5: weak (meta skills likely chosen)
   >=0.5: strong (skill is relevant but didn't help / hurt / noise)

Pass metric aligned with daily report: resolved OR score>=0.75.
"""
import os
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJ = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))
EXPERIMENTS = PROJ / "experiments"

BENCHES = {
    "tb2": {
        "baseline": EXPERIMENTS / "20260420/20260420_v6_baseline/results/tb2/v6_baseline/incremental.jsonl",
        "retrieval": EXPERIMENTS / "20260420/20260420_v6_retrieval/results/tb2/v6_retrieval/incremental.jsonl",
        "irrelevant": EXPERIMENTS / "20260420/20260420_v6_irrelevant/results/tb2/v6_irrelevant/incremental.jsonl",
        "retrieval_meta": EXPERIMENTS / "20260420/20260420_v6_3stage/retrieval_results/tb2.jsonl",
    },
    "seta": {
        "baseline": EXPERIMENTS / "20260420/20260420_v6_baseline/results/seta/v6_baseline/incremental.jsonl",
        "retrieval": EXPERIMENTS / "20260420/20260420_v6_retrieval/results/seta/v6_retrieval/incremental.jsonl",
        "irrelevant": EXPERIMENTS / "20260420/20260420_v6_irrelevant/results/seta/v6_irrelevant/incremental.jsonl",
        "retrieval_meta": EXPERIMENTS / "20260420/20260420_v6_3stage/retrieval_results/seta.jsonl",
    },
    "swe": {
        "baseline": EXPERIMENTS / "20260420/20260420_v6_baseline/results/swe/v6_baseline/incremental.jsonl",
        "retrieval": EXPERIMENTS / "20260420/20260420_v6_retrieval/results/swe/v6_retrieval/incremental.jsonl",
        "irrelevant": EXPERIMENTS / "20260420/20260420_v6_irrelevant/results/swe/v6_irrelevant/incremental.jsonl",
        "retrieval_meta": EXPERIMENTS / "20260420/20260420_v6_3stage/retrieval_results/swe.jsonl",
    },
}


def is_pass(row):
    if row.get("resolved"):
        return True
    sc = row.get("score")
    if sc is None:
        return False
    return sc >= 0.75


def load_results(path):
    """Return {task_id: latest row}."""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            tid = r.get("task_id") or r.get("instance_id")
            out[str(tid)] = r
    return out


def normalize_tid(s):
    return str(s).replace("_s_", "__")


def load_retrieval_meta(path):
    """Return {task_id: retrieval_row}."""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[normalize_tid(r["task_id"])] = r
    return out


for bench, files in BENCHES.items():
    print(f"\n{'='*80}\n# {bench.upper()}\n{'='*80}")
    bl = load_results(files["baseline"])
    rt = load_results(files["retrieval"])
    ir = load_results(files["irrelevant"])
    meta = load_retrieval_meta(files["retrieval_meta"])

    # use retrieval arm's tasks as reference (all 89 for tb2 / 30 for seta / 19 for swe)
    all_tids = set(rt.keys())
    print(f"tasks (retrieval arm): {len(all_tids)}")
    print(f"baseline coverage: {len(all_tids & set(bl.keys()))}, "
          f"irrelevant coverage: {len(all_tids & set(ir.keys()))}, "
          f"meta coverage: {len(all_tids & set(meta.keys()))}")

    # partitions
    parts = defaultdict(list)
    for tid in sorted(all_tids):
        b_row = bl.get(tid)
        r_row = rt.get(tid)
        i_row = ir.get(tid)
        b = is_pass(b_row) if b_row else None
        r = is_pass(r_row)
        i = is_pass(i_row) if i_row else None
        if b is None:
            continue
        key = (b, r)
        parts[key].append(tid)

    print(f"\npartition (baseline, retrieval):")
    print(f"  retrieval_hurt  (T,F): {len(parts[(True, False)])}")
    print(f"  retrieval_helped(F,T): {len(parts[(False, True)])}")
    print(f"  both pass (T,T): {len(parts[(True, True)])}")
    print(f"  both fail (F,F): {len(parts[(False, False)])}")

    # analyze retrieval_hurt
    print(f"\n## retrieval_hurt (baseline pass, retrieval fail)")
    hurt = parts[(True, False)]
    cause_counts = Counter()
    low_score_cases = []
    mid_score_cases = []
    high_score_cases = []
    for tid in hurt:
        m = meta.get(tid)
        if not m:
            print(f"  [NO META] {tid}")
            continue
        top = m.get("reranked_top10", [])[:5]
        top1 = top[0]["rerank_score"] if top else 0.0
        if top1 < 0.2:
            cause_counts["lib_gap(<0.2)"] += 1
            low_score_cases.append((tid, top1, top))
        elif top1 < 0.5:
            cause_counts["weak(0.2-0.5)"] += 1
            mid_score_cases.append((tid, top1, top))
        else:
            cause_counts["strong(>=0.5)"] += 1
            high_score_cases.append((tid, top1, top))

    print(f"cause_counts: {dict(cause_counts)}")

    def fmt_top(top):
        return ", ".join(f"{s['skill_name']}({s['rerank_score']:.2f})" for s in top)

    print(f"\n### LIB-GAP cases (top-1 rerank<0.2, reranker says nothing relevant):")
    for tid, sc, top in low_score_cases:
        r_row = rt.get(tid, {})
        task_desc = meta.get(tid, {}).get("task_description", "")[:120].replace("\n", " ")
        print(f"  - [{tid}] top1={sc:.3f}")
        print(f"      desc: {task_desc}")
        print(f"      top5: {fmt_top(top)}")
        print(f"      finish: {r_row.get('finish_reason', '?')} / turns={r_row.get('turns','?')} / err={(r_row.get('error') or '')[:80]}")

    print(f"\n### WEAK cases (top-1 rerank 0.2-0.5):")
    for tid, sc, top in mid_score_cases:
        r_row = rt.get(tid, {})
        task_desc = meta.get(tid, {}).get("task_description", "")[:120].replace("\n", " ")
        print(f"  - [{tid}] top1={sc:.3f}")
        print(f"      desc: {task_desc}")
        print(f"      top5: {fmt_top(top)}")
        print(f"      finish: {r_row.get('finish_reason', '?')} / turns={r_row.get('turns','?')} / err={(r_row.get('error') or '')[:80]}")

    print(f"\n### STRONG cases (top-1 rerank>=0.5, skill exists but didn't help):")
    for tid, sc, top in high_score_cases:
        r_row = rt.get(tid, {})
        task_desc = meta.get(tid, {}).get("task_description", "")[:120].replace("\n", " ")
        print(f"  - [{tid}] top1={sc:.3f}")
        print(f"      desc: {task_desc}")
        print(f"      top5: {fmt_top(top)}")
        print(f"      finish: {r_row.get('finish_reason', '?')} / turns={r_row.get('turns','?')} / err={(r_row.get('error') or '')[:80]}")

    # also look at retrieval_helped to understand what works
    helped = parts[(False, True)]
    print(f"\n## retrieval_helped (baseline fail, retrieval pass): {len(helped)} cases")
    for tid in helped:
        m = meta.get(tid)
        if not m:
            continue
        top = m.get("reranked_top10", [])[:5]
        top1 = top[0]["rerank_score"] if top else 0.0
        task_desc = m.get("task_description", "")[:120].replace("\n", " ")
        print(f"  - [{tid}] top1={top1:.3f}")
        print(f"      desc: {task_desc}")
        print(f"      top5: {fmt_top(top)}")

    # top-1 score distribution across ALL tasks in this bench
    print(f"\n## top-1 rerank score distribution (all {len(all_tids)} tasks):")
    bins = Counter()
    for tid in all_tids:
        m = meta.get(tid)
        if not m:
            continue
        top = m.get("reranked_top10", [])
        top1 = top[0]["rerank_score"] if top else 0.0
        if top1 < 0.1: bins["<0.1"] += 1
        elif top1 < 0.2: bins["0.1-0.2"] += 1
        elif top1 < 0.3: bins["0.2-0.3"] += 1
        elif top1 < 0.5: bins["0.3-0.5"] += 1
        elif top1 < 0.7: bins["0.5-0.7"] += 1
        else: bins[">=0.7"] += 1
    for k in ["<0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-0.7", ">=0.7"]:
        print(f"  {k}: {bins[k]}")
