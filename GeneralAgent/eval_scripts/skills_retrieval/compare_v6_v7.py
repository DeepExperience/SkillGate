"""Compare v6 (20260420) retrieval vs v7 (today) retrieval on the same task set.

Usage:
    python compare_v6_v7.py --date 20260422

Reports per bench:
  (1) top-1 rerank score distribution comparison
  (2) meta-skill top-1 占比变化
  (3) # tasks where top-1 skill changed
  (4) sample cases: top-3 pre/post skill names + scores for 5 diverse tasks
"""
import os
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJ = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))
EXPERIMENTS = PROJ / "experiments"
LEGACY_RESULTS = PROJ / "archive/overnight/logs/migrated_20260428/logs/results"

META_SKILLS = {
    "coding-agent", "engineering-advanced-skills", "skill-vetter", "find-skills",
    "capability-evolver", "verification-before-completion", "opencode-controller",
    "safe-exec", "senior-prompt-engineer", "prompt-engineering-patterns",
    "prompt-engineering-expert", "senior-data-scientist", "skill-security-auditor",
    "senior-ml-engineer", "senior-security", "senior-secops", "deep-research-pro",
    "agent-team-orchestration", "agentic-eval", "brainstorming", "writing-plans",
    "executing-plans", "auto-memory-pro", "autoresearch-agent", "template-skill",
    "oracle", "debug-pro", "systematic-debugging", "add-educational-comments",
    "Code", "filesystem", "file-search", "exa-web-search-free", "docker-development",
    "docker-essentials", "dependency-auditor", "env-secrets-manager",
    "ci-cd-pipeline-builder", "performance-profiler",
}


def bin_score(s: float) -> str:
    if s < 0.1: return "<0.1"
    if s < 0.2: return "0.1-0.2"
    if s < 0.3: return "0.2-0.3"
    if s < 0.5: return "0.3-0.5"
    if s < 0.7: return "0.5-0.7"
    return ">=0.7"


def normalize_tid(s: str) -> str:
    return str(s).replace("_s_", "__")


def load(p: Path) -> dict[str, dict]:
    out = {}
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            out[normalize_tid(r["task_id"])] = r
    return out


def analyze(bench: str, old_path: Path, new_path: Path):
    if not old_path.exists():
        print(f"\n## {bench}: OLD missing ({old_path})"); return
    if not new_path.exists():
        print(f"\n## {bench}: NEW missing ({new_path})"); return
    old = load(old_path); new = load(new_path)
    shared = sorted(set(old) & set(new))
    print(f"\n{'='*70}\n## {bench}  (shared tasks: {len(shared)}; old total: {len(old)}; new total: {len(new)})")

    # (1) top-1 rerank score distribution
    old_bins, new_bins = Counter(), Counter()
    for tid in shared:
        t_o = (old[tid].get("reranked_top10") or [{}])[0].get("rerank_score", 0.0)
        t_n = (new[tid].get("reranked_top10") or [{}])[0].get("rerank_score", 0.0)
        old_bins[bin_score(t_o)] += 1
        new_bins[bin_score(t_n)] += 1
    print("\n### (1) top-1 rerank score distribution")
    print(f"  {'bin':<10}{'old':>6}  {'new':>6}  {'Δ':>6}")
    for k in ["<0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-0.7", ">=0.7"]:
        o = old_bins[k]; n = new_bins[k]; d = n - o
        print(f"  {k:<10}{o:>6}  {n:>6}  {d:>+6}")
    # lib-gap summary
    old_gap = old_bins["<0.1"] + old_bins["0.1-0.2"]
    new_gap = new_bins["<0.1"] + new_bins["0.1-0.2"]
    total = len(shared)
    print(f"  **lib-gap** (top-1<0.2): old {old_gap}/{total} ({old_gap/total*100:.1f}%) → new {new_gap}/{total} ({new_gap/total*100:.1f}%)  Δ {new_gap-old_gap:+d}")

    # (2) meta-skill top-1 占比
    old_meta = sum(1 for tid in shared if ((old[tid].get("reranked_top10") or [{}])[0].get("skill_name") in META_SKILLS))
    new_meta = sum(1 for tid in shared if ((new[tid].get("reranked_top10") or [{}])[0].get("skill_name") in META_SKILLS))
    print(f"\n### (2) meta-skill top-1 占比")
    print(f"  old {old_meta}/{total} ({old_meta/total*100:.1f}%) → new {new_meta}/{total} ({new_meta/total*100:.1f}%)  Δ {new_meta-old_meta:+d}")

    # (3) # tasks where top-1 changed
    changed = 0; same = 0
    for tid in shared:
        o = (old[tid].get("reranked_top10") or [{}])[0].get("skill_name")
        n = (new[tid].get("reranked_top10") or [{}])[0].get("skill_name")
        if o != n: changed += 1
        else: same += 1
    print(f"\n### (3) top-1 skill changed: {changed}/{total} ({changed/total*100:.1f}%) — same: {same}")

    # (4) sample cases: pick 5 diverse tasks
    #   priority: highest new top-1 score + (top-1 skill changed) → shows best improvements
    cases = []
    for tid in shared:
        o_top = old[tid].get("reranked_top10") or []
        n_top = new[tid].get("reranked_top10") or []
        if not o_top or not n_top: continue
        o1 = o_top[0]["rerank_score"]; n1 = n_top[0]["rerank_score"]
        desc = (new[tid].get("task_description") or "")[:120].replace("\n", " ")
        cases.append((n1 - o1, tid, o_top[:3], n_top[:3], desc))
    # sort by biggest positive improvement
    cases.sort(key=lambda c: -c[0])
    print(f"\n### (4) top-5 biggest rerank-score improvements (new − old top-1)")
    for delta, tid, o3, n3, desc in cases[:5]:
        print(f"\n- **{tid}**  Δtop1={delta:+.3f}")
        print(f"  desc: {desc}")
        old_line = ", ".join(f"{s['skill_name']}({s['rerank_score']:.2f})" for s in o3)
        new_line = ", ".join(f"{s['skill_name']}({s['rerank_score']:.2f})" for s in n3)
        print(f"  OLD top-3: {old_line}")
        print(f"  NEW top-3: {new_line}")


def retrieval_file(date: str, suffix: str, bench: str) -> Path:
    new_path = EXPERIMENTS / date / f"{date}_{suffix}" / "retrieval_results" / f"{bench}.jsonl"
    if new_path.exists():
        return new_path
    return LEGACY_RESULTS / date / f"{date}_retrieval_{bench}_{suffix}.jsonl"

    # and top-5 worst regressions
    print(f"\n### (5) top-5 biggest regressions (new − old top-1)")
    for delta, tid, o3, n3, desc in cases[-5:]:
        print(f"\n- **{tid}**  Δtop1={delta:+.3f}")
        print(f"  desc: {desc}")
        old_line = ", ".join(f"{s['skill_name']}({s['rerank_score']:.2f})" for s in o3)
        new_line = ", ".join(f"{s['skill_name']}({s['rerank_score']:.2f})" for s in n3)
        print(f"  OLD top-3: {old_line}")
        print(f"  NEW top-3: {new_line}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD of new (v7) run")
    ap.add_argument("--new-suffix", default="v7_3stage")
    ap.add_argument("--old-date", default="20260420")
    ap.add_argument("--old-suffix", default="v6_3stage")
    args = ap.parse_args()

    BENCHES = ["skillsbench", "seta", "tb2", "swe", "claw"]
    for b in BENCHES:
        old_p = retrieval_file(args.old_date, args.old_suffix, b)
        new_p = retrieval_file(args.date, args.new_suffix, b)
        analyze(b, old_p, new_p)


if __name__ == "__main__":
    main()
