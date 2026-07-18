"""Deeper dive: both_fail tasks + meta-skill pollution + lib-gap categorization."""
import os
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJ = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))
EXPERIMENTS = PROJ / "experiments"

BENCHES = {
    "tb2": (EXPERIMENTS / "20260420/20260420_v6_baseline/results/tb2/v6_baseline/incremental.jsonl",
            EXPERIMENTS / "20260420/20260420_v6_retrieval/results/tb2/v6_retrieval/incremental.jsonl",
            EXPERIMENTS / "20260420/20260420_v6_3stage/retrieval_results/tb2.jsonl"),
    "seta": (EXPERIMENTS / "20260420/20260420_v6_baseline/results/seta/v6_baseline/incremental.jsonl",
             EXPERIMENTS / "20260420/20260420_v6_retrieval/results/seta/v6_retrieval/incremental.jsonl",
             EXPERIMENTS / "20260420/20260420_v6_3stage/retrieval_results/seta.jsonl"),
    "swe": (EXPERIMENTS / "20260420/20260420_v6_baseline/results/swe/v6_baseline/incremental.jsonl",
            EXPERIMENTS / "20260420/20260420_v6_retrieval/results/swe/v6_retrieval/incremental.jsonl",
            EXPERIMENTS / "20260420/20260420_v6_3stage/retrieval_results/swe.jsonl"),
}

META_SKILLS = {
    # generic/meta skills that retrieve on anything
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


def is_pass(r):
    if r.get("resolved"):
        return True
    sc = r.get("score")
    return sc is not None and sc >= 0.75


def normalize_tid(s):
    return str(s).replace("_s_", "__")


def load(path, kfn):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[kfn(r)] = r
    return out


DOMAIN_KEYWORDS = {
    "bayesian/stats": ["bayesian", "mcmc", "stan", "pystan", "rstan", "posterior", "probabilit"],
    "proof-assistant": ["coq", "proof", "theorem", "plus_comm", "lean", "isabelle"],
    "ML-training": ["pytorch", "cifar", "caffe", "gpt2", "xgboost", "train", "tensorboard", "torch"],
    "graphics/simulation": ["mjcf", "mujoco", "pov-ray", "povray", "raytrac", "render", "blender", "opengl"],
    "semantic-web/graphs": ["sparql", "rdf", "ttl", "turtle", "ontolog"],
    "systems/low-level": ["gcov", "elf", "ocaml", "cython", "linker", "glibc", "syscall", "kernel"],
    "compilers/parsers": ["parser", "ast", "compile", "interpreter", "javac", "bytecode"],
    "HPC/parallel": ["mpi", "openmp", "cuda", "tensor-parallel", "distributed", "nccl", "slurm"],
    "cryptography": ["7z", "hash", "crypto", "ssl", "cert", "openssl", "rsa", "aes", "gpg"],
    "web/browser": ["selenium", "html", "javascript", "webdriver", "browser", "xss", "csp", "filter-js"],
    "data-eng": ["csv", "parquet", "arrow", "dataset", "huggingface", "json", "token"],
    "databases": ["wal", "grpc", "kv-store", "sqlite", "postgres", "mysql", "db-"],
    "dev-ops": ["docker", "qemu", "vm", "alpine", "ssh", "apt", "dpkg", "ubuntu"],
    "chem/bio": ["protein", "raman", "chemistry", "bioinform", "crystall", "dna", "amino"],
    "scheduling/planning": ["scheduling", "constraint", "pddl", "planning"],
    "terminal-emulation": ["tmux", "terminal", "headless-terminal", "pty"],
}


def categorize(desc: str, tid: str) -> str:
    t = (desc + " " + tid).lower()
    for cat, kws in DOMAIN_KEYWORDS.items():
        if any(k in t for k in kws):
            return cat
    return "OTHER"


for bench, (bl_f, rt_f, meta_f) in BENCHES.items():
    print(f"\n{'='*80}\n# {bench.upper()} deep dive\n{'='*80}")
    bl = load(bl_f, lambda r: str(r.get("task_id") or r.get("instance_id")))
    rt = load(rt_f, lambda r: str(r.get("task_id") or r.get("instance_id")))
    meta = load(meta_f, lambda r: normalize_tid(r["task_id"]))

    # Meta skill pollution — of all retrieved top-3 skills, what frac are meta?
    all_skills = Counter()
    top1_skills = Counter()
    for tid, m in meta.items():
        top = m.get("reranked_top10", [])[:3]
        for s in top:
            all_skills[s["skill_name"]] += 1
        if top:
            top1_skills[top[0]["skill_name"]] += 1
    print(f"\n## top-1 skill distribution ({bench}):")
    for name, n in top1_skills.most_common(10):
        print(f"  {name}: {n}")

    meta_frac_top1 = sum(n for s, n in top1_skills.items() if s in META_SKILLS) / max(1, sum(top1_skills.values()))
    meta_frac_top3 = sum(n for s, n in all_skills.items() if s in META_SKILLS) / max(1, sum(all_skills.values()))
    print(f"meta-skill frac @top1: {meta_frac_top1:.2%}")
    print(f"meta-skill frac @top3: {meta_frac_top3:.2%}")

    # Categorize all tasks + partition counts per category
    task_info = {}
    for tid in rt.keys():
        m = meta.get(tid)
        if not m:
            continue
        desc = m.get("task_description", "") or ""
        cat = categorize(desc, tid)
        b_row = bl.get(tid)
        r_row = rt.get(tid)
        if not b_row:
            continue
        b = is_pass(b_row)
        r = is_pass(r_row)
        top = m.get("reranked_top10", [])
        top1 = top[0]["rerank_score"] if top else 0.0
        task_info[tid] = dict(cat=cat, b=b, r=r, top1=top1, desc=desc[:140])

    # Category breakdown
    print(f"\n## per-category partition ({bench}):")
    by_cat = defaultdict(lambda: dict(n=0, b_pass=0, r_pass=0, hurt=0, helped=0, avg_top1=0.0, lib_gap_n=0, tids=[]))
    for tid, info in task_info.items():
        c = by_cat[info["cat"]]
        c["n"] += 1
        c["b_pass"] += int(info["b"])
        c["r_pass"] += int(info["r"])
        c["avg_top1"] += info["top1"]
        if info["top1"] < 0.2:
            c["lib_gap_n"] += 1
        if info["b"] and not info["r"]:
            c["hurt"] += 1
        if not info["b"] and info["r"]:
            c["helped"] += 1
        c["tids"].append(tid)

    # sort by n desc
    cats_sorted = sorted(by_cat.items(), key=lambda kv: -kv[1]["n"])
    print(f"{'cat':<22}{'n':>4}{'b_pass':>8}{'r_pass':>8}{'hurt':>6}{'helped':>8}{'lib_gap':>9}{'avg_top1':>10}")
    for cat, s in cats_sorted:
        avg = s["avg_top1"] / s["n"] if s["n"] else 0
        print(f"{cat:<22}{s['n']:>4}{s['b_pass']:>8}{s['r_pass']:>8}{s['hurt']:>6}{s['helped']:>8}{s['lib_gap_n']:>9}{avg:>10.3f}")

    # "both_fail" hard tasks - what domains dominate?
    print(f"\n## both_fail tasks (baseline fail, retrieval fail) — hardest + most under-served:")
    both_fail_cats = Counter()
    for tid, info in task_info.items():
        if not info["b"] and not info["r"]:
            both_fail_cats[info["cat"]] += 1
    for cat, n in both_fail_cats.most_common():
        print(f"  {cat}: {n}")

    # lib-gap high-confidence cases — lowest top-1 across the bench
    print(f"\n## lowest 15 top-1 rerank scores (most definite lib gaps):")
    low = sorted(task_info.items(), key=lambda kv: kv[1]["top1"])[:15]
    for tid, info in low:
        pass_str = "B+R" if info["b"] and info["r"] else ("B" if info["b"] else ("R" if info["r"] else "--"))
        print(f"  [{info['cat']}] {tid} top1={info['top1']:.3f} [{pass_str}] {info['desc']}")
