#!/usr/bin/env python3
"""Batch skill retrieval across 5 benches.

For each task in the selected benches, runs coarse (embedding top-20) + fine
(LLM rerank top-5) retrieval, writes one JSONL row per task.

Usage:
    python batch_retrieve.py --dataset all --index skill_index_qwen3emb8b.pkl \
        --out-dir experiments/

    # Only rerank pass on existing coarse result
    python batch_retrieve.py --rerank-only --input existing.jsonl --out new.jsonl
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

BASE = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))
sys.path.insert(0, str(BASE / "GeneralAgent/eval_scripts/skills_retrieval"))

from retrieve import retrieve_skills, _load_index  # noqa: E402


# ---- task description extractors --------------------------------------------

def _read_text(p: Path, limit: int = 4000) -> str:
    try:
        t = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return t[:limit]


def _parse_toml_instruction(toml_path: Path) -> str:
    """Best-effort TOML instruction extraction without tomllib dep fuss."""
    if not toml_path.exists():
        return ""
    txt = toml_path.read_text(encoding="utf-8", errors="replace")
    # Match triple-quoted or single-line instruction
    m = re.search(r'instruction\s*=\s*"""(.+?)"""', txt, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"instruction\s*=\s*'''(.+?)'''", txt, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'instruction\s*=\s*"((?:[^"\\]|\\.)*)"', txt)
    if m:
        return m.group(1).encode().decode("unicode_escape")
    return ""


def collect_skillsbench() -> list[dict]:
    """SkillsBench: 88 unique tasks (with-skills + no-skills share task set)."""
    tasks_dir = BASE / "datasets/skillsbench/tasks"
    out = []
    for td in sorted(tasks_dir.iterdir()):
        if not td.is_dir():
            continue
        desc = ""
        instr_md = td / "instruction.md"
        if instr_md.exists():
            desc = _read_text(instr_md)
        if not desc:
            desc = _parse_toml_instruction(td / "task.toml")
        if not desc:
            continue
        out.append({"task_id": td.name, "dataset": "skillsbench", "task_description": desc[:3000]})
    return out


def collect_seta() -> list[dict]:
    root = BASE / "datasets/seta/dataset/seta_baseline_30"
    out = []
    for td in sorted(root.iterdir()):
        if not td.is_dir():
            continue
        desc = _parse_toml_instruction(td / "task.toml")
        if not desc:
            instr_md = td / "instruction.md"
            if instr_md.exists():
                desc = _read_text(instr_md)
        if not desc:
            continue
        out.append({"task_id": td.name, "dataset": "seta", "task_description": desc[:3000]})
    return out


def collect_tb2() -> list[dict]:
    root = BASE / "datasets/terminal-bench-v2"
    out = []
    for td in sorted(root.iterdir()):
        if not td.is_dir():
            continue
        if not (td / "task.toml").exists() and not (td / "instruction.md").exists():
            continue
        desc = ""
        instr_md = td / "instruction.md"
        if instr_md.exists():
            desc = _read_text(instr_md)
        if not desc:
            desc = _parse_toml_instruction(td / "task.toml")
        if not desc:
            continue
        out.append({"task_id": td.name, "dataset": "tb2", "task_description": desc[:3000]})
    return out


def collect_swe() -> list[dict]:
    """Walk SWE-Gym parquet to get instance_id + problem_statement."""
    out = []
    candidates = [
        BASE / "datasets/swe-gym/lite/data/train-00000-of-00001.parquet",
        BASE / "datasets/swe-bench-verified/data/test-00000-of-00001.parquet",
    ]
    for pq_path in candidates:
        if not pq_path.exists():
            continue
        try:
            import pandas as pd
            df = pd.read_parquet(pq_path)
            for _, row in df.iterrows():
                iid = row.get("instance_id")
                ps = row.get("problem_statement", "")
                if iid and ps:
                    out.append({
                        "task_id": str(iid),
                        "dataset": "swe",
                        "task_description": str(ps)[:3000],
                    })
            if out:
                return out
        except Exception as e:
            print(f"  [swe] parquet {pq_path} failed: {e}", file=sys.stderr)
    return out


def collect_claw() -> list[dict]:
    """Claw-eval tasks live at datasets/claw-eval/tasks/<tid>/task.yaml.

    Task description = prompt.text (often zh); include task_name/category as context.
    """
    root = BASE / "datasets/claw-eval/tasks"
    out = []
    if not root.exists():
        return out
    try:
        import yaml  # PyYAML
    except ImportError:
        yaml = None
    for td in sorted(root.iterdir()):
        if not td.is_dir():
            continue
        # Include T-series (single-turn general) and C-series (multi-turn user_agent).
        # 2026-04-19 update: previously only T was included; C-series was added so
        # retrieval jsonl covers the full 199-task general-tagged subset.
        if not (td.name.startswith("T") or td.name.startswith("C")):
            continue
        yaml_path = td / "task.yaml"
        if not yaml_path.exists():
            continue
        txt = yaml_path.read_text(encoding="utf-8", errors="replace")
        desc = ""
        if yaml is not None:
            try:
                rec = yaml.safe_load(txt)
                tname = rec.get("task_name", "")
                cat = rec.get("category", "")
                tags = rec.get("tags", [])
                prompt = rec.get("prompt", {})
                if isinstance(prompt, dict):
                    ptext = prompt.get("text", "")
                else:
                    ptext = str(prompt)
                parts = []
                if tname:
                    parts.append(f"Task: {tname}")
                if cat:
                    parts.append(f"Category: {cat}")
                if tags:
                    parts.append(f"Tags: {', '.join(tags)}")
                if ptext:
                    parts.append(f"Prompt: {ptext}")
                desc = "\n".join(parts)
            except Exception:
                desc = ""
        if not desc:
            # Regex fallback
            m = re.search(r"prompt:\s*\n\s*text:\s*[\"']?(.+?)[\"']?\s*\n", txt)
            desc = m.group(1) if m else ""
        if not desc:
            continue
        out.append({"task_id": td.name, "dataset": "claw", "task_description": desc[:3000]})
    return out


COLLECTORS = {
    "skillsbench": collect_skillsbench,
    "seta": collect_seta,
    "tb2": collect_tb2,
    "swe": collect_swe,
    "claw": collect_claw,
}


# ---- retrieval driver -------------------------------------------------------

def run_bench(
    bench: str,
    tasks: list[dict],
    index_path: Path,
    out_path: Path,
    sglang_url: str,
    sglang_model: str,
    use_llm: bool = True,
    coarse_k: int = 20,
    top_k: int = 5,
) -> dict:
    idx = _load_index(index_path)
    emb_model = idx["model_name"]
    rerank_model = sglang_model if use_llm else ""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    n_success = 0
    n_err = 0
    coarse_top1_scores = []
    rerank_times = []
    with open(out_path, "w", encoding="utf-8") as fo:
        for i, task in enumerate(tasks, 1):
            t0 = time.time()
            try:
                res = retrieve_skills(
                    task["task_description"],
                    top_k=top_k,
                    use_llm_rerank=use_llm,
                    index_path=index_path,
                    coarse_k=coarse_k,
                    sglang_url=sglang_url,
                    sglang_model=sglang_model,
                    return_coarse=True,
                )
                row = {
                    "task_id": task["task_id"],
                    "dataset": task["dataset"],
                    "task_description": task["task_description"][:2000],
                    "embedding_model": emb_model,
                    "rerank_model": rerank_model,
                    "coarse_top20": res["coarse_top"],
                    "reranked_top5": res["reranked_top"],
                }
                if res["coarse_top"]:
                    coarse_top1_scores.append(res["coarse_top"][0]["embedding_score"])
                n_success += 1
            except Exception as e:
                row = {
                    "task_id": task["task_id"],
                    "dataset": task["dataset"],
                    "task_description": task["task_description"][:2000],
                    "embedding_model": emb_model,
                    "rerank_model": rerank_model,
                    "coarse_top20": [],
                    "reranked_top5": [],
                    "rerank_error": str(e),
                }
                n_err += 1
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            fo.flush()
            rerank_times.append(time.time() - t0)
            if i % 10 == 0 or i == len(tasks):
                print(f"  [{bench}] {i}/{len(tasks)} tasks done, avg={sum(rerank_times)/len(rerank_times):.1f}s/task", flush=True)

    summary = {
        "bench": bench,
        "n_tasks": len(tasks),
        "n_success": n_success,
        "n_err": n_err,
        "avg_time_sec": (sum(rerank_times) / len(rerank_times)) if rerank_times else 0.0,
        "total_time_sec": time.time() - t_start,
        "coarse_top1_mean": (sum(coarse_top1_scores) / len(coarse_top1_scores)) if coarse_top1_scores else 0.0,
        "coarse_top1_min": min(coarse_top1_scores) if coarse_top1_scores else 0.0,
        "coarse_top1_max": max(coarse_top1_scores) if coarse_top1_scores else 0.0,
        "out_path": str(out_path),
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description="Batch skill retrieval across 5 benches")
    ap.add_argument("--dataset", type=str, required=True,
                    choices=list(COLLECTORS.keys()) + ["all"])
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path,
                    default=BASE / "experiments")
    ap.add_argument("--no-llm", action="store_true", help="Skip LLM rerank")
    ap.add_argument("--sglang-url", default="http://localhost:30000/v1/chat/completions")
    ap.add_argument("--sglang-model", default="qwen3.5-27b")
    ap.add_argument("--coarse-k", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="Cap tasks per bench (0=all)")
    args = ap.parse_args()

    today = dt.date.today().strftime("%Y%m%d")
    benches = list(COLLECTORS.keys()) if args.dataset == "all" else [args.dataset]

    all_summary = []
    for bench in benches:
        print(f"\n### Collecting tasks for {bench} ###")
        tasks = COLLECTORS[bench]()
        if args.limit:
            tasks = tasks[:args.limit]
        print(f"  Got {len(tasks)} tasks")
        if not tasks:
            all_summary.append({"bench": bench, "skipped": "no tasks"})
            continue
        out_root = args.out_dir / today / f"{today}_retrieval_batch"
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / "retrieval_results" / f"{bench}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Writing to {out_path}")
        s = run_bench(
            bench, tasks, args.index, out_path,
            args.sglang_url, args.sglang_model,
            use_llm=not args.no_llm,
            coarse_k=args.coarse_k, top_k=args.top_k,
        )
        all_summary.append(s)
        print(f"  [{bench}] done: {s['n_success']} ok / {s['n_err']} err / avg={s['avg_time_sec']:.1f}s/task")

    # Summary markdown
    sum_path = args.out_dir / today / f"{today}_retrieval_batch" / "summary.md"
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write(f"# Batch Retrieval Summary ({today})\n\n")
        f.write(f"Index: `{args.index.name}`  |  LLM rerank: `{not args.no_llm}`\n\n")
        f.write("| Bench | N | ok | err | avg_time | top1 mean / min / max |\n")
        f.write("|---|---|---|---|---|---|\n")
        for s in all_summary:
            if "skipped" in s:
                f.write(f"| {s['bench']} | — | — | — | — | (skipped: {s['skipped']}) |\n")
                continue
            f.write(f"| {s['bench']} | {s['n_tasks']} | {s['n_success']} | {s['n_err']} | "
                    f"{s['avg_time_sec']:.1f}s | {s['coarse_top1_mean']:.3f} / "
                    f"{s['coarse_top1_min']:.3f} / {s['coarse_top1_max']:.3f} |\n")
        f.write("\nSee per-bench jsonl for full results.\n")
    print(f"\nSummary: {sum_path}")


if __name__ == "__main__":
    main()
