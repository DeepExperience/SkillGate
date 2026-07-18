"""v6 retrieval pipeline (2026-04-20):
  Stage 1 — Embedding coarse filter: Qwen3-Embedding-8B cosine sim → top 40-50
  Stage 2 — Reranker precision filter: Qwen3-Reranker-8B yes/no scoring → top 10
  (No LLM rerank — user explicitly dropped it.)

Output jsonl per bench with both tops:
    {
      "task_id": "...",
      "dataset": "...",
      "task_description": "...",
      "embedding_model": "Qwen/Qwen3-Embedding-8B",
      "rerank_model": "Qwen/Qwen3-Reranker-8B",
      "coarse_top50": [{"rank":1,"skill_name","skill_path","embedding_score"}, ...],
      "reranked_top10": [{"rank":1,"skill_name","skill_path","rerank_score","coarse_rank"}, ...]
    }

IMPORTANT (GPU memory):
  Qwen3-Reranker-8B uses ~16GB VRAM. Qwen3.5-27B (SGLang TP=4) is using 4×H800.
  You MUST stop SGLang 27B before running this script. This script auto-verifies
  GPU availability at startup and errors if SGLang is still live on any GPU.

Usage:
    # Stop SGLang first
    tmux kill-session -t sglang
    sleep 10

    # Run pipeline (loads index, batches queries, outputs one jsonl per bench)
    python retrieve_v6_3stage.py --bench claw skillsbench seta tb2 swe \
        --coarse-k 50 --rerank-k 10 \
        --out-dir experiments/ \
        --date $(date +%Y%m%d)

    # Result files:
    #   experiments/YYYYMMDD/YYYYMMDD_v6_3stage/retrieval_results/claw.jsonl
    #   experiments/YYYYMMDD/YYYYMMDD_v6_3stage/retrieval_results/skillsbench.jsonl
    #   ...

    # Restart SGLang
    tmux new-session -d -s sglang 'bash /mnt/.../LLMWeights/qwen3.5-27b.sh'
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent.parent.parent  # .../Projects
INDEX_PATH = SCRIPT_DIR / "skill_index_qwen3emb8b.pkl"
DATASETS_DIR = PROJ_ROOT / "datasets"
SKILLS_RETR_MODULE = SCRIPT_DIR  # for task-listing helpers

# Reranker
RERANKER_MODEL = "Qwen/Qwen3-Reranker-8B"
RERANKER_INSTRUCTION = (
    "Given an agent task description, retrieve skill cards that are most useful "
    "for solving it (tool integration, API usage, workflow templates)."
)
# Prompt format per Qwen3-Reranker HF card
RERANKER_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


# ---------------------------------------------------------------------------
# Preflight: ensure GPUs free (SGLang killed)
# ---------------------------------------------------------------------------

def check_gpu_free(min_free_gb: int = 20) -> None:
    """Ensure each GPU has ≥ min_free_gb free; else abort."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception as e:
        print(f"WARN: nvidia-smi failed: {e}", file=sys.stderr)
        return
    lines = out.splitlines()
    ok = True
    for i, line in enumerate(lines):
        free_mb, total_mb = (int(x) for x in line.split(","))
        free_gb = free_mb / 1024
        total_gb = total_mb / 1024
        print(f"  GPU{i}: free={free_gb:.1f}G / {total_gb:.1f}G")
        if free_gb < min_free_gb:
            print(f"  ❌ GPU{i} has only {free_gb:.1f}G free — need >= {min_free_gb}G")
            print(f"  Is SGLang still running? Check: tmux ls | grep sglang")
            ok = False
    if not ok:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Task sources (mirror batch_retrieve.py; read task descriptions)
# ---------------------------------------------------------------------------

def _read_task_desc(task_dir: Path, limit: int = 3000) -> str:
    """Consistent task-description extractor used by harbor-format benches.
    Priority: instruction.md → task.toml 'instruction' field → fallback task_dir name.
    """
    inst = task_dir / "instruction.md"
    if inst.exists():
        try:
            txt = inst.read_text(encoding="utf-8", errors="replace").strip()
            if txt:
                return txt[:limit]
        except Exception:
            pass
    toml_path = task_dir / "task.toml"
    if toml_path.exists():
        import re as _re
        txt = toml_path.read_text(encoding="utf-8", errors="replace")
        for pat in (
            r'instruction\s*=\s*"""(.+?)"""',
            r"instruction\s*=\s*'''(.+?)'''",
            r'instruction\s*=\s*"((?:[^"\\]|\\.)*)"',
        ):
            m = _re.search(pat, txt, _re.DOTALL)
            if m:
                return m.group(1).strip()[:limit]
    return task_dir.name


def list_skillsbench_tasks() -> list[dict]:
    # Prefer `tasks/`; fall back to `tasks-no-skills/` (same task set).
    base = DATASETS_DIR / "skillsbench"
    tasks_dir = base / "tasks" if (base / "tasks").is_dir() else base / "tasks-no-skills"
    if not tasks_dir.is_dir():
        return []
    seen = set(); out = []
    for td in sorted(tasks_dir.iterdir()):
        if not td.is_dir() or td.name.startswith(".") or td.name in seen:
            continue
        seen.add(td.name)
        out.append({"task_id": td.name, "dataset": "skillsbench",
                    "task_description": _read_task_desc(td)})
    return out


def list_seta_tasks() -> list[dict]:
    p = DATASETS_DIR / "seta" / "dataset" / "seta_baseline_30"
    if not p.is_dir():
        return []
    out = []
    for td in sorted(p.iterdir()):
        if not td.is_dir():
            continue
        out.append({"task_id": td.name, "dataset": "seta",
                    "task_description": _read_task_desc(td)})
    return out


def list_seta_synth_tasks() -> list[dict]:
    """SETA train set: 300 task IDs from prebake_images/seta_300.txt, with
    task descriptions read from datasets/seta/dataset/synth_data_harbor/.

    Used by SFT data collection — the small 30-task seta_baseline set is the
    test holdout, the 300-task synth_data_harbor set is the train pool.
    """
    list_path = (
        DATASETS_DIR.parent / "GeneralAgent" / "eval_scripts"
        / "prebake_images" / "seta_300.txt"
    )
    tasks_root = DATASETS_DIR / "seta" / "dataset" / "synth_data_harbor"
    if not list_path.is_file() or not tasks_root.is_dir():
        return []
    out = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        task_id = line.strip()
        if not task_id or task_id.startswith("#"):
            continue
        td = tasks_root / task_id
        if not td.is_dir():
            # Some seta_300 ids may not be in synth_data_harbor; skip silently
            # rather than fail the whole run.
            continue
        out.append({"task_id": task_id, "dataset": "seta_synth",
                    "task_description": _read_task_desc(td)})
    return out


def list_tb2_tasks() -> list[dict]:
    p = DATASETS_DIR / "terminal-bench-v2"
    if not p.is_dir():
        return []
    out = []
    for td in sorted(p.iterdir()):
        if not td.is_dir():
            continue
        # Accept any task dir that has test.sh OR task.toml OR task.yaml
        if not any((
            (td / "tests" / "test.sh").exists(),
            (td / "task.toml").exists(),
            (td / "task.yaml").exists(),
        )):
            continue
        desc = _read_task_desc(td)
        # task.yaml fallback: some tb2 variants carry prompt here
        if desc == td.name:
            task_yaml = td / "task.yaml"
            if task_yaml.exists():
                try:
                    import yaml
                    y = yaml.safe_load(task_yaml.read_text()) or {}
                    desc = ((y.get("prompt") or {}).get("text") or "").strip() or td.name
                except Exception:
                    pass
        out.append({"task_id": td.name, "dataset": "tb2", "task_description": desc[:3000]})
    return out


def list_swe_tasks() -> list[dict]:
    """SWE-Gym: read problem_statement from parquet for every instance we may run.

    Target set = union of:
      - ALL_IMAGES (legacy 21) from run_unified_swe.py, converted from `owner_s_repo-N` → `owner__repo-N`
      - swe_lite_100.txt (100-instance stratified subset)
      - swe-bench-verified parquet (full)

    task_id is the canonical `owner__repo-N` instance_id (matches runner output).
    """
    out = []
    # 1. Load all parquet rows into {instance_id: problem_statement}
    parquets = [
        PROJ_ROOT / "datasets/swe-gym/lite/data/train-00000-of-00001.parquet",
        PROJ_ROOT / "datasets/swe-bench-verified/data/test-00000-of-00001.parquet",
        PROJ_ROOT / "datasets/swe-bench-verified/data/data/test-00000-of-00001.parquet",  # alt layout
    ]
    ps_map: dict[str, str] = {}
    for pq in parquets:
        if not pq.exists():
            continue
        try:
            import pandas as pd
            df = pd.read_parquet(pq)
            for _, row in df.iterrows():
                iid = row.get("instance_id")
                ps = row.get("problem_statement", "")
                if iid and ps and iid not in ps_map:
                    ps_map[str(iid)] = str(ps)
        except Exception as e:
            print(f"WARN: SWE parquet {pq.name} failed: {e}", file=sys.stderr)
    if not ps_map:
        print("WARN: no SWE parquet loaded; SWE retrieval will be empty", file=sys.stderr)
        return out

    # 2. Figure out which instance_ids we care about
    target_ids: set[str] = set()
    # 2a. ALL_IMAGES from runner module (legacy 21)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_unified_swe",
            str(PROJ_ROOT / "GeneralAgent/eval_scripts/unified_runner/run_unified_swe.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for img in getattr(mod, "ALL_IMAGES", []):
            # e.g. "xingyaoww/sweb.eval.x86_64.dask_s_dask-6626:latest"
            raw = img.split(".")[-1].split(":")[0]          # dask_s_dask-6626
            target_ids.add(raw.replace("_s_", "__"))         # dask__dask-6626
    except Exception as e:
        print(f"WARN: SWE ALL_IMAGES import failed: {e}", file=sys.stderr)
    # 2b. swe_lite_100.txt (100 instance ids, already in __ format)
    lite100 = PROJ_ROOT / "GeneralAgent/eval_scripts/prebake_images/swe_lite_100.txt"
    if lite100.exists():
        for line in lite100.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                target_ids.add(line)
    # 2c. If neither source produced targets, fall back to entire parquet
    if not target_ids:
        target_ids = set(ps_map.keys())

    # 3. Emit rows; skip any instance_id not in parquet (with warning)
    missing = 0
    for iid in sorted(target_ids):
        ps = ps_map.get(iid)
        if not ps:
            missing += 1
            continue
        out.append({
            "task_id": iid,
            "dataset": "swe",
            "task_description": ps[:3000],
        })
    if missing:
        print(f"WARN: {missing} target instance_ids missing from parquet", file=sys.stderr)
    print(f"  SWE: {len(out)} tasks with problem_statement (from {len(target_ids)} targets)")
    return out


def list_claw_tasks() -> list[dict]:
    out = []
    p = DATASETS_DIR / "claw-eval" / "tasks"
    if not p.is_dir():
        return out
    # Only T-series (general tag), 161 tasks (matches claw_161_t_series.txt)
    target_file = PROJ_ROOT / "GeneralAgent/eval_scripts/prebake_images/claw_161_t_series.txt"
    if target_file.exists():
        targets = set(l.strip() for l in target_file.read_text().splitlines()
                      if l.strip() and not l.startswith("#"))
    else:
        targets = None  # include all T*
    for td in sorted(p.iterdir()):
        if not td.is_dir() or not td.name.startswith("T"):
            continue
        if targets is not None and td.name not in targets:
            continue
        try:
            import yaml
            task_def = yaml.safe_load((td / "task.yaml").read_text())
            prompt = (task_def.get("prompt", {}) or {}).get("text", "") or ""
            cat = task_def.get("category", "")
            tags = task_def.get("tags", [])
            desc = f"Task: {task_def.get('task_name', td.name)}\nCategory: {cat}\nTags: {','.join(tags)}\nPrompt: {prompt}"
        except Exception:
            desc = td.name
        out.append({"task_id": td.name, "dataset": "claw", "task_description": desc[:3000]})
    return out


TASK_LISTERS = {
    "skillsbench": list_skillsbench_tasks,
    "seta":        list_seta_tasks,
    "seta_synth":  list_seta_synth_tasks,
    "tb2":         list_tb2_tasks,
    "swe":         list_swe_tasks,
    "claw":        list_claw_tasks,
}


# ---------------------------------------------------------------------------
# Stage 1: Embedding coarse (load index + embed queries + cosine → top-K)
# ---------------------------------------------------------------------------

def load_index() -> dict:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index missing: {INDEX_PATH}. Run build_index.py first or use existing "
            f"skill_index_qwen3emb8b.pkl (8B)."
        )
    with open(INDEX_PATH, "rb") as f:
        idx = pickle.load(f)
    print(f"  Loaded index: {idx['embeddings'].shape} skills, model={idx.get('model_name')}")
    return idx


def embed_queries(queries: list[str], model_name: str,
                  query_instruction: str) -> np.ndarray:
    """Embed N task queries into N × D matrix. Uses same model as the index."""
    os.environ.setdefault("HF_HUB_OFFLINE", "0")  # allow download if missing
    from sentence_transformers import SentenceTransformer
    print(f"  Loading embedding model {model_name}...")
    model = SentenceTransformer(model_name)
    # Qwen3-Embedding uses instruction-prefix for queries
    prefixed = [f"{query_instruction}\n{q}" if query_instruction else q for q in queries]
    print(f"  Encoding {len(queries)} queries...")
    embs = model.encode(prefixed, show_progress_bar=True, batch_size=4,
                        normalize_embeddings=True, convert_to_numpy=True)
    # Release model
    del model
    import torch, gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embs


def coarse_topk(query_emb: np.ndarray, skill_embs: np.ndarray, k: int
                ) -> list[tuple[int, float]]:
    """Cosine similarity top-K (embs already normalized)."""
    scores = skill_embs @ query_emb
    topk_idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in topk_idx]


# ---------------------------------------------------------------------------
# Stage 2: Qwen3-Reranker-8B (batch scoring yes/no for each (q, doc))
# ---------------------------------------------------------------------------

class Qwen3Reranker:
    """Wraps Qwen3-Reranker-8B for (query, doc) → P(yes)."""

    def __init__(self, model_name: str = RERANKER_MODEL,
                 dtype: str = "bfloat16", batch_size: int = 8,
                 max_len: int = 8192):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"  Loading reranker {model_name} (dtype={dtype}, bs={batch_size}, max_len={max_len})...")
        # Left-pad so the trailing yes/no logit position is constant across batch items.
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        torch_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype
        ).cuda().eval()
        # Cache yes/no token ids
        self.yes_id = self.tok.convert_tokens_to_ids("yes")
        self.no_id = self.tok.convert_tokens_to_ids("no")
        # Ensure pad token is configured
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.batch_size = batch_size
        self.max_len = max_len  # hard cap on full sequence (prefix+body+suffix)

    def score_pairs(self, query: str, docs: list[str],
                    instruction: str = RERANKER_INSTRUCTION,
                    batch_size: Optional[int] = None) -> list[float]:
        """Return P(yes) for each (query, doc) pair. Higher = more relevant.

        Build full prompt per example as ONE string (prefix + body + suffix), then
        batch-tokenize with left-padding. This guarantees the suffix's last token
        (where yes/no is predicted) is always at position max_len-1 across the batch,
        so positional embeddings are consistent.
        """
        import torch
        bs = batch_size or self.batch_size
        scores: list[float] = []
        # Build full-prompt strings (prefix+body+suffix). Truncate per-example if body is huge.
        prompts: list[str] = []
        for doc in docs:
            body = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
            prompts.append(RERANKER_PREFIX + body + RERANKER_SUFFIX)

        for i in range(0, len(prompts), bs):
            batch = prompts[i : i + bs]
            enc = self.tok(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_len, add_special_tokens=False,
            )
            input_ids = enc["input_ids"].cuda()
            attention_mask = enc["attention_mask"].cuda()

            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Left-padding ⇒ real last token is at index -1 for every row
            logits = out.logits[:, -1, :]
            yes_logits = logits[:, self.yes_id]
            no_logits = logits[:, self.no_id]
            pair = torch.stack([yes_logits, no_logits], dim=1)
            probs = torch.softmax(pair, dim=1)
            scores.extend(probs[:, 0].cpu().tolist())
        return scores

    def unload(self):
        import torch, gc
        del self.model
        del self.tok
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_bench(bench: str, idx: dict, reranker: Optional[Qwen3Reranker],
              coarse_k: int, rerank_k: int,
              out_path: Path) -> int:
    """Run 3-stage retrieval for one bench. Returns #tasks processed."""
    tasks = TASK_LISTERS[bench]()
    print(f"\n=== {bench}: {len(tasks)} tasks ===")
    if not tasks:
        return 0

    skill_names = idx["skill_names"]
    skill_paths = idx["skill_paths"]
    skill_summaries = idx.get("skill_summaries") or [""] * len(skill_names)
    skill_embs = np.asarray(idx["embeddings"])

    # Stage 1: embed all tasks batch
    query_instr = idx.get("query_instruction", "")
    task_descs = [t["task_description"] for t in tasks]
    query_embs = embed_queries(task_descs, idx["model_name"], query_instr)

    # Stage 2: per-task reranker (if loaded)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w") as f:
        for i, t in enumerate(tasks):
            q_emb = query_embs[i]
            # Coarse top-K
            coarse = coarse_topk(q_emb, skill_embs, coarse_k)
            coarse_entries = [
                {
                    "rank": r + 1,
                    "skill_name": skill_names[idx_i],
                    "skill_path": skill_paths[idx_i],
                    "embedding_score": score,
                }
                for r, (idx_i, score) in enumerate(coarse)
            ]
            # Rerank top-K (if reranker available)
            reranked_entries = []
            if reranker is not None:
                # Doc = summary (or skill_name if summary empty)
                docs = [skill_summaries[idx_i] or skill_names[idx_i]
                        for idx_i, _ in coarse]
                t0 = time.time()
                scores = reranker.score_pairs(t["task_description"], docs)
                elapsed = time.time() - t0

                # Sort by rerank score desc
                ranked = sorted(range(len(scores)), key=lambda k: -scores[k])
                for new_rank, pos in enumerate(ranked[:rerank_k]):
                    idx_i, coarse_score = coarse[pos]
                    reranked_entries.append({
                        "rank": new_rank + 1,
                        "skill_name": skill_names[idx_i],
                        "skill_path": skill_paths[idx_i],
                        "rerank_score": scores[pos],
                        "coarse_rank": pos + 1,
                        "coarse_score": coarse_score,
                    })
                print(f"  [{i+1}/{len(tasks)}] {t['task_id']} reranked in {elapsed:.1f}s "
                      f"top1={reranked_entries[0]['skill_name']}")
            else:
                print(f"  [{i+1}/{len(tasks)}] {t['task_id']} coarse only (no reranker)")

            row = {
                "task_id": t["task_id"],
                "dataset": t["dataset"],
                "task_description": t["task_description"][:2000],
                "embedding_model": idx["model_name"],
                "rerank_model": RERANKER_MODEL if reranker else "",
                "coarse_top50": coarse_entries,
                "reranked_top10": reranked_entries,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    print(f"=== {bench}: wrote {written} rows to {out_path}")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", nargs="+",
                    choices=["skillsbench", "seta", "seta_synth", "tb2", "swe", "claw"],
                    required=True)
    ap.add_argument("--coarse-k", type=int, default=50,
                    help="Coarse top-K from embedding (default 50)")
    ap.add_argument("--rerank-k", type=int, default=10,
                    help="Final top-K after reranker (default 10)")
    ap.add_argument("--no-rerank", action="store_true",
                    help="Skip Stage 2 (coarse only; for quick testing)")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for per-bench jsonl")
    ap.add_argument("--date", default=time.strftime("%Y%m%d"),
                    help="Date prefix for output filename")
    ap.add_argument("--suffix", default="v6_3stage",
                    help="Output filename suffix (default v6_3stage)")
    ap.add_argument("--skip-gpu-check", action="store_true",
                    help="Skip GPU free check (dev/testing only)")
    ap.add_argument("--rerank-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="Reranker weight dtype (default bfloat16)")
    ap.add_argument("--rerank-batch-size", type=int, default=8,
                    help="Reranker batch size (default 8)")
    ap.add_argument("--rerank-max-len", type=int, default=8192,
                    help="Reranker max sequence length (default 8192)")
    args = ap.parse_args()

    # Preflight
    if not args.skip_gpu_check and not args.no_rerank:
        print("=== GPU preflight ===")
        check_gpu_free(min_free_gb=20)

    # Load shared index
    print("\n=== Load embedding index ===")
    idx = load_index()

    # Load reranker once (shared across benches)
    reranker = None
    if not args.no_rerank:
        print("\n=== Load reranker ===")
        reranker = Qwen3Reranker(
            RERANKER_MODEL,
            dtype=args.rerank_dtype,
            batch_size=args.rerank_batch_size,
            max_len=args.rerank_max_len,
        )

    out_base = Path(args.out_dir)
    if out_base.name == "experiments":
        out_dir = out_base / args.date / f"{args.date}_{args.suffix}" / "retrieval_results"
    else:
        # Legacy v8 layout: <results>/<date>/retrieval_results/<suffix>/<bench>.jsonl
        out_dir = out_base / args.date / "retrieval_results" / args.suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    for bench in args.bench:
        out_path = out_dir / f"{bench}.jsonl"
        run_bench(bench, idx, reranker, args.coarse_k, args.rerank_k, out_path)

    if reranker:
        print("\n=== Unload reranker ===")
        reranker.unload()

    print("\n=== DONE. Files written ===")
    for bench in args.bench:
        p = out_dir / f"{bench}.jsonl"
        if p.exists():
            n = sum(1 for _ in open(p))
            print(f"  {p}  ({n} rows)")
    print(f"  (all under {out_dir}/)")


if __name__ == "__main__":
    main()
