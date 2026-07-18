#!/usr/bin/env python3
"""
V8 4-stage skill retrieval pipeline (2026-04-22).

Stages:
  1a) Qwen3-Embedding-8B: cosine top-50 (query has explicit instruction prefix)
  1b) BM25 (bm25s): top-50 over full SKILL.md text
  ── Union of 1a+1b (dedup by skill_name) → typically 50-100 candidates
  2) Qwen3-Reranker-8B: yes/no scoring with explicit usefulness instruction → top-20
  3) DeepSeek-V3.2 (via MAAS): JSON usefulness score 0-10 (parallel) → top-10

Output per task (single jsonl row):
  {
    task_id, dataset, task_description,
    embedding_model, rerank_model, llm_judge_model,
    embedding_instruction, rerank_instruction, llm_judge_prompt,
    stage1_embedding_top50: [...],
    stage1_bm25_top50: [...],
    stage1_union:     [...],   # each entry: {skill_name, skill_path, embedding_score?, bm25_score?}
    stage2_top20_reranked: [...],
    stage3_top10_llm_judge: [...]
  }

Output location:
  experiments/<date>/<date>_v8_4stage/retrieval_results/<bench>.jsonl

Usage:
  python retrieve_v8_4stage.py --bench skillsbench tb2 seta claw swe \
      --out-dir experiments --date 20260422 \
      --llm-concurrency 20
"""
import argparse
import json
import os
import pickle
import re
import sys
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np


# --- Paths --------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent.parent.parent
DATASETS_DIR = PROJ_ROOT / "datasets"

EMB_INDEX_PATH = SCRIPT_DIR / "skill_index_qwen3emb8b.pkl"
BM25_INDEX_PATH = SCRIPT_DIR / "skill_index_bm25.pkl"


# --- Models -------------------------------------------------------------------

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
RERANKER_MODEL = "Qwen/Qwen3-Reranker-8B"
LLM_JUDGE_MODEL = "deepseek-v3.2"

# --- Instructions (explicit "usefulness for solving task" framing) ------------

EMBEDDING_QUERY_INSTRUCTION = (
    "Retrieve skill cards from a skill library that would help an AI agent "
    "solve the given task. The skills contain tool integration patterns, "
    "API references, domain knowledge, and workflow templates."
)

RERANKER_INSTRUCTION = (
    "Evaluate whether this skill card provides useful guidance for an AI agent "
    "attempting to solve the given task. A skill is useful only if reading it "
    "would meaningfully increase the agent's chance of solving the task — for "
    "example by providing relevant APIs, code templates, domain knowledge, or "
    "step-by-step workflows that directly apply. Answer \"yes\" only when the "
    "skill is substantively helpful for this specific task."
)

RERANKER_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


LLM_JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a skill card would help an AI agent solve a specific task.

**Task**:
{task_description}

**Skill card** (`{skill_name}`):
---
{skill_body}
---

Rate how useful this skill is for an agent attempting to solve the task, on a 0-10 scale:
  10 — directly matches the task (the skill essentially solves it)
  7-9 — provides critical APIs / workflows / patterns the agent would use
  4-6 — topically related, provides some helpful context
  1-3 — only tangentially related (same keyword but wrong purpose)
   0 — not useful

Output STRICT JSON only, no preamble, no markdown fencing:
{{"score": <int 0-10>, "reason": "<one sentence explanation>"}}"""


# --- Task listers (mirror v6_3stage) -----------------------------------------

def list_skillsbench_tasks():
    out = []
    seen = set()
    for sub in ("tasks", "tasks-no-skills"):
        p = DATASETS_DIR / "skillsbench" / sub
        if not p.is_dir(): continue
        for td in sorted(p.iterdir()):
            if not td.is_dir() or td.name.startswith("."): continue
            if td.name in seen: continue
            inst = td / "instruction.md"
            desc = inst.read_text(encoding="utf-8", errors="replace")[:3000] if inst.exists() else td.name
            out.append({"task_id": td.name, "dataset": "skillsbench", "task_description": desc})
            seen.add(td.name)
        break
    return out


def list_seta_tasks():
    out = []
    p = DATASETS_DIR / "seta" / "dataset" / "seta_baseline_30"
    if not p.is_dir(): return out
    for td in sorted(p.iterdir()):
        if not td.is_dir(): continue
        md = next(td.glob("*.md"), None)
        desc = md.read_text(encoding="utf-8", errors="replace")[:3000] if md else td.name
        out.append({"task_id": td.name, "dataset": "seta", "task_description": desc})
    return out


def list_tb2_tasks():
    out = []
    p = DATASETS_DIR / "terminal-bench-v2"
    if not p.is_dir(): return out
    for td in sorted(p.iterdir()):
        if not td.is_dir() or not (td / "tests" / "test.sh").exists(): continue
        inst = td / "instruction.md"
        if inst.exists():
            desc = inst.read_text(encoding="utf-8", errors="replace")[:3000]
        else:
            ty = td / "task.yaml"
            if not ty.exists(): continue
            import yaml
            d = yaml.safe_load(ty.read_text())
            desc = (d.get("prompt") or d.get("task_description") or td.name)[:3000]
        out.append({"task_id": td.name, "dataset": "tb2", "task_description": desc})
    return out


def list_claw_tasks():
    """T-series claw tasks filtered by claw_161_t_series.txt, v6-compatible."""
    out = []
    p = DATASETS_DIR / "claw-eval" / "tasks"
    if not p.is_dir():
        return out
    target_file = PROJ_ROOT / "GeneralAgent/eval_scripts/prebake_images/claw_161_t_series.txt"
    if target_file.exists():
        targets = set(l.strip() for l in target_file.read_text().splitlines()
                      if l.strip() and not l.startswith("#"))
    else:
        targets = None
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


def list_swe_tasks():
    """SWE: union of ALL_IMAGES (21) + swe_lite_100.txt + swe-bench-verified parquet.

    Reads problem_statement from multiple parquets to cover both SWE-Gym and
    SWE-Bench Verified instances. Matches v6 coverage (~117 rows).
    """
    out = []
    parquets = [
        PROJ_ROOT / "datasets/swe-gym/lite/data/train-00000-of-00001.parquet",
        PROJ_ROOT / "datasets/swe-bench-verified/data/test-00000-of-00001.parquet",
        PROJ_ROOT / "datasets/swe-bench-verified/data/data/test-00000-of-00001.parquet",
    ]
    ps_map = {}
    import pandas as pd
    for pq in parquets:
        if not pq.exists():
            continue
        try:
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

    target_ids = set()
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_unified_swe",
            str(PROJ_ROOT / "GeneralAgent/eval_scripts/unified_runner/run_unified_swe.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for img in getattr(mod, "ALL_IMAGES", []):
            raw = img.split(".")[-1].split(":")[0]
            target_ids.add(raw.replace("_s_", "__"))
    except Exception as e:
        print(f"WARN: SWE ALL_IMAGES import failed: {e}", file=sys.stderr)
    lite100 = PROJ_ROOT / "GeneralAgent/eval_scripts/prebake_images/swe_lite_100.txt"
    if lite100.exists():
        for line in lite100.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                target_ids.add(line)
    if not target_ids:
        target_ids = set(ps_map.keys())

    missing = 0
    for iid in sorted(target_ids):
        ps = ps_map.get(iid)
        if not ps:
            missing += 1
            continue
        out.append({"task_id": iid, "dataset": "swe", "task_description": ps[:3000]})
    if missing:
        print(f"WARN: {missing} target instance_ids missing from parquet", file=sys.stderr)
    return out


TASK_LISTERS = {
    "skillsbench": list_skillsbench_tasks,
    "seta": list_seta_tasks,
    "tb2": list_tb2_tasks,
    "claw": list_claw_tasks,
    "swe": list_swe_tasks,
}


# --- GPU preflight -----------------------------------------------------------

def check_gpu_free(min_free_gb: int = 20):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception as e:
        print(f"WARN: nvidia-smi failed: {e}", file=sys.stderr)
        return
    ok = True
    for i, line in enumerate(out.splitlines()):
        free_mb, total_mb = (int(x) for x in line.split(","))
        free_gb = free_mb / 1024
        print(f"  GPU{i}: free={free_gb:.1f}G / {total_mb/1024:.1f}G")
        if free_gb < min_free_gb:
            print(f"  ❌ GPU{i} has only {free_gb:.1f}G free — need >= {min_free_gb}G")
            ok = False
    if not ok:
        print("  Kill SGLang: tmux kill-session -t sglang")
        sys.exit(1)


# --- Stage 1a: Embedding ------------------------------------------------------

class Qwen3Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer
        print(f"  Loading embedding model {model_name}...")
        self.model = SentenceTransformer(model_name)

    def encode_queries(self, queries: list[str], instruction: str) -> np.ndarray:
        # Qwen3-Embedding HF-recommended format for queries
        prefixed = [f"Instruct: {instruction}\nQuery: {q}" for q in queries]
        emb = self.model.encode(prefixed, normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(emb, dtype=np.float32)

    def unload(self):
        import torch, gc
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


# --- Stage 2: Reranker --------------------------------------------------------

class Qwen3Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL, dtype="float16",
                 batch_size: int = 8, max_len: int = 8192):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"  Loading reranker {model_name}...")
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype).cuda().eval()
        self.yes_id = self.tok.convert_tokens_to_ids("yes")
        self.no_id = self.tok.convert_tokens_to_ids("no")
        self.prefix_ids = self.tok(RERANKER_PREFIX, return_tensors=None)["input_ids"]
        self.suffix_ids = self.tok(RERANKER_SUFFIX, return_tensors=None)["input_ids"]
        self.batch_size = batch_size
        self.max_len = max_len

    def score_pairs(self, query: str, docs: list[str],
                    instruction: str = RERANKER_INSTRUCTION) -> list[float]:
        import torch
        scores = []
        prompts = [f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
                   for doc in docs]
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            inputs = self.tok(batch, return_tensors="pt", padding=True, truncation=True,
                              max_length=self.max_len - len(self.prefix_ids) - len(self.suffix_ids))
            full_ids = [self.prefix_ids + row.tolist() + self.suffix_ids for row in inputs["input_ids"]]
            mx = max(len(x) for x in full_ids)
            pad_id = self.tok.pad_token_id or self.tok.eos_token_id
            input_ids = torch.tensor([[pad_id]*(mx-len(x)) + x for x in full_ids]).cuda()
            attn = (input_ids != pad_id).long()
            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits[:, -1, :]
            yes_l = logits[:, self.yes_id]
            no_l = logits[:, self.no_id]
            probs = torch.softmax(torch.stack([yes_l, no_l], dim=1), dim=1)
            scores.extend(probs[:, 0].cpu().tolist())
        return scores

    def unload(self):
        import torch, gc
        del self.model; del self.tok
        gc.collect(); torch.cuda.empty_cache()


# --- Stage 3: LLM judge (DeepSeek-V3.2 via MAAS) -----------------------------

def llm_judge_one(task_desc: str, skill_name: str, skill_body: str,
                   api_base: str, api_key: str, timeout: int = 40) -> dict:
    import urllib.request, urllib.error
    prompt = LLM_JUDGE_PROMPT_TEMPLATE.format(
        task_description=task_desc[:3000],
        skill_name=skill_name,
        skill_body=skill_body[:3000],
    )
    payload = {
        "model": LLM_JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": "You output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
    }
    req = urllib.request.Request(
        f"{api_base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = json.loads(resp.read())
            content = raw["choices"][0]["message"]["content"].strip()
            # strip markdown fences if any
            m = re.search(r"```(?:json)?\s*(.+?)```", content, re.DOTALL)
            if m: content = m.group(1).strip()
            parsed = json.loads(content)
            return {"score": int(parsed.get("score", 0)),
                    "reason": str(parsed.get("reason", ""))[:200]}
        except Exception as e:
            if attempt == 2:
                return {"score": 0, "reason": f"LLM judge error: {type(e).__name__}"}
            time.sleep(2 ** attempt)


def read_skill_body(skill_path: str, max_chars: int = 3000) -> str:
    p = Path(skill_path) / "SKILL.md"
    if not p.exists(): return ""
    return p.read_text(encoding="utf-8", errors="replace")[:max_chars]


# --- Index loaders -----------------------------------------------------------

def load_embedding_index():
    if not EMB_INDEX_PATH.exists():
        print(f"ERROR: missing {EMB_INDEX_PATH}. Run build_index.py first.")
        sys.exit(1)
    with open(EMB_INDEX_PATH, "rb") as f:
        idx = pickle.load(f)
    print(f"  Loaded emb index: {len(idx['skill_names'])} skills, "
          f"embeddings shape {idx['embeddings'].shape}")
    return idx


def load_bm25_index():
    if not BM25_INDEX_PATH.exists():
        print(f"ERROR: missing {BM25_INDEX_PATH}. Run build_bm25_index.py first.")
        sys.exit(1)
    with open(BM25_INDEX_PATH, "rb") as f:
        idx = pickle.load(f)
    print(f"  Loaded BM25 index: {idx['n_skills']} skills (stemmer={idx['stemmer']})")
    return idx


# --- Main pipeline -----------------------------------------------------------

def run_bench(bench: str, emb_idx: dict, bm25_idx: dict,
              reranker: Qwen3Reranker, embedder: Qwen3Embedder,
              args, out_path: Path) -> int:
    tasks = TASK_LISTERS[bench]()
    print(f"\n=== {bench}: {len(tasks)} tasks ===")
    if not tasks:
        return 0

    # Stage 1a: encode all queries once, score against emb index
    print(f"  [Stage 1a] Encoding {len(tasks)} queries with embedding model...")
    query_embs = embedder.encode_queries(
        [t["task_description"] for t in tasks],
        instruction=EMBEDDING_QUERY_INSTRUCTION,
    )
    emb_matrix = np.asarray(emb_idx["embeddings"])        # (N_skills, d)
    emb_skill_names = emb_idx["skill_names"]
    emb_skill_paths = emb_idx["skill_paths"]

    # Stage 1b: BM25 — tokenize queries (same stemmer as index), score
    print(f"  [Stage 1b] BM25 scoring (stemmer={bm25_idx['stemmer']})...")
    import bm25s
    if bm25_idx["stemmer"] == "none":
        stemmer = None
    else:
        import Stemmer
        stemmer = Stemmer.Stemmer(bm25_idx["stemmer"])
    query_tokens = bm25s.tokenize([t["task_description"] for t in tasks],
                                   stopwords="en", stemmer=stemmer, show_progress=False)
    bm25_retriever = bm25_idx["bm25"]
    bm25_skill_names = bm25_idx["skill_names"]
    bm25_skill_paths = bm25_idx["skill_paths"]
    bm25_scores_batch, bm25_idx_batch = bm25_retriever.retrieve(
        query_tokens, k=args.coarse_k, show_progress=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    MAX_WORKERS = args.llm_concurrency
    api_base = os.environ["MAAS_API_BASE"]
    api_key = os.environ["MAAS_API_KEY"]

    with open(out_path, "w") as fout:
        for t_idx, t in enumerate(tasks):
            q_emb = query_embs[t_idx]
            # 1a. Embedding top-50
            emb_scores = emb_matrix @ q_emb
            top_emb_idx = np.argsort(-emb_scores)[:args.coarse_k]
            emb_top50 = [{
                "rank": r+1,
                "skill_name": emb_skill_names[int(i)],
                "skill_path": emb_skill_paths[int(i)],
                "embedding_score": float(emb_scores[int(i)]),
            } for r, i in enumerate(top_emb_idx)]

            # 1b. BM25 top-50
            bm_top50 = [{
                "rank": r+1,
                "skill_name": bm25_skill_names[int(i)],
                "skill_path": bm25_skill_paths[int(i)],
                "bm25_score": float(bm25_scores_batch[t_idx][r]),
            } for r, i in enumerate(bm25_idx_batch[t_idx])]

            # Union (dedup by skill_name; merge scores)
            union_map: dict[str, dict] = {}
            for item in emb_top50:
                union_map[item["skill_name"]] = dict(item)
            for item in bm_top50:
                if item["skill_name"] in union_map:
                    union_map[item["skill_name"]]["bm25_score"] = item["bm25_score"]
                else:
                    u = dict(item); u["embedding_score"] = None
                    union_map[item["skill_name"]] = u
            union = list(union_map.values())
            print(f"    [{t_idx+1}/{len(tasks)}] {t['task_id']}  "
                  f"emb_top50={len(emb_top50)}  bm25_top50={len(bm_top50)}  union={len(union)}")

            # Stage 2: reranker on all union candidates → top 20
            doc_texts = []
            for c in union:
                body = read_skill_body(c["skill_path"], max_chars=3000)
                doc_texts.append(f"# {c['skill_name']}\n{body}")
            rerank_scores = reranker.score_pairs(t["task_description"], doc_texts)
            for c, rs in zip(union, rerank_scores):
                c["rerank_score"] = float(rs)
            union.sort(key=lambda x: -x["rerank_score"])
            top20 = union[:args.rerank_k]
            for r, c in enumerate(top20):
                c["rank_after_rerank"] = r + 1

            # Stage 3: LLM judge (parallel) → top 10
            if args.skip_llm:
                # simulate with rerank score × 10
                for c in top20:
                    c["llm_score"] = int(round(c["rerank_score"] * 10))
                    c["llm_reason"] = "(skipped)"
            else:
                def judge_one(c):
                    body = read_skill_body(c["skill_path"], max_chars=3000)
                    return c, llm_judge_one(t["task_description"], c["skill_name"], body,
                                             api_base, api_key)

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                    futs = [pool.submit(judge_one, c) for c in top20]
                    for f in as_completed(futs):
                        c, judged = f.result()
                        c["llm_score"] = judged["score"]
                        c["llm_reason"] = judged["reason"]

            top20_sorted = sorted(top20, key=lambda x: -x["llm_score"])
            top10 = top20_sorted[:args.final_k]
            for r, c in enumerate(top10):
                c["rank_final"] = r + 1

            fout.write(json.dumps({
                "task_id": t["task_id"],
                "dataset": t["dataset"],
                "task_description": t["task_description"],
                "embedding_model": EMBEDDING_MODEL,
                "rerank_model": RERANKER_MODEL,
                "llm_judge_model": LLM_JUDGE_MODEL if not args.skip_llm else "SKIPPED",
                "embedding_instruction": EMBEDDING_QUERY_INSTRUCTION,
                "rerank_instruction": RERANKER_INSTRUCTION,
                "llm_judge_prompt_template": LLM_JUDGE_PROMPT_TEMPLATE,
                "stage1_embedding_top50": emb_top50,
                "stage1_bm25_top50": bm_top50,
                "stage1_union": [
                    {k: v for k, v in c.items() if k in
                     ("skill_name", "skill_path", "embedding_score", "bm25_score")}
                    for c in union
                ],
                "stage2_top20_reranked": [
                    {k: v for k, v in c.items() if k in
                     ("rank_after_rerank", "skill_name", "skill_path",
                      "embedding_score", "bm25_score", "rerank_score")}
                    for c in top20
                ],
                "stage3_top10_llm_judge": [
                    {k: v for k, v in c.items() if k in
                     ("rank_final", "skill_name", "skill_path",
                      "embedding_score", "bm25_score", "rerank_score",
                      "llm_score", "llm_reason")}
                    for c in top10
                ],
            }, ensure_ascii=False, default=str) + "\n")
            fout.flush()
            written += 1
            top1 = top10[0]
            print(f"      → top1={top1['skill_name']} rerank={top1['rerank_score']:.3f} llm={top1['llm_score']}/10")
    print(f"=== {bench}: wrote {written} rows to {out_path}")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", nargs="+", required=True,
                    choices=list(TASK_LISTERS.keys()))
    ap.add_argument("--coarse-k", type=int, default=50)
    ap.add_argument("--rerank-k", type=int, default=20)
    ap.add_argument("--final-k", type=int, default=10)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--date", default=time.strftime("%Y%m%d"))
    ap.add_argument("--suffix", default="v8_4stage")
    ap.add_argument("--llm-concurrency", type=int, default=20)
    ap.add_argument("--skip-llm", action="store_true",
                    help="skip Stage 3 (LLM judge); use rerank score * 10 as stub")
    ap.add_argument("--skip-gpu-check", action="store_true")
    args = ap.parse_args()

    # Require MAAS creds unless skipping LLM
    if not args.skip_llm:
        if not os.environ.get("MAAS_API_KEY"):
            print("ERROR: MAAS_API_KEY not set. Source secrets/.env.secrets first.")
            sys.exit(1)

    if not args.skip_gpu_check:
        check_gpu_free(min_free_gb=20)

    print("\n=== Load indexes ===")
    emb_idx = load_embedding_index()
    bm25_idx = load_bm25_index()

    print("\n=== Load embedder + reranker (both on GPU) ===")
    embedder = Qwen3Embedder(EMBEDDING_MODEL)
    reranker = Qwen3Reranker(RERANKER_MODEL)

    out_base = Path(args.out_dir)
    if out_base.name == "experiments":
        out_dir = out_base / args.date / f"{args.date}_{args.suffix}" / "retrieval_results"
    else:
        out_dir = out_base / args.date / "retrieval_results" / args.suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    for bench in args.bench:
        out_path = out_dir / f"{bench}.jsonl"
        run_bench(bench, emb_idx, bm25_idx, reranker, embedder, args, out_path)

    reranker.unload()
    embedder.unload()

    print(f"\n=== DONE — outputs under {out_dir}/ ===")
    for bench in args.bench:
        p = out_dir / f"{bench}.jsonl"
        if p.exists():
            print(f"  {p}  ({sum(1 for _ in open(p))} rows)")


if __name__ == "__main__":
    main()
