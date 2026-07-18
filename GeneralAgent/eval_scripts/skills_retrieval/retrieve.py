#!/usr/bin/env python3
"""Two-stage skill retrieval: embedding coarse filter + LLM fine filter.

Stage 1: Cosine similarity between task embedding and skill embeddings -> top-20
Stage 2: LLM reranking via SGLang -> top 2-5 with reasons

Usage:
    python retrieve.py "Fix a build failure in a Java Maven project"
    python retrieve.py --no-llm "Create a REST API with FastAPI"
    python retrieve.py --top-k 3 "Set up Docker container orchestration"
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Default paths and config
INDEX_PATH = Path(__file__).resolve().parent / "skill_index.pkl"
SGLANG_URL = "http://localhost:30000/v1/chat/completions"
SGLANG_MODEL = "qwen3.5-27b"


def _load_index(index_path: Path = INDEX_PATH) -> dict:
    """Load the skill index from pickle file."""
    if not index_path.exists():
        raise FileNotFoundError(
            f"Skill index not found at {index_path}. "
            "Run build_index.py first to create it."
        )
    with open(index_path, "rb") as f:
        return pickle.load(f)


def _get_embedding_model(model_name: str):
    """Load the sentence-transformers model (cached after first call)."""
    if not hasattr(_get_embedding_model, "_cache"):
        _get_embedding_model._cache = {}
    if model_name not in _get_embedding_model._cache:
        # Avoid HuggingFace online checks (model already downloaded)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        _get_embedding_model._cache[model_name] = SentenceTransformer(model_name)
    return _get_embedding_model._cache[model_name]


def _embed_query(query: str, model_name: str, query_instruction: str = "") -> np.ndarray:
    """Compute normalized embedding for a query string."""
    model = _get_embedding_model(model_name)
    if query_instruction:
        query = query_instruction + query
    elif "bge" in model_name.lower():
        query = "Represent this sentence for searching relevant documents: " + query
    emb = model.encode([query], normalize_embeddings=True)
    return np.array(emb, dtype=np.float32)[0]


def _cosine_topk(query_emb: np.ndarray, index_embs: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return top-k indices and scores by cosine similarity.

    Assumes both query_emb and index_embs are L2-normalized,
    so cosine similarity = dot product.
    """
    scores = index_embs @ query_emb  # (N,)
    top_indices = np.argsort(scores)[::-1][:k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


def _llm_rerank(
    task_description: str,
    candidates: list[dict],
    top_k: int = 5,
    sglang_url: str = SGLANG_URL,
    sglang_model: str = SGLANG_MODEL,
) -> Optional[list[dict]]:
    """Use LLM to rerank candidate skills and select the most relevant ones.

    Returns list of dicts with keys: skill_name, skill_path, relevance_score, reason.
    Returns None if LLM is unavailable or fails.
    """
    import urllib.request
    import urllib.error

    # Build candidate list for prompt
    candidate_text = ""
    for i, c in enumerate(candidates, 1):
        # Truncate summary to keep prompt manageable
        summary = c["summary"][:300].replace("\n", " ").strip()
        candidate_text += f"{i}. **{c['name']}** (score: {c['embedding_score']:.3f})\n   {summary}\n\n"

    prompt = f"""Given a task and candidate skills, select the {top_k} most relevant. Output ONLY a JSON array.

Task: {task_description}

Candidates:
{candidate_text}
Select {top_k} most relevant. JSON array format:
[{{"rank":1,"candidate_number":<n>,"skill_name":"<name>","reason":"<1 sentence>"}}]"""

    payload = {
        "model": sglang_model,
        "messages": [
            {"role": "system", "content": "You output valid JSON only. No explanations, no markdown."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    # Set NO_PROXY to avoid proxy issues with localhost
    env_backup = os.environ.get("NO_PROXY", "")
    os.environ["NO_PROXY"] = "127.0.0.1,localhost,0.0.0.0"
    no_proxy_lower_backup = os.environ.get("no_proxy", "")
    os.environ["no_proxy"] = "127.0.0.1,localhost,0.0.0.0"

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            sglang_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        content = result["choices"][0]["message"]["content"].strip()
        # Strip markdown code fences if present
        if "```" in content:
            import re as _re
            fence_match = _re.search(r"```(?:json)?\s*\n?(.*?)```", content, _re.DOTALL)
            if fence_match:
                content = fence_match.group(1).strip()
        # Try to find JSON array in the content
        if not content.startswith("["):
            bracket_start = content.find("[")
            bracket_end = content.rfind("]")
            if bracket_start != -1 and bracket_end != -1:
                content = content[bracket_start:bracket_end + 1]

        selections = json.loads(content)

        # Map back to full candidate info
        reranked = []
        for sel in selections:
            cand_num = sel["candidate_number"] - 1  # 0-indexed
            if 0 <= cand_num < len(candidates):
                cand = candidates[cand_num]
                reranked.append({
                    "skill_name": cand["name"],
                    "skill_path": cand["path"],
                    "relevance_score": cand["embedding_score"],
                    "reason": sel.get("reason", "Selected by LLM reranking"),
                })
        return reranked if reranked else None

    except urllib.error.URLError as e:
        print(f"  [LLM rerank] SGLang unavailable: {e}", file=sys.stderr)
        return None
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        print(f"  [LLM rerank] Failed to parse LLM response: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [LLM rerank] Unexpected error: {e}", file=sys.stderr)
        return None
    finally:
        os.environ["NO_PROXY"] = env_backup
        os.environ["no_proxy"] = no_proxy_lower_backup


def retrieve_skills(
    task_description: str,
    top_k: int = 5,
    use_llm_rerank: bool = True,
    index_path: Path = INDEX_PATH,
    coarse_k: int = 20,
    sglang_url: str = SGLANG_URL,
    sglang_model: str = SGLANG_MODEL,
    return_coarse: bool = False,
) -> list[dict]:
    """Retrieve top-k skills for a task description.

    Two-stage retrieval:
      1. Embedding cosine similarity -> top-20 candidates
      2. LLM reranking (optional) -> top-k with explanations

    Args:
        task_description: Natural language description of the task.
        top_k: Number of skills to return (2-5 recommended).
        use_llm_rerank: Whether to use LLM for reranking stage 2.
        index_path: Path to the skill_index.pkl file.
        coarse_k: Number of candidates from embedding stage.
        sglang_url: URL of the SGLang API endpoint.
        sglang_model: Model name for SGLang.

    Returns:
        List of dicts, each with keys:
          - skill_name: Name of the skill
          - skill_path: Path to the skill directory
          - relevance_score: Cosine similarity score
          - reason: Why the skill is relevant (from LLM or "Embedding similarity")
    """
    # Load index
    index = _load_index(index_path)
    model_name = index["model_name"]
    query_instruction = index.get("query_instruction", "")

    # Stage 1: Embedding coarse filter
    query_emb = _embed_query(task_description, model_name, query_instruction)
    top_results = _cosine_topk(query_emb, index["embeddings"], coarse_k)

    candidates = []
    for idx, score in top_results:
        candidates.append({
            "name": index["skill_names"][idx],
            "path": index["skill_paths"][idx],
            "summary": index["skill_summaries"][idx],
            "embedding_score": score,
        })

    coarse_out = [
        {"rank": i + 1,
         "skill_name": c["name"],
         "skill_path": c["path"],
         "embedding_score": c["embedding_score"]}
        for i, c in enumerate(candidates)
    ]

    # Stage 2: LLM reranking
    reranked = None
    if use_llm_rerank:
        reranked = _llm_rerank(
            task_description, candidates, top_k,
            sglang_url=sglang_url, sglang_model=sglang_model,
        )
        if reranked is None:
            print("  [Fallback] Using embedding-only results", file=sys.stderr)

    if reranked is None:
        reranked = []
        for cand in candidates[:top_k]:
            reranked.append({
                "skill_name": cand["name"],
                "skill_path": cand["path"],
                "relevance_score": cand["embedding_score"],
                "reason": "Embedding similarity",
            })

    if return_coarse:
        return {"coarse_top": coarse_out, "reranked_top": reranked}
    return reranked


def main():
    parser = argparse.ArgumentParser(description="Retrieve relevant skills for a task")
    parser.add_argument("task", nargs="?", default="Write a Python script to process CSV files",
                        help="Task description")
    parser.add_argument("--top-k", type=int, default=5, help="Number of skills to return")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM reranking")
    parser.add_argument("--coarse-k", type=int, default=20, help="Coarse filter top-k")
    parser.add_argument("--index", type=Path, default=INDEX_PATH, help="Path to index file")
    parser.add_argument("--show-coarse", action="store_true", help="Also show coarse embedding results")
    args = parser.parse_args()

    print(f"Task: {args.task}")
    print(f"Top-k: {args.top_k}, LLM rerank: {not args.no_llm}")
    print()

    if args.show_coarse:
        # Show raw embedding results
        print("--- Stage 1: Embedding coarse filter (top-20) ---")
        coarse_results = retrieve_skills(
            args.task, top_k=args.coarse_k, use_llm_rerank=False,
            index_path=args.index, coarse_k=args.coarse_k,
        )
        for i, r in enumerate(coarse_results, 1):
            print(f"  {i:2d}. [{r['relevance_score']:.3f}] {r['skill_name']}")
        print()

    print("--- Retrieved Skills ---")
    t0 = time.time()
    results = retrieve_skills(
        args.task,
        top_k=args.top_k,
        use_llm_rerank=not args.no_llm,
        index_path=args.index,
        coarse_k=args.coarse_k,
    )
    elapsed = time.time() - t0

    for r in results:
        print(f"  [{r['relevance_score']:.3f}] {r['skill_name']}")
        print(f"    Path: {r['skill_path']}")
        print(f"    Reason: {r['reason']}")
        print()

    print(f"(Retrieved {len(results)} skills in {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
