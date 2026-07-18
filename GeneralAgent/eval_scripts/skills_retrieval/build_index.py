#!/usr/bin/env python3
"""Build skill embedding index from merged skill SKILL.md files.

Reads all SKILL.md files, extracts name/description/summary, computes
embeddings with BAAI/bge-base-en-v1.5, and saves to skill_index.pkl.

Usage:
    python build_index.py [--skills-dir PATH] [--output PATH] [--model MODEL]
"""

import argparse
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np


DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "skill_libraries" / "merged"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "skill_index.pkl"
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


def parse_skill_md(skill_md_path: Path) -> dict | None:
    """Parse a SKILL.md file and extract name, description, and summary text."""
    try:
        text = skill_md_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  WARNING: Cannot read {skill_md_path}: {e}")
        return None

    result = {
        "name": skill_md_path.parent.name,
        "description": "",
        "summary": "",
        "path": str(skill_md_path.parent),
    }

    # Parse YAML frontmatter
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if fm_match:
        frontmatter = fm_match.group(1)
        # Extract name
        name_match = re.search(r'^name:\s*["\']?(.+?)["\']?\s*$', frontmatter, re.MULTILINE)
        if name_match:
            result["name"] = name_match.group(1).strip()
        # Extract description
        desc_match = re.search(r'^description:\s*["\']?(.+?)["\']?\s*$', frontmatter, re.MULTILINE)
        if desc_match:
            result["description"] = desc_match.group(1).strip().rstrip('"').rstrip("'")
        # Body after frontmatter
        body = text[fm_match.end():]
    else:
        body = text

    # Build summary: description + first ~500 chars of body (stripped of markdown formatting)
    # Remove markdown links, images, code blocks for cleaner text
    clean_body = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
    clean_body = re.sub(r"!\[.*?\]\(.*?\)", "", clean_body)
    clean_body = re.sub(r"\[([^\]]*)\]\([^\)]*\)", r"\1", clean_body)
    clean_body = re.sub(r"#{1,6}\s+", "", clean_body)
    clean_body = re.sub(r"\n{3,}", "\n\n", clean_body)
    clean_body = clean_body.strip()

    # Combine description and body excerpt for embedding
    parts = []
    if result["description"]:
        parts.append(result["description"])
    if clean_body:
        parts.append(clean_body[:500])
    result["summary"] = "\n".join(parts) if parts else result["name"]

    return result


def build_index(
    skills_dir: Path,
    output_path: Path,
    model_name: str,
    doc_instruction: str = "",
    query_instruction: str = "",
):
    """Build and save the skill embedding index."""
    print(f"Skills directory: {skills_dir}")
    print(f"Output path: {output_path}")
    print(f"Embedding model: {model_name}")
    if doc_instruction:
        print(f"Doc instruction:   {doc_instruction[:80]}")
    if query_instruction:
        print(f"Query instruction: {query_instruction[:80]}")
    print()

    # 1. Discover all SKILL.md files
    print("Discovering skills...")
    skill_dirs = sorted([d for d in skills_dir.iterdir() if d.is_dir()])
    print(f"  Found {len(skill_dirs)} skill directories")

    # 2. Parse all SKILL.md files
    print("Parsing SKILL.md files...")
    skills = []
    skipped = 0
    for d in skill_dirs:
        skill_md = d / "SKILL.md"
        if not skill_md.exists():
            skipped += 1
            continue
        parsed = parse_skill_md(skill_md)
        if parsed:
            skills.append(parsed)
        else:
            skipped += 1

    print(f"  Parsed {len(skills)} skills, skipped {skipped}")
    if not skills:
        print("ERROR: No skills found!")
        sys.exit(1)

    # 3. Load embedding model
    print(f"\nLoading embedding model: {model_name}")
    print("  (This may download the model on first run...)")
    t0 = time.time()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # 4. Compute embeddings
    print(f"\nComputing embeddings for {len(skills)} skills...")
    t0 = time.time()
    summaries = [s["summary"] for s in skills]

    # Apply explicit doc_instruction if given (wins over heuristic prefix).
    if doc_instruction:
        summaries = [doc_instruction + s for s in summaries]
    elif "bge" in model_name.lower():
        # bge models recommend prepending for retrieval
        summaries = ["Represent this sentence: " + s for s in summaries]

    embeddings = model.encode(
        summaries,
        show_progress_bar=True,
        batch_size=16,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Computed in {time.time() - t0:.1f}s")

    # 5. Save index
    index = {
        "skill_names": [s["name"] for s in skills],
        "skill_paths": [s["path"] for s in skills],
        "skill_summaries": [s["summary"] for s in skills],
        "embeddings": embeddings,
        "model_name": model_name,
        "doc_instruction": doc_instruction,
        "query_instruction": query_instruction,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nIndex saved to {output_path} ({file_size_mb:.1f} MB)")
    print(f"  {len(skills)} skills indexed")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build skill embedding index")
    parser.add_argument("--skills-dir", type=Path, default=DEFAULT_SKILLS_DIR,
                        help=f"Path to merged skills directory (default: {DEFAULT_SKILLS_DIR})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output pickle file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Embedding model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--doc-instruction", type=str, default="",
                        help="Prefix string to prepend to doc (skill summary) text before embedding.")
    parser.add_argument("--query-instruction", type=str, default="",
                        help="Instruction stored in pickle; retrieve.py uses it on query side.")
    args = parser.parse_args()

    build_index(
        args.skills_dir, args.output, args.model,
        doc_instruction=args.doc_instruction,
        query_instruction=args.query_instruction,
    )
