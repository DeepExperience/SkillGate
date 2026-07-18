#!/usr/bin/env python3
"""Build BM25 sparse index over all skill_libraries/merged/*/SKILL.md texts.

Uses bm25s (the 2024-2026 SOTA Python BM25 implementation, 10-500x faster than
rank_bm25). Stores to `skill_index_bm25.pkl` as a pickled tuple:
    (bm25s.BM25 instance, skill_names: list[str], skill_paths: list[str])

Usage:
    python build_bm25_index.py            # default skill_libraries/merged
    python build_bm25_index.py --stemmer english
"""
import argparse
import pickle
import re
import time
from pathlib import Path

import bm25s


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SKILLS_DIR = SCRIPT_DIR.parent.parent.parent / "skill_libraries" / "merged"
DEFAULT_OUTPUT = SCRIPT_DIR / "skill_index_bm25.pkl"


def read_skill_text(skill_dir: Path) -> str:
    """Concatenate all .md files in a skill dir for indexing."""
    parts = []
    for md in sorted(skill_dir.rglob("*.md")):
        try:
            txt = md.read_text(encoding="utf-8", errors="replace")
            # Strip YAML frontmatter (we keep description which is most important,
            # but don't want raw YAML syntax polluting tokenization)
            fm = re.match(r"^---\s*\n(.*?)\n---\s*\n", txt, re.DOTALL)
            if fm:
                # extract description from frontmatter as plain text
                desc_m = re.search(r'^description:\s*["\']?(.+?)["\']?\s*$',
                                   fm.group(1), re.MULTILINE | re.DOTALL)
                desc = desc_m.group(1).strip() if desc_m else ""
                body = txt[fm.end():]
                parts.append(desc + "\n" + body)
            else:
                parts.append(txt)
        except Exception:
            continue
    # Cap per-skill text at ~20K chars so giant rollup files don't dominate
    return "\n".join(parts)[:20000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills-dir", type=Path, default=DEFAULT_SKILLS_DIR)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--stemmer", type=str, default="english",
                    help="stemmer language; 'none' to disable")
    args = ap.parse_args()

    print(f"Skills dir: {args.skills_dir}")
    print(f"Output:     {args.output}")

    t0 = time.time()
    skill_dirs = sorted(d for d in args.skills_dir.iterdir()
                        if d.is_dir() and (d / "SKILL.md").exists())
    print(f"Discovered {len(skill_dirs)} skill folders")

    skill_names, skill_paths, texts = [], [], []
    for sd in skill_dirs:
        txt = read_skill_text(sd)
        if not txt.strip():
            continue
        skill_names.append(sd.name)
        skill_paths.append(str(sd))
        texts.append(txt)
    print(f"Collected {len(texts)} non-empty texts in {time.time()-t0:.1f}s")

    # Tokenize with bm25s built-in tokenizer (needs PyStemmer object, not str)
    print(f"Tokenizing (stemmer={args.stemmer})...")
    t1 = time.time()
    if args.stemmer == "none":
        stemmer = None
    else:
        import Stemmer
        stemmer = Stemmer.Stemmer(args.stemmer)
    corpus_tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer,
                                   show_progress=True)
    print(f"Tokenized in {time.time()-t1:.1f}s")

    # Fit BM25
    print("Fitting BM25...")
    t2 = time.time()
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    print(f"Indexed in {time.time()-t2:.1f}s")

    # Save as bundle
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({
            "bm25": retriever,
            "skill_names": skill_names,
            "skill_paths": skill_paths,
            "stemmer": args.stemmer,
            "n_skills": len(skill_names),
        }, f)
    print(f"Saved {args.output} ({args.output.stat().st_size / 1024:.1f} KB)")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
