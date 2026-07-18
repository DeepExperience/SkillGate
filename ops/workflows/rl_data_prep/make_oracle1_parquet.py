#!/usr/bin/env python3
"""Build the oracle-top1 RL parquet variant (2026-06-12).

Takes the canonical 4bench factual parquet and, per task:
  1. copies the task's oracle skill dir (latest snapshot) into a FLAT root
     keyed by task_id (task_ids are unique across benches — verified);
  2. replaces the prompt's <available_skills> entries block with a single
     <skill> entry: name=<task_id>, description from the oracle SKILL.md
     frontmatter, location=/root/.claude/skills/<task_id>/SKILL.md;
  3. sets extra_info.retrieval_skills_top_n=[<task_id>] (resolved at rollout
     time through AGENT_BENCH_EXTRA_SKILL_ROOTS pointing at the flat root).

Everything else in the parquet (instructions, reward_model, task_kwargs) is
untouched, and the original parquet/teaching chain is not modified.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BLOCK_RE = re.compile(r"<available_skills>\s*<skill>.*?</available_skills>", re.S)


def frontmatter_description(skill_md: Path) -> str:
    text = skill_md.read_text()
    m = re.search(r"^---\s*\n(.*?)\n---", text, re.S)
    desc = ""
    if m:
        dm = re.search(r"^description:\s*(.+?)(?=\n[a-zA-Z_]+:|\Z)", m.group(1),
                       re.S | re.M)
        if dm:
            desc = " ".join(dm.group(1).split())
    if not desc:
        desc = f"Task-specific oracle skill for {skill_md.parent.name}."
    return desc[:400]


def build_block(task_id: str, desc: str) -> str:
    return (
        "<available_skills>\n"
        "  <skill>\n"
        f"    <name>{task_id}</name>\n"
        f"    <description>{desc}</description>\n"
        f"    <location>/root/.claude/skills/{task_id}/SKILL.md</location>\n"
        "  </skill>\n"
        "</available_skills>"
    )


def transform(df: pd.DataFrame, snapshot: Path, flat_root: Path) -> tuple[pd.DataFrame, list[str]]:
    problems: list[str] = []
    new_prompts, new_extras = [], []
    for _, row in df.iterrows():
        extra = dict(row["extra_info"])
        bench, task_id = str(extra["bench"]), str(extra["task_id"])
        src = snapshot / bench / task_id
        if not (src / "SKILL.md").exists():
            problems.append(f"missing-oracle:{bench}/{task_id}")
            new_prompts.append(row["prompt"]); new_extras.append(extra)
            continue
        dst = flat_root / task_id
        if not dst.exists():
            shutil.copytree(src, dst)
        desc = frontmatter_description(src / "SKILL.md")
        block = build_block(task_id, desc)
        prompt = row["prompt"]
        msgs = list(prompt) if isinstance(prompt, (list, np.ndarray)) else None
        replaced = False
        if msgs is not None:
            out_msgs = []
            for m in msgs:
                m = dict(m)
                c = m.get("content", "")
                if isinstance(c, str) and BLOCK_RE.search(c):
                    m["content"] = BLOCK_RE.sub(block, c, count=1)
                    replaced = True
                out_msgs.append(m)
            new_prompts.append(np.array(out_msgs, dtype=object))
        else:
            text = str(prompt)
            if BLOCK_RE.search(text):
                text = BLOCK_RE.sub(block, text, count=1)
                replaced = True
            new_prompts.append(text)
        if not replaced:
            problems.append(f"no-skills-block:{bench}/{task_id}")
        extra["retrieval_skills_top_n"] = [task_id]
        new_extras.append(extra)
    out = df.copy()
    out["prompt"] = new_prompts
    out["extra_info"] = new_extras
    return out, problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path,
                    default=ROOT / "datasets/rl/parquet_4bench_factual_20260602")
    ap.add_argument("--snapshot", type=Path,
                    default=ROOT / "skill_libraries/snapshots/rl/oracle_skills_full692_20260612")
    ap.add_argument("--flat-root", type=Path,
                    default=ROOT / "skill_libraries/snapshots/rl/oracle_top1_skills_20260612")
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "datasets/rl/parquet_4bench_oracle1_20260612")
    args = ap.parse_args()

    args.flat_root.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for split in ("train", "eval"):
        df = pd.read_parquet(args.input_dir / f"{split}.parquet")
        out, problems = transform(df, args.snapshot, args.flat_root)
        out.to_parquet(args.output_dir / f"{split}.parquet")
        report[split] = {"rows": len(out), "problems": problems}
        print(f"[{split}] rows={len(out)} problems={len(problems)}")
        for p in problems[:10]:
            print("   !", p)
    (args.output_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    n_flat = sum(1 for d in args.flat_root.iterdir() if (d / "SKILL.md").exists())
    print(f"flat oracle root: {args.flat_root} ({n_flat} skills)")


if __name__ == "__main__":
    main()
