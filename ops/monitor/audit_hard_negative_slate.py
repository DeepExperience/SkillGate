#!/usr/bin/env python3
"""Sample hard-negative slate candidates for manual pre-promote review.

This helper is intentionally read-only. It compares oracle, old misleading,
and new hard-negative SKILL.md files so a reviewer can check whether candidate
descriptions are attractive but still logically distinguishable before
replacing the production train/eval slate.
"""

from __future__ import annotations

import os

import argparse
import json
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))
OLD_SLATE_DEFAULT = ROOT / "skill_libraries/snapshots/rl/slate_skills_20260704"
NEW_SLATE_DEFAULT = ROOT / "skill_libraries/snapshots/rl/slate_skills_20260706_hard_negative"
FRONTMATTER_RE = re.compile(r"\A---\s*\n(?P<fm>.*?)\n---\s*\n", re.S)
DESC_RE = re.compile(r"(?m)^description:\s*(?P<desc>.*)$")
NEGATIVE_LABEL_RE = re.compile(
    r"\b(hard[- ]negative|negative skill|misleading skill|corrupted skill|"
    r"adversarial skill|decoy skill|trap skill|fake skill|bad skill)\b",
    re.I,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def description(text: str) -> str:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return ""
    desc = DESC_RE.search(match.group("fm"))
    return desc.group("desc").strip().strip("\"'") if desc else ""


def body_excerpt(text: str, max_chars: int) -> str:
    text = FRONTMATTER_RE.sub("", text, count=1).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:max_chars]


def ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return round(SequenceMatcher(None, a.lower(), b.lower()).ratio(), 4)


def task_key(row: dict[str, Any]) -> str:
    return f"{row['bench']}::{row['task_id']}"


def collect_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    old_root = Path(args.old_slate_root)
    new_root = Path(args.new_slate_root)
    rows = read_jsonl(old_root / "manifest" / f"slate_manifest_{args.split}.jsonl")
    records: list[dict[str, Any]] = []
    for row in rows:
        oracle_entry = row["oracle"][0]
        oracle_path = Path(oracle_entry["path"]) / "SKILL.md"
        if not oracle_path.is_absolute():
            oracle_path = ROOT / oracle_path
        oracle_text = read_text(oracle_path)
        for entry in row["misleading"]:
            old_path = Path(entry["path"]) / "SKILL.md"
            if not old_path.is_absolute():
                old_path = ROOT / old_path
            new_path = new_root / "skills" / entry["name"] / "SKILL.md"
            if not new_path.is_file():
                continue
            old_text = read_text(old_path)
            new_text = read_text(new_path)
            oracle_desc = description(oracle_text)
            old_desc = description(old_text)
            new_desc = description(new_text)
            records.append({
                "task_key": task_key(row),
                "bench": row["bench"],
                "task_id": str(row["task_id"]),
                "skill_name": entry["name"],
                "strategy": entry.get("strategy", ""),
                "oracle_path": str(oracle_path),
                "old_misleading_path": str(old_path),
                "new_candidate_path": str(new_path),
                "oracle_description": oracle_desc,
                "old_misleading_description": old_desc,
                "new_candidate_description": new_desc,
                "desc_similarity_to_oracle": ratio(new_desc, oracle_desc),
                "desc_similarity_to_old_misleading": ratio(new_desc, old_desc),
                "full_similarity_to_old_misleading": ratio(new_text, old_text),
                "full_identical_to_old_misleading": re.sub(r"\s+", " ", new_text.strip()) == re.sub(r"\s+", " ", old_text.strip()),
                "length_ratio_to_oracle": round(len(new_text) / max(1, len(oracle_text)), 4),
                "has_obvious_negative_label": bool(NEGATIVE_LABEL_RE.search(new_text)),
                "oracle_excerpt": body_excerpt(oracle_text, args.excerpt_chars),
                "old_misleading_excerpt": body_excerpt(old_text, args.excerpt_chars),
                "new_candidate_excerpt": body_excerpt(new_text, args.excerpt_chars),
            })
    return records


def sample_records(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    if args.per_bench > 0:
        picked: list[dict[str, Any]] = []
        benches = sorted({r["bench"] for r in records})
        for bench in benches:
            group = [r for r in records if r["bench"] == bench]
            rng.shuffle(group)
            picked.extend(group[:args.per_bench])
        return sorted(picked, key=lambda r: (r["bench"], r["task_id"], r["skill_name"]))
    records = list(records)
    rng.shuffle(records)
    return sorted(records[:args.sample_size], key=lambda r: (r["bench"], r["task_id"], r["skill_name"]))


def write_md(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        "# Hard-Negative Slate Manual Audit Sample",
        "",
        "| task | skill | sim(new,oracle desc) | sim(new,old desc) | sim(full,old) | old-identical | len ratio | label flag |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        lines.append(
            f"| {rec['task_key']} | {rec['skill_name']} | "
            f"{rec['desc_similarity_to_oracle']:.3f} | "
            f"{rec['desc_similarity_to_old_misleading']:.3f} | "
            f"{rec['full_similarity_to_old_misleading']:.3f} | "
            f"{rec['full_identical_to_old_misleading']} | "
            f"{rec['length_ratio_to_oracle']:.2f} | "
            f"{rec['has_obvious_negative_label']} |"
        )
    for i, rec in enumerate(records, 1):
        lines.extend([
            "",
            f"## {i}. {rec['task_key']} / {rec['skill_name']}",
            "",
            f"- oracle: `{rec['oracle_path']}`",
            f"- old misleading: `{rec['old_misleading_path']}`",
            f"- new candidate: `{rec['new_candidate_path']}`",
            "",
            "**Descriptions**",
            "",
            f"- oracle: {rec['oracle_description']}",
            f"- old misleading: {rec['old_misleading_description']}",
            f"- new candidate: {rec['new_candidate_description']}",
            "",
            "**Oracle Excerpt**",
            "",
            "```text",
            rec["oracle_excerpt"],
            "```",
            "",
            "**Old Misleading Excerpt**",
            "",
            "```text",
            rec["old_misleading_excerpt"],
            "```",
            "",
            "**New Candidate Excerpt**",
            "",
            "```text",
            rec["new_candidate_excerpt"],
            "```",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-slate-root", default=str(OLD_SLATE_DEFAULT))
    parser.add_argument("--new-slate-root", default=str(NEW_SLATE_DEFAULT))
    parser.add_argument("--split", choices=["eval70", "train"], default="eval70")
    parser.add_argument("--seed", type=int, default=1063810697)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--per-bench", type=int, default=3)
    parser.add_argument("--excerpt-chars", type=int, default=1800)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    records = collect_records(args)
    sample = sample_records(records, args)
    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = ROOT / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "old_slate_root": str(Path(args.old_slate_root)),
        "new_slate_root": str(Path(args.new_slate_root)),
        "split": args.split,
        "available_candidates": len(records),
        "sampled": len(sample),
        "seed": args.seed,
        "records": sample,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out_md = Path(args.out_md)
    if not out_md.is_absolute():
        out_md = ROOT / out_md
    write_md(out_md, sample)
    print(f"[audit] available={len(records)} sampled={len(sample)} json={out_json} md={out_md}")


if __name__ == "__main__":
    main()
