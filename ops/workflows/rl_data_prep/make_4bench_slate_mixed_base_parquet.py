#!/usr/bin/env python3
"""Build a mixed-slate outcome-only GRPO control parquet.

This is the control for SlateRL regret runs:

* train.parquet contains only the mixed-slate prompt arm from the existing
  SlateRL pair data; no no-skill paired arm is present.
* rewards are unchanged outcome rewards; the launcher must use normal GRPO and
  the standard nonzero-std dynamic sampler.
* eval.parquet converts the current no-skill internal eval rows into mixed
  slate prompts using the eval70 slate manifest, so internal eval matches the
  mixed-prompt train setting.

The training split intentionally preserves the gold-present/gold-absent slate
distribution of ``parquet_4bench_slate_regret_20260704``. That makes this a
clean "train the slate arm alone" control against the paired regret run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---", re.S)


def _plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, dict):
            return dict(item)
    raise TypeError(f"extra_info is not a dict: {type(value)!r}")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _seeded_rng(tag: str, bench: str, task_id: str) -> random.Random:
    seed = int(hashlib.sha256(f"{tag}::{bench}::{task_id}".encode()).hexdigest()[:12], 16)
    return random.Random(seed)


def _frontmatter_description(skill_md: Path) -> str:
    text = skill_md.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    desc = ""
    if match:
        desc_match = re.search(r"^description:\s*(.+?)(?=\n[a-zA-Z_]+:|\Z)", match.group(1), re.S | re.M)
        if desc_match:
            desc = " ".join(desc_match.group(1).split()).strip("\"'")
    if not desc:
        desc = f"Skill document {skill_md.parent.name}."
    return desc[:300]


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[f"{row['bench']}::{row['task_id']}"] = row
    return rows


def _build_slate_block(entries: list[dict[str, Any]]) -> str:
    parts = ["<available_skills>"]
    for entry in entries:
        parts.append(
            "  <skill>\n"
            f"    <name>{entry['name']}</name>\n"
            f"    <description>{entry['description']}</description>\n"
            f"    <location>/root/.claude/skills/{entry['name']}/SKILL.md</location>\n"
            "  </skill>"
        )
    parts.append("</available_skills>")
    return "\n".join(parts)


def _entries_from_manifest(mrow: dict[str, Any], *, shuffle_tag: str) -> list[dict[str, Any]]:
    bench = str(mrow["bench"])
    task_id = str(mrow["task_id"])
    entries = (
        list(mrow.get("oracle") or [])
        + list(mrow.get("misleading") or [])
        + list(mrow.get("relevant") or [])
        + list(mrow.get("irrelevant") or [])
    )
    _seeded_rng(shuffle_tag, bench, task_id).shuffle(entries)
    out: list[dict[str, Any]] = []
    for entry in entries:
        entry = dict(entry)
        md = _resolve_repo_path(entry["path"]) / "SKILL.md"
        entry["description"] = _frontmatter_description(md) if md.is_file() else f"Skill document {entry['name']}."
        out.append(entry)
    return out


def _messages_from_prompt(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, np.ndarray):
        prompt = list(prompt)
    if isinstance(prompt, (list, tuple)):
        return [dict(m) if isinstance(m, dict) else {"role": "user", "content": str(m)} for m in prompt]
    return [{"role": "user", "content": "" if prompt is None else str(prompt)}]


def _insert_slate_block(prompt: Any, block: str) -> tuple[np.ndarray, bool]:
    messages = _messages_from_prompt(prompt)
    inserted = False
    out = []
    for msg in messages:
        msg = dict(msg)
        content = msg.get("content", "")
        if (
            not inserted
            and msg.get("role") == "system"
            and isinstance(content, str)
            and "<available_skills>" not in content
        ):
            marker = "\n## Runtime\n"
            idx = content.find(marker)
            if idx >= 0:
                msg["content"] = content[:idx].rstrip() + "\n" + block + "\n" + content[idx:]
            else:
                msg["content"] = content.rstrip() + "\n" + block + "\n"
            inserted = True
        out.append(msg)
    return np.array(out, dtype=object), inserted


def _normalize_train_extra(extra: dict[str, Any]) -> dict[str, Any]:
    extra = dict(extra)
    extra["update_kind"] = "mixed_grpo"
    extra["hybrid_update_kind"] = "mixed_grpo"
    extra["hybrid_is_shadow"] = 0.0
    extra["hybrid_grpo_weight"] = 1.0
    extra["hybrid_shadow_weight"] = 0.0
    extra["slate_mixed_base_control"] = 1.0
    extra.pop("relax_pair_role", None)
    extra.pop("relax_pair_task_key", None)
    extra.pop("relax_pair_group_index", None)
    return extra


def _normalize_eval_extra(extra: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    extra = dict(extra)
    extra["update_kind"] = "mixed_eval"
    extra["hybrid_update_kind"] = "mixed_eval"
    extra["retrieval_skills_top_n"] = [entry["name"] for entry in entries]
    extra["hybrid_is_shadow"] = 0.0
    extra["hybrid_grpo_weight"] = 0.0
    extra["hybrid_shadow_weight"] = 0.0
    extra["slate_mixed_base_control"] = 1.0
    extra["slate_contains_gold"] = 1.0
    extra["slate_gold_name"] = next((entry["name"] for entry in entries if entry.get("category") == "oracle"), "")
    extra["slate_size"] = float(len(entries))
    return extra


def build(args: argparse.Namespace) -> dict[str, Any]:
    train_src = pd.read_parquet(args.slate_pair_dir / "train.parquet")
    eval_src = pd.read_parquet(args.slate_pair_dir / "eval.parquet")
    train_manifest = _load_manifest(args.train_manifest)
    eval_manifest = _load_manifest(args.eval_manifest)

    train_rows = []
    train_stats: Counter[str] = Counter()
    for _, row in train_src.iterrows():
        extra = _plain_dict(row["extra_info"])
        if str(extra.get("update_kind") or "").strip().lower() != "slate_grpo":
            continue
        names = _as_list(extra.get("retrieval_skills_top_n"))
        if not (10 <= len(names) <= 20):
            raise SystemExit(f"bad slate skill count for train {extra.get('bench')}::{extra.get('task_id')}: {len(names)}")
        messages = _messages_from_prompt(row["prompt"])
        text = "\n".join(str(m.get("content", "")) for m in messages)
        if "<available_skills>" not in text:
            raise SystemExit(f"train slate prompt missing <available_skills>: {extra.get('bench')}::{extra.get('task_id')}")
        new_row = row.copy()
        new_row["extra_info"] = _normalize_train_extra(extra)
        train_rows.append(new_row)
        key = f"{extra.get('bench')}::{extra.get('task_id')}"
        train_stats["rows"] += 1
        train_stats[f"bench:{extra.get('bench')}"] += 1
        train_stats["gold_present" if float(extra.get("slate_contains_gold") or 0.0) >= 1.0 else "gold_absent"] += 1
        if key not in train_manifest:
            train_stats["missing_train_manifest_but_kept"] += 1

    eval_prompts = []
    eval_extras = []
    eval_stats: Counter[str] = Counter()
    missing_eval: list[str] = []
    for _, row in eval_src.iterrows():
        extra = _plain_dict(row["extra_info"])
        bench = str(extra["bench"])
        task_id = str(extra["task_id"])
        key = f"{bench}::{task_id}"
        mrow = eval_manifest.get(key)
        if mrow is None:
            missing_eval.append(key)
            continue
        entries = _entries_from_manifest(mrow, shuffle_tag="slate-mixed-base-eval-shuffle")
        if len(entries) != 16:
            raise SystemExit(f"eval manifest slate size is not 16 for {key}: {len(entries)}")
        prompt, inserted = _insert_slate_block(row["prompt"], _build_slate_block(entries))
        if not inserted:
            raise SystemExit(f"failed to insert slate block for eval {key}")
        eval_prompts.append(prompt)
        eval_extras.append(_normalize_eval_extra(extra, entries))
        eval_stats["rows"] += 1
        eval_stats[f"bench:{bench}"] += 1

    if missing_eval:
        raise SystemExit(f"missing eval manifest rows: {missing_eval[:20]}")
    if not train_rows:
        raise SystemExit("no slate_grpo train rows found")

    train_out = pd.DataFrame(train_rows).reset_index(drop=True)
    eval_out = eval_src.copy().iloc[: len(eval_prompts)].reset_index(drop=True)
    eval_out["prompt"] = eval_prompts
    eval_out["extra_info"] = eval_extras

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_out.to_parquet(args.output_dir / "train.parquet")
    eval_out.to_parquet(args.output_dir / "eval.parquet")
    report = {
        "source_slate_pair_dir": str(args.slate_pair_dir),
        "train_manifest": str(args.train_manifest),
        "eval_manifest": str(args.eval_manifest),
        "train": dict(train_stats),
        "eval": dict(eval_stats),
        "notes": [
            "train keeps only existing slate_grpo mixed prompts from the pair data",
            "eval converts internal no-skill eval rows to mixed slate prompts with all 16 eval70 skills",
            "launcher must use outcome-only reward and standard GRPO",
        ],
    }
    (args.output_dir / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    return report


def validate(args: argparse.Namespace) -> dict[str, Any]:
    report_path = args.output_dir / "build_report.json"
    if not report_path.is_file():
        raise SystemExit(f"missing {report_path}")
    report = json.loads(report_path.read_text())
    out: dict[str, Any] = {"report": report}
    for split, expected_kind in (("train", "mixed_grpo"), ("eval", "mixed_eval")):
        path = args.output_dir / f"{split}.parquet"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        df = pd.read_parquet(path)
        kinds: Counter[str] = Counter()
        skill_sizes: Counter[int] = Counter()
        missing_blocks = 0
        old_oracle_blocks = 0
        for _, row in df.iterrows():
            extra = _plain_dict(row["extra_info"])
            kind = str(extra.get("update_kind") or "")
            kinds[kind] += 1
            names = _as_list(extra.get("retrieval_skills_top_n"))
            skill_sizes[len(names)] += 1
            text = "\n".join(str(m.get("content", "")) for m in _messages_from_prompt(row["prompt"]))
            if "<available_skills>" not in text:
                missing_blocks += 1
            if "## Skills (mandatory)" in text or "preloaded_oracle_skill" in text:
                old_oracle_blocks += 1
        if set(kinds) != {expected_kind}:
            raise SystemExit(f"{split}: expected only {expected_kind}, saw {dict(kinds)}")
        if missing_blocks or old_oracle_blocks:
            raise SystemExit(f"{split}: missing_blocks={missing_blocks}, old_oracle_blocks={old_oracle_blocks}")
        if split == "train" and not all(10 <= size <= 20 for size in skill_sizes):
            raise SystemExit(f"{split}: bad slate sizes {dict(skill_sizes)}")
        if split == "eval" and set(skill_sizes) != {16}:
            raise SystemExit(f"{split}: bad eval slate sizes {dict(skill_sizes)}")
        out[split] = {
            "rows": len(df),
            "kinds": dict(kinds),
            "skill_sizes": dict(skill_sizes),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--slate-pair-dir",
        type=Path,
        default=ROOT / "datasets/rl/parquet_4bench_slate_regret_20260704",
    )
    ap.add_argument(
        "--train-manifest",
        type=Path,
        default=ROOT / "skill_libraries/snapshots/rl/slate_skills_20260704/manifest/slate_manifest_train.jsonl",
    )
    ap.add_argument(
        "--eval-manifest",
        type=Path,
        default=ROOT / "skill_libraries/snapshots/rl/slate_skills_20260704/manifest/slate_manifest_eval70.jsonl",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "datasets/rl/parquet_4bench_slate_mixed_base_20260706",
    )
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    result = validate(args) if args.validate_only else build(args)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
