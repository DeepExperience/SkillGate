#!/usr/bin/env python3
"""Build mixed no-skill-GRPO + oracle-shadow parquet for hybrid RL.

Train split contains two rows per task:

- update_kind=no_skill_grpo: no advertised skill, no retrieval skill.
- update_kind=oracle_shadow: oracle top-1 skill advertised for rollout; M1
  cleaning converts the trajectory to a no-skill transcript before training.
- update_kind=oracle_prompt_bc, with ``--oracle-mode direct_text``: full oracle
  skill text is preloaded into the rollout system prompt; prompt-only cleaning
  removes that private context before weighted BC/AWR training.

Eval split is no-skill only.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "datasets/rl/parquet_4bench_factual_20260602"
DEFAULT_SNAPSHOT = ROOT / "skill_libraries/snapshots/rl/oracle_skills_full692_20260612"
DEFAULT_FLAT_ROOT = ROOT / "skill_libraries/snapshots/rl/oracle_top1_skills_20260612"
DEFAULT_OUTPUT = ROOT / "datasets/rl/parquet_4bench_m1_hybrid_oracle_shadow_noskill_grpo_20260621"
DEFAULT_DIRECT_OUTPUT = ROOT / "datasets/rl/parquet_4bench_oracle_promptbc_pair_noskill_grpo_20260623"

SKILLS_SECTION_RE = re.compile(r"\n## Skills \(mandatory\)\n.*?(?=\n## Memory Recall\n)", re.S)
AVAILABLE_SKILLS_RE = re.compile(r"<available_skills>.*?</available_skills>", re.S)
ORACLE_BLOCK_RE = re.compile(r"<available_skills>\s*<skill>.*?</available_skills>", re.S)
SKILL_LOCATION_RE = re.compile(r"/root/\.claude/skills/[^/\s<>]+/SKILL\.md")
PRELOADED_ORACLE_RE = re.compile(r"<preloaded_oracle_skill>.*?</preloaded_oracle_skill>", re.S)


def _as_messages(prompt: Any) -> list | None:
    if isinstance(prompt, (list, tuple, np.ndarray)):
        return list(prompt)
    return None


def _prompt_text(prompt: Any) -> str:
    messages = _as_messages(prompt)
    if messages is not None:
        return "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict)
        )
    return "" if prompt is None else str(prompt)


def _has_values(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, np.ndarray)):
        return len(value) > 0
    return bool(value)


def _bench_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for extra in frame["extra_info"]:
        bench = str(dict(extra or {}).get("bench") or "unknown")
        counts[bench] = counts.get(bench, 0) + 1
    return dict(sorted(counts.items()))


def _frontmatter_description(skill_md: Path) -> str:
    text = skill_md.read_text(encoding="utf-8")
    match = re.search(r"^---\s*\n(.*?)\n---", text, re.S)
    desc = ""
    if match:
        desc_match = re.search(
            r"^description:\s*(.+?)(?=\n[a-zA-Z_]+:|\Z)",
            match.group(1),
            re.S | re.M,
        )
        if desc_match:
            desc = " ".join(desc_match.group(1).split())
    if not desc:
        desc = f"Task-specific oracle skill for {skill_md.parent.name}."
    return desc[:400]


def _oracle_block(task_id: str, desc: str) -> str:
    return (
        "<available_skills>\n"
        "  <skill>\n"
        f"    <name>{task_id}</name>\n"
        f"    <description>{desc}</description>\n"
        f"    <location>/root/.claude/skills/{task_id}/SKILL.md</location>\n"
        "  </skill>\n"
        "</available_skills>"
    )


def _direct_oracle_block(task_id: str, skill_text: str) -> str:
    return (
        "## Preloaded Oracle Skill\n\n"
        "The following task-specific oracle skill is private rollout context. "
        "Use it directly; do not call tools to read any SKILL.md file for this skill.\n\n"
        "<preloaded_oracle_skill>\n"
        f"<name>{task_id}</name>\n"
        "<content>\n"
        f"{skill_text.strip()}\n"
        "</content>\n"
        "</preloaded_oracle_skill>\n"
    )


def _strip_skill_section(prompt: Any) -> tuple[Any, bool]:
    messages = _as_messages(prompt)
    removed = False
    if messages is not None:
        out = []
        for message in messages:
            if isinstance(message, dict):
                new_message = dict(message)
                content = new_message.get("content")
                if isinstance(content, str):
                    content, count = SKILLS_SECTION_RE.subn("\n", content, count=1)
                    removed = removed or bool(count)
                    new_message["content"] = content
                out.append(new_message)
            else:
                out.append(message)
        return np.array(out, dtype=object), removed

    text = "" if prompt is None else str(prompt)
    text, count = SKILLS_SECTION_RE.subn("\n", text, count=1)
    return text, bool(count)


def _replace_skill_section(prompt: Any, replacement: str) -> tuple[Any, bool]:
    messages = _as_messages(prompt)
    replaced = False
    if messages is not None:
        out = []
        for message in messages:
            if isinstance(message, dict):
                new_message = dict(message)
                content = new_message.get("content")
                if isinstance(content, str):
                    content, count = SKILLS_SECTION_RE.subn(lambda _m: "\n" + replacement + "\n", content, count=1)
                    replaced = replaced or bool(count)
                    new_message["content"] = content
                out.append(new_message)
            else:
                out.append(message)
        return np.array(out, dtype=object), replaced

    text = "" if prompt is None else str(prompt)
    text, count = SKILLS_SECTION_RE.subn(lambda _m: "\n" + replacement + "\n", text, count=1)
    return text, bool(count)


def _replace_oracle_block(prompt: Any, block: str) -> tuple[Any, bool]:
    messages = _as_messages(prompt)
    replaced = False
    if messages is not None:
        out = []
        for message in messages:
            if isinstance(message, dict):
                new_message = dict(message)
                content = new_message.get("content")
                if isinstance(content, str) and ORACLE_BLOCK_RE.search(content):
                    new_message["content"] = ORACLE_BLOCK_RE.sub(lambda _m: block, content, count=1)
                    replaced = True
                out.append(new_message)
            else:
                out.append(message)
        return np.array(out, dtype=object), replaced

    text = "" if prompt is None else str(prompt)
    if ORACLE_BLOCK_RE.search(text):
        text = ORACLE_BLOCK_RE.sub(lambda _m: block, text, count=1)
        replaced = True
    return text, replaced


def _make_extra(
    extra: dict[str, Any],
    *,
    update_kind: str,
    task_id: str | None = None,
    oracle_mode: str | None = None,
) -> dict[str, Any]:
    out = dict(extra)
    out["update_kind"] = update_kind
    out["hybrid_update_kind"] = update_kind
    if update_kind == "oracle_shadow":
        assert task_id is not None
        out["retrieval_skills_top_n"] = [task_id]
        out["hybrid_is_shadow"] = 1.0
        out["hybrid_grpo_weight"] = 0.0
        out["hybrid_shadow_weight"] = 1.0
        out["oracle_skill_mode"] = oracle_mode or "skill_path"
    elif update_kind == "oracle_prompt_bc":
        out["retrieval_skills_top_n"] = []
        out["hybrid_is_shadow"] = 1.0
        out["hybrid_grpo_weight"] = 0.0
        # Pair-gated reward postprocess enables this only when the paired
        # no-skill group is all-fail and the oracle group has success.
        out["hybrid_shadow_weight"] = 0.0
        out["oracle_skill_mode"] = oracle_mode or "direct_text"
    else:
        out["retrieval_skills_top_n"] = []
        out["hybrid_is_shadow"] = 0.0
        out["hybrid_grpo_weight"] = 1.0
        out["hybrid_shadow_weight"] = 0.0
    return out


def _copy_oracle_skill(snapshot: Path, flat_root: Path, bench: str, task_id: str) -> tuple[Path, str]:
    src = snapshot / bench / task_id
    skill_md = src / "SKILL.md"
    if not skill_md.exists():
        raise FileNotFoundError(f"missing oracle skill: {bench}/{task_id}")
    dst = flat_root / task_id
    if not dst.exists():
        shutil.copytree(src, dst)
    return skill_md, _frontmatter_description(skill_md)


def build_train(
    frame: pd.DataFrame,
    snapshot: Path,
    flat_root: Path,
    *,
    oracle_mode: str = "skill_path",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[pd.Series] = []
    problems: list[str] = []
    removed_no_skill = 0
    replaced_oracle = 0
    direct_oracle = 0

    for row_index, row in frame.iterrows():
        extra = dict(row["extra_info"] or {})
        bench = str(extra.get("bench") or "unknown")
        task_id = str(extra.get("task_id") or "unknown")

        no_skill_row = row.copy(deep=True)
        no_skill_prompt, removed = _strip_skill_section(row["prompt"])
        removed_no_skill += int(removed)
        if not removed:
            problems.append(f"missing-skills-section-noskill:{bench}/{task_id}@row{row_index}")
        no_skill_row["prompt"] = no_skill_prompt
        no_skill_row["extra_info"] = _make_extra(extra, update_kind="no_skill_grpo")
        rows.append(no_skill_row)

        oracle_row = row.copy(deep=True)
        try:
            _skill_md, desc = _copy_oracle_skill(snapshot, flat_root, bench, task_id)
        except FileNotFoundError as exc:
            problems.append(str(exc))
            continue
        if oracle_mode == "direct_text":
            skill_text = _skill_md.read_text(encoding="utf-8")
            oracle_prompt, replaced = _replace_skill_section(row["prompt"], _direct_oracle_block(task_id, skill_text))
            direct_oracle += int(replaced)
            update_kind = "oracle_prompt_bc"
        else:
            oracle_prompt, replaced = _replace_oracle_block(row["prompt"], _oracle_block(task_id, desc))
            update_kind = "oracle_shadow"
        replaced_oracle += int(replaced)
        if not replaced:
            problems.append(f"no-skills-block-oracle:{bench}/{task_id}@row{row_index}")
        oracle_row["prompt"] = oracle_prompt
        oracle_row["extra_info"] = _make_extra(
            extra,
            update_kind=update_kind,
            task_id=task_id,
            oracle_mode=oracle_mode,
        )
        rows.append(oracle_row)

    out = pd.DataFrame(rows).reset_index(drop=True)
    report = {
        "rows_in": int(len(frame)),
        "rows_out": int(len(out)),
        "bench_counts": _bench_counts(out),
        "update_kind_counts": {
            str(kind): int(count)
            for kind, count in out["extra_info"].map(lambda x: dict(x or {}).get("update_kind")).value_counts().items()
        },
        "removed_no_skill_sections": int(removed_no_skill),
        "replaced_oracle_blocks": int(replaced_oracle),
        "direct_oracle_sections": int(direct_oracle),
        "oracle_mode": oracle_mode,
        "problems": problems,
    }
    return out, report


def build_eval(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    prompts = []
    extras = []
    problems: list[str] = []
    removed_sections = 0

    for row_index, row in frame.iterrows():
        extra = dict(row["extra_info"] or {})
        bench = str(extra.get("bench") or "unknown")
        task_id = str(extra.get("task_id") or "unknown")
        prompt, removed = _strip_skill_section(row["prompt"])
        removed_sections += int(removed)
        if not removed:
            problems.append(f"missing-skills-section-eval:{bench}/{task_id}@row{row_index}")
        prompts.append(prompt)
        extras.append(_make_extra(extra, update_kind="no_skill_eval"))

    out = frame.copy()
    out["prompt"] = prompts
    out["extra_info"] = extras
    report = {
        "rows_in": int(len(frame)),
        "rows_out": int(len(out)),
        "bench_counts": _bench_counts(out),
        "removed_no_skill_sections": int(removed_sections),
        "problems": problems,
    }
    return out, report


def validate_output(output_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"output_dir": str(output_dir), "files": {}}
    for split in ("train", "eval"):
        frame = pd.read_parquet(output_dir / f"{split}.parquet")
        problems: list[str] = []
        update_counts: dict[str, int] = {}
        for row_index, row in frame.iterrows():
            extra = dict(row["extra_info"] or {})
            update_kind = str(extra.get("update_kind") or "")
            update_counts[update_kind] = update_counts.get(update_kind, 0) + 1
            text = _prompt_text(row["prompt"])
            if update_kind.startswith("no_skill"):
                if _has_values(extra.get("retrieval_skills_top_n")):
                    problems.append(f"noskill-nonempty-retrieval:{row_index}")
                if "## Skills (mandatory)" in text or AVAILABLE_SKILLS_RE.search(text) or SKILL_LOCATION_RE.search(text):
                    problems.append(f"noskill-residual-skill:{row_index}")
            elif update_kind == "oracle_shadow":
                task_id = str(extra.get("task_id") or "")
                if extra.get("retrieval_skills_top_n") != [task_id]:
                    problems.append(f"oracle-bad-retrieval:{row_index}")
                if task_id and f"/root/.claude/skills/{task_id}/SKILL.md" not in text:
                    problems.append(f"oracle-missing-skill-location:{row_index}")
            elif update_kind == "oracle_prompt_bc":
                if _has_values(extra.get("retrieval_skills_top_n")):
                    problems.append(f"oracle-promptbc-nonempty-retrieval:{row_index}")
                if "## Skills (mandatory)" in text or AVAILABLE_SKILLS_RE.search(text) or SKILL_LOCATION_RE.search(text):
                    problems.append(f"oracle-promptbc-residual-path-skill:{row_index}")
                if not PRELOADED_ORACLE_RE.search(text):
                    problems.append(f"oracle-promptbc-missing-preloaded-block:{row_index}")
            else:
                problems.append(f"unknown-update-kind:{row_index}:{update_kind}")
        summary["files"][split] = {
            "rows": int(len(frame)),
            "bench_counts": _bench_counts(frame),
            "update_kind_counts": dict(sorted(update_counts.items())),
            "problems": problems,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--flat-root", type=Path, default=DEFAULT_FLAT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--oracle-mode", choices=("skill_path", "direct_text"), default="skill_path")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.oracle_mode == "direct_text" and args.output_dir == DEFAULT_OUTPUT:
        args.output_dir = DEFAULT_DIRECT_OUTPUT

    if args.validate_only:
        summary = validate_output(args.output_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if any(report["problems"] for report in summary["files"].values()):
            raise SystemExit(2)
        return

    args.flat_root.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_in = pd.read_parquet(args.input_dir / "train.parquet")
    eval_in = pd.read_parquet(args.input_dir / "eval.parquet")
    train_out, train_report = build_train(train_in, args.snapshot, args.flat_root, oracle_mode=args.oracle_mode)
    eval_out, eval_report = build_eval(eval_in)
    train_out.to_parquet(args.output_dir / "train.parquet", index=False)
    eval_out.to_parquet(args.output_dir / "eval.parquet", index=False)

    summary = {
        "input_dir": str(args.input_dir),
        "snapshot": str(args.snapshot),
        "flat_root": str(args.flat_root),
        "output_dir": str(args.output_dir),
        "oracle_mode": args.oracle_mode,
        "intent": (
            "mixed train: no-skill GRPO plus oracle prompt/path weighted BC/AWR; "
            "direct_text mode preloads oracle skill in the prompt and removes it before BC; no-skill eval"
        ),
        "files": {
            "train": train_report,
            "eval": eval_report,
        },
        "validation": validate_output(args.output_dir),
    }

    (args.output_dir / "build_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "# 4bench M1 hybrid parquet\n\n"
        "Train split has one no-skill GRPO row and one oracle-shadow row per task. "
        "Eval split is no-skill only.\n\n"
        "```json\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n```\n",
        encoding="utf-8",
    )

    for split, report in summary["files"].items():
        print(f"[{split}] rows_out={report['rows_out']} problems={len(report['problems'])}")
    validation_problems = [
        problem
        for report in summary["validation"]["files"].values()
        for problem in report["problems"]
    ]
    build_problems = train_report["problems"] + eval_report["problems"]
    if build_problems or validation_problems:
        raise SystemExit("hybrid parquet built with validation problems")


if __name__ == "__main__":
    main()
