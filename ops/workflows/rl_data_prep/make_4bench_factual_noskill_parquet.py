#!/usr/bin/env python3
"""Build a no-skill variant of the canonical 4bench factual RL parquet.

This preserves the task set, prompt profile, tool schema, reward labels, and
task kwargs from ``parquet_4bench_factual_20260602``. The only intended data
delta is removal of the OpenClaw ``## Skills (mandatory)`` section plus clearing
``extra_info.retrieval_skills_top_n`` so rollout setup has no skill files to
inject or advertise.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "datasets/rl/parquet_4bench_factual_20260602"
DEFAULT_OUTPUT = ROOT / "datasets/rl/parquet_4bench_factual_noskills_20260617"

SKILLS_SECTION_RE = re.compile(
    r"\n## Skills \(mandatory\)\n.*?(?=\n## Memory Recall\n)",
    re.S,
)
AVAILABLE_SKILLS_RE = re.compile(r"<available_skills>.*?</available_skills>", re.S)
SKILL_LOCATION_RE = re.compile(r"/root/\.claude/skills/[^/\s<>]+/SKILL\.md")


def _bench_of(extra_info: dict[str, Any]) -> str:
    return str((extra_info or {}).get("bench") or "unknown")


def _strip_skill_section(text: str) -> tuple[str, bool]:
    stripped, count = SKILLS_SECTION_RE.subn("\n", text, count=1)
    return stripped, bool(count)


def _transform_prompt(prompt: Any) -> tuple[Any, bool, int, int]:
    """Return prompt with the skills section removed.

    The second return value records whether the expected section was removed.
    The final two counts are post-transform safety checks for XML skill blocks
    and exact SKILL.md locations.
    """
    removed = False
    if isinstance(prompt, (list, tuple, np.ndarray)):
        out_messages = []
        for message in list(prompt):
            if isinstance(message, dict):
                new_message = dict(message)
                content = new_message.get("content")
                if isinstance(content, str):
                    content, did_remove = _strip_skill_section(content)
                    removed = removed or did_remove
                    new_message["content"] = content
                out_messages.append(new_message)
            else:
                out_messages.append(message)
        joined = "\n".join(str(m.get("content", "")) for m in out_messages if isinstance(m, dict))
        return np.array(out_messages, dtype=object), removed, len(AVAILABLE_SKILLS_RE.findall(joined)), len(SKILL_LOCATION_RE.findall(joined))

    text = "" if prompt is None else str(prompt)
    text, removed = _strip_skill_section(text)
    return text, removed, len(AVAILABLE_SKILLS_RE.findall(text)), len(SKILL_LOCATION_RE.findall(text))


def _counts_by_bench(frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for extra in frame["extra_info"]:
        bench = _bench_of(dict(extra or {}))
        counts[bench] = counts.get(bench, 0) + 1
    return dict(sorted(counts.items()))


def _has_values(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, np.ndarray)):
        return len(value) > 0
    return bool(value)


def transform(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    prompts = []
    extras = []
    removed_sections = 0
    residual_available_blocks = 0
    residual_skill_locations = 0
    nonempty_retrieval_before = 0
    problems: list[str] = []

    for row_index, row in frame.iterrows():
        extra = dict(row["extra_info"] or {})
        bench = _bench_of(extra)
        task_id = str(extra.get("task_id") or "unknown")
        if _has_values(extra.get("retrieval_skills_top_n")):
            nonempty_retrieval_before += 1

        prompt, removed, available_count, location_count = _transform_prompt(row["prompt"])
        if removed:
            removed_sections += 1
        else:
            problems.append(f"missing-skills-section:{bench}/{task_id}@row{row_index}")
        if available_count:
            residual_available_blocks += available_count
            problems.append(f"residual-available-skills:{bench}/{task_id}@row{row_index}:{available_count}")
        if location_count:
            residual_skill_locations += location_count
            problems.append(f"residual-skill-location:{bench}/{task_id}@row{row_index}:{location_count}")

        extra["retrieval_skills_top_n"] = []
        prompts.append(prompt)
        extras.append(extra)

    out = frame.copy()
    out["prompt"] = prompts
    out["extra_info"] = extras
    report = {
        "rows": int(len(out)),
        "bench_counts": _counts_by_bench(out),
        "removed_skill_sections": int(removed_sections),
        "nonempty_retrieval_before": int(nonempty_retrieval_before),
        "nonempty_retrieval_after": int(
            sum(
                1
                for extra in out["extra_info"]
                if _has_values(dict(extra or {}).get("retrieval_skills_top_n"))
            )
        ),
        "residual_available_skill_blocks": int(residual_available_blocks),
        "residual_skill_locations": int(residual_skill_locations),
        "problems": problems,
    }
    return out, report


def validate_output(output_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"output_dir": str(output_dir), "files": {}}
    for split in ("train", "eval"):
        frame = pd.read_parquet(output_dir / f"{split}.parquet")
        residual_available_blocks = 0
        residual_skill_locations = 0
        residual_skill_sections = 0
        nonempty_retrieval = 0
        problems: list[str] = []
        for row_index, row in frame.iterrows():
            extra = dict(row["extra_info"] or {})
            bench = _bench_of(extra)
            task_id = str(extra.get("task_id") or "unknown")
            if _has_values(extra.get("retrieval_skills_top_n")):
                nonempty_retrieval += 1
                problems.append(f"nonempty-retrieval:{bench}/{task_id}@row{row_index}")

            prompt = row["prompt"]
            if isinstance(prompt, (list, tuple, np.ndarray)):
                text = "\n".join(
                    str(message.get("content", ""))
                    for message in list(prompt)
                    if isinstance(message, dict)
                )
            else:
                text = "" if prompt is None else str(prompt)
            if "## Skills (mandatory)" in text:
                residual_skill_sections += 1
                problems.append(f"residual-skills-section:{bench}/{task_id}@row{row_index}")
            available_count = len(AVAILABLE_SKILLS_RE.findall(text))
            location_count = len(SKILL_LOCATION_RE.findall(text))
            residual_available_blocks += available_count
            residual_skill_locations += location_count
            if available_count:
                problems.append(f"residual-available-skills:{bench}/{task_id}@row{row_index}:{available_count}")
            if location_count:
                problems.append(f"residual-skill-location:{bench}/{task_id}@row{row_index}:{location_count}")
        report = {
            "rows": int(len(frame)),
            "bench_counts": _counts_by_bench(frame),
            "nonempty_retrieval_after": int(nonempty_retrieval),
            "residual_skill_sections": int(residual_skill_sections),
            "residual_available_skill_blocks": int(residual_available_blocks),
            "residual_skill_locations": int(residual_skill_locations),
            "problems": problems,
        }
        summary["files"][split] = report
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        summary = validate_output(args.output_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        for split_report in summary["files"].values():
            if split_report["problems"] or split_report["nonempty_retrieval_after"]:
                raise SystemExit(2)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "intent": "same 4bench factual RL data, with no advertised or injectable skills",
        "files": {},
    }
    for split in ("train", "eval"):
        frame = pd.read_parquet(args.input_dir / f"{split}.parquet")
        out, report = transform(frame)
        out.to_parquet(args.output_dir / f"{split}.parquet", index=False)
        summary["files"][split] = report
        print(f"[{split}] rows={len(out)} removed_sections={report['removed_skill_sections']} problems={len(report['problems'])}")

    readme = (
        "# 4bench factual no-skill RL parquet\n\n"
        "Derived from `datasets/rl/parquet_4bench_factual_20260602`.\n"
        "Only the OpenClaw `## Skills (mandatory)` section and "
        "`extra_info.retrieval_skills_top_n` are removed. Task rows, reward "
        "labels, prompt profile, and tool schema are otherwise preserved.\n\n"
        "```json\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n```\n"
    )
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    (args.output_dir / "build_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if any(report["problems"] for report in summary["files"].values()):
        raise SystemExit("no-skill parquet built with residual skill prompt problems")


if __name__ == "__main__":
    main()
