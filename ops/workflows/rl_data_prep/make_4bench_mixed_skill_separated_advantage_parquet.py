#!/usr/bin/env python3
"""Stamp the frozen always-gold mixed slate for separated-advantage GRPO.

The canonical mixed-bonus parquet already contains the exact v8-production
prompt text, skill ordering, and category metadata.  This deterministic
converter changes only experiment identity fields so separated-advantage runs
cannot be confused with the stopped scalar-bonus diagnostic.

Input and output prompt payloads must remain byte-equivalent after normalized
JSON serialization.  Outputs are immutable: use ``--validate-only`` for an
existing root or select a new root if any source fingerprint changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "datasets/rl/parquet_4bench_mixed_skill_bonus_compare_v8prod_allgold_20260710"
DEFAULT_OUTPUT = ROOT / "datasets/rl/parquet_4bench_mixed_skill_separated_continuous_advantage_v8prod_allgold_20260710"
SOURCE_TRAIN_KIND = "mixed_bonus_compare_grpo"
SOURCE_EVAL_KIND = "mixed_bonus_compare_eval"
TRAIN_KIND = "mixed_separated_continuous_advantage_grpo"
EVAL_KIND = "mixed_separated_continuous_advantage_eval"
SCHEMA = "continuous_task_grpo_plus_adaptive_outcome_stratified_behavior_v3"
PROMPT_SKILL_NAME_RE = re.compile(r"<name>([^<]+)</name>")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def plain_extra(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, dict):
            return dict(item)
    raise TypeError(f"extra_info is not a dict: {type(value)!r}")


def plain_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


def prompt_text(prompt: Any) -> str:
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if not isinstance(prompt, (list, tuple)):
        return str(prompt)
    return "\n".join(
        str(item.get("content", "")) if isinstance(item, dict) else str(item)
        for item in prompt
    )


def prompt_fingerprint(prompt: Any) -> str:
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    payload = json.dumps(prompt, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def task_key(extra: dict[str, Any]) -> str:
    return f"{extra.get('bench')}::{extra.get('task_id')}"


def source_fingerprints(input_dir: Path) -> dict[str, str]:
    return {
        "input_train_sha256": file_sha256(input_dir / "train.parquet"),
        "input_eval_sha256": file_sha256(input_dir / "eval.parquet"),
        "input_build_report_sha256": file_sha256(input_dir / "build_report.json"),
    }


def output_fingerprints(output_dir: Path) -> dict[str, str]:
    return {
        "train_parquet_sha256": file_sha256(output_dir / "train.parquet"),
        "eval_parquet_sha256": file_sha256(output_dir / "eval.parquet"),
    }


def stamp_frame(frame: pd.DataFrame, *, evaluation: bool) -> pd.DataFrame:
    source_kind = SOURCE_EVAL_KIND if evaluation else SOURCE_TRAIN_KIND
    target_kind = EVAL_KIND if evaluation else TRAIN_KIND
    output = frame.copy()
    extras: list[dict[str, Any]] = []
    for row_index, raw in enumerate(output["extra_info"]):
        extra = plain_extra(raw)
        kind = str(extra.get("update_kind") or extra.get("hybrid_update_kind") or "")
        if kind != source_kind:
            raise SystemExit(
                f"row {row_index}: expected source kind {source_kind!r}, got {kind!r}"
            )
        extra["update_kind"] = target_kind
        extra["hybrid_update_kind"] = target_kind
        extra.pop("mixed_skill_bonus_compare", None)
        extra.pop("mixed_skill_bonus_category_version", None)
        extra["mixed_skill_separated_advantage"] = 1.0
        extra["mixed_skill_separated_schema"] = SCHEMA
        extra["mixed_skill_task_outcome"] = "pass_at_1"
        extra["mixed_skill_task_advantage"] = "continuous_raw_grpo"
        extra["mixed_skill_raw_score_preserved"] = 1.0
        extras.append(extra)
    output["extra_info"] = extras
    return output


def audit_frame(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    *,
    split: str,
    expected_rows: int,
    expected_kind: str,
) -> dict[str, Any]:
    if len(frame) != expected_rows or len(source) != expected_rows:
        raise SystemExit(
            f"{split}: expected {expected_rows} rows, got output={len(frame)} source={len(source)}"
        )
    problems: list[str] = []
    keys: set[str] = set()
    benches: Counter[str] = Counter()
    prompt_equal = 0
    for row_index, ((_, row), (_, source_row)) in enumerate(
        zip(frame.iterrows(), source.iterrows(), strict=True)
    ):
        extra = plain_extra(row["extra_info"])
        key = task_key(extra)
        keys.add(key)
        benches[str(extra.get("bench"))] += 1
        retrieval = plain_list(extra.get("retrieval_skills_top_n"))
        misleading = plain_list(extra.get("slate_misleading_names"))
        relevant = plain_list(extra.get("slate_relevant_names"))
        irrelevant = plain_list(extra.get("slate_irrelevant_names"))
        gold = str(extra.get("slate_gold_name") or "")
        prompt_names = PROMPT_SKILL_NAME_RE.findall(prompt_text(row["prompt"]))
        if prompt_fingerprint(row["prompt"]) == prompt_fingerprint(source_row["prompt"]):
            prompt_equal += 1
        else:
            problems.append(f"row{row_index}:{key}:prompt-changed")
        if extra.get("update_kind") != expected_kind or extra.get("hybrid_update_kind") != expected_kind:
            problems.append(f"row{row_index}:{key}:bad-kind")
        if float(extra.get("slate_contains_gold") or 0.0) != 1.0:
            problems.append(f"row{row_index}:{key}:gold-absent")
        if (
            len(retrieval) != 16
            or len(set(retrieval)) != 16
            or len(misleading) != 5
            or len(relevant) != 5
            or len(irrelevant) != 5
            or not gold
            or set([gold, *misleading, *relevant, *irrelevant]) != set(retrieval)
            or prompt_names != retrieval
        ):
            problems.append(f"row{row_index}:{key}:bad-slate-categories")
        if float(extra.get("mixed_skill_separated_advantage") or 0.0) != 1.0:
            problems.append(f"row{row_index}:{key}:missing-mode-marker")
        if extra.get("mixed_skill_separated_schema") != SCHEMA:
            problems.append(f"row{row_index}:{key}:bad-schema")
        if extra.get("mixed_skill_task_outcome") != "pass_at_1":
            problems.append(f"row{row_index}:{key}:bad-task-outcome")
        if extra.get("mixed_skill_task_advantage") != "continuous_raw_grpo":
            problems.append(f"row{row_index}:{key}:bad-task-advantage")
        if float(extra.get("mixed_skill_raw_score_preserved") or 0.0) != 1.0:
            problems.append(f"row{row_index}:{key}:raw-score-not-preserved")
        if "mixed_skill_bonus_compare" in extra:
            problems.append(f"row{row_index}:{key}:stale-bonus-marker")
    if len(keys) != expected_rows:
        problems.append(f"unique-tasks={len(keys)} expected={expected_rows}")
    if problems:
        raise SystemExit(f"{split} audit failed ({len(problems)}): {problems[:20]}")
    return {
        "rows": len(frame),
        "unique_tasks": len(keys),
        "gold_present": len(frame),
        "gold_absent": 0,
        "slate_size_16": len(frame),
        "prompt_equal_to_source": prompt_equal,
        "bench_counts": dict(benches),
        "update_kind": expected_kind,
        "schema": SCHEMA,
    }


def validate_output(args: argparse.Namespace, fingerprints: dict[str, str]) -> dict[str, Any]:
    report_path = args.output_dir / "build_report.json"
    if not report_path.is_file():
        raise SystemExit(f"validate-only: missing {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != SCHEMA or report.get("fingerprints") != fingerprints:
        raise SystemExit("validate-only: schema or frozen input fingerprints changed")
    if report.get("output_fingerprints") != output_fingerprints(args.output_dir):
        raise SystemExit("validate-only: output parquet fingerprint mismatch")
    source_train = pd.read_parquet(args.input_dir / "train.parquet")
    source_eval = pd.read_parquet(args.input_dir / "eval.parquet")
    train = pd.read_parquet(args.output_dir / "train.parquet")
    evaluation = pd.read_parquet(args.output_dir / "eval.parquet")
    result = {
        "train": audit_frame(
            train,
            source_train,
            split="train",
            expected_rows=args.expected_train_tasks,
            expected_kind=TRAIN_KIND,
        ),
        "eval": audit_frame(
            evaluation,
            source_eval,
            split="eval",
            expected_rows=args.expected_eval_tasks,
            expected_kind=EVAL_KIND,
        ),
        "output_fingerprints": output_fingerprints(args.output_dir),
    }
    if report.get("audit") != {"train": result["train"], "eval": result["eval"]}:
        raise SystemExit("validate-only: stored audit does not match live output")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-train-tasks", type=int, default=491)
    parser.add_argument("--expected-eval-tasks", type=int, default=56)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    fingerprints = source_fingerprints(args.input_dir)
    if args.validate_only:
        print(json.dumps(validate_output(args, fingerprints), indent=2, ensure_ascii=False))
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty output directory: {args.output_dir}; "
            "use --validate-only or select a new immutable root"
        )

    source_train = pd.read_parquet(args.input_dir / "train.parquet")
    source_eval = pd.read_parquet(args.input_dir / "eval.parquet")
    train = stamp_frame(source_train, evaluation=False)
    evaluation = stamp_frame(source_eval, evaluation=True)
    audit = {
        "train": audit_frame(
            train,
            source_train,
            split="train",
            expected_rows=args.expected_train_tasks,
            expected_kind=TRAIN_KIND,
        ),
        "eval": audit_frame(
            evaluation,
            source_eval,
            split="eval",
            expected_rows=args.expected_eval_tasks,
            expected_kind=EVAL_KIND,
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.output_dir / "train.parquet")
    evaluation.to_parquet(args.output_dir / "eval.parquet")
    report = {
        "format_version": 1,
        "objective": "pure mixed always-gold continuous-task GRPO plus adaptive outcome-stratified behavior advantage",
        "schema": SCHEMA,
        "input_dir": str(args.input_dir.resolve()),
        "fingerprints": fingerprints,
        "output_fingerprints": output_fingerprints(args.output_dir),
        "audit": audit,
        "notes": [
            "prompt payloads and skill ordering are identical to the validated v8prod all-gold source",
            "train contains no no-skill arm",
            "raw verifier score is preserved and directly drives task GRPO/dynamic sampling",
            "raw_score >= 1.0 only defines success/failure behavior strata and dominance",
            "behavior advantage is separated from task advantage",
        ],
    }
    (args.output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
