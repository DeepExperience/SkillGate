#!/usr/bin/env python3
"""Build the gold-only, category-attributed SlateRL regret v2 parquet.

This is a separate reproducible data entrypoint.  It reuses the canonical
mixed-slate prompt construction but pins ``p_gold=1.0`` and writes to a new
output root.  Every training ``slate_grpo`` row therefore contains exactly:

    oracle x1 + misleading x5 + relevant x5 + irrelevant x5

It also stamps exact ``slate_oracle_names`` and ``slate_misleading_names`` in
``extra_info``.  The v2 reward postprocessor uses those immutable manifest
labels to attribute strict assistant reads without guessing from names.

Resume behavior: output is immutable once built.  Use ``--validate-only`` to
recheck source fingerprints, output hashes, counts, and all-gold/category
invariants.  A non-validation build refuses to overwrite any existing output;
choose a new output directory if inputs change.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import make_4bench_slate_parquet as base  # noqa: E402


V8PROD_ROOT = ROOT / "skill_libraries/snapshots/rl/slate_skills_20260708_hard_negative_v8_production"
OLD_SLATE_ROOT = ROOT / "skill_libraries/snapshots/rl/slate_skills_20260704"
DEFAULT_INPUT = ROOT / "datasets/rl/parquet_4bench_oracle_promptbc_pair_noskill_grpo_20260623"
DEFAULT_OUTPUT = ROOT / "datasets/rl/parquet_4bench_slate_regret_v8prod_gold_stratified_v2_20260710"
SCHEMA = "gold_only_stratified_misleading_precedence_v1"


def _as_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    try:
        return [str(item) for item in value if str(item)]
    except TypeError:
        return [str(value)] if str(value) else []


def _task_key(extra: dict[str, Any]) -> str:
    return f"{extra.get('bench')}::{extra.get('task_id')}"


def stamp_category_names(
    df: pd.DataFrame,
    manifest: dict[str, dict[str, Any]],
    *,
    allowed_kinds: set[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Stamp exact category names and fail on any all-gold invariant break."""

    problems: list[str] = []
    extras: list[dict[str, Any]] = []
    for row_index, original in enumerate(df["extra_info"]):
        extra = dict(original)
        kind = str(extra.get("update_kind") or extra.get("hybrid_update_kind") or "").strip().lower()
        if kind not in allowed_kinds:
            extras.append(extra)
            continue

        key = _task_key(extra)
        manifest_row = manifest.get(key)
        if manifest_row is None:
            problems.append(f"row{row_index}:missing-manifest:{key}")
            extras.append(extra)
            continue

        oracle_names = [str(entry["name"]) for entry in manifest_row.get("oracle") or []]
        misleading_names = [str(entry["name"]) for entry in manifest_row.get("misleading") or []]
        advertised = _as_names(extra.get("retrieval_skills_top_n"))
        advertised_set = set(advertised)
        if len(oracle_names) != 1:
            problems.append(f"row{row_index}:oracle-count:{len(oracle_names)}")
        if len(misleading_names) != 5:
            problems.append(f"row{row_index}:misleading-count:{len(misleading_names)}")
        if len(advertised) != 16 or len(advertised_set) != 16:
            problems.append(f"row{row_index}:advertised-count:{len(advertised)}/{len(advertised_set)}")
        if not set(oracle_names).issubset(advertised_set):
            problems.append(f"row{row_index}:oracle-not-advertised:{oracle_names}")
        if not set(misleading_names).issubset(advertised_set):
            problems.append(f"row{row_index}:misleading-not-advertised")
        try:
            has_gold = float(extra.get("slate_contains_gold") or 0.0) == 1.0
        except (TypeError, ValueError):
            has_gold = False
        if not has_gold:
            problems.append(f"row{row_index}:gold-absent:{key}")
        if oracle_names and extra.get("slate_gold_name") != oracle_names[0]:
            problems.append(f"row{row_index}:gold-name-mismatch:{key}")

        extra["slate_contains_gold"] = 1.0
        extra["slate_oracle_names"] = oracle_names
        extra["slate_misleading_names"] = misleading_names
        extra["slate_stratification_schema"] = SCHEMA
        extras.append(extra)

    out = df.copy()
    out["extra_info"] = extras
    return out, problems


def audit_output(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_manifest: dict[str, dict[str, Any]],
    eval_manifest: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    problems: list[str] = []
    train_kinds: Counter[str] = Counter()
    seen_train_tasks: dict[str, set[str]] = {"no_skill_grpo": set(), "slate_grpo": set()}

    for row_index, extra_raw in enumerate(train["extra_info"]):
        extra = dict(extra_raw)
        kind = str(extra.get("update_kind") or "").strip().lower()
        train_kinds[kind] += 1
        if kind in seen_train_tasks:
            seen_train_tasks[kind].add(_task_key(extra))
        if kind != "slate_grpo":
            continue
        oracle_names = _as_names(extra.get("slate_oracle_names"))
        misleading_names = _as_names(extra.get("slate_misleading_names"))
        advertised = _as_names(extra.get("retrieval_skills_top_n"))
        if (
            float(extra.get("slate_contains_gold") or 0.0) != 1.0
            or len(oracle_names) != 1
            or len(misleading_names) != 5
            or len(advertised) != 16
            or len(set(advertised)) != 16
            or not set(oracle_names + misleading_names).issubset(set(advertised))
            or extra.get("slate_stratification_schema") != SCHEMA
        ):
            problems.append(f"train-row{row_index}:bad-gold-category-metadata")
        manifest_row = train_manifest.get(_task_key(extra))
        if manifest_row is None:
            problems.append(f"train-row{row_index}:manifest-missing")

    if train_kinds["no_skill_grpo"] <= 0 or train_kinds["slate_grpo"] <= 0:
        problems.append(f"missing-pair-kind:{dict(train_kinds)}")
    if train_kinds["no_skill_grpo"] != train_kinds["slate_grpo"]:
        problems.append(f"unbalanced-pair-count:{dict(train_kinds)}")
    if seen_train_tasks["no_skill_grpo"] != seen_train_tasks["slate_grpo"]:
        problems.append("no-skill/slate task-key sets differ")

    eval_kinds: Counter[str] = Counter()
    for row_index, extra_raw in enumerate(eval_df["extra_info"]):
        extra = dict(extra_raw)
        kind = str(extra.get("update_kind") or "").strip().lower()
        eval_kinds[kind] += 1
        oracle_names = _as_names(extra.get("slate_oracle_names"))
        misleading_names = _as_names(extra.get("slate_misleading_names"))
        advertised = _as_names(extra.get("retrieval_skills_top_n"))
        if (
            float(extra.get("slate_contains_gold") or 0.0) != 1.0
            or len(oracle_names) != 1
            or len(misleading_names) != 5
            or len(advertised) != 16
            or extra.get("slate_stratification_schema") != SCHEMA
            or _task_key(extra) not in eval_manifest
        ):
            problems.append(f"eval-row{row_index}:bad-gold-category-metadata")

    if problems:
        raise SystemExit(f"gold-only SlateRL v2 audit failed ({len(problems)}): {problems[:10]}")
    return {
        "train_kinds": dict(train_kinds),
        "train_unique_tasks": len(seen_train_tasks["slate_grpo"]),
        "train_gold_present": train_kinds["slate_grpo"],
        "train_gold_absent": 0,
        "eval_kinds": dict(eval_kinds),
        "eval_rows": len(eval_df),
        "schema": SCHEMA,
    }


def source_fingerprints(
    input_dir: Path,
    train_manifest_path: Path,
    eval_manifest_path: Path,
    train_manifest: dict[str, dict[str, Any]],
    eval_manifest: dict[str, dict[str, Any]],
) -> dict[str, str]:
    return {
        "input_train_sha256": base.file_sha256(input_dir / "train.parquet"),
        "input_eval_sha256": base.file_sha256(input_dir / "eval.parquet"),
        "train_manifest_sha256": base.file_sha256(train_manifest_path),
        "eval_manifest_sha256": base.file_sha256(eval_manifest_path),
        "skill_content_sha256": base.skill_content_sha256([train_manifest, eval_manifest]),
    }


def output_fingerprints(output_dir: Path) -> dict[str, str]:
    return {
        "train_parquet_sha256": base.file_sha256(output_dir / "train.parquet"),
        "eval_parquet_sha256": base.file_sha256(output_dir / "eval.parquet"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--manifest", type=Path, default=V8PROD_ROOT / "manifest/slate_manifest_train.jsonl"
    )
    parser.add_argument(
        "--eval-manifest", type=Path, default=V8PROD_ROOT / "manifest/slate_manifest_eval70.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skill-roots",
        default=os.pathsep.join((str(V8PROD_ROOT / "skills"), str(OLD_SLATE_ROOT / "skills"))),
        help="colon-separated runtime skill roots in first-match order",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    train_manifest = base.load_manifest(args.manifest)
    eval_manifest = base.load_manifest(args.eval_manifest)
    skill_roots = [Path(item).resolve() for item in args.skill_roots.split(os.pathsep) if item]
    base.validate_runtime_resolution([train_manifest, eval_manifest], skill_roots)
    fingerprints = source_fingerprints(
        args.input_dir, args.manifest, args.eval_manifest, train_manifest, eval_manifest
    )
    report_path = args.output_dir / "build_report.json"

    if args.validate_only:
        if not report_path.is_file():
            raise SystemExit(f"validate-only: missing {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema") != SCHEMA or report.get("p_gold") != 1.0:
            raise SystemExit(f"validate-only: wrong schema/p_gold in {report_path}")
        if report.get("fingerprints") != fingerprints:
            raise SystemExit("validate-only: source fingerprints changed; rebuild into a new output root")
        if report.get("output_fingerprints") != output_fingerprints(args.output_dir):
            raise SystemExit("validate-only: output parquet fingerprint mismatch")
        train = pd.read_parquet(args.output_dir / "train.parquet")
        eval_df = pd.read_parquet(args.output_dir / "eval.parquet")
        audit = audit_output(train, eval_df, train_manifest, eval_manifest)
        if report.get("audit") != audit:
            raise SystemExit(f"validate-only: audit mismatch: report={report.get('audit')} live={audit}")
        print(f"validate OK: {json.dumps(audit, sort_keys=True)}")
        return

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty {args.output_dir}; validate it or choose a new output root"
        )

    input_train = pd.read_parquet(args.input_dir / "train.parquet")
    input_eval = pd.read_parquet(args.input_dir / "eval.parquet")
    train, train_problems, train_stats = base.transform(
        input_train, train_manifest, p_gold=1.0, validate_only=False
    )
    eval_df, eval_problems, eval_stats = base.transform_eval(input_eval, eval_manifest)
    if train_problems or eval_problems:
        raise SystemExit(
            f"base slate transform failed: train={train_problems[:10]} eval={eval_problems[:10]}"
        )

    train, category_train_problems = stamp_category_names(
        train, train_manifest, allowed_kinds={"slate_grpo"}
    )
    eval_df, category_eval_problems = stamp_category_names(
        eval_df, eval_manifest, allowed_kinds={base.MIXED_EVAL_KIND}
    )
    if category_train_problems or category_eval_problems:
        raise SystemExit(
            "category stamping failed: "
            f"train={category_train_problems[:10]} eval={category_eval_problems[:10]}"
        )
    audit = audit_output(train, eval_df, train_manifest, eval_manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.output_dir / "train.parquet")
    eval_df.to_parquet(args.output_dir / "eval.parquet")
    report = {
        "format_version": 1,
        "schema": SCHEMA,
        "p_gold": 1.0,
        "input_dir": str(args.input_dir.resolve()),
        "train_manifest": str(args.manifest.resolve()),
        "eval_manifest": str(args.eval_manifest.resolve()),
        "skill_roots": [str(path) for path in skill_roots],
        "fingerprints": fingerprints,
        "train_transform_stats": train_stats,
        "eval_transform_stats": eval_stats,
        "audit": audit,
        "output_fingerprints": output_fingerprints(args.output_dir),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir}: {json.dumps(audit, sort_keys=True)}")


if __name__ == "__main__":
    main()
