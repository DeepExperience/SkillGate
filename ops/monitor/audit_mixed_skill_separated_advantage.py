#!/usr/bin/env python3
"""Audit live mixed-skill separated-advantage rollout artifacts.

This checks persisted training JSONL rather than launcher intent: exact 8-way
raw-nonuniform groups, all-gold 16-skill metadata, raw-score preservation, the
adaptive outcome-stratified behavior formula, and strict success-over-failure
dominance whenever both outcomes occur.
"""

from __future__ import annotations

import os

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import pandas as pd


ROOT = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))
DEFAULT_DATA_DIR = (
    ROOT
    / "datasets/rl/parquet_4bench_mixed_skill_separated_continuous_advantage_v8prod_allgold_20260710"
)
EXPECTED_KIND = "mixed_separated_continuous_advantage_grpo"
SKILL_NAME_RE = re.compile(r"<name>([^<]+)</name>")
TOL = 2e-6


def plain_extra(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, dict):
            return dict(item)
    raise TypeError(f"extra_info is not a mapping: {type(value)!r}")


def plain_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [str(item) for item in value]


def load_task_metadata(data_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = pd.read_parquet(data_dir / "train.parquet", columns=["extra_info"])
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in rows["extra_info"]:
        extra = plain_extra(raw)
        key = (str(extra.get("bench")), str(extra.get("task_id")))
        if key in result:
            raise AssertionError(f"duplicate parquet task key: {key}")
        result[key] = extra
    if len(result) != 491:
        raise AssertionError(f"expected 491 parquet tasks, got {len(result)}")
    return result


def close(actual: float, expected: float, *, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=TOL):
        raise AssertionError(f"{label}: got {actual}, expected {expected}")


def audit_step(
    path: Path,
    task_metadata: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    # JSON strings may legally contain Unicode line-separator characters such as
    # U+2028.  str.splitlines() treats those as record boundaries even though
    # JSONL records are delimited only by literal LF bytes.
    rows = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
    if len(rows) != 128:
        raise AssertionError(f"{path.name}: expected 128 rows, got {len(rows)}")

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    behavior_counts: Counter[str] = Counter()
    behavior_pass: Counter[str] = Counter()
    bench_counts: Counter[str] = Counter()
    total_pass = 0

    for row_index, row in enumerate(rows):
        if row.get("update_kind") != EXPECTED_KIND or row.get("hybrid_update_kind") != EXPECTED_KIND:
            raise AssertionError(f"row {row_index}: wrong update kind")
        reward = row.get("reward")
        if not isinstance(reward, dict):
            raise AssertionError(f"row {row_index}: reward is not a dictionary")
        raw = float(reward["raw_score"])
        score = float(reward["score"])
        if not math.isfinite(raw):
            raise AssertionError(f"row {row_index}: non-finite raw_score={raw}")
        outcome = 1.0 if raw >= 1.0 else 0.0
        close(score, raw, label=f"row {row_index} score/raw")
        close(
            float(reward["mixed_sep_verifier_raw_score"]),
            raw,
            label=f"row {row_index} persisted raw score",
        )
        close(float(reward["mixed_sep_task_outcome"]), outcome, label=f"row {row_index} outcome")

        bench = str(reward.get("bench"))
        task_id = str(reward.get("task_id"))
        key = (bench, task_id)
        if key not in task_metadata:
            raise AssertionError(f"row {row_index}: task absent from frozen parquet: {key}")
        extra = task_metadata[key]
        retrieval = plain_list(extra.get("retrieval_skills_top_n"))
        prompt_names = SKILL_NAME_RE.findall(str(row.get("prompt") or ""))
        if len(retrieval) != 16 or len(set(retrieval)) != 16 or prompt_names != retrieval:
            raise AssertionError(f"row {row_index}: prompt is not the frozen ordered 16-skill slate")
        if float(extra.get("slate_contains_gold") or 0.0) != 1.0:
            raise AssertionError(f"row {row_index}: frozen task does not contain gold")
        if str(extra.get("slate_gold_name") or "") not in retrieval:
            raise AssertionError(f"row {row_index}: gold is absent from retrieval slate")

        flags = {
            "oracle": float(reward["mixed_sep_is_oracle"]) > 0.5,
            "misleading": float(reward["mixed_sep_is_misleading"]) > 0.5,
            "other": float(reward["mixed_sep_is_other"]) > 0.5,
            "none": float(reward["mixed_sep_no_read"]) > 0.5,
        }
        active = [name for name, enabled in flags.items() if enabled]
        if len(active) != 1:
            raise AssertionError(f"row {row_index}: behavior flags are not one-hot: {flags}")
        behavior = active[0]
        expected_utility = {"oracle": 1.0, "misleading": -1.0, "other": -0.25, "none": -0.25}[behavior]
        close(float(reward["mixed_sep_behavior_utility"]), expected_utility, label=f"row {row_index} utility")
        close(
            float(reward["mixed_sep_any_read"]),
            0.0 if behavior == "none" else 1.0,
            label=f"row {row_index} any-read",
        )
        behavior_counts[behavior] += 1
        behavior_pass[behavior] += int(outcome)
        total_pass += int(outcome)
        bench_counts[bench] += 1
        groups[int(row["group_index"])].append(row)

    if len(groups) != 16 or any(len(group) != 8 for group in groups.values()):
        raise AssertionError(
            f"{path.name}: expected 16 groups x 8, got {len(groups)} groups and "
            f"sizes={sorted(Counter(len(group) for group in groups.values()).items())}"
        )

    minimum_gap = float("inf")
    minimum_task_gap_reserved_fraction = float("inf")
    dominance_applicable_groups = 0
    success_counts: Counter[int] = Counter()
    for group_index, group in groups.items():
        raw_scores = [float(row["reward"]["raw_score"]) for row in group]
        outcomes = [1.0 if value >= 1.0 else 0.0 for value in raw_scores]
        success_count = int(sum(outcomes))
        if max(raw_scores) - min(raw_scores) <= 1e-12:
            raise AssertionError(f"group {group_index}: verifier raw scores are uniform: {raw_scores}")
        success_counts[success_count] += 1
        task = [(value - mean(raw_scores)) / (stdev(raw_scores) + 1e-6) for value in raw_scores]
        utilities = [float(row["reward"]["mixed_sep_behavior_utility"]) for row in group]
        expected_behavior = [0.0] * 8
        expected_scales = [0.0] * 8
        for outcome in (0.0, 1.0):
            indices = [index for index, value in enumerate(outcomes) if value == outcome]
            if not indices:
                continue
            stratum_mean = mean(utilities[index] for index in indices)
            deviations = [utilities[index] - stratum_mean for index in indices]
            max_abs = max(abs(value) for value in deviations)
            scale = 0.0 if max_abs <= 1e-8 else min(0.30, 0.40 / max_abs)
            for index, deviation in zip(indices, deviations, strict=True):
                expected_behavior[index] = scale * deviation
                expected_scales[index] = scale
            if abs(sum(expected_behavior[index] for index in indices)) > 1e-8:
                raise AssertionError(f"group {group_index}: behavior stratum is not zero mean")

        success_indices = [index for index, value in enumerate(outcomes) if value == 1.0]
        failure_indices = [index for index, value in enumerate(outcomes) if value == 0.0]
        applicable = bool(success_indices and failure_indices)
        task_gap = 0.0
        task_pass_guard = 0.0
        behavior_cross_harm = 0.0
        dominance_scale = 1.0
        if applicable:
            dominance_applicable_groups += 1
            task_gap = min(task[index] for index in success_indices) - max(
                task[index] for index in failure_indices
            )
            if task_gap <= 0.0:
                raise AssertionError(f"group {group_index}: nonpositive continuous task gap {task_gap}")
            task_pass_guard = max(0.0, 1e-5 - task_gap)
            if task_pass_guard > 0.0:
                success_fraction = len(success_indices) / 8
                task = [
                    value + task_pass_guard * (outcome - success_fraction)
                    for value, outcome in zip(task, outcomes, strict=True)
                ]
                task_gap = min(task[index] for index in success_indices) - max(
                    task[index] for index in failure_indices
                )
            behavior_cross_harm = max(
                expected_behavior[index] for index in failure_indices
            ) - min(expected_behavior[index] for index in success_indices)
            allowed_harm = 0.5 * task_gap
            if behavior_cross_harm > allowed_harm:
                dominance_scale = allowed_harm / behavior_cross_harm
                expected_behavior = [value * dominance_scale for value in expected_behavior]
                expected_scales = [value * dominance_scale for value in expected_scales]
                behavior_cross_harm *= dominance_scale

        totals: list[float] = []
        for index, row in enumerate(group):
            reward = row["reward"]
            close(float(reward["mixed_sep_task_advantage"]), task[index], label=f"group {group_index} task adv")
            close(
                float(reward["mixed_sep_behavior_advantage"]),
                expected_behavior[index],
                label=f"group {group_index} behavior adv",
            )
            close(
                float(reward["mixed_sep_behavior_scale"]),
                expected_scales[index],
                label=f"group {group_index} behavior scale",
            )
            expected_total = task[index] + expected_behavior[index]
            close(float(reward["mixed_sep_total_advantage"]), expected_total, label=f"group {group_index} total adv")
            close(float(reward["mixed_sep_success_count"]), float(success_count), label=f"group {group_index} success count")
            close(
                float(reward["mixed_sep_failure_count"]),
                float(8 - success_count),
                label=f"group {group_index} failure count",
            )
            totals.append(expected_total)
        close(
            float(group[0]["reward"]["mixed_sep_behavior_cap"]),
            max(abs(value) for value in expected_behavior),
            label=f"group {group_index} behavior cap",
        )
        close(
            float(group[0]["reward"]["mixed_sep_behavior_dominance_scale"]),
            dominance_scale,
            label=f"group {group_index} dominance scale",
        )
        close(
            float(group[0]["reward"]["mixed_sep_behavior_cross_harm"]),
            behavior_cross_harm,
            label=f"group {group_index} behavior cross harm",
        )
        close(
            float(group[0]["reward"]["mixed_sep_task_pass_guard"]),
            task_pass_guard,
            label=f"group {group_index} task pass guard",
        )
        close(
            float(group[0]["reward"]["mixed_sep_task_success_dominance_gap"]),
            task_gap,
            label=f"group {group_index} task gap",
        )
        close(
            float(group[0]["reward"]["mixed_sep_success_dominance_applicable"]),
            1.0 if applicable else 0.0,
            label=f"group {group_index} dominance applicable",
        )
        if applicable:
            gap = min(totals[index] for index in success_indices) - max(
                totals[index] for index in failure_indices
            )
            if gap <= 0.0 or gap + TOL < 0.5 * task_gap:
                raise AssertionError(
                    f"group {group_index}: success dominance failed: total={gap} task={task_gap}"
                )
            minimum_gap = min(minimum_gap, gap)
            minimum_task_gap_reserved_fraction = min(
                minimum_task_gap_reserved_fraction, gap / task_gap
            )
        else:
            gap = 0.0
        for row in group:
            close(
                float(row["reward"]["mixed_sep_success_dominance_gap"]),
                gap,
                label=f"group {group_index} persisted gap",
            )

    behavior_pass_rates = {
        name: behavior_pass[name] / count if count else None
        for name, count in sorted(behavior_counts.items())
    }
    any_read = 128 - behavior_counts["none"]
    return {
        "step": int(path.stem),
        "rows": 128,
        "groups": 16,
        "raw_pass": total_pass,
        "raw_pass_rate": total_pass / 128,
        "verifier_raw_score_mean": sum(float(row["reward"]["raw_score"]) for row in rows) / 128,
        "fractional_raw_score_count": sum(
            1 for row in rows if float(row["reward"]["raw_score"]) not in (0.0, 1.0)
        ),
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "behavior_pass_rates": behavior_pass_rates,
        "read_rate": any_read / 128,
        "oracle_given_read": behavior_counts["oracle"] / any_read if any_read else None,
        "bench_counts": dict(sorted(bench_counts.items())),
        "success_count_distribution": dict(sorted(success_counts.items())),
        "dominance_applicable_groups": dominance_applicable_groups,
        "minimum_success_dominance_gap": (
            minimum_gap if dominance_applicable_groups else None
        ),
        "minimum_task_gap_reserved_fraction": (
            minimum_task_gap_reserved_fraction if dominance_applicable_groups else None
        ),
        "all_groups_raw_nonuniform": True,
        "all_rows_gold_present_slate16": True,
        "raw_scores_preserved": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--through-step", type=int)
    args = parser.parse_args()

    train_dir = args.run_dir / "rollout_result/train"
    paths = sorted(
        (path for path in train_dir.glob("*.jsonl") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if args.through_step is not None:
        expected = list(range(args.through_step + 1))
        paths = [path for path in paths if int(path.stem) <= args.through_step]
        got = [int(path.stem) for path in paths]
        if got != expected:
            raise SystemExit(f"missing rollout artifacts through step {args.through_step}: got={got}")
    if not paths:
        raise SystemExit(f"no rollout JSONL files under {train_dir}")
    task_metadata = load_task_metadata(args.data_dir)
    result = {
        "run_dir": str(args.run_dir),
        "steps": [audit_step(path, task_metadata) for path in paths],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
