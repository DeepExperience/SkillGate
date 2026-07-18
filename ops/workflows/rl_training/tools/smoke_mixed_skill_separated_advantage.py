#!/usr/bin/env python3
"""Deterministic smoke and exhaustive dominance test for separated advantage."""

from __future__ import annotations

import ast
import itertools
import json
import math
import os
import random
import struct
import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "Relax"))

from examples.agent_bench.mixed_skill_separated_advantage import (  # noqa: E402
    _behavior_advantages,
    _task_advantages,
    compute_group_advantages,
    config_from_env,
    keep_raw_task_reward_nonzero_std,
    post_process_rewards,
)
from relax.utils.types import Sample  # noqa: E402


GOLD = "gold-skill"
MISLEADING = [f"misleading-{index}" for index in range(5)]
RELEVANT = [f"relevant-{index}" for index in range(5)]
IRRELEVANT = [f"irrelevant-{index}" for index in range(5)]
RETRIEVAL = [GOLD, *MISLEADING, *RELEVANT, *IRRELEVANT]


def args() -> Namespace:
    return Namespace(
        reward_key="score",
        n_samples_per_prompt=8,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
    )


def read_call(name: str) -> str:
    return (
        "<tool_call>\n<function=exec>\n<parameter=command>\n"
        f"cat /root/.claude/skills/{name}/SKILL.md\n"
        "</parameter>\n</function>\n</tool_call>"
    )


def extra() -> dict:
    return {
        "bench": "seta_synth",
        "task_id": "separated-smoke",
        "update_kind": "mixed_separated_continuous_advantage_grpo",
        "hybrid_update_kind": "mixed_separated_continuous_advantage_grpo",
        "retrieval_skills_top_n": list(RETRIEVAL),
        "slate_contains_gold": 1.0,
        "slate_gold_name": GOLD,
        "slate_misleading_names": list(MISLEADING),
        "slate_relevant_names": list(RELEVANT),
        "slate_irrelevant_names": list(IRRELEVANT),
        "mixed_skill_separated_advantage": 1.0,
        "mixed_skill_separated_schema": "continuous_task_grpo_plus_adaptive_outcome_stratified_behavior_v3",
        "mixed_skill_task_outcome": "pass_at_1",
        "mixed_skill_task_advantage": "continuous_raw_grpo",
        "mixed_skill_raw_score_preserved": 1.0,
    }


def response(behavior: str) -> str:
    if behavior == "oracle":
        return read_call(GOLD)
    if behavior == "misleading":
        return read_call(MISLEADING[0])
    if behavior == "both":
        return read_call(GOLD) + read_call(MISLEADING[0])
    if behavior == "other":
        return read_call(RELEVANT[0])
    if behavior == "none":
        return "solve directly"
    raise ValueError(behavior)


def sample(outcome: float, behavior: str, index: int, *, metadata: dict | None = None) -> Sample:
    return Sample(
        index=index,
        response=response(behavior),
        response_length=32,
        reward={"score": outcome, "raw_score": outcome},
        metadata={"extra_info": extra() if metadata is None else metadata},
        train_metadata={},
        status=Sample.Status.COMPLETED,
    )


def configure() -> None:
    os.environ["RELAX_MIXED_SEPARATED_ADV_ENABLED"] = "1"
    os.environ["RELAX_MIXED_SEPARATED_BEHAVIOR_COEF"] = "0.30"
    os.environ["RELAX_MIXED_SEPARATED_BEHAVIOR_CLIP"] = "0.40"
    os.environ["PASS_REWARD_THRESHOLD"] = "1.0"


def test_worst_case_group() -> dict:
    outcomes = [1.0] * 4 + [0.0] * 4
    behaviors = ["misleading", "oracle", "oracle", "oracle", "oracle", "misleading", "misleading", "misleading"]
    group = [
        sample(outcome, behavior, index)
        for index, (outcome, behavior) in enumerate(zip(outcomes, behaviors, strict=True))
    ]
    result = compute_group_advantages(args(), group)
    raw_rewards, totals = post_process_rewards(args(), group)
    assert raw_rewards == outcomes
    assert all(math.isclose(got, want, abs_tol=1e-12) for got, want in zip(totals, result.total_advantages))
    assert math.isclose(result.dominance_gap, 1.070825185, rel_tol=0.0, abs_tol=2e-6)
    assert min(totals[:4]) > max(totals[4:])
    assert math.isclose(sum(result.behavior_advantages[:4]), 0.0, abs_tol=1e-9)
    assert math.isclose(sum(result.behavior_advantages[4:]), 0.0, abs_tol=1e-9)
    assert max(abs(value) for value in result.behavior_advantages) <= 0.40000001
    assert [row.reward["score"] for row in group] == outcomes
    assert [row.reward["raw_score"] for row in group] == outcomes
    assert all("mixed_sep_total_advantage" in row.reward for row in group)

    both = sample(1.0, "both", 0)
    mixed_group = [both, *[sample(float(index % 2), "none", index) for index in range(1, 8)]]
    mixed_group[0].reward = {"score": 1.0, "raw_score": 1.0}
    mixed_result = compute_group_advantages(args(), mixed_group)
    assert mixed_result.attributions[0].behavior == "misleading"
    return {
        "task_advantages": list(result.task_advantages),
        "behavior_advantages": list(result.behavior_advantages),
        "total_advantages": list(result.total_advantages),
        "dominance_gap": result.dominance_gap,
        "gold_plus_misleading": mixed_result.attributions[0].behavior,
    }


def test_dynamic_filter() -> dict:
    all_fail = [sample(0.0, behavior, index) for index, behavior in enumerate(
        ["oracle", "misleading", "none", "other", "oracle", "misleading", "none", "other"]
    )]
    all_pass = [sample(1.0, behavior, index) for index, behavior in enumerate(
        ["oracle", "misleading", "none", "other", "oracle", "misleading", "none", "other"]
    )]
    mixed = [sample(float(index % 2), "none", index) for index in range(8)]
    fail_result = keep_raw_task_reward_nonzero_std(args(), all_fail)
    pass_result = keep_raw_task_reward_nonzero_std(args(), all_pass)
    mixed_result = keep_raw_task_reward_nonzero_std(args(), mixed)
    assert not fail_result.keep and fail_result.reason == "zero_std_raw_task_0.000"
    assert not pass_result.keep and pass_result.reason == "zero_std_raw_task_1.000"
    assert mixed_result.keep
    assert all("mixed_sep_total_advantage" in row.reward for row in mixed)

    raw_uniform = [sample(0.0, "none", index) for index in range(8)]
    for index, row in enumerate(raw_uniform):
        row.reward["score"] = float(index % 2)
    raw_guard = keep_raw_task_reward_nonzero_std(args(), raw_uniform)
    assert not raw_guard.keep and raw_guard.reason == "zero_std_raw_task_0.000"

    fractional_raw = [0.167, 0.0, 0.0, 0.667, 1.0, 0.333, 0.667, 0.167]
    fractional = [
        sample(value, ["oracle", "misleading", "none", "other"][index % 4], index)
        for index, value in enumerate(fractional_raw)
    ]
    fractional_result = keep_raw_task_reward_nonzero_std(args(), fractional)
    assert fractional_result.keep
    computed = compute_group_advantages(args(), fractional)
    returned_raw, returned_total = post_process_rewards(args(), fractional)
    assert list(computed.raw_scores) == fractional_raw
    assert list(computed.outcomes) == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert returned_raw == fractional_raw
    assert all(
        math.isclose(got, want, abs_tol=1e-12)
        for got, want in zip(returned_total, computed.total_advantages, strict=True)
    )
    assert min(
        total for total, outcome in zip(computed.total_advantages, computed.outcomes, strict=True)
        if outcome == 1.0
    ) > max(
        total for total, outcome in zip(computed.total_advantages, computed.outcomes, strict=True)
        if outcome == 0.0
    )
    assert [row.reward["raw_score"] for row in fractional] == fractional_raw

    partial_no_pass_raw = [0.0, 0.167, 0.333, 0.667, 0.167, 0.333, 0.5, 0.0]
    partial_no_pass = [
        sample(value, ["oracle", "misleading", "none", "other"][index % 4], index)
        for index, value in enumerate(partial_no_pass_raw)
    ]
    partial_filter = keep_raw_task_reward_nonzero_std(args(), partial_no_pass)
    assert partial_filter.keep
    partial_result = compute_group_advantages(args(), partial_no_pass)
    partial_returned_raw, _ = post_process_rewards(args(), partial_no_pass)
    assert partial_result.success_count == 0 and partial_result.failure_count == 8
    assert not partial_result.dominance_applicable and partial_result.dominance_gap == 0.0
    assert partial_returned_raw == partial_no_pass_raw

    tiny_gap_raw = [1.0, 0.999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    tiny_gap_behaviors = ["misleading", "oracle", "misleading", "misleading", "misleading", "misleading", "misleading", "misleading"]
    tiny_gap = [
        sample(value, behavior, index)
        for index, (value, behavior) in enumerate(
            zip(tiny_gap_raw, tiny_gap_behaviors, strict=True)
        )
    ]
    tiny_result = compute_group_advantages(args(), tiny_gap)
    assert 0.0 < tiny_result.behavior_dominance_scale < 1.0
    assert tiny_result.dominance_gap + 1e-8 >= 0.5 * tiny_result.task_dominance_gap
    return {
        "all_fail_keep": bool(fail_result.keep),
        "all_pass_keep": bool(pass_result.keep),
        "mixed_keep": bool(mixed_result.keep),
        "raw_uniform_score_mixed_keep": bool(raw_guard.keep),
        "fractional_raw_scores": fractional_raw,
        "fractional_pass_outcomes": list(computed.outcomes),
        "fractional_raw_preserved": returned_raw == fractional_raw,
        "fractional_dominance_gap": computed.dominance_gap,
        "partial_no_pass_keep": bool(partial_filter.keep),
        "partial_no_pass_raw_preserved": partial_returned_raw == partial_no_pass_raw,
        "partial_no_pass_success_count": partial_result.success_count,
        "tiny_gap_task_dominance": tiny_result.task_dominance_gap,
        "tiny_gap_behavior_scale": tiny_result.behavior_dominance_scale,
        "tiny_gap_total_dominance": tiny_result.dominance_gap,
    }


def test_exhaustive_dominance() -> dict:
    config = config_from_env()
    utility_values = (config.oracle_utility, config.misleading_utility, config.no_read_utility)
    minimum_gap = float("inf")
    combinations = 0
    for success_count in range(1, 8):
        outcomes = [1.0] * success_count + [0.0] * (8 - success_count)
        task = _task_advantages(outcomes)
        for utilities in itertools.product(utility_values, repeat=8):
            _, _, behavior = _behavior_advantages(
                outcomes, utilities, config, config.behavior_clip
            )
            totals = [left + right for left, right in zip(task, behavior, strict=True)]
            gap = min(totals[:success_count]) - max(totals[success_count:])
            assert gap > 0.0
            assert max(abs(value) for value in behavior) <= 0.40000001
            assert math.isclose(sum(behavior[:success_count]), 0.0, abs_tol=1e-8)
            assert math.isclose(sum(behavior[success_count:]), 0.0, abs_tol=1e-8)
            minimum_gap = min(minimum_gap, gap)
            combinations += 1
    assert minimum_gap > 1.0708
    return {"combinations": combinations, "minimum_gap": minimum_gap}


def test_random_continuous_dominance() -> dict:
    rng = random.Random(20260710)
    behavior_names = ("oracle", "misleading", "none", "other")
    groups = 3000
    applicable = 0
    partial_only = 0
    scaled = 0
    guarded = 0
    minimum_reserved_fraction = float("inf")
    maximum_behavior = 0.0
    for group_index in range(groups):
        if group_index % 3 == 0:
            raw_scores = [round(rng.random() * 0.999, 6) for _ in range(8)]
            partial_only += 1
        else:
            raw_scores = [round(rng.random() * 0.999, 6) for _ in range(8)]
            raw_scores[rng.randrange(8)] = 1.0
        if group_index % 97 == 0:
            # Adversarial near-threshold failure next to a full success.
            raw_scores[0] = 1.0
            raw_scores[1] = 1.0 - 10 ** (-rng.randint(3, 12))
        behaviors = [rng.choice(behavior_names) for _ in range(8)]
        group = [
            sample(value, behavior, index)
            for index, (value, behavior) in enumerate(
                zip(raw_scores, behaviors, strict=True)
            )
        ]
        result = compute_group_advantages(args(), group)
        assert list(result.raw_scores) == raw_scores
        assert math.isclose(sum(result.task_advantages), 0.0, abs_tol=2e-6)
        assert math.isclose(sum(result.behavior_advantages), 0.0, abs_tol=1e-8)
        assert math.isclose(sum(result.total_advantages), 0.0, abs_tol=2e-6)
        maximum_behavior = max(
            maximum_behavior,
            max(abs(value) for value in result.behavior_advantages),
        )
        if result.dominance_applicable:
            applicable += 1
            assert result.dominance_gap > 0.0
            assert result.dominance_gap + 1e-8 >= 0.5 * result.task_dominance_gap
            minimum_reserved_fraction = min(
                minimum_reserved_fraction,
                result.dominance_gap / result.task_dominance_gap,
            )
            float32_totals = [
                struct.unpack("f", struct.pack("f", value))[0]
                for value in result.total_advantages
            ]
            success_totals = [
                value
                for value, outcome in zip(float32_totals, result.outcomes, strict=True)
                if outcome == 1.0
            ]
            failure_totals = [
                value
                for value, outcome in zip(float32_totals, result.outcomes, strict=True)
                if outcome == 0.0
            ]
            assert min(success_totals) > max(failure_totals)
        if result.behavior_dominance_scale < 1.0:
            scaled += 1
        if result.task_pass_guard > 0.0:
            guarded += 1
    return {
        "groups": groups,
        "dominance_applicable_groups": applicable,
        "partial_only_groups": partial_only,
        "adaptive_scaled_groups": scaled,
        "numeric_pass_guard_groups": guarded,
        "minimum_task_gap_reserved_fraction": minimum_reserved_fraction,
        "maximum_abs_behavior_advantage": maximum_behavior,
    }


def test_hard_guards_and_passrate_transport() -> dict:
    bad = extra()
    bad["slate_contains_gold"] = 0.0
    bad_group = [sample(float(index % 2), "none", index, metadata=dict(bad)) for index in range(8)]
    try:
        post_process_rewards(args(), bad_group)
    except ValueError as error:
        gold_guard = str(error)
    else:
        raise AssertionError("gold-absent group did not fail")

    os.environ["RELAX_MIXED_SEPARATED_ADV_ENABLED"] = "0"
    try:
        post_process_rewards(args(), [sample(float(index % 2), "none", index) for index in range(8)])
    except RuntimeError as error:
        enabled_guard = str(error)
    else:
        raise AssertionError("disabled separated mode did not fail")
    finally:
        os.environ["RELAX_MIXED_SEPARATED_ADV_ENABLED"] = "1"

    missing_raw_group = [sample(float(index % 2), "none", index) for index in range(8)]
    missing_raw_group[3].reward.pop("raw_score")
    try:
        post_process_rewards(args(), missing_raw_group)
    except ValueError as error:
        raw_score_guard = str(error)
    else:
        raise AssertionError("missing verifier raw_score did not fail")

    actor_path = ROOT / "Relax/relax/backends/megatron/actor.py"
    tree = ast.parse(actor_path.read_text())
    functions: dict[str, int] = {}
    for name in ("train", "train_async"):
        nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
        assert len(nodes) == 1
        lists = []
        for node in ast.walk(nodes[0]):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.List):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "data_fields" for target in node.targets):
                continue
            values = [
                item.value for item in node.value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            ]
            if "raw_reward" in values:
                lists.append(values)
        assert lists and all("clean_pass_score" in values for values in lists)
        functions[name] = len(lists)

    env_names = {
        "RELAX_MIXED_SEPARATED_ADV_ENABLED",
        "RELAX_MIXED_SEPARATED_BEHAVIOR_COEF",
        "RELAX_MIXED_SEPARATED_BEHAVIOR_CLIP",
    }
    utils_text = (ROOT / "Relax/relax/utils/utils.py").read_text()
    ray_job_text = (ROOT / "Relax/scripts/entrypoint/ray-job.sh").read_text()
    missing_utils = sorted(name for name in env_names if f'"{name}"' not in utils_text)
    # ray-job.sh contains escaped JSON quotes inside a shell string, so match
    # the exact variable name and separately require its shell expansion.
    missing_runtime = sorted(
        name
        for name in env_names
        if name not in ray_job_text or f"${{{name}" not in ray_job_text
    )
    assert not missing_utils, missing_utils
    assert not missing_runtime, missing_runtime
    return {
        "gold_guard": gold_guard,
        "enabled_guard": enabled_guard,
        "raw_score_guard": raw_score_guard,
        "clean_pass_score_actor_fields": functions,
        "ray_env_propagation": sorted(env_names),
    }


def main() -> None:
    configure()
    result = {
        "worst_case": test_worst_case_group(),
        "dynamic_filter": test_dynamic_filter(),
        "exhaustive": test_exhaustive_dominance(),
        "random_continuous": test_random_continuous_dominance(),
        "guards": test_hard_guards_and_passrate_transport(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
