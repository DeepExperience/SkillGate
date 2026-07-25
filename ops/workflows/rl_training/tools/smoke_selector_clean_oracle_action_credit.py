#!/usr/bin/env python3
"""CPU smoke for clean-oracle trajectory utility and action centering."""

from __future__ import annotations

import json
import math
import os
import sys
from argparse import Namespace
from pathlib import Path
from statistics import mean, stdev

import pandas as pd


ROOT = Path(os.environ.get("ROOT", "/path/to/skillRL"))
sys.path.insert(0, str(ROOT / "Relax"))

from examples.agent_bench.selector_action_credit import build_train_fields  # noqa: E402
from examples.agent_bench.selector_clean_oracle_action_credit import (  # noqa: E402
    CREDIT_SCHEMA,
    annotate_group_selector_advantages,
    keep_raw_task_reward_nonzero_std,
    post_process_rewards,
)
from relax.utils.types import Sample  # noqa: E402


def plain_dict(value):
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, dict):
            return dict(item)
    raise TypeError(type(value))


def plain_list(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def action(skill_name: str, category: str, index: int) -> dict:
    start = 2 * index
    return {
        "skill_name": skill_name,
        "category": category,
        "call_token_indices": [start, start + 1],
        "identity_token_indices": [start + 1],
    }


def make_sample(
    extra: dict, actions: list[dict], score: float, *, mismatch: int = 0
) -> Sample:
    return Sample(
        metadata={
            "extra_info": dict(extra),
            "selector_action_credit": {
                "schema": "selector_action_credit_v1",
                "actions": actions,
                "turns_checked": 1,
                "alignment_mismatch": mismatch,
                "parse_dispatch_mismatch": 0,
                "span_mismatch": 0,
            },
        },
        reward={"score": score, "raw_score": score},
        response_length=16,
        loss_mask=[1] * 16,
    )


def main() -> None:
    os.environ["RELAX_SELECTOR_ACTION_CREDIT"] = "1"
    frame = pd.read_parquet(
        Path(os.environ["DATA_DIR"]) / "train.parquet", columns=["extra_info"]
    )
    extra = plain_dict(frame.iloc[0]["extra_info"])
    oracle = str(extra["slate_gold_name"])
    misleading = str(plain_list(extra["slate_misleading_names"])[0])
    relevant = str(plain_list(extra["slate_relevant_names"])[0])
    irrelevant = str(plain_list(extra["slate_irrelevant_names"])[0])

    action_specs = [
        [(oracle, "oracle")],
        [(misleading, "misleading")],
        [],
        [(oracle, "oracle"), (misleading, "misleading")],
        [(oracle, "oracle"), (oracle, "oracle")],
        [(relevant, "relevant")],
        [(irrelevant, "irrelevant")],
        [(oracle, "oracle")],
    ]
    scores = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    samples = [
        make_sample(
            extra,
            [
                action(skill, category, index)
                for index, (skill, category) in enumerate(spec)
            ],
            score,
        )
        for spec, score in zip(action_specs, scores, strict=True)
    ]

    stats = annotate_group_selector_advantages(samples)
    assert stats["active"] == 1.0 and stats["actions"] == 9.0, stats
    assert stats["clean_oracle_trajectories"] == stats["clean_oracle_actions"] == 2.0, (
        stats
    )
    assert (
        stats["zero_utility_actions"] == 7.0 and stats["multi_read_actions"] == 4.0
    ), stats
    assert stats["positive_actions"] == 2.0 and stats["negative_actions"] == 7.0, stats
    assert math.isclose(stats["baseline"], 2 / 9, abs_tol=1e-12), stats
    assert stats["zero_mean_error"] < 1e-12, stats

    for sample_index, sample in enumerate(samples):
        state = sample.metadata["selector_action_credit"]
        assert state["credit_schema"] == CREDIT_SCHEMA
        for credited_action in state["actions"]:
            if sample_index in {0, 7}:
                assert (
                    credited_action["utility"] == 1.0
                    and credited_action["selector_advantage"] > 0
                )
            else:
                assert (
                    credited_action["utility"] == 0.0
                    and credited_action["selector_advantage"] < 0
                )
        if len(state["actions"]) > 1:
            assert all(item["utility"] == 0.0 for item in state["actions"])

    args = Namespace(
        reward_key="score",
        n_samples_per_prompt=8,
        rewards_normalization=True,
        grpo_std_normalization=True,
    )
    filter_result = keep_raw_task_reward_nonzero_std(args, samples)
    assert filter_result.keep, filter_result
    raw, processed = post_process_rewards(args, samples)
    expected = [(value - mean(scores)) / (stdev(scores) + 1e-6) for value in scores]
    assert raw == scores
    assert all(
        math.isclose(left, right, abs_tol=1e-12)
        for left, right in zip(processed, expected, strict=True)
    )
    assert all(
        float(sample.reward["score"]) == score
        for sample, score in zip(samples, scores, strict=True)
    )

    fields = build_train_fields(samples, [sample.loss_mask for sample in samples])
    base_total = sum(sample.response_length for sample in samples)
    assert math.isclose(
        sum(map(sum, fields["selector_task_loss_weights"])), base_total, abs_tol=1e-3
    )
    assert math.isclose(
        sum(map(sum, fields["selector_action_loss_weights"])), base_total, abs_tol=1e-3
    )
    assert all(
        not (task_weight > 0 and selector_weight > 0)
        for task, selector in zip(
            fields["selector_task_loss_weights"],
            fields["selector_action_loss_weights"],
            strict=True,
        )
        for task_weight, selector_weight in zip(task, selector, strict=True)
    )

    no_clean_samples = [
        make_sample(extra, [action(misleading, "misleading", 0)], score)
        for score in scores
    ]
    no_clean_stats = annotate_group_selector_advantages(no_clean_samples)
    assert no_clean_stats["active"] == 0.0 and no_clean_stats["baseline"] == 0.0, (
        no_clean_stats
    )
    assert all(
        item["selector_advantage"] == 0.0
        for sample in no_clean_samples
        for item in sample.metadata["selector_action_credit"]["actions"]
    )

    lone_clean_samples = [make_sample(extra, [], score) for score in scores]
    lone_clean_samples[0] = make_sample(extra, [action(oracle, "oracle", 0)], scores[0])
    lone_clean_stats = annotate_group_selector_advantages(lone_clean_samples)
    assert lone_clean_stats["active"] == 0.0 and lone_clean_stats["baseline"] == 1.0, (
        lone_clean_stats
    )
    assert (
        lone_clean_samples[0].metadata["selector_action_credit"]["actions"][0][
            "selector_advantage"
        ]
        == 0.0
    )

    mismatched = [
        make_sample(extra, [action(oracle, "oracle", 0)], score) for score in scores
    ]
    mismatched[0].metadata["selector_action_credit"]["alignment_mismatch"] = 1
    mismatch_filter = keep_raw_task_reward_nonzero_std(args, mismatched)
    assert not mismatch_filter.keep and mismatch_filter.reason.startswith(
        "selector_attribution_error_"
    ), mismatch_filter

    print(
        json.dumps(
            {
                "ok": True,
                "credit_schema": CREDIT_SCHEMA,
                "active_group": stats,
                "no_clean_group": no_clean_stats,
                "lone_clean_group": lone_clean_stats,
                "task_weight_sum": sum(map(sum, fields["selector_task_loss_weights"])),
                "selector_weight_sum": sum(
                    map(sum, fields["selector_action_loss_weights"])
                ),
                "mismatch_reason": mismatch_filter.reason,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
