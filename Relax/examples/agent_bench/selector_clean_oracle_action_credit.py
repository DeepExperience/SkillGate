"""Token-local selector credit for exactly one oracle skill read per trajectory."""

from __future__ import annotations

from statistics import mean, stdev
from typing import Any, Sequence

from examples.agent_bench.selector_action_credit import (
    EXPECTED_GROUP_SIZE,
    EXPECTED_UPDATE_KIND,
    _annotate_reward,
    _categories,
    _extra_info,
    _flat_samples,
    _groups,
    _raw_score,
    _selector_state,
    build_train_fields,
    enabled,
    record_assistant_turn,
    sample_behavior_metrics,
)
from relax.engine.filters.base_types import DynamicFilterOutput
from relax.utils.types import Sample


CREDIT_SCHEMA = "selector_clean_oracle_action_credit_v1"


def annotate_group_selector_advantages(group: Sequence[Sample]) -> dict[str, float]:
    """Center clean-oracle action utilities over all read actions in a prompt group."""

    if len(group) != EXPECTED_GROUP_SIZE:
        raise ValueError(f"clean-oracle selector credit requires group size {EXPECTED_GROUP_SIZE}, got {len(group)}")

    oracle = _categories(group[0])["oracle"]
    actions: list[dict[str, Any]] = []
    per_sample_actions: list[list[dict[str, Any]]] = []
    clean_oracle_trajectories = 0
    multi_read_actions = 0
    for sample in group:
        categories = _categories(sample)
        if categories["oracle"] != oracle:
            raise ValueError("clean-oracle selector credit group has inconsistent oracle labels")
        state = _selector_state(sample)
        if any(int(state.get(key, 0)) for key in ("alignment_mismatch", "parse_dispatch_mismatch", "span_mismatch")):
            raise ValueError("clean-oracle selector action attribution mismatch")

        sample_actions = list(state.get("actions") or [])
        clean_oracle = len(sample_actions) == 1 and str(sample_actions[0].get("category") or "") == "oracle"
        clean_oracle_trajectories += int(clean_oracle)
        if len(sample_actions) > 1:
            multi_read_actions += len(sample_actions)
        for action in sample_actions:
            action["utility"] = 1.0 if clean_oracle else 0.0
            action["credit_schema"] = CREDIT_SCHEMA
        state["credit_schema"] = CREDIT_SCHEMA
        state["trajectory_clean_oracle"] = 1.0 if clean_oracle else 0.0
        per_sample_actions.append(sample_actions)
        actions.extend(sample_actions)

    utilities = [float(action["utility"]) for action in actions]
    baseline = float(mean(utilities)) if utilities else 0.0
    for action in actions:
        action["selector_advantage"] = float(action["utility"] - baseline)

    selector_advantages = [float(action["selector_advantage"]) for action in actions]
    positive_actions = sum(value > 0 for value in selector_advantages)
    negative_actions = sum(value < 0 for value in selector_advantages)
    active = positive_actions > 0 and negative_actions > 0
    zero_mean_error = abs(sum(selector_advantages))
    advantage_mean = float(mean(selector_advantages)) if selector_advantages else 0.0
    advantage_min = min(selector_advantages, default=0.0)
    advantage_max = max(selector_advantages, default=0.0)
    clean_oracle_actions = sum(float(action["utility"]) > 0 for action in actions)
    zero_utility_actions = len(actions) - clean_oracle_actions

    for sample, sample_actions in zip(group, per_sample_actions, strict=True):
        state = _selector_state(sample)
        clean_oracle = float(state["trajectory_clean_oracle"])
        state["group_action_baseline"] = baseline
        state["group_selector_active"] = 1.0 if active else 0.0
        state["group_action_count"] = len(actions)
        state["group_clean_oracle_action_count"] = clean_oracle_actions
        state["group_zero_utility_action_count"] = zero_utility_actions
        state["group_weighted_zero_mean_error"] = zero_mean_error
        _annotate_reward(
            sample,
            {
                **sample_behavior_metrics(sample),
                "selector_clean_oracle": clean_oracle,
                "selector_success_and_clean_oracle": clean_oracle * _raw_score(sample, None),
                "selector_active_group": 1.0 if active else 0.0,
                "selector_group_action_count": float(len(actions)),
                "selector_group_clean_oracle_action_count": float(clean_oracle_actions),
                "selector_group_zero_utility_action_count": float(zero_utility_actions),
                "selector_group_multi_read_action_count": float(multi_read_actions),
                "selector_group_positive_action_count": float(positive_actions),
                "selector_group_negative_action_count": float(negative_actions),
                "selector_group_action_baseline": baseline,
                "selector_group_zero_mean_error": zero_mean_error,
                "selector_group_advantage_mean": advantage_mean,
                "selector_group_advantage_min": advantage_min,
                "selector_group_advantage_max": advantage_max,
                "selector_group_no_clean_oracle_action": 1.0 if clean_oracle_actions == 0 else 0.0,
            },
        )

    return {
        "active": 1.0 if active else 0.0,
        "actions": float(len(actions)),
        "clean_oracle_trajectories": float(clean_oracle_trajectories),
        "clean_oracle_actions": float(clean_oracle_actions),
        "zero_utility_actions": float(zero_utility_actions),
        "multi_read_actions": float(multi_read_actions),
        "positive_actions": float(positive_actions),
        "negative_actions": float(negative_actions),
        "baseline": baseline,
        "zero_mean_error": zero_mean_error,
    }


def keep_raw_task_reward_nonzero_std(args: Any, samples: list[Sample], **_: Any) -> DynamicFilterOutput:
    """Keep the established task-outcome dynamic-sampling rule."""

    if not enabled():
        raise RuntimeError("clean-oracle selector filter selected while RELAX_SELECTOR_ACTION_CREDIT is disabled")
    if any(sample.reward is None for sample in samples):
        return DynamicFilterOutput(keep=False, reason="selector_missing_reward")
    if any(
        str(_extra_info(sample).get("update_kind") or _extra_info(sample).get("hybrid_update_kind") or "")
        .strip()
        .lower()
        != EXPECTED_UPDATE_KIND
        for sample in samples
    ):
        return DynamicFilterOutput(keep=False, reason="selector_non_train_update_kind")
    try:
        stats = annotate_group_selector_advantages(samples)
        raw_scores = [_raw_score(sample, args) for sample in samples]
    except (TypeError, ValueError) as error:
        return DynamicFilterOutput(keep=False, reason=f"selector_attribution_error_{type(error).__name__}")
    if max(raw_scores) - min(raw_scores) <= 1e-12:
        suffix = "active" if stats["active"] else "inactive"
        return DynamicFilterOutput(keep=False, reason=f"zero_std_raw_task_selector_{suffix}")
    return DynamicFilterOutput(keep=True, reason=None)


def post_process_rewards(args: Any, samples: list[Sample] | list[list[Sample]]) -> tuple[list[float], list[float]]:
    """Preserve factual task GRPO rewards and add clean-oracle action credit."""

    if not enabled():
        raise RuntimeError("clean-oracle selector reward postprocess selected while feature is disabled")
    flat = _flat_samples(samples)
    if any(
        str(_extra_info(sample).get("update_kind") or _extra_info(sample).get("hybrid_update_kind") or "")
        .strip()
        .lower()
        != EXPECTED_UPDATE_KIND
        for sample in flat
    ):
        raise ValueError(f"clean-oracle selector reward postprocess requires update_kind={EXPECTED_UPDATE_KIND!r}")
    group_size = int(getattr(args, "n_samples_per_prompt", EXPECTED_GROUP_SIZE) or EXPECTED_GROUP_SIZE)
    raw_rewards: list[float] = []
    processed: list[float] = []
    for group in _groups(flat, group_size):
        annotate_group_selector_advantages(group)
        scores = [_raw_score(sample, args) for sample in group]
        raw_rewards.extend(scores)
        if getattr(args, "rewards_normalization", True):
            centered = [float(score - mean(scores)) for score in scores]
            if bool(getattr(args, "grpo_std_normalization", True)) and len(scores) > 1:
                scale = float(stdev(scores)) + 1e-6
                centered = [value / scale for value in centered]
            processed.extend(centered)
        else:
            processed.extend(scores)
    return raw_rewards, processed


__all__ = [
    "CREDIT_SCHEMA",
    "annotate_group_selector_advantages",
    "build_train_fields",
    "enabled",
    "keep_raw_task_reward_nonzero_std",
    "post_process_rewards",
    "record_assistant_turn",
    "sample_behavior_metrics",
]
