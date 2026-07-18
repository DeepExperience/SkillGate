"""Pure-mixed GRPO with separated task and skill-choice advantages.

Every prompt group contains eight rollouts of the same always-gold 16-skill
slate.  Factual task success remains the primary optimization signal:

``A_total = A_task + A_behavior``

``A_task`` is ordinary sample-standardized GRPO over the original verifier
``raw_score``, including benchmark partial credit such as 0.167/0.667.
``A_behavior`` is computed independently inside the success and failure
strata, using the fixed ordering ``oracle > no-read/other > misleading``.
The whole stratum is scaled together, preserving zero mean and behavior order,
and each behavior correction is bounded by 0.40.

When a group contains both a full success (raw_score >= 1.0) and a failure,
the behavior term is globally scaled only if its actual direction would
consume more than half of that group's task-advantage success/failure gap.
This keeps the nominal behavior signal whenever it is harmless while reserving
at least half of the factual task gap.  A runtime assertion enforces strict
success dominance for every applicable group.  A zero-mean pass guard is used
only when the continuous gap is below 1e-5, preventing float32 collapse without
changing any ordering inside the success or failure stratum.

Dynamic sampling admission uses variance of the original verifier score,
matching the historical Relax GRPO rule.  Uniform groups are refilled, while
all-failure groups with differing partial scores remain learnable.  Returned
raw rewards stay unchanged; only the processed advantage adds the separated
behavior term, keeping W&B pass@k and raw-score statistics comparable.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any, Sequence

from examples.agent_bench.mixed_skill_bonus_compare import (
    Attribution,
    SkillCategories,
    attribute_behavior,
    factual_outcome_score,
    skill_categories,
)
from relax.engine.filters.base_types import DynamicFilterOutput
from relax.utils.types import Sample


EXPECTED_UPDATE_KIND = "mixed_separated_continuous_advantage_grpo"
EXPECTED_GROUP_SIZE = 8
EXPECTED_SCHEMA = "continuous_task_grpo_plus_adaptive_outcome_stratified_behavior_v3"
MAX_BEHAVIOR_HARM_TASK_GAP_FRACTION = 0.50
MIN_NUMERIC_TASK_DOMINANCE_GAP = 1e-5
_EPS = 1e-8


@dataclass(frozen=True)
class Config:
    enabled: bool
    behavior_coef: float
    behavior_clip: float
    oracle_utility: float
    misleading_utility: float
    no_read_utility: float
    other_utility: float
    pass_threshold: float


@dataclass(frozen=True)
class GroupAdvantages:
    raw_scores: tuple[float, ...]
    outcomes: tuple[float, ...]
    task_advantages: tuple[float, ...]
    attributions: tuple[Attribution, ...]
    behavior_utilities: tuple[float, ...]
    behavior_centered_utilities: tuple[float, ...]
    behavior_scales: tuple[float, ...]
    behavior_advantages: tuple[float, ...]
    total_advantages: tuple[float, ...]
    success_count: int
    failure_count: int
    behavior_cap: float
    behavior_dominance_scale: float
    behavior_cross_harm: float
    task_pass_guard: float
    task_dominance_gap: float
    dominance_applicable: bool
    dominance_gap: float


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return float(value)


def config_from_env() -> Config:
    config = Config(
        enabled=_env_bool("RELAX_MIXED_SEPARATED_ADV_ENABLED", False),
        behavior_coef=_env_float("RELAX_MIXED_SEPARATED_BEHAVIOR_COEF", 0.30),
        behavior_clip=_env_float("RELAX_MIXED_SEPARATED_BEHAVIOR_CLIP", 0.40),
        oracle_utility=1.0,
        misleading_utility=-1.0,
        no_read_utility=-0.25,
        other_utility=-0.25,
        pass_threshold=_env_float("PASS_REWARD_THRESHOLD", 1.0),
    )
    numeric = (
        config.behavior_coef,
        config.behavior_clip,
        config.oracle_utility,
        config.misleading_utility,
        config.no_read_utility,
        config.other_utility,
        config.pass_threshold,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError(f"separated-advantage config must be finite: {config}")
    if config.behavior_coef < 0.0:
        raise ValueError("RELAX_MIXED_SEPARATED_BEHAVIOR_COEF must be non-negative")
    if not 0.0 < config.behavior_clip <= 0.40:
        raise ValueError(
            "RELAX_MIXED_SEPARATED_BEHAVIOR_CLIP must be in (0, 0.40] "
            "to preserve the hard success-over-failure margin"
        )
    if not (
        config.oracle_utility
        > config.no_read_utility
        >= config.misleading_utility
        and config.oracle_utility
        > config.other_utility
        >= config.misleading_utility
    ):
        raise ValueError(
            "behavior utilities must rank oracle above no-read/other and "
            f"no-read/other above or equal to misleading: {config}"
        )
    if not math.isclose(config.pass_threshold, 1.0, abs_tol=1e-12):
        raise ValueError(
            "separated advantage requires PASS_REWARD_THRESHOLD=1.0 for pass/fail behavior strata"
        )
    return config


def _require_enabled() -> Config:
    config = config_from_env()
    if not config.enabled:
        raise RuntimeError(
            "mixed_skill_separated_advantage was selected but "
            "RELAX_MIXED_SEPARATED_ADV_ENABLED is not 1"
        )
    return config


def _extra_info(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    extra = metadata.get("extra_info")
    return extra if isinstance(extra, dict) else metadata


def _validate_train_args(args: Any) -> None:
    if int(getattr(args, "n_samples_per_prompt", 0) or 0) != EXPECTED_GROUP_SIZE:
        raise ValueError(
            f"separated advantage requires n_samples_per_prompt={EXPECTED_GROUP_SIZE}"
        )
    if str(getattr(args, "advantage_estimator", "grpo") or "grpo") != "grpo":
        raise ValueError("separated advantage requires advantage_estimator=grpo")
    if not bool(getattr(args, "rewards_normalization", True)):
        raise ValueError("separated advantage requires rewards_normalization=true")
    if not bool(getattr(args, "grpo_std_normalization", True)):
        raise ValueError("separated advantage requires grpo_std_normalization=true")


def _validate_group(group: Sequence[Sample]) -> SkillCategories:
    if len(group) != EXPECTED_GROUP_SIZE:
        raise ValueError(
            f"separated advantage requires {EXPECTED_GROUP_SIZE} samples, got {len(group)}"
        )
    first = skill_categories(group[0], expected_update_kind=EXPECTED_UPDATE_KIND)
    for index, sample in enumerate(group):
        categories = skill_categories(sample, expected_update_kind=EXPECTED_UPDATE_KIND)
        if categories != first:
            raise ValueError(f"separated-advantage group category mismatch at sample {index}")
        extra = _extra_info(sample)
        if float(extra.get("mixed_skill_separated_advantage") or 0.0) != 1.0:
            raise ValueError("missing mixed_skill_separated_advantage=1 metadata marker")
        if str(extra.get("mixed_skill_separated_schema") or "") != EXPECTED_SCHEMA:
            raise ValueError("unexpected mixed_skill_separated_schema metadata")
        if str(extra.get("mixed_skill_task_outcome") or "") != "pass_at_1":
            raise ValueError("separated advantage requires mixed_skill_task_outcome=pass_at_1")
        if str(extra.get("mixed_skill_task_advantage") or "") != "continuous_raw_grpo":
            raise ValueError(
                "separated advantage requires mixed_skill_task_advantage=continuous_raw_grpo"
            )
        if float(extra.get("mixed_skill_raw_score_preserved") or 0.0) != 1.0:
            raise ValueError("separated advantage requires mixed_skill_raw_score_preserved=1")
    return first


def _raw_scores_and_outcomes(
    group: Sequence[Sample],
    args: Any,
    config: Config,
) -> tuple[list[float], list[float]]:
    raw_scores: list[float] = []
    for index, sample in enumerate(group):
        if not isinstance(sample.reward, dict) or "raw_score" not in sample.reward:
            raise ValueError(
                "separated advantage requires verifier reward['raw_score'] for "
                f"every trajectory; missing at sample {index}"
            )
        try:
            raw_score = float(sample.reward["raw_score"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid verifier raw_score at sample {index}: {sample.reward['raw_score']!r}"
            ) from error
        # Keep this equality check against the shared helper as a regression
        # guard: the helper must continue to prefer the verifier's raw score.
        if not math.isclose(raw_score, float(factual_outcome_score(sample, args)), abs_tol=0.0):
            raise AssertionError("factual outcome helper stopped preferring raw_score")
        raw_scores.append(raw_score)
    if not all(math.isfinite(value) for value in raw_scores):
        raise ValueError(f"separated advantage requires finite verifier raw scores, got {raw_scores}")
    outcomes = [1.0 if value >= config.pass_threshold else 0.0 for value in raw_scores]
    return raw_scores, outcomes


def _task_advantages(raw_scores: Sequence[float]) -> list[float]:
    if max(raw_scores) - min(raw_scores) <= 1e-12:
        raise ValueError(f"separated advantage requires non-uniform raw scores, got {raw_scores}")
    center = mean(raw_scores)
    scale = stdev(raw_scores) + 1e-6
    return [float((value - center) / scale) for value in raw_scores]


def _utility(config: Config, attribution: Attribution) -> float:
    if attribution.behavior == "oracle":
        return config.oracle_utility
    if attribution.behavior == "misleading":
        return config.misleading_utility
    if attribution.behavior == "none":
        return config.no_read_utility
    if attribution.behavior == "other":
        return config.other_utility
    raise ValueError(f"unknown separated-advantage behavior: {attribution.behavior!r}")


def _behavior_advantages(
    outcomes: Sequence[float],
    utilities: Sequence[float],
    config: Config,
    behavior_cap: float,
) -> tuple[list[float], list[float], list[float]]:
    centered = [0.0] * len(outcomes)
    scales = [0.0] * len(outcomes)
    advantages = [0.0] * len(outcomes)
    for outcome in (0.0, 1.0):
        indices = [index for index, value in enumerate(outcomes) if value == outcome]
        if not indices:
            continue
        stratum_mean = mean(utilities[index] for index in indices)
        deviations = [float(utilities[index] - stratum_mean) for index in indices]
        max_abs = max((abs(value) for value in deviations), default=0.0)
        scale = (
            0.0
            if max_abs <= _EPS
            else min(config.behavior_coef, behavior_cap / max_abs)
        )
        for index, deviation in zip(indices, deviations, strict=True):
            centered[index] = deviation
            scales[index] = scale
            advantages[index] = float(scale * deviation)
        if abs(sum(advantages[index] for index in indices)) > _EPS:
            raise AssertionError("behavior advantage is not zero-mean inside outcome stratum")
    if max(abs(value) for value in advantages) > behavior_cap + _EPS:
        raise AssertionError("behavior advantage exceeded effective hard cap")
    return centered, scales, advantages


def compute_group_advantages(
    args: Any,
    group: Sequence[Sample],
    *,
    config: Config | None = None,
) -> GroupAdvantages:
    """Compute and validate separated advantages for one accepted group."""

    config = config or _require_enabled()
    _validate_train_args(args)
    categories = _validate_group(group)
    raw_scores, outcomes = _raw_scores_and_outcomes(group, args, config)
    task_advantages = _task_advantages(raw_scores)
    attributions = [attribute_behavior(sample, categories) for sample in group]
    utilities = [_utility(config, attribution) for attribution in attributions]
    success_indices = [index for index, value in enumerate(outcomes) if value == 1.0]
    failure_indices = [index for index, value in enumerate(outcomes) if value == 0.0]
    dominance_applicable = bool(success_indices and failure_indices)
    if dominance_applicable:
        task_dominance_gap = min(task_advantages[index] for index in success_indices) - max(
            task_advantages[index] for index in failure_indices
        )
        if not task_dominance_gap > 0.0:
            raise AssertionError(
                "continuous task advantage does not rank success above failure: "
                f"gap={task_dominance_gap}, raw_scores={raw_scores}"
            )
        task_pass_guard = max(0.0, MIN_NUMERIC_TASK_DOMINANCE_GAP - task_dominance_gap)
        if task_pass_guard > 0.0:
            success_fraction = len(success_indices) / len(outcomes)
            task_advantages = [
                float(
                    value
                    + task_pass_guard * (outcome - success_fraction)
                )
                for value, outcome in zip(task_advantages, outcomes, strict=True)
            ]
            task_dominance_gap = min(
                task_advantages[index] for index in success_indices
            ) - max(task_advantages[index] for index in failure_indices)
    else:
        task_pass_guard = 0.0
        task_dominance_gap = 0.0
    centered, scales, behavior_advantages = _behavior_advantages(
        outcomes, utilities, config, config.behavior_clip
    )
    behavior_dominance_scale = 1.0
    behavior_cross_harm = 0.0
    if dominance_applicable:
        behavior_cross_harm = max(
            behavior_advantages[index] for index in failure_indices
        ) - min(behavior_advantages[index] for index in success_indices)
        allowed_harm = MAX_BEHAVIOR_HARM_TASK_GAP_FRACTION * task_dominance_gap
        if behavior_cross_harm > allowed_harm:
            behavior_dominance_scale = allowed_harm / behavior_cross_harm
            behavior_advantages = [
                float(value * behavior_dominance_scale)
                for value in behavior_advantages
            ]
            scales = [float(value * behavior_dominance_scale) for value in scales]
            behavior_cross_harm *= behavior_dominance_scale
    behavior_cap = max(abs(value) for value in behavior_advantages)
    totals = [
        float(task + behavior)
        for task, behavior in zip(task_advantages, behavior_advantages, strict=True)
    ]
    if dominance_applicable:
        dominance_gap = min(totals[index] for index in success_indices) - max(
            totals[index] for index in failure_indices
        )
        if not dominance_gap > _EPS:
            raise AssertionError(
                "task-success dominance violated: "
                f"gap={dominance_gap}, raw_scores={raw_scores}, totals={totals}"
            )
        minimum_reserved_gap = (
            1.0 - MAX_BEHAVIOR_HARM_TASK_GAP_FRACTION
        ) * task_dominance_gap
        if dominance_gap + _EPS < minimum_reserved_gap:
            raise AssertionError(
                "success dominance retained less than the guaranteed task gap: "
                f"actual={dominance_gap}, guaranteed={minimum_reserved_gap}"
            )
    else:
        dominance_gap = 0.0
    if abs(sum(task_advantages)) > 1e-6 or abs(sum(behavior_advantages)) > _EPS:
        raise AssertionError("separated task/behavior advantages must each be group-zero-mean")
    if abs(sum(totals)) > 1e-6:
        raise AssertionError("separated total advantage must be group-zero-mean")
    return GroupAdvantages(
        raw_scores=tuple(raw_scores),
        outcomes=tuple(outcomes),
        task_advantages=tuple(task_advantages),
        attributions=tuple(attributions),
        behavior_utilities=tuple(utilities),
        behavior_centered_utilities=tuple(centered),
        behavior_scales=tuple(scales),
        behavior_advantages=tuple(behavior_advantages),
        total_advantages=tuple(totals),
        success_count=len(success_indices),
        failure_count=len(failure_indices),
        behavior_cap=float(behavior_cap),
        behavior_dominance_scale=float(behavior_dominance_scale),
        behavior_cross_harm=float(behavior_cross_harm),
        task_pass_guard=float(task_pass_guard),
        task_dominance_gap=float(task_dominance_gap),
        dominance_applicable=dominance_applicable,
        dominance_gap=float(dominance_gap),
    )


def _annotate(group: Sequence[Sample], result: GroupAdvantages) -> None:
    for index, sample in enumerate(group):
        if not isinstance(sample.reward, dict):
            raise ValueError("separated advantage requires dictionary rewards")
        attribution = result.attributions[index]
        values = {
            "mixed_sep_verifier_raw_score": result.raw_scores[index],
            "mixed_sep_task_outcome": result.outcomes[index],
            "mixed_sep_task_advantage": result.task_advantages[index],
            "mixed_sep_behavior_utility": result.behavior_utilities[index],
            "mixed_sep_behavior_centered_utility": result.behavior_centered_utilities[index],
            "mixed_sep_behavior_scale": result.behavior_scales[index],
            "mixed_sep_behavior_advantage": result.behavior_advantages[index],
            "mixed_sep_total_advantage": result.total_advantages[index],
            "mixed_sep_success_count": float(result.success_count),
            "mixed_sep_failure_count": float(result.failure_count),
            "mixed_sep_behavior_cap": result.behavior_cap,
            "mixed_sep_behavior_dominance_scale": result.behavior_dominance_scale,
            "mixed_sep_behavior_cross_harm": result.behavior_cross_harm,
            "mixed_sep_task_pass_guard": result.task_pass_guard,
            "mixed_sep_task_success_dominance_gap": result.task_dominance_gap,
            "mixed_sep_success_dominance_applicable": (
                1.0 if result.dominance_applicable else 0.0
            ),
            "mixed_sep_success_dominance_gap": result.dominance_gap,
            "mixed_sep_any_read": 1.0 if attribution.any_read else 0.0,
            "mixed_sep_gold_read": 1.0 if attribution.gold_read else 0.0,
            "mixed_sep_misleading_read": 1.0 if attribution.misleading_read else 0.0,
            "mixed_sep_other_read": 1.0 if attribution.other_read else 0.0,
            "mixed_sep_no_read": 0.0 if attribution.any_read else 1.0,
            "mixed_sep_is_oracle": 1.0 if attribution.behavior == "oracle" else 0.0,
            "mixed_sep_is_misleading": 1.0 if attribution.behavior == "misleading" else 0.0,
            "mixed_sep_is_other": 1.0 if attribution.behavior == "other" else 0.0,
        }
        sample.reward.update(values)


def keep_raw_task_reward_nonzero_std(
    args: Any,
    samples: list[Sample],
    **_: Any,
) -> DynamicFilterOutput:
    """Admit only factual mixed groups and pre-annotate accepted trajectories."""

    config = _require_enabled()
    _validate_train_args(args)
    _validate_group(samples)
    if any(sample.reward is None for sample in samples):
        return DynamicFilterOutput(keep=False, reason="missing_reward")
    raw_scores, _ = _raw_scores_and_outcomes(samples, args, config)
    if max(raw_scores) - min(raw_scores) <= 1e-12:
        return DynamicFilterOutput(
            keep=False,
            reason=f"zero_std_raw_task_{raw_scores[0]:.3f}",
        )
    result = compute_group_advantages(args, samples, config=config)
    _annotate(samples, result)
    return DynamicFilterOutput(keep=True, reason=None)


def _groups(
    samples: list[Sample] | list[list[Sample]],
) -> list[list[Sample]]:
    if not samples:
        return []
    if isinstance(samples[0], list):
        groups = [list(group) for group in samples]
    else:
        flat = list(samples)
        if len(flat) % EXPECTED_GROUP_SIZE:
            raise ValueError(
                f"separated-advantage sample count {len(flat)} is not divisible by {EXPECTED_GROUP_SIZE}"
            )
        groups = [
            flat[start : start + EXPECTED_GROUP_SIZE]
            for start in range(0, len(flat), EXPECTED_GROUP_SIZE)
        ]
    return groups


def post_process_rewards(
    args: Any,
    samples: list[Sample] | list[list[Sample]],
) -> tuple[list[float], list[float]]:
    """Return factual raw rewards and separated total training advantages."""

    config = _require_enabled()
    _validate_train_args(args)
    raw_rewards: list[float] = []
    total_advantages: list[float] = []
    for group in _groups(samples):
        result = compute_group_advantages(args, group, config=config)
        _annotate(group, result)
        raw_rewards.extend(result.raw_scores)
        total_advantages.extend(result.total_advantages)
    return raw_rewards, total_advantages


__all__ = [
    "Config",
    "GroupAdvantages",
    "compute_group_advantages",
    "config_from_env",
    "keep_raw_task_reward_nonzero_std",
    "post_process_rewards",
]
