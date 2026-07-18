"""Mixed-skill behavior bonus for a simple, auditable GRPO comparison.

This module is deliberately independent from ``slate_regret_gating``.  It is
used by the mixed-only comparison run where every prompt advertises exactly
one oracle skill, five hard-negative misleading skills, five relevant skills,
and five irrelevant skills.

For each eight-sample prompt group we first preserve the verifier outcome and
add a small behavior term:

* read oracle and no misleading skill: ``+0.30``;
* read any misleading skill (including oracle + misleading): ``-0.30``;
* read no skill at all and solve the task: ``+0.35``;
* read only other skills, or fail without reading: ``0``.

The resulting shaped scores are normalized together with the ordinary GRPO
group normalization.  Task success therefore remains the primary signal: with
the default coefficients the worst successful score (0.70) is still above the
best failed score (0.30).  ``reward['raw_score']`` and ``reward['score']`` are
never changed, so task-quality metrics remain factual.

Skill categories come only from frozen parquet metadata.  Actual behavior is
attributed conservatively from explicit read-like tool calls containing a
``/root/.claude/skills/<name>/SKILL.md`` path.  Skill-name mentions and plans to
read a skill do not count.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any, Iterable, Sequence

from relax.engine.filters.base_types import DynamicFilterOutput
from relax.utils.types import Sample


EXPECTED_UPDATE_KIND = "mixed_bonus_compare_grpo"
EXPECTED_GROUP_SIZE = 8
EXPECTED_SLATE_SIZE = 16
EXPECTED_MISLEADING_COUNT = 5
EXPECTED_RELEVANT_COUNT = 5
EXPECTED_IRRELEVANT_COUNT = 5

_TOOL_CALL_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL | re.IGNORECASE)
_SKILL_PATH_RE = re.compile(
    r"(?:/root|~)?/\.claude/skills/([A-Za-z0-9_.-]+)/(?:SKILL|README)\.md",
    re.IGNORECASE,
)
_EXEC_READ_RE = re.compile(
    r"\b(cat|sed|awk|grep|head|tail|less|more|python3?|perl|ruby|node)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Config:
    enabled: bool
    oracle_bonus: float
    misleading_penalty: float
    no_read_success_bonus: float
    pass_threshold: float


@dataclass(frozen=True)
class SkillCategories:
    retrieval: frozenset[str]
    gold: str
    misleading: frozenset[str]
    relevant: frozenset[str]
    irrelevant: frozenset[str]


@dataclass(frozen=True)
class Attribution:
    behavior: str
    read_names: frozenset[str]
    any_read: bool
    gold_read: bool
    misleading_read: bool
    other_read: bool


@dataclass(frozen=True)
class ScoredBehavior:
    outcome: float
    bonus: float
    shaped_score: float
    attribution: Attribution


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
        enabled=_env_bool("RELAX_MIXED_SKILL_BONUS_ENABLED", False),
        oracle_bonus=_env_float("RELAX_MIXED_SKILL_BONUS_ORACLE", 0.30),
        misleading_penalty=_env_float("RELAX_MIXED_SKILL_BONUS_MISLEADING", 0.30),
        no_read_success_bonus=_env_float(
            "RELAX_MIXED_SKILL_BONUS_NO_READ_SUCCESS", 0.35
        ),
        pass_threshold=_env_float("PASS_REWARD_THRESHOLD", 1.0),
    )
    coefficients = (
        config.oracle_bonus,
        config.misleading_penalty,
        config.no_read_success_bonus,
    )
    if any(value < 0.0 or value > 1.0 for value in coefficients):
        raise ValueError(
            "mixed-skill behavior coefficients must be in [0, 1], got "
            f"oracle={config.oracle_bonus}, misleading={config.misleading_penalty}, "
            f"no_read_success={config.no_read_success_bonus}"
        )
    return config


def _require_enabled() -> Config:
    config = config_from_env()
    if not config.enabled:
        raise RuntimeError(
            "mixed_skill_bonus_compare was selected but "
            "RELAX_MIXED_SKILL_BONUS_ENABLED is not 1"
        )
    return config


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _read_names_from_text(text: str) -> set[str]:
    """Extract strictly executed skill-file reads from assistant tool calls."""

    if not text or "/.claude/skills/" not in text:
        return set()
    names: set[str] = set()
    for match in _TOOL_CALL_RE.finditer(text):
        tool_name = match.group(1).strip().lower().rsplit(".", 1)[-1]
        arguments = match.group(2)
        if tool_name == "read":
            pass
        elif tool_name == "exec":
            if not _EXEC_READ_RE.search(arguments):
                continue
        else:
            continue
        names.update(path_match.group(1) for path_match in _SKILL_PATH_RE.finditer(arguments))
    return names


def strict_read_skill_names(sample: Sample) -> frozenset[str]:
    """Return names of skill files actually read by this trajectory.

    Live rollouts retain assistant-only per-turn response text, which avoids
    interpreting a tool response that merely echoes a path as agent behavior.
    Archived samples used by tests may only have ``sample.response``.
    """

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    traces = metadata.get("rollout_traces") or []
    names: set[str] = set()
    saw_trace_text = False
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        inference = trace.get("inference")
        if not isinstance(inference, dict):
            continue
        response_text = inference.get("response_text")
        if not isinstance(response_text, str):
            continue
        saw_trace_text = True
        names.update(_read_names_from_text(response_text))
    if not saw_trace_text:
        names.update(_read_names_from_text(sample.response or ""))
    return frozenset(names)


def _plain_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple, set, frozenset)):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


def _extra_info(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    extra = metadata.get("extra_info")
    return extra if isinstance(extra, dict) else metadata


def _categories(
    sample: Sample,
    *,
    expected_update_kind: str = EXPECTED_UPDATE_KIND,
) -> SkillCategories:
    extra = _extra_info(sample)
    kind = str(extra.get("update_kind") or extra.get("hybrid_update_kind") or "").strip().lower()
    if kind != expected_update_kind:
        raise ValueError(
            f"unexpected update_kind for mixed skill group: {kind!r}; "
            f"expected {expected_update_kind!r}"
        )
    try:
        has_gold = float(extra.get("slate_contains_gold") or 0.0) == 1.0
    except (TypeError, ValueError):
        has_gold = False
    if not has_gold:
        raise ValueError("mixed bonus compare requires slate_contains_gold=1")

    retrieval_list = _plain_list(extra.get("retrieval_skills_top_n"))
    misleading_list = _plain_list(extra.get("slate_misleading_names"))
    relevant_list = _plain_list(extra.get("slate_relevant_names"))
    irrelevant_list = _plain_list(extra.get("slate_irrelevant_names"))
    gold = str(extra.get("slate_gold_name") or "").strip()
    if len(retrieval_list) != EXPECTED_SLATE_SIZE or len(set(retrieval_list)) != EXPECTED_SLATE_SIZE:
        raise ValueError(
            f"mixed bonus compare requires {EXPECTED_SLATE_SIZE} unique skills, "
            f"got {len(retrieval_list)}/{len(set(retrieval_list))}"
        )
    expected_counts = (
        ("misleading", misleading_list, EXPECTED_MISLEADING_COUNT),
        ("relevant", relevant_list, EXPECTED_RELEVANT_COUNT),
        ("irrelevant", irrelevant_list, EXPECTED_IRRELEVANT_COUNT),
    )
    for label, names, expected in expected_counts:
        if len(names) != expected or len(set(names)) != expected:
            raise ValueError(
                f"mixed bonus compare requires {expected} unique {label} skills, "
                f"got {len(names)}/{len(set(names))}"
            )
    if not gold:
        raise ValueError("mixed bonus compare metadata is missing slate_gold_name")

    category_sets = [
        {gold},
        set(misleading_list),
        set(relevant_list),
        set(irrelevant_list),
    ]
    category_union = set().union(*category_sets)
    category_total = sum(len(names) for names in category_sets)
    if len(category_union) != category_total:
        raise ValueError("mixed bonus compare skill categories overlap")
    if category_union != set(retrieval_list):
        missing = sorted(set(retrieval_list) - category_union)
        extra_names = sorted(category_union - set(retrieval_list))
        raise ValueError(
            "mixed bonus compare categories do not exactly cover retrieval list: "
            f"uncategorized={missing}, absent_from_retrieval={extra_names}"
        )
    return SkillCategories(
        retrieval=frozenset(retrieval_list),
        gold=gold,
        misleading=frozenset(misleading_list),
        relevant=frozenset(relevant_list),
        irrelevant=frozenset(irrelevant_list),
    )


def skill_categories(
    sample: Sample,
    *,
    expected_update_kind: str = EXPECTED_UPDATE_KIND,
) -> SkillCategories:
    """Return validated frozen skill categories for a mixed-skill sample."""

    return _categories(sample, expected_update_kind=expected_update_kind)


def attribute_behavior(sample: Sample, categories: SkillCategories | None = None) -> Attribution:
    categories = categories or _categories(sample)
    read_names = strict_read_skill_names(sample)
    gold_read = categories.gold in read_names
    misleading_read = bool(read_names & categories.misleading)
    any_read = bool(read_names)
    other_read = bool(read_names - {categories.gold} - categories.misleading)
    if misleading_read:
        behavior = "misleading"
    elif gold_read:
        behavior = "oracle"
    elif any_read:
        behavior = "other"
    else:
        behavior = "none"
    return Attribution(
        behavior=behavior,
        read_names=read_names,
        any_read=any_read,
        gold_read=gold_read,
        misleading_read=misleading_read,
        other_read=other_read,
    )


def _outcome_score(sample: Sample, args: Any) -> float:
    reward = sample.reward
    if isinstance(reward, dict) and "raw_score" in reward:
        try:
            return float(reward["raw_score"])
        except (TypeError, ValueError):
            pass
    return float(sample.get_reward_value(args))


def factual_outcome_score(sample: Sample, args: Any) -> float:
    """Return the verifier's unshaped task score, preferring ``raw_score``."""

    return _outcome_score(sample, args)


def _score_behavior(
    sample: Sample,
    args: Any,
    config: Config,
    categories: SkillCategories,
) -> ScoredBehavior:
    outcome = _outcome_score(sample, args)
    attribution = attribute_behavior(sample, categories)
    if attribution.misleading_read:
        bonus = -config.misleading_penalty
    elif attribution.gold_read:
        bonus = config.oracle_bonus
    elif not attribution.any_read and outcome >= config.pass_threshold:
        bonus = config.no_read_success_bonus
    else:
        bonus = 0.0
    return ScoredBehavior(
        outcome=float(outcome),
        bonus=float(bonus),
        shaped_score=float(outcome + bonus),
        attribution=attribution,
    )


def _validate_group(group: Sequence[Sample]) -> SkillCategories:
    if len(group) != EXPECTED_GROUP_SIZE:
        raise ValueError(
            f"mixed bonus compare requires {EXPECTED_GROUP_SIZE} samples per prompt, got {len(group)}"
        )
    first = _categories(group[0])
    for sample in group[1:]:
        if _categories(sample) != first:
            raise ValueError("mixed bonus compare group has inconsistent skill-category metadata")
    return first


def _annotate(sample: Sample, scored: ScoredBehavior) -> None:
    attr = scored.attribution
    values: dict[str, float] = {
        "mixed_skill_bonus_outcome": scored.outcome,
        "mixed_skill_bonus_value": scored.bonus,
        "mixed_skill_bonus_shaped_score": scored.shaped_score,
        "mixed_skill_bonus_any_read": float(attr.any_read),
        "mixed_skill_bonus_gold_read": float(attr.gold_read),
        "mixed_skill_bonus_misleading_read": float(attr.misleading_read),
        "mixed_skill_bonus_other_read": float(attr.other_read),
        "mixed_skill_bonus_no_read": float(not attr.any_read),
    }
    if isinstance(sample.reward, dict):
        sample.reward.update(values)
    sample.train_metadata = dict(sample.train_metadata or {})
    sample.train_metadata.update(values)


def _groups(
    samples: list[Sample] | list[list[Sample]],
    group_size: int,
) -> list[list[Sample]]:
    if samples and isinstance(samples[0], list):
        return [list(group) for group in samples]  # type: ignore[arg-type]
    flat = list(samples)  # type: ignore[arg-type]
    if len(flat) % group_size:
        raise ValueError(
            f"mixed bonus compare received {len(flat)} samples, not divisible by group size {group_size}"
        )
    return [flat[start : start + group_size] for start in range(0, len(flat), group_size)]


def _center(values: Sequence[float], *, use_std: bool) -> list[float]:
    center = mean(values)
    processed = [float(value - center) for value in values]
    if use_std and len(values) > 1:
        scale = float(stdev(values)) + 1e-6
        processed = [value / scale for value in processed]
    return processed


def keep_shaped_reward_nonzero_std(
    args: Any,
    samples: list[Sample],
    **_: Any,
) -> DynamicFilterOutput:
    """Keep groups whose final outcome+behavior scores contain GRPO signal."""

    config = _require_enabled()
    if any(sample.reward is None for sample in samples):
        return DynamicFilterOutput(keep=False, reason="missing_reward")
    categories = _validate_group(samples)
    scored = [_score_behavior(sample, args, config, categories) for sample in samples]
    for sample, item in zip(samples, scored, strict=False):
        _annotate(sample, item)
    values = [item.shaped_score for item in scored]
    keep = max(values) - min(values) > 1e-12
    return DynamicFilterOutput(
        keep=bool(keep),
        reason=None if keep else f"zero_std_mixed_bonus_{values[0]:.2f}",
    )


def keep_raw_task_reward_nonzero_std(
    args: Any,
    samples: list[Sample],
    **_: Any,
) -> DynamicFilterOutput:
    """Keep only groups with factual task-reward variance.

    Behavior bonuses deliberately do not participate in rollout admission.
    This preserves the standard dynamic-sampling task distribution: an
    all-fail or all-pass eight-sample task is refilled even when its samples
    made different skill-reading choices.
    """

    _require_enabled()
    if any(sample.reward is None for sample in samples):
        return DynamicFilterOutput(keep=False, reason="missing_reward")
    _validate_group(samples)
    outcomes = [_outcome_score(sample, args) for sample in samples]
    keep = max(outcomes) - min(outcomes) > 1e-12
    return DynamicFilterOutput(
        keep=bool(keep),
        reason=None if keep else f"zero_std_raw_task_{outcomes[0]:.2f}",
    )


def post_process_rewards(args: Any, samples: list[Sample] | list[list[Sample]]):
    """Return shaped raw scores and their ordinary group-normalized advantages."""

    config = _require_enabled()
    group_size = int(getattr(args, "n_samples_per_prompt", EXPECTED_GROUP_SIZE) or 0)
    if group_size != EXPECTED_GROUP_SIZE:
        raise ValueError(
            f"mixed bonus compare requires n_samples_per_prompt={EXPECTED_GROUP_SIZE}, got {group_size}"
        )

    all_raw: list[float] = []
    all_processed: list[float] = []
    for group in _groups(samples, group_size):
        categories = _validate_group(group)
        scored = [_score_behavior(sample, args, config, categories) for sample in group]
        for sample, item in zip(group, scored, strict=False):
            _annotate(sample, item)
        shaped_scores = [item.shaped_score for item in scored]
        all_raw.extend(shaped_scores)
        if bool(getattr(args, "rewards_normalization", True)):
            processed = _center(
                shaped_scores,
                use_std=bool(getattr(args, "grpo_std_normalization", True)),
            )
        else:
            processed = list(shaped_scores)
        all_processed.extend(processed)
    return all_raw, all_processed


__all__ = [
    "Attribution",
    "Config",
    "SkillCategories",
    "attribute_behavior",
    "config_from_env",
    "factual_outcome_score",
    "keep_raw_task_reward_nonzero_std",
    "keep_shaped_reward_nonzero_std",
    "post_process_rewards",
    "skill_categories",
    "strict_read_skill_names",
]
