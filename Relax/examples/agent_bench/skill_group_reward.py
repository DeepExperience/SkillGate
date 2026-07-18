"""Skill-choice reward shaping for agent_bench GRPO runs.

This module implements an env-gated, reversible variant inspired by GiGPO's
group-in-group credit assignment: within each GRPO prompt group, trajectories
are split by whether the agent actually read a retrieved skill file. The better
subgroup can receive a small extra reward, while optional subgroup-relative
advantage keeps signal inside the preferred subgroup.

The primary task metrics remain anchored on ``raw_score``. This module only
changes the training reward path (``reward[args.reward_key]`` / processed
``rewards``), so pass@k and task_score_mean can continue to report factual task
performance through ``clean_pass_score``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any, Iterable

from relax.utils.types import Sample


SKILL_FILE_RE = re.compile(
    r"/(?:root/)?\.claude/skills/[^\s'\"<>|;&]+/(?:SKILL|README)\.md"
)
EXEC_READ_RE = re.compile(
    r"\b(cat|sed|awk|grep|head|tail|less|more|python3?|perl|ruby|node)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SkillGroupConfig:
    enabled: bool
    bonus_coef: float
    bonus_max: float
    margin: float
    subgroup_adv_coef: float
    require_both: bool
    no_read_success_bonus: float
    no_read_success_threshold: float


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return float(raw)


def config_from_env() -> SkillGroupConfig:
    success_threshold = _env_float(
        "RELAX_SKILL_GROUP_NO_READ_SUCCESS_THRESHOLD",
        _env_float("PASS_REWARD_THRESHOLD", 1.0),
    )
    return SkillGroupConfig(
        enabled=_env_bool("RELAX_SKILL_GROUP_REWARD", False),
        bonus_coef=_env_float("RELAX_SKILL_GROUP_BONUS_COEF", 0.1),
        bonus_max=_env_float("RELAX_SKILL_GROUP_BONUS_MAX", 0.2),
        margin=_env_float("RELAX_SKILL_GROUP_MARGIN", 0.0),
        subgroup_adv_coef=_env_float("RELAX_SKILL_GROUP_SUBGROUP_ADV_COEF", 0.0),
        require_both=_env_bool("RELAX_SKILL_GROUP_REQUIRE_BOTH", True),
        no_read_success_bonus=_env_float("RELAX_SKILL_GROUP_NO_READ_SUCCESS_BONUS", 0.0),
        no_read_success_threshold=success_threshold,
    )


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _tool_call_reads_skill(name: str, arguments: Any) -> bool:
    text = _stringify(arguments)
    if not SKILL_FILE_RE.search(text):
        return False
    if name == "read":
        return True
    if name == "exec":
        return bool(EXEC_READ_RE.search(text))
    return False


def _extract_tool_calls_from_text(text: str) -> Iterable[tuple[str, str]]:
    """Best-effort XML tool-call parser for Qwen/OpenClaw-style text.

    The live environment also stores assistant raw text in rollout traces. We
    intentionally keep this parser conservative: it only counts explicit
    SKILL.md/README.md reads, not mere skill-name mentions.
    """

    if not text or "/.claude/skills/" not in text and "/root/.claude/skills/" not in text:
        return []
    calls: list[tuple[str, str]] = []
    for match in re.finditer(r"<function=([^>]+)>(.*?)</function>", text, flags=re.DOTALL):
        calls.append((match.group(1).strip(), match.group(2)))
    return calls


def used_skill_strict(sample: Sample) -> bool:
    """Return True iff the trajectory actually reads a retrieved skill file."""

    metadata = sample.metadata or {}
    if metadata.get("used_skill_strict") is True:
        return True
    if metadata.get("skill_group_used_strict") is True:
        return True

    # Fallback for archived/older traces: parse raw assistant text.
    for trace in metadata.get("rollout_traces") or []:
        response_text = ((trace or {}).get("inference") or {}).get("response_text") or ""
        for name, args_text in _extract_tool_calls_from_text(response_text):
            if _tool_call_reads_skill(name, args_text):
                return True

    # Last-resort fallback: full response text.
    for name, args_text in _extract_tool_calls_from_text(sample.response or ""):
        if _tool_call_reads_skill(name, args_text):
            return True
    return False


def _reward_dict(sample: Sample) -> dict[str, Any] | None:
    return sample.reward if isinstance(sample.reward, dict) else None


def _clean_score(sample: Sample, args: Any) -> float:
    reward = _reward_dict(sample)
    if reward is not None and "raw_score" in reward:
        try:
            return float(reward["raw_score"])
        except (TypeError, ValueError):
            pass
    try:
        return float(sample.get_reward_value(args))
    except (TypeError, ValueError):
        return 0.0


def _train_score(sample: Sample, args: Any) -> float:
    try:
        return float(sample.get_reward_value(args))
    except (TypeError, ValueError):
        return 0.0


def _set_train_score(sample: Sample, args: Any, value: float) -> None:
    reward_key = getattr(args, "reward_key", None)
    if isinstance(sample.reward, dict) and reward_key:
        sample.reward[reward_key] = float(value)
    else:
        sample.reward = float(value)


def _groups(samples: list[Sample], group_size: int) -> list[list[Sample]]:
    if group_size <= 0:
        return [samples]
    return [samples[i : i + group_size] for i in range(0, len(samples), group_size)]


def _center(values: list[float], *, use_std: bool) -> list[float]:
    if not values:
        return []
    m = mean(values)
    centered = [float(v - m) for v in values]
    if use_std and len(values) > 1:
        std = float(stdev(values))
        centered = [v / (std + 1e-6) for v in centered]
    return centered


def _subgroup_centered_values(values: list[float], used_flags: list[bool]) -> list[float]:
    """Return per-sample centered values inside each read/no-read subgroup."""

    centered_by_idx = [0.0 for _ in values]
    for target_used in (True, False):
        idxs = [idx for idx, used in enumerate(used_flags) if used is target_used]
        if len(idxs) < 2:
            continue
        centered = _center([values[idx] for idx in idxs], use_std=False)
        for idx, value in zip(idxs, centered, strict=False):
            centered_by_idx[idx] = float(value)
    return centered_by_idx


def apply_skill_group_reward(args: Any, group: list[Sample]) -> dict[str, float]:
    """Mutate rewards for one prompt group and return summary metrics."""

    cfg = config_from_env()
    if not cfg.enabled or not group:
        return {}

    # Idempotency: this function can be called both from rollout and from custom
    # postprocess during tests. Avoid double-applying the same bonus.
    if all((_reward_dict(sample) or {}).get("_skill_group_reward_applied") for sample in group):
        return {}

    used_flags = [used_skill_strict(sample) for sample in group]
    clean_scores = [_clean_score(sample, args) for sample in group]
    base_scores = [_train_score(sample, args) for sample in group]
    read_scores = [score for score, used in zip(clean_scores, used_flags, strict=False) if used]
    no_read_scores = [score for score, used in zip(clean_scores, used_flags, strict=False) if not used]

    has_both = bool(read_scores and no_read_scores)
    mean_read = float(mean(read_scores)) if read_scores else 0.0
    mean_no_read = float(mean(no_read_scores)) if no_read_scores else 0.0
    gap = mean_read - mean_no_read if has_both else 0.0

    preferred: str = "none"
    if has_both:
        if gap > cfg.margin:
            preferred = "read"
        elif -gap > cfg.margin:
            preferred = "no_read"
        else:
            preferred = "tie"
    elif not cfg.require_both:
        preferred = "read" if read_scores else "no_read"

    bonus_mag = 0.0
    if preferred in {"read", "no_read"}:
        bonus_mag = min(cfg.bonus_max, cfg.bonus_coef * abs(gap))

    subgroup_adv_values = _subgroup_centered_values(base_scores, used_flags)

    bonuses: list[float] = []
    preference_bonuses: list[float] = []
    no_read_success_bonuses: list[float] = []
    no_read_success_count = 0
    for idx, (sample, used, base_score, clean_score) in enumerate(
        zip(group, used_flags, base_scores, clean_scores, strict=False)
    ):
        is_preferred = (preferred == "read" and used) or (preferred == "no_read" and not used)
        subgroup_adv = subgroup_adv_values[idx]
        preference_bonus = bonus_mag if is_preferred else 0.0
        no_read_success_bonus = 0.0
        if (not used) and cfg.no_read_success_bonus and clean_score >= cfg.no_read_success_threshold:
            no_read_success_bonus = cfg.no_read_success_bonus
            no_read_success_count += 1
        bonus = preference_bonus + no_read_success_bonus
        shaped_score = base_score + bonus
        _set_train_score(sample, args, shaped_score)

        reward = _reward_dict(sample)
        if reward is not None:
            reward["_skill_group_reward_applied"] = True
            reward["skill_group_used_strict"] = float(1.0 if used else 0.0)
            reward["skill_group_has_both"] = float(1.0 if has_both else 0.0)
            reward["skill_group_preferred_read"] = float(1.0 if preferred == "read" else 0.0)
            reward["skill_group_preferred_no_read"] = float(1.0 if preferred == "no_read" else 0.0)
            reward["skill_group_tie_or_none"] = float(1.0 if preferred not in {"read", "no_read"} else 0.0)
            # Only write subgroup diagnostics when the subgroup exists: a 0.0
            # placeholder for empty subgroups pollutes the cross-sample W&B
            # mean (e.g. all-read groups dragged mean_no_read down, inflating
            # the apparent read-vs-no-read difference while gap stayed ~0).
            if read_scores:
                reward["skill_group_mean_read"] = mean_read
            if no_read_scores:
                reward["skill_group_mean_no_read"] = mean_no_read
            if has_both:
                reward["skill_group_gap"] = gap
            reward["skill_group_preference_bonus"] = preference_bonus
            reward["skill_group_no_read_success_bonus"] = no_read_success_bonus
            reward["skill_group_bonus"] = bonus
            reward["skill_group_subgroup_adv"] = subgroup_adv

        sample.metadata = sample.metadata or {}
        sample.metadata["used_skill_strict"] = bool(used)
        sample.metadata["skill_group_preferred"] = preferred
        sample.metadata["skill_group_bonus"] = bonus
        sample.metadata["skill_group_no_read_success_bonus"] = no_read_success_bonus
        sample.train_metadata = dict(sample.train_metadata or {})
        sample.train_metadata.update(
            {
                "used_skill_strict": bool(used),
                "skill_group_preferred": preferred,
                "skill_group_bonus": bonus,
                "skill_group_no_read_success_bonus": no_read_success_bonus,
                "skill_group_subgroup_adv": subgroup_adv,
            }
        )
        bonuses.append(bonus)
        preference_bonuses.append(preference_bonus)
        no_read_success_bonuses.append(no_read_success_bonus)

    return {
        "read_rate": float(sum(used_flags) / len(used_flags)),
        "has_both": float(1.0 if has_both else 0.0),
        "preferred_read": float(1.0 if preferred == "read" else 0.0),
        "preferred_no_read": float(1.0 if preferred == "no_read" else 0.0),
        "bonus_mean": float(mean(bonuses)) if bonuses else 0.0,
        "preference_bonus_mean": float(mean(preference_bonuses)) if preference_bonuses else 0.0,
        "no_read_success_bonus_mean": (
            float(mean(no_read_success_bonuses)) if no_read_success_bonuses else 0.0
        ),
        "no_read_success_count": float(no_read_success_count),
        "gap": gap,
    }


def post_process_rewards(args: Any, samples: list[Sample] | list[list[Sample]]):
    """Custom Relax reward postprocess with optional subgroup advantage.

    Signature matches ``--custom-reward-post-process-path``. It intentionally
    mirrors Relax's default GRPO normalization after applying skill-group reward.
    """

    flat_samples = samples
    if flat_samples and isinstance(flat_samples[0], list):
        flat_samples = [sample for group in flat_samples for sample in group]

    group_size = int(getattr(args, "n_samples_per_prompt", 1) or 1)
    all_processed: list[float] = []
    raw_rewards: list[float] = []
    subgroup_adv_coef = config_from_env().subgroup_adv_coef

    for group in _groups(list(flat_samples), group_size):
        if len(group) != group_size:
            # Keep partial groups usable in debug paths.
            group_size_for_norm = max(len(group), 1)
        else:
            group_size_for_norm = group_size
        apply_skill_group_reward(args, group)
        train_scores = [_train_score(sample, args) for sample in group]
        raw_rewards.extend(train_scores)

        if getattr(args, "rewards_normalization", True):
            processed = _center(
                train_scores,
                use_std=bool(getattr(args, "grpo_std_normalization", True)),
            )
        else:
            processed = list(train_scores)

        if subgroup_adv_coef:
            used_flags = [used_skill_strict(sample) for sample in group]
            subgroup_adv_values = _subgroup_centered_values(train_scores, used_flags)
            for idx, subgroup_adv in enumerate(subgroup_adv_values):
                processed[idx] += subgroup_adv_coef * subgroup_adv
                reward = _reward_dict(group[idx])
                if reward is not None:
                    reward["skill_group_subgroup_adv"] = subgroup_adv
                group[idx].train_metadata = dict(group[idx].train_metadata or {})
                group[idx].train_metadata["skill_group_subgroup_adv"] = subgroup_adv

        # Preserve exact length in case debug paths provided a partial group.
        all_processed.extend(processed[:group_size_for_norm])

    return raw_rewards, all_processed
