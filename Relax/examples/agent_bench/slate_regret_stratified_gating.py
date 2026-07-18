"""Gold-only SlateRL regret with behavior-stratified advantage.

This is an opt-in wrapper around :mod:`slate_regret_gating`.  It preserves
the original outcome GRPO and paired-regret shift, then adds a zero-mean
between-stratum advantage inside each ``slate_grpo`` prompt group.

The mutually exclusive behavior strata are intentionally conservative:

``misleading``
    The assistant strictly read at least one misleading SKILL.md.  This takes
    precedence over oracle, so reading both is not credited as a clean oracle
    choice.
``oracle``
    The assistant strictly read the oracle and no misleading SKILL.md.
``no_read``
    The assistant did not strictly read any skill file.
``other``
    It read only relevant/irrelevant/unadvertised skills.  This diagnostic fallback
    keeps the ordinary outcome/regret advantage and is not behavior-shaped.

For the three target strata S = {no_read, oracle, misleading}, let E be the
samples assigned to an observed target stratum, r_i the verifier score, and
mu_E their mean.  For each observed stratum s:

    mu_s_shrunk = (n_s * mu_s + tau * mu_E) / (n_s + tau)
    b_s          = mu_s_shrunk - mu_E

The b_s values are sample-count weighted back to zero mean and jointly scaled
to ``RELAX_SLATE_STRATIFIED_ADV_CLIP``.  The final training advantage is:

    A_i = A_regret_i + RELAX_SLATE_STRATIFIED_ADV_COEF * b_{stratum(i)}

No class gets an unconditional reward.  A stratum only wins when its observed
task outcome is better in the same prompt group.  Shrinkage makes singleton
strata less noisy; zero-meaning prevents this term from becoming another
group-level reward.  If fewer than two target strata are present, b_s = 0.

The feature is gated by ``RELAX_SLATE_STRATIFIED_ADVANTAGE=1`` and additionally
requires ``slate_contains_gold=1`` plus explicit oracle/misleading name lists
from the dedicated gold-only v2 parquet.  Old launchers keep using the original
module and are therefore unaffected.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable

from examples.agent_bench.hybrid_pair_gating import (
    _clean_success_score,
    _sample_extra_info,
)
from examples.agent_bench.skill_group_reward import (
    _extract_tool_calls_from_text,
    _tool_call_reads_skill,
)
from examples.agent_bench.slate_regret_gating import (
    post_process_rewards as _regret_post_process_rewards,
)
from relax.utils.types import Sample


_SLATE_UPDATE_KINDS = {"slate_grpo"}
_TARGET_STRATA = ("no_read", "oracle", "misleading")
_SKILL_NAME_RE = re.compile(
    r"/(?:root/)?\.claude/skills/([^/\s'\"<>|;&]+)/(?:SKILL|README)\.md"
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _enabled() -> bool:
    return _env_bool("RELAX_SLATE_STRATIFIED_ADVANTAGE", False)


def _coef() -> float:
    return _env_float("RELAX_SLATE_STRATIFIED_ADV_COEF", 1.0)


def _shrinkage() -> float:
    return max(0.0, _env_float("RELAX_SLATE_STRATIFIED_SHRINKAGE", 1.0))


def _clip() -> float:
    return max(0.0, _env_float("RELAX_SLATE_STRATIFIED_ADV_CLIP", 0.5))


def _as_names(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value} if value else set()
    try:
        return {str(item) for item in value if str(item)}
    except TypeError:
        return {str(value)} if str(value) else set()


def _assistant_texts(sample: Sample) -> Iterable[str]:
    """Yield assistant inference text without counting tool-response echoes."""

    metadata = sample.metadata or {}
    if isinstance(metadata, dict):
        traces = metadata.get("rollout_traces") or []
        for trace in traces:
            text = ((trace or {}).get("inference") or {}).get("response_text") or ""
            if text:
                yield str(text)
        # ``sample.response`` is the only source retained by older/debug paths.
        # Do not parse it twice when rollout traces supplied assistant turns.
        if any(
            ((trace or {}).get("inference") or {}).get("response_text")
            for trace in traces
        ):
            return
    if sample.response:
        yield str(sample.response)


def strict_read_skill_names(sample: Sample) -> set[str]:
    """Return skill directory names from explicit assistant read/exec calls."""

    names: set[str] = set()
    metadata = sample.metadata or {}
    if isinstance(metadata, dict):
        # Optional direct attribution from newer rollout paths.  These fields
        # are agent-initiated names, never tool-response echo attribution.
        for key in ("read_skill_names_agent", "skill_read_names_agent"):
            names.update(_as_names(metadata.get(key)))

    for text in _assistant_texts(sample):
        for tool_name, arguments in _extract_tool_calls_from_text(text):
            if not _tool_call_reads_skill(tool_name, arguments):
                continue
            names.update(_SKILL_NAME_RE.findall(str(arguments)))
    return names


def behavior_stratum(sample: Sample) -> tuple[str, set[str]]:
    """Classify one trajectory using names stamped by the v2 parquet."""

    extra = _sample_extra_info(sample)
    advertised = _as_names(extra.get("retrieval_skills_top_n"))
    oracle_names = _as_names(extra.get("slate_oracle_names"))
    if not oracle_names:
        oracle_names = _as_names(extra.get("slate_gold_name"))
    misleading_names = _as_names(extra.get("slate_misleading_names"))
    read_names = strict_read_skill_names(sample)
    advertised_reads = read_names & advertised

    if advertised_reads & misleading_names:
        return "misleading", read_names
    if advertised_reads & oracle_names:
        return "oracle", read_names
    if not read_names:
        return "no_read", read_names
    return "other", read_names


def _stratum_advantages(
    scores: list[float], strata: list[str], *, shrinkage: float, clip: float
) -> tuple[list[float], dict[str, float], float]:
    """Compute zero-mean, clipped between-stratum advantages."""

    target_indices = [idx for idx, stratum in enumerate(strata) if stratum in _TARGET_STRATA]
    observed = {strata[idx] for idx in target_indices}
    if len(observed) < 2:
        return [0.0] * len(scores), {}, mean([scores[idx] for idx in target_indices]) if target_indices else 0.0

    eligible_mean = mean(scores[idx] for idx in target_indices)
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for idx in target_indices:
        by_stratum[strata[idx]].append(idx)

    raw_by_stratum: dict[str, float] = {}
    for stratum, indices in by_stratum.items():
        n = len(indices)
        stratum_mean = mean(scores[idx] for idx in indices)
        shrunk_mean = (n * stratum_mean + shrinkage * eligible_mean) / (n + shrinkage)
        raw_by_stratum[stratum] = float(shrunk_mean - eligible_mean)

    weighted_mean = sum(
        len(by_stratum[stratum]) * value for stratum, value in raw_by_stratum.items()
    ) / len(target_indices)
    centered = {stratum: value - weighted_mean for stratum, value in raw_by_stratum.items()}

    max_abs = max((abs(value) for value in centered.values()), default=0.0)
    scale = 1.0 if clip <= 0.0 or max_abs <= clip else clip / max_abs
    shaped_by_stratum = {stratum: float(value * scale) for stratum, value in centered.items()}
    per_sample = [
        shaped_by_stratum.get(stratum, 0.0) if stratum in _TARGET_STRATA else 0.0
        for stratum in strata
    ]
    return per_sample, shaped_by_stratum, float(eligible_mean)


def _require_gold_metadata(group: list[Sample]) -> None:
    extra = _sample_extra_info(group[0])
    try:
        has_gold = float(extra.get("slate_contains_gold") or 0.0) == 1.0
    except (TypeError, ValueError):
        has_gold = False
    oracle_names = _as_names(extra.get("slate_oracle_names"))
    misleading_names = _as_names(extra.get("slate_misleading_names"))
    advertised = _as_names(extra.get("retrieval_skills_top_n"))
    if not has_gold or len(oracle_names) != 1 or len(misleading_names) != 5:
        raise RuntimeError(
            "gold-stratified SlateRL requires slate_contains_gold=1, exactly one "
            "slate_oracle_names entry, and exactly five slate_misleading_names entries"
        )
    if not oracle_names.issubset(advertised) or not misleading_names.issubset(advertised):
        raise RuntimeError("gold-stratified SlateRL category names are missing from advertised slate")


def post_process_rewards(args: Any, samples: list[Sample] | list[list[Sample]]):
    """Apply original slate regret, then opt-in behavior-stratified advantage."""

    raw_rewards, rewards = _regret_post_process_rewards(args, samples)
    if not _enabled():
        return raw_rewards, rewards

    flat_samples = samples
    if flat_samples and isinstance(flat_samples[0], list):
        flat_samples = [sample for group in flat_samples for sample in group]
    flat_samples = list(flat_samples)
    rewards = list(rewards)
    group_size = max(1, int(getattr(args, "n_samples_per_prompt", 1) or 1))

    for start in range(0, len(flat_samples), group_size):
        group = flat_samples[start : start + group_size]
        if not group:
            continue
        extra0 = _sample_extra_info(group[0])
        kind = str(extra0.get("update_kind") or extra0.get("hybrid_update_kind") or "").strip().lower()
        if kind not in _SLATE_UPDATE_KINDS:
            continue
        _require_gold_metadata(group)

        scores = [_clean_success_score(sample, args) for sample in group]
        classified = [behavior_stratum(sample) for sample in group]
        strata = [item[0] for item in classified]
        read_names = [item[1] for item in classified]
        behavior_adv, means, eligible_mean = _stratum_advantages(
            scores,
            strata,
            shrinkage=_shrinkage(),
            clip=_clip(),
        )
        counts = {stratum: strata.count(stratum) for stratum in (*_TARGET_STRATA, "other")}
        score_means = {
            stratum: float(mean(score for score, observed in zip(scores, strata) if observed == stratum))
            for stratum in set(strata)
        }

        for offset, sample in enumerate(group):
            idx = start + offset
            if idx >= len(rewards):
                break
            addition = _coef() * behavior_adv[offset]
            rewards[idx] = float(rewards[idx]) + addition
            stratum = strata[offset]
            sample.train_metadata = dict(sample.train_metadata or {})
            sample.train_metadata.update(
                {
                    "slate_stratified_stratum": stratum,
                    "slate_stratified_read_names": sorted(read_names[offset]),
                    "slate_stratified_eligible_mean": float(eligible_mean),
                    "slate_stratified_stratum_score_mean": float(score_means[stratum]),
                    "slate_stratified_stratum_behavior_adv": float(means.get(stratum, 0.0)),
                    "slate_stratified_behavior_adv": float(behavior_adv[offset]),
                    "slate_stratified_adv_addition": float(addition),
                    "slate_stratified_count_no_read": float(counts["no_read"]),
                    "slate_stratified_count_oracle": float(counts["oracle"]),
                    "slate_stratified_count_misleading": float(counts["misleading"]),
                    "slate_stratified_count_other": float(counts["other"]),
                }
            )
            if isinstance(sample.reward, dict):
                sample.reward["slate_stratified_is_no_read"] = float(stratum == "no_read")
                sample.reward["slate_stratified_is_oracle"] = float(stratum == "oracle")
                sample.reward["slate_stratified_is_misleading"] = float(stratum == "misleading")
                sample.reward["slate_stratified_is_other"] = float(stratum == "other")
                sample.reward["slate_stratified_behavior_adv"] = float(behavior_adv[offset])
                sample.reward["slate_stratified_adv_addition"] = float(addition)

    return raw_rewards, rewards
