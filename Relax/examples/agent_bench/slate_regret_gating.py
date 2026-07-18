"""Slate-regret advantage shaping (SlateRL) reward post-process.

Wraps ``hybrid_pair_gating.post_process_rewards`` and, ONLY when
``RELAX_SLATE_REGRET_GRPO=1``, adds a paired-regret shift to the
post-centered advantages of ``slate_grpo`` groups:

    delta = clip(mean(slate group raw scores) - paired no-skill group mean, -1, 1)
    processed[i] += coef * delta        for every sample i of the slate group

where the paired no-skill mean was stamped on the slate group by the
pair-atomic rollout path (``relax_pair_no_skill_mean_reward``), and
``coef`` is ``RELAX_SLATE_REGRET_COEF`` when the slate contains the gold
skill (``slate_contains_gold=1`` from the parquet) else
``RELAX_SLATE_REGRET_COEF_NOGOLD``.

Semantics (design section 4.2 of the original skill-reliability proposal;
that internal document is not shipped with this repository):
  gold present:  delta<0 = dragged down by slate (penalize), delta>0 = used it well (reward)
  gold absent:   delta<0 = misled (penalize), delta~0 = correctly ignored (no shift)

The shift is applied AFTER group mean-centering (a uniform pre-centering
shift would be nulled by the mean subtraction), mirroring the
``subgroup_adv_coef`` precedent in ``skill_group_reward.post_process_rewards``.

When ``RELAX_SLATE_REGRET_GRPO`` is unset/0 this module returns the wrapped
result verbatim - byte-identical behavior for every existing run. Existing
launchers keep ``CUSTOM_REWARD_POST_PROCESS_PATH=examples.agent_bench.
hybrid_pair_gating.post_process_rewards``; only the slate launcher points at
``examples.agent_bench.slate_regret_gating.post_process_rewards``.
"""

from __future__ import annotations

import os
from typing import Any

from examples.agent_bench.hybrid_pair_gating import (
    _clean_success_score,
    _sample_extra_info,
    post_process_rewards as _pair_post_process_rewards,
)
from relax.utils.types import Sample

_SLATE_UPDATE_KINDS = {"slate_grpo"}


def _enabled() -> bool:
    return os.environ.get("RELAX_SLATE_REGRET_GRPO", "0").lower() in {"1", "true", "yes", "on"}


def _coef_gold() -> float:
    try:
        return float(os.environ.get("RELAX_SLATE_REGRET_COEF", "0.5"))
    except ValueError:
        return 0.5


def _coef_nogold() -> float:
    raw = os.environ.get("RELAX_SLATE_REGRET_COEF_NOGOLD")
    if raw is None or raw == "":
        return _coef_gold()
    try:
        return float(raw)
    except ValueError:
        return _coef_gold()


def post_process_rewards(args: Any, samples: list[Sample] | list[list[Sample]]):
    """Delegate to pair gating, then apply the slate regret shift (env-gated)."""

    raw_rewards, rewards = _pair_post_process_rewards(args, samples)
    if not _enabled():
        return raw_rewards, rewards

    flat_samples = samples
    if flat_samples and isinstance(flat_samples[0], list):
        flat_samples = [sample for group in flat_samples for sample in group]
    flat_samples = list(flat_samples)

    group_size = int(getattr(args, "n_samples_per_prompt", 1) or 1)
    if group_size <= 0:
        group_size = 1
    rewards = list(rewards)

    for start in range(0, len(flat_samples), group_size):
        group = flat_samples[start : start + group_size]
        if not group:
            continue
        extra0 = _sample_extra_info(group[0])
        kind = str(extra0.get("update_kind") or extra0.get("hybrid_update_kind") or "").strip().lower()
        if kind not in _SLATE_UPDATE_KINDS:
            continue
        noskill_mean_raw = extra0.get("relax_pair_no_skill_mean_reward")
        if noskill_mean_raw is None:
            # Unpaired slate group (should not happen in pair-atomic mode);
            # leave its plain GRPO advantages unshifted.
            continue
        try:
            noskill_mean = float(noskill_mean_raw)
        except (TypeError, ValueError):
            continue
        scores = [_clean_success_score(sample, args) for sample in group]
        delta = sum(scores) / len(scores) - noskill_mean
        delta = max(-1.0, min(1.0, delta))
        has_gold = False
        try:
            has_gold = float(extra0.get("slate_contains_gold") or 0.0) > 0.0
        except (TypeError, ValueError):
            pass
        coef = _coef_gold() if has_gold else _coef_nogold()
        shift = coef * delta
        for offset, sample in enumerate(group):
            idx = start + offset
            if idx >= len(rewards):
                break
            rewards[idx] = float(rewards[idx]) + shift
            # Audit trail for train dumps (mirrors skill_group_subgroup_adv).
            sample.train_metadata = dict(sample.train_metadata or {})
            sample.train_metadata["slate_regret_delta"] = float(delta)
            sample.train_metadata["slate_regret_shift"] = float(shift)
            sample.train_metadata["slate_regret_has_gold"] = float(has_gold)
            if isinstance(sample.reward, dict):
                sample.reward["slate_regret_delta"] = float(delta)
                sample.reward["slate_regret_shift"] = float(shift)

    return raw_rewards, rewards
