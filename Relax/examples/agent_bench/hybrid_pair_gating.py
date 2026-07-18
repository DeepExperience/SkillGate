"""Pair-gated no-skill GRPO + oracle-prompt BC controls.

This module is for the prompt-only oracle-BC variant:

- Every task is represented by two prompt groups in the same rollout batch:
  ``no_skill_grpo`` and ``oracle_prompt_bc``.
- No-skill groups train with GRPO only when they are mixed success/fail.
- If the paired no-skill group is all-fail, the oracle group may train with
  weighted BC/AWR, but only successful oracle samples receive nonzero AWR
  weight in ``hybrid_shadow_grpo_loss``.
- If the no-skill group is all-pass, no BC is used for that task.

The dynamic filter keeps candidate pair groups. The actual gate is in reward
postprocess because it needs to see both arms of the same task in the training
batch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from examples.agent_bench.skill_group_reward import post_process_rewards as _base_post_process_rewards
from relax.engine.filters.base_types import DynamicFilterOutput
from relax.engine.filters.dynamic_sampling_filters import check_reward_nonzero_std
from relax.utils.types import Sample


_GRPO_UPDATE_KINDS = {"no_skill_grpo", "noskill_grpo", "no_skill", "noskill"}
_ORACLE_PROMPT_UPDATE_KINDS = {"oracle_prompt_bc", "oracle_direct_bc", "prompt_shadow"}


@dataclass
class _TaskGate:
    no_skill_rewards: list[float] = field(default_factory=list)
    oracle_rewards: list[float] = field(default_factory=list)

    @property
    def no_skill_has_success(self) -> bool:
        return any(reward >= _threshold() for reward in self.no_skill_rewards)

    @property
    def no_skill_has_failure(self) -> bool:
        return any(reward < _threshold() for reward in self.no_skill_rewards)

    @property
    def no_skill_all_fail(self) -> bool:
        return bool(self.no_skill_rewards) and not self.no_skill_has_success

    @property
    def no_skill_mixed(self) -> bool:
        return self.no_skill_has_success and self.no_skill_has_failure

    @property
    def no_skill_all_pass(self) -> bool:
        return bool(self.no_skill_rewards) and not self.no_skill_has_failure

    @property
    def oracle_has_success(self) -> bool:
        return any(reward >= _threshold() for reward in self.oracle_rewards)

    @property
    def bc_enabled(self) -> bool:
        return self.no_skill_all_fail and self.oracle_has_success


def _threshold() -> float:
    raw = os.environ.get("RELAX_PAIR_BC_PASS_THRESHOLD") or os.environ.get("PASS_REWARD_THRESHOLD") or "1.0"
    try:
        return float(raw)
    except ValueError:
        return 1.0


def _oracle_grpo_enabled() -> bool:
    """Accepted oracle groups train with in-group GRPO instead of BC."""
    return os.environ.get("RELAX_PAIR_ORACLE_GRPO", "0").lower() in {"1", "true", "yes", "on"}


def _oracle_grpo_cross_arm_adv_enabled() -> bool:
    """Use paired no-skill mean as the oracle-GRPO reward baseline."""

    return os.environ.get("RELAX_PAIR_ORACLE_GRPO_CROSS_ARM_ADV", "0").lower() in {"1", "true", "yes", "on"}


def _oracle_grpo_adv_clip() -> float | None:
    raw = os.environ.get("RELAX_PAIR_ORACLE_GRPO_CROSS_ARM_ADV_CLIP")
    if raw in (None, "", "0"):
        return None
    try:
        value = abs(float(raw))
    except ValueError:
        return None
    return value if value > 0 else None


def _sample_extra_info(sample: Sample) -> dict:
    metadata = sample.metadata or {}
    if not isinstance(metadata, dict):
        return {}
    extra = metadata.get("extra_info")
    if isinstance(extra, dict):
        return extra
    return metadata


def _set_extra(sample: Sample, key: str, value: Any) -> None:
    sample.metadata = sample.metadata or {}
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    extra = sample.metadata.get("extra_info")
    if not isinstance(extra, dict):
        extra = sample.metadata
    extra[key] = value
    if extra is not sample.metadata:
        sample.metadata["extra_info"] = extra


def _extra_float(extra: dict, key: str) -> float | None:
    if key not in extra:
        return None
    try:
        return float(extra[key])
    except (TypeError, ValueError):
        return None


def _clip_value(value: float, limit: float | None) -> float:
    if limit is None:
        return value
    return max(-limit, min(limit, value))


def _update_kind(sample: Sample) -> str:
    extra = _sample_extra_info(sample)
    return str(extra.get("update_kind") or extra.get("hybrid_update_kind") or "").strip().lower()


def _task_id(sample: Sample) -> str:
    extra = _sample_extra_info(sample)
    return str(extra.get("task_id") or sample.index or "unknown")


def _clean_success_score(sample: Sample, args: Any) -> float:
    reward = sample.reward
    if isinstance(reward, dict) and "raw_score" in reward:
        try:
            return float(reward["raw_score"])
        except (TypeError, ValueError):
            pass
    return float(sample.get_reward_value(args))


def keep_pair_candidate_groups(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """Keep paired no-skill/oracle groups so batch-level gating can compare them."""

    kinds = {_update_kind(sample) for sample in samples}
    if kinds and all(kind in _GRPO_UPDATE_KINDS or kind in _ORACLE_PROMPT_UPDATE_KINDS for kind in kinds):
        return DynamicFilterOutput(keep=True, reason=None)
    return check_reward_nonzero_std(args, samples, **kwargs)


def post_process_rewards(args: Any, samples: list[Sample] | list[list[Sample]]):
    """Apply base reward processing, then write pair-gated hybrid loss weights."""

    raw_rewards, rewards = _base_post_process_rewards(args, samples)

    flat_samples = samples
    if flat_samples and isinstance(flat_samples[0], list):
        flat_samples = [sample for group in flat_samples for sample in group]
    flat_samples = list(flat_samples)

    gates: dict[str, _TaskGate] = {}
    for sample in flat_samples:
        kind = _update_kind(sample)
        task_id = _task_id(sample)
        gate = gates.setdefault(task_id, _TaskGate())
        score = _clean_success_score(sample, args)
        if kind in _GRPO_UPDATE_KINDS:
            gate.no_skill_rewards.append(score)
        elif kind in _ORACLE_PROMPT_UPDATE_KINDS:
            gate.oracle_rewards.append(score)

    cross_arm_adv_enabled = _oracle_grpo_enabled() and _oracle_grpo_cross_arm_adv_enabled()
    cross_arm_adv_clip = _oracle_grpo_adv_clip()

    for idx, sample in enumerate(flat_samples):
        kind = _update_kind(sample)
        task_id = _task_id(sample)
        gate = gates.get(task_id, _TaskGate())
        extra = _sample_extra_info(sample)
        if kind in _GRPO_UPDATE_KINDS:
            explicit_grpo = _extra_float(extra, "hybrid_grpo_weight")
            grpo_weight = explicit_grpo if explicit_grpo is not None else (1.0 if gate.no_skill_mixed else 0.0)
            _set_extra(sample, "hybrid_is_shadow", 0.0)
            _set_extra(sample, "hybrid_grpo_weight", grpo_weight)
            _set_extra(sample, "hybrid_shadow_weight", 0.0)
            _set_extra(sample, "hybrid_pair_no_skill_all_fail", float(gate.no_skill_all_fail))
            _set_extra(sample, "hybrid_pair_no_skill_all_pass", float(gate.no_skill_all_pass))
            _set_extra(sample, "hybrid_pair_no_skill_mixed", float(gate.no_skill_mixed))
        elif kind in _ORACLE_PROMPT_UPDATE_KINDS:
            explicit_shadow = _extra_float(extra, "hybrid_shadow_weight")
            explicit_bc = _extra_float(extra, "hybrid_pair_bc_enabled")
            if explicit_shadow is not None:
                shadow_weight = explicit_shadow
            elif explicit_bc is not None:
                shadow_weight = 1.0 if explicit_bc > 0 else 0.0
            else:
                shadow_weight = 1.0 if gate.bc_enabled else 0.0
            explicit_no_skill_all_fail = _extra_float(extra, "hybrid_pair_no_skill_all_fail")
            explicit_oracle_has_success = _extra_float(extra, "hybrid_pair_oracle_has_success")
            if _oracle_grpo_enabled():
                # Oracle-GRPO60: the rollout accept path already marked the oracle group
                # for in-group GRPO (grpo_weight=1, shadow off). Respect the
                # explicit marks instead of forcing the BC weight layout. A
                # sample with no explicit marks falls back to the pair gate
                # under GRPO semantics (train only when gate fired).
                explicit_grpo = _extra_float(extra, "hybrid_grpo_weight")
                grpo_weight = explicit_grpo if explicit_grpo is not None else (1.0 if gate.bc_enabled else 0.0)
                _set_extra(sample, "hybrid_is_shadow", 0.0)
                _set_extra(sample, "hybrid_grpo_weight", grpo_weight)
                _set_extra(sample, "hybrid_shadow_weight", 0.0)
                bc_enabled = 0.0
            else:
                _set_extra(sample, "hybrid_is_shadow", 1.0)
                _set_extra(sample, "hybrid_grpo_weight", 0.0)
                _set_extra(sample, "hybrid_shadow_weight", shadow_weight)
                bc_enabled = float(shadow_weight > 0)
            _set_extra(sample, "hybrid_pair_bc_enabled", bc_enabled)
            _set_extra(
                sample,
                "hybrid_pair_no_skill_all_fail",
                explicit_no_skill_all_fail if explicit_no_skill_all_fail is not None else float(gate.no_skill_all_fail),
            )
            _set_extra(
                sample,
                "hybrid_pair_oracle_has_success",
                explicit_oracle_has_success if explicit_oracle_has_success is not None else float(gate.oracle_has_success),
            )
            if cross_arm_adv_enabled and _extra_float(extra, "hybrid_grpo_weight") not in (None, 0.0):
                no_skill_mean = _extra_float(extra, "relax_pair_no_skill_mean_reward")
                if no_skill_mean is not None:
                    raw_score = _clean_success_score(sample, args)
                    advantage = _clip_value(float(raw_score - no_skill_mean), cross_arm_adv_clip)
                    rewards[idx] = advantage
                    _set_extra(sample, "relax_pair_advantage_mode", "oracle_cross_arm_no_skill_mean")
                    _set_extra(sample, "relax_pair_cross_arm_advantage", float(advantage))
                    _set_extra(sample, "relax_pair_oracle_raw_score", float(raw_score))
                    _set_extra(sample, "relax_pair_oracle_adv_clip", float(cross_arm_adv_clip or 0.0))
                else:
                    _set_extra(sample, "relax_pair_advantage_mode", "oracle_cross_arm_missing_no_skill_mean")

    return raw_rewards, rewards
