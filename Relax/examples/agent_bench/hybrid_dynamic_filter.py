"""Dynamic sampling filter for M1 hybrid tier B.

Oracle-shadow groups are auxiliary BC/AWR data, so zero reward std alone is not
a reason to discard them: all-success shadow groups are valuable imitation data.
All-fail shadow groups have zero BC/AWR weight and would only occupy batch
capacity, so they are refilled. Real no-skill GRPO groups still use the
standard nonzero-std filter.
"""

from __future__ import annotations

from relax.engine.filters.dynamic_sampling_filters import check_reward_nonzero_std
from relax.engine.filters.base_types import DynamicFilterOutput
from relax.utils.types import Sample

_SHADOW_UPDATE_KINDS = {"oracle_shadow", "shadow", "m1_shadow", "hybrid_shadow"}


def _sample_extra_info(sample: Sample) -> dict:
    metadata = sample.metadata or {}
    if not isinstance(metadata, dict):
        return {}
    extra = metadata.get("extra_info")
    if isinstance(extra, dict):
        return extra
    return metadata


def _update_kind(sample: Sample) -> str:
    extra = _sample_extra_info(sample)
    return str(extra.get("update_kind") or extra.get("hybrid_update_kind") or "").strip().lower()


def keep_shadow_or_nonzero_std(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """Keep useful shadow groups; apply std filtering to GRPO groups."""

    kinds = {_update_kind(sample) for sample in samples}
    if kinds and all(kind in _SHADOW_UPDATE_KINDS for kind in kinds):
        rewards = []
        for sample in samples:
            if sample.reward is None:
                return DynamicFilterOutput(keep=False, reason="shadow_missing_reward")
            rewards.append(float(sample.get_reward_value(args)))

        threshold = float(getattr(args, "pass_reward_threshold", 1.0) or 1.0)
        keep = any(reward >= threshold for reward in rewards)
        return DynamicFilterOutput(
            keep=keep,
            reason=None if keep else f"shadow_all_fail_lt_{threshold:g}",
        )
    return check_reward_nonzero_std(args, samples, **kwargs)
