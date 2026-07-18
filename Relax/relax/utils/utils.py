# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import socket
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import ray
import torch
from tensordict import TensorDict

from relax.utils.logging_utils import get_logger
from relax.utils.misc import load_function
from relax.utils.types import Sample


logger = get_logger(__name__)
CURRENT_ROLLOUT_BATCH = []
_HYBRID_SHADOW_UPDATE_KINDS = {
    "oracle_shadow",
    "shadow",
    "m1_shadow",
    "hybrid_shadow",
    "oracle_prompt_bc",
    "oracle_direct_bc",
    "prompt_shadow",
}
_HYBRID_GRPO_UPDATE_KINDS = {"no_skill_grpo", "noskill_grpo", "no_skill", "noskill"}
_ACTION_MASK_TOKENIZER = None
_ACTION_MASK_TOKENIZER_PATH = None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float env %s=%r; using default %s", name, raw, default)
        return default


def _sample_extra_info(sample: Sample) -> dict:
    metadata = sample.metadata or {}
    if not isinstance(metadata, dict):
        return {}
    extra = metadata.get("extra_info")
    if isinstance(extra, dict):
        return extra
    return metadata


def _mutable_sample_extra_info(sample: Sample) -> dict:
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    extra = sample.metadata.get("extra_info")
    if isinstance(extra, dict):
        return extra
    return sample.metadata


def _sample_update_kind(sample: Sample) -> str:
    extra = _sample_extra_info(sample)
    return str(extra.get("update_kind") or extra.get("hybrid_update_kind") or "").strip().lower()


def _metadata_float(extra: dict, key: str) -> float | None:
    if key not in extra or extra.get(key) is None:
        return None
    try:
        return float(extra[key])
    except (TypeError, ValueError):
        return None


def _hybrid_is_shadow_sample(sample: Sample) -> bool:
    # Explicit mark from the rollout accept path / pair gating wins: oracle-GRPO
    # oracle-GRPO groups keep update_kind=oracle_prompt_bc for provenance but
    # are explicitly marked hybrid_is_shadow=0. Existing BC paths set the
    # explicit mark to the same value the kind would derive, so this is
    # behavior-preserving for them.
    explicit = _metadata_float(_sample_extra_info(sample), "hybrid_is_shadow")
    if explicit is not None:
        return explicit > 0
    update_kind = _sample_update_kind(sample)
    if update_kind:
        return update_kind in _HYBRID_SHADOW_UPDATE_KINDS
    metadata = sample.metadata or {}
    return bool(isinstance(metadata, dict) and metadata.get("m1_cleaned"))


def _hybrid_grpo_weight(sample: Sample, is_shadow: bool) -> float:
    extra = _sample_extra_info(sample)
    explicit = _metadata_float(extra, "hybrid_grpo_weight")
    if explicit is not None:
        return explicit
    update_kind = _sample_update_kind(sample)
    if update_kind in _HYBRID_GRPO_UPDATE_KINDS:
        return 1.0
    return 0.0 if is_shadow else 1.0


def _hybrid_shadow_weight(sample: Sample, is_shadow: bool) -> float:
    extra = _sample_extra_info(sample)
    explicit = _metadata_float(extra, "hybrid_shadow_weight")
    if explicit is not None:
        return explicit
    return 1.0 if is_shadow else 0.0


def _get_action_mask_tokenizer(args: Any):
    global _ACTION_MASK_TOKENIZER, _ACTION_MASK_TOKENIZER_PATH
    checkpoint = getattr(args, "hf_checkpoint", None)
    if not checkpoint:
        raise ValueError("action-span mask generation requires args.hf_checkpoint for tokenizer alignment")
    if _ACTION_MASK_TOKENIZER is not None and _ACTION_MASK_TOKENIZER_PATH == checkpoint:
        return _ACTION_MASK_TOKENIZER

    from relax.utils.data.processing_utils import load_tokenizer

    _ACTION_MASK_TOKENIZER = load_tokenizer(checkpoint, trust_remote_code=True)
    _ACTION_MASK_TOKENIZER_PATH = checkpoint
    logger.info("Loaded tokenizer for shadow action-span masks from %s", checkpoint)
    return _ACTION_MASK_TOKENIZER


def _build_shadow_action_masks(
    args: Any,
    samples: list[Sample],
    loss_masks: list[list[int]],
    hybrid_is_shadow: list[float],
    hybrid_shadow_weights: list[float],
    raw_rewards: list[float],
) -> dict[str, list]:
    from examples.agent_bench.action_span_mask import build_action_token_mask, make_tokenize_len

    mode = os.environ.get("RELAX_SHADOW_ACTION_MASK_MODE", "tool_call")
    min_reward = _env_float("RELAX_SHADOW_AWR_MIN_REWARD", 1.0)
    tokenize_len = None

    action_masks: list[list[bool]] = []
    token_counts: list[float] = []
    token_fracs: list[float] = []
    span_counts: list[float] = []
    char_fracs: list[float] = []

    for i, sample in enumerate(samples):
        response_length = int(sample.response_length or 0)
        base_loss_mask = loss_masks[i]
        should_mask = (
            hybrid_is_shadow[i] > 0
            and hybrid_shadow_weights[i] > 0
            and float(raw_rewards[i]) >= min_reward
            and response_length > 0
        )
        if should_mask:
            if tokenize_len is None:
                tokenizer = _get_action_mask_tokenizer(args)
                tokenize_len = make_tokenize_len(tokenizer)
            mask, stats = build_action_token_mask(
                getattr(sample, "response", "") or "",
                response_token_count=response_length,
                tokenize_len=tokenize_len,
                mode=mode,
            )
            if len(mask) != response_length:
                raise ValueError(
                    f"shadow action mask length mismatch: got {len(mask)}, expected {response_length}"
                )
            mask = [bool(mask_value and bool(loss_value)) for mask_value, loss_value in zip(mask, base_loss_mask)]
            action_count = float(sum(1 for value in mask if value))
            stats["action_token_count"] = action_count
            stats["action_token_frac"] = action_count / float(response_length) if response_length else 0.0
        else:
            mask = [False] * response_length
            stats = {
                "span_count": 0.0,
                "action_token_count": 0.0,
                "action_token_frac": 0.0,
                "char_span_frac": 0.0,
            }

        extra = _mutable_sample_extra_info(sample)
        extra["shadow_action_mask_enabled"] = 1
        extra["shadow_action_mask_mode"] = mode
        extra["shadow_action_span_count"] = stats["span_count"]
        extra["shadow_action_token_count"] = stats["action_token_count"]
        extra["shadow_action_token_frac"] = stats["action_token_frac"]
        extra["shadow_action_char_span_frac"] = stats["char_span_frac"]

        action_masks.append(mask)
        token_counts.append(float(stats["action_token_count"]))
        token_fracs.append(float(stats["action_token_frac"]))
        span_counts.append(float(stats["span_count"]))
        char_fracs.append(float(stats["char_span_frac"]))

    return {
        "shadow_action_loss_masks": action_masks,
        "shadow_action_token_count": token_counts,
        "shadow_action_token_frac": token_fracs,
        "shadow_action_span_count": span_counts,
        "shadow_action_char_span_frac": char_fracs,
    }


def _build_compat_action_masks(
    args: Any,
    samples: list[Sample],
    loss_masks: list[list[int]],
) -> dict[str, list]:
    from examples.agent_bench.action_span_mask import build_action_token_mask, make_tokenize_len

    mode = os.environ.get("RELAX_COMPAT_ACTION_MASK_MODE") or os.environ.get(
        "RELAX_SHADOW_ACTION_MASK_MODE", "tool_call"
    )
    tokenize_len = None

    action_masks: list[list[bool]] = []
    token_counts: list[float] = []
    token_fracs: list[float] = []
    span_counts: list[float] = []
    char_fracs: list[float] = []

    for i, sample in enumerate(samples):
        response_length = int(sample.response_length or 0)
        base_loss_mask = loss_masks[i]
        if response_length > 0:
            if tokenize_len is None:
                tokenizer = _get_action_mask_tokenizer(args)
                tokenize_len = make_tokenize_len(tokenizer)
            mask, stats = build_action_token_mask(
                getattr(sample, "response", "") or "",
                response_token_count=response_length,
                tokenize_len=tokenize_len,
                mode=mode,
            )
            if len(mask) != response_length:
                raise ValueError(
                    f"compat action mask length mismatch: got {len(mask)}, expected {response_length}"
                )
            mask = [bool(mask_value and bool(loss_value)) for mask_value, loss_value in zip(mask, base_loss_mask)]
            action_count = float(sum(1 for value in mask if value))
            stats["action_token_count"] = action_count
            stats["action_token_frac"] = action_count / float(response_length) if response_length else 0.0
        else:
            mask = []
            stats = {
                "span_count": 0.0,
                "action_token_count": 0.0,
                "action_token_frac": 0.0,
                "char_span_frac": 0.0,
            }

        extra = _mutable_sample_extra_info(sample)
        extra["compat_action_mask_enabled"] = 1
        extra["compat_action_mask_mode"] = mode
        extra["compat_action_span_count"] = stats["span_count"]
        extra["compat_action_token_count"] = stats["action_token_count"]
        extra["compat_action_token_frac"] = stats["action_token_frac"]
        extra["compat_action_char_span_frac"] = stats["char_span_frac"]

        action_masks.append(mask)
        token_counts.append(float(stats["action_token_count"]))
        token_fracs.append(float(stats["action_token_frac"]))
        span_counts.append(float(stats["span_count"]))
        char_fracs.append(float(stats["char_span_frac"]))

    return {
        "compat_action_loss_masks": action_masks,
        "compat_action_token_count": token_counts,
        "compat_action_token_frac": token_fracs,
        "compat_action_span_count": span_counts,
        "compat_action_char_span_frac": char_fracs,
    }


def _build_hard_span_masks(
    args: Any,
    samples: list[Sample],
    loss_masks: list[list[int]],
    hybrid_is_shadow: list[float],
    hybrid_shadow_weights: list[float],
    raw_rewards: list[float],
) -> dict[str, list]:
    from examples.agent_bench.hard_span_mask import build_hard_span_token_mask

    mode = os.environ.get("RELAX_HARD_SPAN_ACTION_MASK_MODE") or os.environ.get(
        "RELAX_SHADOW_ACTION_MASK_MODE", "tool_call"
    )
    hard_span_version = os.environ.get("RELAX_HARD_SPAN_VERSION", "v1")
    min_reward = _env_float("RELAX_SHADOW_AWR_MIN_REWARD", 1.0)
    reasoning_max_chars = int(_env_float("RELAX_HARD_SPAN_REASONING_MAX_CHARS", 4096.0))
    final_max_chars = int(_env_float("RELAX_HARD_SPAN_FINAL_MAX_CHARS", 4096.0))
    max_response_tokens = int(_env_float("RELAX_HARD_SPAN_MAX_RESPONSE_TOKENS", 0.0))
    keep_final = _env_bool("RELAX_HARD_SPAN_KEEP_FINAL", True)
    require_useful_reasoning = _env_bool("RELAX_HARD_SPAN_REQUIRE_USEFUL_REASONING", True)
    tokenizer = None

    hard_masks: list[list[bool]] = []
    token_counts: list[float] = []
    token_fracs: list[float] = []
    action_counts: list[float] = []
    reasoning_counts: list[float] = []
    final_counts: list[float] = []
    skill_counts: list[float] = []
    think_counts: list[float] = []
    tool_response_counts: list[float] = []
    span_counts: list[float] = []
    contaminated_action_counts: list[float] = []
    zero_fracs: list[float] = []

    for i, sample in enumerate(samples):
        response_length = int(sample.response_length or 0)
        base_loss_mask = loss_masks[i]
        should_mask = (
            hybrid_is_shadow[i] > 0
            and hybrid_shadow_weights[i] > 0
            and float(raw_rewards[i]) >= min_reward
            and response_length > 0
        )
        if should_mask:
            if tokenizer is None:
                tokenizer = _get_action_mask_tokenizer(args)
            status = sample.status.value if hasattr(sample.status, "value") else str(sample.status)
            mask, stats = build_hard_span_token_mask(
                getattr(sample, "response", "") or "",
                response_token_count=response_length,
                tokenizer=tokenizer,
                mode=mode,
                base_loss_mask=base_loss_mask,
                sample_status=status,
                reasoning_max_chars=reasoning_max_chars,
                final_max_chars=final_max_chars,
                max_response_tokens=max_response_tokens,
                keep_final=keep_final,
                require_useful_reasoning=require_useful_reasoning,
                version=hard_span_version,
            )
            if len(mask) != response_length:
                raise ValueError(f"hard-span mask length mismatch: got {len(mask)}, expected {response_length}")
        else:
            mask = [False] * response_length
            stats = {
                "drop_reason": "ineligible",
                "version": hard_span_version,
                "token_count": 0.0,
                "token_frac": 0.0,
                "action_token_count": 0.0,
                "reasoning_token_count": 0.0,
                "final_token_count": 0.0,
                "excluded_skill_token_count": 0.0,
                "excluded_think_token_count": 0.0,
                "excluded_tool_response_token_count": 0.0,
                "span_count": 0.0,
                "action_span_count": 0.0,
                "reasoning_span_count": 0.0,
                "skill_reasoning_span_count": 0.0,
                "final_span_count": 0.0,
                "contaminated_action_span_count": 0.0,
            }

        hard_count = float(stats["token_count"])
        extra = _mutable_sample_extra_info(sample)
        extra["hard_span_mask_enabled"] = 1
        extra["hard_span_version"] = stats["version"]
        extra["hard_span_action_mask_mode"] = mode
        extra["hard_span_drop_reason"] = stats["drop_reason"]
        extra["hard_span_token_count"] = hard_count
        extra["hard_span_token_frac"] = stats["token_frac"]
        extra["hard_span_action_token_count"] = stats["action_token_count"]
        extra["hard_span_reasoning_token_count"] = stats["reasoning_token_count"]
        extra["hard_span_final_token_count"] = stats["final_token_count"]
        extra["hard_span_excluded_skill_token_count"] = stats["excluded_skill_token_count"]
        extra["hard_span_excluded_think_token_count"] = stats["excluded_think_token_count"]
        extra["hard_span_excluded_tool_response_token_count"] = stats["excluded_tool_response_token_count"]
        extra["hard_span_span_count"] = stats["span_count"]
        extra["hard_span_action_span_count"] = stats["action_span_count"]
        extra["hard_span_reasoning_span_count"] = stats["reasoning_span_count"]
        extra["hard_span_skill_reasoning_span_count"] = stats["skill_reasoning_span_count"]
        extra["hard_span_final_span_count"] = stats["final_span_count"]
        extra["hard_span_contaminated_action_span_count"] = stats["contaminated_action_span_count"]

        hard_masks.append(mask)
        token_counts.append(hard_count)
        token_fracs.append(float(stats["token_frac"]))
        action_counts.append(float(stats["action_token_count"]))
        reasoning_counts.append(float(stats["reasoning_token_count"]))
        final_counts.append(float(stats["final_token_count"]))
        skill_counts.append(float(stats["excluded_skill_token_count"]))
        think_counts.append(float(stats["excluded_think_token_count"]))
        tool_response_counts.append(float(stats["excluded_tool_response_token_count"]))
        span_counts.append(float(stats["span_count"]))
        contaminated_action_counts.append(float(stats["contaminated_action_span_count"]))
        zero_fracs.append(1.0 if should_mask and hard_count <= 0 else 0.0)

    return {
        "hard_span_loss_masks": hard_masks,
        "hard_span_token_count": token_counts,
        "hard_span_token_frac": token_fracs,
        "hard_span_action_token_count": action_counts,
        "hard_span_reasoning_token_count": reasoning_counts,
        "hard_span_final_token_count": final_counts,
        "hard_span_excluded_skill_token_count": skill_counts,
        "hard_span_excluded_think_token_count": think_counts,
        "hard_span_excluded_tool_response_token_count": tool_response_counts,
        "hard_span_span_count": span_counts,
        "hard_span_contaminated_action_span_count": contaminated_action_counts,
        "hard_span_zero_frac": zero_fracs,
    }


def convert_samples_to_train_data(args: Any, samples: list[Sample] | list[list[Sample]]):
    """Convert inference generated samples to training data."""
    raw_rewards, rewards = post_process_rewards(args, samples)

    assert len(raw_rewards) == len(samples)
    assert len(rewards) == len(samples)

    # Clean (penalty-free) score for monitoring metrics such as pass@k.
    # `raw_reward` above carries the reward-shaped score (e.g. soft-overlong
    # penalty applied to sample.reward[reward_key]) which is correct for
    # advantage, but pollutes pass@k. We expose the original verifier score via
    # reward["raw_score"] so pass@k reflects true task success, not length.
    def _clean_score(sample):
        rw = sample.reward
        if isinstance(rw, dict) and "raw_score" in rw:
            try:
                return float(rw["raw_score"])
            except (TypeError, ValueError):
                pass
        return sample.get_reward_value(args)

    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        # some reward model, e.g. remote rm, may return multiple rewards,
        # we could use key to select the reward.
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "clean_pass_score": [_clean_score(sample) for sample in samples],
        "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample.index for sample in samples],
    }

    # loss mask
    # TODO: compress the loss mask
    loss_masks = []
    for sample in samples:
        # always instantiate loss_mask if not provided
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        else:
            # NOTE(jiajia): loss_mask is not None only if args.mask_offpolicy_in_partial_rollout is True, so we need to pad it to response_length.
            sample.loss_mask += [1] * (sample.response_length - len(sample.loss_mask))

        assert len(sample.loss_mask) == sample.response_length, (
            f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
        )
        if sample.remove_sample:
            sample.loss_mask = [0] * sample.response_length
        loss_masks.append(sample.loss_mask)
    train_data["loss_masks"] = loss_masks

    if _env_bool("RELAX_SELECTOR_ACTION_CREDIT", False):
        from examples.agent_bench.selector_action_credit import build_train_fields

        train_data.update(build_train_fields(samples, loss_masks))

    # overwriting the raw reward
    # populate this field for a subset of samples (e.g. SWE but not code).
    if any(sample.metadata and "raw_reward" in sample.metadata for sample in samples):
        train_data["raw_reward"] = [
            sample.metadata["raw_reward"] if sample.metadata and "raw_reward" in sample.metadata else sample.reward
            for sample in samples
        ]

    hybrid_is_shadow = []
    hybrid_grpo_weights = []
    hybrid_shadow_weights = []
    for sample in samples:
        is_shadow = _hybrid_is_shadow_sample(sample)
        hybrid_is_shadow.append(1.0 if is_shadow else 0.0)
        hybrid_grpo_weights.append(_hybrid_grpo_weight(sample, is_shadow))
        hybrid_shadow_weights.append(_hybrid_shadow_weight(sample, is_shadow))
    train_data["hybrid_is_shadow"] = hybrid_is_shadow
    train_data["hybrid_grpo_weight"] = hybrid_grpo_weights
    train_data["hybrid_shadow_weight"] = hybrid_shadow_weights

    if _env_bool("RELAX_SHADOW_BC_ACTION_MASK", False):
        train_data.update(
            _build_shadow_action_masks(
                args,
                samples,
                loss_masks,
                hybrid_is_shadow,
                hybrid_shadow_weights,
                raw_rewards,
            )
        )
    if _env_bool("RELAX_SHADOW_BC_HARD_SPAN_MASK", False):
        train_data.update(
            _build_hard_span_masks(
                args,
                samples,
                loss_masks,
                hybrid_is_shadow,
                hybrid_shadow_weights,
                raw_rewards,
            )
        )
    if _env_bool("RELAX_COMPAT_ACTION_MONITOR", False):
        train_data.update(_build_compat_action_masks(args, samples, loss_masks))

    # For rollout buffer
    if samples[0].metadata and "round_number" in samples[0].metadata:
        train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

    # Add rollout log probabilities for off-policy correction
    if samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

    if samples[0].rollout_routed_experts is not None:
        train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

    if samples[0].train_metadata is not None:
        metadata_values = [sample.train_metadata for sample in samples]
        if all(isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)) for value in metadata_values):
            train_data["metadata"] = metadata_values
        else:
            logger.warning(
                "Skipping non-scalar train_metadata before TensorDict conversion; first_type=%s",
                type(metadata_values[0]).__name__,
            )

    if args.multimodal_keys is not None:
        train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

    if _env_bool("RELAX_OPSD_MODE", False):
        # OPSD safety fill: train-data packing keys on samples[0], so every
        # sample must carry a length-aligned teacher_log_probs list. Samples
        # missed by prompt-swap scoring (e.g. buffered pre-OPSD partials) get
        # a fallback so packing/loss shapes stay consistent. The fallback is
        # NOT numerically inert (student side is Megatron-recomputed logp),
        # so a per-sample validity flag gates the reverse-KL term in
        # apply_opd_kl_to_advantages: only opsd_scored samples contribute.
        from relax.engine.rollout.on_policy_distillation import _fallback_teacher_log_probs

        opsd_teacher_valid: list[float] = []
        for sample in samples:
            if sample.teacher_log_probs is None:
                sample.teacher_log_probs = _fallback_teacher_log_probs(sample, int(sample.response_length or 0))
            extra = _sample_extra_info(sample)
            scored = _metadata_float(extra, "opsd_scored")
            opsd_teacher_valid.append(1.0 if (scored or 0.0) > 0.0 else 0.0)
        train_data["opsd_teacher_valid"] = opsd_teacher_valid

        if _env_bool("RELAX_OPSD_SKILL_TOKEN_MASK", False):
            # Skill-register gating for the OPSD distillation term: reuse the
            # hard-span v1-v4 keyword recipes to zero the distill weight on
            # skill-register tokens (skill_reasoning blocks, skill-referencing
            # lines, privileged path/source prose). GRPO is unaffected.
            from examples.agent_bench.hard_span_mask import build_skill_register_token_mask

            register_tokenizer = _get_action_mask_tokenizer(args)
            opsd_distill_token_masks: list[list[float]] = []
            opsd_register_token_frac: list[float] = []
            for sample in samples:
                response_length = int(sample.response_length or 0)
                register_mask, register_stats = build_skill_register_token_mask(
                    getattr(sample, "response", "") or "",
                    response_token_count=response_length,
                    tokenizer=register_tokenizer,
                )
                if len(register_mask) != response_length:
                    raise ValueError(
                        f"skill-register mask length mismatch: got {len(register_mask)}, expected {response_length}"
                    )
                opsd_distill_token_masks.append([0.0 if flagged else 1.0 for flagged in register_mask])
                frac = float(register_stats.get("register_token_frac") or 0.0)
                opsd_register_token_frac.append(frac)
                _mutable_sample_extra_info(sample)["opsd_register_token_frac"] = frac
            train_data["opsd_distill_token_masks"] = opsd_distill_token_masks
            train_data["opsd_register_token_frac"] = opsd_register_token_frac

    if samples[0].teacher_log_probs is not None:
        train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

    if args.debug_train_only and any(hasattr(sample, "ref_log_probs") for sample in samples):
        train_data["ref_log_probs"] = [
            getattr(sample, "ref_log_probs", None) or [0.0] * sample.response_length for sample in samples
        ]
    if args.debug_train_only and any(hasattr(sample, "log_probs") for sample in samples):
        train_data["log_probs"] = [
            getattr(sample, "log_probs", None) or [0.0] * sample.response_length for sample in samples
        ]
    if args.debug_train_only and any(hasattr(sample, "advantages") for sample in samples):
        train_data["advantages"] = [
            getattr(sample, "advantages", None) or [0.0] * sample.response_length for sample in samples
        ]
    if args.debug_train_only and any(hasattr(sample, "returns") for sample in samples):
        train_data["returns"] = [
            getattr(sample, "returns", None) or [0.0] * sample.response_length for sample in samples
        ]

    if any(sample.teacher_topk_token_ids is not None for sample in samples):
        topk_k = max(
            (
                len(sample.teacher_topk_token_ids[0])
                for sample in samples
                if sample.teacher_topk_token_ids is not None and len(sample.teacher_topk_token_ids) > 0
            ),
            default=0,
        )
        train_data["teacher_topk_token_ids"] = [
            (
                [token_id for step_topk in sample.teacher_topk_token_ids for token_id in step_topk]
                if sample.teacher_topk_token_ids is not None
                else []
            )
            for sample in samples
        ]
        train_data["teacher_topk_k"] = [topk_k for _ in samples]

    total_lengths = [len(t) for t in train_data["tokens"]]
    train_data["total_lengths"] = total_lengths
    if args.debug_train_only:
        return train_data
    rollout_batch = dict_to_tensordict(train_data, len(total_lengths))
    return rollout_batch


def post_process_rewards(args: Any, samples: list[Sample] | list[list[Sample]]):
    """Post-process rewards and return (raw_rewards, possibly-normalized
    rewards).

    Returns:
        Tuple[List[float], List[float]]
    """
    if args.custom_reward_post_process_path is not None:
        custom_reward_post_process_func = load_function(args.custom_reward_post_process_path)
        return custom_reward_post_process_func(args, samples)

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if (
        args.advantage_estimator in ["grpo", "gspo", "sapo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        # group norm
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        else:
            # when samples count are not equal in each group
            rewards = rewards.view(-1, rewards.shape[-1])
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean

        if args.advantage_estimator in ["grpo", "gspo", "sapo"] and args.grpo_std_normalization:
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)

        return raw_rewards, rewards.flatten().tolist()

    return raw_rewards, raw_rewards


def dict_to_tensordict(
    data: Dict[str, List],
    batch_size: Union[int, torch.Size, None] = None,
    device: Optional[torch.device] = None,
) -> TensorDict:
    """Convert a nested-list dictionary to a TensorDict.

    Args:
        data: Mapping of keys to nested lists (supports depth 1 or 2).
        batch_size: Optional batch size. If None, caller may set an appropriate
            batch size (TensorDict accepts None or an int/torch.Size).
        device: Optional target torch.device for created tensors.

    Returns:
        A TensorDict built from the input nested lists.
    """
    if not data:
        return TensorDict({}, batch_size=0 if batch_size is None else batch_size, device=device)

    def _nesting_depth(x):
        if isinstance(x, list) and x:
            return 1 + _nesting_depth(x[0])
        return 0

    def _scalar_dtype(sample) -> Optional[torch.dtype]:
        """Return an explicit dtype only for bool/float; None lets torch.tensor
        infer."""
        if isinstance(sample, bool):
            return torch.bool
        if isinstance(sample, float):
            return torch.float32
        # int or mixed int/float: let torch.tensor auto-promote (C++ level, zero overhead)
        return None

    def _to_tensor_1d(lst):
        dtype = _scalar_dtype(lst[0])
        return torch.tensor(lst, dtype=dtype, device=device)

    def _to_tensor_2d(lst):
        dtype = _scalar_dtype(lst[0][0])
        tensors = [torch.tensor(seq, dtype=dtype, device=device) for seq in lst]
        return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)

    result = {}

    for key, value in data.items():
        if not isinstance(value, list):
            raise TypeError(f"Value for key '{key}' must be a list, got {type(value)}")
        if key == "rollout_routed_experts":
            # Flatten 3D numpy (seq_i, num_layers, topk) -> 2D tensor (seq_i, num_layers*topk)
            # so NestedTensor jagged layout can handle variable seq_len efficiently.
            # This avoids NonTensorStack wrapping which forces slow pickle serialization
            # during dist.broadcast_object_list (~377 MB pickle -> ~14s overhead).
            tensors = [
                torch.from_numpy(np.ascontiguousarray(arr.reshape(arr.shape[0], -1))).to(torch.int32) for arr in value
            ]
            result[key] = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
            continue
        depth = _nesting_depth(value)
        if depth == 0:  # empty list []
            tensor = torch.empty(0)
        elif depth == 1:
            if key == "multimodal_train_inputs":
                tensor = value
            else:
                try:
                    tensor = _to_tensor_1d(value)
                except Exception as exc:
                    first_type = type(value[0]).__name__ if value else "empty"
                    raise TypeError(
                        f"Failed to convert key '{key}' with depth {depth} and first_type {first_type} to TensorDict"
                    ) from exc
        elif depth == 2:
            try:
                tensor = _to_tensor_2d(value)
            except Exception as exc:
                first_type = type(value[0][0]).__name__ if value and value[0] else "empty"
                raise TypeError(
                    f"Failed to convert key '{key}' with depth {depth} and first_type {first_type} to TensorDict"
                ) from exc
        else:
            raise ValueError(f"Unsupported nesting depth {depth} for key '{key}'. Max supported: 2.")

        result[key] = tensor

    return TensorDict(result, batch_size=batch_size, device=device)


def _resolve_to_ip(addr: str) -> str:
    """Resolve *addr* to an IPv4/IPv6 address string.

    If *addr* is already a valid IP literal, return it unchanged. Otherwise,
    treat it as a hostname and resolve it via DNS. Falls back to
    ``"127.0.0.1"`` if resolution fails.
    """
    import ipaddress as _ipaddress

    # Fast path: addr is already an IP literal
    try:
        _ipaddress.ip_address(addr)
        return addr
    except ValueError:
        pass

    # addr is a hostname — resolve it
    try:
        return socket.gethostbyname(addr)
    except socket.gaierror:
        logger.warning("Failed to resolve hostname %r to IP; falling back to 127.0.0.1", addr)
        return "127.0.0.1"


def post_process_env(args, env):
    """Set and return environment variables required for rollout workers.

    Populates common env keys used by the rollout processes.
    """
    cur_dir = Path(__file__).resolve().parent
    repo_dir = cur_dir.parent.parent

    if "env_vars" not in env or not isinstance(env["env_vars"], dict):
        env["env_vars"] = {}

    env["env_vars"]["TQ_PRE_ALLOC_SAMPLE_NUM"] = str(args.rollout_batch_size * args.n_samples_per_prompt)
    env["env_vars"]["TQ_ZERO_COPY_SERIALIZATION"] = "true"
    env["env_vars"]["SLIME_HOST_IP"] = _resolve_to_ip(os.getenv("MASTER_ADDR", "127.0.0.1"))

    if os.getenv("RAY_DEBUG", "0") == "1":
        env["env_vars"]["RAY_DEBUG_POST_MORTEM"] = "1"
        env["env_vars"]["RAY_DEBUG"] = "1"

    # Runtime env from configs/env.yaml is static; launch scripts often need to
    # override host-specific values such as Docker/proxy/W&B/CUDA paths.  Apply
    # these after loading YAML so the actual launch environment wins.
    for key in (
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDNN_HOME",
        "LD_LIBRARY_PATH",
        "TOKENIZERS_PARALLELISM",
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCH_COMPILE_DISABLE",
        "TORCHDYNAMO_DISABLE",
        "RELAX_TRAIN_STEP_DIAG",
        "RELAX_TRAIN_SUBSTEP_DIAG",
        "RELAX_ENABLE_QWEN35_COMPAT",
        "DOCKER_HOST",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
        "NCCL_IB_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "NCCL_NVLS_ENABLE",
        "NCCL_CUMEM_ENABLE",
        "NCCL_SOCKET_FAMILY",
        "NCCL_RAS_ENABLE",
        "NCCL_ASYNC_ERROR_HANDLING",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "TORCH_NCCL_ENABLE_MONITORING",
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
        "TORCH_NCCL_DUMP_ON_TIMEOUT",
        "TORCH_NCCL_TRACE_BUFFER_SIZE",
        "TORCH_NCCL_DESYNC_DEBUG",
        "GLOO_SOCKET_IFNAME",
        "RAY_SERVE_CONTROLLER_NODE_RESOURCE",
        "RAY_SERVE_HTTP_PROXY_TIMEOUT_S",
        "UNIFIED_LAUNCHER_MODE",
        "UNIFIED_CLAW_USE_DOCKER_SANDBOX",
        "UNIFIED_CLAW_SANDBOX_FAIL_HARD",
        "UNIFIED_DISABLE_THINKING",
        "UNIFIED_ROLLOUT_WALLCLOCK_CAP_SEC",
        "AGENT_BENCH_DOCKER_START_CONCURRENCY",
        "AGENT_BENCH_ACTIVE_ENV_CONCURRENCY",
        "AGENT_BENCH_SETUP_ATTEMPTS",
        "AGENT_BENCH_SETUP_TOTAL_TIMEOUT_SEC",
        "AGENT_BENCH_RETRIEVAL_TOP_N",
        "AGENT_BENCH_EXTRA_SKILL_ROOTS",
        "UNIFIED_HARBOR_BUILD_TIMEOUT_SEC",
        "UNIFIED_HARBOR_REQUIRE_PREBUILT_LOCAL",
        "UNIFIED_VERIFIER_TIMEOUT_CAP_SEC",
        "UNIFIED_DOCKER_PIDS_LIMIT",
        "UNIFIED_DOCKER_ULIMIT_FSIZE_GB",
        "UNIFIED_DOCKER_CPUSET",
        "UNIFIED_DOCKER_BUILD_JOBS",
        "UNIFIED_DOCKER_NETWORK_HOST",
        "UNIFIED_CONTAINER_PROXY",
        "UNIFIED_TOOL_TIMEOUT_CHILD_CLEANUP",
        "SETA_CONTINUOUS_REWARD",
        "UNIFIED_SWE_VERIFIER_TIMEOUT_SEC",
        "AGENT_BENCH_SKIP_CLOSE_GRADING_ON_ABORT",
        "RELAX_REQUEUE_ABORTED_GROUPS",
        "RELAX_MAX_DROPPED_ABORT_GROUPS_PER_ROLLOUT",
        "RELAX_DYNAMIC_FILTER_MAX_REJECTS_PER_ROLLOUT",
        "RELAX_DYNAMIC_FILTER_MAX_REJECT_SAMPLES_PER_ROLLOUT",
        "RELAX_DYNAMIC_FILTER_MIN_SKILL_READ_FRAC",
        "RELAX_DYNAMIC_FILTER_MIN_NO_SKILL_READ_FRAC",
        "RELAX_DYNAMIC_FILTER_SKILL_READ_MAX_SAMPLES",
        "RELAX_ABORT_PENDING_TIMEOUT_SEC",
        "RELAX_ABORT_PROTECTED_TIMEOUT_SEC",
        "RELAX_ABORT_CANCEL_WAIT_SEC",
        "RELAX_CP64K_DIAG",
        "RELAX_SOFT_OVERLONG_PENALTY",
        "RELAX_SOFT_OVERLONG_LMAX",
        "RELAX_SOFT_OVERLONG_CACHE",
        "RELAX_SKILL_GROUP_REWARD",
        "RELAX_SKILL_GROUP_BONUS_COEF",
        "RELAX_SKILL_GROUP_BONUS_MAX",
        "RELAX_SKILL_GROUP_MARGIN",
        "RELAX_SKILL_GROUP_SUBGROUP_ADV_COEF",
        "RELAX_SKILL_GROUP_REQUIRE_BOTH",
        "RELAX_SKILL_GROUP_NO_READ_SUCCESS_BONUS",
        "RELAX_SKILL_GROUP_NO_READ_SUCCESS_THRESHOLD",
        "RELAX_MIXED_SKILL_BONUS_ENABLED",
        "RELAX_MIXED_SKILL_BONUS_ORACLE",
        "RELAX_MIXED_SKILL_BONUS_MISLEADING",
        "RELAX_MIXED_SKILL_BONUS_NO_READ_SUCCESS",
        "RELAX_MIXED_SEPARATED_ADV_ENABLED",
        "RELAX_MIXED_SEPARATED_BEHAVIOR_COEF",
        "RELAX_MIXED_SEPARATED_BEHAVIOR_CLIP",
        "RELAX_SELECTOR_ACTION_CREDIT",
        "RELAX_SELECTOR_ACTION_LOSS_COEF",
        "RELAX_M1_CLEAN",
        "RELAX_PROMPT_ONLY_SHADOW_CLEAN",
        "RELAX_PAIR_ATOMIC_SAMPLING",
        "RELAX_PAIR_SPECULATIVE_EXTRA_GROUPS",
        "RELAX_PAIR_BC_PASS_THRESHOLD",
        "RELAX_PAIR_ORACLE_BC_UNTIL_STEP",
        "RELAX_PAIR_ORACLE_GRPO",
        "RELAX_PAIR_ORACLE_GRPO_CROSS_ARM_ADV",
        "RELAX_PAIR_ORACLE_GRPO_CROSS_ARM_ADV_CLIP",
        "RELAX_PAIR_ORACLE_GRPO_DROP_ALL_PASS",
        "RELAX_SLATE_REGRET_GRPO",
        "RELAX_SLATE_UNIFORM_MIN_DELTA",
        "RELAX_SLATE_REGRET_COEF",
        "RELAX_SLATE_REGRET_COEF_NOGOLD",
        "RELAX_SLATE_STRATIFIED_ADVANTAGE",
        "RELAX_SLATE_STRATIFIED_ADV_COEF",
        "RELAX_SLATE_STRATIFIED_SHRINKAGE",
        "RELAX_SLATE_STRATIFIED_ADV_CLIP",
        "RELAX_EVAL_BEFORE_ROLLOUT_ID",
        "RELAX_SHADOW_BC_ACTION_MASK",
        "RELAX_SHADOW_ACTION_MASK_MODE",
        "RELAX_SHADOW_ACTION_MASK_RENORMALIZE",
        "RELAX_SHADOW_BC_HARD_SPAN_MASK",
        "RELAX_HARD_SPAN_VERSION",
        "RELAX_HARD_SPAN_ACTION_MASK_MODE",
        "RELAX_HARD_SPAN_MASK_RENORMALIZE",
        "RELAX_HARD_SPAN_REASONING_MAX_CHARS",
        "RELAX_HARD_SPAN_FINAL_MAX_CHARS",
        "RELAX_HARD_SPAN_MAX_RESPONSE_TOKENS",
        "RELAX_HARD_SPAN_KEEP_FINAL",
        "RELAX_HARD_SPAN_REQUIRE_USEFUL_REASONING",
        "RELAX_SHADOW_BC_COMPAT_WEIGHTS",
        "RELAX_COMPAT_GAP_LOW",
        "RELAX_COMPAT_GAP_HIGH",
        "RELAX_COMPAT_LOW_WEIGHT",
        "RELAX_COMPAT_MID_WEIGHT",
        "RELAX_COMPAT_OOD_WEIGHT",
        "RELAX_COMPAT_RENORMALIZE",
        "RELAX_COMPAT_ACTION_MONITOR",
        "RELAX_COMPAT_ACTION_MASK_MODE",
        "RELAX_COMPAT_REWARD_THRESHOLD",
        "RELAX_COMPAT_TEACHER_ACTION_NLL_SFT_BASELINE",
        "RELAX_SHADOW_AWR_WEIGHT_MODE",
        "RELAX_SHADOW_AWR_TEMPERATURE",
        "RELAX_SHADOW_AWR_MAX_WEIGHT",
        "RELAX_SHADOW_AWR_MIN_REWARD",
        "RELAX_SHADOW_AWR_ZERO_BELOW_MIN",
        "RELAX_SHADOW_AWR_NORMALIZE_WEIGHTS",
        "RELAX_SHADOW_LOSS_COEF",
        "RELAX_OPSD_MODE",
        "RELAX_OPSD_FORM",
        "RELAX_OPSD_K3_COEF",
        "RELAX_OPSD_K3_ELL_CLAMP",
        "RELAX_OPSD_K3_RHO_CLAMP",
        "RELAX_OPSD_SKILL_TOKEN_MASK",
        "RELAX_OPSD_KL_COEF",
        "RELAX_OPSD_SCOPE",
        "RELAX_OPSD_TEACHER_SELF_ROUTER",
        "RELAX_OPSD_TEACHER_TIMEOUT_S",
        "RELAX_PIN_NODE_ACTOR",
        "RELAX_PIN_NODE_ACTOR_FWD",
        "RELAX_PIN_NODE_REFERENCE",
        "RELAX_PIN_NODE_ROLLOUT",
        "RELAX_DCS_MASTER_PORT_MIN",
        "RELAX_DCS_MASTER_PORT_MAX",
        "RELAX_DCS_ACTOR_FWD_REF_PORT_MIN",
        "RELAX_DCS_ACTOR_FWD_REF_PORT_MAX",
        "RELAX_TRAIN_MASTER_PORT_ACTOR_MIN",
        "RELAX_TRAIN_MASTER_PORT_ACTOR_MAX",
        "RELAX_TRAIN_MASTER_PORT_ACTOR_FWD_MIN",
        "RELAX_TRAIN_MASTER_PORT_ACTOR_FWD_MAX",
        "RELAX_TRAIN_MASTER_PORT_REFERENCE_MIN",
        "RELAX_TRAIN_MASTER_PORT_REFERENCE_MAX",
        "RELAX_TRAIN_MASTER_PORT_CRITIC_MIN",
        "RELAX_TRAIN_MASTER_PORT_CRITIC_MAX",
        "RELAX_TRAIN_MASTER_PORT_DEFAULT_MIN",
        "RELAX_TRAIN_MASTER_PORT_DEFAULT_MAX",
        "WANDB_API_KEY",
        "WANDB_PROJECT",
        "WANDB_BASE_URL",
        "WANDB_MODE",
        "JINA_API_KEY",
        "EXA_API_KEY",
        "GOOGLE_CSE_API_KEY",
        "GOOGLE_CSE_ID",
    ):
        if key in os.environ:
            env["env_vars"][key] = os.environ[key]

    # Propagate PYTHONPATH to Ray workers so external packages (e.g. Megatron-LM)
    # are available in Serve replicas and remote actors.
    python_paths = [str(repo_dir)]
    if pp := os.environ.get("PYTHONPATH"):
        python_paths += pp.split(":")
    if pp := env["env_vars"].get("PYTHONPATH"):
        python_paths += pp.split(":")

    # deduplicate with order
    python_paths = list(dict.fromkeys(python_paths))

    env["env_vars"]["PYTHONPATH"] = ":".join(python_paths)
    log_env = dict(env["env_vars"])
    for key in list(log_env):
        upper_key = key.upper()
        if any(marker in upper_key for marker in ("KEY", "TOKEN", "SECRET", "PASS")):
            log_env[key] = "<redacted>"
    logger.info(f"Ray runtime env: {log_env}")
    return env


def merge_dict_list(dict_list):
    """Merge a list of (dict, something) pairs into a single dict of lists.

    Each input item is expected to be a (dict, <unused>) tuple. For each key,
    values that are list/tuple are extended, otherwise appended.

    Args:
        dict_list: Iterable of (dict, any) pairs.

    Returns:
        A dict mapping keys to lists of aggregated values.
    """
    merged: Dict[str, List[Any]] = {}
    for d, _ in dict_list:
        for key, value in d.items():
            # ensure target key maps to a list
            if key not in merged:
                merged[key] = []
            # extend if iterable (list/tuple) and not a string/bytes, else append
            if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
                merged[key].extend(value)
            else:
                merged[key].append(value)
    return merged


def get_debug_data(args, rollout_id: int, batch_size, dp_rank: int) -> Dict[str, Any]:
    """Fetch debug data for a given rollout_id from the data system.

    Parameters:
        rollout_id: The rollout ID for which to fetch debug data.
    Returns:
        A dictionary containing the debug data for the specified rollout ID.
    """

    data = torch.load(
        open(args.load_debug_rollout_data.format(rollout_id=rollout_id), "rb"),
        weights_only=False,
    )["samples"]
    data = [Sample.from_dict(sample) for sample in data]
    if (ratio := args.load_debug_rollout_data_subsample) is not None:
        original_num_rows = len(data)
        rough_subsample_num_rows = int(original_num_rows * ratio)
        data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
        logger.info(
            f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
        )
    rollout_batch = convert_samples_to_train_data(args, data)

    for key in rollout_batch:
        rollout_batch[key] = rollout_batch[key][dp_rank * batch_size : (dp_rank + 1) * batch_size]
    return rollout_batch


async def transfer_batch_to_data_system(
    args: Namespace,
    batch_samples: List,
    batch_count: int,
    rollout_id: int,
    data_system_client: Any,
) -> None:
    """Helper function to transfer a batch of samples to the data system
    client.

    Args:
        batch_samples: List of sample groups
        batch_count: Batch sequence number
        rollout_id: Rollout identifier
        data_system_client: Client for async data transfer
    """
    try:
        # Guard against empty batch_samples
        if not batch_samples:
            logger.warning(
                f"transfer_batch_to_data_system called with empty batch_samples for rollout_id={rollout_id}, batch_count={batch_count}"
            )
            return
        batch_samples = sorted(
            batch_samples, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
        )
        # Flatten nested groups of samples into a single list
        while isinstance(batch_samples[0], list):
            batch_samples = sum(batch_samples, [])
        global CURRENT_ROLLOUT_BATCH
        CURRENT_ROLLOUT_BATCH.extend(batch_samples)
        rollout_batch = convert_samples_to_train_data(args, batch_samples)
        logger.info(f"Prepared rollout batch {batch_count} with {rollout_batch.numel()} samples for transfer")
        logger.info(f"Transferring batch rollout_batch: {rollout_batch}")
        metadata = await data_system_client.async_put(data=rollout_batch, partition_id=f"train_{rollout_id}")

        # Store total_lengths in custom_meta so that the TransferQueue sampler
        # can use it for seqlen-balanced partitioning across DP ranks.
        if metadata and metadata.size > 0:
            total_lengths = rollout_batch.get("total_lengths", None)
            if total_lengths is not None:
                custom_meta = [{"total_lengths": int(tl)} for tl in total_lengths]
                metadata.update_custom_meta(custom_meta)
                await data_system_client.async_set_custom_meta(metadata)

        logger.info(f"Batch {batch_count} transferred successfully for rollout_id: {rollout_id}")
    except Exception as e:
        logger.error(f"Error transferring batch {batch_count}: {e}")
        raise


def process_args(args: Namespace, role: str) -> None:
    """Process args for reference actor and actor fwd."""
    # Adjust max tokens per GPU for reference actor and actor fwd
    for key in args.ref_actor_config:
        setattr(args, key, args.ref_actor_config[key])
    args.only_load_weight = True
    if role == "reference" or role == "actor_fwd":
        args.load = args.ref_load


def get_serve_url(route_prefix: str = "") -> str:
    """Return an accessible HTTP URL for the current Ray Serve deployment.

    Notes:
        - Call after `serve.run()` from a client that can reach the Ray
          cluster (typically the head node).
        - Returns a URL like: http://<head-node-ip>:<http-port><route_prefix>

    Args:
        route_prefix: Optional route prefix to append to the base URL.
    """
    # 1. Determine Serve proxy IP. If we intentionally pin ServeController away
    #    from a small head node, use the same node's proxy for coordinator calls.
    controller_node_resource = os.environ.get("RAY_SERVE_CONTROLLER_NODE_RESOURCE", "")
    if controller_node_resource.startswith("node:"):
        head_ip = controller_node_resource.split("node:", 1)[1]
    else:
        head_ip = None

    # 2. Determine head node IP. Prefer Ray cluster state; fall back to
    #    local hostname resolution for client-on-head scenarios.
    try:
        # ray.nodes() returns info for all nodes. Ray 2.x auto-registers
        # "node:__internal_head__" on the head node; some legacy setups also
        # mark it with a custom "head" resource. Accept either.
        if not head_ip:
            for node in ray.nodes():
                if not node["Alive"]:
                    continue
                resources = node.get("Resources", {})
                if "node:__internal_head__" in resources or resources.get("head"):
                    head_ip = node["NodeManagerAddress"]
                    break
            else:
                # If no head marker, fall back to the first alive node
                head_ip = ray.nodes()[0]["NodeManagerAddress"]
    except Exception:
        # Fallback: resolve local hostname (works when client runs on head)
        head_ip = head_ip or socket.gethostbyname(socket.gethostname())

    # 3. 格式化 route_prefix
    if route_prefix and not route_prefix.startswith("/"):
        route_prefix = "/" + route_prefix

    serve_url = f"http://{head_ip}:{8000}{route_prefix}"
    logger.info("Serve URL: %s", serve_url)
    return serve_url


def recovery_load_path(args: Namespace) -> Optional[str]:
    """Determine the checkpoint path to load for recovery, if applicable."""
    if args.save is not None and os.path.exists(os.path.join(args.save, "latest_checkpointed_iteration.txt")):
        args.no_load_optim = False
        args.no_load_rng = False
        args.finetune = False
        args.load = args.save


def compute_dp_size(config) -> int:
    """Compute data-parallel size from config for the actor role.

    For Megatron backend: dp_size = total_actor_gpus / (tp * pp * cp)
    """
    _, actor_total_gpus = config.resource.get("actor", (1, 1))
    tp = getattr(config, "tensor_model_parallel_size", 1)
    pp = getattr(config, "pipeline_model_parallel_size", 1)
    cp = getattr(config, "context_parallel_size", 1)
    dp_size = actor_total_gpus // (tp * pp * cp)
    if dp_size <= 0:
        raise ValueError(
            f"Computed dp_size={dp_size} is invalid. actor_total_gpus={actor_total_gpus}, tp={tp}, pp={pp}, cp={cp}"
        )
    return dp_size
