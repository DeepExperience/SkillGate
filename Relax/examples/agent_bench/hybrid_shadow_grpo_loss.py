"""Hybrid no-skill GRPO + oracle-shadow weighted BC/AWR loss.

Use with mixed batches where each rollout sample carries numeric tags:

- ``hybrid_grpo_weight``: sample participates in normal GRPO/PPO loss.
- ``hybrid_shadow_weight``: sample participates in shadow BC/AWR loss.
- ``hybrid_is_shadow``: metric-only sample type tag.

The rollout side may use oracle skills for shadow rows, but M1 cleaning rewrites
those trajectories to no-skill transcripts before this loss sees them.
"""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import torch

from examples.agent_bench.shadow_awr_loss import _as_reward_tensor, _build_weights, _env_bool, _env_float


def _cat_field(values: Any, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=dtype).flatten()
    if values is None:
        raise ValueError("required batch field is missing")
    pieces = []
    for value in values:
        if isinstance(value, torch.Tensor):
            pieces.append(value.to(device=device, dtype=dtype).flatten())
        else:
            pieces.append(torch.tensor([float(value)], device=device, dtype=dtype))
    if not pieces:
        return torch.empty(0, device=device, dtype=dtype)
    return torch.cat(pieces, dim=0)


def _env_optional_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return float(value)


def _masked_mean(values: torch.Tensor, weights: torch.Tensor, *, default: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=torch.float32)
    denom = weights.sum()
    return torch.where(
        denom > 0,
        (values.to(torch.float32) * weights).sum() / denom.clamp_min(1e-12),
        default,
    )


def _masked_median(values: torch.Tensor, mask: torch.Tensor, *, default: torch.Tensor) -> torch.Tensor:
    selected = values.to(torch.float32)[mask.to(device=values.device) > 0]
    if selected.numel() == 0:
        return default
    return selected.median()


def _optional_sample_weights(
    values: Any,
    *,
    device: torch.device,
    count: int,
    default: torch.Tensor | float,
) -> torch.Tensor:
    if values is None:
        if isinstance(default, torch.Tensor):
            weights = default.to(device=device, dtype=torch.float32).flatten()
        else:
            weights = torch.full((count,), float(default), device=device, dtype=torch.float32)
    else:
        weights = _cat_field(values, device=device, dtype=torch.float32)

    if weights.numel() != count:
        raise ValueError(f"hybrid sample weight count mismatch: got {weights.numel()}, expected {count}")
    return weights


def _expand_sample_weights(weights: torch.Tensor, log_prob_chunks: list[torch.Tensor]) -> torch.Tensor:
    pieces = [
        torch.ones_like(log_prob, dtype=torch.float32) * weight
        for log_prob, weight in zip(log_prob_chunks, weights, strict=False)
    ]
    if not pieces:
        return torch.empty(0, device=weights.device, dtype=torch.float32)
    return torch.cat(pieces, dim=0)


def _response_mask_chunks(
    values: Any,
    *,
    args: Namespace,
    log_prob_chunks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    device: torch.device,
    max_seq_lens: list[int] | None = None,
    padded_total_lengths: list[int] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    from relax.backends.megatron.cp_utils import slice_log_prob_with_cp

    if values is None:
        raise ValueError("response-token mask field is missing from batch")

    full_masks: list[torch.Tensor] = []
    sliced_masks: list[torch.Tensor] = []
    for idx, (value, log_prob, total_length, response_length) in enumerate(
        zip(values, log_prob_chunks, total_lengths, response_lengths, strict=False)
    ):
        if isinstance(value, torch.Tensor):
            full_mask = value.to(device=device, dtype=torch.float32).flatten()
        else:
            full_mask = torch.tensor(value, device=device, dtype=torch.float32).flatten()

        if full_mask.numel() != int(response_length):
            raise ValueError(
                "response mask length mismatch for sample "
                f"{idx}: got {full_mask.numel()}, expected {int(response_length)}"
            )

        max_seq_len = max_seq_lens[idx] if max_seq_lens is not None else None
        padded_total_length = padded_total_lengths[idx] if padded_total_lengths is not None else None
        sliced = slice_log_prob_with_cp(
            full_mask,
            int(total_length),
            int(response_length),
            getattr(args, "qkv_format", "thd"),
            max_seq_len,
            padded_total_length,
        )
        if not isinstance(sliced, torch.Tensor):
            sliced = torch.tensor(sliced, device=device, dtype=torch.float32)
        else:
            sliced = sliced.to(device=device, dtype=torch.float32)
        if sliced.numel() != log_prob.numel():
            raise ValueError(
                "sliced response mask length mismatch for sample "
                f"{idx}: got {sliced.numel()}, expected log_probs={log_prob.numel()}"
            )
        full_masks.append(full_mask)
        sliced_masks.append(sliced)

    return full_masks, sliced_masks


def _build_action_token_weights(
    *,
    args: Namespace,
    batch: dict[str, Any],
    log_prob_chunks: list[torch.Tensor],
    shadow_weights: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total_lengths = batch["total_lengths"]
    response_lengths = batch["response_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)
    padded_total_lengths = batch.get("padded_total_lengths", None)

    action_full_masks, action_chunks = _response_mask_chunks(
        batch.get("shadow_action_loss_masks"),
        args=args,
        log_prob_chunks=log_prob_chunks,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        device=device,
        max_seq_lens=max_seq_lens,
        padded_total_lengths=padded_total_lengths,
    )
    loss_full_masks, loss_chunks = _response_mask_chunks(
        batch.get("loss_masks"),
        args=args,
        log_prob_chunks=log_prob_chunks,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        device=device,
        max_seq_lens=max_seq_lens,
        padded_total_lengths=padded_total_lengths,
    )

    renormalize = _env_bool("RELAX_SHADOW_ACTION_MASK_RENORMALIZE", True)
    effective_chunks: list[torch.Tensor] = []
    action_token_counts: list[torch.Tensor] = []
    full_token_counts: list[torch.Tensor] = []
    for action_full, loss_full, action_chunk, loss_chunk in zip(
        action_full_masks, loss_full_masks, action_chunks, loss_chunks, strict=False
    ):
        action_full = (action_full > 0).to(dtype=torch.float32) * (loss_full > 0).to(dtype=torch.float32)
        action_chunk = (action_chunk > 0).to(dtype=torch.float32) * (loss_chunk > 0).to(dtype=torch.float32)
        action_count = action_full.sum()
        full_count = (loss_full > 0).to(dtype=torch.float32).sum()
        if renormalize:
            scale = torch.where(action_count > 0, full_count / action_count.clamp_min(1), action_count)
            action_chunk = action_chunk * scale
        effective_chunks.append(action_chunk)
        action_token_counts.append(action_count)
        full_token_counts.append(full_count)

    action_token_mask = torch.cat(effective_chunks, dim=0) if effective_chunks else torch.empty(0, device=device)
    action_counts = torch.stack(action_token_counts) if action_token_counts else torch.empty(0, device=device)
    full_counts = torch.stack(full_token_counts) if full_token_counts else torch.empty(0, device=device)
    bc_candidates = (shadow_weights > 0).to(dtype=torch.float32)
    denom = bc_candidates.sum().clamp_min(1)
    action_frac_per_sample = action_counts / full_counts.clamp_min(1)

    metrics = {
        "hybrid_shadow_action_mask_enabled": torch.ones((), device=device),
        "hybrid_shadow_action_mask_renorm": torch.tensor(float(renormalize), device=device),
        "hybrid_shadow_action_token_frac": (action_frac_per_sample * bc_candidates).sum() / denom,
        "hybrid_shadow_action_token_count": (action_counts * bc_candidates).sum() / denom,
        "hybrid_shadow_action_zero_frac": (((action_counts <= 0).to(torch.float32) * bc_candidates).sum() / denom),
    }
    return action_token_mask, metrics


def _default_hard_span_metrics(device: torch.device) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), device=device)
    return {
        "hard_span/mask_enabled": zero,
        "hard_span/mask_renorm": zero,
        "hard_span/token_frac": zero,
        "hard_span/token_count": zero,
        "hard_span/action_token_count": zero,
        "hard_span/reasoning_token_count": zero,
        "hard_span/final_token_count": zero,
        "hard_span/excluded_skill_token_count": zero,
        "hard_span/excluded_think_token_count": zero,
        "hard_span/excluded_tool_response_token_count": zero,
        "hard_span/span_count": zero,
        "hard_span/contaminated_action_span_count": zero,
        "hard_span/zero_frac": zero,
        "hard_span/recomputed_token_frac": zero,
        "hard_span/recomputed_token_count": zero,
    }


def _masked_sample_field_mean(
    batch: dict[str, Any],
    field_name: str,
    weights: torch.Tensor,
    *,
    device: torch.device,
    default: torch.Tensor,
) -> torch.Tensor:
    values = batch.get(field_name)
    if values is None:
        return default
    tensor = _cat_field(values, device=device, dtype=torch.float32)
    if tensor.numel() != weights.numel():
        raise ValueError(f"{field_name} count mismatch: got {tensor.numel()}, expected {weights.numel()}")
    return _masked_mean(tensor, weights, default=default)


def _build_hard_span_token_weights(
    *,
    args: Namespace,
    batch: dict[str, Any],
    log_prob_chunks: list[torch.Tensor],
    shadow_weights: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total_lengths = batch["total_lengths"]
    response_lengths = batch["response_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)
    padded_total_lengths = batch.get("padded_total_lengths", None)

    hard_full_masks, hard_chunks = _response_mask_chunks(
        batch.get("hard_span_loss_masks"),
        args=args,
        log_prob_chunks=log_prob_chunks,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        device=device,
        max_seq_lens=max_seq_lens,
        padded_total_lengths=padded_total_lengths,
    )
    loss_full_masks, loss_chunks = _response_mask_chunks(
        batch.get("loss_masks"),
        args=args,
        log_prob_chunks=log_prob_chunks,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        device=device,
        max_seq_lens=max_seq_lens,
        padded_total_lengths=padded_total_lengths,
    )

    renormalize = _env_bool("RELAX_HARD_SPAN_MASK_RENORMALIZE", True)
    effective_chunks: list[torch.Tensor] = []
    hard_token_counts: list[torch.Tensor] = []
    full_token_counts: list[torch.Tensor] = []
    for hard_full, loss_full, hard_chunk, loss_chunk in zip(
        hard_full_masks, loss_full_masks, hard_chunks, loss_chunks, strict=False
    ):
        hard_full = (hard_full > 0).to(dtype=torch.float32) * (loss_full > 0).to(dtype=torch.float32)
        hard_chunk = (hard_chunk > 0).to(dtype=torch.float32) * (loss_chunk > 0).to(dtype=torch.float32)
        hard_count = hard_full.sum()
        full_count = (loss_full > 0).to(dtype=torch.float32).sum()
        if renormalize:
            scale = torch.where(hard_count > 0, full_count / hard_count.clamp_min(1), hard_count)
            hard_chunk = hard_chunk * scale
        effective_chunks.append(hard_chunk)
        hard_token_counts.append(hard_count)
        full_token_counts.append(full_count)

    hard_token_mask = torch.cat(effective_chunks, dim=0) if effective_chunks else torch.empty(0, device=device)
    hard_counts = torch.stack(hard_token_counts) if hard_token_counts else torch.empty(0, device=device)
    full_counts = torch.stack(full_token_counts) if full_token_counts else torch.empty(0, device=device)
    bc_candidates = (shadow_weights > 0).to(dtype=torch.float32)
    denom = bc_candidates.sum().clamp_min(1)
    hard_frac_per_sample = hard_counts / full_counts.clamp_min(1)

    metrics = _default_hard_span_metrics(device)
    metrics.update(
        {
            "hard_span/mask_enabled": torch.ones((), device=device),
            "hard_span/mask_renorm": torch.tensor(float(renormalize), device=device),
            "hard_span/recomputed_token_frac": (hard_frac_per_sample * bc_candidates).sum() / denom,
            "hard_span/recomputed_token_count": (hard_counts * bc_candidates).sum() / denom,
        }
    )
    for field_name, metric_name in (
        ("hard_span_token_frac", "hard_span/token_frac"),
        ("hard_span_token_count", "hard_span/token_count"),
        ("hard_span_action_token_count", "hard_span/action_token_count"),
        ("hard_span_reasoning_token_count", "hard_span/reasoning_token_count"),
        ("hard_span_final_token_count", "hard_span/final_token_count"),
        ("hard_span_excluded_skill_token_count", "hard_span/excluded_skill_token_count"),
        ("hard_span_excluded_think_token_count", "hard_span/excluded_think_token_count"),
        ("hard_span_excluded_tool_response_token_count", "hard_span/excluded_tool_response_token_count"),
        ("hard_span_span_count", "hard_span/span_count"),
        ("hard_span_contaminated_action_span_count", "hard_span/contaminated_action_span_count"),
        ("hard_span_zero_frac", "hard_span/zero_frac"),
    ):
        metrics[metric_name] = _masked_sample_field_mean(
            batch,
            field_name,
            bc_candidates,
            device=device,
            default=metrics[metric_name],
        )
    return hard_token_mask, metrics


def _default_compat_metrics(device: torch.device) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), device=device)
    return {
        "compat/bc_traj_weight_mean": zero,
        "compat/bc_traj_easy_frac": zero,
        "compat/bc_traj_mid_frac": zero,
        "compat/bc_traj_ood_frac": zero,
        "compat/bc_action_ood_frac": zero,
        "compat/teacher_action_nll_current_mean": zero,
        "compat/teacher_action_nll_current_p50": zero,
        "compat/teacher_action_nll_vs_sft_delta": zero,
        "compat/teacher_action_nll_sft_baseline_present": zero,
        "compat/teacher_action_nll_vs_previous_ckpt_delta": zero,
        "compat/no_skill_good_action_nll_current": zero,
        "compat/no_skill_good_action_nll_delta_vs_prev": zero,
        "compat/no_skill_good_bad_margin": zero,
        "compat/fixed_grpo_surrogate": zero,
        "compat/bc_traj_weighted_token_frac": zero,
    }


def _optional_response_mask(
    *,
    args: Namespace,
    batch: dict[str, Any],
    field_name: str,
    log_prob_chunks: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor | None:
    values = batch.get(field_name)
    if values is None:
        return None
    _, chunks = _response_mask_chunks(
        values,
        args=args,
        log_prob_chunks=log_prob_chunks,
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        device=device,
        max_seq_lens=batch.get("max_seq_lens", None),
        padded_total_lengths=batch.get("padded_total_lengths", None),
    )
    return torch.cat([(chunk > 0).to(dtype=torch.float32) for chunk in chunks], dim=0) if chunks else None


def _build_compat_traj_token_weights(
    *,
    args: Namespace,
    batch: dict[str, Any],
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    log_prob_chunks: list[torch.Tensor],
    shadow_weights: torch.Tensor,
    grpo_weights: torch.Tensor,
    rewards: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compatibility-weight full teacher trajectories for oracle-shadow BC.

    ``rollout_log_probs`` are the teacher/oracle-prompt logprobs on the same
    response tokens. ``log_probs`` are the current actor logprobs after prompt
    cleaning, i.e. under the no-skill prompt. Large positive gaps indicate
    tokens that are much more teacher-specific than the student can support.
    """

    teacher_log_probs = _cat_field(batch.get("rollout_log_probs"), device=device, dtype=torch.float32)
    if teacher_log_probs.numel() != log_probs.numel():
        raise ValueError(
            "CompatTraj-BC requires rollout_log_probs aligned with current log_probs: "
            f"teacher={teacher_log_probs.numel()} current={log_probs.numel()}"
        )
    if old_log_probs.numel() != log_probs.numel():
        raise ValueError(
            "CompatTraj-BC requires old log_probs aligned with current log_probs: "
            f"old={old_log_probs.numel()} current={log_probs.numel()}"
        )

    _, loss_chunks = _response_mask_chunks(
        batch.get("loss_masks"),
        args=args,
        log_prob_chunks=log_prob_chunks,
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        device=device,
        max_seq_lens=batch.get("max_seq_lens", None),
        padded_total_lengths=batch.get("padded_total_lengths", None),
    )
    loss_token_mask = (
        torch.cat([(chunk > 0).to(dtype=torch.float32) for chunk in loss_chunks], dim=0)
        if loss_chunks
        else torch.empty(0, device=device, dtype=torch.float32)
    )
    if loss_token_mask.numel() != log_probs.numel():
        raise ValueError(
            "CompatTraj-BC loss mask length mismatch: "
            f"mask={loss_token_mask.numel()} current={log_probs.numel()}"
        )

    gap_low = _env_float("RELAX_COMPAT_GAP_LOW", 0.25)
    gap_high = _env_float("RELAX_COMPAT_GAP_HIGH", 4.0)
    if gap_high <= gap_low:
        raise ValueError(f"RELAX_COMPAT_GAP_HIGH must be > RELAX_COMPAT_GAP_LOW, got {gap_high} <= {gap_low}")
    low_weight = _env_float("RELAX_COMPAT_LOW_WEIGHT", 0.25)
    mid_weight = _env_float("RELAX_COMPAT_MID_WEIGHT", 1.0)
    ood_weight = _env_float("RELAX_COMPAT_OOD_WEIGHT", 0.0)
    renormalize = _env_bool("RELAX_COMPAT_RENORMALIZE", True)

    gap = teacher_log_probs.detach() - log_probs.detach().to(torch.float32)
    compat_chunks: list[torch.Tensor] = []
    cursor = 0
    for loss_chunk in loss_chunks:
        n_tokens = int(loss_chunk.numel())
        local_gap = gap[cursor : cursor + n_tokens]
        valid = (loss_chunk > 0).to(dtype=torch.float32)
        local_weight = torch.where(
            local_gap > gap_high,
            torch.full_like(local_gap, ood_weight),
            torch.where(
                local_gap > gap_low,
                torch.full_like(local_gap, mid_weight),
                torch.full_like(local_gap, low_weight),
            ),
        )
        local_weight = local_weight * valid
        if renormalize:
            valid_count = valid.sum()
            weight_sum = local_weight.sum()
            scale = torch.where(weight_sum > 0, valid_count / weight_sum.clamp_min(1e-12), weight_sum)
            local_weight = local_weight * scale
        compat_chunks.append(local_weight)
        cursor += n_tokens

    compat_weights = (
        torch.cat(compat_chunks, dim=0) if compat_chunks else torch.empty(0, device=device, dtype=torch.float32)
    )
    shadow_token_mask = (
        (_expand_sample_weights((shadow_weights > 0).to(dtype=torch.float32), log_prob_chunks) > 0).to(torch.float32)
        * loss_token_mask
    )
    grpo_token_mask = (
        (_expand_sample_weights((grpo_weights > 0).to(dtype=torch.float32), log_prob_chunks) > 0).to(torch.float32)
        * loss_token_mask
    )

    easy_mask = (gap <= gap_low).to(dtype=torch.float32) * shadow_token_mask
    mid_mask = ((gap > gap_low) & (gap <= gap_high)).to(dtype=torch.float32) * shadow_token_mask
    ood_mask = (gap > gap_high).to(dtype=torch.float32) * shadow_token_mask
    compat_shadow_weights = compat_weights.detach() * shadow_token_mask
    shadow_token_denom = shadow_token_mask.sum().clamp_min(1e-12)

    metrics = _default_compat_metrics(device)
    metrics.update(
        {
            "compat/bc_traj_weight_mean": _masked_mean(
                compat_weights.detach(),
                shadow_token_mask,
                default=metrics["compat/bc_traj_weight_mean"],
            ),
            "compat/bc_traj_easy_frac": easy_mask.sum() / shadow_token_denom,
            "compat/bc_traj_mid_frac": mid_mask.sum() / shadow_token_denom,
            "compat/bc_traj_ood_frac": ood_mask.sum() / shadow_token_denom,
        }
    )

    action_mask = _optional_response_mask(
        args=args,
        batch=batch,
        field_name="compat_action_loss_masks",
        log_prob_chunks=log_prob_chunks,
        device=device,
    )
    if action_mask is not None and action_mask.numel() != log_probs.numel():
        raise ValueError(
            "CompatTraj action monitor mask length mismatch: "
            f"mask={action_mask.numel()} current={log_probs.numel()}"
        )

    current_nll = -log_probs.detach().to(torch.float32)
    previous_nll = -old_log_probs.detach().to(torch.float32)
    if action_mask is not None:
        bc_action_mask = shadow_token_mask * action_mask
        bc_action_current = _masked_mean(
            current_nll,
            bc_action_mask,
            default=metrics["compat/teacher_action_nll_current_mean"],
        )
        bc_action_prev = _masked_mean(
            previous_nll,
            bc_action_mask,
            default=bc_action_current.detach(),
        )
        metrics["compat/teacher_action_nll_current_mean"] = bc_action_current
        metrics["compat/teacher_action_nll_current_p50"] = _masked_median(
            current_nll,
            bc_action_mask,
            default=metrics["compat/teacher_action_nll_current_p50"],
        )
        metrics["compat/teacher_action_nll_vs_previous_ckpt_delta"] = bc_action_current - bc_action_prev
        metrics["compat/bc_action_ood_frac"] = (ood_mask * action_mask).sum() / bc_action_mask.sum().clamp_min(1e-12)

        sft_baseline = _env_optional_float("RELAX_COMPAT_TEACHER_ACTION_NLL_SFT_BASELINE")
        if sft_baseline is not None:
            metrics["compat/teacher_action_nll_vs_sft_delta"] = bc_action_current - torch.tensor(
                float(sft_baseline), device=device, dtype=torch.float32
            )
            metrics["compat/teacher_action_nll_sft_baseline_present"] = torch.ones((), device=device)

        reward_threshold = _env_float("RELAX_COMPAT_REWARD_THRESHOLD", 1.0)
        good_samples = ((grpo_weights > 0) & (rewards >= reward_threshold)).to(dtype=torch.float32)
        bad_samples = ((grpo_weights > 0) & (rewards < reward_threshold)).to(dtype=torch.float32)
        good_token_mask = _expand_sample_weights(good_samples, log_prob_chunks) * loss_token_mask * action_mask
        bad_token_mask = _expand_sample_weights(bad_samples, log_prob_chunks) * loss_token_mask * action_mask
        good_current = _masked_mean(
            current_nll,
            good_token_mask,
            default=metrics["compat/no_skill_good_action_nll_current"],
        )
        good_previous = _masked_mean(previous_nll, good_token_mask, default=good_current.detach())
        bad_current = _masked_mean(current_nll, bad_token_mask, default=good_current.detach())
        metrics["compat/no_skill_good_action_nll_current"] = good_current
        metrics["compat/no_skill_good_action_nll_delta_vs_prev"] = good_current - good_previous
        metrics["compat/no_skill_good_bad_margin"] = bad_current - good_current

        grpo_action_mask = grpo_token_mask * action_mask
        metrics["compat/fixed_grpo_surrogate"] = _masked_mean(
            (-advantages.detach().to(torch.float32)) * log_probs.detach().to(torch.float32),
            grpo_action_mask,
            default=metrics["compat/fixed_grpo_surrogate"],
        )
    else:
        metrics["compat/fixed_grpo_surrogate"] = _masked_mean(
            (-advantages.detach().to(torch.float32)) * log_probs.detach().to(torch.float32),
            grpo_token_mask,
            default=metrics["compat/fixed_grpo_surrogate"],
        )

    # Keep this metric available even when action monitor masks are disabled.
    metrics["compat/bc_traj_weighted_token_frac"] = (
        (compat_shadow_weights > 0).to(dtype=torch.float32).sum() / shadow_token_denom
    )
    return compat_weights, metrics


def hybrid_shadow_grpo_loss(
    args: Namespace,
    batch: dict[str, Any],
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Route no-skill samples to GRPO and oracle-shadow samples to AWR BC."""

    from relax.backends.megatron.loss import compute_approx_kl, get_log_probs_and_entropy
    from relax.utils.training.ppo_utils import compute_policy_loss

    if getattr(args, "use_tis", False):
        raise ValueError(
            "hybrid_shadow_grpo_loss requires RELAX_DISABLE_TIS=1 in the initial implementation"
        )

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
        padded_total_lengths=batch.get("padded_total_lengths", None),
    )

    log_prob_chunks: list[torch.Tensor] = log_probs_and_entropy["log_probs"]
    sample_count = len(log_prob_chunks)
    log_probs = torch.cat(log_prob_chunks, dim=0) if log_prob_chunks else logits.new_empty(0)

    is_shadow = _optional_sample_weights(
        batch.get("hybrid_is_shadow"),
        device=logits.device,
        count=sample_count,
        default=0.0,
    )
    grpo_weights = _optional_sample_weights(
        batch.get("hybrid_grpo_weight"),
        device=logits.device,
        count=sample_count,
        default=1.0 - is_shadow,
    )
    shadow_gate = _optional_sample_weights(
        batch.get("hybrid_shadow_weight"),
        device=logits.device,
        count=sample_count,
        default=is_shadow,
    )

    old_log_probs_source = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]
    old_log_probs = _cat_field(old_log_probs_source, device=logits.device, dtype=torch.float32)
    advantages = _cat_field(batch["advantages"], device=logits.device, dtype=torch.float32)
    if old_log_probs.numel() != log_probs.numel() or advantages.numel() != log_probs.numel():
        raise ValueError(
            "hybrid GRPO token count mismatch: "
            f"log_probs={log_probs.numel()} old={old_log_probs.numel()} advantages={advantages.numel()}"
        )

    ppo_kl_tokens = old_log_probs - log_probs
    pg_loss_tokens, pg_clipfrac_tokens = compute_policy_loss(
        ppo_kl_tokens,
        advantages,
        args.eps_clip,
        args.eps_clip_high,
    )
    grpo_token_weights = _expand_sample_weights(grpo_weights, log_prob_chunks)
    pg_loss = sum_of_sample_mean(pg_loss_tokens * grpo_token_weights)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac_tokens * grpo_token_weights)
    ppo_kl = sum_of_sample_mean(ppo_kl_tokens * grpo_token_weights)

    rewards_source = batch.get("raw_reward")
    if rewards_source is None:
        rewards_source = batch.get("rewards")
    rewards = _as_reward_tensor(rewards_source, logits.device, sample_count)
    awr_weights = _build_weights(rewards, args)
    shadow_weights = shadow_gate * awr_weights
    shadow_token_weights = _expand_sample_weights(shadow_weights, log_prob_chunks)
    compat_metrics: dict[str, torch.Tensor] = {}
    action_metrics: dict[str, torch.Tensor] = {
        "hybrid_shadow_action_mask_enabled": logits.new_zeros(()),
        "hybrid_shadow_action_mask_renorm": logits.new_zeros(()),
        "hybrid_shadow_action_token_frac": logits.new_zeros(()),
        "hybrid_shadow_action_token_count": logits.new_zeros(()),
        "hybrid_shadow_action_zero_frac": logits.new_zeros(()),
    }
    hard_span_metrics = _default_hard_span_metrics(logits.device)
    use_action_mask = _env_bool("RELAX_SHADOW_BC_ACTION_MASK", False)
    use_hard_span_mask = _env_bool("RELAX_SHADOW_BC_HARD_SPAN_MASK", False)
    use_compat_weights = _env_bool("RELAX_SHADOW_BC_COMPAT_WEIGHTS", False)
    if sum(1 for enabled in (use_action_mask, use_hard_span_mask, use_compat_weights) if enabled) > 1:
        raise ValueError(
            "RELAX_SHADOW_BC_ACTION_MASK, RELAX_SHADOW_BC_HARD_SPAN_MASK, and "
            "RELAX_SHADOW_BC_COMPAT_WEIGHTS are mutually exclusive; token-support variants "
            "must be separate experiments."
        )
    if use_compat_weights:
        compat_token_weights, compat_metrics = _build_compat_traj_token_weights(
            args=args,
            batch=batch,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            log_prob_chunks=log_prob_chunks,
            shadow_weights=shadow_weights,
            grpo_weights=grpo_weights,
            rewards=rewards,
            device=logits.device,
        )
        if compat_token_weights.numel() != shadow_token_weights.numel():
            raise ValueError(
                "compat-token weight length mismatch: "
                f"compat={compat_token_weights.numel()} weights={shadow_token_weights.numel()}"
            )
        shadow_token_weights = shadow_token_weights * compat_token_weights
    elif use_action_mask:
        action_token_mask, action_metrics = _build_action_token_weights(
            args=args,
            batch=batch,
            log_prob_chunks=log_prob_chunks,
            shadow_weights=shadow_weights,
            device=logits.device,
        )
        if action_token_mask.numel() != shadow_token_weights.numel():
            raise ValueError(
                "action-token mask length mismatch: "
                f"mask={action_token_mask.numel()} weights={shadow_token_weights.numel()}"
            )
        shadow_token_weights = shadow_token_weights * action_token_mask
    elif use_hard_span_mask:
        hard_token_mask, hard_span_metrics = _build_hard_span_token_weights(
            args=args,
            batch=batch,
            log_prob_chunks=log_prob_chunks,
            shadow_weights=shadow_weights,
            device=logits.device,
        )
        if hard_token_mask.numel() != shadow_token_weights.numel():
            raise ValueError(
                "hard-span token mask length mismatch: "
                f"mask={hard_token_mask.numel()} weights={shadow_token_weights.numel()}"
            )
        shadow_token_weights = shadow_token_weights * hard_token_mask
    shadow_bc_loss = -sum_of_sample_mean(log_probs * shadow_token_weights)

    shadow_loss_coef = _env_float("RELAX_SHADOW_LOSS_COEF", 0.1)
    loss = pg_loss + shadow_loss_coef * shadow_bc_loss

    grpo_coeff = (-advantages.detach()) * grpo_token_weights.detach()
    bc_coeff = -shadow_token_weights.detach()
    coeff_dot = (grpo_coeff * bc_coeff).sum()
    coeff_norm = grpo_coeff.norm() * bc_coeff.norm()
    coeff_cosine = torch.where(coeff_norm > 0, coeff_dot / coeff_norm.clamp_min(1e-12), logits.new_zeros(()))
    support_overlap = (
        ((grpo_token_weights > 0) & (shadow_token_weights > 0)).to(dtype=torch.float32).mean()
        if grpo_token_weights.numel() > 0
        else logits.new_zeros(())
    )
    bc_supervised_token_frac = (
        (shadow_token_weights > 0).to(dtype=torch.float32).mean()
        if shadow_token_weights.numel() > 0
        else logits.new_zeros(())
    )
    bc_action_nll_mean = torch.where(
        shadow_token_weights.sum() > 0,
        (-(log_probs.detach()) * shadow_token_weights.detach()).sum() / shadow_token_weights.detach().sum().clamp_min(1e-12),
        logits.new_zeros(()),
    )

    reported_loss: dict[str, torch.Tensor] = {
        "loss": loss.clone().detach(),
        "hybrid_pg_loss": pg_loss.clone().detach(),
        "hybrid_pg_clipfrac": pg_clipfrac.clone().detach(),
        "hybrid_ppo_kl": ppo_kl.clone().detach(),
        "hybrid_shadow_bc_loss": shadow_bc_loss.clone().detach(),
        "hybrid_shadow_loss_coef": logits.new_tensor(shadow_loss_coef),
        "hybrid_shadow_frac": is_shadow.mean().clone().detach(),
        "hybrid_grpo_weight_mean": grpo_weights.mean().clone().detach(),
        "hybrid_shadow_gate_mean": shadow_gate.mean().clone().detach(),
        "hybrid_shadow_weight_mean": shadow_weights.mean().clone().detach(),
        "hybrid_shadow_nonzero_frac": (shadow_weights > 0).to(dtype=torch.float32).mean().clone().detach(),
        "hybrid_shadow_bc_supervised_token_frac": bc_supervised_token_frac.clone().detach(),
        "hybrid_shadow_bc_action_nll_mean": bc_action_nll_mean.clone().detach(),
        "hybrid_grad_proxy_grpo_bc_coeff_cosine": coeff_cosine.clone().detach(),
        "hybrid_grad_proxy_grpo_bc_support_overlap": support_overlap.clone().detach(),
        "hybrid_reward_mean": rewards.mean().clone().detach(),
    }
    reported_loss.update({key: value.clone().detach() for key, value in action_metrics.items()})
    reported_loss.update({key: value.clone().detach() for key, value in hard_span_metrics.items()})
    reported_loss.update({key: value.clone().detach() for key, value in compat_metrics.items()})

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        if ref_log_probs is None:
            raise ValueError("hybrid loss was launched with --use-kl-loss but batch['ref_log_probs'] is missing")
        ref_log_probs = _cat_field(ref_log_probs, device=logits.device, dtype=torch.float32)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)
        loss = loss + args.kl_loss_coef * kl_loss
        reported_loss["loss"] = loss.clone().detach()
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if log_probs.numel() == 0:
        loss = loss + 0 * logits.sum()

    return loss, reported_loss
