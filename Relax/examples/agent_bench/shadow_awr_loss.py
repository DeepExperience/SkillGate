"""Weighted BC/AWR loss for skill-free shadow trajectories.

This loss is used by the M1 shadow update: rollouts may use an oracle skill for
exploration, rollout.py cleans the resulting trajectory to a no-skill transcript,
and the actor is trained by weighted behavior cloning instead of treating the
cleaned transcript as on-policy GRPO data.
"""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import torch


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _as_reward_tensor(values: Any, device: torch.device, count: int) -> torch.Tensor:
    if values is None:
        raise ValueError("shadow AWR loss requires batch['raw_reward'] or batch['rewards']")

    if isinstance(values, torch.Tensor):
        rewards = values.detach().to(device=device, dtype=torch.float32).flatten()
    else:
        converted: list[float] = []
        for value in values:
            if isinstance(value, torch.Tensor):
                converted.append(float(value.detach().float().mean().cpu().item()))
            else:
                converted.append(float(value))
        rewards = torch.tensor(converted, dtype=torch.float32, device=device)

    if rewards.numel() != count:
        raise ValueError(f"shadow AWR reward count mismatch: got {rewards.numel()}, expected {count}")
    return rewards


def _group_centered_advantages(rewards: torch.Tensor, n_samples_per_prompt: int) -> torch.Tensor:
    if rewards.numel() == 0:
        return rewards

    group_size = max(int(n_samples_per_prompt or 1), 1)
    pieces: list[torch.Tensor] = []
    for start in range(0, rewards.numel(), group_size):
        group = rewards[start : start + group_size]
        mean = group.mean()
        if group.numel() > 1:
            std = group.std(unbiased=False).clamp_min(1e-6)
            pieces.append((group - mean) / std)
        else:
            pieces.append(group - mean)
    return torch.cat(pieces, dim=0)


def _build_weights(rewards: torch.Tensor, args: Namespace) -> torch.Tensor:
    mode = os.environ.get("RELAX_SHADOW_AWR_WEIGHT_MODE", "reward").strip().lower()
    max_weight = _env_float("RELAX_SHADOW_AWR_MAX_WEIGHT", 1.0)
    temperature = max(_env_float("RELAX_SHADOW_AWR_TEMPERATURE", 1.0), 1e-6)

    if mode in {"reward", "raw_reward", "linear"}:
        min_reward = _env_float("RELAX_SHADOW_AWR_MIN_REWARD", 0.0)
        weights = torch.clamp(rewards, min=0.0)
        weights = torch.where(rewards >= min_reward, weights, torch.zeros_like(weights))
    elif mode in {"binary", "success"}:
        min_reward = _env_float("RELAX_SHADOW_AWR_MIN_REWARD", 1.0)
        weights = (rewards >= min_reward).to(dtype=torch.float32)
    elif mode in {"advantage_exp", "awr", "exp"}:
        min_reward = _env_float("RELAX_SHADOW_AWR_MIN_REWARD", 1.0)
        advantages = _group_centered_advantages(rewards, getattr(args, "n_samples_per_prompt", 1))
        weights = torch.exp(advantages / temperature)
        if _env_bool("RELAX_SHADOW_AWR_ZERO_BELOW_MIN", True):
            weights = torch.where(rewards >= min_reward, weights, torch.zeros_like(weights))
    else:
        raise ValueError(
            "Unknown RELAX_SHADOW_AWR_WEIGHT_MODE="
            f"{mode!r}; expected reward, binary, or advantage_exp"
        )

    if max_weight > 0:
        weights = weights.clamp(max=max_weight)

    if _env_bool("RELAX_SHADOW_AWR_NORMALIZE_WEIGHTS", False):
        denom = weights[weights > 0].mean() if torch.any(weights > 0) else None
        if denom is not None:
            weights = weights / denom.clamp_min(1e-6)
            if max_weight > 0:
                weights = weights.clamp(max=max_weight)

    return weights


def shadow_weighted_bc_loss(
    args: Namespace,
    batch: dict[str, Any],
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute weighted BC/AWR over response tokens in cleaned shadow data."""

    from relax.backends.megatron.loss import compute_approx_kl, get_log_probs_and_entropy

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
    rewards_source = batch.get("raw_reward")
    if rewards_source is None:
        rewards_source = batch.get("rewards")
    rewards = _as_reward_tensor(rewards_source, logits.device, len(log_prob_chunks))
    weights = _build_weights(rewards, args)

    token_weights = torch.cat(
        [
            torch.ones_like(log_prob, dtype=torch.float32) * weight
            for log_prob, weight in zip(log_prob_chunks, weights, strict=False)
        ],
        dim=0,
    )
    log_probs = torch.cat(log_prob_chunks, dim=0)
    bc_loss = -sum_of_sample_mean(log_probs * token_weights)
    loss = bc_loss

    reported_loss = {
        "loss": loss.clone().detach(),
        "shadow_bc_loss": bc_loss.clone().detach(),
        "shadow_reward_mean": rewards.mean().clone().detach(),
        "shadow_weight_mean": weights.mean().clone().detach(),
        "shadow_weight_nonzero_frac": (weights > 0).to(dtype=torch.float32).mean().clone().detach(),
        "shadow_weight_max": weights.max().clone().detach() if weights.numel() > 0 else logits.new_zeros(()),
    }

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        if ref_log_probs is None:
            raise ValueError("shadow AWR loss was launched with --use-kl-loss but batch['ref_log_probs'] is missing")
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        kl = compute_approx_kl(log_probs, ref_log_probs, kl_loss_type=args.kl_loss_type)
        kl_loss = sum_of_sample_mean(kl)
        loss = loss + args.kl_loss_coef * kl_loss
        reported_loss["loss"] = loss.clone().detach()
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if log_probs.numel() == 0:
        loss = loss + 0 * logits.sum()

    return loss, reported_loss
