"""One-forward PPO loss with disjoint task and selector token credit."""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import torch

from examples.agent_bench.hybrid_shadow_grpo_loss import _cat_field, _response_mask_chunks


def _coefficient() -> float:
    return float(os.environ.get("RELAX_SELECTOR_ACTION_LOSS_COEF", "0.2"))


def _weighted_sum(
    values: torch.Tensor,
    weights: torch.Tensor,
    reducer: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    return reducer(values.to(torch.float32) * weights.to(torch.float32))


def selector_action_grpo_loss(
    args: Namespace,
    batch: dict[str, Any],
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply task PPO off skill-read calls and selector PPO on identity tokens."""

    from relax.backends.megatron.loss import compute_approx_kl, get_log_probs_and_entropy
    from relax.utils.training.ppo_utils import compute_policy_loss

    if getattr(args, "use_tis", False):
        raise ValueError("selector action credit requires RELAX_DISABLE_TIS=1")

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
    log_probs = torch.cat(log_prob_chunks, dim=0) if log_prob_chunks else logits.new_empty(0)
    old_source = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]
    old_log_probs = _cat_field(old_source, device=logits.device, dtype=torch.float32)
    task_advantages = _cat_field(batch["advantages"], device=logits.device, dtype=torch.float32)
    if old_log_probs.numel() != log_probs.numel() or task_advantages.numel() != log_probs.numel():
        raise ValueError(
            "selector task token count mismatch: "
            f"current={log_probs.numel()} old={old_log_probs.numel()} adv={task_advantages.numel()}"
        )

    slice_kwargs = dict(
        args=args,
        log_prob_chunks=log_prob_chunks,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        device=logits.device,
        max_seq_lens=batch.get("max_seq_lens", None),
        padded_total_lengths=batch.get("padded_total_lengths", None),
    )
    _, task_weight_chunks = _response_mask_chunks(batch.get("selector_task_loss_weights"), **slice_kwargs)
    _, selector_weight_chunks = _response_mask_chunks(batch.get("selector_action_loss_weights"), **slice_kwargs)
    _, selector_advantage_chunks = _response_mask_chunks(batch.get("selector_action_advantages"), **slice_kwargs)
    task_weights = torch.cat(task_weight_chunks, dim=0) if task_weight_chunks else logits.new_empty(0)
    selector_weights = torch.cat(selector_weight_chunks, dim=0) if selector_weight_chunks else logits.new_empty(0)
    selector_advantages = (
        torch.cat(selector_advantage_chunks, dim=0) if selector_advantage_chunks else logits.new_empty(0)
    )
    for name, value in (
        ("task_weights", task_weights),
        ("selector_weights", selector_weights),
        ("selector_advantages", selector_advantages),
    ):
        if value.numel() != log_probs.numel():
            raise ValueError(
                f"selector {name} token count mismatch: {value.numel()} != {log_probs.numel()}"
            )
    if torch.any((task_weights > 0) & (selector_weights > 0)):
        raise ValueError("selector/task loss support overlap reached actor")

    ppo_kl_tokens = old_log_probs - log_probs
    task_pg_tokens, task_clip_tokens = compute_policy_loss(
        ppo_kl_tokens,
        task_advantages,
        args.eps_clip,
        args.eps_clip_high,
    )
    selector_pg_tokens, selector_clip_tokens = compute_policy_loss(
        ppo_kl_tokens,
        selector_advantages,
        args.eps_clip,
        args.eps_clip_high,
    )
    task_pg_loss = _weighted_sum(task_pg_tokens, task_weights, sum_of_sample_mean)
    selector_pg_loss = _weighted_sum(selector_pg_tokens, selector_weights, sum_of_sample_mean)
    selector_coef = _coefficient()
    loss = task_pg_loss + selector_coef * selector_pg_loss

    zero = logits.new_zeros(())
    selector_weight_sum = selector_weights.sum()
    task_weight_sum = task_weights.sum()
    positive_weights = selector_weights * (selector_advantages > 0).to(torch.float32)
    negative_weights = selector_weights * (selector_advantages < 0).to(torch.float32)
    ratio = torch.exp(-ppo_kl_tokens.detach().to(torch.float32))

    def local_weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.where(
            weights.sum() > 0,
            (values.to(torch.float32) * weights).sum() / weights.sum().clamp_min(1e-12),
            zero,
        )

    task_coeff = (-task_advantages.detach()) * task_weights.detach()
    selector_coeff = (-selector_advantages.detach()) * selector_weights.detach() * selector_coef
    reported: dict[str, torch.Tensor] = {
        "loss": loss.detach(),
        "selector/task_pg_loss": task_pg_loss.detach(),
        "selector/pg_loss": selector_pg_loss.detach(),
        "selector/loss_coef": logits.new_tensor(selector_coef),
        "selector/task_clipfrac": _weighted_sum(task_clip_tokens, task_weights, sum_of_sample_mean).detach(),
        "selector/clipfrac": _weighted_sum(
            selector_clip_tokens, selector_weights, sum_of_sample_mean
        ).detach(),
        "selector/task_weight_sum": task_weight_sum.detach(),
        "selector/action_weight_sum": selector_weight_sum.detach(),
        "selector/positive_ratio": local_weighted_mean(ratio, positive_weights).detach(),
        "selector/negative_ratio": local_weighted_mean(ratio, negative_weights).detach(),
        "selector/positive_weight_frac": (positive_weights.sum() / selector_weight_sum.clamp_min(1e-12)).detach(),
        "selector/negative_weight_frac": (negative_weights.sum() / selector_weight_sum.clamp_min(1e-12)).detach(),
        "selector/task_coeff_norm": task_coeff.norm().detach(),
        "selector/weighted_coeff_norm": selector_coeff.norm().detach(),
        "selector/weighted_to_task_coeff_norm": (
            selector_coeff.norm() / task_coeff.norm().clamp_min(1e-12)
        ).detach(),
        "selector/support_overlap": (
            ((task_weights > 0) & (selector_weights > 0)).to(torch.float32).mean()
            if task_weights.numel()
            else zero
        ).detach(),
    }

    if args.use_kl_loss:
        ref_log_probs = batch.get("ref_log_probs")
        if ref_log_probs is None:
            raise ValueError("selector loss launched with KL but ref_log_probs is missing")
        ref_log_probs = _cat_field(ref_log_probs, device=logits.device, dtype=torch.float32)
        if ref_log_probs.numel() != log_probs.numel():
            raise ValueError("selector reference logprob token count mismatch")
        importance_ratio = torch.exp(log_probs - old_log_probs) if args.use_unbiased_kl else None
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)
        loss = loss + args.kl_loss_coef * kl_loss
        reported["loss"] = loss.detach()
        reported["kl_loss"] = kl_loss.detach()

    if log_probs.numel() == 0:
        loss = loss + 0 * logits.sum()
    return loss, reported


__all__ = ["selector_action_grpo_loss"]
