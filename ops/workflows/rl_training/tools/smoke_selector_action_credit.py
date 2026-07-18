#!/usr/bin/env python3
"""CPU smoke for selector attribution, group credit, and disjoint token masks."""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import types
from argparse import Namespace
from pathlib import Path
from statistics import mean, stdev

import pandas as pd
from transformers import AutoTokenizer


ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[4])).resolve()
sys.path.insert(0, str(ROOT / "Relax"))
sys.path.insert(0, str(ROOT / "GeneralAgent" / "eval_scripts"))

from examples.agent_bench.selector_action_credit import (  # noqa: E402
    annotate_group_selector_advantages,
    build_train_fields,
    keep_raw_task_reward_nonzero_std,
    post_process_rewards,
    record_assistant_turn,
)
from relax.utils.types import Sample  # noqa: E402


def plain_dict(value):
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, dict):
            return dict(item)
    raise TypeError(type(value))


def plain_list(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def xml_read(skill: str, *, repeats: int = 1) -> str:
    call = (
        "<tool_call>\n<function=read>\n<parameter=path>\n"
        f"/root/.claude/skills/{skill}/SKILL.md\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    return "\n".join([call] * repeats)


def json_read(skill: str) -> str:
    return (
        '<tool_call>{"name":"read","arguments":{"path":'
        f'"/root/.claude/skills/{skill}/SKILL.md"'
        "}}</tool_call>"
    )


def make_sample(extra: dict, tokenizer, text: str, score: float, *, dispatches: int) -> Sample:
    text = f"I am choosing the next action.\n{text}\nI will continue with the task."
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    terminator = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if terminator in set(tokenizer.all_special_ids):
        tokens.append(terminator)
    observation_tokens = [900001, 900002, 900003]
    sample = Sample(
        metadata={"extra_info": dict(extra)},
        reward={"score": score, "raw_score": score},
        response_length=len(tokens) + len(observation_tokens),
        loss_mask=[1] * len(tokens) + [0] * len(observation_tokens),
    )
    record_assistant_turn(
        sample,
        response_text=text,
        new_tokens=tokens,
        response_token_start=0,
        tokenizer=tokenizer,
        turn_index=0,
        dispatched_tool_call_count=dispatches,
    )
    return sample


def main() -> None:
    os.environ["RELAX_SELECTOR_ACTION_CREDIT"] = "1"
    data_dir = Path(os.environ["DATA_DIR"])
    model_path = Path(os.environ["MODEL_DIR"]) / os.environ["QWEN35_9B_SFT_SUBDIR"]
    if not (model_path / "config.json").is_file():
        raise SystemExit(
            f"[smoke_selector_action_credit] no model/tokenizer at {model_path} "
            "(missing config.json). Restore or train the init model first — see "
            "`./skillrl show rl.selector-action-credit` — or point "
            "MODEL_DIR/QWEN35_9B_SFT_SUBDIR at any complete HF model directory "
            "(this smoke only needs its tokenizer)."
        )
    frame = pd.read_parquet(data_dir / "train.parquet", columns=["extra_info"])
    extra = plain_dict(frame.iloc[0]["extra_info"])
    oracle = str(extra["slate_gold_name"])
    misleading = str(plain_list(extra["slate_misleading_names"])[0])
    relevant = str(plain_list(extra["slate_relevant_names"])[0])
    irrelevant = str(plain_list(extra["slate_irrelevant_names"])[0])
    unadvertised = "selector-smoke-unadvertised"

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    texts = [
        xml_read(oracle),
        xml_read(misleading),
        "I will solve this directly without opening a skill.",
        xml_read(relevant),
        xml_read(oracle, repeats=2),
        xml_read(irrelevant),
        xml_read(unadvertised),
        json_read(oracle),
    ]
    dispatches = [1, 1, 0, 1, 2, 1, 1, 1]
    scores = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    samples = [
        make_sample(extra, tokenizer, text, score, dispatches=count)
        for text, score, count in zip(texts, scores, dispatches, strict=True)
    ]

    stats = annotate_group_selector_advantages(samples)
    assert stats["active"] == 1.0 and stats["actions"] == 8.0, stats
    assert stats["oracle_actions"] == 3.0 and stats["nonoracle_actions"] == 5.0, stats
    assert math.isclose(stats["baseline"], 3 / 8, abs_tol=1e-12), stats
    assert stats["zero_mean_error"] < 1e-12, stats

    args = Namespace(
        reward_key="score",
        n_samples_per_prompt=8,
        rewards_normalization=True,
        grpo_std_normalization=True,
    )
    filter_result = keep_raw_task_reward_nonzero_std(args, samples)
    assert filter_result.keep, filter_result
    raw, processed = post_process_rewards(args, samples)
    expected = [(value - mean(scores)) / (stdev(scores) + 1e-6) for value in scores]
    assert raw == scores
    assert all(math.isclose(left, right, abs_tol=1e-12) for left, right in zip(processed, expected, strict=True))

    base_masks = [sample.loss_mask for sample in samples]
    fields = build_train_fields(samples, base_masks)
    base_total = sum(sum(mask) for mask in base_masks)
    task_total = sum(sum(mask) for mask in fields["selector_task_loss_weights"])
    selector_total = sum(sum(mask) for mask in fields["selector_action_loss_weights"])
    assert math.isclose(task_total, base_total, rel_tol=1e-9, abs_tol=1e-3)
    assert math.isclose(selector_total, base_total, rel_tol=1e-9, abs_tol=1e-3)

    for sample, task, selector, advantage in zip(
        samples,
        fields["selector_task_loss_weights"],
        fields["selector_action_loss_weights"],
        fields["selector_action_advantages"],
        strict=True,
    ):
        state = sample.metadata["selector_action_credit"]
        call_tokens = {
            index for action in state["actions"] for index in action["call_token_indices"]
        }
        identity_tokens = {
            index for action in state["actions"] for index in action["identity_token_indices"]
        }
        assert all(task[index] == 0 for index in call_tokens)
        assert all(selector[index] > 0 for index in identity_tokens)
        assert all(not (task_value > 0 and selector_value > 0) for task_value, selector_value in zip(task, selector))
        for index in range(sample.response_length - 3, sample.response_length):
            assert sample.loss_mask[index] == task[index] == selector[index] == advantage[index] == 0

    # CP2 must partition each full response mask without dropping or
    # duplicating weighted selector/task support.
    import torch

    fake_mpu = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
    )
    fake_megatron = types.ModuleType("megatron")
    fake_megatron_core = types.ModuleType("megatron.core")
    fake_megatron_core.mpu = fake_mpu
    fake_megatron.core = fake_megatron_core
    sys.modules.setdefault("megatron", fake_megatron)
    sys.modules.setdefault("megatron.core", fake_megatron_core)

    with contextlib.redirect_stdout(sys.stderr):
        from relax.backends.megatron.cp_utils import slice_log_prob_with_cp

    original_cp_size = fake_mpu.get_context_parallel_world_size
    original_cp_rank = fake_mpu.get_context_parallel_rank
    try:
        fake_mpu.get_context_parallel_world_size = lambda: 2
        for field_name in ("selector_task_loss_weights", "selector_action_loss_weights"):
            for sample, full_mask in zip(samples, fields[field_name], strict=True):
                rank_slices = []
                for rank in (0, 1):
                    fake_mpu.get_context_parallel_rank = lambda rank=rank: rank
                    rank_slices.append(
                        slice_log_prob_with_cp(
                            torch.tensor(full_mask),
                            sample.response_length + 17,
                            sample.response_length,
                            "thd",
                        )
                    )
                assert sum(piece.numel() for piece in rank_slices) == sample.response_length
                assert math.isclose(
                    sum(float(piece.sum()) for piece in rank_slices),
                    sum(full_mask),
                    rel_tol=1e-6,
                    abs_tol=1e-4,
                )
    finally:
        fake_mpu.get_context_parallel_world_size = original_cp_size
        fake_mpu.get_context_parallel_rank = original_cp_rank

    # Exercise the custom one-forward loss with a differentiable synthetic
    # log-prob tensor. Identity-token gradients must move positive actions up
    # and negative actions down; ordinary task tokens follow task advantage.
    # Some optional Megatron imports print a diagnostic to stdout. Keep the
    # smoke artifact itself strict JSON while retaining that diagnostic on stderr.
    with contextlib.redirect_stdout(sys.stderr):
        from examples.agent_bench.selector_action_grpo_loss import selector_action_grpo_loss
    from relax.utils.training import ppo_utils

    response_lengths = [sample.response_length for sample in samples]
    log_probs = torch.zeros(sum(response_lengths), dtype=torch.float32, requires_grad=True)

    def fake_get_log_probs_and_entropy(_logits, **kwargs):
        return torch.empty(0), {"log_probs": list(log_probs.split(kwargs["response_lengths"]))}

    fake_loss_module = types.ModuleType("relax.backends.megatron.loss")
    fake_loss_module.get_log_probs_and_entropy = fake_get_log_probs_and_entropy
    fake_loss_module.compute_approx_kl = ppo_utils.compute_approx_kl
    original_loss_module = sys.modules.get("relax.backends.megatron.loss")
    original_compute_policy_loss = ppo_utils.compute_policy_loss
    eager_compute_policy_loss = getattr(
        original_compute_policy_loss,
        "__wrapped__",
        getattr(original_compute_policy_loss, "_torchdynamo_orig_callable", original_compute_policy_loss),
    )
    ppo_utils.compute_policy_loss = eager_compute_policy_loss
    sys.modules["relax.backends.megatron.loss"] = fake_loss_module
    try:
        loss_args = Namespace(
            use_tis=False,
            use_rollout_logprobs=True,
            eps_clip=0.2,
            eps_clip_high=0.28,
            use_kl_loss=False,
            qkv_format="thd",
            allgather_cp=False,
            log_probs_chunk_size=4096,
            rollout_temperature=1.0,
        )
        loss_batch = {
            "response_lengths": response_lengths,
            "total_lengths": [length + 1 for length in response_lengths],
            "unconcat_tokens": [torch.zeros(length + 1, dtype=torch.long) for length in response_lengths],
            "rollout_log_probs": [torch.zeros(length) for length in response_lengths],
            "log_probs": [torch.zeros(length) for length in response_lengths],
            "advantages": [
                torch.full((length,), float(task_advantage))
                for length, task_advantage in zip(response_lengths, processed, strict=True)
            ],
            "selector_task_loss_weights": fields["selector_task_loss_weights"],
            "selector_action_loss_weights": fields["selector_action_loss_weights"],
            "selector_action_advantages": fields["selector_action_advantages"],
        }
        combined_loss, loss_metrics = selector_action_grpo_loss(
            loss_args,
            loss_batch,
            log_probs,
            lambda value: value.sum(),
        )
        combined_loss.backward()
    finally:
        if original_loss_module is None:
            sys.modules.pop("relax.backends.megatron.loss", None)
        else:
            sys.modules["relax.backends.megatron.loss"] = original_loss_module
        ppo_utils.compute_policy_loss = original_compute_policy_loss

    offsets = []
    cursor = 0
    for length in response_lengths:
        offsets.append(cursor)
        cursor += length
    positive_selector_gradients = []
    negative_selector_gradients = []
    for sample_index, sample in enumerate(samples):
        for action in sample.metadata["selector_action_credit"]["actions"]:
            advantage = float(action["selector_advantage"])
            gradients = [
                float(log_probs.grad[offsets[sample_index] + token_index])
                for token_index in action["identity_token_indices"]
            ]
            assert gradients and all(gradient * advantage < 0 for gradient in gradients)
            (positive_selector_gradients if advantage > 0 else negative_selector_gradients).extend(gradients)
    task_index = next(
        index
        for index, (task_weight, selector_weight) in enumerate(
            zip(
                fields["selector_task_loss_weights"][0],
                fields["selector_action_loss_weights"][0],
                strict=True,
            )
        )
        if task_weight > 0 and selector_weight == 0
    )
    assert float(log_probs.grad[task_index]) * processed[0] < 0
    assert float(loss_metrics["selector/support_overlap"]) == 0.0

    # Raw-task-variable groups without an oracle read remain valid task GRPO
    # groups, but correctly carry no selector gradient.
    no_oracle_samples = [
        make_sample(extra, tokenizer, xml_read(misleading), score, dispatches=1)
        for score in scores
    ]
    no_oracle_filter = keep_raw_task_reward_nonzero_std(args, no_oracle_samples)
    assert no_oracle_filter.keep, no_oracle_filter
    _, _ = post_process_rewards(args, no_oracle_samples)
    no_oracle_fields = build_train_fields(
        no_oracle_samples, [sample.loss_mask for sample in no_oracle_samples]
    )
    assert sum(map(sum, no_oracle_fields["selector_action_loss_weights"])) == 0

    # Parser/dispatch disagreement must fail closed and make the group ineligible.
    mismatched = [
        make_sample(extra, tokenizer, xml_read(oracle), score, dispatches=(0 if index == 0 else 1))
        for index, score in enumerate(scores)
    ]
    mismatch_filter = keep_raw_task_reward_nonzero_std(args, mismatched)
    assert not mismatch_filter.keep and mismatch_filter.reason.startswith("selector_attribution_error_"), mismatch_filter

    print(
        json.dumps(
            {
                "ok": True,
                "model": str(model_path),
                "oracle": oracle,
                "active_group": stats,
                "base_tokens": base_total,
                "task_weight_sum": task_total,
                "selector_weight_sum": selector_total,
                "combined_loss": float(combined_loss.detach()),
                "positive_selector_gradient_mean": mean(positive_selector_gradients),
                "negative_selector_gradient_mean": mean(negative_selector_gradients),
                "no_oracle_selector_weight_sum": 0.0,
                "mismatch_reason": mismatch_filter.reason,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
