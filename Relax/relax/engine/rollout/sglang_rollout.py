# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import copy
import inspect
import math
import os
import uuid
from argparse import Namespace
from collections.abc import Callable
from contextlib import contextmanager
from time import monotonic
from typing import Any

import numpy as np
import pybase64
import ray
import sglang_router
import torch
from packaging.version import parse
from tqdm import tqdm

from relax.distributed.ray.rollout import _log_rollout_data
from relax.engine.filters.base_types import MetricGatherer, call_dynamic_filter
from relax.engine.rewards import async_rm, batched_async_rm
from relax.engine.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from relax.utils.async_utils import run
from relax.utils.data.data import Dataset
from relax.utils.data.processing_utils import (
    _ENCODE_EXECUTOR,
    async_encode_audio_for_rollout_engine,
    async_encode_image_for_rollout_engine,
    async_encode_video_tensor_for_rollout_engine,
    load_processor,
    load_tokenizer,
)
from relax.utils.data.processor_pool import ProcessorPool, prepare_mm_inputs_for_ipc, process_sample_in_worker
from relax.utils.http_utils import get, post
from relax.utils.logging_utils import get_logger
from relax.utils.misc import SingletonMeta, load_function
from relax.utils.profile_utils import start_sglang_profile, stop_sglang_profile
from relax.utils.timer import Timer
from relax.utils.training.eval_config import EvalDatasetConfig
from relax.utils.training.train_dump_utils import save_debug_rollout_data
from relax.utils.types import Sample
from relax.utils.utils import CURRENT_ROLLOUT_BATCH, transfer_batch_to_data_system


__all__ = ["generate_rollout"]

logger = get_logger(__name__)

_PAIR_GRPO_UPDATE_KINDS = {"no_skill_grpo", "noskill_grpo", "no_skill", "noskill"}
_PAIR_ORACLE_UPDATE_KINDS = {
    "oracle_prompt_bc",
    "oracle_direct_bc",
    "prompt_shadow",
    "oracle_shadow",
    "shadow",
    "m1_shadow",
    "hybrid_shadow",
}


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %s.", name, raw, default)
        return default


def _sample_extra_info(sample: Sample) -> dict:
    metadata = sample.metadata or {}
    if not isinstance(metadata, dict):
        return {}
    extra = metadata.get("extra_info")
    if isinstance(extra, dict):
        return extra
    return metadata


def _set_sample_extra(sample: Sample, key: str, value: Any) -> None:
    sample.metadata = sample.metadata or {}
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    extra = sample.metadata.get("extra_info")
    if not isinstance(extra, dict):
        extra = sample.metadata
    extra[key] = value
    if extra is not sample.metadata:
        sample.metadata["extra_info"] = extra


def _group_update_kind(group: list[Sample]) -> str:
    if not group:
        return ""
    extra = _sample_extra_info(group[0])
    return str(extra.get("update_kind") or extra.get("hybrid_update_kind") or "").strip().lower()


def _group_task_key(group: list[Sample]) -> str:
    if not group:
        return "unknown/unknown"
    sample = group[0]
    extra = _sample_extra_info(sample)
    label = sample.label if isinstance(sample.label, dict) else {}
    bench = extra.get("bench") or label.get("bench") or "unknown"
    task_id = extra.get("task_id") or label.get("task_id") or sample.index or "unknown"
    return f"{bench}/{task_id}"


def _stable_extra_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _group_pair_key(group: list[Sample]) -> str:
    if not group:
        return "unknown/unknown"
    extra = _sample_extra_info(group[0])
    sft_record_idx = _stable_extra_value(extra.get("sft_record_idx"))
    source = _stable_extra_value(extra.get("source"))
    if sft_record_idx:
        return f"{_group_task_key(group)}|source={source}|sft={sft_record_idx}"
    return _group_task_key(group)


def _group_pair_role(group: list[Sample]) -> str:
    if not group:
        return ""
    extra = _sample_extra_info(group[0])
    role = str(extra.get("relax_pair_role") or "").strip().lower()
    if role:
        return role
    kind = _group_update_kind(group)
    if kind in _PAIR_GRPO_UPDATE_KINDS:
        return "no_skill"
    if kind in _PAIR_ORACLE_UPDATE_KINDS:
        return "oracle"
    if kind in _SLATE_UPDATE_KINDS and _slate_regret_enabled():
        return "slate"
    return ""


def _group_reward_values(args: Namespace, group: list[Sample]) -> list[float]:
    rewards: list[float] = []
    for sample in group:
        reward = sample.reward
        if isinstance(reward, dict) and "raw_score" in reward:
            try:
                rewards.append(float(reward["raw_score"]))
                continue
            except (TypeError, ValueError):
                pass
        rewards.append(float(sample.get_reward_value(args)))
    return rewards


def _pair_pass_threshold() -> float:
    raw = os.environ.get("RELAX_PAIR_BC_PASS_THRESHOLD") or os.environ.get("PASS_REWARD_THRESHOLD") or "1.0"
    try:
        return float(raw)
    except ValueError:
        return 1.0


def _pair_oracle_bc_until_step() -> int:
    """Return first rollout id where oracle-BC pairing is disabled.

    A negative value means unlimited oracle-BC pairing.  With value ``60``,
    rollout ids 0..59 may use oracle BC, and rollout id 60 onward is pure
    no-skill GRPO dynamic sampling.
    """

    raw = os.environ.get("RELAX_PAIR_ORACLE_BC_UNTIL_STEP")
    if raw is None or raw == "":
        return -1
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"RELAX_PAIR_ORACLE_BC_UNTIL_STEP must be an integer, got {raw!r}") from exc


def _pair_oracle_grpo_enabled() -> bool:
    """Accepted oracle groups train with in-group GRPO instead of BC.

    Same pair gate as oracle BC (paired no-skill all-fail + oracle has
    success); only the update rule for the accepted oracle group changes.
    """
    return os.environ.get("RELAX_PAIR_ORACLE_GRPO", "0").lower() in {"1", "true", "yes", "on"}


def _pair_oracle_grpo_cross_arm_adv_enabled() -> bool:
    """Use paired no-skill mean as the oracle-GRPO reward baseline."""

    return os.environ.get("RELAX_PAIR_ORACLE_GRPO_CROSS_ARM_ADV", "0").lower() in {"1", "true", "yes", "on"}


def _pair_oracle_grpo_drop_all_pass_enabled() -> bool:
    """Drop all-pass oracle-GRPO rescue groups instead of training inert slots."""

    return os.environ.get("RELAX_PAIR_ORACLE_GRPO_DROP_ALL_PASS", "0").lower() in {"1", "true", "yes", "on"}


def _opsd_mode_enabled() -> bool:
    """Prompt-swap self-teacher OPD (OPSD): master switch, default off."""
    return os.environ.get("RELAX_OPSD_MODE", "0").lower() in {"1", "true", "yes", "on"}


def _opsd_scope() -> str:
    """Which no-skill groups get teacher-scored: 'mixed' (default) or 'all'."""
    return (os.environ.get("RELAX_OPSD_SCOPE") or "mixed").strip().lower()


_SLATE_UPDATE_KINDS = {"slate_grpo"}


def _slate_regret_enabled() -> bool:
    """SlateRL: mixed-skill slate arm paired with the no-skill arm.

    When on, ``slate_grpo`` rows take the deferred (oracle) slot of the
    pair-atomic machinery, the slate arm is rolled out for EVERY task outcome
    (mixed/all-pass/all-fail no-skill), and the paired no-skill group mean is
    stamped on the slate group for regret advantage shaping downstream
    (examples.agent_bench.slate_regret_gating). Default off: byte-identical
    behavior, ``slate_grpo`` stays an unknown kind.
    """
    return os.environ.get("RELAX_SLATE_REGRET_GRPO", "0").lower() in {"1", "true", "yes", "on"}


def _slate_uniform_min_delta() -> float:
    """Accept a uniform-outcome slate group (all-pass/all-fail) only when
    |slate mean - paired no-skill mean| >= this threshold: such groups carry
    pure group-level regret signal; below the threshold they are dropped like
    zero-variance groups."""
    raw = os.environ.get("RELAX_SLATE_UNIFORM_MIN_DELTA", "0.25")
    try:
        return float(raw)
    except ValueError:
        return 0.25


def _mark_group_extra(group: list[Sample], **values: Any) -> None:
    for sample in group:
        for key, value in values.items():
            _set_sample_extra(sample, key, value)


def _prepare_pair_atomic_candidates(
    samples: list[list[Sample]],
    *,
    rollout_id: int,
    pending_no_skill: dict[str, list[list[Sample]]],
    pending_oracle: dict[str, list[list[Sample]]],
    require_oracle_pair: bool = True,
) -> tuple[list[list[Sample]], dict[str, list[Sample]], dict[str, float]]:
    """Return no-skill first-stage groups and deferred oracle groups.

    Rollout shuffling and streaming data loading can return the two rows of a
    pair in different batches. Keep step-local pending caches keyed by task so
    the no-skill arm is submitted only after its oracle arm is available.

    When ``require_oracle_pair`` is false, oracle rows are dropped before
    rollout and no-skill rows are submitted immediately. This is used after the
    BC warmup horizon to turn the pair dataset into pure no-skill GRPO without
    paying oracle rollout cost.
    """

    no_skill_groups: list[list[Sample]] = []
    oracle_by_pair_id: dict[str, list[Sample]] = {}
    stats = {
        "candidate_groups": float(len(samples)),
        "candidate_pairs": 0.0,
        "candidate_no_skill_groups": 0.0,
        "candidate_oracle_groups": 0.0,
        "candidate_unpaired_no_skill": 0.0,
        "candidate_oracle_without_no_skill": 0.0,
        "pending_no_skill_groups": 0.0,
        "pending_oracle_groups": 0.0,
        "candidate_unknown_kind_groups": 0.0,
    }

    def _pop_pending(bucket: dict[str, list[list[Sample]]], key: str) -> list[Sample] | None:
        queued = bucket.get(key)
        if not queued:
            return None
        group = queued.pop(0)
        if not queued:
            bucket.pop(key, None)
        return group

    def _append_pending(bucket: dict[str, list[list[Sample]]], key: str, group: list[Sample]) -> None:
        bucket.setdefault(key, []).append(group)

    def _ready_pair(no_skill_group: list[Sample], oracle_group: list[Sample], pair_key: str) -> None:
        pair_id = (
            f"{rollout_id}:{pair_key}:"
            f"{getattr(no_skill_group[0], 'group_index', 'g')}:"
            f"{getattr(oracle_group[0], 'group_index', 'o')}"
        )
        oracle_by_pair_id[pair_id] = oracle_group
        task_key = _group_task_key(no_skill_group)
        _mark_group_extra(
            no_skill_group,
            relax_pair_id=pair_id,
            relax_pair_role="no_skill",
            relax_pair_task_key=task_key,
        )
        deferred_role = (
            "slate"
            if _slate_regret_enabled() and _group_update_kind(oracle_group) in _SLATE_UPDATE_KINDS
            else "oracle"
        )
        _mark_group_extra(
            oracle_group,
            relax_pair_id=pair_id,
            relax_pair_role=deferred_role,
            relax_pair_task_key=task_key,
        )
        if _opsd_mode_enabled() and oracle_group and oracle_group[0].prompt:
            # OPSD: the oracle arm is a prompt donor only. Stash its (oracle
            # skill preloaded) prompt on each no-skill sample so the agent
            # rollout can render teacher prompt ids for prompt-swap scoring.
            donor_prompt = oracle_group[0].prompt
            for no_skill_sample in no_skill_group:
                no_skill_sample.metadata = no_skill_sample.metadata or {}
                no_skill_sample.metadata["opsd_teacher_prompt"] = donor_prompt
        no_skill_groups.append(no_skill_group)
        stats["candidate_pairs"] += 1.0

    for group in samples:
        kind = _group_update_kind(group)
        pair_key = _group_pair_key(group)
        if kind in _PAIR_GRPO_UPDATE_KINDS:
            stats["candidate_no_skill_groups"] += 1.0
            if not require_oracle_pair:
                pair_id = f"{rollout_id}:{pair_key}:{getattr(group[0], 'group_index', 'g')}:noskill_only"
                _mark_group_extra(
                    group,
                    relax_pair_id=pair_id,
                    relax_pair_role="no_skill",
                    relax_pair_task_key=_group_task_key(group),
                )
                no_skill_groups.append(group)
                continue
            oracle_group = _pop_pending(pending_oracle, pair_key)
            if oracle_group is not None:
                _ready_pair(group, oracle_group, pair_key)
            else:
                stats["candidate_unpaired_no_skill"] += 1.0
                _append_pending(pending_no_skill, pair_key, group)
        elif kind in _PAIR_ORACLE_UPDATE_KINDS or (
            kind in _SLATE_UPDATE_KINDS and _slate_regret_enabled()
        ):
            stats["candidate_oracle_groups"] += 1.0
            if not require_oracle_pair:
                stats["candidate_oracle_without_no_skill"] += 1.0
                continue
            no_skill_group = _pop_pending(pending_no_skill, pair_key)
            if no_skill_group is not None:
                _ready_pair(no_skill_group, group, pair_key)
            else:
                stats["candidate_oracle_without_no_skill"] += 1.0
                _append_pending(pending_oracle, pair_key, group)
        else:
            # Non-pair rows are treated as normal no-skill candidates so this
            # env-gated mode can fail soft on old data rather than deadlocking.
            pair_id = f"{rollout_id}:{getattr(group[0], 'group_index', 'g')}"
            _mark_group_extra(group, relax_pair_id=pair_id, relax_pair_role="no_skill")
            no_skill_groups.append(group)
            stats["candidate_unknown_kind_groups"] += 1.0
    stats["pending_no_skill_groups"] = float(sum(len(v) for v in pending_no_skill.values()))
    stats["pending_oracle_groups"] = float(sum(len(v) for v in pending_oracle.values()))
    return no_skill_groups, oracle_by_pair_id, stats


class GenerateState(metaclass=SingletonMeta):
    """The global state for the generation process."""

    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        # Process pool for running HuggingFace processor without GIL contention.
        # Controlled by --mm-processor-pool-size (0 = disabled).
        self.processor_pool = None
        if self.processor is not None:
            pool_size = getattr(args, "mm_processor_pool_size", 0)
            if pool_size > 0:
                try:
                    self.processor_pool = ProcessorPool(
                        model_path=args.hf_checkpoint,
                        pool_size=pool_size,
                        trust_remote_code=True,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create ProcessorPool, falling back to ThreadPoolExecutor: {e}")

        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        # dp rank balancing
        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        self.dp_rank = 0
        self.eval_abort_lock = asyncio.Lock()
        self.abort_complete = asyncio.Event()

        self.reset()

    @contextmanager
    def dp_rank_context(self):
        candidates = [i for i, count in enumerate(self.dp_counts) if count == min(self.dp_counts)]
        dp_rank = int(np.random.choice(candidates))
        self.dp_counts[dp_rank] += 1
        self.dp_rank = dp_rank
        try:
            yield dp_rank
        finally:
            self.dp_counts[dp_rank] -= 1
            assert self.dp_counts[dp_rank] >= 0

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings = set()
        self.protected_pendings = (
            set()
        )  # tasks that should not be aborted (abort_count >= partial_rollout_max_aborted_count)
        self.aborted = False
        self.abort_in_progress = False
        self.abort_complete.set()
        self.evaluating = getattr(self, "evaluating", 0)  # preserve eval state across resets
        # Pre-fetched data ObjectRef for cross-step overlap.
        # Persisted across reset() calls so the ref submitted at the end of
        # step N is consumed at the beginning of step N+1.
        if not hasattr(self, "prefetched_samples_ref"):
            self.prefetched_samples_ref: ray.ObjectRef | None = None

    async def wait_for_abort_complete(self, reason: str) -> None:
        logged = False
        while self.abort_in_progress or self.aborted or not self.abort_complete.is_set():
            if not logged:
                logger.info("%s deferred until rollout abort cleanup completes.", reason)
                logged = True
            if self.abort_complete.is_set():
                await asyncio.sleep(0.1)
            else:
                await self.abort_complete.wait()
        if logged:
            logger.info("%s resumed after rollout abort cleanup completed.", reason)

    def submit_generate_group(self, group: list[Sample], *, count_capacity: bool = True) -> asyncio.Task:
        max_aborted_count = getattr(self.args, "partial_rollout_max_aborted_count", None)
        task = asyncio.create_task(
            generate_and_rm_group(
                self.args,
                group,
                sampling_params=self.sampling_params.copy(),
                evaluation=False,
            )
        )
        # If any sample in the group has been aborted >= partial_rollout_max_aborted_count,
        # mark this task as protected so it won't be aborted again.
        if max_aborted_count is not None and any(sample.abort_count >= max_aborted_count for sample in group):
            self.protected_pendings.add(task)
        else:
            self.pendings.add(task)
        if count_capacity:
            self.remaining_batch_size += 1
        return task

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            self.submit_generate_group(group)


async def _run_image_processor(
    state: GenerateState, args: Namespace, prompt: str | list[dict[str, str]], multimodal_inputs: dict
) -> tuple[list[int], dict | None, float]:
    """Run HF processor and return (prompt_ids, mm_train_inputs,
    elapsed_seconds)."""
    t_start = monotonic()
    loop = asyncio.get_running_loop()

    if state.processor_pool is not None:
        mm_inputs_ipc = prepare_mm_inputs_for_ipc(multimodal_inputs)
        processor_kwargs = {
            "use_audio_in_video": args.use_audio_in_video,
            "return_mm_token_type_ids": False,
        }
        processor_prompt_ids, mm_train_inputs = await loop.run_in_executor(
            state.processor_pool.executor,
            process_sample_in_worker,
            prompt,
            mm_inputs_ipc,
            processor_kwargs,
        )
    else:

        def _run_processor():
            processor_output = state.processor(
                text=prompt,
                use_audio_in_video=args.use_audio_in_video,
                return_mm_token_type_ids=False,
                **multimodal_inputs,
            )
            prompt_ids = processor_output["input_ids"][0]
            train_inputs = {
                k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v)
                for k, v in processor_output.items()
                if k not in ["input_ids", "attention_mask"]
            } or None
            return prompt_ids, train_inputs

        processor_prompt_ids, mm_train_inputs = await loop.run_in_executor(_ENCODE_EXECUTOR, _run_processor)

    return processor_prompt_ids, mm_train_inputs, monotonic() - t_start


async def _encode_multimodal_inputs(multimodal_inputs: dict) -> tuple[dict[str, list], float]:
    """Base64-encode multimodal data and return (encoded_data,
    elapsed_seconds)."""
    t_start = monotonic()
    encode_coros = []

    if image_data := multimodal_inputs["images"]:
        encode_coros.extend(async_encode_image_for_rollout_engine(image) for image in image_data)
    image_count = len(image_data) if multimodal_inputs.get("images") else 0

    if video_data := multimodal_inputs["videos"]:
        encode_coros.extend(async_encode_video_tensor_for_rollout_engine(video) for video in video_data)
    video_count = len(video_data) if multimodal_inputs.get("videos") else 0

    if audio_data := multimodal_inputs["audio"]:
        encode_coros.extend(async_encode_audio_for_rollout_engine(audio) for audio in audio_data)

    encoded: dict[str, list] = {}
    if encode_coros:
        results = await asyncio.gather(*encode_coros)
        offset = 0
        if image_count:
            encoded["image_data"] = list(results[offset : offset + image_count])
            offset += image_count
        if video_count:
            encoded["video_data"] = list(results[offset : offset + video_count])
            offset += video_count
        if offset < len(results):
            encoded["audio_data"] = list(results[offset:])

    return encoded, monotonic() - t_start


async def generate(
    args: Namespace, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False
) -> Sample:
    """Generate using traditional SGLang router with token-based workflow."""
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED, (
        f"Sample status is {sample.status}"
    )

    tokenizer_prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)

    _t_image_processor: float | None = None
    if state.processor:
        processor_prompt_ids, sample.multimodal_train_inputs, _t_image_processor = await _run_image_processor(
            state, args, sample.prompt, sample.multimodal_inputs
        )
    else:
        processor_prompt_ids = tokenizer_prompt_ids

    if len(sample.response) > 0:
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(processor_prompt_ids)

    assert sampling_params["max_new_tokens"] >= 0, (
        f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    )
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": not evaluation,
    }

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    _t_mm_encode: float | None = None
    if sample.multimodal_inputs:
        # Use pre-encoded data from group-level de-dup if available; otherwise encode inline.
        pre_encoded = getattr(sample, "_pre_encoded_mm", None)
        if pre_encoded is not None:
            encoded_mm = pre_encoded
            _t_mm_encode = getattr(sample, "_pre_encoded_mm_elapsed", 0.0)
            del sample._pre_encoded_mm
            if hasattr(sample, "_pre_encoded_mm_elapsed"):
                del sample._pre_encoded_mm_elapsed
        else:
            encoded_mm, _t_mm_encode = await _encode_multimodal_inputs(sample.multimodal_inputs)
        payload.update(encoded_mm)

    # Use existing tokens for multi-turn or tokenize the new prompt
    if len(sample.response) > 0:
        payload["input_ids"] = sample.rollout_tokens
    else:
        payload["input_ids"] = tokenizer_prompt_ids
        # Initialize sample.tokens for the first turn
        if not sample.tokens:
            sample.tokens = processor_prompt_ids
        if not sample.rollout_tokens:
            sample.rollout_tokens = tokenizer_prompt_ids

    # Use session_id for consistent hashing routing if router uses consistent_hashing policy
    headers = None
    if args.sglang_router_policy == "consistent_hashing" and sample.session_id:
        headers = {"X-SMG-Routing-Key": sample.session_id}

    _t_generate_start = monotonic()
    output = await post(url, payload, headers=headers)
    _t_generate = monotonic() - _t_generate_start

    _t_post_generate_start = monotonic()
    if args.use_slime_router and "RadixTreeMiddleware" in args.slime_router_middleware_paths:
        from relax.engine.router.middleware.radix_tree_middleware import postprocess_sample_with_radix_tree

        sample = await postprocess_sample_with_radix_tree(args, sample, output)
    else:
        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            new_response_tokens = state.tokenizer.encode(output["text"], add_special_tokens=False)
            new_response_log_probs = []

        while hasattr(state.tokenizer, "image_token_id") and state.tokenizer.image_token_id in new_response_tokens:
            index = new_response_tokens.index(state.tokenizer.image_token_id)
            new_response_tokens[index] = state.tokenizer.pad_token_id
            logger.warning(
                "Image token found in output tokens, replaced with pad_token_id. Consider updating the model's stop condition to stop at image_token_id if you want to avoid this."
            )

        while hasattr(state.tokenizer, "audio_token_id") and state.tokenizer.audio_token_id in new_response_tokens:
            index = new_response_tokens.index(state.tokenizer.audio_token_id)
            new_response_tokens[index] = state.tokenizer.pad_token_id
            logger.warning(
                "Audio token found in output tokens, replaced with pad_token_id. Consider updating the model's stop condition to stop at audio_token_id if you want to avoid this."
            )

        while hasattr(state.tokenizer, "video_token_id") and state.tokenizer.video_token_id in new_response_tokens:
            index = new_response_tokens.index(state.tokenizer.video_token_id)
            new_response_tokens[index] = state.tokenizer.pad_token_id
            logger.warning(
                "Video token found in output tokens, replaced with pad_token_id. Consider updating the model's stop condition to stop at video_token_id if you want to avoid this."
            )

        # Update sample with tokens directly - avoiding re-tokenization
        sample.tokens = sample.tokens + new_response_tokens
        sample.rollout_tokens = sample.rollout_tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response += output["text"]

        # When partial rollout and masking off policy is enabled, update the loss mask
        if sample.loss_mask is not None:
            assert args.partial_rollout and args.mask_offpolicy_in_partial_rollout
            sample.loss_mask += [1] * len(new_response_tokens)

        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += new_response_log_probs

    if "routed_experts" in output["meta_info"]:
        sample.rollout_routed_experts = np.frombuffer(
            pybase64.b64decode(output["meta_info"]["routed_experts"].encode("ascii")),
            dtype=np.int32,
        ).reshape(
            len(sample.tokens) - 1,
            args.num_layers,
            args.moe_router_topk,
        )

    sample.update_from_meta_info(args, output["meta_info"])
    _t_post_generate = monotonic() - _t_post_generate_start

    _timing: dict[str, float] = {"generate": _t_generate, "post_generate": _t_post_generate}
    if _t_image_processor is not None:
        _timing["image_processor"] = _t_image_processor
    if _t_mm_encode is not None:
        _timing["mm_encode"] = _t_mm_encode
    sample.metadata["_timing"] = _timing

    return sample


def apply_soft_overlong_penalty(args: Namespace, sample: Sample) -> None:
    """DAPO-style soft overlong punishment (env-gated, default OFF).

    Adds a length-based negative penalty to ``sample.reward`` in-place when
    ``RELAX_SOFT_OVERLONG_PENALTY=1``. Three-stage piecewise-linear shaping
    (DAPO, arXiv:2503.14476):

        L <= L_max - L_cache              -> no penalty
        L_max - L_cache < L <= L_max      -> linear 0 .. -1
        L > L_max                         -> -1

    L_max defaults to ``args.rollout_max_response_len`` and L_cache to env
    ``RELAX_SOFT_OVERLONG_CACHE`` (default 4096). The penalty is ADDED to the
    existing 0/1 task reward (so a passing-but-overlong sample keeps positive
    signal minus the length cost). Only the scalar reward path is shaped; when
    ``args.reward_key`` selects a dict field, that field is shaped in place.
    """
    if os.environ.get("RELAX_SOFT_OVERLONG_PENALTY") != "1":
        return
    if sample.reward is None:
        return

    l_max = int(os.environ.get("RELAX_SOFT_OVERLONG_LMAX") or getattr(args, "rollout_max_response_len", 0) or 0)
    if l_max <= 0:
        return
    l_cache = int(os.environ.get("RELAX_SOFT_OVERLONG_CACHE") or 4096)
    length = int(sample.response_length or 0)

    threshold = l_max - l_cache
    if length <= threshold:
        return
    if length >= l_max:
        penalty = -1.0
    else:
        penalty = -float(length - threshold) / float(max(l_cache, 1))

    reward_key = getattr(args, "reward_key", None)
    if reward_key:
        if isinstance(sample.reward, dict) and reward_key in sample.reward:
            try:
                sample.reward[reward_key] = float(sample.reward[reward_key]) + penalty
            except (TypeError, ValueError):
                return
    elif isinstance(sample.reward, (int, float)):
        sample.reward = float(sample.reward) + penalty


async def generate_and_rm(
    args: Namespace,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        with state.dp_rank_context() as _:
            # Check sample.generate_function_path for per-sample custom_generate_function_path (e.g., from eval dataset config)
            custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

            if custom_func_path is not None:
                custom_generate_func = load_function(custom_func_path)
                # if signature has evaluation, pass evaluation
                if "evaluation" in inspect.signature(custom_generate_func).parameters:
                    sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
                else:
                    sample = await custom_generate_func(args, sample, sampling_params)
            else:
                sample = await generate(args, sample, sampling_params, evaluation=evaluation)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any(sample.status == Sample.Status.ABORTED for sample in samples):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
            if not evaluation:
                apply_soft_overlong_penalty(args, sample)

        # OPD sglang: fetch teacher log-probs for each sample (independent of reward)
        if getattr(args, "use_opd", False) and getattr(args, "opd_type", None) == "sglang" and not evaluation:
            from relax.engine.rollout.on_policy_distillation import (
                create_teacher_client_session,
                fetch_teacher_log_probs,
            )

            async with create_teacher_client_session(args) as teacher_session:
                await asyncio.gather(*[fetch_teacher_log_probs(args, s, session=teacher_session) for s in samples])

        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        if sample.reward is None:
            sample.reward = await async_rm(args, sample)
        if not evaluation:
            apply_soft_overlong_penalty(args, sample)

        # OPD sglang: fetch teacher log-probs (independent of reward)
        if getattr(args, "use_opd", False) and getattr(args, "opd_type", None) == "sglang" and not evaluation:
            from relax.engine.rollout.on_policy_distillation import (
                create_teacher_client_session,
                fetch_teacher_log_probs,
            )

            async with create_teacher_client_session(args) as teacher_session:
                await fetch_teacher_log_probs(args, sample, session=teacher_session)

    return sample


def _collect_timing_from_samples(samples: list[Sample]) -> dict[str, list[float]]:
    """Extract per-phase timing lists from sample metadata written by
    generate()."""
    collected: dict[str, list[float]] = {}
    for sample in samples:
        timing = sample.metadata.get("_timing")
        if not timing:
            continue
        for key, value in timing.items():
            collected.setdefault(key, []).append(value)
    return collected


def _aggregate_rollout_timing(all_samples: list[Sample], get_samples_times: list[float]) -> dict[str, float]:
    timing_data = _collect_timing_from_samples(all_samples)
    metrics: dict[str, float] = {}

    for phase in ("image_processor", "mm_encode", "generate", "post_generate"):
        values = timing_data.get(phase, [])
        if not values:
            continue
        metrics[f"perf_detail/rollout/{phase}_time/mean"] = sum(values) / len(values)
        metrics[f"perf_detail/rollout/{phase}_time/max"] = max(values)

    if get_samples_times:
        metrics["perf_detail/rollout/get_samples_time/total"] = sum(get_samples_times)
        metrics["perf_detail/rollout/get_samples_time/mean"] = sum(get_samples_times) / len(get_samples_times)

    return metrics


def _sample_used_skill_strict(sample: Sample) -> bool:
    reward = sample.reward if isinstance(sample.reward, dict) else {}
    if reward.get("skill_group_used_strict") == 1.0:
        return True
    metadata = sample.metadata or {}
    if metadata.get("used_skill_strict") is True or metadata.get("skill_group_used_strict") is True:
        return True
    try:
        from examples.agent_bench.skill_group_reward import used_skill_strict

        return bool(used_skill_strict(sample))
    except Exception:
        return "/.claude/skills/" in (sample.response or "") or "/root/.claude/skills/" in (sample.response or "")


def _count_skill_reads(group: list[Sample]) -> int:
    return sum(1 for sample in group if _sample_used_skill_strict(sample))


async def _opsd_score_group(args: Namespace, group: list[Sample]) -> None:
    """OPSD prompt-swap self-teacher scoring for a completed no-skill group.

    For each sample the teacher input is (oracle_prompt_ids + response token
    ids); the same live rollout engine scores it, so teacher == current policy
    weights conditioned on the privileged oracle-skill prompt. Every sample in
    the group is guaranteed a length-aligned ``teacher_log_probs`` afterwards
    (real score or inert rollout-log-prob fallback) so train-data packing never
    sees a partially populated field.
    """
    from relax.engine.rollout.on_policy_distillation import (
        _fallback_teacher_log_probs,
        create_teacher_client_session,
        fetch_teacher_log_probs,
    )

    if any(sample.status == Sample.Status.ABORTED for sample in group):
        return
    if _group_update_kind(group) not in _PAIR_GRPO_UPDATE_KINDS:
        return

    should_score = True
    if _opsd_scope() != "all":
        try:
            rewards = _group_reward_values(args, group)
        except Exception:
            logger.exception(
                "OPSD: failed to read group rewards for task=%s; skipping teacher scoring for this group.",
                _group_task_key(group),
            )
            rewards = []
        threshold = _pair_pass_threshold()
        has_success = any(reward >= threshold for reward in rewards)
        has_failure = any(reward < threshold for reward in rewards)
        # Paper-faithful default: only mixed groups survive the pair filter,
        # so scoring anything else would be wasted teacher prefill.
        should_score = bool(rewards) and has_success and has_failure

    to_score: list[tuple[Sample, list[int]]] = []
    for sample in group:
        response_length = int(sample.response_length or 0)
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        teacher_prompt_ids = metadata.pop("opsd_teacher_prompt_ids", None)
        metadata.pop("opsd_teacher_prompt", None)
        if (
            should_score
            and teacher_prompt_ids
            and response_length > 0
            and len(sample.tokens) >= response_length
        ):
            metadata.pop("opd_teacher_error", None)
            teacher_input_ids = list(teacher_prompt_ids) + list(sample.tokens[-response_length:])
            to_score.append((sample, teacher_input_ids))
        else:
            sample.teacher_log_probs = _fallback_teacher_log_probs(sample, response_length)
            _set_sample_extra(sample, "opsd_scored", 0.0)

    if not to_score:
        return

    async with create_teacher_client_session(args) as teacher_session:
        await asyncio.gather(
            *[
                fetch_teacher_log_probs(args, sample, session=teacher_session, teacher_input_ids=teacher_input_ids)
                for sample, teacher_input_ids in to_score
            ]
        )
    for sample, teacher_input_ids in to_score:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        scored = 0.0 if metadata.get("opd_teacher_error") else 1.0
        _set_sample_extra(sample, "opsd_scored", scored)
        _set_sample_extra(sample, "opsd_teacher_input_len", float(len(teacher_input_ids)))


async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    state = GenerateState(args)

    # eval requests should not be affected by abort state; only skip for training rollout
    if state.aborted and not evaluation:
        return group

    # Generate a unique session_id for each sample in the group
    for sample in group:
        if sample.session_id is None:
            sample.session_id = str(uuid.uuid4())

    # Group-level multimodal encoding de-duplication: when samples in the same
    # group share the same multimodal_inputs object (e.g. after shallow-copy in
    # data_source), encode once and attach the result to every sample so that
    # generate() picks up the pre-encoded data instead of re-encoding per sample.
    first_mm = getattr(group[0], "multimodal_inputs", None)
    if first_mm is not None and all(getattr(s, "multimodal_inputs", None) is first_mm for s in group[1:]):
        encoded_mm, t_enc = await _encode_multimodal_inputs(first_mm)
        for sample in group:
            sample._pre_encoded_mm = encoded_mm
            sample._pre_encoded_mm_elapsed = t_enc

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(
            asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))
        )

    group = await asyncio.gather(*tasks)

    # eval should still compute group reward even if abort was triggered by a concurrent rollout
    if (not state.aborted or evaluation) and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward
            if not evaluation:
                apply_soft_overlong_penalty(args, sample)
        if not evaluation and os.environ.get("RELAX_SKILL_GROUP_REWARD") == "1":
            from examples.agent_bench.skill_group_reward import apply_skill_group_reward

            apply_skill_group_reward(args, group)

        # OPD sglang: fetch teacher log-probs for group_rm samples (independent of reward)
        if getattr(args, "use_opd", False) and getattr(args, "opd_type", None) == "sglang" and not evaluation:
            from relax.engine.rollout.on_policy_distillation import (
                create_teacher_client_session,
                fetch_teacher_log_probs,
            )

            async with create_teacher_client_session(args) as teacher_session:
                await asyncio.gather(*[fetch_teacher_log_probs(args, s, session=teacher_session) for s in group])

    # OPSD prompt-swap self-teacher scoring: runs after all rewards are final
    # (env-assigned during generation and/or group_rm above) so the mixed-group
    # gate sees the same reward view as the pair-atomic accept logic.
    if _opsd_mode_enabled() and not evaluation and not state.aborted:
        try:
            await _opsd_score_group(args, group)
        except Exception:
            logger.exception("OPSD group scoring failed; filling inert fallback teacher log-probs.")
            from relax.engine.rollout.on_policy_distillation import _fallback_teacher_log_probs

            for sample in group:
                if sample.teacher_log_probs is None:
                    sample.teacher_log_probs = _fallback_teacher_log_probs(sample, int(sample.response_length or 0))
                    _set_sample_extra(sample, "opsd_scored", 0.0)

    return group


async def abort(args: Namespace, rollout_id: int) -> tuple[list[list[Sample]], list[list[Sample]]]:
    aborted_samples = []
    completed_protected_samples = []
    abort_pending_timeout = float(os.getenv("RELAX_ABORT_PENDING_TIMEOUT_SEC", "180"))
    abort_protected_timeout = float(os.getenv("RELAX_ABORT_PROTECTED_TIMEOUT_SEC", "180"))
    cancel_wait_timeout = float(os.getenv("RELAX_ABORT_CANCEL_WAIT_SEC", "5"))

    async def _cancel_leftovers(tasks: set[asyncio.Task], label: str) -> None:
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        if cancel_wait_timeout <= 0:
            logger.warning(
                "Cancelled %s %s rollout task(s) without waiting for cancellation acknowledgement.",
                len(tasks),
                label,
            )
            return
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=cancel_wait_timeout)
        except TimeoutError:
            logger.warning(
                "Timed out after %.1fs waiting for %s cancelled %s rollout task(s); "
                "continuing so the next rollout can advance.",
                cancel_wait_timeout,
                len(tasks),
                label,
            )

    state = GenerateState(args)
    assert not state.aborted

    async with state.eval_abort_lock:
        state.abort_in_progress = True
        state.abort_complete.clear()

    # Wait for any in-progress eval to finish before aborting.
    # Aborting during eval would send abort_all to SGLang workers and kill eval requests.
    if state.evaluating > 0:
        logger.info(
            f"Abort deferred: {state.evaluating} eval task(s) in progress. "
            f"Waiting for eval to complete before aborting rollout {rollout_id}."
        )
        while state.evaluating > 0:
            await asyncio.sleep(0.5)
        logger.info(f"Eval completed. Proceeding with abort for rollout {rollout_id}.")

    # Step 1: Wait for protected tasks (abort_count >= partial_rollout_max_aborted_count) to finish naturally.
    if state.protected_pendings:
        logger.info(
            f"Waiting for {len(state.protected_pendings)} protected tasks "
            f"(abort_count >= partial_rollout_max_aborted_count) to complete before aborting others."
        )
        protected_deadline = monotonic() + abort_protected_timeout if abort_protected_timeout > 0 else None
        while state.protected_pendings:
            timeout = None
            if protected_deadline is not None:
                timeout = max(0.0, protected_deadline - monotonic())
                if timeout <= 0:
                    logger.warning(
                        "Timed out waiting for %s protected rollout task(s) during abort of rollout_id=%s; "
                        "cancelling them so rollout can advance.",
                        len(state.protected_pendings),
                        rollout_id,
                    )
                    await _cancel_leftovers(state.protected_pendings, "protected")
                    state.protected_pendings = set()
                    break
            done, state.protected_pendings = await asyncio.wait(
                state.protected_pendings, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                continue
            for task in done:
                try:
                    group = task.result()
                except BaseException as exc:
                    logger.warning(
                        "Protected rollout task failed during abort of rollout_id=%s: %r",
                        rollout_id,
                        exc,
                    )
                    continue
                completed_protected_samples.append(group)

        logger.info(f"All {len(completed_protected_samples)} protected tasks completed.")

    # Step 2: Now abort the remaining (non-protected) pending tasks.
    state.aborted = True

    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_slime_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

    logger.info(f"Abort request for {urls}")
    abort_tasks = [post(f"{url}/abort_request", {"abort_all": True}) for url in urls]
    abort_results = await asyncio.gather(*abort_tasks, return_exceptions=True)
    for url, result in zip(urls, abort_results, strict=False):
        if isinstance(result, BaseException):
            logger.warning(f"Failed to abort worker at {url}: {result}")

    # make sure all the pending tasks are finished
    count = 0
    pending_deadline = monotonic() + abort_pending_timeout if abort_pending_timeout > 0 else None
    while state.pendings:
        timeout = None
        if pending_deadline is not None:
            timeout = max(0.0, pending_deadline - monotonic())
            if timeout <= 0:
                logger.warning(
                    "Timed out waiting for %s pending rollout task(s) after abort_request for rollout_id=%s; "
                    "cancelling leftovers so complete train partitions are not blocked by stale env cleanup.",
                    len(state.pendings),
                    rollout_id,
                )
                await _cancel_leftovers(state.pendings, "pending")
                state.pendings = set()
                break

        done, state.pendings = await asyncio.wait(
            state.pendings, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if not done:
            continue

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            try:
                group = task.result()
            except BaseException as exc:
                logger.warning(
                    "Pending rollout task failed during abort of rollout_id=%s: %r",
                    rollout_id,
                    exc,
                )
                continue
            for sample in group:
                if sample.status == Sample.Status.ABORTED:
                    sample.abort_count += 1
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    return aborted_samples, completed_protected_samples


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]], data_system_client: Any
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based
    rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch
        data_system_client: the data system client to use for transferring batches

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    timer = Timer()
    timer.start("rollout")
    assert args.rollout_global_dataset

    save_root = str(getattr(args, "save", "") or "")
    os.environ["RELAX_ROLLOUT_ID"] = str(rollout_id)
    if save_root:
        os.environ["RELAX_SAVE_ROOT"] = save_root
        os.environ.setdefault("RELAX_RL_RUN_ID", os.path.basename(save_root.rstrip(os.sep)))

    state = GenerateState(args)

    # Start SGLang profiling if enabled
    await start_sglang_profile(args, rollout_id)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    transfer_tasks = []
    batch_to_transfer = []
    aborted_samples = []
    dropped_abort_groups = 0
    dynamic_filter_rejects = 0
    dynamic_filter_forced_accepts = 0
    max_dynamic_filter_rejects = int(os.getenv("RELAX_DYNAMIC_FILTER_MAX_REJECTS_PER_ROLLOUT", "0") or 0)
    max_dynamic_filter_reject_samples = int(
        os.getenv("RELAX_DYNAMIC_FILTER_MAX_REJECT_SAMPLES_PER_ROLLOUT", "0") or 0
    )
    force_accept_exclude_prefixes = tuple(
        prefix.strip()
        for prefix in os.getenv(
            "RELAX_DYNAMIC_FILTER_FORCE_ACCEPT_EXCLUDE_PREFIXES",
            "shadow_all_fail,shadow_missing_reward",
        ).split(",")
        if prefix.strip()
    )
    skill_gate_min_frac = float(os.getenv("RELAX_DYNAMIC_FILTER_MIN_SKILL_READ_FRAC", "0") or 0)
    skill_gate_min_no_read_frac = float(os.getenv("RELAX_DYNAMIC_FILTER_MIN_NO_SKILL_READ_FRAC", "0") or 0)
    skill_gate_enabled = skill_gate_min_frac > 0 or skill_gate_min_no_read_frac > 0
    skill_gate_target_samples = args.rollout_batch_size * args.n_samples_per_prompt
    skill_gate_min_reads = math.ceil(skill_gate_target_samples * skill_gate_min_frac)
    skill_gate_min_no_reads = math.ceil(skill_gate_target_samples * skill_gate_min_no_read_frac)
    skill_gate_max_samples = int(os.getenv("RELAX_DYNAMIC_FILTER_SKILL_READ_MAX_SAMPLES", "0") or 0)
    skill_gate_seen_samples = 0
    skill_gate_accepted_samples = 0
    skill_gate_accepted_reads = 0
    skill_gate_accepted_no_reads = 0
    skill_gate_forced_accepts = 0
    if skill_gate_enabled:
        logger.info(
            "Enabled rollout skill-read/no-read gate: min_read_frac=%.3f min_reads=%s/%s "
            "min_no_read_frac=%.3f min_no_reads=%s/%s max_seen_samples=%s",
            skill_gate_min_frac,
            skill_gate_min_reads,
            skill_gate_target_samples,
            skill_gate_min_no_read_frac,
            skill_gate_min_no_reads,
            skill_gate_target_samples,
            skill_gate_max_samples or "disabled",
        )
    requeue_aborted_groups = os.getenv("RELAX_REQUEUE_ABORTED_GROUPS", "0").lower() in {"1", "true", "yes"}
    max_dropped_abort_groups = int(
        os.getenv(
            "RELAX_MAX_DROPPED_ABORT_GROUPS_PER_ROLLOUT",
            str(max(16, args.rollout_batch_size * 8)),
        )
    )
    pair_atomic_enabled = _env_flag("RELAX_PAIR_ATOMIC_SAMPLING")
    pair_atomic_oracle_groups: dict[str, list[Sample]] = {}
    pair_atomic_pending_no_skill: dict[str, list[list[Sample]]] = {}
    pair_atomic_pending_oracle: dict[str, list[list[Sample]]] = {}
    pair_atomic_ready_no_skill: list[list[Sample]] = []
    pair_atomic_stats: dict[str, float] = {
        "candidate_groups": 0.0,
        "candidate_pairs": 0.0,
        "candidate_no_skill_groups": 0.0,
        "candidate_oracle_groups": 0.0,
        "candidate_unpaired_no_skill": 0.0,
        "candidate_oracle_without_no_skill": 0.0,
        "pending_no_skill_groups": 0.0,
        "pending_oracle_groups": 0.0,
        "ready_no_skill_backlog_groups": 0.0,
        "candidate_unknown_kind_groups": 0.0,
        "accepted_no_skill_mixed_groups": 0.0,
        "accepted_oracle_bc_groups": 0.0,
        "dropped_no_skill_all_pass_groups": 0.0,
        "dropped_no_skill_all_fail_no_oracle_groups": 0.0,
        "dropped_oracle_all_fail_groups": 0.0,
        "dropped_oracle_grpo_all_pass_groups": 0.0,
        "dropped_no_skill_all_fail_bc_disabled_groups": 0.0,
        "dropped_oracle_bc_disabled_groups": 0.0,
        "deferred_oracle_groups": 0.0,
        "late_completed_groups": 0.0,
    }
    if _slate_regret_enabled() and not pair_atomic_enabled:
        raise RuntimeError(
            "RELAX_SLATE_REGRET_GRPO=1 requires RELAX_PAIR_ATOMIC_SAMPLING=1: the slate "
            "arm rides the pair-atomic deferred slot."
        )
    if pair_atomic_enabled:
        if _pair_oracle_grpo_enabled() and _opsd_mode_enabled():
            raise RuntimeError(
                "RELAX_PAIR_ORACLE_GRPO=1 and RELAX_OPSD_MODE=1 are mutually exclusive: "
                "OPSD drops no-skill all-fail groups before the oracle rescue, so the "
                "oracle-GRPO accept branch could never fire."
            )
        if _slate_regret_enabled():
            if _pair_oracle_grpo_enabled() or _opsd_mode_enabled():
                raise RuntimeError(
                    "RELAX_SLATE_REGRET_GRPO=1 is mutually exclusive with "
                    "RELAX_PAIR_ORACLE_GRPO=1 and RELAX_OPSD_MODE=1: the deferred pair "
                    "slot is taken by the slate arm."
                )
            if _pair_oracle_bc_until_step() >= 0:
                raise RuntimeError(
                    "RELAX_SLATE_REGRET_GRPO=1 requires RELAX_PAIR_ORACLE_BC_UNTIL_STEP "
                    "to stay unset: the slate arm must roll out for every task at every "
                    "step (no BC warmup schedule)."
                )
        pair_atomic_speculative_extra_groups = max(0, _env_int("RELAX_PAIR_SPECULATIVE_EXTRA_GROUPS", 0))
        pair_oracle_bc_until_step = _pair_oracle_bc_until_step()
        pair_oracle_bc_enabled_for_rollout = pair_oracle_bc_until_step < 0 or rollout_id < pair_oracle_bc_until_step
        pair_atomic_stats["speculative_extra_groups"] = float(pair_atomic_speculative_extra_groups)
        pair_atomic_stats["oracle_bc_until_step"] = float(pair_oracle_bc_until_step)
        pair_atomic_stats["oracle_bc_phase_enabled"] = float(pair_oracle_bc_enabled_for_rollout)
        logger.info(
            "Enabled atomic no-skill/oracle pair sampling: no-skill mixed groups train with GRPO; "
            "no-skill all-fail groups defer to paired oracle prompt BC; all-pass no-skill and "
            "all-fail oracle groups are dropped/refilled; speculative_extra_groups=%s; "
            "oracle_bc_until_step=%s; oracle_bc_enabled_this_rollout=%s.",
            pair_atomic_speculative_extra_groups,
            pair_oracle_bc_until_step if pair_oracle_bc_until_step >= 0 else "unlimited",
            pair_oracle_bc_enabled_for_rollout,
        )
    else:
        pair_atomic_speculative_extra_groups = 0
        pair_oracle_bc_until_step = -1
        pair_oracle_bc_enabled_for_rollout = False
    num_old_samples = 0
    total_transfer_samples = 0
    get_samples_times: list[float] = []

    def _submit_slate_arm(
        slate_group: list[Sample],
        no_skill_rewards: list[float],
        outcome: str,
        slate_task_key: str,
    ) -> None:
        """SlateRL: submit the deferred slate arm after its no-skill arm completed.

        Stamps the paired no-skill group mean so the reward post-process can
        compute the regret delta. Uses count_capacity=True: the slate group
        occupies (and later releases) its own capacity slot regardless of the
        no-skill group's fate, keeping the 1-slot-per-group invariant.
        """
        mean_reward = sum(no_skill_rewards) / max(1, len(no_skill_rewards))
        pair_atomic_stats["slate_submitted_groups"] = (
            pair_atomic_stats.get("slate_submitted_groups", 0.0) + 1.0
        )
        pair_atomic_stats[f"slate_submitted_after_no_skill_{outcome}_groups"] = (
            pair_atomic_stats.get(f"slate_submitted_after_no_skill_{outcome}_groups", 0.0) + 1.0
        )
        _mark_group_extra(
            slate_group,
            relax_pair_no_skill_mean_reward=float(mean_reward),
            hybrid_pair_no_skill_mixed=1.0 if outcome == "mixed" else 0.0,
            hybrid_pair_no_skill_all_pass=1.0 if outcome == "all_pass" else 0.0,
            hybrid_pair_no_skill_all_fail=1.0 if outcome == "all_fail" else 0.0,
            relax_pair_decision=f"slate_pending_after_no_skill_{outcome}",
        )
        state.submit_generate_group(slate_group, count_capacity=True)
        logger.info(
            "Submitted deferred atomic-pair SLATE group for rollout_id=%s task=%s "
            "after no-skill %s (no_skill_mean=%.3f)",
            rollout_id,
            slate_task_key,
            outcome,
            mean_reward,
        )

    loop = asyncio.get_running_loop()

    while len(data) < target_data_size:
        capacity_target_size = target_data_size + pair_atomic_speculative_extra_groups if pair_atomic_enabled else target_data_size
        while state.remaining_batch_size < capacity_target_size:
            if pair_atomic_enabled and pair_atomic_ready_no_skill:
                slot_capacity = max(capacity_target_size - state.remaining_batch_size, 0)
                submit_no_skill_groups = pair_atomic_ready_no_skill[:slot_capacity]
                pair_atomic_ready_no_skill = pair_atomic_ready_no_skill[slot_capacity:]
                pair_atomic_stats["ready_no_skill_backlog_groups"] = float(len(pair_atomic_ready_no_skill))
                logger.info(
                    "Atomic pair rollout step %s: submitting %s cached ready no-skill groups "
                    "(ready_backlog=%s capacity=%s/%s train_target=%s)",
                    rollout_id,
                    len(submit_no_skill_groups),
                    len(pair_atomic_ready_no_skill),
                    state.remaining_batch_size,
                    capacity_target_size,
                    target_data_size,
                )
                state.submit_generate_tasks(submit_no_skill_groups)
                continue

            _t_get_samples = monotonic()

            if state.prefetched_samples_ref is not None:
                ref = state.prefetched_samples_ref
                state.prefetched_samples_ref = None
                logger.info(f"Rollout step {rollout_id}: using pre-fetched data from previous step")
            else:
                ref = data_source.get_samples.remote(args.over_sampling_batch_size, args.fully_async)

            samples = await loop.run_in_executor(None, ray.get, ref)

            get_samples_times.append(monotonic() - _t_get_samples)
            num_old_samples = len(samples) - args.over_sampling_batch_size
            logger.info(
                f"Starting rollout step {rollout_id}, but had {num_old_samples} old samples for step {rollout_id - 1}"
            )
            target_data_size += num_old_samples

            if args.fully_async and num_old_samples != 0:
                pbar.close()
                pbar = tqdm(total=len(samples) * args.n_samples_per_prompt, desc="Rollout generation")
            if pair_atomic_enabled:
                no_skill_groups, oracle_groups, prep_stats = _prepare_pair_atomic_candidates(
                    samples,
                    rollout_id=rollout_id,
                    pending_no_skill=pair_atomic_pending_no_skill,
                    pending_oracle=pair_atomic_pending_oracle,
                    require_oracle_pair=pair_oracle_bc_enabled_for_rollout,
                )
                if pair_oracle_bc_enabled_for_rollout:
                    pair_atomic_oracle_groups.update(oracle_groups)
                else:
                    pair_atomic_stats["dropped_oracle_bc_disabled_groups"] += float(
                        prep_stats.get("candidate_oracle_groups", 0.0)
                    )
                if pair_atomic_ready_no_skill:
                    no_skill_groups = pair_atomic_ready_no_skill + no_skill_groups
                    pair_atomic_ready_no_skill = []
                capacity_target_size = (
                    target_data_size + pair_atomic_speculative_extra_groups
                    if pair_atomic_enabled
                    else target_data_size
                )
                slot_capacity = max(capacity_target_size - state.remaining_batch_size, 0)
                submit_no_skill_groups = no_skill_groups[:slot_capacity]
                if len(no_skill_groups) > slot_capacity:
                    pair_atomic_ready_no_skill.extend(no_skill_groups[slot_capacity:])
                for key, value in prep_stats.items():
                    pair_atomic_stats[key] = pair_atomic_stats.get(key, 0.0) + value
                pair_atomic_stats["pending_no_skill_groups"] = prep_stats.get("pending_no_skill_groups", 0.0)
                pair_atomic_stats["pending_oracle_groups"] = prep_stats.get("pending_oracle_groups", 0.0)
                pair_atomic_stats["ready_no_skill_backlog_groups"] = float(len(pair_atomic_ready_no_skill))
                logger.info(
                    "Atomic pair rollout step %s: fetched_groups=%s candidate_pairs=%s "
                    "submit_no_skill_groups=%s deferred_oracle_groups=%s "
                    "pending_no_skill=%s pending_oracle=%s ready_backlog=%s capacity=%s/%s train_target=%s "
                    "oracle_bc_enabled=%s",
                    rollout_id,
                    len(samples),
                    int(prep_stats.get("candidate_pairs", 0.0)),
                    len(submit_no_skill_groups),
                    len(oracle_groups) if pair_oracle_bc_enabled_for_rollout else 0,
                    int(prep_stats.get("pending_no_skill_groups", 0.0)),
                    int(prep_stats.get("pending_oracle_groups", 0.0)),
                    len(pair_atomic_ready_no_skill),
                    state.remaining_batch_size,
                    capacity_target_size,
                    target_data_size,
                    pair_oracle_bc_enabled_for_rollout,
                )
                state.submit_generate_tasks(submit_no_skill_groups)
            else:
                state.submit_generate_tasks(samples)
        # wait for the generation to finish (from both normal and protected pending sets)
        all_pendings = state.pendings | state.protected_pendings
        done, remaining = await asyncio.wait(all_pendings, return_when=asyncio.FIRST_COMPLETED)
        state.pendings = state.pendings & remaining
        state.protected_pendings = state.protected_pendings & remaining
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            if pair_atomic_enabled:
                role = _group_pair_role(group)
                pair_id = str(_sample_extra_info(group[0]).get("relax_pair_id") or "")
                task_key = _group_task_key(group)

                if len(data) >= target_data_size:
                    pair_atomic_stats["late_completed_groups"] += 1.0
                    state.remaining_batch_size = max(0, state.remaining_batch_size - 1)
                    logger.info(
                        "Dropping late speculative atomic-pair group after train batch is full "
                        "for rollout_id=%s role=%s task=%s accepted=%s/%s",
                        rollout_id,
                        role,
                        task_key,
                        len(data),
                        target_data_size,
                    )
                    continue

                if any(sample.status == Sample.Status.ABORTED for sample in group):
                    dropped_abort_groups += 1
                    state.remaining_batch_size -= 1
                    metric_gatherer.on_dynamic_filter_drop(reason=f"pair_{role or 'unknown'}_aborted")
                    logger.warning(
                        "Dropping ABORTED atomic-pair group for rollout_id=%s role=%s task=%s "
                        "(dropped=%s, max=%s).",
                        rollout_id,
                        role,
                        task_key,
                        dropped_abort_groups,
                        max_dropped_abort_groups,
                    )
                    if dropped_abort_groups > max_dropped_abort_groups:
                        raise RuntimeError(
                            f"Too many ABORTED groups in rollout {rollout_id}: "
                            f"{dropped_abort_groups} > {max_dropped_abort_groups}."
                        )
                    continue

                if any(sample.reward is None for sample in group):
                    dropped_abort_groups += 1
                    state.remaining_batch_size -= 1
                    metric_gatherer.on_dynamic_filter_drop(reason=f"pair_{role or 'unknown'}_missing_reward")
                    logger.warning(
                        "Dropping atomic-pair group with missing reward for rollout_id=%s role=%s task=%s "
                        "(dropped=%s, max=%s).",
                        rollout_id,
                        role,
                        task_key,
                        dropped_abort_groups,
                        max_dropped_abort_groups,
                    )
                    if dropped_abort_groups > max_dropped_abort_groups:
                        raise RuntimeError(
                            f"Too many reward-missing groups in rollout {rollout_id}: "
                            f"{dropped_abort_groups} > {max_dropped_abort_groups}."
                        )
                    continue

                rewards = _group_reward_values(args, group)
                threshold = _pair_pass_threshold()
                has_success = any(reward >= threshold for reward in rewards)
                has_failure = any(reward < threshold for reward in rewards)

                if role == "no_skill":
                    if has_success and has_failure:
                        deferred_group = pair_atomic_oracle_groups.pop(pair_id, None)
                        if (
                            _slate_regret_enabled()
                            and deferred_group is not None
                            and _group_update_kind(deferred_group) in _SLATE_UPDATE_KINDS
                        ):
                            _submit_slate_arm(deferred_group, rewards, "mixed", task_key)
                        pair_atomic_stats["accepted_no_skill_mixed_groups"] += 1.0
                        pass_count = sum(1 for reward in rewards if reward >= threshold)
                        _mark_group_extra(
                            group,
                            hybrid_is_shadow=0.0,
                            hybrid_grpo_weight=1.0,
                            hybrid_shadow_weight=0.0,
                            hybrid_pair_no_skill_mixed=1.0,
                            hybrid_pair_no_skill_all_fail=0.0,
                            hybrid_pair_no_skill_all_pass=0.0,
                            relax_pair_decision="no_skill_grpo_mixed",
                        )
                        if _opsd_mode_enabled():
                            group_scored = sum(
                                1.0
                                for sample in group
                                if float(_sample_extra_info(sample).get("opsd_scored") or 0.0) > 0.0
                            )
                            pair_atomic_stats["opsd_scored_samples"] = (
                                pair_atomic_stats.get("opsd_scored_samples", 0.0) + group_scored
                            )
                            pair_atomic_stats["opsd_unscored_samples"] = (
                                pair_atomic_stats.get("opsd_unscored_samples", 0.0) + (len(group) - group_scored)
                            )
                        logger.info(
                            "Accepted atomic-pair no-skill GRPO mixed group for rollout_id=%s task=%s "
                            "pass=%s/%s accepted_next=%s/%s",
                            rollout_id,
                            task_key,
                            pass_count,
                            len(rewards),
                            len(data) + 1,
                            target_data_size,
                        )
                    elif has_success and not has_failure:
                        pair_atomic_stats["dropped_no_skill_all_pass_groups"] += 1.0
                        dynamic_filter_rejects += 1
                        state.remaining_batch_size -= 1
                        metric_gatherer.on_dynamic_filter_drop(reason="pair_no_skill_all_pass")
                        logger.info(
                            "Dropping atomic-pair no-skill all-pass group for rollout_id=%s task=%s pass=%s/%s",
                            rollout_id,
                            task_key,
                            sum(1 for reward in rewards if reward >= threshold),
                            len(rewards),
                        )
                        deferred_group = pair_atomic_oracle_groups.pop(pair_id, None)
                        if (
                            _slate_regret_enabled()
                            and deferred_group is not None
                            and _group_update_kind(deferred_group) in _SLATE_UPDATE_KINDS
                        ):
                            _submit_slate_arm(deferred_group, rewards, "all_pass", task_key)
                        continue
                    else:
                        if _slate_regret_enabled():
                            # SlateRL: no BC rescue. The all-fail no-skill group is
                            # zero-advantage under GRPO, so drop it, but ALWAYS
                            # submit the deferred slate arm (all-fail no-skill is
                            # exactly where "was the slate used well / resisted"
                            # carries the most regret signal).
                            deferred_group = pair_atomic_oracle_groups.pop(pair_id, None)
                            pair_atomic_stats["dropped_no_skill_all_fail_slate_groups"] = (
                                pair_atomic_stats.get("dropped_no_skill_all_fail_slate_groups", 0.0) + 1.0
                            )
                            dynamic_filter_rejects += 1
                            state.remaining_batch_size -= 1
                            metric_gatherer.on_dynamic_filter_drop(reason="pair_no_skill_all_fail_slate")
                            if (
                                deferred_group is not None
                                and _group_update_kind(deferred_group) in _SLATE_UPDATE_KINDS
                            ):
                                _submit_slate_arm(deferred_group, rewards, "all_fail", task_key)
                            else:
                                logger.warning(
                                    "Slate mode: no deferred slate group for all-fail no-skill "
                                    "rollout_id=%s task=%s pair_id=%s",
                                    rollout_id,
                                    task_key,
                                    pair_id,
                                )
                            continue
                        if _opsd_mode_enabled():
                            # OPSD: the oracle arm is a prompt donor only; there is
                            # no oracle-BC rescue. All-fail no-skill groups are
                            # zero-advantage under GRPO, so drop/refill them
                            # (paper-faithful "pop zero-advantage" behavior).
                            pair_atomic_oracle_groups.pop(pair_id, None)
                            pair_atomic_stats["dropped_no_skill_all_fail_opsd_groups"] = (
                                pair_atomic_stats.get("dropped_no_skill_all_fail_opsd_groups", 0.0) + 1.0
                            )
                            dynamic_filter_rejects += 1
                            state.remaining_batch_size -= 1
                            metric_gatherer.on_dynamic_filter_drop(reason="pair_no_skill_all_fail_opsd")
                            logger.info(
                                "Dropping atomic-pair no-skill all-fail group for rollout_id=%s task=%s "
                                "under OPSD mode (oracle arm is prompt-donor only; no BC rescue).",
                                rollout_id,
                                task_key,
                            )
                            continue
                        if not pair_oracle_bc_enabled_for_rollout:
                            pair_atomic_stats["dropped_no_skill_all_fail_bc_disabled_groups"] += 1.0
                            dynamic_filter_rejects += 1
                            state.remaining_batch_size -= 1
                            metric_gatherer.on_dynamic_filter_drop(reason="pair_no_skill_all_fail_bc_disabled")
                            logger.info(
                                "Dropping atomic-pair no-skill all-fail group for rollout_id=%s task=%s "
                                "because oracle BC is disabled for this phase (bc_until_step=%s).",
                                rollout_id,
                                task_key,
                                pair_oracle_bc_until_step,
                            )
                            pair_atomic_oracle_groups.pop(pair_id, None)
                            continue
                        oracle_group = pair_atomic_oracle_groups.pop(pair_id, None)
                        if oracle_group is None:
                            pair_atomic_stats["dropped_no_skill_all_fail_no_oracle_groups"] += 1.0
                            dynamic_filter_rejects += 1
                            state.remaining_batch_size -= 1
                            metric_gatherer.on_dynamic_filter_drop(reason="pair_no_skill_all_fail_no_oracle")
                            logger.warning(
                                "Dropping atomic-pair no-skill all-fail group without paired oracle for "
                                "rollout_id=%s task=%s pair_id=%s",
                                rollout_id,
                                task_key,
                                pair_id,
                            )
                            continue
                        pair_atomic_stats["deferred_oracle_groups"] += 1.0
                        no_skill_mean = float(sum(rewards) / max(1, len(rewards)))
                        no_skill_pass_count = sum(1 for reward in rewards if reward >= threshold)
                        _mark_group_extra(
                            oracle_group,
                            hybrid_pair_no_skill_all_fail=1.0,
                            hybrid_pair_no_skill_all_pass=0.0,
                            hybrid_pair_no_skill_mixed=0.0,
                            relax_pair_no_skill_mean_reward=no_skill_mean,
                            relax_pair_no_skill_pass_count=float(no_skill_pass_count),
                            relax_pair_no_skill_group_size=float(len(rewards)),
                            relax_pair_decision="oracle_pending_after_no_skill_all_fail",
                        )
                        state.submit_generate_group(oracle_group, count_capacity=False)
                        logger.info(
                            "Deferred atomic-pair oracle group for rollout_id=%s task=%s after no-skill all-fail "
                            "(no_skill_mean=%.3f pass=%s/%s)",
                            rollout_id,
                            task_key,
                            no_skill_mean,
                            no_skill_pass_count,
                            len(rewards),
                        )
                        continue
                elif role == "oracle":
                    if not pair_oracle_bc_enabled_for_rollout:
                        pair_atomic_stats["dropped_oracle_bc_disabled_groups"] += 1.0
                        dynamic_filter_rejects += 1
                        state.remaining_batch_size -= 1
                        metric_gatherer.on_dynamic_filter_drop(reason="pair_oracle_bc_disabled")
                        logger.info(
                            "Dropping atomic-pair oracle group for rollout_id=%s task=%s because oracle BC "
                            "is disabled for this phase (bc_until_step=%s).",
                            rollout_id,
                            task_key,
                            pair_oracle_bc_until_step,
                        )
                        continue
                    if has_success:
                        pass_count = sum(1 for reward in rewards if reward >= threshold)
                        if _pair_oracle_grpo_enabled():
                            if not has_failure and _pair_oracle_grpo_drop_all_pass_enabled():
                                pair_atomic_stats["dropped_oracle_grpo_all_pass_groups"] = (
                                    pair_atomic_stats.get("dropped_oracle_grpo_all_pass_groups", 0.0) + 1.0
                                )
                                dynamic_filter_rejects += 1
                                state.remaining_batch_size -= 1
                                metric_gatherer.on_dynamic_filter_drop(reason="pair_oracle_grpo_all_pass")
                                logger.info(
                                    "Dropping atomic-pair oracle GRPO all-pass group for rollout_id=%s task=%s "
                                    "pass=%s/%s because RELAX_PAIR_ORACLE_GRPO_DROP_ALL_PASS=1.",
                                    rollout_id,
                                    task_key,
                                    pass_count,
                                    len(rewards),
                                )
                                continue
                            # Oracle-GRPO60: same accept gate, but the oracle group trains
                            # with in-group GRPO advantages instead of BC. An
                            # all-pass oracle group has zero in-group advantage
                            # (inert slot); count it separately for monitoring.
                            pair_atomic_stats["accepted_oracle_grpo_groups"] = (
                                pair_atomic_stats.get("accepted_oracle_grpo_groups", 0.0) + 1.0
                            )
                            if not has_failure:
                                pair_atomic_stats["accepted_oracle_grpo_allpass_groups"] = (
                                    pair_atomic_stats.get("accepted_oracle_grpo_allpass_groups", 0.0) + 1.0
                                )
                            _mark_group_extra(
                                group,
                                hybrid_is_shadow=0.0,
                                hybrid_grpo_weight=1.0,
                                hybrid_shadow_weight=0.0,
                                hybrid_pair_bc_enabled=0.0,
                                hybrid_pair_no_skill_all_fail=1.0,
                                hybrid_pair_oracle_has_success=1.0,
                                relax_pair_oracle_grpo_cross_arm_adv=(
                                    1.0 if _pair_oracle_grpo_cross_arm_adv_enabled() else 0.0
                                ),
                                relax_pair_oracle_grpo_drop_all_pass=(
                                    1.0 if _pair_oracle_grpo_drop_all_pass_enabled() else 0.0
                                ),
                                relax_pair_decision="oracle_grpo_after_no_skill_all_fail",
                            )
                            logger.info(
                                "Accepted atomic-pair oracle GRPO group for rollout_id=%s task=%s "
                                "pass=%s/%s accepted_next=%s/%s",
                                rollout_id,
                                task_key,
                                pass_count,
                                len(rewards),
                                len(data) + 1,
                                target_data_size,
                            )
                        else:
                            pair_atomic_stats["accepted_oracle_bc_groups"] += 1.0
                            _mark_group_extra(
                                group,
                                hybrid_is_shadow=1.0,
                                hybrid_grpo_weight=0.0,
                                hybrid_shadow_weight=1.0,
                                hybrid_pair_bc_enabled=1.0,
                                hybrid_pair_no_skill_all_fail=1.0,
                                hybrid_pair_oracle_has_success=1.0,
                                relax_pair_decision="oracle_bc_after_no_skill_all_fail",
                            )
                            logger.info(
                                "Accepted atomic-pair oracle BC group for rollout_id=%s task=%s "
                                "pass=%s/%s accepted_next=%s/%s",
                                rollout_id,
                                task_key,
                                pass_count,
                                len(rewards),
                                len(data) + 1,
                                target_data_size,
                            )
                    else:
                        pair_atomic_stats["dropped_oracle_all_fail_groups"] += 1.0
                        dynamic_filter_rejects += 1
                        state.remaining_batch_size -= 1
                        metric_gatherer.on_dynamic_filter_drop(reason="pair_oracle_all_fail")
                        logger.info(
                            "Dropping atomic-pair oracle all-fail group for rollout_id=%s task=%s",
                            rollout_id,
                            task_key,
                        )
                        continue
                elif role == "slate" and _slate_regret_enabled():
                    # SlateRL completion: the slate arm trains with in-group GRPO.
                    # Mixed groups carry per-sample advantage; uniform groups are
                    # accepted only when the regret delta vs the paired no-skill
                    # mean is large enough to carry group-level signal (the
                    # advantage shift itself is applied in
                    # examples.agent_bench.slate_regret_gating post-process).
                    extra0 = _sample_extra_info(group[0])
                    try:
                        slate_no_skill_mean = float(
                            extra0.get("relax_pair_no_skill_mean_reward") or 0.0
                        )
                    except (TypeError, ValueError):
                        slate_no_skill_mean = 0.0
                    slate_mean = sum(rewards) / max(1, len(rewards))
                    slate_delta = slate_mean - slate_no_skill_mean
                    _mark_group_extra(group, slate_regret_delta=float(slate_delta))
                    if has_success and has_failure:
                        pair_atomic_stats["accepted_slate_grpo_mixed_groups"] = (
                            pair_atomic_stats.get("accepted_slate_grpo_mixed_groups", 0.0) + 1.0
                        )
                        _mark_group_extra(
                            group,
                            hybrid_is_shadow=0.0,
                            hybrid_grpo_weight=1.0,
                            hybrid_shadow_weight=0.0,
                            hybrid_pair_bc_enabled=0.0,
                            relax_pair_decision="slate_grpo_mixed",
                        )
                        logger.info(
                            "Accepted atomic-pair SLATE GRPO mixed group for rollout_id=%s task=%s "
                            "pass=%s/%s delta=%.3f accepted_next=%s/%s",
                            rollout_id,
                            task_key,
                            sum(1 for reward in rewards if reward >= threshold),
                            len(rewards),
                            slate_delta,
                            len(data) + 1,
                            target_data_size,
                        )
                    elif abs(slate_delta) >= _slate_uniform_min_delta():
                        pair_atomic_stats["accepted_slate_grpo_uniform_groups"] = (
                            pair_atomic_stats.get("accepted_slate_grpo_uniform_groups", 0.0) + 1.0
                        )
                        _mark_group_extra(
                            group,
                            hybrid_is_shadow=0.0,
                            hybrid_grpo_weight=1.0,
                            hybrid_shadow_weight=0.0,
                            hybrid_pair_bc_enabled=0.0,
                            relax_pair_decision="slate_grpo_uniform",
                        )
                        logger.info(
                            "Accepted atomic-pair SLATE uniform group (all-%s) for rollout_id=%s "
                            "task=%s delta=%.3f accepted_next=%s/%s",
                            "pass" if has_success else "fail",
                            rollout_id,
                            task_key,
                            slate_delta,
                            len(data) + 1,
                            target_data_size,
                        )
                    else:
                        pair_atomic_stats["dropped_slate_uniform_small_delta_groups"] = (
                            pair_atomic_stats.get("dropped_slate_uniform_small_delta_groups", 0.0) + 1.0
                        )
                        dynamic_filter_rejects += 1
                        state.remaining_batch_size -= 1
                        metric_gatherer.on_dynamic_filter_drop(reason="pair_slate_uniform_small_delta")
                        logger.info(
                            "Dropping atomic-pair SLATE uniform group with small delta for "
                            "rollout_id=%s task=%s delta=%.3f (min=%.3f)",
                            rollout_id,
                            task_key,
                            slate_delta,
                            _slate_uniform_min_delta(),
                        )
                        continue
                else:
                    dynamic_filter_rejects += 1
                    state.remaining_batch_size -= 1
                    metric_gatherer.on_dynamic_filter_drop(reason="pair_unknown_role")
                    logger.warning(
                        "Dropping atomic-pair group with unknown role for rollout_id=%s task=%s kind=%s",
                        rollout_id,
                        task_key,
                        _group_update_kind(group),
                    )
                    continue
            else:
                dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
                if not dynamic_filter_output.keep:
                    dynamic_filter_rejects += 1
                    metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                    force_by_rejects = max_dynamic_filter_rejects > 0 and dynamic_filter_rejects > max_dynamic_filter_rejects
                    force_by_samples = (
                        max_dynamic_filter_reject_samples > 0
                        and dynamic_filter_rejects * args.n_samples_per_prompt >= max_dynamic_filter_reject_samples
                    )
                    reject_reason = dynamic_filter_output.reason or ""
                    force_accept_excluded = any(
                        reject_reason.startswith(prefix) for prefix in force_accept_exclude_prefixes
                    )
                    if (force_by_rejects or force_by_samples) and not force_accept_excluded:
                        dynamic_filter_forced_accepts += 1
                        logger.warning(
                            "Forcing dynamic-filter rejected group into rollout_id=%s after %s rejects/%s samples "
                            "(forced=%s, max_rejects=%s, max_reject_samples=%s, reason=%s). This prevents "
                            "infinite refills when the filter cannot find enough nonzero-std reward groups.",
                            rollout_id,
                            dynamic_filter_rejects,
                            dynamic_filter_rejects * args.n_samples_per_prompt,
                            dynamic_filter_forced_accepts,
                            max_dynamic_filter_rejects,
                            max_dynamic_filter_reject_samples,
                            dynamic_filter_output.reason,
                        )
                    else:
                        if force_accept_excluded and (force_by_rejects or force_by_samples):
                            logger.info(
                                "Hard-dropping dynamic-filter rejected group for rollout_id=%s despite force-accept "
                                "threshold (rejects=%s/%s samples, reason=%s)",
                                rollout_id,
                                dynamic_filter_rejects,
                                dynamic_filter_rejects * args.n_samples_per_prompt,
                                reject_reason,
                            )
                        state.remaining_batch_size -= 1
                        continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if any(sample.status == Sample.Status.ABORTED for sample in group):
                abort_details = []
                for sample in group:
                    if sample.response and "start_rollout_id" not in sample.metadata:
                        sample.metadata["start_rollout_id"] = rollout_id
                    sample.abort_count += 1
                    if sample.status == Sample.Status.ABORTED:
                        label = getattr(sample, "label", {}) or {}
                        metadata = getattr(sample, "metadata", {}) or {}
                        traces = metadata.get("rollout_traces") or []
                        last_trace = traces[-1] if traces else {}
                        inference = last_trace.get("inference") or {}
                        env_step = last_trace.get("env_step") or {}
                        info = env_step.get("info") or {}
                        abort_details.append(
                            {
                                "bench": label.get("bench") or metadata.get("extra_info", {}).get("bench"),
                                "task_id": label.get("task_id") or metadata.get("extra_info", {}).get("task_id"),
                                "stop_reason": metadata.get("rollout_stop_reason"),
                                "finish_type": inference.get("finish_type"),
                                "env_error": info.get("error"),
                                "response_length": getattr(sample, "response_length", None),
                            }
                        )
                dropped_abort_groups += 1
                state.remaining_batch_size -= 1
                logger.warning(
                    "Dropping ABORTED group for rollout_id=%s and refilling it "
                    "(dropped=%s, max=%s, requeue=%s, details=%s). This prevents a partial "
                    "train partition from being exposed to logprob/train consumers.",
                    rollout_id,
                    dropped_abort_groups,
                    max_dropped_abort_groups,
                    requeue_aborted_groups,
                    abort_details[:8],
                )
                if requeue_aborted_groups:
                    aborted_samples.append(group)
                if dropped_abort_groups > max_dropped_abort_groups:
                    raise RuntimeError(
                        f"Too many ABORTED groups in rollout {rollout_id}: "
                        f"{dropped_abort_groups} > {max_dropped_abort_groups}. "
                        "Failing fast instead of leaving a partial train partition."
                    )
                continue
            elif len(data) < target_data_size:
                if skill_gate_enabled:
                    group_reads = _count_skill_reads(group)
                    group_size = len(group)
                    group_no_reads = group_size - group_reads
                    skill_gate_seen_samples += group_size
                    projected_samples = skill_gate_accepted_samples + group_size
                    projected_reads = skill_gate_accepted_reads + group_reads
                    projected_no_reads = skill_gate_accepted_no_reads + group_no_reads
                    remaining_slots = max(skill_gate_target_samples - projected_samples, 0)
                    max_possible_reads = projected_reads + remaining_slots
                    max_possible_no_reads = projected_no_reads + remaining_slots
                    force_skill_accept = (
                        skill_gate_max_samples > 0 and skill_gate_seen_samples >= skill_gate_max_samples
                    )
                    missing_read = max_possible_reads < skill_gate_min_reads
                    missing_no_read = max_possible_no_reads < skill_gate_min_no_reads
                    if not force_skill_accept and (missing_read or missing_no_read):
                        state.remaining_batch_size -= 1
                        reason = (
                            "low_skill_read_gbs"
                            if missing_read and not missing_no_read
                            else "low_no_skill_read_gbs"
                            if missing_no_read and not missing_read
                            else "low_skill_read_and_no_read_gbs"
                        )
                        metric_gatherer.on_dynamic_filter_drop(reason=reason)
                        logger.info(
                            "Dropping group for rollout_id=%s due to skill-read/no-read gate (%s): "
                            "group_reads=%s/%s group_no_reads=%s/%s "
                            "accepted_reads=%s/%s accepted_no_reads=%s/%s "
                            "projected_max_reads=%s/%s projected_max_no_reads=%s/%s "
                            "seen_samples=%s max_seen_samples=%s",
                            rollout_id,
                            reason,
                            group_reads,
                            group_size,
                            group_no_reads,
                            group_size,
                            skill_gate_accepted_reads,
                            skill_gate_accepted_samples,
                            skill_gate_accepted_no_reads,
                            skill_gate_accepted_samples,
                            max_possible_reads,
                            skill_gate_min_reads,
                            max_possible_no_reads,
                            skill_gate_min_no_reads,
                            skill_gate_seen_samples,
                            skill_gate_max_samples or "disabled",
                        )
                        continue
                    if force_skill_accept and (projected_reads < skill_gate_min_reads or projected_no_reads < skill_gate_min_no_reads):
                        skill_gate_forced_accepts += 1
                        logger.warning(
                            "Forcing skill-read/no-read gate accept for rollout_id=%s after seeing %s samples: "
                            "projected_reads=%s/%s projected_no_reads=%s/%s, forced=%s",
                            rollout_id,
                            skill_gate_seen_samples,
                            projected_reads,
                            skill_gate_min_reads,
                            projected_no_reads,
                            skill_gate_min_no_reads,
                            skill_gate_forced_accepts,
                        )
                    skill_gate_accepted_samples = projected_samples
                    skill_gate_accepted_reads = projected_reads
                    skill_gate_accepted_no_reads = projected_no_reads
                batch_to_transfer.append(group)
                total_transfer_samples += 1

            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)
                # When batch is ready, spawn background transfer task (don't block generator)
                # Create background task for transfer (don't await here!)
        # Only spawn a transfer task when there are samples to transfer.
        transfer_batch_size = (
            args.global_batch_size // args.num_iters_per_train_update // args.n_samples_per_prompt
            if not args.colocate
            else args.rollout_batch_size
        )  # Samples per batch to transfer
        # in fully async mode, we transfer all remaining samples when we reach the target size
        if len(batch_to_transfer) >= transfer_batch_size:
            if total_transfer_samples <= num_old_samples:
                transfer_task = asyncio.create_task(
                    transfer_batch_to_data_system(
                        args,
                        batch_to_transfer,
                        len(batch_to_transfer),
                        rollout_id - 1,
                        data_system_client,
                    )
                )
                transfer_tasks.append(transfer_task)
                batch_to_transfer = []
                logger.info(
                    f"Total yielded: {target_data_size - 2 * num_old_samples + total_transfer_samples}/{target_data_size - num_old_samples} for step: {rollout_id - 1}"
                )
            else:
                if len(batch_to_transfer) > total_transfer_samples - num_old_samples:
                    cutoff_batch = len(batch_to_transfer) - total_transfer_samples + num_old_samples
                    transfer_task = asyncio.create_task(
                        transfer_batch_to_data_system(
                            args,
                            batch_to_transfer[:cutoff_batch],
                            len(batch_to_transfer[:cutoff_batch]),
                            rollout_id - 1,
                            data_system_client,
                        )
                    )
                    transfer_tasks.append(transfer_task)
                    batch_to_transfer = batch_to_transfer[cutoff_batch:]
                    logger.info(
                        f"{num_old_samples} old samples completed! Total yielded: {target_data_size - num_old_samples}/{target_data_size - num_old_samples} for step: {rollout_id - 1}"
                    )
                else:
                    transfer_task = asyncio.create_task(
                        transfer_batch_to_data_system(
                            args,
                            batch_to_transfer,
                            len(batch_to_transfer),
                            rollout_id,
                            data_system_client,
                        )
                    )
                    transfer_tasks.append(transfer_task)
                    batch_to_transfer = []
                    logger.info(
                        f"Total yielded: {total_transfer_samples - num_old_samples}/{target_data_size - num_old_samples} for step: {rollout_id}"
                    )

    if len(batch_to_transfer) > 0:
        transfer_task = asyncio.create_task(
            transfer_batch_to_data_system(
                args,
                batch_to_transfer,
                len(batch_to_transfer),
                rollout_id,
                data_system_client,
            )
        )
        transfer_tasks.append(transfer_task)
        batch_to_transfer = []
        logger.info(
            f"Total yielded: {total_transfer_samples - num_old_samples}/{target_data_size - num_old_samples} for step: {rollout_id}"
        )

    if not args.fully_async:
        state.prefetched_samples_ref = data_source.get_samples.remote(args.over_sampling_batch_size, args.fully_async)
        logger.info(f"Rollout step {rollout_id}: pre-submitted data fetch for next step")

    logger.info(f"Generator exhausted. Waiting for {len(transfer_tasks)} transfer tasks to complete...")
    # Wait for all transfer tasks to complete
    if transfer_tasks:
        await asyncio.gather(*transfer_tasks)
    pbar.close()

    # Stop SGLang profiling if enabled (no-op if num_steps was set — SGLang auto-stops)
    await stop_sglang_profile(args, rollout_id)

    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
    )

    rollout_time = timer.end("rollout")

    all_samples = [sample for group in data for sample in (group if isinstance(group, list) else [group])]
    timing_metrics = _aggregate_rollout_timing(all_samples, get_samples_times)
    # Acceptance-layer visibility: how much rollout compute was structurally
    # discarded before training (effective-group-ratio diagnostics).
    timing_metrics["rollout/dynamic_filter/rejected_groups"] = float(dynamic_filter_rejects)
    timing_metrics["rollout/dynamic_filter/forced_accepts"] = float(dynamic_filter_forced_accepts)
    timing_metrics["rollout/dynamic_filter/dropped_abort_groups"] = float(dropped_abort_groups)
    if pair_atomic_enabled:
        for key, value in pair_atomic_stats.items():
            timing_metrics[f"rollout/pair_atomic/{key}"] = float(value)
        accepted_groups = (
            pair_atomic_stats["accepted_no_skill_mixed_groups"]
            + pair_atomic_stats["accepted_oracle_bc_groups"]
        )
        if accepted_groups > 0:
            timing_metrics["rollout/pair_atomic/oracle_bc_group_frac"] = (
                pair_atomic_stats["accepted_oracle_bc_groups"] / accepted_groups
            )
        if _opsd_mode_enabled():
            opsd_scored = pair_atomic_stats.get("opsd_scored_samples", 0.0)
            opsd_unscored = pair_atomic_stats.get("opsd_unscored_samples", 0.0)
            opsd_total = opsd_scored + opsd_unscored
            opsd_scored_frac = (opsd_scored / opsd_total) if opsd_total > 0 else 0.0
            timing_metrics["rollout/opsd/scored_frac"] = opsd_scored_frac
            timing_metrics["rollout/opsd/scored_samples"] = opsd_scored
            timing_metrics["rollout/opsd/unscored_samples"] = opsd_unscored
            if opsd_total > 0 and opsd_scored_frac < 0.8:
                logger.warning(
                    "OPSD WARNING: teacher scoring succeeded for only %.1f%% of accepted no-skill samples "
                    "in rollout %s (%s/%s). Unscored samples contribute ZERO reverse-KL (validity-gated); "
                    "a persistently low fraction means the run is degrading toward plain no-skill GRPO — "
                    "check opd_teacher_error in the train dump and the rollout engine load/timeouts.",
                    opsd_scored_frac * 100.0,
                    rollout_id,
                    int(opsd_scored),
                    int(opsd_total),
                )
    if skill_gate_enabled and all_samples:
        skill_read_count = sum(1 for sample in all_samples if _sample_used_skill_strict(sample))
        timing_metrics["rollout/skill_read_gate/read_frac"] = skill_read_count / len(all_samples)
        timing_metrics["rollout/skill_read_gate/no_read_frac"] = 1.0 - (skill_read_count / len(all_samples))
        # Gate-pressure signal: stays at batch size while the gate never
        # rejects; climbing above it means the policy is drifting away from
        # skill reading and the gate is actively resampling.
        timing_metrics["rollout/skill_read_gate/seen_samples"] = skill_gate_seen_samples
        timing_metrics["rollout/skill_read_gate/forced_accepts"] = skill_gate_forced_accepts

    global CURRENT_ROLLOUT_BATCH
    if CURRENT_ROLLOUT_BATCH:
        save_debug_rollout_data(
            args, CURRENT_ROLLOUT_BATCH, rollout_id=rollout_id, evaluation=False, tokenizer=state.tokenizer
        )
        _log_rollout_data(rollout_id, args, CURRENT_ROLLOUT_BATCH, timing_metrics, rollout_time)
        if args.debug_rollout_only:
            logger.info("Debug rollout only mode - data system cleanup")
            await data_system_client.async_clear_partition(partition_id=f"train_{rollout_id}")
        # Cleanup
        CURRENT_ROLLOUT_BATCH.clear()

    try:
        # there are still some unfinished requests, abort them
        # abort() returns (aborted_samples, completed_protected_samples)
        new_aborted, completed_protected = await abort(args, rollout_id)
        aborted_samples.extend(new_aborted)
        aborted_samples.extend(completed_protected)
        if aborted_samples:
            logger.info(
                f"Rollout not completed for rollout_id: {rollout_id}, have {len(aborted_samples)} samples aborted."
            )
        else:
            logger.info(f"Rollout fully completed for rollout_id: {rollout_id}.")
    finally:
        state.reset()

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:

    state = GenerateState(args)
    # Increment evaluating counter so that abort() knows to wait for eval to finish.
    # This prevents abort_all from killing in-flight eval requests on SGLang workers.
    # The inverse is also required: if a previous train rollout is still inside
    # abort cleanup, eval must not start because generate_and_rm short-circuits
    # samples to ABORTED while GenerateState.aborted is set.
    async with state.eval_abort_lock:
        await state.wait_for_abort_complete(f"Eval rollout {rollout_id}")
        state.evaluating += 1
    try:
        coros = []
        for dataset_cfg in getattr(args, "eval_datasets", []) or []:
            coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
        results_list = await asyncio.gather(*coros)
        results = {}
        for r in results_list:
            results.update(r)
        return RolloutFnEvalOutput(data=results), []
    finally:
        state.evaluating -= 1


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm
    rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    global EVAL_PROMPT_DATASET

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            use_audio_in_video=args.use_audio_in_video,
            system_prompt=args.system_prompt,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    sample_index = 0

    if args.group_rm:
        # group_rm mode: group samples by prompt and use generate_and_rm_group
        # so that the RM can see all responses for the same prompt together.
        tasks = []
        for _i, prompt_sample in enumerate(dataset.samples):
            group = []
            for j in range(dataset_cfg.n_samples_per_eval_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.index = sample_index
                sample_index += 1
                sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
                sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
                group.append(sample)
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed
            tasks.append(
                asyncio.create_task(
                    generate_and_rm_group(args, group, sampling_params=sampling_params, evaluation=True)
                )
            )

        data = []
        do_print = True
        pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
        for coro in asyncio.as_completed(tasks):
            group = await coro
            if do_print:
                sample = group[0]
                logger.info(
                    "eval_rollout_single_dataset example data: "
                    f"{[str(sample.prompt) + sample.response]} "
                    f"reward={sample.reward}"
                )
                do_print = False
            data.extend(group)
            pbar.update(1)
        pbar.close()
    else:
        tasks = []
        for _i, prompt_sample in enumerate(dataset.samples):
            for j in range(dataset_cfg.n_samples_per_eval_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.index = sample_index
                sample_index += 1
                sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
                sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
                sampling_params = base_sampling_params
                if getattr(args, "sglang_enable_deterministic_inference", False):
                    sampling_params = base_sampling_params.copy()
                    sampling_params["sampling_seed"] = args.rollout_seed + j
                tasks.append(
                    asyncio.create_task(
                        generate_and_rm(args, sample, sampling_params=sampling_params, evaluation=True)
                    )
                )

        data = []
        do_print = True
        pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
        for coro in asyncio.as_completed(tasks):
            sample = await coro
            if do_print:
                logger.info(
                    "eval_rollout_single_dataset example data: "
                    f"{[str(sample.prompt) + sample.response]} "
                    f"reward={sample.reward}"
                )
                do_print = False
            if isinstance(sample, list):
                data.extend(sample)
            else:
                data.append(sample)
            pbar.update(1)
        pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key

    def _extract_eval_reward(sample: Sample) -> Any:
        reward = sample.reward
        if reward is None:
            logger.warning(
                "Eval sample has no reward; counting as 0.0. "
                f"index={getattr(sample, 'index', None)} status={getattr(sample, 'status', None)}"
            )
            return 0.0
        if not reward_key:
            return reward
        if isinstance(reward, dict) and reward_key in reward:
            try:
                return float(reward[reward_key])
            except (TypeError, ValueError):
                logger.warning(
                    "Eval sample reward value is not numeric; counting as 0.0. "
                    f"index={getattr(sample, 'index', None)} reward_key={reward_key} reward={reward}"
                )
                return 0.0
        logger.warning(
            "Eval sample reward does not contain reward_key; counting as 0.0. "
            f"index={getattr(sample, 'index', None)} reward_key={reward_key} reward={reward}"
        )
        return 0.0

    return {
        dataset_cfg.name: {
            "rewards": [_extract_eval_reward(sample) for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


def generate_rollout(
    args: Namespace, rollout_id: int, data_buffer: Any, data_system_client: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based
    rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        data_system_client: the data system client to use for transferring batches
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        RolloutFnTrainOutput | RolloutFnEvalOutput: the output of the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_buffer, data_system_client))
    data_buffer.add_samples.remote(aborted_samples)
    return output
