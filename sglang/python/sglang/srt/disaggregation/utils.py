from __future__ import annotations

import os
import random
from collections import deque
from contextlib import nullcontext
from enum import Enum
from typing import TYPE_CHECKING, Optional, Type

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

#########################
# Constants & Enums
#########################
FAKE_BOOTSTRAP_HOST = "2.2.2.2"
PREFILL_TIMING_AUX_BUFFER_NAME = "prefill_timing"
PREFILL_TIMING_DEST_ATTRS = (
    ("fwd_prefill_bootstrap_queue_duration", float),
    ("fwd_prefill_forward_duration", float),
    ("fwd_prefill_transfer_queue_duration", float),
    ("fwd_bootstrap_duration", float),
    ("fwd_alloc_waiting_duration", float),
    ("fwd_transfer_speed_gb_s", float),
    ("fwd_transfer_total_mb", float),
    ("fwd_prefill_retry_count", int),
)


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


#########################
# Synchronization
#########################

# env var for testing failure, convert to float explicitly
FAILURE_PROB = float(os.getenv("DISAGGREGATION_TEST_FAILURE_PROB", 0))


def poll_and_all_reduce(pollers, gloo_group):
    # at a certain prob, the poll is failed to simulate failure
    if FAILURE_PROB > 0:
        from sglang.srt.disaggregation.base import KVPoll

        polls = [
            int(KVPoll.Failed) if random.random() < FAILURE_PROB else int(poller.poll())
            for poller in pollers
        ]
    else:
        polls = [int(poller.poll()) for poller in pollers]
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
    return tensor_to_reduce.tolist()


#########################
# Metadata Buffers
#########################


class ReqToMetadataIdxAllocator:
    """A memory pool that maps a request to its first output token location."""

    def __init__(
        self,
        size: int,
    ):
        self.size = size
        self.free_slots = deque(list(range(size)))

    def available_size(self):
        return len(self.free_slots)

    def alloc(self) -> Optional[int]:
        if len(self.free_slots) == 0:
            return None

        return self.free_slots.popleft()

    def free(self, free_index: int):
        self.free_slots.append(free_index)


class MetadataBuffers:
    def __init__(
        self,
        size: int,
        hidden_size: int,
        hidden_states_dtype: torch.dtype,
        max_top_logprobs_num: int = 128,
        custom_mem_pool: torch.cuda.MemPool = None,
    ):
        self.custom_mem_pool = custom_mem_pool
        device = "cpu"
        if is_npu():
            # For ascend backend, output tokens are placed in the NPU and will be transferred by D2D channel.
            device = "npu"
        elif self.custom_mem_pool:
            # TODO(shangming): Fix me (use 'cuda') when nvlink_transport of Mooncake is bug-free
            device = "cpu"
        elif envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() == "INTRA_NODE_NVLINK":
            device = "cuda"
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            # TODO: abort top_logprobs_num > 128 in PD

            # We transfer the metadata of first output token to decode
            # The minimal size for RDMA is 64Bytes, so we pad it to > 64Bytes
            self.output_ids = torch.zeros((size, 16), dtype=torch.int32, device=device)
            self.cached_tokens = torch.zeros(
                (size, 16), dtype=torch.int32, device=device
            )
            self.output_token_logprobs_val = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
            self.output_token_logprobs_idx = torch.zeros(
                (size, 16), dtype=torch.int32, device=device
            )
            self.output_top_logprobs_val = torch.zeros(
                (size, max_top_logprobs_num), dtype=torch.float32, device=device
            )
            self.output_top_logprobs_idx = torch.zeros(
                (size, max_top_logprobs_num), dtype=torch.int32, device=device
            )
            # For PD + spec decode
            self.output_topk_p = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
            self.output_topk_index = torch.zeros(
                (size, 16), dtype=torch.int64, device=device
            )
            self.output_hidden_states = torch.zeros(
                (size, hidden_size), dtype=hidden_states_dtype, device=device
            )
            # Request validation: store bootstrap_room to detect metadata corruption
            self.bootstrap_room = torch.zeros(
                (size, 8), dtype=torch.uint64, device=device
            )
            # Prefill-side PD timing (8 floats, padded to 16 for RDMA alignment).
            # Layout: [bootstrap_queue, forward, transfer_queue, bootstrap,
            #          alloc_waiting, transfer_speed, transfer_mb, retry_count]
            self.prefill_timing = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
        self.aux_buffers = [
            ("output_ids", self.output_ids),
            ("cached_tokens", self.cached_tokens),
            ("output_token_logprobs_val", self.output_token_logprobs_val),
            ("output_token_logprobs_idx", self.output_token_logprobs_idx),
            ("output_top_logprobs_val", self.output_top_logprobs_val),
            ("output_top_logprobs_idx", self.output_top_logprobs_idx),
            ("output_topk_p", self.output_topk_p),
            ("output_topk_index", self.output_topk_index),
            ("output_hidden_states", self.output_hidden_states),
            ("bootstrap_room", self.bootstrap_room),
            (PREFILL_TIMING_AUX_BUFFER_NAME, self.prefill_timing),
        ]

    def get_buf_infos(self):
        ptrs = [buffer.data_ptr() for _, buffer in self.aux_buffers]
        data_lens = [buffer.nbytes for _, buffer in self.aux_buffers]
        item_lens = [buffer[0].nbytes for _, buffer in self.aux_buffers]
        return ptrs, data_lens, item_lens

    def get_aux_buffer_names(self):
        return [name for name, _ in self.aux_buffers]

    def get_buf(self, idx: int):
        return (
            self.output_ids[idx],
            self.cached_tokens[idx],
            self.output_token_logprobs_val[idx],
            self.output_token_logprobs_idx[idx],
            self.output_top_logprobs_val[idx],
            self.output_top_logprobs_idx[idx],
            self.output_topk_p[idx],
            self.output_topk_index[idx],
            self.output_hidden_states[idx],
            self.bootstrap_room[idx],
            self.prefill_timing[idx],
        )

    def clear_profiling_buf(self, idx: int):
        self.prefill_timing[idx].zero_()

    def set_buf(self, req: Req):

        self.output_ids[req.metadata_buffer_index][0] = req.output_ids[0]
        self.cached_tokens[req.metadata_buffer_index][0] = req.cached_tokens
        if req.return_logprob:
            if req.output_token_logprobs_val:  # not none or empty list
                self.output_token_logprobs_val[req.metadata_buffer_index][0] = (
                    req.output_token_logprobs_val[0]
                )
            if req.output_token_logprobs_idx:  # not none or empty list
                self.output_token_logprobs_idx[req.metadata_buffer_index][0] = (
                    req.output_token_logprobs_idx[0]
                )

            if req.output_top_logprobs_val:  # not none or empty list
                self.output_top_logprobs_val[req.metadata_buffer_index][
                    : len(req.output_top_logprobs_val[0])
                ] = torch.tensor(
                    req.output_top_logprobs_val[0], dtype=torch.float32, device="cpu"
                )
            if req.output_top_logprobs_idx:  # not none or empty list
                self.output_top_logprobs_idx[req.metadata_buffer_index][
                    : len(req.output_top_logprobs_idx[0])
                ] = torch.tensor(
                    req.output_top_logprobs_idx[0], dtype=torch.int32, device="cpu"
                )
        # For PD + spec decode
        if req.hidden_states_tensor is not None:
            # speculative_eagle_topk should not be greater than 16 currently
            topk = req.output_topk_p.size(0)

            self.output_topk_p[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_p
            )
            self.output_topk_index[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_index
            )
            self.output_hidden_states[req.metadata_buffer_index].copy_(
                req.hidden_states_tensor
            )
        # Store bootstrap_room for validation on decode side
        self.bootstrap_room[req.metadata_buffer_index, 0] = (
            req.bootstrap_room if req.bootstrap_room is not None else 0
        )
        # Pack prefill-side PD timing durations for transfer to decode instance.
        # Note: set_buf is called at the START of the last KV chunk send, so
        # completion_time and prefill_transfer_queue_entry_time are not yet set.
        # We use time.perf_counter() as the "forward just completed" timestamp.
        import time

        ts = req.time_stats
        timing = self.prefill_timing[req.metadata_buffer_index]
        self.clear_profiling_buf(req.metadata_buffer_index)
        if not is_slime_profiling_enabled():
            return
        for idx, value in enumerate(
            build_prefill_timing_payload(ts, now=time.perf_counter())
        ):
            if value > 0:
                timing[idx] = value


def is_slime_profiling_enabled() -> bool:
    return envs.SLIME_ENABLE_PROFILING.get()


def build_prefill_timing_payload(time_stats, now: float) -> tuple[float, ...]:
    bootstrap_queue_duration = 0.0
    if (
        time_stats.prefill_bootstrap_queue_entry_time > 0
        and time_stats.wait_queue_entry_time > 0
    ):
        bootstrap_queue_duration = (
            time_stats.wait_queue_entry_time
            - time_stats.prefill_bootstrap_queue_entry_time
        )

    prefill_forward_duration = (
        now - time_stats.forward_entry_time
        if time_stats.forward_entry_time > 0
        else 0.0
    )

    return (
        bootstrap_queue_duration,
        prefill_forward_duration,
        0.0,
        max(0.0, time_stats.bootstrap_duration),
        max(0.0, time_stats.alloc_waiting_duration),
        max(0.0, time_stats.transfer_speed_gb_s),
        max(0.0, time_stats.transfer_total_mb),
        float(max(0, time_stats.prefill_retry_count)),
    )


def apply_prefill_timing_payload(time_stats, timing) -> None:
    for value, (attr_name, caster) in zip(
        timing[: len(PREFILL_TIMING_DEST_ATTRS)].tolist(),
        PREFILL_TIMING_DEST_ATTRS,
    ):
        if value > 0:
            setattr(time_stats, attr_name, caster(value))


def iter_aux_transfer_specs(
    aux_buffer_names: list[str],
    prefill_aux_ptrs: list[int],
    prefill_aux_item_lens: list[int],
    dst_aux_ptrs: list[int],
    prefill_aux_index: int,
    dst_aux_index: int,
):
    profiling_enabled = is_slime_profiling_enabled()
    for i, (buffer_name, dst_aux_ptr) in enumerate(zip(aux_buffer_names, dst_aux_ptrs)):
        if not profiling_enabled and buffer_name == PREFILL_TIMING_AUX_BUFFER_NAME:
            continue
        length = prefill_aux_item_lens[i]
        if length <= 0:
            continue
        src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
        dst_addr = dst_aux_ptr + length * dst_aux_index
        yield i, src_addr, dst_addr, length


#########################
# Transfer Backend
#########################


class TransferBackend(Enum):
    MOONCAKE = "mooncake"
    MORI = "mori"
    NIXL = "nixl"
    ASCEND = "ascend"
    FAKE = "fake"


class KVClassType(Enum):
    KVARGS = "kvargs"
    MANAGER = "manager"
    SENDER = "sender"
    RECEIVER = "receiver"
    BOOTSTRAP_SERVER = "bootstrap_server"


def get_kv_class(
    transfer_backend: TransferBackend, class_type: KVClassType
) -> Optional[Type]:
    from sglang.srt.disaggregation.fake import FakeKVReceiver, FakeKVSender

    if transfer_backend == TransferBackend.MOONCAKE:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.mooncake import (
            MooncakeKVBootstrapServer,
            MooncakeKVManager,
            MooncakeKVReceiver,
            MooncakeKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: MooncakeKVManager,
            KVClassType.SENDER: MooncakeKVSender,
            KVClassType.RECEIVER: (MooncakeKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MooncakeKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.MORI:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.mori import (
            MoriKVBootstrapServer,
            MoriKVManager,
            MoriKVReceiver,
            MoriKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: MoriKVManager,
            KVClassType.SENDER: MoriKVSender,
            KVClassType.RECEIVER: (MoriKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MoriKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.ASCEND:
        from sglang.srt.disaggregation.ascend import (
            AscendKVBootstrapServer,
            AscendKVManager,
            AscendKVReceiver,
            AscendKVSender,
        )
        from sglang.srt.disaggregation.base import KVArgs

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: AscendKVManager,
            KVClassType.SENDER: AscendKVSender,
            KVClassType.RECEIVER: (AscendKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: AscendKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.NIXL:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.nixl import (
            NixlKVBootstrapServer,
            NixlKVManager,
            NixlKVReceiver,
            NixlKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: NixlKVManager,
            KVClassType.SENDER: NixlKVSender,
            KVClassType.RECEIVER: (NixlKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: NixlKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.FAKE:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.fake import (
            FakeKVManager,
            FakeKVReceiver,
            FakeKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: FakeKVManager,
            KVClassType.SENDER: FakeKVSender,
            KVClassType.RECEIVER: (FakeKVReceiver),
        }
        return class_mapping.get(class_type)

    raise ValueError(f"Unsupported transfer backend: {transfer_backend}")


#########################
# KV Pages
#########################


def kv_to_page_indices(kv_indices: np.ndarray, page_size: int):
    # 1. The page is guaranteed to be full except the last page.
    # 2. page index = kv_index // page_size
    # The return vector is kv_indices[::page_size] // page_size
    if page_size == 1:  # shortcut
        return kv_indices

    return kv_indices[::page_size] // page_size


def kv_to_page_num(num_kv_indices: int, page_size: int):
    # ceil(num_kv_indices / page_size)
    return (num_kv_indices + page_size - 1) // page_size


#########################
# Misc
#########################


def is_mla_backend(target_kv_pool) -> bool:
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    return isinstance(target_kv_pool, MLATokenToKVPool)


def prepare_abort(req: Req, error_message: str, status_code=None):
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    # populate finish metadata and stream output
    req.finished_reason = FINISH_ABORT(error_message, status_code)

    if req.return_logprob:
        req.input_token_logprobs_val = []
        req.input_token_logprobs_idx = []
        req.input_top_logprobs_val = []
        req.input_top_logprobs_idx = []
        req.input_token_ids_logprobs_val = []
        req.input_token_ids_logprobs_idx = []
