#!/usr/bin/env python3
"""Convert CP2-sharded Qwen3.5 Megatron DCP checkpoints to HuggingFace.

Purpose:
  Export Relax/Megatron torch_dist checkpoints saved with TP4 x CP2 into a
  HuggingFace model directory that Transformers and SGLang can load.

Resume behavior:
  Re-run with --force to replace a partial output directory. The input
  checkpoint is read-only and no vendored Megatron files are modified.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import types
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

# Must happen before importing megatron/transformer_engine.
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.9")
os.environ.setdefault("CUDA_PATH", "/usr/local/cuda-12.9")
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/flashinfer")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")
os.environ.setdefault("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface/transformers")
os.environ.setdefault("MEGATRON_CONFIG_LOCK_DIR", "/tmp/huggingface/locks")

from transformers import AutoConfig, PretrainedConfig  # noqa: E402


def _install_glm_stub() -> None:
    """Avoid unrelated GLM bridge imports requiring newer Transformers."""

    glm_mod = types.ModuleType("megatron.bridge.models.glm_moe_dsa")
    glm_mod.__file__ = "<glm_moe_dsa_dummy>"
    glm_mod.__package__ = "megatron.bridge.models"
    glm_mod.__path__ = []

    class GlmMoeDsaForCausalLM:  # noqa: D401
        """Dummy placeholder; GLM is irrelevant for Qwen3.5 export."""

    class GLM5Bridge:  # noqa: D401
        """Dummy placeholder; GLM is irrelevant for Qwen3.5 export."""

    glm_mod.GlmMoeDsaForCausalLM = GlmMoeDsaForCausalLM
    glm_mod.GLM5Bridge = GLM5Bridge
    sys.modules.setdefault("megatron.bridge.models.glm_moe_dsa", glm_mod)


_install_glm_stub()


# Checkpoint common.pt may pickle TransferQueue sampler classes. They are not
# needed for HF export; provide lazy dummy classes so torch.load can recover args.
class _LazyTransferQueueModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        cls = type(
            name,
            (),
            {"__module__": self.__name__, "__init__": lambda self, *args, **kwargs: None},
        )
        setattr(self, name, cls)
        return cls


def _make_lazy_tq_module(module_name: str):
    mod = _LazyTransferQueueModule(module_name)
    mod.__file__ = f"<{module_name}_dummy>"
    mod.__package__ = module_name.rpartition(".")[0] or module_name
    mod.__path__ = []
    return mod


class _TransferQueueDummyLoader:
    def create_module(self, spec):
        return _make_lazy_tq_module(spec.name)

    def exec_module(self, module):
        return None


class _TransferQueueDummyFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "transfer_queue" or fullname.startswith("transfer_queue."):
            from importlib.machinery import ModuleSpec

            return ModuleSpec(fullname, _TransferQueueDummyLoader(), is_package=True)
        return None


sys.meta_path.insert(0, _TransferQueueDummyFinder())
sys.modules.setdefault("transfer_queue", _make_lazy_tq_module("transfer_queue"))


class _AttrConfig(PretrainedConfig):
    model_type = "qwen3_5_attr"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class Qwen35Config(PretrainedConfig):
    model_type = "qwen3_5"

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        super().__init__(**kwargs)
        self.text_config = _AttrConfig(**(text_config or {}))
        self.vision_config = _AttrConfig(**(vision_config or {}))
        for key, value in kwargs.items():
            setattr(self, key, value)


try:
    AutoConfig.register("qwen3_5", Qwen35Config)
except ValueError:
    pass


import megatron.bridge.training.model_load_save as _model_load_save_module  # noqa: E402
from megatron.bridge import AutoBridge  # noqa: E402


class _FakeWork:
    def wait(self):
        return None


class _FakeProcessGroup:
    def __init__(self, ranks=None):
        self.ranks = list(ranks or [0])

    def size(self):
        return len(self.ranks)

    def rank(self):
        return self.ranks.index(0) if 0 in self.ranks else 0


@contextmanager
def _single_rank_distributed_context(backend: str = "gloo"):
    """Single-rank distributed facade for socket-restricted export sandboxes."""

    import torch.distributed as dist
    from megatron.core import parallel_state

    fake_world = _FakeProcessGroup([0])

    def _fake_init_process_group(*args, **kwargs):
        return None

    def _fake_destroy_process_group(group=None):
        return None

    def _fake_is_initialized():
        return True

    def _fake_is_available():
        return True

    def _fake_get_rank(group=None):
        if isinstance(group, _FakeProcessGroup):
            return group.rank()
        return 0

    def _fake_get_world_size(group=None):
        if isinstance(group, _FakeProcessGroup):
            return group.size()
        return 1

    def _fake_get_backend(group=None):
        return backend

    def _fake_new_group(
        ranks=None,
        timeout=None,
        backend=None,
        pg_options=None,
        use_local_synchronization=False,
        group_desc=None,
    ):
        return _FakeProcessGroup(ranks or [0])

    def _fake_get_process_group_ranks(group):
        if isinstance(group, _FakeProcessGroup):
            return list(group.ranks)
        return [0]

    def _fake_get_global_rank(group, group_rank):
        if isinstance(group, _FakeProcessGroup):
            return group.ranks[group_rank]
        return group_rank

    def _fake_barrier(*args, **kwargs):
        return None

    def _fake_broadcast(tensor, src, group=None, async_op=False):
        return _FakeWork() if async_op else None

    def _fake_all_reduce(tensor, op=None, group=None, async_op=False):
        return _FakeWork() if async_op else None

    patch_names = {
        "init_process_group": _fake_init_process_group,
        "destroy_process_group": _fake_destroy_process_group,
        "is_initialized": _fake_is_initialized,
        "is_available": _fake_is_available,
        "get_rank": _fake_get_rank,
        "get_world_size": _fake_get_world_size,
        "get_backend": _fake_get_backend,
        "new_group": _fake_new_group,
        "get_process_group_ranks": _fake_get_process_group_ranks,
        "get_global_rank": _fake_get_global_rank,
        "barrier": _fake_barrier,
        "broadcast": _fake_broadcast,
        "all_reduce": _fake_all_reduce,
    }
    originals = {name: getattr(dist, name) for name in patch_names}

    for name, replacement in patch_names.items():
        setattr(dist, name, replacement)

    old_world_sentinel = object()
    old_world = getattr(dist.group, "WORLD", old_world_sentinel)
    try:
        dist.group.WORLD = fake_world
    except Exception:
        old_world = old_world_sentinel

    try:
        parallel_state.initialize_model_parallel(create_gloo_process_groups=False)
        print("[convert] Using single-rank fake distributed context", flush=True)
        yield
    finally:
        try:
            parallel_state.destroy_model_parallel()
        finally:
            for name, original in originals.items():
                setattr(dist, name, original)
            try:
                if old_world is old_world_sentinel:
                    delattr(dist.group, "WORLD")
                else:
                    dist.group.WORLD = old_world
            except Exception:
                pass


def _install_fake_distributed_context() -> None:
    """Use only in socket-restricted rescue shells.

    The normal eval host can initialize a real one-rank Gloo process group.
    Keeping that path matters because torch.distributed.checkpoint expects a
    real ProcessGroup during Bridge export. A fake group can trip the loader and
    accidentally force a debug-only direct export.
    """

    _model_load_save_module.temporary_distributed_context = _single_rank_distributed_context


def _is_extra_state_group(sharded_key: object, recovered_keys: Iterable[object]) -> bool:
    keys = [str(sharded_key)]
    keys.extend(str(key) for key in recovered_keys)
    return any("_extra_state" in key for key in keys)


def _install_cp2_extra_state_patch() -> None:
    """Drop TE _extra_state object payloads during DCP key restoration.

    CP2 checkpoints contain one BytesIO ShardedObject per CP rank for TE
    _extra_state. Single-process export asks torch.distributed.checkpoint to
    restore into a CP1 view, and MCore's key restoration assumes list-like
    values. The TE object payloads are not needed for Bridge HF export, so this
    patch records placeholder keys for shape/type restoration and lets the
    Bridge checkpointing layer delete them before model load.
    """

    import megatron.core.dist_checkpointing.strategies.torch as torch_strategy

    if getattr(torch_strategy, "_cp2_extra_state_patch_installed", False):
        return

    def _patched_replace_sharded_keys_with_state_dict_keys(
        state_dict,
        flat_mapping,
        rename_mapping,
    ):
        recovered_sd = {}
        skipped_extra_state = 0

        for sharded_key, tensors in state_dict.items():
            recovered_keys = rename_mapping[sharded_key]
            if _is_extra_state_group(sharded_key, recovered_keys):
                for recovered_key in recovered_keys:
                    recovered_sd[recovered_key] = None
                    skipped_extra_state += 1
                continue

            if isinstance(tensors, io.BytesIO):
                if len(recovered_keys) != 1:
                    raise TypeError(
                        "Non-extra-state BytesIO restore group has "
                        f"{len(recovered_keys)} destination keys: {sharded_key}"
                    )
                tensors = [tensors]

            if not hasattr(tensors, "__len__"):
                if len(recovered_keys) != 1:
                    raise TypeError(
                        "Non-list restore group has "
                        f"{len(recovered_keys)} destination keys: {sharded_key}"
                    )
                tensors = [tensors]

            assert len(tensors) == len(recovered_keys), (
                sharded_key,
                len(tensors),
                len(recovered_keys),
            )
            for tensor, recovered_key in zip(tensors, recovered_keys):
                recovered_sd[recovered_key] = tensor

        if skipped_extra_state:
            print(
                "[convert] CP2 patch replaced "
                f"{skipped_extra_state} _extra_state payloads with placeholders",
                flush=True,
            )

        return torch_strategy.unflatten_state_dict(recovered_sd, flat_mapping)

    torch_strategy._replace_sharded_keys_with_state_dict_keys = (
        _patched_replace_sharded_keys_with_state_dict_keys
    )
    torch_strategy._cp2_extra_state_patch_installed = True
    print("[convert] Installed CP2 _extra_state DCP restore patch", flush=True)


_install_cp2_extra_state_patch()


# Patch Qwen3.5 provider availability gate. The original provider only needs a
# default vision config if one is absent; provider_bridge replaces it with the
# actual config loaded from HF config.json.
import megatron.bridge.models.qwen_vl.qwen35_vl_provider as _qwen35_provider  # noqa: E402

_qwen35_provider._TRANSFORMERS_HAS_QWEN3_5 = True


class Qwen3_5VisionConfig(_AttrConfig):
    pass


_qwen35_provider.Qwen3_5VisionConfig = Qwen3_5VisionConfig

_provider_override = {}
_original_load_model_config = _model_load_save_module.load_model_config


def _patched_load_model_config(checkpoint_path):
    model_cfg, mlm_args = _original_load_model_config(checkpoint_path)
    provider = _provider_override.get("provider")
    if provider is not None:
        from megatron.bridge.models.model_provider import ModelProviderMixin

        if not isinstance(model_cfg, ModelProviderMixin):
            print(
                "[convert] Overriding MLM TransformerConfig with Bridge provider: "
                f"{type(provider).__name__}",
                flush=True,
            )
            return provider, mlm_args
    return model_cfg, mlm_args


_model_load_save_module.load_model_config = _patched_load_model_config


class _DirectQwen35Config:
    hidden_size = 4096
    num_attention_heads = 16
    num_query_groups = 4
    kv_channels = 256
    attention_output_gate = True
    linear_key_head_dim = 128
    linear_value_head_dim = 128
    linear_num_key_heads = 16
    linear_num_value_heads = 32


def _is_linear_attention_layer(layer_idx: int) -> bool:
    return layer_idx % 4 != 3


def _copy_hf_assets(origin_hf_dir: Path, output_dir: Path) -> None:
    for src in origin_hf_dir.iterdir():
        if not src.is_file():
            continue
        name = src.name
        if name.endswith(".safetensors") or name == "model.safetensors.index.json":
            continue
        if name == "config.json":
            config = json.loads(src.read_text())
            (output_dir / name).write_text(json.dumps(config, indent=4) + "\n")
            continue
        shutil.copy2(src, output_dir / name)


def _normalize_qwen35_text_config(origin_hf_dir: Path, output_dir: Path) -> None:
    """Preserve the qwen3_5 text config expected by SGLang.

    The local Transformers stub used for conversion gives nested text_config a
    placeholder model_type. Bridge export writes a valid text-architecture
    config otherwise, so normalize only the architecture identifiers from the
    original HF model.
    """

    origin_config_path = origin_hf_dir / "config.json"
    output_config_path = output_dir / "config.json"
    if not origin_config_path.exists() or not output_config_path.exists():
        return

    origin_config = json.loads(origin_config_path.read_text())
    output_config = json.loads(output_config_path.read_text())
    if origin_config.get("model_type") != "qwen3_5":
        return

    output_config["model_type"] = "qwen3_5"
    output_config["architectures"] = origin_config.get(
        "architectures",
        ["Qwen3_5ForConditionalGeneration"],
    )

    origin_text_config = origin_config.get("text_config")
    output_text_config = output_config.get("text_config")
    if isinstance(origin_text_config, dict) and isinstance(output_text_config, dict):
        output_text_config["model_type"] = origin_text_config.get(
            "model_type",
            "qwen3_5_text",
        )

    output_config_path.write_text(json.dumps(output_config, indent=4) + "\n")


def _direct_export_dcp_to_hf(
    input_dir: str,
    origin_hf_dir: str,
    output_dir: str,
    force: bool,
) -> None:
    """Export by loading DCP tensors directly and writing HF safetensors."""

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from torch.distributed import checkpoint
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    from megatron.bridge.models.conversion.param_mapping import split_qkv_weights

    input_path = Path(input_dir)
    origin_path = Path(origin_hf_dir)
    output_path = Path(output_dir)
    if output_path.exists():
        if not force:
            raise ValueError(f"Output directory exists: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    _copy_hf_assets(origin_path, output_path)
    _normalize_qwen35_text_config(origin_path, output_path)

    index_path = origin_path / "model.safetensors.index.json"
    base_index = json.loads(index_path.read_text())
    weight_map = base_index["weight_map"]
    by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        by_file[filename].append(key)

    reader = FileSystemReader(str(input_path))
    dcp_metadata = reader.read_metadata()
    tensor_metadata = {
        str(key): value
        for key, value in dcp_metadata.state_dict_metadata.items()
        if isinstance(value, TensorStorageMetadata)
    }

    cfg = _DirectQwen35Config()
    dcp_cache: dict[str, torch.Tensor] = {}

    def load_dcp(key: str) -> torch.Tensor:
        if key not in dcp_cache:
            if key not in tensor_metadata:
                raise KeyError(f"DCP tensor not found: {key}")
            meta = tensor_metadata[key]
            state_dict = {key: torch.empty(tuple(meta.size), dtype=meta.properties.dtype)}
            checkpoint.load(state_dict, reader, no_dist=True)
            dcp_cache[key] = state_dict[key]
        return dcp_cache[key]

    simple_map = {
        "model.language_model.embed_tokens.weight": "language_model.embedding.word_embeddings.weight",
        "lm_head.weight": "language_model.output_layer.weight",
        "model.language_model.norm.weight": "language_model.decoder.final_layernorm.weight",
        "model.visual.patch_embed.proj.weight": "vision_model.patch_embed.proj.weight",
        "model.visual.patch_embed.proj.bias": "vision_model.patch_embed.proj.bias",
        "model.visual.pos_embed.weight": "vision_model.pos_embed.weight",
        "model.visual.merger.norm.weight": "vision_model.merger.patch_norm.weight",
        "model.visual.merger.norm.bias": "vision_model.merger.patch_norm.bias",
        "model.visual.merger.linear_fc1.weight": "vision_model.merger.linear_fc1.weight",
        "model.visual.merger.linear_fc1.bias": "vision_model.merger.linear_fc1.bias",
        "model.visual.merger.linear_fc2.weight": "vision_model.merger.linear_fc2.weight",
        "model.visual.merger.linear_fc2.bias": "vision_model.merger.linear_fc2.bias",
    }

    def make_language_tensor(layer: int, suffix: str) -> torch.Tensor | None:
        prefix = f"language_model.decoder.layers.{layer}."
        if suffix == "input_layernorm.weight":
            source = (
                "self_attention.in_proj.layer_norm_weight"
                if _is_linear_attention_layer(layer)
                else "self_attention.linear_qkv.layer_norm_weight"
            )
            return load_dcp(prefix + source)
        if suffix == "post_attention_layernorm.weight":
            return load_dcp(prefix + "mlp.linear_fc1.layer_norm_weight")
        if suffix == "mlp.down_proj.weight":
            return load_dcp(prefix + "mlp.linear_fc2.weight")
        if suffix in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
            gate, up = torch.chunk(load_dcp(prefix + "mlp.linear_fc1.weight"), 2, dim=0)
            return gate if suffix == "mlp.gate_proj.weight" else up

        if _is_linear_attention_layer(layer):
            attn = "self_attention."
            if suffix == "linear_attn.dt_bias":
                return load_dcp(prefix + attn + "dt_bias")
            if suffix == "linear_attn.A_log":
                return load_dcp(prefix + attn + "A_log")
            if suffix == "linear_attn.in_proj_qkv.weight":
                return torch.cat(
                    [
                        load_dcp(prefix + attn + "in_proj.weight.query"),
                        load_dcp(prefix + attn + "in_proj.weight.key"),
                        load_dcp(prefix + attn + "in_proj.weight.value"),
                    ],
                    dim=0,
                )
            if suffix == "linear_attn.in_proj_z.weight":
                return load_dcp(prefix + attn + "in_proj.weight.z")
            if suffix == "linear_attn.in_proj_b.weight":
                return load_dcp(prefix + attn + "in_proj.weight.beta")
            if suffix == "linear_attn.in_proj_a.weight":
                return load_dcp(prefix + attn + "in_proj.weight.alpha")
            if suffix == "linear_attn.conv1d.weight":
                return torch.cat(
                    [
                        load_dcp(prefix + attn + "conv1d.weight.query"),
                        load_dcp(prefix + attn + "conv1d.weight.key"),
                        load_dcp(prefix + attn + "conv1d.weight.value"),
                    ],
                    dim=0,
                )
            if suffix == "linear_attn.norm.weight":
                return load_dcp(prefix + attn + "out_norm.weight")
            if suffix == "linear_attn.out_proj.weight":
                return load_dcp(prefix + attn + "out_proj.weight")
            return None

        attn = "self_attention."
        if suffix in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
        ):
            q, k, v = split_qkv_weights(cfg, load_dcp(prefix + attn + "linear_qkv.weight"))
            return {
                "self_attn.q_proj.weight": q,
                "self_attn.k_proj.weight": k,
                "self_attn.v_proj.weight": v,
            }[suffix]
        if suffix == "self_attn.o_proj.weight":
            return load_dcp(prefix + attn + "linear_proj.weight")
        if suffix == "self_attn.q_norm.weight":
            return load_dcp(prefix + attn + "q_layernorm.weight")
        if suffix == "self_attn.k_norm.weight":
            return load_dcp(prefix + attn + "k_layernorm.weight")
        return None

    def make_visual_block_tensor(layer: int, suffix: str) -> torch.Tensor | None:
        prefix = "vision_model.decoder.layers."
        mapping = {
            "attn.proj.weight": "self_attention.linear_proj.weight",
            "attn.proj.bias": "self_attention.linear_proj.bias",
            "attn.qkv.weight": "self_attention.linear_qkv.weight",
            "attn.qkv.bias": "self_attention.linear_qkv.bias",
            "norm1.weight": "self_attention.linear_qkv.layer_norm_weight",
            "norm1.bias": "self_attention.linear_qkv.layer_norm_bias",
            "norm2.weight": "mlp.linear_fc1.layer_norm_weight",
            "norm2.bias": "mlp.linear_fc1.layer_norm_bias",
            "mlp.linear_fc1.weight": "mlp.linear_fc1.weight",
            "mlp.linear_fc1.bias": "mlp.linear_fc1.bias",
            "mlp.linear_fc2.weight": "mlp.linear_fc2.weight",
            "mlp.linear_fc2.bias": "mlp.linear_fc2.bias",
        }
        source = mapping.get(suffix)
        if source is None:
            return None
        return load_dcp(prefix + source)[layer].contiguous()

    def make_tensor(hf_key: str) -> torch.Tensor | None:
        if hf_key in simple_map:
            return load_dcp(simple_map[hf_key])

        lang_prefix = "model.language_model.layers."
        if hf_key.startswith(lang_prefix):
            rest = hf_key[len(lang_prefix) :]
            layer_str, suffix = rest.split(".", 1)
            return make_language_tensor(int(layer_str), suffix)

        visual_prefix = "model.visual.blocks."
        if hf_key.startswith(visual_prefix):
            rest = hf_key[len(visual_prefix) :]
            layer_str, suffix = rest.split(".", 1)
            return make_visual_block_tensor(int(layer_str), suffix)

        return None

    total_tensors = 0
    base_fallback_tensors = []
    for filename, keys in by_file.items():
        dcp_cache.clear()
        tensors = {}
        with safe_open(str(origin_path / filename), framework="pt", device="cpu") as base_sf:
            for hf_key in keys:
                tensor = make_tensor(hf_key)
                if tensor is None:
                    tensor = base_sf.get_tensor(hf_key)
                    base_fallback_tensors.append(hf_key)
                expected_shape = tuple(base_sf.get_slice(hf_key).get_shape())
                if tuple(tensor.shape) != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for {hf_key}: got {tuple(tensor.shape)}, expected {expected_shape}"
                    )
                tensors[hf_key] = tensor.contiguous()

        save_file(tensors, str(output_path / filename))
        total_tensors += len(tensors)
        print(f"[convert-direct] wrote {filename}: {len(tensors)} tensors", flush=True)

    (output_path / "model.safetensors.index.json").write_text(json.dumps(base_index, indent=2) + "\n")
    print(
        "[convert-direct] used base HF tensors for "
        f"{len(base_fallback_tensors)} keys: {', '.join(base_fallback_tensors)}",
        flush=True,
    )
    print(f"[convert-direct] wrote {total_tensors} HF tensors to {output_path}", flush=True)


def _stamp_export_source(input_dir: str, origin_hf_dir: str, output_dir: str, mode: str) -> None:
    """Write export_source.json into the HF output dir: checkpoint lineage for
    eval manifests to join against (2026-06-12 repo-management patch)."""
    import getpass
    import socket
    from datetime import datetime, timezone
    info = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "input_ckpt_dir": str(Path(input_dir).resolve()),
        "origin_hf_dir": str(Path(origin_hf_dir).resolve()),
        "mode": mode,
        "host": socket.gethostname(),
        "user": getpass.getuser(),
    }
    lineage = Path(input_dir).resolve()
    for parent in [lineage, *lineage.parents]:
        cand = parent / "lineage.json"
        if cand.exists():
            try:
                info["train_lineage"] = json.loads(cand.read_text())
            except Exception:
                pass
            break
    out = Path(output_dir) / "export_source.json"
    out.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"[convert] export_source.json -> {out}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--origin-hf-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("bridge", "direct"),
        default="bridge",
        help=(
            "bridge is the eval-safe path. direct is a debug-only tensor dump "
            "and should not be used for model quality evaluation."
        ),
    )
    parser.add_argument(
        "--fake-distributed",
        action="store_true",
        help="Install the fake one-rank distributed facade for rescue shells without usable sockets.",
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory exists: {args.output_dir}")

    if args.fake_distributed or os.environ.get("RELAX_EVAL_CP2_FAKE_DIST") == "1":
        _install_fake_distributed_context()

    if args.mode == "direct":
        _direct_export_dcp_to_hf(args.input_dir, args.origin_hf_dir, args.output_dir, args.force)
        _stamp_export_source(args.input_dir, args.origin_hf_dir, args.output_dir, "direct")
        print("Done!", flush=True)
        return 0

    print(f"Loading config from {args.origin_hf_dir}", flush=True)
    bridge = AutoBridge.from_hf_pretrained(args.origin_hf_dir, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    _provider_override["provider"] = provider
    print(f"[convert] Using Bridge provider: {type(provider).__name__}", flush=True)
    print(f"[convert] provider layers={provider.num_layers} hidden={provider.hidden_size}", flush=True)
    print(f"Exporting checkpoint from {args.input_dir} to {args.output_dir}", flush=True)
    bridge.export_ckpt(args.input_dir, args.output_dir)
    _normalize_qwen35_text_config(Path(args.origin_hf_dir), Path(args.output_dir))
    _stamp_export_source(args.input_dir, args.origin_hf_dir, args.output_dir, "bridge")
    print("Done!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
