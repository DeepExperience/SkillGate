# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import hashlib
import json
import sys
import types
from argparse import Namespace

import pytest


if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})
    torch_stub.Size = tuple
    torch_stub.dtype = type("dtype", (), {})
    sys.modules["torch"] = torch_stub

from relax.utils.training import train_dump_utils
from relax.utils.training.train_dump_utils import mark_eval_pending, save_eval_summary_jsonl


def _sample():
    return Namespace(
        metadata={},
        prompt=[],
        response="done",
        reward={"score": 1.0},
        response_length=1,
        tokens=[1],
        status="completed",
        group_index=0,
        label=None,
        multimodal_inputs=None,
        multimodal_train_inputs=None,
    )


def _args(tmp_path, *, rows: int = 1, n_samples: int = 1):
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text("".join(json.dumps({"prompt": i}) + "\n" for i in range(rows)), encoding="utf-8")
    return Namespace(
        rollout_result_dir=str(tmp_path),
        keep_best_actor_ckpt=True,
        best_actor_ckpt_eval_fingerprint="mixed-eval-v1",
        eval_datasets=[
            Namespace(
                name="agent_eval",
                path=str(dataset),
                n_samples_per_eval_prompt=n_samples,
            )
        ],
    )


def test_eval_pending_and_completion_markers_share_contract(tmp_path):
    args = _args(tmp_path)
    sample = _sample()

    mark_eval_pending(args, rollout_id=7)
    pending = tmp_path / "eval" / "7.pending.json"
    assert json.loads(pending.read_text())["eval_fingerprint"] == "mixed-eval-v1"

    save_eval_summary_jsonl(args, rollout_id=7, data={"agent_eval": {"samples": [sample]}})

    summary = tmp_path / "eval" / "7.jsonl"
    complete = json.loads((tmp_path / "eval" / "7.complete.json").read_text())
    assert not pending.exists()
    assert complete["records"] == 1
    assert complete["datasets"] == {"agent_eval": 1}
    assert complete["expected_datasets"] == {"agent_eval": 1}
    assert complete["eval_fingerprint"] == "mixed-eval-v1"
    assert complete["sha256"] == hashlib.sha256(summary.read_bytes()).hexdigest()


def test_keep_best_eval_expected_count_uses_slice_and_n_samples(tmp_path):
    args = _args(tmp_path, rows=3, n_samples=2)
    args.eval_datasets[0].path += "@[0:2]"
    samples = [_sample() for _ in range(4)]

    save_eval_summary_jsonl(args, rollout_id=8, data={"agent_eval": {"samples": samples}})

    complete = json.loads((tmp_path / "eval" / "8.complete.json").read_text())
    assert complete["datasets"] == {"agent_eval": 4}
    assert complete["expected_datasets"] == {"agent_eval": 4}


@pytest.mark.parametrize("samples", [[], [_sample()]])
def test_keep_best_eval_rejects_empty_or_partial_results(tmp_path, samples):
    args = _args(tmp_path, rows=2)
    mark_eval_pending(args, rollout_id=9)

    with pytest.raises(RuntimeError, match="no summary records|sample counts"):
        save_eval_summary_jsonl(args, rollout_id=9, data={"agent_eval": {"samples": samples}})

    assert (tmp_path / "eval" / "9.pending.json").is_file()
    assert not (tmp_path / "eval" / "9.complete.json").exists()


@pytest.mark.parametrize("writer", ["_write_summary_jsonl", "_write_json_atomic"])
def test_keep_best_eval_propagates_summary_write_failures(tmp_path, monkeypatch, writer):
    args = _args(tmp_path)
    mark_eval_pending(args, rollout_id=10)

    def fail_write(path, payload):
        raise OSError("injected eval summary write failure")

    monkeypatch.setattr(train_dump_utils, writer, fail_write)
    with pytest.raises(RuntimeError, match="injected eval summary write failure"):
        save_eval_summary_jsonl(args, rollout_id=10, data={"agent_eval": {"samples": [_sample()]}})

    assert (tmp_path / "eval" / "10.pending.json").is_file()
    assert not (tmp_path / "eval" / "10.complete.json").exists()
