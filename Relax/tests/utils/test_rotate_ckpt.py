# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import hashlib
import json
import os
import shutil
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from relax.utils.rotate_ckpt import (
    _hardlink_checkpoint,
    import_best_eval_checkpoint,
    release_checkpoint_save_reservation,
    reserve_checkpoint_save,
    rotate_ckpt,
    write_checkpoint_complete_marker,
)


def _checkpoint(root: Path, step: int) -> Path:
    (root / "transformer_config.pkl").write_text("test transformer config\n", encoding="utf-8")
    rollout_root = root / "rollout"
    rollout_root.mkdir(parents=True, exist_ok=True)
    (rollout_root / f"global_dataset_state_dict_{step}.pt").write_text(
        f"dataset state {step}\n",
        encoding="utf-8",
    )
    path = root / f"iter_{step:07d}"
    path.mkdir(parents=True)
    for name in (".metadata", "metadata.json", "modelopt_run_config.yaml", "common.pt", "__0_0.distcp"):
        (path / name).write_text(f"{step}:{name}\n", encoding="utf-8")
    (path / ".metadata").write_text("expected=__0_0.distcp\n", encoding="utf-8")
    write_checkpoint_complete_marker(root, step)
    return path


def _eval(
    root: Path,
    step: int,
    rewards: list[float | None],
    dataset: str = "agent_eval",
    fingerprint: str = "eval-v1",
) -> None:
    path = root / "rollout_result" / "eval" / f"{step}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for reward in rewards:
            payload = None if reward is None else {"score": reward}
            handle.write(json.dumps({"dataset": dataset, "reward": payload}) + "\n")
    path.with_suffix(".complete.json").write_text(
        json.dumps(
            {
                "rollout_id": step,
                "records": len(rewards),
                "datasets": {dataset: len(rewards)},
                "expected_datasets": {dataset: len(rewards)},
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "eval_fingerprint": fingerprint,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _config(root: Path) -> Namespace:
    eval_root = root / "rollout_result" / "eval"
    configured_rows = 1
    if eval_root.is_dir():
        configured_rows = max(
            (
                sum(1 for line in path.open(encoding="utf-8") if line.strip())
                for path in eval_root.glob("*.jsonl")
                if (root / f"iter_{int(path.stem):07d}").is_dir()
            ),
            default=1,
        )
    configured_eval = root / "configured_eval.jsonl"
    configured_eval.write_text("{}\n" * configured_rows, encoding="utf-8")
    return Namespace(
        save=str(root),
        rollout_result_dir=None,
        rotate_ckpt=False,
        max_actor_ckpt_to_keep=1,
        keep_best_actor_ckpt=True,
        best_actor_ckpt_eval_dataset="agent_eval",
        best_actor_ckpt_eval_fingerprint="eval-v1",
        eval_interval=2,
        eval_datasets=[
            Namespace(name="agent_eval", path=str(configured_eval), n_samples_per_eval_prompt=1)
        ],
        load=None,
    )


def test_keep_latest_and_later_best_on_tie(tmp_path: Path):
    first = _checkpoint(tmp_path, 1)
    latest = _checkpoint(tmp_path, 2)
    _eval(tmp_path, 1, [1.0, None])  # Missing reward stays in denominator: 0.5.
    _eval(tmp_path, 2, [0.5, 0.5])
    _eval(tmp_path, 60, [1.0])  # No matching checkpoint: ignore epoch-only eval.
    source_inode = (latest / "__0_0.distcp").stat().st_ino

    rotate_ckpt(_config(tmp_path), global_step=3)

    marker = json.loads((tmp_path / "best_eval" / "BEST_EVAL.json").read_text())
    preserved = Path(marker["preserved_checkpoint"])
    assert marker["iteration"] == 2
    assert marker["score"] == 0.5
    assert (tmp_path / "best_eval" / "latest_checkpointed_iteration.txt").read_text().strip() == "2"
    assert (preserved / "__0_0.distcp").stat().st_ino == source_inode
    assert (tmp_path / "best_eval" / "transformer_config.pkl").is_file()
    assert (tmp_path / "best_eval" / "rollout" / "global_dataset_state_dict_2.pt").is_file()
    assert not first.exists()
    assert latest.exists()


def test_new_higher_eval_replaces_preserved_best(tmp_path: Path):
    _checkpoint(tmp_path, 1)
    _checkpoint(tmp_path, 2)
    _eval(tmp_path, 1, [0.6])
    _eval(tmp_path, 2, [0.4])
    config = _config(tmp_path)
    rotate_ckpt(config, global_step=3)
    assert (tmp_path / "best_eval" / "iter_0000001").is_dir()

    _checkpoint(tmp_path, 3)
    _eval(tmp_path, 3, [0.8])
    rotate_ckpt(config, global_step=4)

    marker = json.loads((tmp_path / "best_eval" / "BEST_EVAL.json").read_text())
    assert marker["iteration"] == 3
    assert (tmp_path / "best_eval" / "iter_0000003").is_dir()
    assert not (tmp_path / "best_eval" / "iter_0000001").exists()


def test_preserve_failure_skips_destructive_rotation(tmp_path: Path, monkeypatch):
    first = _checkpoint(tmp_path, 1)
    latest = _checkpoint(tmp_path, 2)
    _eval(tmp_path, 1, [1.0])

    def fail_copytree(*_args, **_kwargs):
        raise OSError("injected hardlink failure")

    monkeypatch.setattr("relax.utils.rotate_ckpt.shutil.copytree", fail_copytree)
    rotate_ckpt(_config(tmp_path), global_step=3)

    assert first.exists()
    assert latest.exists()
    assert not (tmp_path / "best_eval" / "BEST_EVAL.json").exists()


def test_eval_without_completion_marker_is_not_selected(tmp_path: Path):
    pending = _checkpoint(tmp_path, 1)
    latest = _checkpoint(tmp_path, 2)
    path = tmp_path / "rollout_result" / "eval" / "1.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"dataset": "agent_eval", "reward": {"score": 1.0}}) + "\n")

    rotate_ckpt(_config(tmp_path), global_step=3)

    assert pending.exists()
    assert latest.exists()
    assert not (tmp_path / "best_eval" / "BEST_EVAL.json").exists()


def test_eval_with_stale_completion_digest_is_not_selected(tmp_path: Path):
    pending = _checkpoint(tmp_path, 1)
    latest = _checkpoint(tmp_path, 2)
    _eval(tmp_path, 1, [1.0])
    path = tmp_path / "rollout_result" / "eval" / "1.jsonl"
    path.write_text(json.dumps({"dataset": "agent_eval", "reward": {"score": 0.0}}) + "\n")

    rotate_ckpt(_config(tmp_path), global_step=3)

    assert pending.exists()
    assert latest.exists()
    assert not (tmp_path / "best_eval" / "BEST_EVAL.json").exists()


def test_eval_count_smaller_than_configured_dataset_is_not_selected(tmp_path: Path):
    pending = _checkpoint(tmp_path, 1)
    latest = _checkpoint(tmp_path, 2)
    _eval(tmp_path, 1, [1.0])
    config = _config(tmp_path)
    configured_eval = Path(config.eval_datasets[0].path)
    configured_eval.write_text("{}\n{}\n", encoding="utf-8")

    assert not rotate_ckpt(config, global_step=3)

    assert pending.exists()
    assert latest.exists()
    assert not (tmp_path / "best_eval" / "BEST_EVAL.json").exists()


def test_same_step_rewrite_replaces_stale_preserved_checkpoint(tmp_path: Path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = _checkpoint(source_root, 1)
    preserved_root = tmp_path / "preserved"
    _hardlink_checkpoint(source, preserved_root)
    preserved = preserved_root / "__0_0.distcp"
    old_inode = preserved.stat().st_ino

    shutil.rmtree(source)
    rewritten = _checkpoint(source_root, 1)
    (rewritten / "__0_0.distcp").write_text("rewritten checkpoint\n", encoding="utf-8")
    rewritten_inode = (rewritten / "__0_0.distcp").stat().st_ino
    _hardlink_checkpoint(rewritten, preserved_root)

    assert preserved.stat().st_ino == rewritten_inode
    assert preserved.stat().st_ino != old_inode
    assert preserved.read_text(encoding="utf-8") == "rewritten checkpoint\n"


def test_zero_keep_cap_does_not_delete_pending_eval_checkpoint(tmp_path: Path):
    pending = _checkpoint(tmp_path, 1)
    config = _config(tmp_path)
    config.max_actor_ckpt_to_keep = 0

    rotate_ckpt(config, global_step=2)

    assert pending.exists()


def test_truncated_shard_is_not_a_complete_checkpoint(tmp_path: Path):
    incomplete = _checkpoint(tmp_path, 1)
    (incomplete / "__0_0.distcp").write_text("x", encoding="utf-8")
    latest = _checkpoint(tmp_path, 2)
    _eval(tmp_path, 1, [1.0])

    rotate_ckpt(_config(tmp_path), global_step=3)

    assert latest.exists()
    assert not (tmp_path / "best_eval" / "BEST_EVAL.json").exists()


def test_completion_marker_rejects_metadata_missing_shard(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path, 1)
    (checkpoint / ".relax_complete.json").unlink()
    (checkpoint / ".metadata").write_text("expected=__0_0.distcp,__1_0.distcp\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="shard set"):
        write_checkpoint_complete_marker(tmp_path, 1)


def test_epoch_pending_marker_defers_rotation(tmp_path: Path):
    pending = _checkpoint(tmp_path, 2)  # Not periodic for eval_interval=2.
    latest = _checkpoint(tmp_path, 3)
    eval_dir = tmp_path / "rollout_result" / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "2.pending.json").write_text("{}\n", encoding="utf-8")

    rotate_ckpt(_config(tmp_path), global_step=4)

    assert pending.exists()
    assert latest.exists()


def test_corrupt_existing_best_marker_skips_destructive_rotation(tmp_path: Path):
    _checkpoint(tmp_path, 1)
    latest = _checkpoint(tmp_path, 2)
    _eval(tmp_path, 1, [1.0])
    config = _config(tmp_path)
    rotate_ckpt(config, global_step=3)
    _checkpoint(tmp_path, 3)
    marker_path = tmp_path / "best_eval" / "BEST_EVAL.json"
    marker = json.loads(marker_path.read_text())
    marker["score"] = "not-a-number"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    rotate_ckpt(config, global_step=4)

    assert latest.exists()
    assert (tmp_path / "iter_0000003").exists()


def test_changed_eval_fingerprint_does_not_compare_old_best(tmp_path: Path):
    _checkpoint(tmp_path, 1)
    latest = _checkpoint(tmp_path, 2)
    _eval(tmp_path, 1, [0.9])
    config = _config(tmp_path)
    rotate_ckpt(config, global_step=3)
    new_checkpoint = _checkpoint(tmp_path, 3)
    _eval(tmp_path, 3, [0.5], fingerprint="eval-v2")
    config.best_actor_ckpt_eval_fingerprint = "eval-v2"

    rotate_ckpt(config, global_step=4)

    assert latest.exists()
    assert new_checkpoint.exists()
    marker = json.loads((tmp_path / "best_eval" / "BEST_EVAL.json").read_text())
    assert marker["eval_fingerprint"] == "eval-v1"


def test_checkpoint_save_reservation_is_cross_driver_exclusive(tmp_path: Path):
    reservation = reserve_checkpoint_save(tmp_path, 7, "run-a", "owner-a")
    second_step = reserve_checkpoint_save(tmp_path, 8, "run-a", "owner-a")

    with pytest.raises(FileExistsError):
        reserve_checkpoint_save(tmp_path, 9, "run-b", "owner-b")

    assert json.loads(reservation.read_text())["run_id"] == "run-a"
    assert second_step.is_file()


def test_stale_same_run_owner_requires_explicit_audited_reclaim(tmp_path: Path):
    owner_path = tmp_path / ".relax_save_owner.json"
    owner_path.write_text(
        json.dumps(
            {
                "run_id": "run-a",
                "owner_token": "old-owner",
                "pid": 999_999_999,
                "host": os.uname().nodename,
            }
        )
        + "\n"
    )

    with pytest.raises(FileExistsError):
        reserve_checkpoint_save(tmp_path, 7, "run-a", "new-owner")
    reservation = reserve_checkpoint_save(tmp_path, 7, "run-a", "new-owner", allow_reclaim=True)

    assert reservation.is_file()
    assert list(tmp_path.glob(".relax_save_owner.json.reclaimed.*"))


def test_stale_owner_has_only_one_concurrent_reclaim_winner(tmp_path: Path):
    owner_path = tmp_path / ".relax_save_owner.json"
    owner_path.write_text(
        json.dumps(
            {
                "run_id": "run-a",
                "owner_token": "old-owner",
                "pid": 999_999_999,
                "host": os.uname().nodename,
            }
        )
        + "\n"
    )

    def reclaim(step: int, owner: str) -> str:
        try:
            reserve_checkpoint_save(tmp_path, step, "run-a", owner, allow_reclaim=True)
            return "won"
        except FileExistsError:
            return "lost"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda item: reclaim(*item), [(7, "owner-a"), (8, "owner-b")]))

    assert sorted(results) == ["lost", "won"]
    current_owner = json.loads(owner_path.read_text())["owner_token"]
    reservations = list(tmp_path.glob(".iter_*.save_reservation.json"))
    assert len(reservations) == 1
    assert json.loads(reservations[0].read_text())["owner_token"] == current_owner


def test_preserved_best_uses_immutable_eval_snapshot(tmp_path: Path):
    _checkpoint(tmp_path, 1)
    _eval(tmp_path, 1, [1.0])
    config = _config(tmp_path)

    assert rotate_ckpt(config, global_step=2)
    marker = json.loads((tmp_path / "best_eval" / "BEST_EVAL.json").read_text())
    snapshot = Path(marker["eval_file"])
    assert snapshot.parent.parent == tmp_path / "best_eval" / "eval"
    assert snapshot.parent.name.startswith("iter_0000001_")
    assert snapshot.read_bytes() == (tmp_path / "rollout_result" / "eval" / "1.jsonl").read_bytes()

    # A same-step eval rerun replaces the mutable source, but must not invalidate
    # the evidence that selected the already preserved checkpoint.
    _eval(tmp_path, 1, [0.0])
    assert rotate_ckpt(config, global_step=2)
    assert json.loads((tmp_path / "best_eval" / "BEST_EVAL.json").read_text())["score"] == 1.0
    assert json.loads(snapshot.read_text())["reward"]["score"] == 1.0


def test_cross_root_resume_imports_better_existing_best(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    source_checkpoint = _checkpoint(source, 1)
    _eval(source, 1, [1.0])
    source_config = _config(source)
    assert rotate_ckpt(source_config, global_step=2)
    source_best_shard = source / "best_eval" / "iter_0000001" / "__0_0.distcp"
    source_best_inode = source_best_shard.stat().st_ino

    destination = tmp_path / "destination"
    destination.mkdir()
    latest = _checkpoint(destination, 2)
    _eval(destination, 2, [0.5])
    destination_config = _config(destination)
    destination_config.load = str(source)

    import_best_eval_checkpoint(destination_config)
    shutil.rmtree(source / "best_eval")
    assert rotate_ckpt(destination_config, global_step=3)

    marker = json.loads((destination / "best_eval" / "BEST_EVAL.json").read_text())
    imported = Path(marker["preserved_checkpoint"])
    assert marker["iteration"] == 1
    assert marker["score"] == 1.0
    assert marker["imported_from"] == str((source / "best_eval").resolve())
    assert imported.parent == destination / "best_eval"
    assert (imported / "__0_0.distcp").stat().st_ino == source_best_inode
    assert (destination / "best_eval" / "latest_checkpointed_iteration.txt").read_text().strip() == "1"
    assert (destination / "best_eval" / "transformer_config.pkl").is_file()
    assert (destination / "best_eval" / "rollout" / "global_dataset_state_dict_1.pt").is_file()
    assert source_checkpoint.exists()
    assert latest.exists()


def test_rotation_failure_can_release_unstarted_step_reservation(tmp_path: Path):
    reservation = reserve_checkpoint_save(tmp_path, 7, "run-a", "owner-a")

    release_checkpoint_save_reservation(tmp_path, 7, "owner-a")

    assert not reservation.exists()
    assert list(tmp_path.glob(".iter_0000007.save_reservation.json.released.*"))


def test_valid_complete_clears_stale_pending_marker(tmp_path: Path):
    _checkpoint(tmp_path, 1)
    latest = _checkpoint(tmp_path, 2)
    _eval(tmp_path, 1, [1.0])
    pending = tmp_path / "rollout_result" / "eval" / "1.pending.json"
    pending.write_text(json.dumps({"rollout_id": 1, "eval_fingerprint": "eval-v1"}) + "\n")

    assert rotate_ckpt(_config(tmp_path), global_step=3)

    assert not pending.exists()
    assert latest.exists()
    assert (tmp_path / "best_eval" / "BEST_EVAL.json").exists()


def test_orphaned_hardlink_is_recovered_before_rotation(tmp_path: Path):
    source = _checkpoint(tmp_path, 1)
    latest = _checkpoint(tmp_path, 2)
    _eval(tmp_path, 1, [1.0])
    orphan = tmp_path / "best_eval" / "iter_0000001"
    _hardlink_checkpoint(source, orphan)

    assert rotate_ckpt(_config(tmp_path), global_step=3)

    marker = json.loads((tmp_path / "best_eval" / "BEST_EVAL.json").read_text())
    assert marker["iteration"] == 1
    assert latest.exists()


def test_interrupted_directory_swap_restores_backup(tmp_path: Path):
    _checkpoint(tmp_path, 1)
    latest = _checkpoint(tmp_path, 2)
    _eval(tmp_path, 1, [1.0])
    _eval(tmp_path, 2, [0.0])
    config = _config(tmp_path)
    assert rotate_ckpt(config, global_step=3)
    destination = tmp_path / "best_eval" / "iter_0000001"
    backup = tmp_path / "best_eval" / ".iter_0000001.old.999"
    destination.rename(backup)

    assert rotate_ckpt(config, global_step=4)

    assert destination.exists()
    assert not backup.exists()
    assert latest.exists()
