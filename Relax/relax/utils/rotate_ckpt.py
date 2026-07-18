# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from relax.utils.logging_utils import get_logger


_ITER_DIR_PATTERN = re.compile(r"^iter_\d{7}$")
_DISTCP_FILE_PATTERN = re.compile(r"^__[0-9]+_[0-9]+\.distcp$")
_CHECKPOINT_COMPLETE_MARKER = ".relax_complete.json"


logger = get_logger(__name__)


def rotate_ckpt(config: Namespace, global_step: int):
    keep_best = bool(getattr(config, "keep_best_actor_ckpt", False))
    if config.max_actor_ckpt_to_keep is None and not config.rotate_ckpt and not keep_best:
        return True

    ckpt_dirs = list(Path(config.save).glob("iter_*"))
    if not ckpt_dirs:
        return True

    ckpt_dirs = [
        (int(ckpt_dir.name.split("_")[-1]), ckpt_dir)
        for ckpt_dir in ckpt_dirs
        if ckpt_dir.name.split("_")[-1].isdigit()
    ]
    if not ckpt_dirs:
        return True

    ckpt_dirs.sort(key=lambda x: x[0], reverse=True)
    if keep_best:
        for step, path in ckpt_dirs:
            if not (path / _CHECKPOINT_COMPLETE_MARKER).exists():
                try:
                    write_checkpoint_complete_marker(config.save, step)
                    logger.warning("Recovered missing checkpoint completion marker for %s", path)
                except BaseException as exc:
                    logger.error("Could not recover checkpoint completion marker for %s: %s", path, exc)
        incomplete = [str(path) for _step, path in ckpt_dirs if not _checkpoint_complete(path)]
        if incomplete:
            logger.error("Refusing best-checkpoint rotation with incomplete or legacy checkpoints: %s", incomplete)
            return False

    # Eval can finish after the checkpoint was written. Preserve a hard-linked
    # best copy before normal rotation can remove its top-level iter directory.
    # If preservation fails, skip cleanup entirely: extra checkpoints are safer
    # than silently deleting the best candidate.
    if keep_best:
        try:
            if not _preserve_best_eval_checkpoint(config, ckpt_dirs):
                logger.error("Best-eval checkpoint preservation failed; skipping checkpoint rotation")
                return False
            if _has_pending_eval_checkpoint(config, ckpt_dirs):
                logger.warning("An eval-aligned checkpoint is still awaiting a complete eval; skipping checkpoint rotation")
                return True
        except BaseException as exc:
            # Rotation runs on rank zero. Never let marker corruption strand
            # the other ranks at the following distributed barrier.
            logger.exception(f"Best-eval checkpoint validation failed; skipping checkpoint rotation: {exc}")
            return False

    if config.rotate_ckpt:
        _rotate_ckpt_cleanup(config, global_step, ckpt_dirs)
    elif config.max_actor_ckpt_to_keep is not None:
        _max_keep_cleanup(config, ckpt_dirs)
    return True


def _rotate_ckpt_cleanup(config: Namespace, global_step: int, ckpt_dirs: list):
    """Cleanup for rotate_ckpt mode: keep latest + up to max_ckpt save_interval
    checkpoints, delete intermediates."""
    # +1 是因为要额外保存 latest，如果 latest 同时满足 save_interval ，下面会减掉
    max_ckpt = global_step // config.save_interval + 1
    if config.max_actor_ckpt_to_keep is not None:
        max_ckpt = min(max_ckpt, config.max_actor_ckpt_to_keep + 1)

    logger.info(f"max checkpoint to keep: {max_ckpt}")

    latest_ckpt = ckpt_dirs.pop(0)
    if latest_ckpt[0] % config.save_interval == 0:
        max_ckpt -= 1

    logger.info(f"latest checkpoint: {latest_ckpt}")
    ckpt_num = 1
    for step, ckpt_dir in ckpt_dirs:
        if step % config.save_interval != 0 or ckpt_num >= max_ckpt:
            _remove_ckpt(ckpt_dir)
        else:
            ckpt_num += 1
            logger.info(f"keep checkpoint dir {ckpt_dir}, current ckpt num: {ckpt_num}")


def _max_keep_cleanup(config: Namespace, ckpt_dirs: list):
    """Cleanup for non-rotate mode: simply keep the latest
    max_actor_ckpt_to_keep checkpoints."""
    max_keep = config.max_actor_ckpt_to_keep
    logger.info(f"max checkpoint to keep: {max_keep}")

    # ckpt_dirs is sorted descending by step; keep the first max_keep, remove the rest
    for i, (step, ckpt_dir) in enumerate(ckpt_dirs):
        if i < max_keep:
            logger.info(f"keep checkpoint dir {ckpt_dir}")
        else:
            _remove_ckpt(ckpt_dir)


def _remove_ckpt(ckpt_dir: Path):
    if not _ITER_DIR_PATTERN.match(ckpt_dir.name):
        logger.error(f"Refusing to remove {ckpt_dir}: directory name does not match iter_NNNNNNN pattern")
        return
    shutil.rmtree(ckpt_dir)
    if ckpt_dir.exists():
        raise RuntimeError(f"checkpoint directory still exists after removal: {ckpt_dir}")
    logger.warning(f"remove checkpoint dir {ckpt_dir}")


def _metadata_shard_names(checkpoint: Path) -> set[str]:
    try:
        return {
            name.decode("ascii")
            for name in re.findall(rb"__[0-9]+_[0-9]+\.distcp", (checkpoint / ".metadata").read_bytes())
        }
    except OSError as exc:
        raise RuntimeError(f"cannot read checkpoint metadata: {checkpoint / '.metadata'}") from exc


def _checkpoint_complete(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        marker = json.loads((path / _CHECKPOINT_COMPLETE_MARKER).read_text(encoding="utf-8"))
        expected_files = marker["files"]
        if int(marker["iteration"]) != int(path.name.rsplit("_", 1)[-1]):
            return False
        if not isinstance(expected_files, dict) or not expected_files:
            return False
        expected_shards = {name for name in expected_files if _DISTCP_FILE_PATTERN.fullmatch(name)}
        if not expected_shards:
            return False
        for name, expected_size in expected_files.items():
            file_path = path / name
            if not file_path.is_file() or file_path.stat().st_size != int(expected_size) or int(expected_size) <= 0:
                return False
        if marker.get("metadata_sha256") != _file_sha256(path / ".metadata"):
            return False
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def write_checkpoint_complete_marker(save_root: str | Path, iteration: int) -> Path:
    """Record exact finalized checkpoint file sizes for later safe rotation."""
    checkpoint = Path(save_root) / f"iter_{iteration:07d}"
    required = (".metadata", "metadata.json", "modelopt_run_config.yaml", "common.pt")
    actual_shards = {
        path.name for path in checkpoint.iterdir() if path.is_file() and _DISTCP_FILE_PATTERN.fullmatch(path.name)
    }
    expected_shards = _metadata_shard_names(checkpoint)
    if not expected_shards or actual_shards != expected_shards:
        raise RuntimeError(
            f"checkpoint shard set does not match .metadata: expected={sorted(expected_shards)} "
            f"actual={sorted(actual_shards)}"
        )
    files = list(required) + sorted(actual_shards)
    if not files or any(not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size <= 0 for name in files):
        raise RuntimeError(f"checkpoint is not finalized: {checkpoint}")
    payload = {
        "format_version": 1,
        "iteration": iteration,
        "files": {name: (checkpoint / name).stat().st_size for name in files},
        "metadata_sha256": _file_sha256(checkpoint / ".metadata"),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    marker = checkpoint / _CHECKPOINT_COMPLETE_MARKER
    temp = checkpoint / f".{_CHECKPOINT_COMPLETE_MARKER}.tmp.{os.getpid()}"
    try:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, marker)
    finally:
        temp.unlink(missing_ok=True)
    return marker


def _owner_process_is_alive(record: dict) -> bool:
    if record.get("host") != os.uname().nodename:
        return True
    try:
        os.kill(int(record["pid"]), 0)
        return True
    except ProcessLookupError:
        return False
    except (OSError, TypeError, ValueError):
        return True


def _write_exclusive_record(path: Path, payload: dict) -> None:
    # Publish only a fully written inode. Creating the final path before
    # serializing could leave a truncated, permanently blocking owner record
    # if the process died between open() and close().
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.claim.", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _acquire_owner_record(path: Path, payload: dict, allow_reclaim: bool, same_owner_ok: bool) -> None:
    try:
        _write_exclusive_record(path, payload)
        return
    except FileExistsError:
        pass
    existing = _read_owner_record(path)
    if existing.get("owner_token") == payload["owner_token"]:
        if same_owner_ok:
            return
        raise FileExistsError(f"checkpoint iteration is already reserved by this driver: {existing}")
    if not allow_reclaim:
        raise FileExistsError(f"checkpoint save root is owned by another driver: {existing}")
    reclaim_lock = path.with_name(f".{path.name}.reclaim_lock")
    try:
        os.mkdir(reclaim_lock)
    except FileExistsError as exc:
        raise FileExistsError(f"another driver is already reclaiming {path}") from exc
    try:
        # Re-read under the reclaim mutex. Another contender may have replaced
        # the stale owner after our initial read.
        existing = _read_owner_record(path)
        if existing.get("owner_token") == payload["owner_token"]:
            if same_owner_ok:
                return
            raise FileExistsError(f"checkpoint iteration is already reserved by this driver: {existing}")
        reclaimable = (
            existing.get("run_id") == payload["run_id"]
            and existing.get("host") == payload["host"]
            and not _owner_process_is_alive(existing)
        )
        if not reclaimable:
            raise FileExistsError(f"checkpoint save root is owned by another driver: {existing}")
        audit = path.with_name(
            f"{path.name}.reclaimed.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.{os.getpid()}"
        )
        os.replace(path, audit)
        _write_exclusive_record(path, payload)
    finally:
        os.rmdir(reclaim_lock)


def _read_owner_record(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid checkpoint owner record {path}: {exc}") from exc


def reserve_checkpoint_save(
    save_root: str | Path,
    iteration: int,
    run_id: str,
    owner_token: str,
    allow_reclaim: bool = False,
) -> Path:
    """Atomically own a save root and reserve one iteration across drivers."""
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration": iteration,
        "run_id": run_id,
        "owner_token": owner_token,
        "pid": os.getpid(),
        "host": os.uname().nodename,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _acquire_owner_record(save_root / ".relax_save_owner.json", payload, allow_reclaim, same_owner_ok=True)
    checkpoint = save_root / f"iter_{iteration:07d}"
    reservation = save_root / f".iter_{iteration:07d}.save_reservation.json"
    if checkpoint.exists():
        raise FileExistsError(f"checkpoint target already exists: {checkpoint}")
    _acquire_owner_record(reservation, payload, allow_reclaim, same_owner_ok=False)
    if checkpoint.exists():
        raise FileExistsError(f"checkpoint target appeared while reserving: {checkpoint}")
    return reservation


def release_checkpoint_save_reservation(save_root: str | Path, iteration: int, owner_token: str) -> None:
    """Release a step reservation only when no checkpoint write has started."""
    save_root = Path(save_root)
    checkpoint = save_root / f"iter_{iteration:07d}"
    reservation = save_root / f".iter_{iteration:07d}.save_reservation.json"
    if checkpoint.exists() or not reservation.is_file():
        return
    record = json.loads(reservation.read_text(encoding="utf-8"))
    if record.get("owner_token") != owner_token:
        raise RuntimeError(f"refusing to release another driver's reservation: {reservation}")
    audit = reservation.with_name(
        f"{reservation.name}.released.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.{os.getpid()}"
    )
    os.replace(reservation, audit)


def _record_reward(record: dict) -> float:
    reward = record.get("reward")
    value = None
    if isinstance(reward, dict):
        value = reward.get("score", reward.get("raw_score"))
    elif isinstance(reward, (int, float)):
        value = reward
    if value is None:
        value = record.get("score", record.get("raw_score"))
    try:
        result = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        result = 0.0
    return result if math.isfinite(result) else 0.0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _write_best_tracker(best_root: Path, iteration: int) -> None:
    _write_text_atomic(best_root / "latest_checkpointed_iteration.txt", f"{iteration}\n")


def _link_file_atomic(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size <= 0:
        raise RuntimeError(f"missing resume file for best checkpoint: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temp.unlink(missing_ok=True)
    try:
        os.link(source, temp)
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def _prepare_best_resume_files(save_root: Path, best_root: Path, iteration: int) -> None:
    _link_file_atomic(save_root / "transformer_config.pkl", best_root / "transformer_config.pkl")
    state_name = f"global_dataset_state_dict_{iteration}.pt"
    _link_file_atomic(save_root / "rollout" / state_name, best_root / "rollout" / state_name)


def _cleanup_old_best_resume_files(best_root: Path, iteration: int) -> None:
    current = f"global_dataset_state_dict_{iteration}.pt"
    rollout_root = best_root / "rollout"
    if not rollout_root.is_dir():
        return
    for path in rollout_root.glob("global_dataset_state_dict_*.pt"):
        if path.name != current:
            path.unlink()


def _score_eval_file(
    path: Path,
    dataset_name: str,
    eval_fingerprint: str,
    expected_samples: int,
    expected_iteration: int,
) -> tuple[float, int, str] | None:
    rewards = []
    total_records = 0
    try:
        complete = json.loads(path.with_suffix(".complete.json").read_text(encoding="utf-8"))
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                total_records += 1
                if record.get("dataset") != dataset_name:
                    continue
                # Match Relax's eval aggregation: missing/non-numeric rewards
                # remain in the fixed denominator and count as zero.
                rewards.append(_record_reward(record))
        expected_dataset = int((complete.get("datasets") or {}).get(dataset_name, 0))
        marker_expected_dataset = int((complete.get("expected_datasets") or {}).get(dataset_name, -1))
        expected_sha256 = str(complete.get("sha256") or "")
        marker_fingerprint = str(complete.get("eval_fingerprint") or "")
        actual_sha256 = _file_sha256(path)
        if (
            int(complete.get("rollout_id", -1)) != expected_iteration
            or int(complete.get("records", -1)) != total_records
            or expected_dataset != len(rewards)
            or marker_expected_dataset != expected_samples
            or len(rewards) != expected_samples
            or expected_sha256 != actual_sha256
            or marker_fingerprint != eval_fingerprint
        ):
            logger.warning(
                "Ignoring incomplete or incompatible eval summary %s: marker records=%s/%s "
                "dataset=%s/%s configured=%s/%s sha256_match=%s fingerprint_match=%s",
                path,
                complete.get("records"),
                total_records,
                expected_dataset,
                len(rewards),
                marker_expected_dataset,
                expected_samples,
                expected_sha256 == actual_sha256,
                marker_fingerprint == eval_fingerprint,
            )
            return None
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(f"Ignoring eval summary without a valid completion marker {path}: {exc}")
        return None
    if not rewards:
        return None
    return sum(rewards) / len(rewards), len(rewards), actual_sha256


def _load_preserved_best(
    best_root: Path,
    dataset_name: str,
    eval_fingerprint: str,
    expected_samples: int,
) -> dict | None:
    marker = best_root / "BEST_EVAL.json"
    if not marker.is_file():
        return None
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
        score = float(record["score"])
        iteration = int(record["iteration"])
        num_samples = int(record["num_samples"])
        preserved = Path(record["preserved_checkpoint"]).resolve()
        eval_path = Path(record["eval_file"]).resolve()
        if not math.isfinite(score) or iteration < 0 or num_samples <= 0:
            raise ValueError("invalid best-eval numeric fields")
        if record.get("dataset") != dataset_name or record.get("eval_fingerprint") != eval_fingerprint:
            raise ValueError("best-eval marker belongs to a different eval contract")
        if best_root.resolve() not in preserved.parents or not _checkpoint_complete(preserved):
            raise ValueError("preserved best checkpoint is missing or incomplete")
        if best_root.resolve() not in eval_path.parents:
            raise ValueError("best-eval evidence is not preserved under the best root")
        if not (best_root / "transformer_config.pkl").is_file():
            raise ValueError("preserved best root is missing transformer_config.pkl")
        if not (best_root / "rollout" / f"global_dataset_state_dict_{iteration}.pt").is_file():
            raise ValueError("preserved best root is missing its rollout dataset state")
        scored = _score_eval_file(eval_path, dataset_name, eval_fingerprint, expected_samples, iteration)
        if scored is None:
            raise ValueError("best-eval source summary is no longer valid")
        current_score, current_samples, current_sha256 = scored
        if (
            not math.isclose(current_score, score, rel_tol=0.0, abs_tol=1e-12)
            or current_samples != num_samples
            or record.get("eval_sha256") != current_sha256
        ):
            raise ValueError("best-eval marker no longer matches its eval summary")
        record.update(score=score, iteration=iteration, num_samples=num_samples)
        return record
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid existing best-eval marker {marker}: {exc}") from exc


def _has_pending_eval_checkpoint(config: Namespace, ckpt_dirs: list[tuple[int, Path]]) -> bool:
    eval_interval = getattr(config, "eval_interval", None)
    if not eval_interval:
        return False
    dataset_name = str(getattr(config, "best_actor_ckpt_eval_dataset", "agent_eval"))
    eval_fingerprint = str(getattr(config, "best_actor_ckpt_eval_fingerprint", ""))
    expected_samples = _expected_eval_samples(config, dataset_name)
    save_root = Path(config.save).resolve()
    eval_root = Path(getattr(config, "rollout_result_dir", None) or save_root / "rollout_result") / "eval"
    # Be conservative and defer the entire rotation while any eval-aligned
    # checkpoint is pending. This also protects direct callers configured with
    # a zero keep cap; argument validation rejects that unsafe combination.
    for step, _checkpoint in ckpt_dirs:
        eval_path = eval_root / f"{step}.jsonl"
        pending_path = eval_root / f"{step}.pending.json"
        is_periodic_eval = (step + 1) % int(eval_interval) == 0
        if not pending_path.is_file() and not is_periodic_eval:
            continue
        scored = _score_eval_file(eval_path, dataset_name, eval_fingerprint, expected_samples, step)
        if pending_path.is_file():
            if scored is not None:
                # Completion is written atomically before pending is removed. A
                # crash in that narrow window leaves a harmless stale pending.
                pending_path.unlink(missing_ok=True)
                continue
            try:
                pending = json.loads(pending_path.read_text(encoding="utf-8"))
                if int(pending.get("rollout_id", -1)) != step or pending.get("eval_fingerprint") != eval_fingerprint:
                    raise ValueError("pending marker belongs to a different eval contract")
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"invalid eval pending marker {pending_path}: {exc}") from exc
            logger.warning("Checkpoint step %s has an active eval pending marker", step)
            return True
        if not is_periodic_eval:
            continue
        if scored is None:
            complete_path = eval_path.with_suffix(".complete.json")
            if eval_path.exists() or complete_path.exists():
                raise RuntimeError(
                    f"eval-aligned checkpoint step {step} has invalid completed evidence: "
                    f"summary={eval_path.exists()} completion={complete_path.exists()}"
                )
            logger.warning("Checkpoint step %s is awaiting a complete %s eval", step, dataset_name)
            return True
    return False


def _hardlink_checkpoint(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.parent / f".{destination.name}.tmp.{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        shutil.copytree(source, tmp, copy_function=os.link, symlinks=True)
        if destination.exists():
            if _checkpoint_hardlinks_match(source, destination):
                shutil.rmtree(tmp)
                return
            backup = destination.parent / f".{destination.name}.old.{os.getpid()}"
            shutil.rmtree(backup, ignore_errors=True)
            destination.rename(backup)
            try:
                tmp.rename(destination)
            except BaseException:
                backup.rename(destination)
                raise
            shutil.rmtree(backup)
        else:
            tmp.rename(destination)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _snapshot_eval_evidence(source: Path, destination: Path) -> Path:
    """Atomically preserve an eval JSONL and its completion marker."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.parent / f".{destination.name}.tmp.{os.getpid()}"
    backup = destination.parent / f".{destination.name}.old.{os.getpid()}"
    shutil.rmtree(temp, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    try:
        temp.mkdir()
        for source_file, name in (
            (source, "summary.jsonl"),
            (source.with_suffix(".complete.json"), "summary.complete.json"),
        ):
            if not source_file.is_file():
                raise FileNotFoundError(f"missing eval evidence file: {source_file}")
            shutil.copy2(source_file, temp / name)
        if destination.exists():
            destination.rename(backup)
            try:
                temp.rename(destination)
            except BaseException:
                backup.rename(destination)
                raise
            shutil.rmtree(backup)
        else:
            temp.rename(destination)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return destination / "summary.jsonl"


def _checkpoint_hardlinks_match(source: Path, destination: Path) -> bool:
    """Return true only when every preserved file is the source hardlink."""

    def entries(root: Path) -> dict[str, tuple]:
        result = {}
        for path in root.rglob("*"):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                result[relative] = ("symlink", os.readlink(path))
            elif path.is_file():
                stat = path.stat()
                result[relative] = ("file", stat.st_dev, stat.st_ino, stat.st_size)
        return result

    return _checkpoint_complete(destination) and entries(source) == entries(destination)


def _recover_best_root(best_root: Path) -> None:
    """Recover or remove uncommitted directory swaps before reading the marker."""
    if not best_root.is_dir():
        return
    for temp in best_root.glob(".iter_*.tmp.*"):
        if temp.is_dir():
            shutil.rmtree(temp)
    backups: dict[str, list[Path]] = {}
    for backup in best_root.glob(".iter_*.old.*"):
        match = re.match(r"^\.(iter_[0-9]{7})\.old\.", backup.name)
        if match and backup.is_dir():
            backups.setdefault(match.group(1), []).append(backup)
    for destination_name, candidates in backups.items():
        if len(candidates) != 1:
            raise RuntimeError(f"ambiguous best-checkpoint backups for {destination_name}: {candidates}")
        destination = best_root / destination_name
        backup = candidates[0]
        if destination.exists():
            shutil.rmtree(backup)
        else:
            backup.rename(destination)
    eval_root = best_root / "eval"
    if eval_root.is_dir():
        for temp in eval_root.glob(".iter_*.tmp.*"):
            if temp.is_dir():
                shutil.rmtree(temp)
        eval_backups: dict[str, list[Path]] = {}
        for backup in eval_root.glob(".iter_*.old.*"):
            match = re.match(r"^\.(iter_[^.]+)\.old\.", backup.name)
            if match and backup.is_dir():
                eval_backups.setdefault(match.group(1), []).append(backup)
        for destination_name, candidates in eval_backups.items():
            if len(candidates) != 1:
                raise RuntimeError(f"ambiguous eval-evidence backups for {destination_name}: {candidates}")
            destination = eval_root / destination_name
            backup = candidates[0]
            if destination.exists():
                shutil.rmtree(backup)
            else:
                backup.rename(destination)


def _expected_eval_samples(config: Namespace, dataset_name: str) -> int:
    from relax.utils.training.train_dump_utils import _expected_eval_dataset_counts

    counts = _expected_eval_dataset_counts(config)
    if dataset_name not in counts:
        raise RuntimeError(
            f"best eval dataset {dataset_name!r} is absent from configured eval counts: {sorted(counts)}"
        )
    return counts[dataset_name]


def _preserve_best_eval_checkpoint(config: Namespace, ckpt_dirs: list[tuple[int, Path]]) -> bool:
    save_root = Path(config.save).resolve()
    best_root = save_root / "best_eval"
    dataset_name = str(getattr(config, "best_actor_ckpt_eval_dataset", "agent_eval"))
    eval_fingerprint = str(getattr(config, "best_actor_ckpt_eval_fingerprint", ""))
    expected_samples = _expected_eval_samples(config, dataset_name)
    if not eval_fingerprint:
        raise RuntimeError("missing best_actor_ckpt_eval_fingerprint")
    _recover_best_root(best_root)
    existing = _load_preserved_best(best_root, dataset_name, eval_fingerprint, expected_samples)
    orphaned = list(best_root.glob("iter_*")) if existing is None else []
    existing_key = None
    if existing is not None:
        existing_key = (existing["score"], existing["iteration"])
        preserved = Path(existing["preserved_checkpoint"]).resolve()
        _write_best_tracker(best_root, existing["iteration"])
        for old_dir in best_root.glob("iter_*"):
            if old_dir.resolve() != preserved and old_dir.is_dir():
                shutil.rmtree(old_dir)

    complete_by_step = {step: path for step, path in ckpt_dirs if _checkpoint_complete(path)}
    candidate = None
    eval_root = Path(getattr(config, "rollout_result_dir", None) or save_root / "rollout_result") / "eval"
    eval_paths = sorted(eval_root.glob("*.jsonl")) if eval_root.is_dir() else []
    for eval_path in eval_paths:
        try:
            step = int(eval_path.stem)
        except ValueError:
            continue
        checkpoint = complete_by_step.get(step)
        if checkpoint is None:
            # Epoch-boundary evals can exist without a matching save step.
            continue
        scored = _score_eval_file(eval_path, dataset_name, eval_fingerprint, expected_samples, step)
        if scored is None:
            continue
        score, num_samples, eval_sha256 = scored
        item = (score, step, num_samples, eval_sha256, checkpoint, eval_path)
        if candidate is None or (score, step) > (candidate[0], candidate[1]):
            candidate = item

    if candidate is None:
        if orphaned:
            raise RuntimeError(f"cannot recover orphaned preserved checkpoints under {best_root}: {orphaned}")
        return True
    score, step, num_samples, eval_sha256, source, eval_path = candidate
    destination = best_root / f"iter_{step:07d}"
    if orphaned and {path.resolve() for path in orphaned} != {destination.resolve()}:
        raise RuntimeError(f"orphaned best checkpoint does not match recoverable candidate {destination}: {orphaned}")
    if existing_key is not None and (score, step) <= existing_key:
        return True

    try:
        _hardlink_checkpoint(source, destination)
        # Include the digest so a same-step eval rerun never mutates evidence
        # referenced by the already committed BEST_EVAL marker.
        eval_snapshot_dir = best_root / "eval" / f"iter_{step:07d}_{eval_sha256[:16]}"
        eval_snapshot = _snapshot_eval_evidence(eval_path, eval_snapshot_dir)
        snapshot_scored = _score_eval_file(
            eval_snapshot,
            dataset_name,
            eval_fingerprint,
            expected_samples,
            step,
        )
        if snapshot_scored != (score, num_samples, eval_sha256):
            raise RuntimeError("preserved eval evidence does not match the selected source summary")
        _prepare_best_resume_files(save_root, best_root, step)
        marker_record = {
            "iteration": step,
            "score": score,
            "num_samples": num_samples,
            "dataset": dataset_name,
            "eval_fingerprint": eval_fingerprint,
            "eval_file": str(eval_snapshot.resolve()),
            "source_eval_file": str(eval_path.resolve()),
            "eval_sha256": eval_sha256,
            "source_checkpoint": str(source.resolve()),
            "preserved_checkpoint": str(destination.resolve()),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        marker_tmp = best_root / f".BEST_EVAL.json.tmp.{os.getpid()}"
        marker_tmp.write_text(json.dumps(marker_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(marker_tmp, best_root / "BEST_EVAL.json")
        _write_best_tracker(best_root, step)
        _cleanup_old_best_resume_files(best_root, step)
        for old_dir in best_root.glob("iter_*"):
            if old_dir != destination and old_dir.is_dir():
                shutil.rmtree(old_dir)
        for old_eval_dir in (best_root / "eval").glob("iter_*"):
            if old_eval_dir != eval_snapshot_dir and old_eval_dir.is_dir():
                shutil.rmtree(old_eval_dir)
        logger.info(
            "Preserved best eval checkpoint: dataset=%s step=%s score=%.6f samples=%s path=%s",
            dataset_name,
            step,
            score,
            num_samples,
            destination,
        )
        return True
    except BaseException as exc:
        logger.exception(f"Failed to preserve best eval checkpoint {source}: {exc}")
        return False


def import_best_eval_checkpoint(config: Namespace) -> None:
    """Carry a compatible best checkpoint across a load-root/save-root resume."""
    raw_load = getattr(config, "load", None)
    if not raw_load:
        return
    save_root = Path(config.save).resolve()
    load_root = Path(raw_load).resolve()
    source_best = load_root if (load_root / "BEST_EVAL.json").is_file() else load_root / "best_eval"
    destination = save_root / "best_eval"
    if not (source_best / "BEST_EVAL.json").is_file() or source_best.resolve() == destination.resolve():
        return

    dataset_name = str(getattr(config, "best_actor_ckpt_eval_dataset", "agent_eval"))
    eval_fingerprint = str(getattr(config, "best_actor_ckpt_eval_fingerprint", ""))
    expected_samples = _expected_eval_samples(config, dataset_name)
    source_record = _load_preserved_best(source_best, dataset_name, eval_fingerprint, expected_samples)
    if source_record is None:
        return

    # Recover only an interrupted whole-root import. Inner checkpoint/eval
    # swaps are handled by _recover_best_root when the root is validated.
    import_temps = list(save_root.glob(".best_eval.import.tmp.*"))
    for temp in import_temps:
        if temp.is_dir():
            shutil.rmtree(temp)
    import_backups = [path for path in save_root.glob(".best_eval.import.old.*") if path.is_dir()]
    if import_backups:
        if len(import_backups) != 1:
            raise RuntimeError(f"ambiguous interrupted best-root imports: {import_backups}")
        backup = import_backups[0]
        if destination.exists():
            shutil.rmtree(backup)
        else:
            backup.rename(destination)

    destination_record = None
    if destination.exists():
        _recover_best_root(destination)
        destination_record = _load_preserved_best(
            destination,
            dataset_name,
            eval_fingerprint,
            expected_samples,
        )
        if destination_record is None:
            raise RuntimeError(f"existing destination best root has no marker: {destination}")
        if (destination_record["score"], destination_record["iteration"]) >= (
            source_record["score"],
            source_record["iteration"],
        ):
            return

    temp = save_root / f".best_eval.import.tmp.{os.getpid()}"
    backup = save_root / f".best_eval.import.old.{os.getpid()}"
    shutil.rmtree(temp, ignore_errors=True)
    try:
        shutil.copytree(source_best, temp, copy_function=os.link, symlinks=True)
        source_preserved = Path(source_record["preserved_checkpoint"]).resolve()
        source_eval = Path(source_record["eval_file"]).resolve()
        preserved_relative = source_preserved.relative_to(source_best.resolve())
        eval_relative = source_eval.relative_to(source_best.resolve())

        marker = temp / "BEST_EVAL.json"
        marker.unlink()
        temp_record = dict(source_record)
        temp_record.update(
            preserved_checkpoint=str((temp / preserved_relative).resolve()),
            eval_file=str((temp / eval_relative).resolve()),
        )
        marker.write_text(json.dumps(temp_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_best_tracker(temp, source_record["iteration"])
        _load_preserved_best(temp, dataset_name, eval_fingerprint, expected_samples)

        marker.unlink()
        imported_record = dict(source_record)
        imported_record.update(
            preserved_checkpoint=str((destination / preserved_relative).resolve()),
            eval_file=str((destination / eval_relative).resolve()),
            imported_from=str(source_best.resolve()),
            imported_at=datetime.now(timezone.utc).isoformat(),
        )
        marker.write_text(json.dumps(imported_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if destination.exists():
            destination.rename(backup)
        try:
            temp.rename(destination)
        except BaseException:
            if backup.exists() and not destination.exists():
                backup.rename(destination)
            raise
        shutil.rmtree(backup, ignore_errors=True)
        _load_preserved_best(destination, dataset_name, eval_fingerprint, expected_samples)
        logger.info(
            "Imported compatible best eval checkpoint from %s to %s: step=%s score=%.6f",
            source_best,
            destination,
            source_record["iteration"],
            source_record["score"],
        )
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
