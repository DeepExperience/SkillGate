#!/usr/bin/env python3
"""Build fixed train/test splits for SFT data collection.

The split is the foundation of every later artifact: trial plans, collected
trajectories, and final SFT records. To prevent test-set contamination after
SFT training, it MUST be:
  - deterministic (reproducible from the same source files)
  - frozen (this script's output checked into outputs/splits/default/)
  - never re-rolled mid-collection (would silently leak test → train)

Re-running this script overwrites outputs/splits/default/. If you do that
during an active SFT run, holdout_split.json will lose its provenance link
to whatever data was already collected. Don't.

Per-bench split rules (see config['split_sources']):
  claw / tb2 / sb_ns: deterministic even-spaced holdout from the full task list
  seta_synth / swe_lite: train comes from a large pre-baked task list, test is
    the existing small eval set (seta_baseline_30 / SWE ALL_IMAGES); any
    overlap is removed from train (so the test set wins ties).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    dump_json,
    filter_known_bad_tasks,
    load_json,
    parse_swe_all_images,
    read_task_lines,
    repo_path,
    sha256_file,
    stable_even_holdout,
    tree_manifest,
    write_task_lines,
)


# ---------------------------------------------------------------------------
# Per-bench task discovery (each runs once at split time)
# ---------------------------------------------------------------------------

def discover_tb2_tasks(dataset_dir: str) -> list[str]:
    """TB 2.0: a task is a subdir with tests/test.sh."""
    root = repo_path(dataset_dir)
    return sorted(
        task_dir.name
        for task_dir in root.iterdir()
        if task_dir.is_dir() and (task_dir / "tests" / "test.sh").exists()
    )


def discover_skillsbench_no_skill_tasks(dataset_dir: str, exclude_tasks: list[str]) -> list[str]:
    """SkillsBench-no-skills: any non-hidden subdir, minus exclusions.

    `scheduling-email-assistant` is excluded because it requires Google OAuth
    credentials that we don't provide; including it just adds noise. See
    `_EXCLUDED_TASKS` in run_unified_harbor.py.
    """
    root = repo_path(dataset_dir)
    excluded = set(exclude_tasks)
    return sorted(
        task_dir.name
        for task_dir in root.iterdir()
        if task_dir.is_dir()
        and not task_dir.name.startswith(".")
        and task_dir.name not in excluded
    )


def discover_seta_holdout(holdout_dataset_dir: str) -> list[str]:
    """SETA holdout = whatever lives in the small eval dataset dir."""
    root = repo_path(holdout_dataset_dir)
    return sorted(task_dir.name for task_dir in root.iterdir() if task_dir.is_dir())


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_even(tasks: list[str], holdout_count: int) -> tuple[list[str], list[str]]:
    """Even-spaced holdout. Train = sorted(tasks) - holdout."""
    holdout = stable_even_holdout(tasks, holdout_count)
    holdout_set = set(holdout)
    train = [task_id for task_id in sorted(tasks) if task_id not in holdout_set]
    return train, holdout


def build_splits(config: dict[str, Any]) -> dict[str, Any]:
    sources = config["split_sources"]

    # claw: source is a fixed task_id list (T-series, 161 tasks).
    claw_tasks = read_task_lines(sources["claw"]["task_file"])
    claw_train, claw_test = split_even(claw_tasks, int(sources["claw"]["holdout_count"]))

    # tb2: discover from filesystem (subdirs with tests/test.sh).
    tb2_tasks = filter_known_bad_tasks("tb2", discover_tb2_tasks(sources["tb2"]["dataset_dir"]))
    tb2_train, tb2_test = split_even(tb2_tasks, int(sources["tb2"]["holdout_count"]))

    # sb_ns: discover from filesystem, drop scheduling-email-assistant et al.
    sb_tasks = filter_known_bad_tasks(
        "sb_ns",
        discover_skillsbench_no_skill_tasks(
            sources["sb_ns"]["dataset_dir"],
            sources["sb_ns"].get("exclude_tasks", []),
        ),
    )
    sb_train, sb_test = split_even(sb_tasks, int(sources["sb_ns"]["holdout_count"]))

    # seta_synth: train from a large pre-baked list (300 task_ids); test from
    # the existing small eval set. Any overlap is removed from train so the
    # test set wins ties (test is sacred, train is replaceable).
    seta_train_source = filter_known_bad_tasks(
        "seta_synth",
        read_task_lines(sources["seta_synth"]["train_task_file"]),
    )
    seta_test = filter_known_bad_tasks(
        "seta_synth",
        discover_seta_holdout(sources["seta_synth"]["holdout_dataset_dir"]),
    )
    seta_test_set = set(seta_test)
    seta_train = [t for t in sorted(seta_train_source) if t not in seta_test_set]
    seta_overlap = sorted(set(seta_train_source) & seta_test_set)

    # swe_lite: same pattern; test comes from the legacy ALL_IMAGES list
    # parsed out of run_unified_swe.py (avoids importing the heavy module).
    swe_train_source = filter_known_bad_tasks(
        "swe_lite",
        read_task_lines(sources["swe_lite"]["train_task_file"]),
    )
    swe_test = sorted(filter_known_bad_tasks(
        "swe_lite",
        parse_swe_all_images(sources["swe_lite"]["holdout_from_runner_all_images"]),
    ))
    swe_test_set = set(swe_test)
    swe_train = [t for t in sorted(swe_train_source) if t not in swe_test_set]
    swe_overlap = sorted(set(swe_train_source) & swe_test_set)

    # Hard sanity: train and test MUST be disjoint per-bench. If this ever
    # fires it indicates a bug in the splitting logic above.
    benches: dict[str, dict[str, list[str]]] = {
        "claw": {"train": claw_train, "test": claw_test},
        "tb2": {"train": tb2_train, "test": tb2_test},
        "sb_ns": {"train": sb_train, "test": sb_test},
        "seta_synth": {"train": seta_train, "test": seta_test},
        "swe_lite": {"train": swe_train, "test": swe_test},
    }
    for bench, split in benches.items():
        overlap = set(split["train"]) & set(split["test"])
        if overlap:
            raise AssertionError(
                f"split bug: train ∩ test not empty for {bench}: {sorted(overlap)[:5]}..."
            )

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "unit": "task_id",
            "strict_rule": "No task_id may appear in both train and test for the same bench.",
            "seta_swe_note": (
                "SETA/SWE use existing small evaluation sets as external heldout; "
                "most heldout ids are outside the large train source files."
            ),
            "tie_breaking": "Test set wins on overlap (overlap removed from train).",
        },
        "benches": benches,
        "source_counts": {
            "claw_161_source": len(claw_tasks),
            "tb2_runnable_source": len(tb2_tasks),
            "sb_ns_runnable_source": len(sb_tasks),
            "seta_train_source": len(seta_train_source),
            "seta_external_holdout": len(seta_test),
            "seta_overlap_removed_from_train": len(seta_overlap),
            "seta_overlap_ids": seta_overlap[:20],  # capped for readability
            "swe_train_source": len(swe_train_source),
            "swe_external_holdout": len(swe_test),
            "swe_overlap_removed_from_train": len(swe_overlap),
            "swe_overlap_ids": swe_overlap[:20],
        },
        "freeze_manifest": build_freeze_manifest(config),
    }


def build_freeze_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """Snapshot the inputs we depend on (skill library + retrieval files) so
    a future run can detect 'oh, the lib changed since we collected this'."""
    retrieval_files = config["frozen_retrieval"]["files"]
    return {
        "skill_library": tree_manifest(config["frozen_skill_library"]),
        "retrieval": {
            bench: {
                "path": path_value,
                "sha256": sha256_file(path_value),
            }
            for bench, path_value in retrieval_files.items()
        },
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_split_files(split_payload: dict[str, Any], output_dir: Path) -> None:
    """Write one train.txt + one test.txt per bench, plus a JSON of everything."""
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    for bench, split in split_payload["benches"].items():
        write_task_lines(train_dir / f"{bench}.txt", split["train"])
        write_task_lines(test_dir / f"{bench}.txt", split["test"])
    dump_json(output_dir / "holdout_split.json", split_payload)


def print_summary(split_payload: dict[str, Any], output_dir: Path) -> None:
    print(f"wrote split to {output_dir.relative_to(PROJECT_ROOT)}")
    for bench, split in split_payload["benches"].items():
        print(f"  {bench:12s} train={len(split['train']):4d}  test={len(split['test']):3d}")
    counts = split_payload["source_counts"]
    if counts["seta_overlap_removed_from_train"]:
        print(f"  ⚠ seta train lost {counts['seta_overlap_removed_from_train']} task to test holdout")
    if counts["swe_overlap_removed_from_train"]:
        print(f"  ⚠ swe train lost {counts['swe_overlap_removed_from_train']} instance to test holdout")
    skill_count = split_payload["freeze_manifest"]["skill_library"]["skill_count"]
    print(f"  freeze: {skill_count} skills, {len(split_payload['freeze_manifest']['retrieval'])} retrieval files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--out-dir",
        default="GeneralAgent/sft_data_collection/outputs/splits/default",
        help="Output directory; will be overwritten on re-run.",
    )
    args = parser.parse_args()

    config = load_json(args.config)
    output_dir = repo_path(args.out_dir)
    split_payload = build_splits(config)
    write_split_files(split_payload, output_dir)
    print_summary(split_payload, output_dir)


if __name__ == "__main__":
    main()
