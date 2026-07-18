#!/usr/bin/env python3
"""Build a live data-quality dashboard for an SFT collection run.

This script is intentionally read-only. It joins:
  - the phase1 plan plus any generated teacher plans,
  - the append-only launcher status file,
  - completed trajectories and incremental verifier rows.

It writes a compact Markdown report and JSON snapshot under the run's
`reports/` directory, so it can be run repeatedly while collection is active.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    PROJECT_ROOT,
    display_path,
    experiment_combined_plan_path,
    experiment_plan_path,
    experiment_root,
    experiment_status_path,
    read_jsonl,
    repo_path,
)
from collect_successes import (
    USE_SKILL_BRANCH_MODES,
    NO_SKILL_BRANCH_MODES,
    TEACHER_MODES,
    detect_meta_talk,
    detect_skill_use,
    estimate_tokens,
)


def phase_for_mode(mode: str) -> str:
    if mode in TEACHER_MODES or mode.startswith("teacher_"):
        return "phase2_teacher"
    if mode.startswith("student_"):
        return "phase1_student"
    if mode.startswith("eval_"):
        return "eval"
    return "other"


def read_plan_records(run_id: str, run_root: Path) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    """Load phase1, combined, and generated teacher plans without duplicates."""
    candidates: list[Path] = []

    def add_candidate(path: Path) -> None:
        path = repo_path(path)
        if path.exists() and path not in candidates:
            candidates.append(path)

    for path in [
        experiment_combined_plan_path(run_id),
        experiment_plan_path(run_id),
        run_root / "plans" / f"{run_id}.combined.jsonl",
        run_root / "plans" / f"{run_id}.jsonl",
    ]:
        add_candidate(path)

    for resume_dir in [
        run_root / "plans" / "resume",
    ]:
        if resume_dir.exists():
            for path in sorted(resume_dir.glob("*.jsonl")):
                add_candidate(path)

    for teacher_dir in [
        run_root / "plans" / "chunks" / "teacher",
        run_root / "plans" / "chunks" / "done",
        run_root / "plans" / "chunks" / "queue",
        run_root / "plans" / "chunks" / "running",
        run_root / "plans" / "chunks" / "failed",
    ]:
        if teacher_dir.exists():
            for path in sorted(teacher_dir.glob("*.teacher.jsonl")):
                add_candidate(path)

    quarantine_root = run_root / "plans" / "chunks" / "quarantine_infra_teacher"
    if quarantine_root.exists():
        for path in sorted(quarantine_root.glob("*/teacher/*.teacher.jsonl")):
            add_candidate(path)

    by_trial: dict[str, dict[str, Any]] = {}
    used_paths: list[Path] = []
    for path in candidates:
        rows = read_jsonl(path)
        if not rows:
            continue
        used_paths.append(path)
        for row in rows:
            trial_id = str(row.get("trial_id", ""))
            if trial_id:
                by_trial[trial_id] = row
    return by_trial, used_paths


def latest_status_rows(run_root: Path) -> list[dict[str, Any]]:
    status_path = run_root / "logs" / "sft_collection" / "status.jsonl"
    if not status_path.exists():
        return []
    by_trial: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(status_path):
        trial_id = str(row.get("trial_id", ""))
        if trial_id:
            by_trial[trial_id] = row
    return list(by_trial.values())


def load_incremental_row(path_value: str | Path, task_id: str, cache: dict[Path, list[dict[str, Any]]]) -> dict[str, Any]:
    path = repo_path(path_value)
    if path not in cache:
        cache[path] = read_jsonl(path)
    # A retry appends a newer row for the same task to incremental.jsonl.
    # Use the latest matching row so dashboards reflect refill/rerun results.
    for row in reversed(cache[path]):
        if str(row.get("task_id") or row.get("instance_id") or "") == str(task_id):
            return row
    return {}


def load_trial_payload(
    record: dict[str, Any],
    status: dict[str, Any],
    incremental_cache: dict[Path, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    trajectory_path = repo_path(status.get("trajectory_path") or record.get("trajectory_path", ""))
    if not trajectory_path.exists():
        return None

    try:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return {
            "trial_id": record.get("trial_id"),
            "bench": record.get("bench"),
            "task_id": record.get("task_id"),
            "mode": record.get("mode"),
            "model": record.get("model"),
            "phase": phase_for_mode(str(record.get("mode", ""))),
            "trajectory_path": display_path(trajectory_path),
            "trajectory_load_error": type(exc).__name__,
            "resolved": False,
            "used_skill": False,
            "meta_talk_detected": False,
            "estimated_tokens": 0,
        }

    result_row = load_incremental_row(
        status.get("incremental_path") or record.get("incremental_path", ""),
        str(record.get("task_id", "")),
        incremental_cache,
    )
    messages = trajectory.get("messages") or []
    skill_use = detect_skill_use(messages, injected_skill_names=[])
    meta_talk = detect_meta_talk(messages)
    resolved = bool(result_row.get("resolved", trajectory.get("resolved", False)))
    return {
        "trial_id": record.get("trial_id"),
        "run_id": record.get("run_id"),
        "bench": record.get("bench"),
        "task_id": str(record.get("task_id", "")),
        "split": record.get("split"),
        "mode": record.get("mode"),
        "model_role": record.get("model_role"),
        "model": record.get("model"),
        "phase": phase_for_mode(str(record.get("mode", ""))),
        "trial_index": record.get("trial_index"),
        "returncode": status.get("returncode"),
        "error_kind": status.get("error_kind") or "",
        "elapsed_sec": status.get("elapsed_sec"),
        "resolved": resolved,
        "score": result_row.get("score", trajectory.get("score")),
        "turns": result_row.get("turns"),
        "time_sec": result_row.get("time_sec") or result_row.get("wall_sec"),
        "used_skill": bool(skill_use["used_skill"]),
        "used_skill_via_path": bool(skill_use["used_skill_via_path"]),
        "meta_talk_detected": bool(meta_talk["meta_talk_detected"]),
        "estimated_tokens": estimate_tokens(messages),
        "had_reflection_context": bool(trajectory.get("reflection_context", "")),
        "direct_sft_candidate": bool(record.get("direct_sft_candidate", False)),
        "trajectory_path": display_path(trajectory_path),
    }


def bench_from_eval_row(row: dict[str, Any], incremental_path: Path) -> str:
    """Normalize benchmark names from dynamic eval rows.

    Dynamic full-eval launchers do not have SFT collection plans/status files.
    Their only authoritative index is the per-bench incremental verifier row,
    whose `dataset` names use runner-native labels.  Normalize them here so the
    dashboard remains comparable with SFT collection dashboards.
    """
    dataset = str(row.get("dataset") or "")
    mapping = {
        "claw-eval": "claw",
        "seta": "seta_synth",
        "swe-gym-lite": "swe_lite",
        "skillsbench-no-skills": "sb_ns",
        "tb2": "tb2",
    }
    if dataset in mapping:
        return mapping[dataset]
    parts = incremental_path.parts
    if "results" in parts:
        index = parts.index("results")
        if index + 1 < len(parts):
            return parts[index + 1]
    return dataset or "unknown"


def mode_from_eval_row(row: dict[str, Any], incremental_path: Path) -> str:
    """Infer eval mode from row/path for dynamic eval dashboards."""
    skill_arm = str(row.get("skill_arm") or "")
    if skill_arm in {"baseline", "retrieval", "top1_skill_text"}:
        return f"eval_{skill_arm}"
    path_text = str(incremental_path)
    if "retrieval" in path_text:
        return "eval_retrieval"
    return "eval_baseline"


def trajectory_path_for_eval_row(row: dict[str, Any], incremental_path: Path) -> Path:
    task_id = str(row.get("task_id") or row.get("instance_id") or "")
    return incremental_path.parent / "trajectories" / f"{task_id}.json"


def load_eval_payloads_from_incrementals(run_root: Path) -> list[dict[str, Any]]:
    """Fallback for dynamic eval runs that do not emit collection plans/status.

    `run_dynamic_bench.sh` writes benchmark-local `incremental.jsonl` files plus
    sibling trajectory files, but no `logs/sft_collection/status.jsonl`.  The
    normal SFT dashboard path therefore reports zero trajectories.  This fallback
    reconstructs a read-only dashboard directly from those verifier rows.
    """
    payloads: list[dict[str, Any]] = []
    for incremental_path in sorted((run_root / "results").glob("*/**/incremental.jsonl")):
        for row in read_jsonl(incremental_path):
            task_id = str(row.get("task_id") or row.get("instance_id") or "")
            trajectory_path = trajectory_path_for_eval_row(row, incremental_path)
            messages: list[dict[str, Any]] = []
            if trajectory_path.exists():
                try:
                    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8", errors="ignore"))
                    messages = trajectory.get("messages") or []
                except Exception:
                    messages = []
            skill_use = detect_skill_use(messages, injected_skill_names=[]) if messages else {
                "used_skill": False,
                "used_skill_via_path": False,
            }
            meta_talk = detect_meta_talk(messages) if messages else {"meta_talk_detected": False}
            estimated_tokens = estimate_tokens(messages) if messages else int(
                (row.get("input_tokens") or 0) + (row.get("output_tokens") or 0)
            )
            mode = mode_from_eval_row(row, incremental_path)
            payloads.append({
                "trial_id": f"{bench_from_eval_row(row, incremental_path)}::{task_id}::{mode}",
                "run_id": run_root.name,
                "bench": bench_from_eval_row(row, incremental_path),
                "task_id": task_id,
                "split": "eval",
                "mode": mode,
                "model_role": "eval",
                "model": "",
                "phase": phase_for_mode(mode),
                "trial_index": 0,
                "returncode": 0 if not row.get("error") else None,
                "error_kind": row.get("error") or "",
                "elapsed_sec": row.get("wall_sec") or row.get("time_sec"),
                "resolved": bool(row.get("resolved")),
                "score": row.get("score"),
                "turns": row.get("turns"),
                "time_sec": row.get("time_sec") or row.get("wall_sec"),
                "used_skill": bool(skill_use["used_skill"]),
                "used_skill_via_path": bool(skill_use["used_skill_via_path"]),
                "meta_talk_detected": bool(meta_talk["meta_talk_detected"]),
                "estimated_tokens": estimated_tokens,
                "had_reflection_context": bool(row.get("reflection_context") or row.get("reflection_text")),
                "direct_sft_candidate": False,
                "trajectory_path": display_path(trajectory_path) if trajectory_path.exists() else "",
                "incremental_path": display_path(incremental_path),
            })
    return payloads


def pct(num: int | float, den: int | float) -> str:
    if not den:
        return "0.0%"
    return f"{100.0 * num / den:.1f}%"


def numeric_summary(values: list[float]) -> dict[str, float | int]:
    values = sorted(value for value in values if isinstance(value, (int, float)))
    if not values:
        return {"n": 0}

    def q(frac: float) -> float:
        index = int(round(frac * (len(values) - 1)))
        return round(values[max(0, min(len(values) - 1, index))], 1)

    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 1),
        "p50": q(0.50),
        "p90": q(0.90),
        "p95": q(0.95),
        "max": round(values[-1], 1),
    }


def summarize_groups(payloads: list[dict[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        key = tuple(payload.get(field, "") for field in key_fields)
        groups[key].append(payload)

    rows: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        trajectories = len(items)
        resolved = sum(1 for item in items if item.get("resolved"))
        used = sum(1 for item in items if item.get("used_skill"))
        success_used = sum(1 for item in items if item.get("resolved") and item.get("used_skill"))
        success_used_non_meta = sum(
            1 for item in items
            if item.get("resolved") and item.get("used_skill") and not item.get("meta_talk_detected")
        )
        meta = sum(1 for item in items if item.get("meta_talk_detected"))
        row = {field: value for field, value in zip(key_fields, key)}
        row.update({
            "trajectories": trajectories,
            "resolved": resolved,
            "resolved_rate": resolved / trajectories if trajectories else 0.0,
            "strict_used_skill": used,
            "strict_used_skill_rate": used / trajectories if trajectories else 0.0,
            "success_strict_used_skill": success_used,
            "success_strict_used_skill_rate_of_success": success_used / resolved if resolved else 0.0,
            "success_strict_used_skill_non_meta": success_used_non_meta,
            "meta_talk": meta,
            "meta_talk_rate": meta / trajectories if trajectories else 0.0,
            "tokens": numeric_summary([item.get("estimated_tokens") for item in items]),
            "elapsed_sec": numeric_summary([item.get("elapsed_sec") for item in items]),
        })
        rows.append(row)
    return rows


def summarize_tasks(payloads: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], Counter]:
    by_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        by_task[(str(payload.get("bench", "")), str(payload.get("task_id", "")))].append(payload)

    tasks: dict[str, dict[str, Any]] = {}
    bucket_counts: Counter = Counter()
    for (bench, task_id), items in sorted(by_task.items()):
        use_success = any(item.get("resolved") and item.get("mode") in USE_SKILL_BRANCH_MODES for item in items)
        no_success = any(item.get("resolved") and item.get("mode") in NO_SKILL_BRANCH_MODES for item in items)
        teacher_success = any(item.get("resolved") and item.get("mode") in TEACHER_MODES for item in items)
        strict_used_success = any(item.get("resolved") and item.get("used_skill") for item in items)
        if use_success and no_success:
            bucket = "both_solvable"
        elif use_success:
            bucket = "skill_helpful"
        elif no_success:
            bucket = "no_skill_solvable"
        elif teacher_success:
            bucket = "teacher_only"
        else:
            bucket = "unresolved"
        key = f"{bench}/{task_id}"
        tasks[key] = {
            "bench": bench,
            "task_id": task_id,
            "bucket": bucket,
            "trials": len(items),
            "success_trials": sum(1 for item in items if item.get("resolved")),
            "strict_used_skill_trials": sum(1 for item in items if item.get("used_skill")),
            "success_strict_used_skill_trials": sum(
                1 for item in items if item.get("resolved") and item.get("used_skill")
            ),
            "strict_used_success": strict_used_success,
            "phase1_success": use_success or no_success,
            "teacher_success": teacher_success,
        }
        bucket_counts[bucket] += 1
    return tasks, bucket_counts


def chunk_progress(run_root: Path) -> dict[str, Any]:
    chunks = run_root / "plans" / "chunks"
    phase1 = chunks / "phase1"
    progress: dict[str, Any] = {}
    if phase1.exists():
        progress["phase1_chunks_total"] = len(list(phase1.glob("chunk_*.jsonl")))
    for name in ["teacher", "queue", "running", "done", "failed"]:
        directory = chunks / name
        if directory.exists():
            progress[f"{name}_teacher_chunks"] = len(list(directory.glob("*.teacher.jsonl")))
    return progress


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def write_markdown(snapshot: dict[str, Any], output_path: Path) -> None:
    totals = snapshot["totals"]
    lines = [
        f"# SFT Data Quality Dashboard: {snapshot['run_id']}",
        "",
        f"- generated_at: {snapshot['generated_at']}",
        f"- run_root: `{snapshot['run_root']}`",
        f"- plan_records_loaded: {totals['plan_records_loaded']}",
        f"- status_rows_latest: {totals['status_rows_latest']}",
        f"- trajectories_loaded: {totals['trajectories_loaded']}",
        f"- raw_success: {totals['resolved']} / {totals['trajectories_loaded']} ({pct(totals['resolved'], totals['trajectories_loaded'])})",
        f"- strict_used_skill_success: {totals['success_strict_used_skill']} / {totals['resolved']} ({pct(totals['success_strict_used_skill'], totals['resolved'])})",
        f"- strict_used_skill_success_non_meta: {totals['success_strict_used_skill_non_meta']}",
        f"- meta_talk: {totals['meta_talk']}",
        "",
        "## Chunk Progress",
        "",
    ]
    for key, value in sorted(snapshot["chunk_progress"].items()):
        lines.append(f"- {key}: {value}")

    lines += ["", "## Phase Summary", ""]
    phase_rows = []
    for row in snapshot["by_phase"]:
        phase_rows.append([
            row["phase"],
            row["trajectories"],
            f"{row['resolved']} ({pct(row['resolved'], row['trajectories'])})",
            f"{row['strict_used_skill']} ({pct(row['strict_used_skill'], row['trajectories'])})",
            f"{row['success_strict_used_skill']} ({pct(row['success_strict_used_skill'], row['resolved'])})",
            row["success_strict_used_skill_non_meta"],
            row["meta_talk"],
        ])
    lines += markdown_table(
        ["phase", "traj", "success", "strict_used", "success_used", "success_used_non_meta", "meta"],
        phase_rows,
    )

    lines += ["", "## Bench / Mode Summary", ""]
    mode_rows = []
    for row in snapshot["by_phase_bench_mode"]:
        mode_rows.append([
            row["phase"],
            row["bench"],
            row["mode"],
            row["trajectories"],
            f"{row['resolved']} ({pct(row['resolved'], row['trajectories'])})",
            f"{row['strict_used_skill']} ({pct(row['strict_used_skill'], row['trajectories'])})",
            f"{row['success_strict_used_skill']} ({pct(row['success_strict_used_skill'], row['resolved'])})",
            row["success_strict_used_skill_non_meta"],
            row["tokens"].get("p50", "-"),
            row["tokens"].get("p90", "-"),
        ])
    lines += markdown_table(
        ["phase", "bench", "mode", "traj", "success", "strict_used", "success_used", "success_used_non_meta", "tok_p50", "tok_p90"],
        mode_rows,
    )

    lines += ["", "## Task Buckets", ""]
    bucket_counts = snapshot["task_bucket_counts"]
    for bucket in ["both_solvable", "skill_helpful", "no_skill_solvable", "teacher_only", "unresolved"]:
        lines.append(f"- {bucket}: {bucket_counts.get(bucket, 0)}")

    lines += ["", "## Teacher Added Tasks", ""]
    teacher_rows = []
    for item in snapshot["teacher_added_tasks"][:30]:
        teacher_rows.append([
            item["bench"],
            item["task_id"],
            item["teacher_success_trials"],
            item["teacher_success_strict_used_skill_trials"],
        ])
    lines += markdown_table(["bench", "task", "teacher_success", "teacher_success_used"], teacher_rows)

    lines += ["", "## Sample Strict Used-Skill Successes", ""]
    sample_rows = []
    for item in snapshot["sample_success_strict_used_skill"][:20]:
        sample_rows.append([
            item["bench"],
            item["task_id"],
            item["mode"],
            item["model"],
            item["estimated_tokens"],
            f"`{item['trajectory_path']}`",
        ])
    lines += markdown_table(["bench", "task", "mode", "model", "tokens", "trajectory"], sample_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_snapshot(run_id: str, run_root: Path) -> dict[str, Any]:
    plan_records, plan_paths = read_plan_records(run_id, run_root)
    status_rows = latest_status_rows(run_root)
    incremental_cache: dict[Path, list[dict[str, Any]]] = {}
    payloads: list[dict[str, Any]] = []
    missing_plan = 0
    for status in status_rows:
        trial_id = str(status.get("trial_id", ""))
        record = plan_records.get(trial_id)
        if not record:
            missing_plan += 1
            continue
        payload = load_trial_payload(record, status, incremental_cache)
        if payload is not None:
            payloads.append(payload)
    if not payloads and not status_rows:
        payloads = load_eval_payloads_from_incrementals(run_root)

    task_summary, bucket_counts = summarize_tasks(payloads)
    teacher_added_tasks = []
    for task in task_summary.values():
        if task["teacher_success"] and not task["phase1_success"]:
            teacher_added_tasks.append({
                "bench": task["bench"],
                "task_id": task["task_id"],
                "teacher_success_trials": task["success_trials"],
                "teacher_success_strict_used_skill_trials": task["success_strict_used_skill_trials"],
            })

    totals = {
        "plan_records_loaded": len(plan_records),
        "plan_files_loaded": len(plan_paths),
        "status_rows_latest": len(status_rows),
        "status_missing_plan": missing_plan,
        "trajectories_loaded": len(payloads),
        "resolved": sum(1 for item in payloads if item.get("resolved")),
        "strict_used_skill": sum(1 for item in payloads if item.get("used_skill")),
        "success_strict_used_skill": sum(
            1 for item in payloads if item.get("resolved") and item.get("used_skill")
        ),
        "success_strict_used_skill_non_meta": sum(
            1 for item in payloads
            if item.get("resolved") and item.get("used_skill") and not item.get("meta_talk_detected")
        ),
        "meta_talk": sum(1 for item in payloads if item.get("meta_talk_detected")),
    }
    snapshot = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "run_root": display_path(run_root),
        "plan_files": [display_path(path) for path in plan_paths],
        "totals": totals,
        "chunk_progress": chunk_progress(run_root),
        "by_phase": summarize_groups(payloads, ("phase",)),
        "by_phase_bench": summarize_groups(payloads, ("phase", "bench")),
        "by_phase_bench_mode": summarize_groups(payloads, ("phase", "bench", "mode")),
        "task_bucket_counts": dict(bucket_counts),
        "task_summary": task_summary,
        "teacher_added_tasks": teacher_added_tasks,
        "sample_success_strict_used_skill": [
            item for item in payloads
            if item.get("resolved") and item.get("used_skill") and not item.get("meta_talk_detected")
        ][:50],
    }
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_id")
    parser.add_argument("--run-root", default="", help="Override experiments/<date>/<run_id>")
    parser.add_argument("--out-dir", default="", help="Default: <run_root>/reports")
    args = parser.parse_args()

    run_root = repo_path(args.run_root) if args.run_root else experiment_root(args.run_id)
    output_dir = repo_path(args.out_dir) if args.out_dir else run_root / "reports"
    snapshot = build_snapshot(args.run_id, run_root)
    json_path = output_dir / "data_quality_dashboard.json"
    md_path = output_dir / "data_quality_dashboard.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(snapshot, md_path)

    totals = snapshot["totals"]
    print(f"run_id={args.run_id}")
    print(f"run_root={display_path(run_root)}")
    print(f"trajectories={totals['trajectories_loaded']} status={totals['status_rows_latest']} plan={totals['plan_records_loaded']}")
    print(f"resolved={totals['resolved']} ({pct(totals['resolved'], totals['trajectories_loaded'])})")
    print(
        "success_strict_used_skill="
        f"{totals['success_strict_used_skill']} ({pct(totals['success_strict_used_skill'], totals['resolved'])} of success)"
    )
    print(f"success_strict_used_skill_non_meta={totals['success_strict_used_skill_non_meta']}")
    print(f"dashboard_md={display_path(md_path)}")
    print(f"dashboard_json={display_path(json_path)}")


if __name__ == "__main__":
    main()
