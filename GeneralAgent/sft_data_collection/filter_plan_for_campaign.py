#!/usr/bin/env python3
"""Filter a newly generated SFT plan against campaign-level prior successes.

The campaign contract is task-level:
  - previous full runs are treated as evidence sources;
  - a task is considered solved for SFT only if it has a usable success;
  - new runs only schedule tasks without a usable prior success.

Default usable success is intentionally strict: resolved=True, actual skill-file
access detected, and no meta-talk contamination. That matches the current SFT
goal: collect trajectories that demonstrate real skill use, not just verifier
success.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    PROJECT_ROOT,
    display_path,
    infer_experiment_date,
    read_jsonl,
    repo_path,
)
from collect_successes import detect_meta_talk, detect_skill_use


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def run_root_from_value(value: str) -> Path:
    """Resolve a source run value that may be a run_id or a path.

    Do not call common.experiment_root() here: run_sft_pipeline exports
    EXPERIMENT_ROOT for the *current* run, which would misresolve old run ids.
    """
    raw = value.strip()
    if not raw:
        raise ValueError("empty source run")
    candidate = repo_path(raw)
    if "/" in raw or candidate.exists():
        return candidate
    date = infer_experiment_date(raw)
    return PROJECT_ROOT / "experiments" / date / raw


def run_id_from_root(run_root: Path) -> str:
    return run_root.name


def source_plan_records(run_root: Path) -> dict[str, dict[str, Any]]:
    run_id = run_id_from_root(run_root)
    candidates = [
        run_root / "plans" / f"{run_id}.combined.jsonl",
        run_root / "plans" / f"{run_id}.jsonl",
    ]
    for subdir in ["teacher", "done", "queue", "running"]:
        teacher_dir = run_root / "plans" / "chunks" / subdir
        if teacher_dir.exists():
            candidates.extend(sorted(teacher_dir.glob("*.teacher.jsonl")))

    by_trial: dict[str, dict[str, Any]] = {}
    for path in candidates:
        for row in read_jsonl(path):
            trial_id = str(row.get("trial_id", ""))
            if trial_id:
                by_trial[trial_id] = row
    return by_trial


def latest_status_rows(run_root: Path) -> dict[str, dict[str, Any]]:
    status_path = run_root / "logs" / "sft_collection" / "status.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(status_path):
        trial_id = str(row.get("trial_id", ""))
        if trial_id:
            latest[trial_id] = row
    return latest


def load_incremental_row(
    path_value: str | Path,
    task_id: str,
    cache: dict[Path, list[dict[str, Any]]],
) -> dict[str, Any]:
    path = repo_path(path_value)
    if not path.exists():
        return {}
    if path not in cache:
        cache[path] = read_jsonl(path)
    for row in cache[path]:
        if str(row.get("task_id") or row.get("instance_id") or "") == str(task_id):
            return row
    return {}


def load_fast_payload(
    record: dict[str, Any],
    status: dict[str, Any],
    incremental_cache: dict[Path, list[dict[str, Any]]],
    success_policy: str,
) -> dict[str, Any] | None:
    """Load only the evidence needed for campaign filtering.

    The data-quality dashboard intentionally parses every trajectory because it
    computes token statistics and samples. Campaign filtering only needs to know
    whether a task has a usable prior success. Reading every failed trajectory is
    unnecessarily slow on this filesystem, so this function first checks the
    verifier row and only opens trajectory JSON for resolved trials that may pass
    a strict skill-use policy.
    """
    task_id = str(record.get("task_id", ""))
    result_row = load_incremental_row(
        status.get("incremental_path") or record.get("incremental_path", ""),
        task_id,
        incremental_cache,
    )
    resolved = bool(result_row.get("resolved", False))
    if not result_row and status.get("returncode") == 0:
        trajectory_path = repo_path(status.get("trajectory_path") or record.get("trajectory_path", ""))
        if trajectory_path.exists():
            try:
                trajectory = json.loads(trajectory_path.read_text(encoding="utf-8", errors="ignore"))
                resolved = bool(trajectory.get("resolved", False))
            except Exception:
                resolved = False

    payload = {
        "trial_id": record.get("trial_id"),
        "run_id": record.get("run_id"),
        "bench": record.get("bench"),
        "task_id": task_id,
        "split": record.get("split"),
        "mode": record.get("mode"),
        "model_role": record.get("model_role"),
        "model": record.get("model"),
        "trial_index": record.get("trial_index"),
        "returncode": status.get("returncode"),
        "error_kind": status.get("error_kind") or "",
        "elapsed_sec": status.get("elapsed_sec"),
        "resolved": resolved,
        "score": result_row.get("score"),
        "trajectory_path": display_path(repo_path(status.get("trajectory_path") or record.get("trajectory_path", ""))),
        "used_skill": False,
        "used_skill_via_path": False,
        "meta_talk_detected": False,
    }
    if not resolved or success_policy == "any_resolved":
        return payload

    trajectory_path = repo_path(status.get("trajectory_path") or record.get("trajectory_path", ""))
    if not trajectory_path.exists():
        return payload
    try:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        payload["trajectory_load_error"] = type(exc).__name__
        return payload

    messages = trajectory.get("messages") or []
    skill_use = detect_skill_use(messages, injected_skill_names=[])
    meta_talk = detect_meta_talk(messages)
    payload["used_skill"] = bool(skill_use["used_skill"])
    payload["used_skill_via_path"] = bool(skill_use["used_skill_via_path"])
    payload["meta_talk_detected"] = bool(meta_talk["meta_talk_detected"])
    return payload


def success_ok(payload: dict[str, Any], policy: str) -> bool:
    if not payload.get("resolved"):
        return False
    if policy == "any_resolved":
        return True
    if policy == "strict_used":
        return bool(payload.get("used_skill"))
    if policy == "strict_used_non_meta":
        return bool(payload.get("used_skill")) and not bool(payload.get("meta_talk_detected"))
    raise ValueError(f"unknown success policy: {policy}")


def collect_source_evidence(
    source_roots: list[Path],
    success_policy: str,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    task_state: dict[tuple[str, str], dict[str, Any]] = {}
    success_payloads: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    incremental_cache: dict[Path, list[dict[str, Any]]] = {}

    for run_root in source_roots:
        records = source_plan_records(run_root)
        status = latest_status_rows(run_root)
        counters = Counter()
        for record in records.values():
            key = (str(record.get("bench", "")), str(record.get("task_id", "")))
            if not key[0] or not key[1]:
                continue
            state = task_state.setdefault(key, {
                "bench": key[0],
                "task_id": key[1],
                "planned_runs": [],
                "covered_runs": [],
                "usable_success_runs": [],
                "success_trials": [],
            })
            if run_root.name not in state["planned_runs"]:
                state["planned_runs"].append(run_root.name)

        for trial_id, row in status.items():
            record = records.get(trial_id)
            if not record:
                counters["status_without_record"] += 1
                continue
            key = (str(record.get("bench", "")), str(record.get("task_id", "")))
            state = task_state.setdefault(key, {
                "bench": key[0],
                "task_id": key[1],
                "planned_runs": [],
                "covered_runs": [],
                "usable_success_runs": [],
                "success_trials": [],
            })
            if run_root.name not in state["covered_runs"]:
                state["covered_runs"].append(run_root.name)
            payload = load_fast_payload(record, row, incremental_cache, success_policy)
            if payload is None:
                counters["missing_payload"] += 1
                continue
            counters["payloads"] += 1
            if payload.get("resolved"):
                counters["resolved"] += 1
            if payload.get("used_skill"):
                counters["used_skill"] += 1
            if success_ok(payload, success_policy):
                counters["usable_success"] += 1
                if run_root.name not in state["usable_success_runs"]:
                    state["usable_success_runs"].append(run_root.name)
                state["success_trials"].append({
                    "run_id": run_root.name,
                    "trial_id": payload.get("trial_id"),
                    "mode": payload.get("mode"),
                    "model": payload.get("model"),
                    "trajectory_path": payload.get("trajectory_path"),
                })
                success_payloads.append(payload)

        source_summaries.append({
            "run_id": run_root.name,
            "run_root": display_path(run_root),
            "records": len(records),
            "status": len(status),
            **dict(counters),
        })
    return task_state, success_payloads, source_summaries


def create_source_links(campaign_root: Path, source_roots: list[Path]) -> None:
    source_dir = campaign_root / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    for run_root in source_roots:
        link_path = source_dir / run_root.name
        if link_path.exists() or link_path.is_symlink():
            continue
        target = os.path.relpath(run_root, source_dir)
        try:
            link_path.symlink_to(target, target_is_directory=True)
        except OSError:
            write_json(link_path.with_suffix(".json"), {
                "run_id": run_root.name,
                "run_root": display_path(run_root),
                "note": "symlink creation failed; path recorded instead",
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--source-run", action="append", default=[])
    parser.add_argument("--source-runs", default="", help="Comma/space separated run ids or run roots.")
    parser.add_argument("--success-policy", default="strict_used_non_meta",
                        choices=["strict_used_non_meta", "strict_used", "any_resolved"])
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()

    plan_path = repo_path(args.plan)
    out_path = repo_path(args.out)
    campaign_root = repo_path(args.campaign_root)
    raw_sources: list[str] = []
    raw_sources.extend(args.source_run)
    for item in args.source_runs.replace(",", " ").split():
        if item.strip():
            raw_sources.append(item.strip())
    source_roots = []
    seen_roots = set()
    for raw in raw_sources:
        root = run_root_from_value(raw)
        if not root.exists():
            print(f"[campaign-filter] warning: source run missing: {display_path(root)}")
            continue
        resolved = root.resolve()
        if resolved in seen_roots:
            continue
        seen_roots.add(resolved)
        source_roots.append(root)

    campaign_root.mkdir(parents=True, exist_ok=True)
    create_source_links(campaign_root, source_roots)

    plan_rows = read_jsonl(plan_path)
    task_state, success_payloads, source_summaries = collect_source_evidence(
        source_roots,
        success_policy=args.success_policy,
    )
    solved_keys = {
        key for key, state in task_state.items()
        if state.get("usable_success_runs")
    }

    selected_rows: list[dict[str, Any]] = []
    all_keys: set[tuple[str, str]] = set()
    selected_keys: set[tuple[str, str]] = set()
    for row in plan_rows:
        key = (str(row.get("bench", "")), str(row.get("task_id", "")))
        all_keys.add(key)
        if key in solved_keys:
            continue
        selected_rows.append(row)
        selected_keys.add(key)

    write_jsonl(out_path, selected_rows)

    state_rows = []
    for bench, task_id in sorted(all_keys):
        state = task_state.get((bench, task_id), {
            "bench": bench,
            "task_id": task_id,
            "planned_runs": [],
            "covered_runs": [],
            "usable_success_runs": [],
            "success_trials": [],
        })
        state_rows.append({
            **state,
            "selected_for_next_run": (bench, task_id) in selected_keys,
            "previously_planned": bool(state.get("planned_runs")),
            "previously_covered": bool(state.get("covered_runs")),
            "previous_usable_success": bool(state.get("usable_success_runs")),
        })

    merged_dir = campaign_root / "merged"
    write_jsonl(merged_dir / "usable_success_payloads.jsonl", success_payloads)
    write_jsonl(merged_dir / "task_state.jsonl", state_rows)

    by_bench = defaultdict(Counter)
    for row in state_rows:
        bench = row["bench"]
        by_bench[bench]["total_tasks"] += 1
        if row["previously_planned"]:
            by_bench[bench]["previously_planned"] += 1
        if row["previously_covered"]:
            by_bench[bench]["previously_covered"] += 1
        if row["previous_usable_success"]:
            by_bench[bench]["previous_usable_success"] += 1
        if row["selected_for_next_run"]:
            by_bench[bench]["selected_tasks"] += 1

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign_root": display_path(campaign_root),
        "input_plan": display_path(plan_path),
        "output_plan": display_path(out_path),
        "success_policy": args.success_policy,
        "source_runs": source_summaries,
        "plan_records_before": len(plan_rows),
        "plan_records_after": len(selected_rows),
        "tasks_total": len(all_keys),
        "tasks_with_previous_usable_success": len(solved_keys & all_keys),
        "tasks_selected": len(selected_keys),
        "by_bench": {bench: dict(counter) for bench, counter in sorted(by_bench.items())},
        "artifacts": {
            "usable_success_payloads": display_path(merged_dir / "usable_success_payloads.jsonl"),
            "task_state": display_path(merged_dir / "task_state.jsonl"),
        },
    }
    manifest_path = repo_path(args.manifest) if args.manifest else campaign_root / "campaign_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
