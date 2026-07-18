#!/usr/bin/env python3
"""Strict, read-only completion gate for a plan-driven eval70 run.

The evaluator may append more than one status row for a retried trial.  This
audit joins the plan and status log by ``trial_id`` and treats the last status
row in file order as authoritative.  A row is complete only when that latest
status is successful and its unique incremental result and trajectory are both
present, valid, and consistent with the plan.

The command never writes to the run directory.  It exits 0 only for a complete
and internally consistent run, and exits 2 for an incomplete or invalid run.

Example:
    python3 ops/monitor/audit_eval70_artifacts.py \
      --run-root experiments/rl/runs/<experiment>/eval/<eval-id>/rows/<row-id> \
      --expected-records 280
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def payload_task_id(row: dict[str, Any]) -> str:
    """Return the task identity used by all eval70 payload schemas."""
    if "task_id" in row:
        return str(row["task_id"])
    if "instance_id" in row:  # SWE payloads use instance_id.
        return str(row["instance_id"])
    return ""


class Audit:
    def __init__(self, run_root: Path, expected_records: int, expected_tasks: int) -> None:
        self.run_root = run_root
        self.expected_records = expected_records
        self.expected_tasks = expected_tasks
        self.issues: list[str] = []

    def issue(self, message: str) -> None:
        self.issues.append(message)

    def read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8", errors="strict") as handle:
                for line_number, raw_line in enumerate(handle, 1):
                    if not raw_line.strip():
                        continue
                    value = json.loads(raw_line)
                    if not isinstance(value, dict):
                        raise TypeError(f"line {line_number} is {type(value).__name__}, not object")
                    rows.append(value)
        except Exception as exc:
            self.issue(f"bad JSONL {path}: {type(exc).__name__}: {exc}")
        return rows

    def load_trajectory(self, path: Path, trial_id: str) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
            if not isinstance(value, dict):
                raise TypeError(f"top-level value is {type(value).__name__}, not object")
            return value
        except Exception as exc:
            self.issue(
                f"{trial_id}: bad or missing trajectory {path}: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

    def check_unique_plan_field(self, plan: list[dict[str, Any]], field: str) -> None:
        values = [str(row.get(field, "")) for row in plan]
        if len(values) != self.expected_records or len(set(values)) != self.expected_records or "" in values:
            self.issue(
                f"plan {field}: rows={len(values)} unique={len(set(values))} "
                f"blank={values.count('')} expected={self.expected_records}"
            )

    def check_plan_shape(self, plan: list[dict[str, Any]]) -> None:
        if self.expected_tasks <= 0:
            self.issue(f"expected_tasks must be positive, got {self.expected_tasks}")
            return
        if self.expected_records % self.expected_tasks:
            self.issue(
                f"expected_records={self.expected_records} is not divisible by "
                f"expected_tasks={self.expected_tasks}"
            )
            return

        repeats = self.expected_records // self.expected_tasks
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in plan:
            key = (str(row.get("bench", "")), str(row.get("task_id", "")))
            groups.setdefault(key, []).append(row)
        repeat_histogram = Counter(len(rows) for rows in groups.values())
        if len(groups) != self.expected_tasks or set(repeat_histogram) != {repeats}:
            self.issue(
                f"plan task/repeat shape: tasks={len(groups)} expected_tasks={self.expected_tasks} "
                f"repeat_histogram={dict(sorted(repeat_histogram.items()))}"
            )
        expected_indices = set(range(repeats))
        for (bench, task_id), rows in groups.items():
            indices = {row.get("trial_index") for row in rows}
            if indices != expected_indices:
                self.issue(
                    f"plan {bench}/{task_id}: trial_index={sorted(map(str, indices))} "
                    f"expected={sorted(expected_indices)}"
                )

    def compare_artifact_sets(self, plan: list[dict[str, Any]]) -> None:
        planned_incrementals = {
            repo_path(row["incremental_path"])
            for row in plan
            if row.get("incremental_path")
        }
        actual_incrementals = {
            path.resolve()
            for path in self.run_root.glob("results/*/*/incremental.jsonl")
        }
        if planned_incrementals != actual_incrementals:
            self.issue(
                "incremental path set mismatch: "
                f"missing={len(planned_incrementals - actual_incrementals)} "
                f"extra={len(actual_incrementals - planned_incrementals)}"
            )

        planned_trajectories = {
            repo_path(row["trajectory_path"])
            for row in plan
            if row.get("trajectory_path")
        }
        actual_trajectories = {
            path.resolve()
            for path in self.run_root.glob("results/*/*/trajectories/*.json")
        }
        if planned_trajectories != actual_trajectories:
            self.issue(
                "trajectory path set mismatch: "
                f"missing={len(planned_trajectories - actual_trajectories)} "
                f"extra={len(actual_trajectories - planned_trajectories)}"
            )

    def audit_record(
        self,
        record: dict[str, Any],
        latest_status: dict[str, dict[str, Any]],
    ) -> bool:
        trial_id = str(record.get("trial_id", ""))
        status = latest_status.get(trial_id)
        if status is None:
            self.issue(f"{trial_id}: no status row")
            return False
        if status.get("returncode") != 0 or status.get("error_kind"):
            self.issue(
                f"{trial_id}: latest status returncode={status.get('returncode')} "
                f"error_kind={status.get('error_kind')!r}"
            )
            return False

        for field in ("run_id", "bench", "task_id", "mode", "model"):
            if str(status.get(field, "")) != str(record.get(field, "")):
                self.issue(f"{trial_id}: status/plan {field} mismatch")
                return False
        for field in ("incremental_path", "trajectory_path"):
            if repo_path(status.get(field, "")) != repo_path(record.get(field, "")):
                self.issue(f"{trial_id}: status/plan {field} mismatch")
                return False

        incremental_path = repo_path(record["incremental_path"])
        incremental_rows = self.read_jsonl(incremental_path) if incremental_path.exists() else []
        if len(incremental_rows) != 1:
            self.issue(
                f"{trial_id}: incremental rows={len(incremental_rows)} "
                f"path={incremental_path}"
            )
            return False

        trajectory_path = repo_path(record["trajectory_path"])
        trajectory = self.load_trajectory(trajectory_path, trial_id)
        if trajectory is None:
            return False
        incremental = incremental_rows[0]
        expected_task_id = str(record.get("task_id", ""))
        incremental_task_id = payload_task_id(incremental)
        trajectory_task_id = payload_task_id(trajectory)
        if incremental_task_id != expected_task_id or trajectory_task_id != expected_task_id:
            self.issue(
                f"{trial_id}: payload task mismatch incremental={incremental_task_id!r} "
                f"trajectory={trajectory_task_id!r} plan={expected_task_id!r}"
            )
            return False

        incremental_resolved = incremental.get("resolved")
        trajectory_resolved = trajectory.get("resolved")
        if (
            not isinstance(incremental_resolved, bool)
            or not isinstance(trajectory_resolved, bool)
            or incremental_resolved != trajectory_resolved
        ):
            self.issue(
                f"{trial_id}: invalid or mismatched resolved values "
                f"incremental={incremental_resolved!r} trajectory={trajectory_resolved!r}"
            )
            return False

        if incremental.get("dataset") != trajectory.get("dataset") or not incremental.get("dataset"):
            self.issue(
                f"{trial_id}: invalid or mismatched dataset values "
                f"incremental={incremental.get('dataset')!r} "
                f"trajectory={trajectory.get('dataset')!r}"
            )
            return False
        messages = trajectory.get("messages")
        if not isinstance(messages, list) or not messages:
            self.issue(f"{trial_id}: trajectory messages are missing or empty")
            return False
        return True

    def run(self) -> tuple[int, int, int, dict[int, int]]:
        plan_paths = sorted((self.run_root / "plans").glob("*.jsonl"))
        if len(plan_paths) != 1:
            self.issue(f"expected one plan JSONL, found {len(plan_paths)}: {plan_paths}")
        plan = self.read_jsonl(plan_paths[0]) if plan_paths else []
        for field in ("trial_id", "incremental_path", "trajectory_path"):
            self.check_unique_plan_field(plan, field)
        self.check_plan_shape(plan)

        status_path = self.run_root / "logs" / "sft_collection" / "status.jsonl"
        status_rows = self.read_jsonl(status_path)
        latest_status: dict[str, dict[str, Any]] = {}
        for row in status_rows:
            trial_id = str(row.get("trial_id", ""))
            if not trial_id:
                self.issue("status row is missing trial_id")
                continue
            # File append order is canonical. A retry status is appended later.
            latest_status[trial_id] = row

        plan_trial_ids = {str(row.get("trial_id", "")) for row in plan}
        unplanned_status = set(latest_status) - plan_trial_ids
        if unplanned_status:
            self.issue(f"unplanned latest status trial_ids={len(unplanned_status)}")

        self.compare_artifact_sets(plan)
        valid_records = sum(self.audit_record(row, latest_status) for row in plan)
        attempt_counts = Counter(str(row.get("trial_id", "")) for row in status_rows)
        attempt_histogram = dict(sorted(Counter(attempt_counts.values()).items()))
        return len(plan), len(latest_status), valid_records, attempt_histogram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", required=True, help="Eval70 row root containing plans/results/logs")
    parser.add_argument("--expected-records", required=True, type=int, help="Expected unique trials")
    parser.add_argument(
        "--expected-tasks",
        type=int,
        default=70,
        help="Expected unique (bench, task_id) pairs (default: 70)",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=40,
        help="Maximum individual audit errors to print (default: 40)",
    )
    args = parser.parse_args()
    if args.expected_records <= 0:
        parser.error("--expected-records must be positive")
    if args.max_errors < 0:
        parser.error("--max-errors must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    audit = Audit(repo_path(args.run_root), args.expected_records, args.expected_tasks)
    plan_records, latest_statuses, valid_records, attempt_histogram = audit.run()
    print(
        f"run_root={audit.run_root} plan={plan_records} "
        f"latest_status={latest_statuses} valid={valid_records}/{args.expected_records} "
        f"issues={len(audit.issues)} status_attempt_histogram={attempt_histogram}"
    )
    for message in audit.issues[: args.max_errors]:
        print(f"ERROR {message}")
    if len(audit.issues) > args.max_errors:
        print(f"ERROR ... {len(audit.issues) - args.max_errors} more")
    return 0 if not audit.issues and valid_records == args.expected_records else 2


if __name__ == "__main__":
    sys.exit(main())
