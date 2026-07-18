#!/usr/bin/env python3
"""Phase 2: generate teacher fallback trials for unresolved Phase 1 tasks.

Phase 1 runs the 9B student in two branches:
  - student_use_skill: retrieval prompt + hidden use-skill nudge
  - student_no_skill:  retrieval prompt + hidden no-skill nudge

This script is intentionally task-level, not failed-rollout-level:
  1. group Phase 1 records by (bench, task_id)
  2. skip tasks where any Phase 1 trial resolved
  3. for tasks where every Phase 1 attempt is terminal but unsuccessful,
     create K `teacher_retrieval_reflection` trials on the configured teacher model
  4. put a compact failure dossier into UNIFIED_REFLECTION_CONTEXT:
       - each prior attempt's mode/trial/status
       - verifier/result row feedback when available
       - launcher error/log tail for missing-trajectory failures
       - last assistant action from the failed trajectory

The launcher remains unchanged: it reads the JSONL, routes records to the
model in each record, and skips completed trajectory files on re-run. That
gives resumability as long as the same RUN_ID is reused.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import DEFAULT_CONFIG, PROJECT_ROOT, experiment_status_path, load_json, read_jsonl, repo_path, secrets_path
from make_trial_plan import make_record


STUDENT_PHASE1_MODES = {"student_use_skill", "student_no_skill"}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def set_argv_option(argv: list[str], option: str, value: str) -> None:
    if option in argv:
        index = argv.index(option)
        if index + 1 >= len(argv):
            argv.append(value)
        else:
            argv[index + 1] = value
    else:
        argv.extend([option, value])


def secret_env() -> dict[str, str]:
    path = secrets_path()
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.replace("export ", "").strip()] = value.strip().strip("\"'")
    return values


def apply_teacher_endpoint(record: dict[str, Any]) -> None:
    """Route teacher records to the configured teacher endpoint when provided.

    Phase 1 records usually use the student endpoint from OPENAI_API_BASE.
    Phase 2 may use a remote MaaS teacher, so teacher fallback must not inherit
    the student endpoint or dummy local API key.
    """
    secrets = secret_env()
    teacher_api_base = (
        os.environ.get("TEACHER_OPENAI_API_BASE", "").strip()
        or os.environ.get("MAAS_API_BASE", "").strip()
        or secrets.get("MAAS_API_BASE", "").strip()
    )
    if teacher_api_base:
        env = record.setdefault("env", {})
        env["OPENAI_API_BASE"] = teacher_api_base
        argv = record.get("argv")
        if isinstance(argv, list):
            set_argv_option(argv, "--api-base", teacher_api_base)
            record["command_preview"] = shlex.join(str(part) for part in argv)
        # The launcher injects the real key at runtime from secrets/.env.secrets.
        # Keep plan files portable and safe to inspect.
        env.pop("OPENAI_API_KEY", None)
        env["UNIFIED_LLM_MAX_RETRIES"] = os.environ.get("TEACHER_LLM_MAX_RETRIES", "8")
        env["UNIFIED_LLM_RETRY_BACKOFF_SEC"] = os.environ.get("TEACHER_LLM_RETRY_BACKOFF_SEC", "5")
        env["UNIFIED_LLM_RETRY_MAX_BACKOFF_SEC"] = os.environ.get("TEACHER_LLM_RETRY_MAX_BACKOFF_SEC", "60")
        env["UNIFIED_LLM_REQUEST_TIMEOUT_SEC"] = os.environ.get("TEACHER_LLM_REQUEST_TIMEOUT_SEC", "300")
        env["UNIFIED_LLM_RETRY_HTTP_STATUSES"] = os.environ.get(
            "TEACHER_LLM_RETRY_HTTP_STATUSES",
            "408,409,425,429,500,502,503,504,529",
        )


def load_status_by_trial(run_id: str) -> dict[str, dict[str, Any]]:
    """Latest launcher status per trial_id.

    status.jsonl is append-only. If a trial was retried, the last row is the
    current state. Missing status means the launcher never attempted it.
    """
    rows = read_jsonl(experiment_status_path(run_id))
    status: dict[str, dict[str, Any]] = {}
    for row in rows:
        trial_id = row.get("trial_id")
        if trial_id:
            status[str(trial_id)] = row
    return status


def load_trajectory(record: dict[str, Any]) -> dict[str, Any] | None:
    path = repo_path(record["trajectory_path"])
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def load_result_row(record: dict[str, Any]) -> dict[str, Any]:
    """Find this trial's verifier row in incremental.jsonl.

    Harbor/claw rows usually key by task_id; SWE uses instance_id. Return {}
    if the runner failed before writing incremental output.
    """
    target = str(record["task_id"])
    for row in read_jsonl(record["incremental_path"]):
        row_task_id = row.get("task_id") or row.get("instance_id")
        if str(row_task_id) == target:
            return row
    return {}


def tail_file(path_value: str | Path | None, max_chars: int) -> str:
    if not path_value:
        return ""
    path = repo_path(path_value)
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def truncate(text: Any, max_chars: int) -> str:
    if text is None:
        return ""
    value = str(text).strip()
    if not value or len(value) <= max_chars:
        return value
    return value[: max_chars - 22].rstrip() + "\n... [truncated]"


def compact_json(value: Any, max_chars: int) -> str:
    if value in (None, "", [], {}):
        return ""
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        text = str(value)
    return truncate(text, max_chars)


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content or "")


def last_assistant_text(trajectory: dict[str, Any] | None, max_chars: int) -> str:
    if not trajectory:
        return ""
    for message in reversed(trajectory.get("messages") or []):
        if message.get("role") == "assistant":
            text = message_text(message).strip()
            if text:
                return truncate(text, max_chars)
    return ""


def compact_tool_call(tool_call: dict[str, Any], max_chars: int = 500) -> str:
    function = tool_call.get("function") or {}
    name = function.get("name") or tool_call.get("name") or "tool_call"
    arguments = function.get("arguments") or tool_call.get("arguments") or ""
    return f"{name}({truncate(arguments, max_chars)})"


def compact_trajectory_trace(
    trajectory: dict[str, Any] | None,
    *,
    max_chars: int,
    max_entries: int = 24,
) -> str:
    """Condense assistant/tool interaction history for reflection.

    Full trajectories can be tens of thousands of characters and do not fit
    safely in an env var. This trace preserves the action sequence: assistant
    plans, tool calls, and short tool-result snippets.
    """
    if not trajectory:
        return ""
    entries: list[str] = []
    for message in trajectory.get("messages") or []:
        role = message.get("role")
        if role == "assistant":
            text = truncate(message_text(message), 600)
            tool_calls = [
                compact_tool_call(tool_call)
                for tool_call in (message.get("tool_calls") or [])
                if isinstance(tool_call, dict)
            ]
            parts = []
            if text:
                parts.append(f"text={text}")
            if tool_calls:
                parts.append("tool_calls=[" + "; ".join(tool_calls[:4]) + "]")
            if parts:
                entries.append("assistant: " + " | ".join(parts))
        elif role == "tool":
            text = truncate(message_text(message), 500)
            if text:
                entries.append("tool: " + text)
    if not entries:
        return ""
    trace = "\n".join(entries[-max_entries:])
    return truncate(trace, max_chars)


def verifier_feedback(
    result_row: dict[str, Any],
    trajectory: dict[str, Any] | None,
    status_row: dict[str, Any] | None,
    max_chars: int,
) -> str:
    """Extract concrete verifier/runner signals without assuming bench schema."""
    pieces: list[str] = []

    scalar_keys = [
        "resolved", "score", "error", "finish_reason", "turns",
        "time_sec", "wall_sec", "status", "verdict",
    ]
    for key in scalar_keys:
        if key in result_row and result_row[key] not in (None, ""):
            pieces.append(f"{key}: {truncate(result_row[key], 300)}")

    for key in [
        "test_output", "tests_output", "pytest_output", "failure_output",
        "stderr", "stdout", "traceback", "verification_output",
        "verifier_output", "result", "details",
    ]:
        if key in result_row and result_row[key] not in (None, "", [], {}):
            pieces.append(f"{key}: {compact_json(result_row[key], max_chars // 2)}")

    if trajectory:
        for key in ["finish_reason", "error", "score", "resolved"]:
            if key in trajectory and trajectory[key] not in (None, ""):
                pieces.append(f"trajectory.{key}: {truncate(trajectory[key], 300)}")

    if status_row:
        status_bits = {
            key: status_row.get(key)
            for key in ["returncode", "error_kind", "elapsed_sec", "started_at", "finished_at"]
            if status_row.get(key) not in (None, "")
        }
        if status_bits:
            pieces.append(f"launcher_status: {compact_json(status_bits, 600)}")
        log_tail = tail_file(status_row.get("log_path"), max_chars // 2)
        if log_tail:
            pieces.append("launcher_log_tail:\n" + truncate(log_tail, max_chars // 2))

    if not pieces:
        return "(no verifier feedback was written; runner likely failed before verification)"
    return truncate("\n".join(pieces), max_chars)


def is_terminal_attempt(record: dict[str, Any], status_row: dict[str, Any] | None) -> bool:
    """A Phase 1 attempt is terminal only after an agent trajectory exists.

    Launcher status alone is not enough: Docker/tunnel outages can write a
    status row and incremental verifier row with turns=0, but those are
    infrastructure failures, not model attempts. Treating them as terminal
    incorrectly triggers teacher reflection on Docker errors.
    """
    return repo_path(record["trajectory_path"]).exists()


def build_attempt_outcome(
    record: dict[str, Any],
    status_row: dict[str, Any] | None,
    *,
    max_assistant_chars: int,
    max_trace_chars: int,
    max_feedback_chars: int,
) -> dict[str, Any] | None:
    if not is_terminal_attempt(record, status_row):
        return None
    trajectory = load_trajectory(record)
    result_row = load_result_row(record)
    resolved = bool(result_row.get("resolved", trajectory.get("resolved", False) if trajectory else False))
    return {
        "record": record,
        "status": status_row or {},
        "trajectory": trajectory,
        "result_row": result_row,
        "resolved": resolved,
        "last_assistant": last_assistant_text(trajectory, max_assistant_chars),
        "action_trace": compact_trajectory_trace(trajectory, max_chars=max_trace_chars),
        "feedback": verifier_feedback(result_row, trajectory, status_row, max_feedback_chars),
    }


def build_teacher_context(
    bench: str,
    task_id: str,
    outcomes: list[dict[str, Any]],
    *,
    max_attempts: int,
    max_total_chars: int,
) -> str:
    """Build the reflection context injected into the teacher prompt."""
    mode_order = {"student_use_skill": 0, "student_no_skill": 1}
    ordered = sorted(
        outcomes,
        key=lambda outcome: (
            int(outcome["record"].get("trial_index", 0)),
            mode_order.get(str(outcome["record"].get("mode", "")), 99),
        ),
    )
    if max_attempts > 0:
        ordered = ordered[:max_attempts]

    header = (
        f"Phase 1 student search failed for task {bench}/{task_id}. "
        f"There were {len(outcomes)} terminal student attempts and none passed. "
        "Use the concrete verifier feedback and prior actions below to avoid "
        "repeating failed approaches. You may read the retrieved skill files "
        "if they are relevant."
    )
    sections = [header, ""]

    for index, outcome in enumerate(ordered, start=1):
        record = outcome["record"]
        status = outcome["status"]
        section = [
            f"### Prior attempt {index}",
            f"- mode: {record.get('mode')} / trial_index: {record.get('trial_index')}",
            f"- trajectory_path: {record.get('trajectory_path')}",
            f"- returncode: {status.get('returncode', 'unknown')} / error_kind: {status.get('error_kind', '') or 'none'}",
            "- verifier feedback:",
            textwrap.indent(outcome["feedback"], "  "),
            "- compact trajectory trace:",
            textwrap.indent(outcome["action_trace"] or "(no trajectory trace available)", "  "),
            "- last assistant action:",
            textwrap.indent(outcome["last_assistant"] or "(empty assistant turn)", "  "),
        ]
        sections.append("\n".join(section))
        sections.append("")

    context = "\n".join(sections).strip()
    return truncate(context, max_total_chars)


def phase1_records_only(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record for record in records
        if record.get("mode") in STUDENT_PHASE1_MODES
        and record.get("model_role") == "student"
        and "_reflection" not in str(record.get("mode", ""))
    ]


def build_teacher_records(
    plan_records: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    teacher_model: str,
    teacher_trials: int,
    require_complete_phase1: bool,
    max_attempts: int,
    max_assistant_chars: int,
    max_trace_chars: int,
    max_feedback_chars: int,
    max_context_chars: int,
    limit_tasks: int,
) -> tuple[list[dict[str, Any]], dict[str, str], Counter]:
    """Return teacher fallback records plus task-level skip reasons."""
    budgets = config["budgets"]
    retrieval_top_n = int(budgets["retrieval_top_n"])
    per_bench_budgets = budgets.get("per_bench_budgets", {})

    phase1 = phase1_records_only(plan_records)
    if not phase1:
        return [], {}, Counter({"no_phase1_records": 1})

    run_ids = {str(record["run_id"]) for record in phase1}
    if len(run_ids) != 1:
        raise ValueError(f"expected one run_id in phase1 plan, got {sorted(run_ids)}")
    status_by_trial = load_status_by_trial(next(iter(run_ids)))

    by_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in phase1:
        by_task[(str(record["bench"]), str(record["task_id"]))].append(record)

    records: list[dict[str, Any]] = []
    skip_reasons: dict[str, str] = {}
    counters: Counter = Counter()
    generated_tasks = 0

    for (bench, task_id), task_records in sorted(by_task.items()):
        outcomes: list[dict[str, Any]] = []
        missing: list[str] = []
        for record in task_records:
            outcome = build_attempt_outcome(
                record,
                status_by_trial.get(record["trial_id"]),
                max_assistant_chars=max_assistant_chars,
                max_trace_chars=max_trace_chars,
                max_feedback_chars=max_feedback_chars,
            )
            if outcome is None:
                missing.append(record["trial_id"])
            else:
                outcomes.append(outcome)

        if missing and require_complete_phase1:
            skip_reasons[f"{bench}/{task_id}"] = f"phase1_incomplete {len(missing)}/{len(task_records)} missing"
            counters["phase1_incomplete"] += 1
            continue
        if not outcomes:
            skip_reasons[f"{bench}/{task_id}"] = "phase1_not_run"
            counters["phase1_not_run"] += 1
            continue
        if any(outcome["resolved"] for outcome in outcomes):
            skip_reasons[f"{bench}/{task_id}"] = "phase1_already_succeeded"
            counters["phase1_already_succeeded"] += 1
            continue
        if limit_tasks > 0 and generated_tasks >= limit_tasks:
            skip_reasons[f"{bench}/{task_id}"] = "limit_tasks"
            counters["limit_tasks"] += 1
            continue

        bench_budgets = {
            **budgets,
            **(per_bench_budgets.get(bench, {}) if isinstance(per_bench_budgets, dict) else {}),
        }
        max_turns = int(bench_budgets["max_turns"])
        max_time = int(bench_budgets["max_time"]) if bench_budgets.get("max_time") is not None else None
        context = build_teacher_context(
            bench,
            task_id,
            outcomes,
            max_attempts=max_attempts,
            max_total_chars=max_context_chars,
        )
        first = task_records[0]
        for trial_index in range(teacher_trials):
            record = make_record(
                run_id=first["run_id"],
                date=first["date"],
                bench=bench,
                task_id=task_id,
                split=first["split"],
                mode="teacher_retrieval_reflection",
                model_role="teacher",
                model=teacher_model,
                arm="retrieval",
                trial_index=90 + trial_index,
                retrieval_jsonl=first.get("retrieval_jsonl"),
                retrieval_top_n=retrieval_top_n,
                max_turns=max_turns,
                max_time=max_time,
                retrieval_covered=bool(first.get("retrieval_covered", True)),
                implicit_mode="use_skill",
                reflection_context=context,
            )
            apply_teacher_endpoint(record)
            records.append(record)
        generated_tasks += 1
        counters["teacher_tasks"] += 1

    counters["teacher_records"] = len(records)
    counters["phase1_tasks"] = len(by_task)
    return records, skip_reasons, counters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", required=True, help="Phase 1 plan JSONL")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default="",
                        help="Teacher fallback plan output path; default <plan>.teacher_reflection.jsonl")
    parser.add_argument("--teacher-model", default="",
                        help="Teacher model id. Default reads models.teacher from config.")
    parser.add_argument("--teacher-trials", type=int, default=None,
                        help="Teacher rollouts per unresolved task. Default budgets.teacher_trials.")
    parser.add_argument("--allow-incomplete-phase1", action="store_true",
                        help="Generate teacher fallback from whatever Phase 1 attempts exist. "
                             "Default requires every planned Phase 1 trial to be terminal.")
    parser.add_argument("--max-attempts", type=int, default=8,
                        help="Max prior attempts summarized into each teacher prompt.")
    parser.add_argument("--max-assistant-chars", type=int, default=900)
    parser.add_argument("--max-trace-chars", type=int, default=2200)
    parser.add_argument("--max-feedback-chars", type=int, default=1800)
    parser.add_argument("--max-context-chars", type=int, default=24000)
    parser.add_argument("--limit-tasks", type=int, default=0,
                        help="Debug cap on unresolved tasks to emit (0 = all).")
    args = parser.parse_args()

    plan_path = repo_path(args.plan)
    if not plan_path.exists():
        raise SystemExit(f"plan file not found: {display_path(plan_path)}")
    config = load_json(args.config)
    teacher_model = args.teacher_model or str(config["models"]["teacher"])
    teacher_trials = (
        int(args.teacher_trials)
        if args.teacher_trials is not None
        else int(config["budgets"].get("teacher_trials", 4))
    )

    plan_records = read_jsonl(plan_path)
    teacher_records, skip_reasons, counters = build_teacher_records(
        plan_records,
        config=config,
        teacher_model=teacher_model,
        teacher_trials=teacher_trials,
        require_complete_phase1=not args.allow_incomplete_phase1,
        max_attempts=args.max_attempts,
        max_assistant_chars=args.max_assistant_chars,
        max_trace_chars=args.max_trace_chars,
        max_feedback_chars=args.max_feedback_chars,
        max_context_chars=args.max_context_chars,
        limit_tasks=args.limit_tasks,
    )

    out_path = repo_path(args.out) if args.out else plan_path.with_suffix(".teacher_reflection.jsonl")
    write_jsonl(out_path, teacher_records)
    skip_path = out_path.with_suffix(".skipped.txt")
    skip_path.write_text(
        "\n".join(f"{key}\t{reason}" for key, reason in sorted(skip_reasons.items())) + ("\n" if skip_reasons else ""),
        encoding="utf-8",
    )

    print(f"loaded Phase 1 plan: {len(plan_records)} records")
    print(f"teacher fallback model: {teacher_model}")
    print(f"teacher trials per unresolved task: {teacher_trials}")
    print(f"teacher fallback records generated: {len(teacher_records)}")
    for key in [
        "phase1_tasks", "teacher_tasks", "phase1_already_succeeded",
        "phase1_incomplete", "phase1_not_run", "limit_tasks",
    ]:
        if counters.get(key):
            print(f"  {key}: {counters[key]}")
    print(f"wrote {len(teacher_records)} teacher records to {display_path(out_path)}")
    print(f"skipped tasks written to {display_path(skip_path)}")


if __name__ == "__main__":
    main()
