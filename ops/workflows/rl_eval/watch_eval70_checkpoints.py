#!/usr/bin/env python3
"""Evaluate every new Relax checkpoint and publish the latest row to a paper table.

The watcher is intentionally thin: checkpoint validation, CP2-to-HF export,
serving, trial resume, and row finalization remain owned by
``run_eval70_checkpoint_set.sh``. This process only discovers complete markers,
invokes that canonical workflow one checkpoint at a time, validates the 70x4
result, and atomically replaces one named method in the paper master.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "ops/workflows/rl_eval"
sys.path.insert(0, str(EVAL_DIR))

from analyze_eval70_3tables import analyze, collect  # noqa: E402
from analyze_slate_reads import BENCH_MAP, load_manifest, summarize  # noqa: E402


BENCH_ORDER = ("claw", "sb_ns", "seta", "swe", "tb2")
EXPECTED_TASKS = {"claw": 14, "sb_ns": 8, "seta": 30, "swe": 10, "tb2": 8}
SECTION_HEADINGS = (
    ("## 一、", "## 二、"),
    ("## 二、", "## 三、"),
    ("## 三、", "## 四、"),
)
STATUS_START = "<!-- masked-task-only-watch:start -->"
STATUS_END = "<!-- masked-task-only-watch:end -->"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def complete_iterations(checkpoint_root: Path, start: int, final: int) -> list[int]:
    found = []
    for marker in checkpoint_root.glob("iter_*/.relax_complete.json"):
        match = re.fullmatch(r"iter_(\d+)", marker.parent.name)
        if not match:
            continue
        iteration = int(match.group(1))
        if not start <= iteration <= final:
            continue
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if int(record.get("iteration", -1)) == iteration and record.get("files"):
            found.append(iteration)
    return sorted(set(found))


def row_label(iteration: int, final: int) -> str:
    return f"masked-task-only-{'final' if iteration == final else 'iter'}{iteration}"


def evaluator_command(args: argparse.Namespace, iteration: int, report: Path) -> list[str]:
    group = f"{args.group_prefix}_iter_{iteration:04d}"
    return [
        "bash",
        str(EVAL_DIR / "run_eval70_checkpoint_set.sh"),
        "--group",
        group,
        "--eval-id",
        args.eval_id,
        "--skill-mode",
        "mixed",
        "--snapshot",
        str(args.snapshot),
        "--manifest",
        str(args.manifest),
        "--local-only",
        "--workers",
        str(args.workers),
        "--report",
        str(report),
        "--checkpoint",
        args.owner,
        row_label(iteration, args.final_iteration),
        str(args.checkpoint_root),
        str(iteration),
        "none",
    ]


def find_completed_row(args: argparse.Namespace, iteration: int) -> Path:
    rows = ROOT / "experiments/rl/runs" / args.owner / "eval" / args.eval_id / "rows"
    expected_label = row_label(iteration, args.final_iteration)
    matches = []
    for row_json in rows.glob("*/row.json"):
        row = read_json(row_json)
        if row.get("label") == expected_label and row.get("status") == "completed":
            matches.append(row_json.parent)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one completed row for label={expected_label!r}, found {len(matches)} under {rows}"
        )
    row_root = matches[0]
    if not (row_root / "ROW_DONE").is_file():
        raise RuntimeError(f"completed row lacks ROW_DONE: {row_root}")
    return row_root


def summarize_row(row_root: Path, manifest_path: Path) -> dict[str, Any]:
    row_root = row_root.resolve()
    manifest_path = manifest_path.resolve()
    trials = collect(str(row_root))
    if len(trials) != 280:
        raise RuntimeError(f"expected 280 trials, got {len(trials)}: {row_root}")
    task_counts = Counter((row["bench"], row["task"]) for row in trials)
    if len(task_counts) != 70 or set(task_counts.values()) != {4}:
        raise RuntimeError(
            f"expected 70 tasks x4; tasks={len(task_counts)} repeat_counts={sorted(set(task_counts.values()))}"
        )
    bench_tasks = Counter(bench for bench, _task in task_counts)
    if dict(bench_tasks) != EXPECTED_TASKS:
        raise RuntimeError(f"unexpected task distribution: {dict(bench_tasks)}")
    if any(not row.get("has_traj") for row in trials):
        raise RuntimeError("one or more finalized records lack a trajectory")

    manifest = load_manifest(str(manifest_path))
    task_keys = {
        f"{BENCH_MAP.get(row['bench'], row['bench'])}::{row['task']}" for row in trials
    }
    missing_manifest_tasks = sorted(task_keys - manifest.keys())
    if missing_manifest_tasks:
        raise RuntimeError(f"manifest lacks eval tasks: {missing_manifest_tasks}")
    malformed_slates = {
        task_key: len(manifest[task_key])
        for task_key in sorted(task_keys)
        if len(manifest[task_key]) != 16
    }
    if malformed_slates:
        raise RuntimeError(f"expected 16 unique skills per task: {malformed_slates}")

    outcome = analyze(trials)
    behavior = summarize(trials, manifest, "read_names_agent")
    unknown_name_counts: Counter[str] = Counter()
    for row in trials:
        task_key = f"{BENCH_MAP.get(row['bench'], row['bench'])}::{row['task']}"
        for name in row.get("read_names_agent") or []:
            if name not in manifest[task_key]:
                unknown_name_counts[name] += 1
    if unknown_name_counts:
        print(
            "[row-warning] model attempted unmapped skill paths; excluding them from "
            f"category attribution: {dict(unknown_name_counts)}",
            flush=True,
        )
    return {
        "records": len(trials),
        "row_root": str(row_root.relative_to(ROOT)),
        "t1": outcome["t1"],
        "t1_task": outcome["t1_task"],
        "behavior": {
            "read_any": int(behavior["read_any"]),
            "oracle": int(behavior["per_cat_read"]["oracle"]),
            "misleading": int(behavior["per_cat_read"]["misleading"]),
            "p_oracle_given_read": float(behavior["p_gold_given_read"]),
            "avg_names_per_trial": float(behavior["avg_names_per_trial"]),
            "unknown_names": int(behavior["unknown_names"]),
            "unknown_name_counts": dict(sorted(unknown_name_counts.items())),
        },
    }


def percent(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.1f}%"


def paper_rows(summary: dict[str, Any], iteration: int, final: int) -> tuple[str, str, str]:
    display = f"masked-task-only（{'final99' if iteration == final else f'iter{iteration}；训练中'}）"
    t1 = summary["t1"]
    trial_cells = []
    for bench in ("ALL", *BENCH_ORDER):
        passed, total, _errors = t1[bench]
        trial_cells.append(f"{passed} ({percent(passed, total)})")
    trial_row = f"| {display} | " + " | ".join(trial_cells) + " |"

    t1_task = summary["t1_task"]
    task_cells = []
    for bench in ("ALL", *BENCH_ORDER):
        passed, total = t1_task[bench]
        task_cells.append(f"{passed} ({percent(passed, total)})" if bench == "ALL" else str(passed))
    task_row = f"| {display} | " + " | ".join(task_cells) + " |"

    behavior = summary["behavior"]
    n = int(summary["records"])
    behavior_cells = (
        percent(behavior["read_any"], n),
        percent(behavior["oracle"], n),
        percent(behavior["misleading"], n),
        f"{100 * behavior['p_oracle_given_read']:.1f}%",
        f"{behavior['avg_names_per_trial']:.2f}",
    )
    behavior_row = f"| {display} | " + " | ".join(behavior_cells) + " |"
    return trial_row, task_row, behavior_row


def replace_method_row(section: str, replacement: str) -> str:
    pattern = re.compile(r"^\| (?:\*\*)?masked-task-only[^\n]*$", re.MULTILINE)
    matches = pattern.findall(section)
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one masked-task-only row in section, found {len(matches)}")
    return pattern.sub(replacement, section, count=1)


def publish_master(
    master: Path,
    summary: dict[str, Any],
    iteration: int,
    final: int,
    report: Path,
) -> None:
    text = master.read_text(encoding="utf-8")
    replacements = paper_rows(summary, iteration, final)
    for (start_heading, end_heading), replacement in zip(SECTION_HEADINGS, replacements, strict=True):
        start = text.index(start_heading)
        end = text.index(end_heading, start)
        section = replace_method_row(text[start:end], replacement)
        text = text[:start] + section + text[end:]

    status = (
        f"{STATUS_START}\n"
        f"> **masked-task-only 自动评测状态**：最新发布 checkpoint `iter{iteration}`，"
        f"280/280 records 已验证；owner-local row：`{summary['row_root']}`；"
        f"单 checkpoint 报告：`{report.relative_to(ROOT)}`；"
        f"未映射的 agent skill 路径尝试：{summary['behavior']['unknown_names']}"
        f"（作为模型行为保留，不计入 oracle/misleading 等类别）。"
        f"{'final99 已完成。' if iteration == final else 'watcher 将继续等待并评测后续完整 checkpoint。'}\n"
        f"{STATUS_END}"
    )
    block_pattern = re.compile(
        re.escape(STATUS_START) + r".*?" + re.escape(STATUS_END), re.DOTALL
    )
    if block_pattern.search(text):
        text = block_pattern.sub(status, text, count=1)
    else:
        insertion = text.index("## 四、")
        text = text[:insertion] + status + "\n\n" + text[insertion:]
    atomic_write_text(master, text)


def validate_args(args: argparse.Namespace) -> None:
    for path, kind in (
        (args.checkpoint_root, "directory"),
        (args.snapshot, "directory"),
        (args.manifest, "file"),
        (args.master, "file"),
    ):
        valid = path.is_dir() if kind == "directory" else path.is_file()
        if not valid:
            raise SystemExit(f"missing {kind}: {path}")
    owner_manifest = ROOT / "experiments/rl/runs" / args.owner / "experiment.json"
    if not owner_manifest.is_file():
        raise SystemExit(f"missing owner experiment: {owner_manifest}")
    if args.start_iteration > args.final_iteration:
        raise SystemExit("--start-iteration must not exceed --final-iteration")


def load_state(args: argparse.Namespace, path: Path) -> dict[str, Any]:
    state = read_json(path)
    identity = {
        "owner": args.owner,
        "checkpoint_root": str(args.checkpoint_root),
        "eval_id": args.eval_id,
        "start_iteration": args.start_iteration,
        "final_iteration": args.final_iteration,
    }
    if state:
        for key, expected in identity.items():
            if state.get(key) != expected:
                raise RuntimeError(f"watch state mismatch for {key}: {state.get(key)!r} != {expected!r}")
        return state
    return {
        "schema_version": 1,
        **identity,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "completed": {},
        "failures": [],
    }


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    args.work_root.mkdir(parents=True, exist_ok=True)
    (args.work_root / "reports").mkdir(exist_ok=True)
    lock_handle = (args.work_root / "watch.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(f"another watcher holds {args.work_root / 'watch.lock'}") from exc

    state_path = args.work_root / "state.json"
    state = load_state(args, state_path)
    completed = {int(value) for value in state["completed"]}
    print(f"[watch-start] {utc_now()} completed={sorted(completed)}", flush=True)

    while True:
        available = complete_iterations(
            args.checkpoint_root, args.start_iteration, args.final_iteration
        )
        pending = [iteration for iteration in available if iteration not in completed]
        if args.dry_run:
            print(f"[dry-run] available={available} pending={pending}")
            for iteration in pending:
                report = args.work_root / "reports" / f"iter_{iteration:07d}.md"
                print(subprocess.list2cmdline(evaluator_command(args, iteration, report)))
            return 0
        if args.final_iteration in completed:
            print(f"[watch-complete] final iteration {args.final_iteration} published", flush=True)
            return 0
        if not pending:
            print(
                f"[watch-wait] {utc_now()} no new complete checkpoint; available={available}",
                flush=True,
            )
            if args.once:
                return 0
            time.sleep(args.poll_seconds)
            continue

        iteration = pending[0]
        report = args.work_root / "reports" / f"iter_{iteration:07d}.md"
        command = evaluator_command(args, iteration, report)
        env = os.environ.copy()
        env.update(
            WORKERS=str(args.workers),
            DOCKER_START_CAP=str(args.docker_start_cap),
            DOCKER_HOST_VALUE=args.docker_host,
            CHECKPOINT_WAIT_SEC=str(min(args.poll_seconds, 120)),
        )
        print(f"[eval-start] {utc_now()} iteration={iteration}", flush=True)
        return_code = subprocess.run(command, cwd=ROOT, env=env, check=False).returncode
        if return_code:
            failure = {"iteration": iteration, "at": utc_now(), "return_code": return_code}
            state["failures"].append(failure)
            state["updated_at"] = utc_now()
            atomic_write_json(state_path, state)
            print(f"[eval-failed] {failure}; retrying after {args.retry_seconds}s", flush=True)
            if args.once:
                return return_code
            time.sleep(args.retry_seconds)
            continue

        row_root = find_completed_row(args, iteration)
        summary = summarize_row(row_root, args.manifest)
        publish_master(args.master, summary, iteration, args.final_iteration, report)
        state["completed"][str(iteration)] = {
            "completed_at": utc_now(),
            "report": str(report.relative_to(ROOT)),
            **summary,
        }
        state["latest_published_iteration"] = iteration
        state["updated_at"] = utc_now()
        atomic_write_json(state_path, state)
        completed.add(iteration)
        print(
            f"[publish-done] {utc_now()} iteration={iteration} row={summary['row_root']}",
            flush=True,
        )
        if args.once:
            return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--start-iteration", type=int, required=True)
    parser.add_argument("--final-iteration", type=int, default=99)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--master", type=Path, default=ROOT / "z_cc_terminal_imgs/skillgate_paper_master.md")
    parser.add_argument("--eval-id", default="eval70-mixed-r4-4023950044")
    parser.add_argument("--group-prefix", default="masked_task_only_checkpoint_watch")
    parser.add_argument("--work-root", type=Path, default=ROOT / "experiments/skillgate_paper/masked_task_only_checkpoint_watch")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--docker-start-cap", type=int, default=24)
    parser.add_argument("--docker-host", default="unix:///tmp/local-docker-overlay2.sock")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--retry-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("checkpoint_root", "snapshot", "manifest", "master", "work_root"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
