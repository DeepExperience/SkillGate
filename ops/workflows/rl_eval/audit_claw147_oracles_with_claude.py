#!/usr/bin/env python3
"""Audit and repair Claw147 oracle skill bodies with Claude Code.

The workflow is restartable. Each oracle is reviewed independently, accepted
frontmatter is preserved byte-for-byte, and only a validated replacement body
may be written back to the canonical eval_claw_147 snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import build_eval_claw_147_slate as builder


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "skill_libraries/snapshots/rl/eval_claw_147"
DEFAULT_WORK = ROOT / "experiments/skill_slate_build/eval_claw_147/claude_oracle_audit"
DEFAULT_REPORT = ROOT / "z_cc_terminal_imgs/claw oracle skill fix.md"
AUDIT_ID = builder.CLAUDE_ORACLE_AUDIT_ID
QUOTA_PATTERNS = (
    "usage limit",
    "you've hit your limit",
    "you have hit your limit",
    "weekly limit",
    "5-hour limit",
    "five-hour limit",
    "resets at",
    "resets in",
)
RAW_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n.*?\n---[ \t]*\n", re.S)
SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "needs_rewrite"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "replacement_body": {"type": "string"},
        "replacement_verified": {"type": "boolean"},
    },
    "required": [
        "verdict",
        "confidence",
        "issues",
        "evidence",
        "replacement_body",
        "replacement_verified",
    ],
    "additionalProperties": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def load_rows(output: Path) -> list[dict[str, Any]]:
    manifest = output / "slate_manifest_eval_claw_147.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 147:
        raise SystemExit(f"expected 147 Claw rows, found {len(rows)} in {manifest}")
    task_ids = [str(row["task_id"]) for row in rows]
    oracle_names = [str(row["oracle"][0]["name"]) for row in rows]
    if len(set(task_ids)) != 147 or len(set(oracle_names)) != 147:
        raise SystemExit("Claw147 task ids and oracle skill names must both be unique")
    return rows


def result_path(work: Path, task_id: str) -> Path:
    return work / "results" / f"{task_id}.json"


def load_result(work: Path, task_id: str) -> dict[str, Any] | None:
    path = result_path(work, task_id)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def current_result_is_valid(work: Path, row: dict[str, Any]) -> bool:
    task_id = str(row["task_id"])
    skill_dir = Path(row["oracle"][0]["path"])
    skill_path = skill_dir / "SKILL.md"
    meta_path = skill_dir / "meta.json"
    result = load_result(work, task_id)
    if result is None or result.get("status") not in {"correct", "fixed"}:
        return False
    if not skill_path.is_file() or not meta_path.is_file():
        return False
    current_sha = sha256_file(skill_path)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        result.get("final_skill_sha256") == current_sha
        and meta.get("claude_audit_id") == AUDIT_ID
        and meta.get("claude_audited") is True
        and meta.get("claude_audit_skill_sha256") == current_sha
    )


def render_report(
    output: Path,
    work: Path,
    report: Path,
    rows: list[dict[str, Any]],
    quota_message: str,
) -> None:
    counts = {"correct": 0, "fixed": 0, "pending": 0, "failed": 0}
    pending_rows = []
    for row in rows:
        task_id = str(row["task_id"])
        name = str(row["oracle"][0]["name"])
        result = load_result(work, task_id)
        if current_result_is_valid(work, row):
            counts[str(result["status"])] += 1
            continue
        status = str((result or {}).get("status") or "pending")
        if status == "failed":
            counts["failed"] += 1
        else:
            counts["pending"] += 1
        issues = (result or {}).get("issues") or []
        error = str((result or {}).get("error") or "")
        reason = "; ".join(str(item) for item in issues[:3]) or error or "待 Claude 审核"
        pending_rows.append((task_id, name, status, reason.replace("|", "\\|")))

    reviewed = counts["correct"] + counts["fixed"]
    lines = [
        "# Claw Oracle Skill Fix",
        "",
        f"- canonical snapshot: `{output}`",
        f"- audit: Claude Code Opus, effort medium, one task per invocation",
        f"- progress: {reviewed}/147 reviewed; {counts['correct']} unchanged; "
        f"{counts['fixed']} rewritten; {counts['pending']} pending; {counts['failed']} failed",
        "- invariant: oracle name and YAML frontmatter are preserved byte-for-byte; only the body may change",
        f"- detailed state: `{work / 'results'}` and `{work / 'audit_events.jsonl'}`",
    ]
    if quota_message:
        lines.append(f"- Claude quota state: `{quota_message.replace('`', '')[:400]}`")
    lines.extend(
        [
            "",
            "## Pending Review Or Unresolved Problems",
            "",
            "Resolved rewrites are removed from this table; complete evidence remains in the JSON audit state.",
            "",
            "| task | oracle skill | state | issue |",
            "|---|---|---|---|",
        ]
    )
    if pending_rows:
        lines.extend(f"| `{task}` | `{name}` | `{status}` | {reason} |" for task, name, status, reason in pending_rows)
    else:
        lines.append("| - | - | complete | - |")
    atomic_write_text(report, "\n".join(lines) + "\n")


def prompt_for(row: dict[str, Any]) -> str:
    task_id = str(row["task_id"])
    skill_dir = Path(row["oracle"][0]["path"])
    skill_path = skill_dir / "SKILL.md"
    meta_path = skill_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source_paths = []
    for key in ("deep_audit_sources", "grounding_sources"):
        for source in meta.get(key) or []:
            path = str(source.get("path") or "")
            if path and path not in source_paths and Path(path).is_file():
                source_paths.append(path)
    source_lines = "\n".join(f"- {path}" for path in source_paths)
    return f"""Review exactly one Claw oracle skill for executable correctness.

Task id: {task_id}
Current skill: {skill_path}
Authoritative task directory: {builder.CLAW_TASK_ROOT / task_id}
Known grounding files:
{source_lines}

Read the current skill, task.yaml, grader.py including any peer grader it imports, every relevant
fixture, and the actual mock-service implementation used by the task. Treat service and fixture
behavior as authoritative for executable behavior, then the grader and intended task contract.
Ignore all existing 27B generation and audit verdicts.

Check every endpoint, method, payload, response field, action sequence, safety condition, decision
rule, dynamic date behavior, and final-answer requirement. The result must be a genuinely correct
procedure, not a grader exploit. Task-YAML tools are HTTP schemas exposed through OpenClaw HTTP
Tools; instructions must use exec/curl or a small script against the runtime-provided exact URL,
never a hardcoded localhost port or an invented direct function tool. Do not include any build-time
91xx port literal anywhere in the replacement, even as an explanation or example; use a named
angle-bracket URL copied from HTTP Tools.

Do not edit any file. If the body has any factual error, nonexistent operation, wrong value, unsafe
step, or materially missing instruction, return needs_rewrite and a complete replacement Markdown
body beginning after the closing YAML frontmatter delimiter. The replacement must preserve the
task deliverable, be concise but fully executable, avoid mentioning graders/evaluation/hidden tests,
and must not contain YAML frontmatter. Set replacement_verified true only after checking the new
body itself against all authoritative sources. If the current skill is already correct and adequate,
return correct, an empty replacement_body, and replacement_verified true. Cosmetic preferences alone
are not grounds for rewriting."""


def parse_claude_output(stdout: str) -> tuple[dict[str, Any], dict[str, Any]]:
    outer = json.loads(stdout)
    if not isinstance(outer, dict):
        raise ValueError("Claude output is not a JSON object")
    if outer.get("is_error"):
        raise RuntimeError(str(outer.get("result") or outer))
    payload = outer.get("structured_output")
    if not isinstance(payload, dict):
        raw = outer.get("result")
        payload = json.loads(raw) if isinstance(raw, str) else None
    if not isinstance(payload, dict):
        raise ValueError("Claude result lacks structured_output")
    return outer, payload


def invoke_claude(prompt: str, timeout: int) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        "claude",
        "-p",
        "--safe-mode",
        "--no-session-persistence",
        "--model",
        "opus",
        "--effort",
        "medium",
        "--output-format",
        "json",
        "--permission-mode",
        "dontAsk",
        "--allowedTools",
        "Read,Grep,Glob",
        "--json-schema",
        json.dumps(SCHEMA, separators=(",", ":")),
        prompt,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        env={**os.environ, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
    )
    combined = (completed.stdout + "\n" + completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(f"Claude rc={completed.returncode}: {combined[-4000:]}")
    return parse_claude_output(completed.stdout)


def validate_payload(payload: dict[str, Any]) -> None:
    verdict = payload.get("verdict")
    body = str(payload.get("replacement_body") or "")
    if payload.get("replacement_verified") is not True:
        raise ValueError("Claude did not verify its verdict/replacement")
    if verdict == "correct" and body.strip():
        raise ValueError("correct verdict must have an empty replacement_body")
    if verdict == "needs_rewrite":
        if len(body.strip()) < 300:
            raise ValueError("replacement body is too short")
        if len(body) > 20000:
            raise ValueError("replacement body is too long")
        if builder.FRONTMATTER_RE.match(body.lstrip()):
            raise ValueError("replacement body unexpectedly includes YAML frontmatter")
        if not payload.get("issues"):
            raise ValueError("needs_rewrite verdict must identify at least one issue")


def apply_result(row: dict[str, Any], payload: dict[str, Any], outer: dict[str, Any]) -> dict[str, Any]:
    task_id = str(row["task_id"])
    name = str(row["oracle"][0]["name"])
    skill_dir = Path(row["oracle"][0]["path"])
    skill_path = skill_dir / "SKILL.md"
    meta_path = skill_dir / "meta.json"
    original = skill_path.read_text(encoding="utf-8")
    match = RAW_FRONTMATTER_RE.match(original)
    if not match:
        raise ValueError(f"{task_id}: current skill lacks frontmatter")
    original_prefix = original[: match.end()]
    original_front, _ = builder.parse_frontmatter(original)
    if str(original_front.get("name") or "").strip() != name:
        raise ValueError(f"{task_id}: current oracle name mismatch")
    original_sha = sha256_bytes(original.encode("utf-8"))

    verdict = str(payload["verdict"])
    rewritten = verdict == "needs_rewrite"
    final_text = original
    if rewritten:
        body = str(payload["replacement_body"]).strip() + "\n"
        final_text = original_prefix + "\n" + body
        final_match = RAW_FRONTMATTER_RE.match(final_text)
        if not final_match or final_text[: final_match.end()] != original_prefix:
            raise ValueError(f"{task_id}: frontmatter was not preserved byte-for-byte")
        final_front, _ = builder.parse_frontmatter(final_text)
        if final_front != original_front:
            raise ValueError(f"{task_id}: parsed frontmatter changed")
        problems = builder.static_skill_problems(final_text, name)
        if problems:
            raise ValueError(f"{task_id}: replacement failed static checks: {problems}")

    final_sha = sha256_bytes(final_text.encode("utf-8"))
    if rewritten:
        atomic_write_text(skill_path, final_text)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.setdefault("pre_claude_skill_sha256", original_sha)
    meta.update(
        {
            "claude_audited": True,
            "claude_audit_id": AUDIT_ID,
            "claude_audit_model": "opus",
            "claude_audit_effort": "medium",
            "claude_audit_completed_at": utc_now(),
            "claude_audit_verdict": verdict,
            "claude_audit_confidence": payload.get("confidence"),
            "claude_audit_issues": payload.get("issues") or [],
            "claude_audit_evidence": payload.get("evidence") or [],
            "claude_audit_input_sha256": original_sha,
            "claude_audit_skill_sha256": final_sha,
            "claude_rewritten": rewritten,
            "skill_sha256": final_sha,
        }
    )
    atomic_write_json(meta_path, meta)
    return {
        "task_id": task_id,
        "oracle_name": name,
        "status": "fixed" if rewritten else "correct",
        "audit_id": AUDIT_ID,
        "model": "opus",
        "effort": "medium",
        "completed_at": utc_now(),
        "confidence": payload.get("confidence"),
        "issues": payload.get("issues") or [],
        "evidence": payload.get("evidence") or [],
        "original_skill_sha256": original_sha,
        "final_skill_sha256": final_sha,
        "claude_session_id": outer.get("session_id"),
        "duration_ms": outer.get("duration_ms"),
        "num_turns": outer.get("num_turns"),
        "total_cost_usd": outer.get("total_cost_usd"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--limit", type=int, default=0, help="0 means all pending tasks")
    args = parser.parse_args()

    output = args.output_root.resolve()
    work = args.work_root.resolve()
    report = args.report.resolve()
    work.mkdir(parents=True, exist_ok=True)
    rows = load_rows(output)
    pending = [row for row in rows if not current_result_is_valid(work, row)]
    if pending:
        (output / "COMPLETE").unlink(missing_ok=True)
    if args.limit > 0:
        pending = pending[: args.limit]

    write_lock = threading.Lock()
    quota_event = threading.Event()
    quota_message = {"value": ""}
    render_report(output, work, report, rows, quota_message["value"])
    print(f"[claude-audit] pending={len(pending)} workers={args.workers}", flush=True)

    def one(row: dict[str, Any]) -> str:
        task_id = str(row["task_id"])
        if quota_event.is_set():
            return "quota-skipped"
        last_error = ""
        for attempt in range(1, args.attempts + 1):
            if quota_event.is_set():
                return "quota-skipped"
            started = utc_now()
            try:
                retry_instruction = ""
                if last_error:
                    retry_instruction = (
                        "\n\nA prior attempt was rejected locally for this reason; repair it and do not "
                        f"repeat it:\n{last_error[-1200:]}"
                    )
                outer, payload = invoke_claude(
                    prompt_for(row) + retry_instruction, args.timeout
                )
                validate_payload(payload)
                if payload["verdict"] == "needs_rewrite":
                    pending_record = {
                        "task_id": task_id,
                        "oracle_name": row["oracle"][0]["name"],
                        "status": "rewrite_pending",
                        "issues": payload.get("issues") or [],
                        "started_at": started,
                    }
                    atomic_write_json(result_path(work, task_id), pending_record)
                    with write_lock:
                        render_report(output, work, report, rows, quota_message["value"])
                record = apply_result(row, payload, outer)
                atomic_write_json(result_path(work, task_id), record)
                append_jsonl(work / "audit_events.jsonl", record, write_lock)
                with write_lock:
                    render_report(output, work, report, rows, quota_message["value"])
                print(
                    f"[claude-audit] {task_id} status={record['status']} "
                    f"turns={record.get('num_turns')} duration_ms={record.get('duration_ms')}",
                    flush=True,
                )
                return str(record["status"])
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                lowered = last_error.lower()
                event = {
                    "task_id": task_id,
                    "status": "attempt_failed",
                    "attempt": attempt,
                    "started_at": started,
                    "completed_at": utc_now(),
                    "error": last_error[-4000:],
                }
                append_jsonl(work / "audit_events.jsonl", event, write_lock)
                if any(pattern in lowered for pattern in QUOTA_PATTERNS):
                    quota_message["value"] = last_error[-1000:]
                    quota_event.set()
                    break
                if attempt < args.attempts:
                    time.sleep(15 * attempt)
        failed = {
            "task_id": task_id,
            "oracle_name": row["oracle"][0]["name"],
            "status": "failed",
            "completed_at": utc_now(),
            "error": last_error[-4000:],
            "issues": [],
        }
        atomic_write_json(result_path(work, task_id), failed)
        with write_lock:
            render_report(output, work, report, rows, quota_message["value"])
        print(f"[claude-audit] {task_id} failed: {last_error[-500:]}", flush=True)
        return "failed"

    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(one, row): row for row in pending}
        for future in as_completed(futures):
            status = future.result()
            counts[status] = counts.get(status, 0) + 1

    render_report(output, work, report, rows, quota_message["value"])
    remaining = sum(not current_result_is_valid(work, row) for row in rows)
    final_counts = {"correct": 0, "fixed": 0, "remaining": 0}
    for row in rows:
        result = load_result(work, str(row["task_id"]))
        if current_result_is_valid(work, row):
            final_counts[str(result["status"])] += 1
        else:
            final_counts["remaining"] += 1
    summary = {
        "audit_id": AUDIT_ID,
        "completed_at": utc_now(),
        "run_counts": counts,
        "final_counts": final_counts,
        "remaining": remaining,
        "quota_exhausted": quota_event.is_set(),
        "quota_message": quota_message["value"],
    }
    atomic_write_json(work / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if remaining:
        raise SystemExit(3 if quota_event.is_set() else 2)


if __name__ == "__main__":
    main()
