#!/usr/bin/env python3
"""Collect successful trajectories from a trial plan into SFT-ready JSONL.

Reads a plan JSONL (from make_trial_plan.py) + the trajectory and incremental
files each trial wrote, then produces:
  - successful_trials.jsonl  — metadata for every successful trial (no msgs)
  - sft_messages.jsonl       — actual SFT training samples (messages + meta)
  - task_buckets.json        — per-task classification (no_skill_solvable / ...)
  - summary.md               — human-readable rollup

The interesting decisions live in three places:

  1. detect_skill_use(): how we tell whether a trial actually USED an injected
     skill, vs just had it sitting in the prompt unused. The primary signal is
     an assistant tool call that explicitly targets a mounted SKILL.md or
     README.md; broader path/name mentions are retained only for diagnostics.
     The strict signal is conservative because a false-positive would silently
     mis-bucket the trial.

  2. bucket_student_tasks(): per-task bucket assignment from {baseline-success,
     retrieval-success-without-skill, retrieval-success-with-skill, teacher-
     success}. Determines what bucket each task falls into and gets attached
     to every SFT record's metadata.

  3. dedupe_per_task(): keep at most --max-successes-per-task per
     (task, mode, used_skill), with a higher cap for use-skill/strict
     skill-use demos. Without this, a popular easy task with 8 successes
     contributes 8 nearly-identical trajectories and skews training. We keep
     the shortest ones because they're the cleanest demos.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    dump_json,
    experiment_collected_dir,
    is_bad_task,
    load_json,
    read_jsonl,
    repo_path,
)


# Regex evidence that an agent tool call targeted an injected skill document.
# Bare prose such as "I will read SKILL.md" is not a read.
SKILL_ENTRY_CAPTURE = re.compile(
    # The 7 mount namespaces used by retrieval_skill_inject.py. README.md is
    # included because a few skills advertise it as their entry document.
    r"/root/\.(?:claude|codex|agents|gemini|factory|goose|opencode)/skills?/"
    r"([A-Za-z0-9][A-Za-z0-9._-]*)/(?:SKILL|README)\.md\b",
    re.IGNORECASE,
)

# Captures any mounted skill path for a broader diagnostic signal. This can
# include prose and tool-result echoes, so it is never the primary read metric.
SKILL_NAME_CAPTURE = re.compile(
    r"/root/\.(?:claude|codex|agents|gemini|factory|goose|opencode)/skills?/"
    r"([A-Za-z0-9][A-Za-z0-9._-]*)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Plan / trajectory loading
# ---------------------------------------------------------------------------

def load_plan(plan_path: str | Path) -> list[dict[str, Any]]:
    path = repo_path(plan_path)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            line = raw_line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_trajectory(record: dict[str, Any]) -> dict[str, Any] | None:
    path = repo_path(record["trajectory_path"])
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_result_row(record: dict[str, Any]) -> dict[str, Any]:
    """Find this trial's row in incremental.jsonl by task_id.

    Returns {} if not found. (Earlier version fell back to rows[-1] which
    silently grabbed an UNRELATED task's result row when the right one was
    missing — a wrong-row contamination bug.)

    SWE rows use `instance_id` instead of `task_id`; we accept either.
    """
    rows = read_jsonl(record["incremental_path"])
    for row in rows:
        row_task_id = row.get("task_id") or row.get("instance_id")
        if row_task_id == record["task_id"]:
            return row
    return {}


# ---------------------------------------------------------------------------
# Skill-use detection
# ---------------------------------------------------------------------------

def _stringify_tool_message_content(content: Any) -> str:
    """tool message content can be a string or a list of {type,text} blocks
    (multimodal-style). Normalize to one concatenated string so regex sees it."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content or "")


def extract_searchable_texts_with_source(
    messages: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Like extract_searchable_texts but tags each text with its source:
    assistant prose, assistant tool-call arguments, or tool results."""
    texts: list[tuple[str, str]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            content = message.get("content")
            if isinstance(content, str) and content:
                texts.append(("assistant_text", content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        texts.append(("assistant_text", str(block["text"])))
            for tool_call in message.get("tool_calls") or []:
                # tool_calls have a function dict with name+arguments. We
                # serialize the whole tool_call so path arguments are searchable.
                texts.append(("agent_tool_call", json.dumps(tool_call, ensure_ascii=False, default=str)))
        elif role == "tool":
            texts.append(("tool", _stringify_tool_message_content(message.get("content"))))
    return texts


def extract_searchable_texts(messages: list[dict[str, Any]]) -> list[str]:
    """Collect all text the agent emitted that COULD reference a skill.

    Includes:
      - assistant text content (where agent mentions skill name in prose)
      - assistant tool_call arguments (where agent does `read /root/.claude/skills/...`)
      - tool result content (where the read returned skill file content)
    """
    return [text for _source, text in extract_searchable_texts_with_source(messages)]


def detect_skill_use(
    messages: list[dict[str, Any]],
    injected_skill_names: list[str] | None = None,
) -> dict[str, Any]:
    """Decide whether the agent actually USED an injected skill.

    Primary signal:
      (a) an assistant tool call explicitly targets a mounted SKILL.md or
          README.md -> strong: the agent attempted to read a skill document

    Auxiliary signal only:
      (b) any injected skill name appeared in agent text/tool_calls
          (case-insensitive whole-word match)
          → weaker: agent referred to the skill by name but may not have opened it.
            We preserve this evidence for debugging, but do NOT count it as
            used_skill because the project standard is "agent read the skill".

    Returns evidence snippets (capped at 5 each) for debugging mis-bucketing.
    """
    path_evidence: list[str] = []
    name_evidence: list[str] = []
    read_names: set[str] = set()
    read_names_agent: set[str] = set()

    # Pre-compile per-skill name regex (whole-word, case-insensitive).
    name_patterns: list[tuple[str, re.Pattern]] = []
    for skill_name in injected_skill_names or []:
        if not skill_name:
            continue
        # \b doesn't work well with hyphens in skill names, so we match
        # name surrounded by non-alnum or string boundary.
        escaped = re.escape(skill_name)
        name_patterns.append(
            (skill_name, re.compile(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", re.IGNORECASE))
        )

    for source, text in extract_searchable_texts_with_source(messages):
        compact = text.replace("\n", " ")[:800]
        strict_agent_names: set[str] = set()
        if source == "agent_tool_call":
            strict_agent_names = {match.group(1) for match in SKILL_ENTRY_CAPTURE.finditer(text)}
            if strict_agent_names:
                path_evidence.append(compact)
        for match in SKILL_NAME_CAPTURE.finditer(text):
            read_names.add(match.group(1))
        read_names_agent.update(strict_agent_names)
        for skill_name, name_pattern in name_patterns:
            if name_pattern.search(text):
                name_evidence.append(f"[{skill_name}] {compact}")
                break

    return {
        "used_skill": bool(path_evidence),
        "used_skill_via_path": bool(path_evidence),
        "used_skill_via_name": bool(name_evidence),
        "evidence_path": path_evidence[:5],
        "evidence_name": name_evidence[:5],
        # Per-skill attribution (additive). `read_skill_names` is the broad
        # diagnostic over all searchable texts; `_agent` restricts to assistant
        # tool-call arguments that explicitly target SKILL.md/README.md.
        "read_skill_names": sorted(read_names),
        "read_skill_names_agent": sorted(read_names_agent),
    }


# ---------------------------------------------------------------------------
# Trajectory token estimate
# ---------------------------------------------------------------------------

def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token count: chars / 4. Accurate enough for a length filter
    (we'd over-/under-count by 30% but that's fine for "is it under 40k?")."""
    serialized = json.dumps(messages, ensure_ascii=False, default=str)
    return max(1, len(serialized) // 4)


# ---------------------------------------------------------------------------
# Per-trial payload (one element of trials list)
# ---------------------------------------------------------------------------

def _injected_skill_names_from_record(record: dict[str, Any]) -> list[str]:
    """Look up which skills were retrieved for this task. Returns empty list
    if the retrieval jsonl is missing or task not present."""
    retrieval_jsonl = record.get("retrieval_jsonl")
    if not retrieval_jsonl:
        return []
    path = repo_path(retrieval_jsonl)
    if not path.exists():
        return []
    target = str(record["task_id"])
    top_n = int(record.get("retrieval_top_n", 10))
    with path.open(encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("task_id", "")) != target:
                continue
            top10 = row.get("reranked_top10") or []
            return [str(entry.get("skill_name", "")) for entry in top10[:top_n] if entry.get("skill_name")]
    return []


# ---------------------------------------------------------------------------
# Meta-talk detection — risk that the implicit instruction leaks into
# assistant text ("as you instructed I won't read the skills"). If detected,
# the trajectory is excluded from SFT export by default; user can override
# with --include-meta-talk.
# ---------------------------------------------------------------------------

# Compiled once. Patterns target English meta-talk styles 9B tends to produce.
META_TALK_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bas (you|the user|the (instructions?|note)) (asked|requested|instructed|noted|said|wanted|suggested|directed|specified)\b",
        r"\bsince (you|the user|the (instructions?|note)) (told|asked|wanted|said|noted|requested)\b",
        r"\bper (your|the user'?s|the) (instruction|request|note|guidance)\b",
        r"\b(as|per) (instructed|noted|requested|directed)\b",
        r"\b(without|not) using (the )?(skills?|skill files?)\b",
        r"\b(don'?t|won'?t|will not|shall not) (open|read|use) (the )?(skills?|skill files?)\b",
        r"\bsolving (this )?(without|using only) (skills?|the skills?|my own knowledge)\b",
        r"\bthe (note|instructions?) (says?|tells? me|asks me to|requires me)\b",
        # Reflection-aware leaks (Phase 2)
        r"\b(previous|last|prior|the earlier) attempt (failed|didn'?t|did not)\b",
        r"\b(learning from|based on) (the )?(previous|last|prior) (attempt|failure|mistake)\b",
        r"\btry(ing)? a different approach (this time|now)\b",
    ]
]


def detect_meta_talk(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Scan assistant text for phrases that suggest the agent acknowledged
    the implicit instruction or reflection context. Such phrases would leak
    the artificial framing into SFT data even after we strip the system-prompt
    suffix.

    Returns dict with:
      meta_talk_detected: bool
      evidence: list of up to 3 example snippets (for debugging)
    """
    evidence: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                str(b.get("text", "")) for b in content
                if isinstance(b, dict)
            )
        if not text:
            continue
        for pattern in META_TALK_PATTERNS:
            match = pattern.search(text)
            if match:
                # Return ~80 char window around the match for debugging
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 50)
                evidence.append(text[start:end].replace("\n", " "))
                break  # one match per assistant turn suffices
        if len(evidence) >= 3:
            break
    return {"meta_talk_detected": bool(evidence), "evidence": evidence}


# ---------------------------------------------------------------------------
# System-prompt cleanup — strip the implicit/reflection text we appended
# at runtime, so SFT data has the same prompt distribution as deployment.
# ---------------------------------------------------------------------------

def strip_runtime_suffixes(
    messages: list[dict[str, Any]],
    implicit_text: str,
    reflection_text: str,
) -> list[dict[str, Any]]:
    """Return a copy of messages with implicit_text + reflection_text removed
    from messages[0]['content'] (the system message).

    Critical: we strip by exact byte-sequence match. The runner appends with
    `sys_prompt + "\\n\\n" + implicit_text` (and reflection_text starts with
    "\\n\\n" already) — so the suffix in the saved trajectory ends with
    `implicit_text` then optionally `reflection_text`. We strip both, leaving
    the original system prompt intact. If the suffix isn't found exactly
    (shouldn't happen but defensive), we return messages unchanged so SFT
    isn't silently corrupted by partial strips.
    """
    if not messages or not (implicit_text or reflection_text):
        return messages

    out = [dict(m) for m in messages]
    sys_msg = out[0]
    if sys_msg.get("role") != "system":
        return out  # unexpected layout, leave alone

    content = sys_msg.get("content")
    if not isinstance(content, str):
        # Multi-block system prompt is rare in practice — bail rather than
        # guess which block has the suffix.
        return out

    cleaned = content
    # Reflection text comes AFTER implicit (per implicit_instruction.py
    # apply_implicit_and_reflection ordering). Strip in reverse order.
    if reflection_text and cleaned.endswith(reflection_text):
        cleaned = cleaned[: -len(reflection_text)]
    if implicit_text:
        # implicit_text was prefixed with "\n\n" by the runner; strip that
        # together with the implicit text itself.
        suffix_with_separator = "\n\n" + implicit_text
        if cleaned.endswith(suffix_with_separator):
            cleaned = cleaned[: -len(suffix_with_separator)]
        elif cleaned.endswith(implicit_text):
            # Defensive: if for some reason the separator wasn't there.
            cleaned = cleaned[: -len(implicit_text)]

    sys_msg["content"] = cleaned
    return out


def build_trial_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    """Load + analyze one trial. None if trajectory file missing (trial
    didn't complete or was killed before writing)."""
    trajectory = load_trajectory(record)
    if trajectory is None:
        return None
    result_row = load_result_row(record)
    messages = trajectory.get("messages") or []
    resolved = bool(result_row.get("resolved", trajectory.get("resolved", False)))
    injected_names = _injected_skill_names_from_record(record)
    skill_use = detect_skill_use(messages, injected_names)
    meta_talk = detect_meta_talk(messages)
    # implicit/reflection text the runner appended — saved by all 3 runners
    # in the trajectory file. We need them at SFT-export time to strip.
    implicit_text = str(trajectory.get("implicit_text", "") or "")
    reflection_text = str(trajectory.get("reflection_text", "") or "")
    return {
        "trial_id": record["trial_id"],
        "run_id": record["run_id"],
        "bench": record["bench"],
        "task_id": record["task_id"],
        "split": record["split"],
        "mode": record["mode"],
        "model_role": record["model_role"],
        "model": record["model"],
        "arm": record["arm"],
        "trial_index": record["trial_index"],
        "resolved": resolved,
        "score": result_row.get("score", trajectory.get("score")),
        "error": result_row.get("error", ""),
        "turns": result_row.get("turns"),
        "time_sec": result_row.get("time_sec") or result_row.get("wall_sec"),
        "retrieval_skills_injected": result_row.get(
            "retrieval_skills_injected",
            trajectory.get("retrieval_skills_injected", 0),
        ),
        "injected_skill_names": injected_names,
        "used_skill": skill_use["used_skill"],
        "used_skill_via_path": skill_use["used_skill_via_path"],
        "used_skill_via_name": skill_use["used_skill_via_name"],
        "used_skill_evidence": (skill_use["evidence_path"] + skill_use["evidence_name"])[:5],
        "meta_talk_detected": meta_talk["meta_talk_detected"],
        "meta_talk_evidence": meta_talk["evidence"],
        "implicit_mode": str(trajectory.get("implicit_mode", "") or record.get("implicit_mode", "")),
        "implicit_text": implicit_text,
        "reflection_text": reflection_text,
        "had_reflection_context": bool(trajectory.get("reflection_context", "")),
        "direct_sft_candidate": record.get("direct_sft_candidate", False),
        "classification_candidate": record.get("classification_candidate", False),
        "trajectory_path": record["trajectory_path"],
        "messages": messages,  # raw, not stripped — strip happens at SFT export
        "estimated_tokens": estimate_tokens(messages),
    }


# ---------------------------------------------------------------------------
# Per-task bucketing
# ---------------------------------------------------------------------------

# 2026-04-26 design: explicit branches replace implicit "did the agent happen
# to use skill" classification. The mode itself indicates which branch.
USE_SKILL_BRANCH_MODES = {"student_use_skill", "student_use_skill_reflection"}
NO_SKILL_BRANCH_MODES = {"student_no_skill", "student_no_skill_reflection"}
TEACHER_MODES = {"teacher_retrieval", "teacher_retrieval_reflection"}


def bucket_student_tasks(trials: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-task bucket assignment.

    Two-branch design (Phase 1 + reflection):
      - student_use_skill / _reflection: prompt-nudged toward using skills
      - student_no_skill  / _reflection: prompt-nudged AWAY from using skills

    The bucket reflects which branch(es) found a successful path, NOT the
    post-hoc used_skill detection (that's still computed for diagnostics
    but doesn't drive bucketing now that branches are explicit):

      both_solvable        = both branches succeeded
      skill_helpful        = only use_skill branch succeeded
      no_skill_solvable    = only no_skill branch succeeded
      teacher_only         = no student succeeded but teacher did
      unresolved           = nobody succeeded
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        grouped[(trial["bench"], trial["task_id"])].append(trial)

    buckets: dict[str, dict[str, Any]] = {}
    for (bench, task_id), task_trials in grouped.items():
        # Per-branch success (any trial in that branch resolved).
        use_skill_success = any(
            t["resolved"] and t["mode"] in USE_SKILL_BRANCH_MODES
            for t in task_trials
        )
        no_skill_success = any(
            t["resolved"] and t["mode"] in NO_SKILL_BRANCH_MODES
            for t in task_trials
        )
        teacher_success = any(
            t["resolved"] and t["mode"] in TEACHER_MODES
            for t in task_trials
        )
        # student_baseline (no retrieval prompt) — diagnostic only, not bucketed.
        baseline_success = any(
            t["resolved"] and t["mode"] == "student_baseline"
            for t in task_trials
        )

        if use_skill_success and no_skill_success:
            bucket = "both_solvable"
        elif use_skill_success:
            bucket = "skill_helpful"
        elif no_skill_success:
            bucket = "no_skill_solvable"
        elif teacher_success:
            bucket = "teacher_only"
        else:
            bucket = "unresolved"

        # Diagnostic: how many trials in each branch + how many actually
        # touched skill files (post-hoc detection).
        use_skill_trials = [t for t in task_trials if t["mode"] in USE_SKILL_BRANCH_MODES]
        no_skill_trials = [t for t in task_trials if t["mode"] in NO_SKILL_BRANCH_MODES]
        buckets[f"{bench}/{task_id}"] = {
            "bench": bench,
            "task_id": task_id,
            "bucket": bucket,
            "use_skill_success": use_skill_success,
            "no_skill_success": no_skill_success,
            "teacher_success": teacher_success,
            "baseline_success": baseline_success,
            "n_trials": len(task_trials),
            "n_success": sum(1 for t in task_trials if t["resolved"]),
            # Sanity diagnostic: in use_skill branch, did agents actually read
            # skill files? Low ratio = nudge ineffective, agent ignored implicit.
            "use_skill_branch_actually_used_skill": (
                f"{sum(1 for t in use_skill_trials if t['used_skill'])}"
                f"/{len(use_skill_trials)}"
            ),
            "no_skill_branch_unexpectedly_used_skill": (
                f"{sum(1 for t in no_skill_trials if t['used_skill'])}"
                f"/{len(no_skill_trials)}"
            ),
        }
    return buckets


# ---------------------------------------------------------------------------
# Per-task de-dup (keep best K successes per (task, mode, used_skill))
# ---------------------------------------------------------------------------

def dedupe_per_task(
    trials: list[dict[str, Any]],
    max_per_group: int,
    max_use_skill_group: int,
) -> list[dict[str, Any]]:
    """Cap successful trajectories per (task, mode, used_skill).

    Why: a popular easy task can have 8 trials succeed, contributing 8
    nearly-identical trajectories that skew training. We keep the SHORTEST
    K because they tend to be the cleanest demos (less faff, less noise
    from agent recovery loops).

    `max_per_group <= 0` disables dedup (pass everything through). The
    use-skill cap applies to `student_use_skill*` modes and to any trajectory
    with strict `used_skill=True`, because both are useful for increasing the
    skill-conditioned slice. If max_use_skill_group is smaller than the base
    cap, the base cap wins.

    Note: only successful + direct_sft_candidate trials are subject to
    the cap. Failed/baseline trials don't enter SFT data anyway, so we
    don't filter them here (keeps stats accurate downstream).
    """
    if max_per_group <= 0:
        return trials

    # Bucket by (bench, task_id, mode, used_skill). used_skill is part of
    # the key because skill-used and skill-unused are different DEMOs even
    # for the same task — we want at least one of each if both exist.
    groups: dict[tuple[str, str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    keep: list[dict[str, Any]] = []
    for trial in trials:
        if not trial["resolved"] or not trial.get("direct_sft_candidate"):
            keep.append(trial)
            continue
        key = (trial["bench"], trial["task_id"], trial["mode"], trial["used_skill"])
        groups[key].append(trial)

    for group_trials in groups.values():
        # Sort: shortest token count first, then most turns ascending as
        # tie-breaker (shorter = cleaner demo).
        group_trials.sort(key=lambda t: (t.get("estimated_tokens", 0), t.get("turns") or 0))
        first = group_trials[0]
        group_cap = max_per_group
        if first["mode"] in USE_SKILL_BRANCH_MODES or first["used_skill"]:
            group_cap = max(max_per_group, max_use_skill_group)
        keep.extend(group_trials[:group_cap])
    return keep


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_summary(
    path: Path,
    trials: list[dict[str, Any]],
    sft_records: list[dict[str, Any]],
    buckets: dict[str, dict[str, Any]],
    max_successes_per_task: int,
    max_successes_per_use_skill_task: int,
) -> None:
    """Human-readable rollup. Important for sanity-checking before training."""
    bucket_counts: dict[str, int] = defaultdict(int)
    for bucket in buckets.values():
        bucket_counts[bucket["bucket"]] += 1

    # Per-bench × per-mode success breakdown (the most useful at-a-glance table).
    per_bench_mode: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"total": 0, "resolved": 0, "used_skill": 0, "missing_traj": 0}
    )
    for trial in trials:
        cell = per_bench_mode[(trial["bench"], trial["mode"])]
        cell["total"] += 1
        if trial["resolved"]:
            cell["resolved"] += 1
        if trial.get("used_skill"):
            cell["used_skill"] += 1

    # SFT records: per-bench length distribution.
    per_bench_lengths: dict[str, list[int]] = defaultdict(list)
    for sft in sft_records:
        meta = sft.get("metadata") or {}
        per_bench_lengths[meta.get("bench", "?")].append(int(meta.get("estimated_tokens") or 0))

    lines: list[str] = [
        "# SFT Collection Summary",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"- loaded_trials: {len(trials)}",
        f"- successful_trials: {sum(1 for t in trials if t['resolved'])}",
        f"- sft_records (after dedup + length filter): {len(sft_records)}",
        "",
        "## Task buckets (per task)",
        "",
    ]
    for bucket_name in ("no_skill_solvable", "skill_helpful", "both_solvable", "teacher_only", "unresolved"):
        lines.append(f"- {bucket_name}: {bucket_counts.get(bucket_name, 0)}")
    lines += [
        "",
        "## Per (bench, mode) — trial counts",
        "",
        "| bench | mode | total | resolved | rate | used_skill | used_skill_rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (bench, mode) in sorted(per_bench_mode):
        cell = per_bench_mode[(bench, mode)]
        rate = (cell["resolved"] / cell["total"] * 100.0) if cell["total"] else 0.0
        skill_rate = (cell["used_skill"] / cell["total"] * 100.0) if cell["total"] else 0.0
        lines.append(
            f"| {bench} | {mode} | {cell['total']} | {cell['resolved']} | "
            f"{rate:.1f}% | {cell['used_skill']} | {skill_rate:.1f}% |"
        )
    lines += [
        "",
        "## SFT records — token length per bench",
        "",
        "| bench | n | min | median | p90 | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for bench in sorted(per_bench_lengths):
        lengths = sorted(per_bench_lengths[bench])
        n = len(lengths)
        if n == 0:
            continue
        median = lengths[n // 2]
        p90 = lengths[min(n - 1, int(n * 0.9))]
        lines.append(f"| {bench} | {n} | {lengths[0]} | {median} | {p90} | {lengths[-1]} |")
    lines += [
        "",
        "## SFT policy",
        "",
        "- `direct_sft_candidate=True` only for student_retrieval and teacher_retrieval.",
        "- student_baseline successes inform task bucketing but DO NOT enter SFT messages",
        "  (different prompt distribution from deployment).",
        f"- Per (task, mode, used_skill), at most {max_successes_per_task} SHORTEST trajectories kept by default.",
        f"- Use-skill branch or strict used_skill=True groups keep up to {max_successes_per_use_skill_task} trajectories.",
        "- Trajectories with estimated_tokens > config.loss_policy.max_trajectory_tokens are dropped",
        "  unless --include-overlong is passed.",
        "- Loss policy at training time: mask system + user + tool messages; train only assistant.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", required=True, help="Plan JSONL from make_trial_plan.py")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-dir", default="",
                        help="Override output dir; default experiments/<plan_stem>/collected/")
    parser.add_argument("--include-overlong", action="store_true",
                        help="Don't drop trajectories exceeding max_trajectory_tokens")
    parser.add_argument("--max-successes-per-task", type=int, default=2,
                        help="Cap successful SFT trajectories per (task, mode, used_skill); "
                             "0 disables. Default 2 to preserve ~one-of-each-style demos.")
    parser.add_argument("--max-successes-per-use-skill-task", type=int, default=4,
                        help="Higher cap for student_use_skill* branch groups and strict "
                             "used_skill=True groups. Default 4 to increase skill-use "
                             "coverage while keeping no-skill demos capped at 2.")
    parser.add_argument("--include-meta-talk", action="store_true",
                        help="Don't drop trajectories with meta-talk leakage. "
                             "Default off — meta-talk = agent acknowledged the implicit "
                             "instruction or reflection in its own text, contaminating "
                             "SFT data even after we strip the system-prompt suffix.")
    parser.add_argument("--include-reflection-when-phase1-succeeded", action="store_true",
                        help="Include reflection trial successes even if Phase 1 already "
                             "succeeded for that task. Default off — Phase 1 successes are "
                             "preferred (no reflection contamination needed).")
    args = parser.parse_args()

    config = load_json(args.config)
    max_tokens = int(config["loss_policy"]["max_trajectory_tokens"])

    plan_records = load_plan(args.plan)
    trials = [
        payload
        for payload in (build_trial_payload(record) for record in plan_records)
        if payload is not None
    ]
    print(f"loaded {len(trials)} / {len(plan_records)} trial trajectories")

    buckets = bucket_student_tasks(trials)

    # Build the set of (bench, task_id) where Phase 1 (non-reflection trials)
    # already produced a success. By default we drop reflection successes for
    # those tasks since Phase 1 wins are cleaner (no failure-summary baggage
    # in the system prompt that we'd have to strip). Override with
    # --include-reflection-when-phase1-succeeded.
    phase1_success_tasks: set[tuple[str, str]] = set()
    for trial in trials:
        if trial["resolved"] and "_reflection" not in trial["mode"]:
            phase1_success_tasks.add((trial["bench"], trial["task_id"]))

    # Filtering pipeline (in order):
    #   1. resolved + direct_sft_candidate
    #   2. token length under max_trajectory_tokens (else dropped unless --include-overlong)
    #   3. Phase 2 reflection successes filtered out if Phase 1 succeeded for same task
    #   4. meta-talk free (else dropped unless --include-meta-talk)
    #   5. dedup per (task, mode, used_skill) keeping shortest K
    eligible: list[dict[str, Any]] = []
    dropped_overlong = 0
    dropped_meta_talk = 0
    dropped_reflection_redundant = 0
    dropped_bad_task = 0
    for trial in trials:
        if is_bad_task(trial["bench"], trial["task_id"]):
            dropped_bad_task += 1
            continue
        if not trial["resolved"] or not trial["direct_sft_candidate"]:
            continue
        if trial["estimated_tokens"] > max_tokens and not args.include_overlong:
            dropped_overlong += 1
            continue
        if (
            "_reflection" in trial["mode"]
            and (trial["bench"], trial["task_id"]) in phase1_success_tasks
            and not args.include_reflection_when_phase1_succeeded
        ):
            dropped_reflection_redundant += 1
            continue
        if trial.get("meta_talk_detected") and not args.include_meta_talk:
            dropped_meta_talk += 1
            continue
        eligible.append(trial)
    eligible = dedupe_per_task(
        eligible,
        max_per_group=args.max_successes_per_task,
        max_use_skill_group=args.max_successes_per_use_skill_task,
    )
    if dropped_overlong:
        print(f"dropped {dropped_overlong} over-long trajectories (>{max_tokens} estimated tokens)")
    if dropped_reflection_redundant:
        print(f"dropped {dropped_reflection_redundant} reflection successes (Phase 1 already succeeded)")
    if dropped_meta_talk:
        print(f"dropped {dropped_meta_talk} trajectories with meta-talk leakage")
    if dropped_bad_task:
        print(f"dropped {dropped_bad_task} trajectories from known-broken docker tasks")
    print(f"eligible SFT trials after filter+dedup: {len(eligible)}")

    sft_records: list[dict[str, Any]] = []
    for trial in eligible:
        bucket_key = f"{trial['bench']}/{trial['task_id']}"
        # Strip the implicit/reflection suffixes from messages[0] (system).
        # This is the critical step that makes the SFT prompt match deployment
        # distribution: at runtime, the agent saw extra instructions; at SFT
        # train time, the model sees only the original retrieval prompt.
        cleaned_messages = strip_runtime_suffixes(
            trial["messages"],
            implicit_text=trial.get("implicit_text", ""),
            reflection_text=trial.get("reflection_text", ""),
        )
        # Strip large fields out of metadata; messages get their own key,
        # and *_evidence are debugging info that shouldn't bloat training.
        metadata = {
            key: value
            for key, value in trial.items()
            if key not in {"messages", "used_skill_evidence", "meta_talk_evidence"}
        }
        metadata["task_bucket"] = buckets.get(bucket_key, {}).get("bucket", "unknown")
        metadata["loss_policy"] = config["loss_policy"]
        sft_records.append({
            "messages": cleaned_messages,
            "metadata": metadata,
        })

    # Output dir defaults to experiments/<plan_filename_stem>/collected/.
    if args.out_dir:
        output_dir = repo_path(args.out_dir)
    else:
        plan_stem = Path(args.plan).stem
        output_dir = experiment_collected_dir(plan_stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    # successful_trials.jsonl: lightweight metadata for analysis (no messages).
    trials_without_messages = [
        {key: value for key, value in trial.items() if key != "messages"}
        for trial in trials
        if trial["resolved"] and not is_bad_task(trial["bench"], trial["task_id"])
    ]
    write_jsonl(output_dir / "successful_trials.jsonl", trials_without_messages)
    write_jsonl(output_dir / "sft_messages.jsonl", sft_records)
    dump_json(output_dir / "task_buckets.json", buckets)
    write_summary(
        output_dir / "summary.md",
        trials,
        sft_records,
        buckets,
        max_successes_per_task=args.max_successes_per_task,
        max_successes_per_use_skill_task=args.max_successes_per_use_skill_task,
    )

    try:
        display_output_dir = output_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        display_output_dir = output_dir
    print(f"output: {display_output_dir}")
    print(f"  - sft_messages.jsonl: {len(sft_records)} records")
    print(f"  - successful_trials.jsonl: {len(trials_without_messages)} trials")
    print(f"  - task_buckets.json: {len(buckets)} tasks")


if __name__ == "__main__":
    main()
