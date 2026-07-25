#!/usr/bin/env python3
"""Build task-specific oracle-skill experiments for the RL train/eval universe.

The pipeline is intentionally additive:
  1. run empty-skill baseline rollouts with existing unified runners;
  2. synthesize one concise SKILL.md per (bench, task_id) from those rollouts;
  3. evaluate with a retrieval jsonl that exposes only that one task skill;
  4. optionally revise skills for tasks that still fail once, then freeze.

No core runner behavior is changed. The generated oracle skill library is just
a standard skill directory tree plus standard retrieval jsonl files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLAW_TASKS_DIR = PROJECT_ROOT / "datasets/claw-eval/tasks"
SFT_COLLECTION_DIR = PROJECT_ROOT / "GeneralAgent" / "sft_data_collection"
if str(SFT_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_COLLECTION_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import display_path, filter_known_bad_tasks, repo_path, safe_slug  # type: ignore  # noqa: E402
from make_trial_plan import make_record, write_jsonl  # type: ignore  # noqa: E402
from ops.workflows.rl_eval.make_full_parquet_eval_plan import load_task_universe  # noqa: E402


BENCHES = ("claw", "seta_synth", "swe_lite", "sb_ns", "tb2")
DEFAULT_TRAIN_PARQUET = "datasets/rl/parquet_4bench_base_20260523/train.parquet"
DEFAULT_EVAL_PARQUET = "datasets/rl/parquet_4bench_base_20260523/eval.parquet"
SKILL_CONTAMINATION_MARKERS = (
    "<think>",
    "</think>",
    "Thinking Process:",
    "Self-Correction",
    "Analyze the Request",
    "Let's write the skill",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path_value: str | Path) -> list[dict[str, Any]]:
    path = repo_path(path_value)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path_value: str | Path, payload: Any) -> None:
    path = repo_path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def task_key(bench: str, task_id: str) -> str:
    return f"{bench}/{task_id}"


def skill_name_for(bench: str, task_id: str) -> str:
    return safe_slug(f"oracle-{bench}-{task_id}", max_len=90)


def skill_dir_for(skills_root: Path, bench: str, task_id: str) -> Path:
    return skills_root / bench / safe_slug(task_id, max_len=90)


def apply_task_limit(tasks: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0 or limit >= len(tasks):
        return tasks
    per_bench: dict[str, list[dict[str, str]]] = defaultdict(list)
    for task in tasks:
        per_bench[task["bench"]].append(task)
    selected: list[dict[str, str]] = []
    while len(selected) < limit and any(per_bench.values()):
        for bench in BENCHES:
            if per_bench[bench] and len(selected) < limit:
                selected.append(per_bench[bench].pop(0))
    return selected


def load_task_list(path_value: str | Path) -> dict[tuple[str, str], int | None]:
    """Parse `bench\\ttask_id[\\trepeats]` rows into task repeat overrides."""
    tasks: dict[tuple[str, str], int | None] = {}
    for line_number, line in enumerate(
        repo_path(path_value).read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("\t")]
        if len(parts) not in {2, 3} or not all(parts):
            raise ValueError(
                f"invalid task-list row {line_number}: expected "
                "bench<TAB>task_id[<TAB>repeats]"
            )
        bench, task_id = parts[:2]
        repeats = None
        if len(parts) == 3:
            try:
                repeats = int(parts[2])
            except ValueError as exc:
                raise ValueError(
                    f"invalid task-list repeats on row {line_number}: {parts[2]!r}"
                ) from exc
            if repeats <= 0:
                raise ValueError(
                    f"task-list repeats must be positive on row {line_number}: {repeats}"
                )
        key = (bench, task_id)
        if key in tasks:
            raise ValueError(f"duplicate task-list row {line_number}: {key}")
        tasks[key] = repeats
    return tasks


def load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks = load_task_universe(repo_path(args.train_parquet), repo_path(args.eval_parquet))
    allowed: set[tuple[str, str]] | None = None
    task_repeat_overrides: dict[tuple[str, str], int | None] = {}
    if getattr(args, "task_list", ""):
        task_repeat_overrides = load_task_list(args.task_list)
        allowed = set(task_repeat_overrides)
        if not allowed:
            raise SystemExit(f"--task-list {args.task_list} parsed to 0 tasks")
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for task in tasks:
        bench = task["bench"]
        task_id = str(task["task_id"])
        if args.bench and bench not in args.bench:
            continue
        if allowed is not None and (bench, task_id) not in allowed:
            continue
        if task_id not in filter_known_bad_tasks(bench, [task_id]):
            continue
        key = (bench, task_id)
        if key in seen:
            continue
        seen.add(key)
        selected_task = dict(task)
        override = task_repeat_overrides.get(key)
        if override is not None:
            selected_task["_eval_trials"] = override
        deduped.append(selected_task)
    if allowed is not None:
        # The historical RL parquets cover only the earlier Claw task subset.
        # An explicit task list may select newer filesystem-backed Claw tasks;
        # these use the same unified Claw runner and do not need a parquet row.
        for bench, task_id in sorted(allowed - seen):
            if bench != "claw":
                continue
            if task_id not in filter_known_bad_tasks(bench, [task_id]):
                continue
            if not (CLAW_TASKS_DIR / task_id / "task.yaml").is_file():
                continue
            selected_task: dict[str, Any] = {
                "bench": bench,
                "task_id": task_id,
                "split": "task_list",
            }
            override = task_repeat_overrides.get((bench, task_id))
            if override is not None:
                selected_task["_eval_trials"] = override
            deduped.append(selected_task)
            seen.add((bench, task_id))
        missing = allowed - seen
        if missing:
            print(f"[make-plan] warning: {len(missing)} task-list entries not in universe: "
                  f"{sorted(missing)[:5]}{' ...' if len(missing) > 5 else ''}")
    return apply_task_limit(deduped, int(args.task_limit or 0))


def load_prompt_index(train_parquet: str | Path, eval_parquet: str | Path) -> dict[str, dict[str, str]]:
    import pandas as pd

    out: dict[str, dict[str, str]] = {}
    for split_name, path_value in (("rl_train", train_parquet), ("rl_eval", eval_parquet)):
        df = pd.read_parquet(repo_path(path_value))
        for row in df.itertuples(index=False):
            extra = row.extra_info
            if isinstance(extra, str):
                extra = json.loads(extra)
            bench = str(extra.get("bench") or "")
            task_id = str(extra.get("task_id") or "")
            if not bench or not task_id:
                continue
            key = task_key(bench, task_id)
            prompt_text = extract_task_prompt(row.prompt)
            if key not in out:
                out[key] = {"bench": bench, "task_id": task_id, "split": split_name, "prompt": prompt_text}
    return out


def extract_task_prompt(prompt: Any) -> str:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, str):
        try:
            prompt = json.loads(prompt)
        except Exception:
            return prompt[:6000]
    if isinstance(prompt, list):
        for item in reversed(prompt):
            if isinstance(item, dict) and item.get("role") == "user":
                return str(item.get("content") or "")[:6000]
        for item in reversed(prompt):
            if isinstance(item, dict) and item.get("content"):
                return str(item.get("content") or "")[:6000]
    return str(prompt)[:6000]


def build_plan_records(
    *,
    tasks: list[dict[str, Any]],
    run_id: str,
    date: str,
    model: str,
    mode: str,
    arm: str,
    trials: int,
    max_turns: int,
    max_time: int,
    docker_host: str,
    api_base: str,
    retrieval_jsonl_by_bench: dict[str, str] | None = None,
    retrieval_top_n: int = 1,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    retrieval_coverage: dict[str, set[str]] = {}
    for bench, jsonl_path in (retrieval_jsonl_by_bench or {}).items():
        retrieval_coverage[bench] = {
            str(row.get("task_id") or "")
            for row in read_jsonl(jsonl_path)
            if row.get("task_id") is not None
        }
    for task in tasks:
        bench = task["bench"]
        task_id = str(task["task_id"])
        retrieval_jsonl = None
        retrieval_covered = True
        if arm == "retrieval":
            retrieval_jsonl = (retrieval_jsonl_by_bench or {}).get(bench)
            retrieval_covered = bool(retrieval_jsonl and task_id in retrieval_coverage.get(bench, set()))
        task_trials = int(task.get("_eval_trials", trials))
        for trial_index in range(task_trials):
            record = make_record(
                run_id=run_id,
                date=date,
                bench=bench,
                task_id=task_id,
                split=task.get("split", "rl"),
                mode=mode,
                model_role="eval",
                model=model,
                arm=arm,
                trial_index=trial_index,
                retrieval_jsonl=retrieval_jsonl,
                retrieval_top_n=retrieval_top_n,
                max_turns=max_turns,
                max_time=max_time,
                retrieval_covered=retrieval_covered,
                implicit_mode="",
            )
            env = record.setdefault("env", {})
            env["OPENAI_API_BASE"] = api_base.rstrip("/")
            env["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "dummy")
            env["DOCKER_HOST"] = docker_host
            env["UNIFIED_PROMPT_PROFILE"] = "openclaw_full"
            env["UNIFIED_TOOLS_SCHEMA_MODE"] = "openai_tools"
            env["UNIFIED_CLAW_USE_DOCKER_SANDBOX"] = "1"
            env["UNIFIED_DISABLE_THINKING"] = "1"
            env["UNIFIED_PRESENCE_PENALTY"] = "1.5"
            env["UNIFIED_EARLY_STOP_N"] = "3"
            env["UNIFIED_ROLLOUT_WALLCLOCK_CAP_SEC"] = str(max_time)
            env["UNIFIED_VERIFIER_TIMEOUT_CAP_SEC"] = "300"
            env["UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS"] = "1"
            env["UNIFIED_HARBOR_REQUIRE_PREBUILT_LOCAL"] = "1"
            env["AGENT_BENCH_DOCKER_START_CONCURRENCY"] = os.environ.get("DOCKER_START_CAP", "32")
            records.append(record)
    return records


def cmd_make_plan(args: argparse.Namespace) -> None:
    tasks = load_tasks(args)
    retrieval_by_bench = {}
    if args.retrieval_root:
        retrieval_root = repo_path(args.retrieval_root)
        retrieval_by_bench = {
            bench: display_path(retrieval_root / f"{bench}.jsonl")
            for bench in BENCHES
            if (retrieval_root / f"{bench}.jsonl").exists()
        }
    records = build_plan_records(
        tasks=tasks,
        run_id=args.run_id,
        date=args.date,
        model=args.model,
        mode=args.mode,
        arm=args.arm,
        trials=args.trials,
        max_turns=args.max_turns,
        max_time=args.max_time,
        docker_host=args.docker_host,
        api_base=args.api_base,
        retrieval_jsonl_by_bench=retrieval_by_bench,
        retrieval_top_n=args.retrieval_top_n,
    )
    out = repo_path(args.out)
    write_jsonl(out, records)
    counts = Counter((r["split"], r["bench"]) for r in records)
    write_json(
        out.with_suffix(".summary.json"),
        {
            "run_id": args.run_id,
            "date": args.date,
            "model": args.model,
            "mode": args.mode,
            "arm": args.arm,
            "records": len(records),
            "trials": args.trials,
            "task_repeat_schedule": dict(sorted(Counter(
                int(task.get("_eval_trials", args.trials)) for task in tasks
            ).items())),
            "task_count": len(tasks),
            "counts": {f"{split}/{bench}": count for (split, bench), count in sorted(counts.items())},
            "retrieval_root": display_path(args.retrieval_root) if args.retrieval_root else "",
        },
    )
    print(f"wrote {len(records)} records to {display_path(out)}")


def load_latest_status(run_root: Path) -> dict[str, dict[str, Any]]:
    status_path = run_root / "logs" / "sft_collection" / "status.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(status_path):
        trial_id = str(row.get("trial_id") or "")
        if trial_id:
            latest[trial_id] = row
    return latest


def load_completed_attempts(plan: list[dict[str, Any]], run_root: Path) -> dict[str, list[dict[str, Any]]]:
    status = load_latest_status(run_root)
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in plan:
        row = status.get(str(record.get("trial_id") or ""))
        trajectory_path = repo_path((row or {}).get("trajectory_path") or record.get("trajectory_path", ""))
        if not trajectory_path.exists():
            continue
        try:
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        incremental = latest_incremental(record, row)
        resolved = bool(incremental.get("resolved", trajectory.get("resolved", False)))
        attempts[task_key(str(record["bench"]), str(record["task_id"]))].append(
            {
                "trial_id": record.get("trial_id"),
                "bench": record.get("bench"),
                "task_id": str(record.get("task_id")),
                "split": record.get("split"),
                "trial_index": record.get("trial_index"),
                "resolved": resolved,
                "score": incremental.get("score", trajectory.get("score")),
                "error": incremental.get("error", trajectory.get("error", "")),
                "turns": incremental.get("turns"),
                "trajectory_path": display_path(trajectory_path),
                "messages": trajectory.get("messages") or trajectory.get("trajectory") or [],
            }
        )
    return attempts


def latest_incremental(record: dict[str, Any], status: dict[str, Any] | None) -> dict[str, Any]:
    path = repo_path((status or {}).get("incremental_path") or record.get("incremental_path", ""))
    task_id = str(record.get("task_id", ""))
    for row in reversed(read_jsonl(path)):
        if str(row.get("task_id") or row.get("instance_id") or "") == task_id:
            return row
    return {}


def compact_message_content(content: Any, max_chars: int) -> str:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n...[truncated]"
    return text


def summarize_attempt(attempt: dict[str, Any], max_chars: int = 6500) -> str:
    messages = attempt.get("messages") or []
    chunks = [
        f"trial_id: {attempt.get('trial_id')}",
        f"resolved: {attempt.get('resolved')} score: {attempt.get('score')} turns: {attempt.get('turns')}",
        f"error: {attempt.get('error') or ''}",
        f"trajectory_path: {attempt.get('trajectory_path')}",
    ]
    selected = []
    for msg in messages[-14:]:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "")
        content = compact_message_content(msg.get("content", ""), 900)
        selected.append(f"[{role}]\n{content}")
    body = "\n\n".join(chunks + selected)
    if len(body) > max_chars:
        body = body[-max_chars:]
        body = "[leading trajectory context truncated]\n" + body
    return body


def build_skill_prompt(
    *,
    prompt_index: dict[str, dict[str, str]],
    bench: str,
    task_id: str,
    attempts: list[dict[str, Any]],
    previous_skill: str = "",
    revision_failure: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    key = task_key(bench, task_id)
    task_prompt = prompt_index.get(key, {}).get("prompt", "")
    successes = [a for a in attempts if a.get("resolved")]
    failures = [a for a in attempts if not a.get("resolved")]
    evidence = []
    for label, bucket in (("SUCCESS", successes[:2]), ("FAILURE", failures[:2])):
        for attempt in bucket:
            evidence.append(f"### {label} ATTEMPT\n{summarize_attempt(attempt)}")
    if not evidence and attempts:
        evidence.append(summarize_attempt(attempts[0]))
    if revision_failure:
        evidence.append(f"### ORACLE-SKILL FAILURE TO FIX\n{summarize_attempt(revision_failure, max_chars=9000)}")

    system = (
        "You write concise task-specific agent skills. Return only a complete SKILL.md file. "
        "Do not wrap it in markdown fences. Do not include analysis outside the file."
    )
    mode = "revise" if previous_skill else "create"
    user = f"""Create mode: {mode}

Benchmark: {bench}
Task id: {task_id}

Task prompt excerpt:
{task_prompt[:6000]}

Baseline rollout evidence:
{chr(10).join(evidence) if evidence else '(no completed trajectory evidence available)'}

{"Previous SKILL.md to revise:" if previous_skill else ""}
{previous_skill[:5000] if previous_skill else ""}

Write a single SKILL.md that helps an OpenClaw-style coding agent solve this exact task family.

Hard requirements:
- Include YAML frontmatter with `name:` and `description:`.
- Keep the whole file under 2500 words and preferably under 1400 words.
- Optimize for operational steps, checks, verifier pitfalls, and common failure modes visible in the evidence.
- Do not mention hidden labels, rewards, this prompt, or that the skill was generated from trajectories.
- Do not paste long code from a trajectory. Give compact procedures and command patterns instead.
- If the task needs reading files or running commands, say what to inspect and how to verify.
- If the evidence shows a bad approach, include a short "Avoid" bullet.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def chat_completion(api_base: str, api_key: str, model: str, messages: list[dict[str, str]], *, temperature: float = 0.2, max_tokens: int = 1800) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(api_base.rstrip("/") + "/chat/completions", data=data, headers=headers)
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8"))
            return str(body["choices"][0]["message"]["content"]).strip()
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_exc = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"chat completion failed after retries: {last_exc}")


def extract_skill_candidate(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:markdown)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    text = text.replace("<think>", "").replace("</think>", "").strip()

    frontmatter = re.compile(r"(?ms)^\s*---\s*\n(?P<body>.*?\n)---\s*\n")
    matches = list(frontmatter.finditer(text))
    valid = [
        match for match in matches
        if "name:" in match.group("body") and "description:" in match.group("body")
    ]
    if valid:
        return text[valid[-1].start():].lstrip()
    lines = text.splitlines()
    nonempty_index = next((idx for idx, line in enumerate(lines) if line.strip()), -1)
    if nonempty_index >= 0 and lines[nonempty_index].strip().startswith("name:"):
        description_index = next(
            (
                idx for idx in range(nonempty_index + 1, min(len(lines), nonempty_index + 6))
                if lines[idx].strip().startswith("description:")
            ),
            -1,
        )
        if description_index >= 0:
            body_start = description_index + 1
            while body_start < len(lines) and not lines[body_start].strip():
                body_start += 1
            if body_start < len(lines) and lines[body_start].strip() == "---":
                body_start += 1
            front = "\n".join(line.strip() for line in lines[nonempty_index:description_index + 1])
            body = "\n".join(lines[body_start:]).lstrip()
            return f"---\n{front}\n---\n\n{body}".rstrip()
    return text


def normalize_skill_text(text: str, name: str, description: str) -> str:
    text = extract_skill_candidate(text)
    if not text.startswith("---"):
        text = f"---\nname: {name}\ndescription: {description}\n---\n\n{text}"
    parts = text.split("---", 2)
    frontmatter = parts[1] if len(parts) > 2 else ""
    if "name:" not in frontmatter:
        text = text.replace("---\n", f"---\nname: {name}\n", 1)
        parts = text.split("---", 2)
        frontmatter = parts[1] if len(parts) > 2 else ""
    if "description:" not in frontmatter:
        text = text.replace("---\n", f"---\nname: {name}\ndescription: {description}\n", 1)
    return text.rstrip() + "\n"


def skill_text_is_clean(text: str) -> bool:
    if not text.startswith("---"):
        return False
    parts = text.split("---", 2)
    if len(parts) < 3 or "name:" not in parts[1] or "description:" not in parts[1]:
        return False
    lowered = text.lower()
    return not any(marker.lower() in lowered for marker in SKILL_CONTAMINATION_MARKERS)


def repair_skill_text(api_base: str, api_key: str, model: str, dirty_text: str, name: str, description: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the provided text into a clean SKILL.md file. "
                "Return only the file content. Do not include reasoning, analysis, or markdown fences."
            ),
        },
        {
            "role": "user",
            "content": f"""Required frontmatter:
---
name: {name}
description: {description}
---

Dirty/generated text to rewrite:
{dirty_text[:7000]}

Write a concise task skill with operational steps, checks, and avoid bullets. Return only SKILL.md.
""",
        },
    ]
    repaired = chat_completion(api_base, api_key, model, messages, temperature=0.1, max_tokens=2400)
    return normalize_skill_text(repaired, name, description)


def make_clean_skill_text(
    raw: str,
    *,
    name: str,
    description: str,
    api_base: str,
    api_key: str,
    model: str,
) -> tuple[str, bool]:
    skill_text = normalize_skill_text(raw, name, description)
    if skill_text_is_clean(skill_text):
        return skill_text, False
    repaired = repair_skill_text(api_base, api_key, model, skill_text, name, description)
    return repaired, True


def run_keyed_tasks(keys: list[str], worker_fn, workers: int) -> list[Any]:
    """Run worker_fn over keys with a thread pool, preserving keys order.

    SGLang continuous batching serves concurrent chat completions at nearly
    the same latency as one, so the serial skill loop left the GPUs idle.
    Workers only write inside their own task dir; the returned entries are the
    only shared state."""
    workers = max(1, int(workers))
    if workers == 1:
        return [worker_fn(key) for key in keys]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(worker_fn, keys))


def cmd_generate_skills(args: argparse.Namespace) -> None:
    plan = read_jsonl(args.plan)
    run_root = repo_path(args.run_root)
    skills_root = repo_path(args.skills_root)
    prompt_index = load_prompt_index(args.train_parquet, args.eval_parquet)
    attempts = load_completed_attempts(plan, run_root)

    def generate_one(key: str) -> dict[str, Any]:
        bench, task_id = key.split("/", 1)
        out_dir = skill_dir_for(skills_root, bench, task_id)
        final_path = out_dir / "SKILL.md"
        if final_path.exists() and not args.force:
            return {"bench": bench, "task_id": task_id, "skill_dir": display_path(out_dir), "status": "exists"}
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.force:
            for stale_revision in out_dir.glob("SKILL.v[2-9]*.md"):
                stale_revision.unlink()
        name = skill_name_for(bench, task_id)
        description = f"Task-specific skill for {bench}/{task_id}."
        messages = build_skill_prompt(
            prompt_index=prompt_index,
            bench=bench,
            task_id=task_id,
            attempts=attempts.get(key, []),
        )
        raw = chat_completion(args.api_base, args.api_key, args.model, messages, max_tokens=2400)
        skill_text, repaired = make_clean_skill_text(
            raw,
            name=name,
            description=description,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
        )
        (out_dir / "SKILL.v1.md").write_text(skill_text, encoding="utf-8")
        final_path.write_text(skill_text, encoding="utf-8")
        write_json(out_dir / "meta.json", {
            "bench": bench,
            "task_id": task_id,
            "created_at": utc_now(),
            "source_run_root": display_path(run_root),
            "source_plan": display_path(args.plan),
            "attempts_used": len(attempts.get(key, [])),
            "successes_used": sum(1 for a in attempts.get(key, []) if a.get("resolved")),
            "version": "v1",
            "repaired_generation": repaired,
        })
        print(f"[skill] {bench}/{task_id}: {display_path(final_path)}")
        return {"bench": bench, "task_id": task_id, "skill_dir": display_path(out_dir), "status": "generated"}

    keys = sorted({task_key(str(r["bench"]), str(r["task_id"])) for r in plan})
    manifest = run_keyed_tasks(keys, generate_one, args.workers)

    write_json(skills_root / "manifest.json", {"generated_at": utc_now(), "skills": manifest})
    write_retrieval_jsonls(skills_root, repo_path(args.retrieval_root), manifest)


def write_retrieval_jsonls(skills_root: Path, retrieval_root: Path, manifest: list[dict[str, Any]] | None = None) -> None:
    if manifest is None:
        manifest = []
        for skill_file in sorted(skills_root.glob("*/*/SKILL.md")):
            bench = skill_file.parent.parent.name
            task_id = skill_file.parent.name
            meta_path = skill_file.parent / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    task_id = str(meta.get("task_id") or task_id)
                    bench = str(meta.get("bench") or bench)
                except Exception:
                    pass
            manifest.append({"bench": bench, "task_id": task_id, "skill_dir": display_path(skill_file.parent), "status": "final"})
    retrieval_root.mkdir(parents=True, exist_ok=True)
    by_bench: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        if row.get("status") not in {"generated", "exists", "revised", "final"}:
            continue
        bench = str(row["bench"])
        task_id = str(row["task_id"])
        skill_dir = repo_path(row["skill_dir"])
        if not (skill_dir / "SKILL.md").exists():
            continue
        skill_name = skill_name_for(bench, task_id)
        item = {
            "task_id": task_id,
            "reranked_top10": [{
                "skill_name": skill_name,
                "skill_path": str(skill_dir),
                "score": 1.0,
            }],
            "reranked_top5": [{
                "skill_name": skill_name,
                "skill_path": str(skill_dir),
                "score": 1.0,
            }],
            "coarse_top20": [{
                "skill_name": skill_name,
                "skill_path": str(skill_dir),
                "score": 1.0,
            }],
        }
        by_bench[bench].append(item)
    for bench, rows in by_bench.items():
        path = retrieval_root / f"{bench}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in sorted(rows, key=lambda x: str(x["task_id"])):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[retrieval] {bench}: {len(rows)} -> {display_path(path)}")


def cmd_write_retrieval(args: argparse.Namespace) -> None:
    write_retrieval_jsonls(repo_path(args.skills_root), repo_path(args.retrieval_root))


def failed_attempts_from_eval(plan: list[dict[str, Any]], run_root: Path) -> dict[str, dict[str, Any]]:
    attempts = load_completed_attempts(plan, run_root)
    failed: dict[str, dict[str, Any]] = {}
    for key, rows in attempts.items():
        if rows and not any(row.get("resolved") for row in rows):
            failed[key] = rows[-1]
    return failed


def cmd_revise_failed(args: argparse.Namespace) -> None:
    baseline_plan = read_jsonl(args.baseline_plan)
    eval_plan = read_jsonl(args.eval_plan)
    baseline_attempts = load_completed_attempts(baseline_plan, repo_path(args.baseline_run_root))
    failed = failed_attempts_from_eval(eval_plan, repo_path(args.eval_run_root))
    prompt_index = load_prompt_index(args.train_parquet, args.eval_parquet)
    skills_root = repo_path(args.skills_root)
    def revise_one(key: str) -> tuple[dict[str, Any], str] | None:
        failure = failed[key]
        bench, task_id = key.split("/", 1)
        out_dir = skill_dir_for(skills_root, bench, task_id)
        skill_path = out_dir / "SKILL.md"
        if not skill_path.exists():
            return None
        previous = skill_path.read_text(encoding="utf-8", errors="replace")
        if (out_dir / "SKILL.v2.md").exists() and not args.force:
            return (
                {"bench": bench, "task_id": task_id, "skill_dir": display_path(out_dir), "status": "exists"},
                f"{bench}\t{task_id}\n",
            )
        messages = build_skill_prompt(
            prompt_index=prompt_index,
            bench=bench,
            task_id=task_id,
            attempts=baseline_attempts.get(key, []),
            previous_skill=previous,
            revision_failure=failure,
        )
        name = skill_name_for(bench, task_id)
        description = f"Revised task-specific skill for {bench}/{task_id}."
        raw = chat_completion(args.api_base, args.api_key, args.model, messages, temperature=0.15, max_tokens=2400)
        revised, repaired = make_clean_skill_text(
            raw,
            name=name,
            description=description,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
        )
        (out_dir / "SKILL.v2.md").write_text(revised, encoding="utf-8")
        skill_path.write_text(revised, encoding="utf-8")
        meta_path = out_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        meta.update({
            "bench": bench,
            "task_id": task_id,
            "revised_at": utc_now(),
            "revision_source_run_root": display_path(args.eval_run_root),
            "revision_failure_trial_id": failure.get("trial_id"),
            "version": "v2",
            "repaired_revision": repaired,
        })
        write_json(meta_path, meta)
        print(f"[revise] {bench}/{task_id}: {display_path(skill_path)}")
        return (
            {"bench": bench, "task_id": task_id, "skill_dir": display_path(out_dir), "status": "revised"},
            f"{bench}\t{task_id}\n",
        )

    keys = sorted(failed.keys())
    results = [r for r in run_keyed_tasks(keys, revise_one, args.workers) if r is not None]
    manifest = [entry for entry, _ in results]
    failed_task_lines = [line for _, line in results]

    write_json(args.revision_manifest, {"generated_at": utc_now(), "failed_tasks": len(failed), "revisions": manifest})
    task_list_path = repo_path(args.failed_task_list)
    task_list_path.parent.mkdir(parents=True, exist_ok=True)
    task_list_path.write_text("".join(failed_task_lines), encoding="utf-8")
    write_retrieval_jsonls(skills_root, repo_path(args.retrieval_root))
    print(f"failed/revised task list: {display_path(task_list_path)} ({len(failed_task_lines)})")


def cmd_report(args: argparse.Namespace) -> None:
    rows = []
    for name, plan_path, root_path in args.run:
        plan = read_jsonl(plan_path)
        plan_tasks = {(str(r["bench"]), str(r["task_id"])) for r in plan}
        attempts = load_completed_attempts(plan, repo_path(root_path))
        resolved = sum(1 for values in attempts.values() if any(v.get("resolved") for v in values))
        trials_total = sum(len(values) for values in attempts.values())
        trials_resolved = sum(
            sum(1 for v in values if v.get("resolved")) for values in attempts.values()
        )
        per_bench: dict[str, dict[str, int]] = defaultdict(lambda: {"trials": 0, "resolved": 0, "tasks": 0, "tasks_resolved": 0})
        for key, values in attempts.items():
            bench = key.split("/", 1)[0]
            per_bench[bench]["tasks"] += 1
            per_bench[bench]["tasks_resolved"] += int(any(v.get("resolved") for v in values))
            per_bench[bench]["trials"] += len(values)
            per_bench[bench]["resolved"] += sum(1 for v in values if v.get("resolved"))
        rows.append({
            "name": name,
            "plan_tasks": len(plan_tasks),
            "plan_records": len(plan),
            "tasks_with_attempts": len(attempts),
            "tasks_resolved": resolved,
            # pass@1 = resolved trials / completed trials (mean over rollouts);
            # the comparison metric across baseline / oracle_v1 / oracle_v2 arms.
            "trials_total": trials_total,
            "trials_resolved": trials_resolved,
            "pass_at_1": round(trials_resolved / trials_total, 4) if trials_total else 0.0,
            "per_bench": {bench: dict(stats) for bench, stats in sorted(per_bench.items())},
        })
    write_json(args.out, {"generated_at": utc_now(), "runs": rows})
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def add_common_task_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-parquet", default=DEFAULT_TRAIN_PARQUET)
    parser.add_argument("--eval-parquet", default=DEFAULT_EVAL_PARQUET)
    parser.add_argument("--bench", action="append", choices=BENCHES)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument(
        "--task-list",
        default="",
        help=(
            "Optional TSV of `bench\\ttask_id[\\trepeats]` lines. When set, "
            "restrict the plan to exactly these tasks; the optional third column "
            "overrides --trials for that task."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    make_plan = sub.add_parser("make-plan", help="Create a launch_trials-compatible plan")
    add_common_task_args(make_plan)
    make_plan.add_argument("--run-id", required=True)
    make_plan.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    make_plan.add_argument("--model", default="qwen3.5-27b")
    make_plan.add_argument("--mode", required=True)
    make_plan.add_argument("--arm", choices=["baseline", "retrieval"], required=True)
    make_plan.add_argument("--trials", type=int, default=1)
    make_plan.add_argument("--max-turns", type=int, default=30)
    make_plan.add_argument("--max-time", type=int, default=850)
    make_plan.add_argument("--docker-host", default="tcp://127.0.0.1:2376")
    make_plan.add_argument("--api-base", default="http://127.0.0.1:30000/v1")
    make_plan.add_argument("--retrieval-root", default="")
    make_plan.add_argument("--retrieval-top-n", type=int, default=1)
    make_plan.add_argument("--out", required=True)
    make_plan.set_defaults(func=cmd_make_plan)

    gen = sub.add_parser("generate-skills", help="Generate v1 SKILL.md files from baseline trajectories")
    gen.add_argument("--plan", required=True)
    gen.add_argument("--run-root", required=True)
    gen.add_argument("--skills-root", required=True)
    gen.add_argument("--retrieval-root", required=True)
    gen.add_argument("--train-parquet", default=DEFAULT_TRAIN_PARQUET)
    gen.add_argument("--eval-parquet", default=DEFAULT_EVAL_PARQUET)
    gen.add_argument("--api-base", default="http://127.0.0.1:30000/v1")
    gen.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "dummy"))
    gen.add_argument("--model", default="qwen3.5-27b")
    gen.add_argument("--force", action="store_true")
    gen.add_argument("--workers", type=int, default=8, help="concurrent chat completions (SGLang batches them)")
    gen.set_defaults(func=cmd_generate_skills)

    wr = sub.add_parser("write-retrieval", help="Write per-bench retrieval jsonls from oracle skill dirs")
    wr.add_argument("--skills-root", required=True)
    wr.add_argument("--retrieval-root", required=True)
    wr.set_defaults(func=cmd_write_retrieval)

    rev = sub.add_parser("revise-failed", help="Revise skills for tasks that failed oracle v1 eval")
    rev.add_argument("--baseline-plan", required=True)
    rev.add_argument("--baseline-run-root", required=True)
    rev.add_argument("--eval-plan", required=True)
    rev.add_argument("--eval-run-root", required=True)
    rev.add_argument("--skills-root", required=True)
    rev.add_argument("--retrieval-root", required=True)
    rev.add_argument("--failed-task-list", required=True)
    rev.add_argument("--revision-manifest", required=True)
    rev.add_argument("--train-parquet", default=DEFAULT_TRAIN_PARQUET)
    rev.add_argument("--eval-parquet", default=DEFAULT_EVAL_PARQUET)
    rev.add_argument("--api-base", default="http://127.0.0.1:30000/v1")
    rev.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "dummy"))
    rev.add_argument("--model", default="qwen3.5-27b")
    rev.add_argument("--force", action="store_true")
    rev.add_argument("--workers", type=int, default=8, help="concurrent chat completions (SGLang batches them)")
    rev.set_defaults(func=cmd_revise_failed)

    report = sub.add_parser("report", help="Write a compact JSON report")
    report.add_argument("--run", nargs=3, action="append", metavar=("NAME", "PLAN", "RUN_ROOT"), required=True)
    report.add_argument("--out", required=True)
    report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
