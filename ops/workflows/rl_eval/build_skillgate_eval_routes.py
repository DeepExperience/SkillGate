#!/usr/bin/env python3
"""Build audited single-skill retrieval routes for SkillGate paper evals.

The generated directory is accepted directly by ``run_eval70_checkpoint_set.sh
--skill-mode retrieve --snapshot DIR``.  All modes preserve the frozen skill
paths from the input slate and write one ``reranked_top10`` entry per task.

Modes:
  oracle      expose the manifest oracle only;
  misleading expose one deterministic misleading candidate only;
  router      ask an OpenAI-compatible frozen model to choose one candidate;
  reranker    score the 16 task/description pairs with Qwen3-Reranker-8B.

Router generation is restartable: each valid selection is atomically recorded
in selections.jsonl before the next task is requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "GeneralAgent/eval_scripts"))

from unified_runner.retrieval_skill_inject import _read_skill_description  # noqa: E402


DEFAULT_MANIFEST = ROOT / "skill_libraries/snapshots/rl/eval70_final_v8prod_fixed4/slate_manifest_eval70.jsonl"
DEFAULT_SNAPSHOT = ROOT / "skill_libraries/snapshots/rl/eval70_final_v8prod_fixed4/snapshot_eval70"
DEFAULT_TASK_LIST = ROOT / "ops/workflows/rl_eval/specs/eval70_v1/tasks.tsv"
DEFAULT_TRAIN = ROOT / "datasets/rl/parquet_4bench_base_20260523/train.parquet"
DEFAULT_EVAL = ROOT / "datasets/rl/parquet_4bench_base_20260523/eval.parquet"
DEFAULT_RERANKER = "Qwen/Qwen3-Reranker-8B"
ROUTER_PROMPT_VERSION = "single-selection-v3-eval-description-json-schema"
RERANKER_PROMPT_VERSION = "qwen3-reranker-v1-eval-description"
BENCH_FILES = ("claw", "sb_ns", "seta_synth", "swe_lite", "tb2")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "mode": args.mode,
        "manifest": sha256_file(args.manifest),
        "task_list": sha256_file(args.task_list),
        "snapshot": {p.name: sha256_file(p) for p in sorted(args.snapshot.glob("*.jsonl"))},
        "router_model": args.router_model if args.mode == "router" else "",
        "router_prompt_version": ROUTER_PROMPT_VERSION if args.mode == "router" else "",
        "reranker_model": args.reranker_model if args.mode == "reranker" else "",
        **({"reranker_prompt_version": RERANKER_PROMPT_VERSION} if args.mode == "reranker" else {}),
        "misleading_index": args.misleading_index if args.mode == "misleading" else None,
        "max_task_chars": args.max_task_chars,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_task_list(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        bench, task_id = (part.strip() for part in raw.split("\t", 1))
        rows.append((bench, task_id))
    if not rows or len(rows) != len(set(rows)):
        raise SystemExit(f"task list must be non-empty and unique: {path}")
    return rows


def load_snapshot(snapshot: Path, *, validate_all_skills: bool) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(snapshot.glob("*.jsonl")):
        bench = path.stem
        for row in read_jsonl(path):
            key = (bench, str(row["task_id"]))
            if key in out:
                raise SystemExit(f"duplicate snapshot task: {key}")
            candidates = row.get("reranked_top10") or []
            categories = row.get("slate_categories") or {}
            if len(candidates) != 16 or len(categories) != 16:
                raise SystemExit(f"expected 16 candidates/categories for {key}")
            if validate_all_skills:
                for candidate in candidates:
                    path_value = Path(str(candidate.get("skill_path") or ""))
                    if not (path_value / "SKILL.md").is_file():
                        raise SystemExit(f"missing frozen skill for {key}: {path_value}")
            out[key] = row
    return out


def extract_prompt(value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value
    if isinstance(value, list):
        for item in reversed(value):
            if isinstance(item, dict) and item.get("role") == "user":
                return str(item.get("content") or "")
        for item in reversed(value):
            if isinstance(item, dict) and item.get("content"):
                return str(item["content"])
    return str(value)


def load_task_prompts(train_path: Path, eval_path: Path) -> dict[tuple[str, str], str]:
    import pandas as pd

    prompts: dict[tuple[str, str], str] = {}
    for path in (train_path, eval_path):
        frame = pd.read_parquet(path, columns=["prompt", "extra_info"])
        for row in frame.itertuples(index=False):
            extra = row.extra_info
            if isinstance(extra, str):
                extra = json.loads(extra)
            key = (str(extra.get("bench") or ""), str(extra.get("task_id") or ""))
            if all(key) and key not in prompts:
                prompts[key] = extract_prompt(row.prompt)
    return prompts


def skill_description(skill_path: str) -> str:
    return _read_skill_description(Path(skill_path))


def candidates_for(row: dict[str, Any], *, include_descriptions: bool) -> list[dict[str, Any]]:
    categories = row["slate_categories"]
    result = []
    for position, candidate in enumerate(row["reranked_top10"], 1):
        name = str(candidate["skill_name"])
        path = str(candidate["skill_path"])
        result.append(
            {
                "position": position,
                "skill_name": name,
                "skill_path": path,
                "category": str(categories[name]),
                "description": skill_description(path) if include_descriptions else "",
            }
        )
    return result


def router_prompt(task: str, candidates: list[dict[str, Any]], max_task_chars: int) -> tuple[str, str]:
    system = (
        "You are a frozen skill router. Select exactly one candidate skill that is most likely "
        "to help solve the task. Use only the task and candidate names/descriptions. Return one "
        "JSON object with exactly this schema: {\"skill_name\": \"candidate-name\"}."
    )
    lines = ["<task>", task[:max_task_chars], "</task>", "<candidate_skills>"]
    for item in candidates:
        lines.append(
            f"{item['position']}. name={item['skill_name']}\n"
            f"   description={item['description']}"
        )
    lines.extend(["</candidate_skills>", "Select one candidate now."])
    return system, "\n".join(lines)


def request_chat(
    api_base: str,
    model: str,
    system: str,
    user: str,
    candidate_names: list[str],
    timeout: int,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": 256,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "skill_selection",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "skill_name": {"type": "string", "enum": candidate_names},
                        },
                        "required": ["skill_name"],
                        "additionalProperties": False,
                    },
                },
            },
        }
    ).encode()
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    return str(body["choices"][0]["message"]["content"])


def parse_router_choice(text: str, candidates: list[dict[str, Any]]) -> str:
    names = [item["skill_name"] for item in candidates]
    match = re.search(r'"skill_name"\s*:\s*"([^"]+)"', text)
    if match and match.group(1).strip() in names:
        return match.group(1).strip()
    lowered = text.casefold()
    matches = [name for name in names if name.casefold() in lowered]
    return matches[0] if len(matches) == 1 else ""


def choose_router(
    args: argparse.Namespace,
    task: str,
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    system, user = router_prompt(task, candidates, args.max_task_chars)
    errors: list[str] = []
    for attempt in range(1, args.max_attempts + 1):
        try:
            raw = request_chat(
                args.api_base,
                args.router_model,
                system,
                user,
                [item["skill_name"] for item in candidates],
                args.request_timeout,
            )
            name = parse_router_choice(raw, candidates)
            if name:
                selected = next(item for item in candidates if item["skill_name"] == name)
                return selected, {"raw_response": raw, "attempts": attempt}
            errors.append(f"attempt {attempt}: invalid choice: {raw[:400]}")
        except (OSError, KeyError, ValueError, urllib.error.URLError) as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        time.sleep(min(2**attempt, 8))
    raise RuntimeError("; ".join(errors))


def choose_reranker(
    reranker: Any,
    task: str,
    candidates: list[dict[str, Any]],
    max_task_chars: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    docs = [f"name: {item['skill_name']}\ndescription: {item['description']}" for item in candidates]
    scores = reranker.score_pairs(task[:max_task_chars], docs)
    best = max(range(len(scores)), key=lambda index: (scores[index], -index))
    return candidates[best], {"scores": {item["skill_name"]: score for item, score in zip(candidates, scores)}}


def selection_record(
    bench: str,
    task_id: str,
    selected: dict[str, Any],
    details: dict[str, Any],
    fingerprint: str,
    mode: str,
) -> dict[str, Any]:
    return {
        "bench": bench,
        "task_id": task_id,
        "selection_method": mode,
        "selected_name": selected["skill_name"],
        "selected_path": selected["skill_path"],
        "selected_category": selected["category"],
        "selected_position": selected["position"],
        "description": selected["description"],
        "details": details,
        "input_fingerprint": fingerprint,
        "created_at": utc_now(),
    }


def route_record(selection: dict[str, Any]) -> dict[str, Any]:
    score = 1.0
    scores = (selection.get("details") or {}).get("scores") or {}
    if selection["selected_name"] in scores:
        score = float(scores[selection["selected_name"]])
    return {
        "task_id": selection["task_id"],
        "dataset": selection["bench"],
        "selection_method": selection["selection_method"],
        "reranked_top10": [
            {
                "rank": 1,
                "score": score,
                "skill_name": selection["selected_name"],
                "skill_path": selection["selected_path"],
            }
        ],
        "slate_categories": {selection["selected_name"]: selection["selected_category"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("oracle", "misleading", "router", "reranker"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--task-list", type=Path, default=DEFAULT_TASK_LIST)
    parser.add_argument("--train-parquet", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-parquet", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--router-model", default="")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER)
    parser.add_argument("--max-task-chars", type=int, default=12000)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--misleading-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test prefix; final output remains incomplete")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in (args.manifest, args.task_list, args.train_parquet, args.eval_parquet):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if not args.snapshot.is_dir():
        raise SystemExit(f"missing snapshot: {args.snapshot}")
    if args.mode == "router" and not args.router_model:
        raise SystemExit("--router-model is required in router mode")

    args.output_root.mkdir(parents=True, exist_ok=True)
    fingerprint = input_fingerprint(args)
    summary_path = args.output_root / "summary.json"
    if summary_path.exists() and not args.overwrite:
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        if prior.get("input_fingerprint") != fingerprint:
            raise SystemExit(f"output fingerprint mismatch; use --overwrite: {args.output_root}")
    if args.overwrite:
        for name in (*BENCH_FILES, "selections"):
            (args.output_root / f"{name}.jsonl").unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)

    task_keys = load_task_list(args.task_list)
    snapshot = load_snapshot(
        args.snapshot,
        validate_all_skills=args.mode in ("router", "reranker"),
    )
    missing = [key for key in task_keys if key not in snapshot]
    if missing:
        raise SystemExit(f"snapshot misses {len(missing)} task-list entries: {missing[:5]}")
    manifest_keys = {
        (str(row["bench"]), str(row["task_id"])) for row in read_jsonl(args.manifest)
    }
    if set(task_keys) - manifest_keys:
        raise SystemExit(f"manifest misses task-list entries: {sorted(set(task_keys) - manifest_keys)[:5]}")

    prompts: dict[tuple[str, str], str] = {}
    if args.mode in ("router", "reranker"):
        prompts = load_task_prompts(args.train_parquet, args.eval_parquet)
        missing_prompts = [key for key in task_keys if not prompts.get(key)]
        if missing_prompts:
            raise SystemExit(f"parquets miss {len(missing_prompts)} task prompts: {missing_prompts[:5]}")

    selections_path = args.output_root / "selections.jsonl"
    prior_rows = read_jsonl(selections_path) if selections_path.exists() else []
    selected_by_key = {
        (str(row["bench"]), str(row["task_id"])): row
        for row in prior_rows
        if row.get("input_fingerprint") == fingerprint and row.get("selected_name")
    }

    reranker = None
    if args.mode == "reranker":
        sys.path.insert(0, str(ROOT / "GeneralAgent/eval_scripts/skills_retrieval"))
        from retrieve_v6_3stage import Qwen3Reranker

        reranker = Qwen3Reranker(model_name=args.reranker_model, batch_size=16, max_len=8192)

    active_keys = task_keys[: args.limit] if args.limit > 0 else task_keys
    try:
        for index, key in enumerate(active_keys, 1):
            if key in selected_by_key:
                print(f"[{index}/{len(active_keys)}] resume {key[0]}/{key[1]} -> {selected_by_key[key]['selected_name']}")
                continue
            bench, task_id = key
            candidates = candidates_for(
                snapshot[key],
                include_descriptions=args.mode in ("router", "reranker"),
            )
            details: dict[str, Any] = {}
            if args.mode == "oracle":
                matches = [item for item in candidates if item["category"] == "oracle"]
                if len(matches) != 1:
                    raise RuntimeError(f"expected one oracle for {key}, got {len(matches)}")
                selected = matches[0]
            elif args.mode == "misleading":
                matches = [item for item in candidates if item["category"] == "misleading"]
                if not 0 <= args.misleading_index < len(matches):
                    raise RuntimeError(f"misleading index out of range for {key}")
                selected = matches[args.misleading_index]
                details = {"misleading_index": args.misleading_index}
            elif args.mode == "router":
                selected, details = choose_router(args, prompts[key], candidates)
            else:
                selected, details = choose_reranker(reranker, prompts[key], candidates, args.max_task_chars)
            if not (Path(selected["skill_path"]) / "SKILL.md").is_file():
                raise RuntimeError(f"selected skill is missing for {key}: {selected['skill_path']}")
            record = selection_record(bench, task_id, selected, details, fingerprint, args.mode)
            selected_by_key[key] = record
            write_jsonl(selections_path, [selected_by_key[k] for k in active_keys if k in selected_by_key])
            print(
                f"[{index}/{len(active_keys)}] {bench}/{task_id} -> {selected['skill_name']} "
                f"({selected['category']}, pos={selected['position']})"
            )
    finally:
        if reranker is not None:
            reranker.unload()

    complete = len(selected_by_key) == len(task_keys) and args.limit <= 0
    if complete:
        ordered = [selected_by_key[key] for key in task_keys]
        for bench in BENCH_FILES:
            write_jsonl(args.output_root / f"{bench}.jsonl", [route_record(row) for row in ordered if row["bench"] == bench])
    counts = Counter(row["selected_category"] for row in selected_by_key.values())
    summary = {
        "schema_version": 1,
        "mode": args.mode,
        "status": "complete" if complete else "incomplete",
        "input_fingerprint": fingerprint,
        "created_at": utc_now(),
        "task_count": len(task_keys),
        "selected_count": len(selected_by_key),
        "category_counts": dict(sorted(counts.items())),
        "oracle_selection_rate": counts.get("oracle", 0) / len(selected_by_key) if selected_by_key else 0.0,
        "router_model": args.router_model,
        "router_prompt_version": ROUTER_PROMPT_VERSION if args.mode == "router" else "",
        "reranker_model": args.reranker_model if args.mode == "reranker" else "",
        "reranker_prompt_version": RERANKER_PROMPT_VERSION if args.mode == "reranker" else "",
        "inputs": {
            "manifest": str(args.manifest),
            "snapshot": str(args.snapshot),
            "task_list": str(args.task_list),
            "train_parquet": str(args.train_parquet),
            "eval_parquet": str(args.eval_parquet),
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not complete:
        raise SystemExit("route generation is intentionally incomplete (smoke-test limit or missing selections)")


if __name__ == "__main__":
    main()
