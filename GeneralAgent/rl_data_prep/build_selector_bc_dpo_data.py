#!/usr/bin/env python3
"""Build the SkillGate selection-turn BC and DPO baseline datasets.

The source parquet contains one Hybrid mixed-slate prompt per training task.
BC teaches a single first-turn ``read(gold)`` action.  Whenever possible the
target is copied from a real clean-oracle rollout whose *first* tool call is
that read; otherwise a deterministic target is synthesized from the advertised
gold description.  DPO pairs the gold target with each of the five advertised
misleading candidates while keeping the response style fixed.

This is an output-data builder: it does not alter the FINAL parquet, skill
snapshots, rollout artifacts, or any historical training path.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = (
    PROJECT_ROOT
    / "datasets/rl/parquet_4bench_final_hybridtrain_v8prodfixed4eval_20260720/train.parquet"
)
DEFAULT_ROLLOUT_ROOT = (
    PROJECT_ROOT
    / "experiments/rl/runs/selector-clean-oracle-action-credit-sft9b-"
    "hybridv8b0704d-20260716_121116/segments"
)
DEFAULT_BC_DIR = (
    PROJECT_ROOT / "GeneralAgent/sft_training/llamafactory_data/20260721_selection_bc"
)
DEFAULT_DPO_DIR = (
    PROJECT_ROOT / "GeneralAgent/sft_training/llamafactory_data/20260721_selection_dpo"
)
EXPECTED_PARQUET_SHA256 = "6dd2350879c6337fc0304f6ea08973ee9d8697ed6a72c70467aab9ae41f30732"
THINK_PREFIX = "<think>\n\n</think>\n\n"
EXPECTED_TASKS = 491
EXPECTED_MISLEADING_PER_TASK = 5

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(?P<function>[^>\n]+)>"
    r"(?P<body>.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
PATH_RE = re.compile(r"<parameter=path>\s*(?P<path>.*?)\s*</parameter>", re.DOTALL)
SKILL_RE = re.compile(
    r"<skill>\s*<name>(?P<name>.*?)</name>\s*"
    r"<description>(?P<description>.*?)</description>\s*"
    r"<location>(?P<location>.*?)</location>\s*</skill>",
    re.DOTALL,
)


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else [value]


def task_key(bench: Any, task_id: Any) -> tuple[str, str]:
    return str(bench), str(task_id)


def normalize_messages(raw_prompt: Any) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for raw in as_list(raw_prompt):
        if not isinstance(raw, dict):
            raise ValueError(f"prompt entry is not a dict: {type(raw).__name__}")
        role = str(raw.get("role", ""))
        content = raw.get("content", "")
        if role not in {"system", "user"} or not isinstance(content, str):
            raise ValueError(f"unexpected prompt message: role={role!r}, content={type(content).__name__}")
        messages.append({"role": role, "content": content})
    if [message["role"] for message in messages] != ["system", "user"]:
        raise ValueError(f"expected [system,user] prompt, got {[m['role'] for m in messages]}")
    return messages


def advertised_skills(system_prompt: str) -> dict[str, dict[str, str]]:
    skills: dict[str, dict[str, str]] = {}
    for match in SKILL_RE.finditer(system_prompt):
        item = {
            key: html.unescape(match.group(key).strip())
            for key in ("name", "description", "location")
        }
        skills[item["name"]] = item
    return skills


def skill_name_from_path(path: str) -> str:
    clean = path.strip().rstrip("/")
    parts = clean.split("/")
    if len(parts) >= 2 and parts[-1] == "SKILL.md":
        return parts[-2]
    return ""


def first_tool_call(response: str) -> tuple[re.Match[str], str, str] | None:
    match = TOOL_CALL_RE.search(response)
    if match is None:
        return None
    path_match = PATH_RE.search(match.group("body"))
    path = path_match.group("path").strip() if path_match else ""
    return match, match.group("function").strip(), path


def iter_rollouts(root: Path) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    files = sorted(root.glob("*/rollout_result/train/*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no rollout JSONL files under {root}")
    for path in files:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    yield path, line_number, json.loads(line)


def collect_real_targets(root: Path) -> tuple[dict[tuple[str, str], str], dict[str, Any]]:
    candidates: dict[tuple[str, str], list[tuple[int, str, int, str]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    seen_files: set[Path] = set()
    for path, line_number, record in iter_rollouts(root):
        seen_files.add(path)
        counts["records"] += 1
        reward = record.get("reward") or {}
        if reward.get("selector_clean_oracle") != 1:
            continue
        counts["clean_oracle_records"] += 1
        actions = (record.get("selector_action_credit") or {}).get("actions") or []
        parsed = first_tool_call(str(record.get("response") or ""))
        if len(actions) != 1 or parsed is None:
            counts["rejected_not_single_or_unparsed"] += 1
            continue
        match, function, path_value = parsed
        action = actions[0]
        if (
            function != "read"
            or action.get("turn_index") != 0
            or action.get("category") != "oracle"
            or action.get("skill_name") != skill_name_from_path(path_value)
        ):
            counts["rejected_not_first_read_gold"] += 1
            continue
        target = str(record["response"])[: match.end()].strip()
        if "<image>" in target:
            counts["rejected_image"] += 1
            continue
        key = task_key(reward.get("bench"), reward.get("task_id"))
        candidates[key].append((len(target), str(path), line_number, target))
        counts["strict_first_read_gold_records"] += 1

    # The shortest valid real turn is deterministic and avoids teaching long,
    # meandering pre-read prose when several trajectories cover the same task.
    selected = {key: min(values)[3] for key, values in candidates.items()}
    return selected, {
        **dict(counts),
        "files": len(seen_files),
        "strict_tasks": len(selected),
    }


def render_selection_turn(name: str, description: str) -> str:
    reason = " ".join(description.split())
    return (
        "<skill_reasoning>\n"
        f"The `{name}` skill is the strongest match for this task: {reason}\n"
        "</skill_reasoning>\n\n"
        "I will read the most relevant skill once before proceeding.\n\n"
        "<tool_call>\n"
        "<function=read>\n"
        "<parameter=path>\n"
        f"/root/.claude/skills/{name}/SKILL.md\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )


def replace_selection(target: str, old: dict[str, str], new: dict[str, str]) -> str:
    # Replace the full path first, then description/name occurrences.  This
    # preserves the chosen target's reasoning length and style, so DPO cannot
    # win by exploiting a systematic real-vs-synthetic formatting difference.
    replacements = [
        (old["location"], new["location"]),
        (old["description"], new["description"]),
        (old["name"], new["name"]),
    ]
    transformed = target
    for source, destination in replacements:
        if source:
            transformed = transformed.replace(source, destination)
    parsed = first_tool_call(transformed)
    if parsed is None or parsed[1] != "read" or skill_name_from_path(parsed[2]) != new["name"]:
        raise ValueError(f"failed to transform read({old['name']}) to read({new['name']})")
    return transformed


def dataset_info(dataset_name: str, file_name: str, *, ranking: bool) -> dict[str, Any]:
    columns = {"messages": "messages"}
    if ranking:
        columns.update({"chosen": "chosen", "rejected": "rejected"})
    item: dict[str, Any] = {
        "file_name": file_name,
        "formatting": "openai",
        "columns": columns,
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "observation_tag": "tool",
            "function_tag": "function",
            "system_tag": "system",
        },
    }
    if ranking:
        item["ranking"] = True
    return {dataset_name: item}


def validate_outputs(
    bc_path: Path,
    dpo_path: Path,
    bc_dataset_name: str,
    dpo_dataset_name: str,
) -> dict[str, Any]:
    bc = json.loads(bc_path.read_text(encoding="utf-8"))
    dpo = json.loads(dpo_path.read_text(encoding="utf-8"))
    if len(bc) != EXPECTED_TASKS:
        raise ValueError(f"BC expected {EXPECTED_TASKS} records, found {len(bc)}")
    if len(dpo) != EXPECTED_TASKS * EXPECTED_MISLEADING_PER_TASK:
        raise ValueError(f"DPO expected 2455 records, found {len(dpo)}")

    bc_keys: set[tuple[str, str]] = set()
    for item in bc:
        messages = item.get("messages") or []
        if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            raise ValueError("invalid BC role sequence")
        target = str(messages[-1].get("content") or "")
        if not target.startswith(THINK_PREFIX) or "<image>" in json.dumps(item):
            raise ValueError("invalid BC target prefix or image placeholder")
        parsed = first_tool_call(target[len(THINK_PREFIX) :])
        meta = item.get("metadata") or {}
        if parsed is None or parsed[1] != "read" or skill_name_from_path(parsed[2]) != meta.get("gold_name"):
            raise ValueError("BC target does not read its gold skill first")
        bc_keys.add(task_key(meta.get("bench"), meta.get("task_id")))
    if len(bc_keys) != EXPECTED_TASKS:
        raise ValueError(f"BC task keys are not unique: {len(bc_keys)}")

    pair_counts: Counter[tuple[str, str]] = Counter()
    for item in dpo:
        messages = item.get("messages") or []
        if [m.get("role") for m in messages] != ["system", "user"]:
            raise ValueError("invalid DPO prompt role sequence")
        chosen = str((item.get("chosen") or {}).get("content") or "")
        rejected = str((item.get("rejected") or {}).get("content") or "")
        meta = item.get("metadata") or {}
        if not chosen.startswith(THINK_PREFIX) or not rejected.startswith(THINK_PREFIX):
            raise ValueError("DPO target missing think wrapper")
        chosen_call = first_tool_call(chosen[len(THINK_PREFIX) :])
        rejected_call = first_tool_call(rejected[len(THINK_PREFIX) :])
        if chosen_call is None or skill_name_from_path(chosen_call[2]) != meta.get("gold_name"):
            raise ValueError("DPO chosen target is not read(gold)")
        if rejected_call is None or skill_name_from_path(rejected_call[2]) != meta.get("misleading_name"):
            raise ValueError("DPO rejected target is not read(misleading)")
        if "<image>" in json.dumps(item):
            raise ValueError("DPO record contains an image placeholder")
        pair_counts[task_key(meta.get("bench"), meta.get("task_id"))] += 1
    if set(pair_counts.values()) != {EXPECTED_MISLEADING_PER_TASK} or len(pair_counts) != EXPECTED_TASKS:
        raise ValueError("DPO does not contain exactly five pairs per task")

    bc_info = json.loads((bc_path.parent / "dataset_info.json").read_text())
    dpo_info = json.loads((dpo_path.parent / "dataset_info.json").read_text())
    if bc_dataset_name not in bc_info or dpo_dataset_name not in dpo_info:
        raise ValueError("dataset_info.json is missing its dataset entry")
    if dpo_info[dpo_dataset_name].get("ranking") is not True:
        raise ValueError("DPO dataset_info.json is missing ranking=true")
    return {
        "bc_records": len(bc),
        "bc_unique_tasks": len(bc_keys),
        "dpo_records": len(dpo),
        "dpo_unique_tasks": len(pair_counts),
        "dpo_pairs_per_task": sorted(set(pair_counts.values())),
        "bc_sha256": sha256_file(bc_path),
        "dpo_sha256": sha256_file(dpo_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-parquet", default=str(DEFAULT_PARQUET))
    parser.add_argument("--rollout-root", default=str(DEFAULT_ROLLOUT_ROOT))
    parser.add_argument("--bc-output-dir", default=str(DEFAULT_BC_DIR))
    parser.add_argument("--dpo-output-dir", default=str(DEFAULT_DPO_DIR))
    parser.add_argument("--bc-dataset-name", default="skillgate_gold_selector_bc_20260721")
    parser.add_argument("--dpo-dataset-name", default="skillgate_selskill_dpo_20260721")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    train_parquet = repo_path(args.train_parquet)
    rollout_root = repo_path(args.rollout_root)
    bc_dir = repo_path(args.bc_output_dir)
    dpo_dir = repo_path(args.dpo_output_dir)
    bc_path = bc_dir / f"{args.bc_dataset_name}.json"
    dpo_path = dpo_dir / f"{args.dpo_dataset_name}.json"

    if args.validate_only:
        print(json.dumps(validate_outputs(
            bc_path, dpo_path, args.bc_dataset_name, args.dpo_dataset_name
        ), ensure_ascii=False, indent=2))
        return

    parquet_hash = sha256_file(train_parquet)
    if parquet_hash != EXPECTED_PARQUET_SHA256:
        raise ValueError(
            f"FINAL train parquet hash drifted: expected {EXPECTED_PARQUET_SHA256}, got {parquet_hash}"
        )
    frame = pd.read_parquet(train_parquet)
    if len(frame) != EXPECTED_TASKS:
        raise ValueError(f"expected {EXPECTED_TASKS} parquet rows, found {len(frame)}")

    real_targets, rollout_report = collect_real_targets(rollout_root)
    bc_records: list[dict[str, Any]] = []
    dpo_records: list[dict[str, Any]] = []
    task_keys: set[tuple[str, str]] = set()
    source_counts: Counter[str] = Counter()
    source_by_bench: dict[str, Counter[str]] = defaultdict(Counter)
    target_lengths: list[int] = []
    image_rows: list[dict[str, Any]] = []

    for row_index, row in frame.iterrows():
        reward_model = row["reward_model"]
        extra = row["extra_info"]
        key = task_key(reward_model["bench"], reward_model["task_id"])
        if key in task_keys:
            raise ValueError(f"duplicate parquet task: {key}")
        task_keys.add(key)
        messages = normalize_messages(row["prompt"])
        serialized_prompt = "\n".join(message["content"] for message in messages)
        if "<image>" in serialized_prompt:
            image_rows.append({"row": int(row_index), "bench": key[0], "task_id": key[1]})
            continue

        skills = advertised_skills(messages[0]["content"])
        gold_name = str(extra["slate_gold_name"])
        misleading_names = [str(name) for name in as_list(extra["slate_misleading_names"])]
        if len(skills) != 16:
            raise ValueError(f"{key}: expected 16 advertised skills, found {len(skills)}")
        if gold_name not in skills:
            raise ValueError(f"{key}: gold skill {gold_name!r} not in prompt")
        if len(misleading_names) != EXPECTED_MISLEADING_PER_TASK or any(
            name not in skills for name in misleading_names
        ):
            raise ValueError(f"{key}: invalid misleading slate {misleading_names}")

        target = real_targets.get(key)
        source = "real_strict_first_read_gold" if target is not None else "synthetic_from_gold_description"
        if target is None:
            target = render_selection_turn(gold_name, skills[gold_name]["description"])
        parsed = first_tool_call(target)
        if parsed is None or parsed[1] != "read" or skill_name_from_path(parsed[2]) != gold_name:
            raise ValueError(f"{key}: selected target is not first-call read({gold_name})")

        metadata = {
            "bench": key[0],
            "task_id": key[1],
            "source_row": int(row_index),
            "gold_name": gold_name,
            "target_source": source,
        }
        chosen_content = THINK_PREFIX + target
        bc_records.append({
            "messages": messages + [{"role": "assistant", "content": chosen_content}],
            "metadata": metadata,
        })
        source_counts[source] += 1
        source_by_bench[key[0]][source] += 1
        target_lengths.append(len(chosen_content))

        for misleading_name in misleading_names:
            rejected_target = replace_selection(target, skills[gold_name], skills[misleading_name])
            dpo_records.append({
                "messages": messages,
                "chosen": {"role": "assistant", "content": chosen_content},
                "rejected": {"role": "assistant", "content": THINK_PREFIX + rejected_target},
                "metadata": {
                    **metadata,
                    "misleading_name": misleading_name,
                    "pair_protocol": "gold_over_advertised_misleading_same_response_style",
                },
            })

    if image_rows:
        raise ValueError(
            f"FINAL data unexpectedly contains {len(image_rows)} <image> rows; refusing to make a <491 baseline"
        )
    if len(bc_records) != EXPECTED_TASKS or len(dpo_records) != 2455:
        raise ValueError(f"unexpected output counts: BC={len(bc_records)}, DPO={len(dpo_records)}")

    atomic_json(bc_path, bc_records)
    atomic_json(bc_dir / "dataset_info.json", dataset_info(
        args.bc_dataset_name, bc_path.name, ranking=False
    ))
    atomic_json(dpo_path, dpo_records)
    atomic_json(dpo_dir / "dataset_info.json", dataset_info(
        args.dpo_dataset_name, dpo_path.name, ranking=True
    ))
    validation = validate_outputs(
        bc_path, dpo_path, args.bc_dataset_name, args.dpo_dataset_name
    )
    report = {
        "schema_version": 1,
        "protocol": "skillgate_selection_bc_dpo_v1_strict_first_call",
        "inputs": {
            "train_parquet": str(train_parquet),
            "train_parquet_sha256": parquet_hash,
            "rollout_root": str(rollout_root),
        },
        "rollout_scan": rollout_report,
        "counts": {
            "parquet_rows": len(frame),
            "unique_tasks": len(task_keys),
            "image_rows_dropped": len(image_rows),
            "bc_records": len(bc_records),
            "dpo_pairs": len(dpo_records),
            "target_source": dict(source_counts),
            "target_source_by_bench": {
                bench: dict(counts) for bench, counts in sorted(source_by_bench.items())
            },
        },
        "target_chars": {
            "min": min(target_lengths),
            "median": statistics.median(target_lengths),
            "max": max(target_lengths),
        },
        "outputs": {
            "bc": str(bc_path),
            "dpo": str(dpo_path),
            **validation,
        },
    }
    atomic_json(bc_dir / "build_report.json", report)
    atomic_json(dpo_dir / "build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
