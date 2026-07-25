#!/usr/bin/env python3
"""Build the paper-facing outcome and selector tables for Claw147 eval rows."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "GeneralAgent/sft_data_collection"))
from collect_successes import (  # noqa: E402
    SKILL_ENTRY_CAPTURE,
    extract_searchable_texts_with_source,
)


CATEGORIES = ("oracle", "misleading", "relevant", "irrelevant")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    categories: dict[str, dict[str, str]] = {}
    strategies: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = str(row["task_id"])
        categories[task] = {}
        strategies[task] = {}
        for category in CATEGORIES:
            for entry in row.get(category) or []:
                name = str(entry["name"])
                categories[task][name] = category
                if category == "misleading":
                    strategies[task][name] = str(entry.get("strategy") or "unknown")
    return categories, strategies


def load_positions(path: Path) -> dict[str, dict[str, int]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["task_id"])] = {
            str(entry["skill_name"]): index
            for index, entry in enumerate(row.get("reranked_top10") or [], 1)
        }
    return out


def ordered_agent_reads(messages: list[dict[str, Any]]) -> list[str]:
    names = []
    seen = set()
    for source, text in extract_searchable_texts_with_source(messages):
        if source != "agent_tool_call":
            continue
        for match in SKILL_ENTRY_CAPTURE.finditer(text):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def load_trials(root: Path) -> list[dict[str, Any]]:
    trials = []
    for leaf_raw in sorted(glob.glob(str(root / "results/claw/*"))):
        leaf = Path(leaf_raw)
        incremental = leaf / "incremental.jsonl"
        if not incremental.is_file():
            continue
        records = [json.loads(line) for line in incremental.read_text(encoding="utf-8").splitlines() if line]
        if not records:
            continue
        record = records[-1]
        task = str(record.get("task_id") or record.get("instance_id") or leaf.name)
        trajectory_path = leaf / "trajectories" / f"{task}.json"
        messages = []
        if trajectory_path.is_file():
            try:
                messages = json.loads(trajectory_path.read_text(encoding="utf-8")).get("messages") or []
            except (OSError, json.JSONDecodeError, TypeError):
                messages = []
        reads = ordered_agent_reads(messages)
        error_text = " ".join(
            str(record.get(key) or "")
            for key in ("error", "finish_reason", "grade_reason")
        ).lower()
        timeout = "timeout" in error_text or "timed out" in error_text
        loop = any(token in error_text for token in ("loop", "repetition", "repeated", "stuck", "same content"))
        trials.append(
            {
                "task": task,
                "resolved": bool(record.get("resolved")),
                "score": float(record.get("score") or 0.0),
                "reads": reads,
                "has_traj": trajectory_path.is_file(),
                "timeout": timeout,
                "loop": loop,
                "wall_sec": float(record.get("wall_sec") or record.get("time_sec") or 0.0),
                "input_tokens": int(record.get("input_tokens") or 0),
                "output_tokens": int(record.get("output_tokens") or 0),
                "leaf": str(leaf),
            }
        )
    return trials


def rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def summarize(
    trials: list[dict[str, Any]],
    categories: dict[str, dict[str, str]],
    strategies: dict[str, dict[str, str]],
    positions: dict[str, dict[str, int]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    details = {}
    first_positions = []
    category_any = {category: 0 for category in CATEGORIES}
    category_first = {category: 0 for category in CATEGORIES}
    misleading_strategy_reads: dict[str, int] = {}
    misleading_strategy_successes: dict[str, int] = {}
    for trial in trials:
        task = trial["task"]
        category_map = categories.get(task, {})
        read_categories = [category_map.get(name, "unknown") for name in trial["reads"]]
        any_categories = set(read_categories)
        first_name = trial["reads"][0] if trial["reads"] else ""
        first_category = read_categories[0] if read_categories else "none"
        first_position = positions.get(task, {}).get(first_name)
        if first_position is not None:
            first_positions.append(first_position)
        for category in CATEGORIES:
            category_any[category] += int(category in any_categories)
            category_first[category] += int(first_category == category)
        for name in trial["reads"]:
            if category_map.get(name) != "misleading":
                continue
            strategy = strategies.get(task, {}).get(name, "unknown")
            misleading_strategy_reads[strategy] = misleading_strategy_reads.get(strategy, 0) + 1
            misleading_strategy_successes[strategy] = (
                misleading_strategy_successes.get(strategy, 0) + int(trial["resolved"])
            )
        details[task] = {
            **trial,
            "read_categories": read_categories,
            "any_oracle": "oracle" in any_categories,
            "any_misleading": "misleading" in any_categories,
            "first_category": first_category,
            "first_position": first_position,
        }

    n = len(trials)
    resolved = sum(int(trial["resolved"]) for trial in trials)
    strict_read = sum(bool(trial["reads"]) for trial in trials)
    any_oracle_trials = [item for item in details.values() if item["any_oracle"]]
    misleading_only = [
        item for item in details.values()
        if item["any_misleading"] and not item["any_oracle"]
    ]
    no_read = [item for item in details.values() if not item["reads"]]
    categorized_read = [
        item for item in details.values()
        if set(item["read_categories"]) & set(CATEGORIES)
    ]
    summary = {
        "n": n,
        "resolved": resolved,
        "pass_rate": rate(resolved, n),
        "mean_score": statistics.fmean(trial["score"] for trial in trials) if trials else 0.0,
        "strict_read": strict_read,
        "avg_read_names": statistics.fmean(len(trial["reads"]) for trial in trials) if trials else 0.0,
        **{f"any_{category}": category_any[category] for category in CATEGORIES},
        **{f"first_{category}": category_first[category] for category in CATEGORIES},
        "first_unknown": sum(item["first_category"] == "unknown" for item in details.values()),
        "no_read": len(no_read),
        "p_oracle_given_categorized_read": rate(category_any["oracle"], len(categorized_read)),
        "p_success_oracle_read": rate(sum(item["resolved"] for item in any_oracle_trials), len(any_oracle_trials)),
        "misleading_no_oracle": len(misleading_only),
        "p_success_misleading_no_oracle": rate(sum(item["resolved"] for item in misleading_only), len(misleading_only)),
        "p_success_no_read": rate(sum(item["resolved"] for item in no_read), len(no_read)),
        "mean_first_position": statistics.fmean(first_positions) if first_positions else 0.0,
        "first_top1": sum(position == 1 for position in first_positions),
        "first_position_n": len(first_positions),
        "timeout_flags": sum(trial["timeout"] for trial in trials),
        "loop_flags": sum(trial["loop"] for trial in trials),
        "median_wall_sec": statistics.median(trial["wall_sec"] for trial in trials) if trials else 0.0,
        "p90_wall_sec": sorted(trial["wall_sec"] for trial in trials)[min(n - 1, int(n * 0.9))] if trials else 0.0,
        "mean_input_tokens": statistics.fmean(trial["input_tokens"] for trial in trials) if trials else 0.0,
        "mean_output_tokens": statistics.fmean(trial["output_tokens"] for trial in trials) if trials else 0.0,
        "misleading_strategy_reads": misleading_strategy_reads,
        "misleading_strategy_successes": misleading_strategy_successes,
    }
    return summary, details


def paired(primary: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    tasks = sorted(set(primary) & set(other))
    return {
        "n": len(tasks),
        "skillgate_only_pass": sum(primary[t]["resolved"] and not other[t]["resolved"] for t in tasks),
        "other_only_pass": sum(other[t]["resolved"] and not primary[t]["resolved"] for t in tasks),
        "both_pass": sum(primary[t]["resolved"] and other[t]["resolved"] for t in tasks),
        "both_fail": sum(not primary[t]["resolved"] and not other[t]["resolved"] for t in tasks),
        "pass_rate_delta_pp": 100 * rate(sum(primary[t]["resolved"] for t in tasks), len(tasks))
        - 100 * rate(sum(other[t]["resolved"] for t in tasks), len(tasks)),
        "oracle_any_skillgate_only": sum(primary[t]["any_oracle"] and not other[t]["any_oracle"] for t in tasks),
        "oracle_any_other_only": sum(other[t]["any_oracle"] and not primary[t]["any_oracle"] for t in tasks),
        "oracle_first_skillgate_only": sum(primary[t]["first_category"] == "oracle" and other[t]["first_category"] != "oracle" for t in tasks),
        "oracle_first_other_only": sum(other[t]["first_category"] == "oracle" and primary[t]["first_category"] != "oracle" for t in tasks),
    }


def render(payload: dict[str, Any]) -> str:
    models = payload["models"]
    lines = ["# Claw147 Paper Evaluation", "", "## Results", ""]
    lines.extend([
        "| method | task pass@1 | grader mean | explicit read | avg reads | loop / timeout |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for label, value in models.items():
        lines.append(
            f"| {label} | {value['resolved']}/{value['n']} ({pct(value['pass_rate'])}) | "
            f"{value['mean_score']:.3f} | {value['strict_read']}/{value['n']} "
            f"({pct(rate(value['strict_read'], value['n']))}) | {value['avg_read_names']:.2f} | "
            f"{value['loop_flags']} / {value['timeout_flags']} |"
        )
    lines.extend(["", "## Selector Behavior", "", "| method | first oracle | first misleading | any oracle | any misleading | P(oracle | read) | no-read | first position / top-1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for label, value in models.items():
        n = value["n"]
        lines.append(
            f"| {label} | {value['first_oracle']}/{n} ({pct(rate(value['first_oracle'], n))}) | "
            f"{value['first_misleading']}/{n} ({pct(rate(value['first_misleading'], n))}) | "
            f"{value['any_oracle']}/{n} ({pct(rate(value['any_oracle'], n))}) | "
            f"{value['any_misleading']}/{n} ({pct(rate(value['any_misleading'], n))}) | "
            f"{pct(value['p_oracle_given_categorized_read'])} | {value['no_read']}/{n} "
            f"({pct(rate(value['no_read'], n))}) | {value['mean_first_position']:.2f} / "
            f"{value['first_top1']}/{value['first_position_n']} |"
        )
    lines.extend(["", "## Conditional Success", "", "| method | P(success | oracle read) | misleading without oracle: N / success | no-read: N / success |", "|---|---:|---:|---:|"])
    for label, value in models.items():
        lines.append(
            f"| {label} | {pct(value['p_success_oracle_read'])} | "
            f"{value['misleading_no_oracle']} / {pct(value['p_success_misleading_no_oracle'])} | "
            f"{value['no_read']} / {pct(value['p_success_no_read'])} |"
        )
    lines.extend(["", "## Paired Against SkillGate", "", "| comparison | SkillGate only pass | other only pass | both pass | both fail | pass delta | first-oracle only SG / other |", "|---|---:|---:|---:|---:|---:|---:|"])
    for label, value in payload["paired_vs_skillgate"].items():
        lines.append(
            f"| {label} | {value['skillgate_only_pass']} | {value['other_only_pass']} | "
            f"{value['both_pass']} | {value['both_fail']} | {value['pass_rate_delta_pp']:+.1f} pp | "
            f"{value['oracle_first_skillgate_only']} / {value['oracle_first_other_only']} |"
        )
    if payload.get("delta_vs_previous"):
        lines.extend(["", "## Change From Previous Slate Body", "", "| method | pass delta | score delta |", "|---|---:|---:|"])
        for label, value in payload["delta_vs_previous"].items():
            lines.append(f"| {label} | {value['pass_delta_pp']:+.1f} pp | {value['score_delta']:+.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--previous-json", type=Path)
    parser.add_argument("runs", nargs="+", help="label=run_root; SkillGate must be first")
    args = parser.parse_args()

    categories, strategies = load_manifest(args.manifest)
    positions = load_positions(args.snapshot / "claw.jsonl")
    models = {}
    details = {}
    for raw in args.runs:
        label, root = raw.split("=", 1)
        trials = load_trials(Path(root))
        if len(trials) != 147 or len({trial["task"] for trial in trials}) != 147:
            raise SystemExit(f"{label}: expected 147 unique completed trials, found {len(trials)}")
        models[label], details[label] = summarize(trials, categories, strategies, positions)
    labels = list(models)
    primary = labels[0]
    paired_rows = {label: paired(details[primary], details[label]) for label in labels[1:]}
    audit_report_path = args.snapshot.parent / "audit_report.json"
    audit_report = (
        json.loads(audit_report_path.read_text(encoding="utf-8"))
        if audit_report_path.is_file()
        else {}
    )
    payload = {
        "protocol": {
            "tasks": 147,
            "repeats": 1,
            "eval_id": args.eval_id,
            "fingerprint": args.fingerprint,
            "manifest_sha256": sha256_file(args.manifest),
            "snapshot_sha256": sha256_file(args.snapshot / "claw.jsonl"),
            "referenced_skill_content_sha256": audit_report.get(
                "referenced_skill_content_sha256", ""
            ),
        },
        "models": models,
        "paired_vs_skillgate": paired_rows,
    }
    if args.previous_json and args.previous_json.is_file():
        old = json.loads(args.previous_json.read_text(encoding="utf-8")).get("models") or {}
        payload["delta_vs_previous"] = {
            label: {
                "pass_delta_pp": 100 * (value["pass_rate"] - old[label]["pass_rate"]),
                "score_delta": value["mean_score"] - old[label]["mean_score"],
            }
            for label, value in models.items()
            if label in old
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    for label, task_rows in details.items():
        safe_label = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
        trial_path = args.output_json.parent / f"trials_{safe_label}.jsonl"
        trial_path.write_text(
            "".join(
                json.dumps(task_rows[task], ensure_ascii=False, sort_keys=True) + "\n"
                for task in sorted(task_rows)
            ),
            encoding="utf-8",
        )
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render(payload), encoding="utf-8")
    print(render(payload))


if __name__ == "__main__":
    main()
