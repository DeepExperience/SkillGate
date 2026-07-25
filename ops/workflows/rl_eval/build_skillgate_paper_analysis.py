#!/usr/bin/env python3
"""Rebuild the paper-facing SkillGate analyses from immutable artifacts.

This is the single maintained offline analysis entrypoint. It deliberately keeps
three evidence families separate:

* the end-to-end main comparison (SkillGate versus mixed task-only RL),
* alternative selector-credit designs (ablation), and
* prompt/router/retrieval interventions (selector diagnostics).

The script never edits evaluation rows or the paper master document. It refreshes
cached trial projections, structured JSON, figures, and one Markdown fragment for
the analysis section. The paper-facing table is updated separately so hand-curated
paper values are never overwritten by an analysis rebuild.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "ops/workflows/rl_eval"))

from analyze_eval70_3tables import analyze, collect  # noqa: E402
from analyze_slate_reads import load_manifest  # noqa: E402


DEFAULT_OUT = ROOT / "experiments/skillgate_paper/analysis"
EVAL_MANIFEST = ROOT / "skill_libraries/snapshots/rl/eval70_final_v8prod_fixed4/slate_manifest_eval70.jsonl"
TRAIN_MANIFEST = ROOT / "skill_libraries/snapshots/rl/slate_skills_final_hybrid_train/manifest/slate_manifest_train.jsonl"
SFT_MODEL = ROOT / "GeneralAgent/sft_training/merged_models/qwen35_9b_sft_campaign_20260512_clean_plus_claw_thinkwrap_4gpu_49k_5epoch_r32_liger"
ANALYSIS_SCHEMA_VERSION = 2
OUTPUT_SCHEMA_VERSION = 3

SG_OWNER = "selector-clean-oracle-action-credit-sft9b-hybridv8b0704d-20260716_121116"
BASE_OWNER = "mixed-skills-task-reward-v8prod-20260713_185407"
BEHAVIOR_OWNER = "mixedskills-separatedcontinuousadv-v8prod-allgold"
SLATE_OWNER = "slate-regret-v8prod-pair-spec8-eval0"
NONCLEAN_OWNER = "selector-action-credit-v1-sft9b-v8prod-20260714_164200"

ROW_ROOTS = {
    "SkillGate": ROOT / f"experiments/rl/runs/{SG_OWNER}/eval/eval70-mixed-r4-4023950044/rows/clean-oracle-final99-v8-fixed4-5c0e606f",
    "SkillGate_claw_fix": ROOT / f"experiments/rl/runs/{SG_OWNER}/eval/claw14-mixed-r4-oraclefix719/rows/cleanoracle-final99-v8",
    "mixed baseRL": ROOT / f"experiments/rl/runs/{BASE_OWNER}/eval/eval70-mixed-r4-4023950044/rows/v8-task-only-final99-v8-fixed4-ffc91b56",
    # This completed row is the artifact whose counts match the manually locked
    # paper registration (118/280). It is also used for route-level analyses.
    "mixed baseRL_paper": ROOT / f"experiments/rl/runs/{BASE_OWNER}/eval/eval70-mixed-r4-2d51f4da2b/rows/v8_task_only_final_iter99-ffc91b56",
    "Slate Regret v2": ROOT / f"experiments/rl/runs/{SLATE_OWNER}/eval/eval70-mixed-r4-4023950044/rows/slate-regret-v2-final99-v8-fixed4-25b8d793",
    "behavior bonus": ROOT / f"experiments/rl/runs/{BEHAVIOR_OWNER}/eval/eval70-mixed-r4-4023950044/rows/behavior-bonus-final99-v8-fixed4-ab820a9f",
    "non-clean action credit": ROOT / f"experiments/rl/runs/{NONCLEAN_OWNER}/eval/eval70-mixed-r4-4023950044/rows/action-credit-nonclean-final99-v8-fixed4-d0336c42",
}

DPO_ROW_ROOT = ROOT / (
    "experiments/rl/runs/selskill-dpo-selection-sft9b-20260721/eval/"
    "eval70-mixed-r4-4023950044/rows/selskill-dpo-sft9b-v8-fixed4-d6826436"
)

CLAW147_ANALYSIS_JSON = ROOT / (
    "experiments/skill_slate_eval/claw147_main_models_20260722/analysis/"
    "claw147_all_main_models_analysis.json"
)
CLAW147_TASK_TRIALS = ROOT / (
    "experiments/skill_slate_eval/claw147_main_models_20260722/analysis/"
    "trials_SkillGate.jsonl"
)

# These are paper-registered T1 values supplied from the completed evaluation
# on the other machine. Do not silently replace them with a locally discovered
# historical row. Behavior analyses continue to use immutable local trajectories.
PAPER_T1_COUNTS = {
    "base27b": (116, 35, 3, 51, 16, 11),
    "base9b": (72, 34, 0, 29, 6, 3),
    "SFT9B": (104, 29, 2, 48, 18, 7),
    "no-skill RL": (116, 29, 3, 57, 17, 10),
    "mixed baseRL": (118, 29, 1, 60, 18, 10),
    "Gold Selector BC": (116, 28, 5, 52, 20, 11),
    "SelSkill DPO": (124, 31, 0, 57, 24, 12),
    "masked-task-only": (117, 28, 3, 58, 18, 10),
    "SkillGate": (140, 32, 5, 65, 26, 12),
    "Slate Regret v2": (117, 27, 2, 57, 21, 10),
    "behavior bonus": (117, 27, 2, 59, 19, 8),
    "non-clean action credit": (126, 29, 5, 57, 20, 15),
}

PAPER_T1_DENOMINATORS = (280, 56, 32, 120, 40, 32)

UNIFIED_METHOD_ORDER = (
    "base27b",
    "base9b",
    "SFT9B",
    "no-skill RL",
    "mixed baseRL",
    "Gold Selector BC",
    "SelSkill DPO",
    "masked-task-only",
    "SkillGate",
    "Slate Regret v2",
    "behavior bonus",
    "non-clean action credit",
)

UNIFIED_EVAL70_CLAW_ROWS = {
    "base27b": ROOT / (
        "experiments/rl/runs/reference-qwen3-5-27b/eval/"
        "eval70-mixed-r4-4023950044/rows/base27b-v8-fixed4-41f6b6f7"
    ),
    "base9b": ROOT / (
        "experiments/rl/runs/reference-qwen3-5-9b-base/eval/"
        "eval70-mixed-r4-4023950044/rows/base9b-v8-fixed4-3e3c0499"
    ),
    "SFT9B": ROOT / (
        "experiments/rl/runs/reference-qwen3-5-9b-sft/eval/"
        "eval70-mixed-r4-4023950044/rows/sft9b-v8-fixed4-83cec000"
    ),
    "no-skill RL": ROOT / (
        "experiments/rl/runs/noskills/eval/eval70-mixed-r4-4023950044/rows/"
        "no-skill-rl-final99-v8-fixed4-3add6bb6"
    ),
    "mixed baseRL": ROW_ROOTS["mixed baseRL_paper"],
    "Gold Selector BC": ROOT / (
        "experiments/rl/runs/goldbc-selection-sft9b-20260721/eval/"
        "eval70-mixed-r4-4023950044/rows/goldbc-sft9b-v8-fixed4-16b384ed"
    ),
    "SelSkill DPO": DPO_ROW_ROOT,
    "masked-task-only": ROOT / (
        "experiments/rl/runs/selector-clean-oracle-maskedtaskonly-sft9b-finalhybrid-lr1e6-20260721_022408/eval/"
        "eval70-mixed-r4-4023950044/rows/masked-task-only-final99-07f3e763"
    ),
    "SkillGate": ROW_ROOTS["SkillGate_claw_fix"],
    "Slate Regret v2": ROW_ROOTS["Slate Regret v2"],
    "behavior bonus": ROW_ROOTS["behavior bonus"],
    "non-clean action credit": ROW_ROOTS["non-clean action credit"],
}

CLAW147_MODEL_LABELS = {
    "base27b": "base27b",
    "base9b": "base9b",
    "SFT9B": "SFT9B",
    "no-skill RL": "no-skill-RL",
    "mixed baseRL": "mixed-baseRL",
    "Gold Selector BC": "Gold-Selector-BC",
    "SelSkill DPO": "SelSkill-DPO",
    "masked-task-only": "masked-task-only-final99",
    "SkillGate": "SkillGate",
    "Slate Regret v2": "Slate-Regret-v2-final99",
    "behavior bonus": "behavior-bonus-final99",
    "non-clean action credit": "action-credit-nonclean",
}

SELECTOR_ROW_ROOTS = {
    "SFT9B + prompt": ROOT / "experiments/rl/runs/reference-qwen3-5-9b-sft/eval/eval70_v1-mixed-r4-8f9155db1e/rows/sft9b-prompt-select-one-v8-fixed4-83cec000",
    "SFT9B router": ROOT / "experiments/rl/runs/reference-qwen3-5-9b-sft/eval/eval70_v1-retrieve-r4-e6a4b50a9c/rows/sft9b-router-select-one-v8-fixed4-83cec000",
    "27B router": ROOT / "experiments/rl/runs/reference-qwen3-5-9b-sft/eval/eval70_v1-retrieve-r4-af26f8361f/rows/qwen27b-router-select-one-v8-fixed4-83cec000",
    "Qwen3 reranker": ROOT / "experiments/rl/runs/reference-qwen3-5-9b-sft/eval/eval70_v1-retrieve-r4-8e24f80215/rows/qwen3-reranker-top1-v8-fixed4-83cec000",
    "oracle-only SFT9B": ROOT / "experiments/rl/runs/reference-qwen3-5-9b-sft/eval/eval70_v1-retrieve-r4-41ac93f8cd/rows/oracle-only-sft9b-83cec000",
    "oracle-only SkillGate": ROOT / f"experiments/rl/runs/{SG_OWNER}/eval/eval70_v1-retrieve-r4-41ac93f8cd/rows/oracle-only-skillgate-5c0e606f",
}

ROUTE_SUMMARIES = {
    "SFT9B router": ROOT / "experiments/skillgate_paper/routes/sft9b_router_eval70/summary.json",
    "27B router": ROOT / "experiments/skillgate_paper/routes/qwen27b_router_eval70/summary.json",
    "Qwen3 reranker": ROOT / "experiments/skillgate_paper/routes/qwen3_reranker_eval70/summary.json",
}

NOSKILL_RL_MIXED_ROW = ROOT / (
    "experiments/rl/runs/noskills/eval/eval70-mixed-r4-4023950044/rows/"
    "no-skill-rl-final99-v8-fixed4-3add6bb6"
)

BASE_TRAIN_SEGMENTS = [
    ROOT / f"experiments/rl/runs/{BASE_OWNER}/segments/20260713_185858-guardfix-retry/rollout_result/train",
]
CLEAN_TRAIN_SEGMENTS = [
    ROOT / f"experiments/rl/runs/{SG_OWNER}/segments/20260716_121116-initial/rollout_result/train",
    ROOT / f"experiments/rl/runs/{SG_OWNER}/segments/20260718_1230-resume74-restart/rollout_result/train",
    ROOT / f"experiments/rl/runs/{SG_OWNER}/segments/20260718_1243-resume74-evalcontract/rollout_result/train",
]

BENCH_TO_MANIFEST = {"seta": "seta_synth", "swe": "swe_lite", "tb2": "tb2", "sb_ns": "sb_ns", "claw": "claw"}
BENCH_ORDER = ("claw", "sb_ns", "seta", "swe", "tb2")
FIRST_CATEGORIES = ("oracle", "misleading", "other", "no_read")
COLORS = {
    "SkillGate": "#087E8B",
    "mixed baseRL": "#6C757D",
    "Slate Regret v2": "#E9C46A",
    "behavior bonus": "#E76F51",
    "non-clean action credit": "#457B9D",
    "SelSkill DPO": "#7B6D8D",
    "oracle": "#087E8B",
    "misleading": "#D1495B",
    "other": "#7A8A93",
    "no_read": "#C7CDD1",
}

JSON_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
XML_TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
SKILL_PATH_RE = re.compile(r"(?:/root|~)?/\.claude/skills/([A-Za-z0-9_.-]+)/(?:SKILL|README)\.md", re.IGNORECASE)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


def pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def pp(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:+.{digits}f} pp"


def signature(paths: Iterable[Path]) -> str:
    rows = []
    for path in sorted(set(paths)):
        stat = path.stat()
        rows.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    payload = {"analysis_schema": ANALYSIS_SCHEMA_VERSION, "files": rows}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cached_trials(label: str, root: Path, cache_dir: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    cache = cache_dir / f"trials_{label}.jsonl"
    stamp = cache.with_suffix(".meta.json")
    inputs = sorted(root.glob("results/*/*/incremental.jsonl"))
    current = signature(inputs)
    try:
        meta = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        meta = {}
    if cache.is_file() and meta.get("signature") == current:
        return [json.loads(line) for line in cache.open(encoding="utf-8") if line.strip()]
    trials = collect(str(root))
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(f".{cache.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for trial in trials:
            handle.write(json.dumps(trial, ensure_ascii=False) + "\n")
    os.replace(tmp, cache)
    write_json(stamp, {"signature": current, "root": str(root), "records": len(trials)})
    return trials


def load_claw_repeat_zero(row_root: Path) -> list[dict[str, Any]]:
    """Load the explicitly planned t00 result for each of the 14 FINAL70 Claw tasks."""

    plans = sorted((row_root / "plans").glob("*.jsonl"))
    if len(plans) != 1:
        raise ValueError(f"{row_root}: expected one JSONL plan, found {len(plans)}")
    entries = [
        json.loads(line)
        for line in plans[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [
        entry
        for entry in entries
        if entry.get("bench") == "claw" and int(entry.get("trial_index", -1)) == 0
    ]
    if len(selected) != 14 or len({str(entry["task_id"]) for entry in selected}) != 14:
        raise ValueError(f"{row_root}: expected 14 unique Claw t00 plan entries")

    trials = []
    for entry in selected:
        incremental = Path(str(entry["incremental_path"]))
        if not incremental.is_absolute():
            incremental = ROOT / incremental
        records = [
            json.loads(line)
            for line in incremental.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            raise ValueError(f"{incremental}: no result records")
        record = records[-1]
        task = str(record.get("task_id") or record.get("instance_id") or "")
        if task != str(entry["task_id"]):
            raise ValueError(f"{incremental}: plan/result task mismatch")
        trials.append(
            {
                "task": task,
                "resolved": bool(record.get("resolved")),
                "incremental_path": str(incremental.relative_to(ROOT)),
            }
        )
    return sorted(trials, key=lambda row: row["task"])


def build_unified_claw161_table() -> dict[str, Any]:
    """Combine registered non-Claw x4 counts with Claw14 t00 and Claw147."""

    claw147 = json.loads(CLAW147_ANALYSIS_JSON.read_text(encoding="utf-8"))
    claw147_models = claw147.get("models") or {}
    claw147_trials = [
        json.loads(line)
        for line in CLAW147_TASK_TRIALS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    claw147_tasks = {str(row["task"]) for row in claw147_trials}
    if len(claw147_trials) != 147 or len(claw147_tasks) != 147:
        raise ValueError("Claw147 analysis must contain 147 unique task trials")

    models = {}
    for label in UNIFIED_METHOD_ORDER:
        repeat_zero = load_claw_repeat_zero(UNIFIED_EVAL70_CLAW_ROWS[label])
        claw14_tasks = {row["task"] for row in repeat_zero}
        overlap = claw14_tasks & claw147_tasks
        if overlap:
            raise ValueError(f"{label}: Claw14 and Claw147 overlap: {sorted(overlap)}")

        claw147_label = CLAW147_MODEL_LABELS[label]
        claw147_row = claw147_models.get(claw147_label)
        if not claw147_row or int(claw147_row.get("n", 0)) != 147:
            raise ValueError(f"{label}: missing complete Claw147 result")

        registered = PAPER_T1_COUNTS[label]
        by_bench = {
            "sb_ns": int(registered[2]),
            "seta": int(registered[3]),
            "swe": int(registered[4]),
            "tb2": int(registered[5]),
        }
        claw14_success = sum(int(row["resolved"]) for row in repeat_zero)
        claw147_success = int(claw147_row["resolved"])
        claw161_success = claw14_success + claw147_success
        all_success = sum(by_bench.values()) + claw161_success
        if not 0 <= all_success <= 385 or not 0 <= claw161_success <= 161:
            raise ValueError(f"{label}: invalid unified counts")
        models[label] = {
            "all": {"success": all_success, "trials": 385},
            "claw161": {"success": claw161_success, "trials": 161},
            "sb_ns": {"success": by_bench["sb_ns"], "trials": 32},
            "seta": {"success": by_bench["seta"], "trials": 120},
            "swe": {"success": by_bench["swe"], "trials": 40},
            "tb2": {"success": by_bench["tb2"], "trials": 32},
            "claw14_repeat0_success": claw14_success,
            "claw147_success": claw147_success,
            "eval70_claw_row": str(UNIFIED_EVAL70_CLAW_ROWS[label].relative_to(ROOT)),
            "claw14_repeat0_tasks": repeat_zero,
        }

    return {
        "protocol": {
            "total_trials": 385,
            "non_claw_trials": 224,
            "claw_trials": 161,
            "non_claw_rule": "56 FINAL70 non-Claw tasks x 4 repeats; paper-registered counts",
            "claw_rule": "14 FINAL70 Claw tasks at trial_index=0 plus 147 corrected Claw tasks x 1",
            "claw147_eval_id": claw147.get("protocol", {}).get("eval_id"),
            "claw147_fingerprint": claw147.get("protocol", {}).get("fingerprint"),
        },
        "models": models,
    }


def replace_claw(full: Sequence[dict[str, Any]], claw: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = [dict(row) for row in full if row["bench"] != "claw"] + [dict(row) for row in claw]
    if len(merged) != 280 or sum(row["bench"] == "claw" for row in merged) != 56:
        raise ValueError(f"invalid paper merge: records={len(merged)}")
    return merged


def load_paper_trials(cache_dir: Path) -> dict[str, list[dict[str, Any]]]:
    data = {
        "SkillGate": replace_claw(
            cached_trials("sg_full", ROW_ROOTS["SkillGate"], cache_dir),
            cached_trials("sg_claw_fix", ROW_ROOTS["SkillGate_claw_fix"], cache_dir),
        ),
        "mixed baseRL": cached_trials("base_paper_118", ROW_ROOTS["mixed baseRL_paper"], cache_dir),
        "Slate Regret v2": cached_trials("slate_regret_fresh", ROW_ROOTS["Slate Regret v2"], cache_dir),
        "behavior bonus": cached_trials("behavior_fresh", ROW_ROOTS["behavior bonus"], cache_dir),
        "non-clean action credit": cached_trials("nonclean_fresh", ROW_ROOTS["non-clean action credit"], cache_dir),
    }
    for label, trials in data.items():
        tasks = {(row["bench"], row["task"]) for row in trials}
        if len(trials) != 280 or len(tasks) != 70:
            raise ValueError(f"{label}: expected 280 trials/70 tasks, got {len(trials)}/{len(tasks)}")
    return data


def by_task(trials: Sequence[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    output: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        output[(row["bench"], row["task"])].append(row)
    return output


def task_manifest_key(trial: dict[str, Any]) -> str:
    bench = BENCH_TO_MANIFEST.get(str(trial["bench"]), str(trial["bench"]))
    return f"{bench}::{trial['task']}"


def read_info(trial: dict[str, Any], manifest: dict[str, dict[str, str]]) -> dict[str, Any]:
    names = [str(name) for name in (trial.get("read_names_agent") or [])]
    category_map = manifest.get(task_manifest_key(trial), {})
    raw_categories = [str(category_map.get(name) or "other") for name in names]
    categories = [category if category in {"oracle", "misleading"} else "other" for category in raw_categories]
    return {
        "names": names,
        "categories": categories,
        "raw_categories": raw_categories,
        "first_name": names[0] if names else "__NO_READ__",
        "first_category": categories[0] if categories else "no_read",
        "first_raw_category": raw_categories[0] if raw_categories else "no_read",
        "oracle_exposure": "oracle" in categories,
        "misleading_exposure": "misleading" in categories,
        "clean_oracle": len(names) == 1 and categories == ["oracle"],
    }


def method_metrics(trials: Sequence[dict[str, Any]], manifest: dict[str, dict[str, str]]) -> dict[str, Any]:
    result = analyze(list(trials))
    infos = [read_info(row, manifest) for row in trials]
    any_read = sum(bool(info["names"]) for info in infos)
    oracle = sum(info["oracle_exposure"] for info in infos)
    misleading = sum(info["misleading_exposure"] for info in infos)
    first = {category: sum(info["first_category"] == category for info in infos) for category in FIRST_CATEGORIES}
    return {
        "trials": len(trials),
        "tasks": result["t1_task"]["ALL"][1],
        "trial_success": result["t1"]["ALL"][0],
        "task_pass4": result["t1_task"]["ALL"][0],
        "by_bench": {bench: result["t1"][bench][0] for bench in BENCH_ORDER},
        "any_read": any_read,
        "oracle_exposure": oracle,
        "misleading_exposure": misleading,
        "oracle_given_read": oracle / any_read if any_read else 0.0,
        "avg_unique_reads": float(np.mean([len(set(info["names"])) for info in infos])),
        "clean_oracle": sum(info["clean_oracle"] for info in infos),
        "first_category": first,
    }


def apply_paper_t1(metrics: dict[str, Any], label: str) -> dict[str, Any]:
    """Replace only task-outcome fields with the locked paper registration."""

    values = PAPER_T1_COUNTS[label]
    updated = dict(metrics)
    updated["trial_success"] = values[0]
    updated["by_bench"] = dict(zip(BENCH_ORDER, values[1:], strict=True))
    updated["task_outcome_source"] = "manual paper registration"
    return updated


def build_main_results_figure(out: Path) -> dict[str, Any]:
    """Plot the manually locked main-table outcomes without re-reading eval rows."""

    import matplotlib.pyplot as plt

    methods = (
        "base27b", "base9b", "SFT9B", "no-skill RL", "mixed baseRL",
        "Gold Selector BC", "SelSkill DPO", "masked-task-only", "SkillGate",
    )
    display = (
        "Base 27B", "Base 9B", "SFT 9B", "No-skill RL", "Mixed baseRL",
        "Gold selector BC", "SelSkill DPO", "Masked task-only", "SkillGate",
    )
    values = np.asarray([PAPER_T1_COUNTS[label] for label in methods], dtype=float)
    rates = 100.0 * values / np.asarray(PAPER_T1_DENOMINATORS, dtype=float)
    colors = [COLORS["SkillGate"] if label == "SkillGate" else "#89959B" for label in methods]

    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2, figsize=(12.8, 5.1), gridspec_kw={"width_ratios": [1.05, 1.55]},
    )
    y = np.arange(len(methods))
    ax_bar.barh(y, rates[:, 0], color=colors, height=0.68)
    ax_bar.set_yticks(y, display)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Trial success (%)")
    ax_bar.set_title("Overall")
    ax_bar.set_xlim(0, max(rates[:, 0]) + 8)
    for index, value in enumerate(rates[:, 0]):
        ax_bar.text(value + 0.7, index, f"{value:.1f}", va="center", fontsize=8)
    ax_bar.grid(axis="x", color="#E5E5E5", linewidth=0.7)
    ax_bar.spines[["top", "right"]].set_visible(False)

    image = ax_heat.imshow(rates[:, 1:], cmap="YlGnBu", vmin=0, vmax=70, aspect="auto")
    ax_heat.set_xticks(np.arange(5), ("Claw", "SB", "SETA", "SWE", "TB2"))
    ax_heat.set_yticks(np.arange(len(methods)), display)
    ax_heat.set_title("Per-benchmark trial success (%)")
    for row in range(rates.shape[0]):
        for column in range(1, rates.shape[1]):
            value = rates[row, column]
            ax_heat.text(column - 1, row, f"{value:.1f}", ha="center", va="center", fontsize=7.5,
                         color="white" if value >= 45 else "#1F2D33")
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.035, pad=0.03)
    colorbar.set_label("Success (%)")
    fig.suptitle("Main result: standard mixed-slate evaluation (280 trials)", fontsize=12)
    fig.tight_layout()
    figure = out / "figures/section1_main_results.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "methods": {label: list(PAPER_T1_COUNTS[label]) for label in methods},
        "source": "manual paper registration",
        "figure": str(figure.relative_to(ROOT)),
    }


def build_ablation(
    data: dict[str, list[dict[str, Any]]],
    manifest: dict[str, dict[str, str]],
    out: Path,
) -> dict[str, Any]:
    methods = ("mixed baseRL", "Slate Regret v2", "behavior bonus", "non-clean action credit", "SkillGate")
    metrics = {
        label: apply_paper_t1(method_metrics(data[label], manifest), label)
        for label in methods
    }

    import matplotlib.pyplot as plt

    display = ["Task-only", "Slate\nRegret", "Trajectory\nbonus", "Non-clean\naction", "SkillGate"]
    colors = [COLORS[label] for label in methods]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    trial = [100 * metrics[label]["trial_success"] / metrics[label]["trials"] for label in methods]
    axes[0].bar(x, trial, color=colors, width=0.72)
    axes[0].set_title("Task outcome")
    axes[0].set_ylabel("Trial success (%)")
    axes[0].set_ylim(0, max(trial) + 8)
    for idx, value in enumerate(trial):
        axes[0].text(idx, value + 0.8, f"{value:.1f}", ha="center", fontsize=8)

    oracle = [100 * metrics[label]["oracle_exposure"] / metrics[label]["trials"] for label in methods]
    misleading = [100 * metrics[label]["misleading_exposure"] / metrics[label]["trials"] for label in methods]
    width = 0.34
    axes[1].bar(x - width / 2, oracle, width, label="Oracle", color=COLORS["oracle"])
    axes[1].bar(x + width / 2, misleading, width, label="Misleading", color=COLORS["misleading"])
    axes[1].set_title("Observed read exposure")
    axes[1].set_ylabel("Trials (%)")
    axes[1].set_ylim(0, 105)
    axes[1].legend(frameon=False, fontsize=8)

    reads = [metrics[label]["avg_unique_reads"] for label in methods]
    axes[2].bar(x, reads, color=colors, width=0.72)
    axes[2].set_title("Read volume")
    axes[2].set_ylabel("Unique skills / trial")
    axes[2].set_ylim(0, max(reads) + 0.35)
    for idx, value in enumerate(reads):
        axes[2].text(idx, value + 0.04, f"{value:.2f}", ha="center", fontsize=8)

    for ax in axes:
        ax.set_xticks(x, display)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Ablation: where oracle-reading credit is assigned", fontsize=12, y=1.02)
    fig.tight_layout()
    figure = out / "figures/section4_ablation_credit_designs.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"methods": metrics, "figure": str(figure.relative_to(ROOT))}


def build_selector_diagnostics(
    cache_dir: Path,
    manifest: dict[str, dict[str, str]],
    data: dict[str, list[dict[str, Any]]],
    out: Path,
) -> dict[str, Any]:
    metrics = {}
    for label, root in SELECTOR_ROW_ROOTS.items():
        trials = cached_trials(f"selector_{label.lower().replace(' ', '_').replace('-', '_')}", root, cache_dir)
        if len(trials) != 280:
            raise ValueError(f"{label}: expected 280 trials, got {len(trials)}")
        metrics[label] = method_metrics(trials, manifest)
    metrics["SkillGate"] = apply_paper_t1(method_metrics(data["SkillGate"], manifest), "SkillGate")

    routes = {}
    for label, path in ROUTE_SUMMARIES.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("selected_count") != 70:
            raise ValueError(f"incomplete route summary: {path}")
        routes[label] = {
            "tasks": int(payload["selected_count"]),
            "oracle_rate": float(payload["oracle_selection_rate"]),
            "category_counts": payload["category_counts"],
        }

    skillgate_first = [read_info(row, manifest)["first_raw_category"] for row in data["SkillGate"]]
    skillgate_counts = {
        category: sum(value == category for value in skillgate_first)
        for category in ("oracle", "misleading", "relevant", "irrelevant", "no_read", "other")
    }
    routes["SkillGate"] = {
        "tasks": len(skillgate_first),
        "oracle_rate": skillgate_counts["oracle"] / len(skillgate_first),
        "category_counts": skillgate_counts,
        "unit": "on-policy trial first attributed read",
    }

    import matplotlib.pyplot as plt

    points = (
        ("SFT9B router", "SFT9B router"),
        ("27B router", "27B router"),
        ("Qwen3 reranker", "Qwen3 reranker"),
        ("SkillGate", "SkillGate"),
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for route_label, metric_label in points:
        x = 100 * routes[route_label]["oracle_rate"]
        y = 100 * metrics[metric_label]["trial_success"] / metrics[metric_label]["trials"]
        color = COLORS["SkillGate"] if route_label == "SkillGate" else COLORS["non-clean action credit"]
        marker = "*" if route_label == "SkillGate" else "o"
        size = 140 if route_label == "SkillGate" else 70
        ax.scatter(x, y, s=size, marker=marker, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        offset = (-4, 8) if route_label == "27B router" else (5, 6)
        ax.annotate(route_label, (x, y), xytext=offset, textcoords="offset points", fontsize=9)
    ax.set_xlabel("Oracle top-1 / agent first-read (%)")
    ax.set_ylabel("End-to-end trial success (%)")
    ax.set_title("Selection accuracy and executor outcome are separate")
    ax.set_xlim(20, 84)
    ax.set_ylim(27, 53)
    ax.grid(color="#E5E5E5", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    figure = out / "figures/section5_selector_tradeoff.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"rows": metrics, "routes": routes, "figure": str(figure.relative_to(ROOT))}


def bootstrap_delta(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
    *,
    bench: str | None,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    left_tasks, right_tasks = by_task(left), by_task(right)
    keys = sorted(set(left_tasks) & set(right_tasks))
    if bench is not None:
        keys = [key for key in keys if key[0] == bench]
    if not keys:
        raise ValueError(f"no paired tasks for bench={bench}")
    trial_left = np.array([np.mean([row["resolved"] for row in left_tasks[key]]) for key in keys], dtype=float)
    trial_right = np.array([np.mean([row["resolved"] for row in right_tasks[key]]) for key in keys], dtype=float)
    pass_left = (trial_left > 0).astype(float)
    pass_right = (trial_right > 0).astype(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(keys), size=(samples, len(keys)))
    trial_boot = np.mean((trial_left - trial_right)[indices], axis=1)
    pass_boot = np.mean((pass_left - pass_right)[indices], axis=1)

    def interval(values: np.ndarray, boot: np.ndarray) -> dict[str, float]:
        low, high = np.quantile(boot, [0.025, 0.975])
        return {"delta": float(np.mean(values)), "ci_low": float(low), "ci_high": float(high)}

    return {
        "bench": bench or "ALL",
        "tasks": len(keys),
        "trial_success": interval(trial_left - trial_right, trial_boot),
        "task_pass4": interval(pass_left - pass_right, pass_boot),
    }


def build_main_bootstrap(
    data: dict[str, list[dict[str, Any]]],
    out: Path,
    samples: int,
) -> dict[str, Any]:
    rows = [
        bootstrap_delta(
            data["SkillGate"], data["mixed baseRL"], bench=bench,
            samples=samples, seed=20260722 + index,
        )
        for index, bench in enumerate((None, *BENCH_ORDER))
    ]

    import matplotlib.pyplot as plt

    display_names = ["ALL", "Claw", "SB", "SETA", "SWE", "TB2"]
    labels = [f"{label} (n={row['tasks']})" for label, row in zip(display_names, rows, strict=True)]
    centers = np.array([row["trial_success"]["delta"] * 100 for row in rows])
    lows = np.array([row["trial_success"]["ci_low"] * 100 for row in rows])
    highs = np.array([row["trial_success"]["ci_high"] * 100 for row in rows])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.errorbar(centers, y, xerr=[centers - lows, highs - centers], fmt="o", color=COLORS["SkillGate"], ecolor="#4C5B61", capsize=3)
    ax.axvline(0, color="#777777", linewidth=1, linestyle="--")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Trial-success difference: SkillGate - mixed baseRL (percentage points)")
    ax.set_title(f"Task-cluster paired bootstrap ({samples:,} resamples, 95% CI)")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    figure = out / "figures/section6_1_paired_bootstrap.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"bootstrap_samples": samples, "seed": 20260722, "rows": rows, "figure": str(figure.relative_to(ROOT))}


def build_matched_route_audit(
    data: dict[str, list[dict[str, Any]]],
    manifest: dict[str, dict[str, str]],
    cache_dir: Path,
    out: Path,
    samples: int,
) -> dict[str, Any]:
    """Compare executor outcomes only on full ordered read sequences shared by all methods."""

    methods = ("SkillGate", "mixed baseRL", "SelSkill DPO")
    trials = {
        "SkillGate": data["SkillGate"],
        "mixed baseRL": data["mixed baseRL"],
        "SelSkill DPO": cached_trials("selskill_dpo_full_sequence", DPO_ROW_ROOT, cache_dir),
    }
    for label, rows in trials.items():
        if len(rows) != 280:
            raise ValueError(f"{label}: expected 280 trials, got {len(rows)}")

    grouped: dict[str, dict[tuple[Any, ...], list[dict[str, Any]]]] = {}
    for label, rows in trials.items():
        strata: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            route = tuple(read_info(row, manifest)["names"])
            strata[(row["bench"], row["task"], route)].append(row)
        grouped[label] = strata

    shared = sorted(set.intersection(*(set(grouped[label]) for label in methods)), key=str)
    tasks = sorted({key[:2] for key in shared})
    if not tasks:
        raise ValueError("no full ordered read sequence shared by all three methods")

    task_values = np.zeros((len(tasks), len(methods)), dtype=float)
    for task_index, task in enumerate(tasks):
        task_keys = [key for key in shared if key[:2] == task]
        for method_index, label in enumerate(methods):
            stratum_rates = [
                float(np.mean([bool(row["resolved"]) for row in grouped[label][key]]))
                for key in task_keys
            ]
            task_values[task_index, method_index] = float(np.mean(stratum_rates))

    rng = np.random.default_rng(20260723 + 202)
    indices = rng.integers(0, len(tasks), size=(samples, len(tasks)))
    boot_rates = np.mean(task_values[indices], axis=1)
    method_rows = {}
    for method_index, label in enumerate(methods):
        raw_rows = [row for key in shared for row in grouped[label][key]]
        low, high = np.quantile(boot_rates[:, method_index], [0.025, 0.975])
        method_rows[label] = {
            "success": sum(bool(row["resolved"]) for row in raw_rows),
            "n": len(raw_rows),
            "raw_rate": float(np.mean([bool(row["resolved"]) for row in raw_rows])),
            "task_macro_rate": float(np.mean(task_values[:, method_index])),
            "ci_low": float(low),
            "ci_high": float(high),
        }

    comparisons = {}
    skillgate_index = methods.index("SkillGate")
    for label in ("mixed baseRL", "SelSkill DPO"):
        other_index = methods.index(label)
        delta_values = task_values[:, skillgate_index] - task_values[:, other_index]
        delta_boot = boot_rates[:, skillgate_index] - boot_rates[:, other_index]
        low, high = np.quantile(delta_boot, [0.025, 0.975])
        comparisons[f"SkillGate - {label}"] = {
            "delta": float(np.mean(delta_values)),
            "ci_low": float(low),
            "ci_high": float(high),
        }

    import matplotlib.pyplot as plt

    centers = np.asarray([100 * method_rows[label]["task_macro_rate"] for label in methods])
    lows = np.asarray([100 * method_rows[label]["ci_low"] for label in methods])
    highs = np.asarray([100 * method_rows[label]["ci_high"] for label in methods])
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars = ax.bar(x, centers, color=[COLORS[label] for label in methods], width=0.62)
    ax.errorbar(x, centers, yerr=[centers - lows, highs - centers], fmt="none", ecolor="#36454D", capsize=4)
    ax.set_xticks(x, ("SkillGate", "Mixed baseRL", "SelSkill DPO"))
    ax.set_ylabel("Task-macro success (%)")
    ax.set_title(f"Same full ordered read sequence ({len(shared)} strata, {len(tasks)} tasks)")
    ax.set_ylim(0, max(highs) + 12)
    for bar, label in zip(bars, methods, strict=True):
        row = method_rows[label]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f"{100 * row['task_macro_rate']:.1f}%\nN={row['n']}", ha="center", fontsize=8)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    figure = out / "figures/section6_2_same_full_sequence.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "trials_per_method": 280,
        "strata": len(shared),
        "tasks": len(tasks),
        "bootstrap_samples": samples,
        "route_definition": "(benchmark, task, ordered attributed read-name sequence), including the empty no-read sequence",
        "repeat_pairing": "Independent repeats have no stable cross-model repeat id; no artificial one-to-one trial matching is used.",
        "methods": method_rows,
        "comparisons": comparisons,
        "figure": str(figure.relative_to(ROOT)),
    }


def build_noskill_route_strata(
    cache_dir: Path,
    manifest: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Audit outcomes after the no-skill RL policy naturally chooses a route.

    These are post-selection strata, not randomized exposure arms. Keeping that
    distinction in the structured result prevents the paper from treating a
    conditional association as a causal intervention.
    """

    trials = cached_trials("noskill_rl_final_mixed_route_strata", NOSKILL_RL_MIXED_ROW, cache_dir)
    if len(trials) != 280 or len(by_task(trials)) != 70:
        raise ValueError(f"no-skill RL mixed row: expected 280 trials/70 tasks, got {len(trials)}/{len(by_task(trials))}")

    strata: dict[str, list[dict[str, Any]]] = {
        "oracle-only": [],
        "misleading-only": [],
        "no-read": [],
    }
    excluded: dict[str, int] = defaultdict(int)
    for row in trials:
        info = read_info(row, manifest)
        categories = set(info["raw_categories"])
        if not info["names"]:
            strata["no-read"].append(row)
        elif categories == {"oracle"}:
            strata["oracle-only"].append(row)
        elif categories == {"misleading"}:
            strata["misleading-only"].append(row)
        elif "oracle" in categories:
            excluded["oracle-plus-other"] += 1
        elif "misleading" in categories:
            excluded["misleading-plus-other"] += 1
        else:
            excluded["relevant-or-irrelevant-only"] += 1

    conditions = {}
    for label, rows in strata.items():
        conditions[label] = {
            "success": sum(bool(row["resolved"]) for row in rows),
            "trials": len(rows),
            "tasks_represented": len(by_task(rows)),
            "rate": float(np.mean([bool(row["resolved"]) for row in rows])),
        }
    return {
        "source_row": str(NOSKILL_RL_MIXED_ROW.relative_to(ROOT)),
        "protocol": "FINAL70 x4 standard mixed-slate; natural policy-selected route strata",
        "total_trials": len(trials),
        "conditions": conditions,
        "excluded_trials": dict(excluded),
        "excluded_total": sum(excluded.values()),
        "causal_intervention": False,
    }


def step_files(segments: Sequence[Path]) -> dict[int, Path]:
    selected = {}
    for segment in segments:
        if not segment.is_dir():
            continue
        for path in segment.glob("*.jsonl"):
            try:
                selected[int(path.stem)] = path
            except ValueError:
                continue
    return dict(sorted(selected.items()))


def manifest_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def task_category_map(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    output = {}
    for row in manifest_rows(path):
        names = {}
        for category in ("oracle", "misleading", "relevant", "irrelevant"):
            for entry in row[category]:
                names[entry["name"]] = category
        output[(row["bench"], str(row["task_id"]))] = names
    return output


def action_names(response: str) -> list[str]:
    blocks = [match.group(0) for match in JSON_TOOL_CALL_RE.finditer(response)]
    if not blocks:
        blocks = [match.group(0) for match in XML_TOOL_CALL_RE.finditer(response)]
    return [match.group(1) for block in blocks for match in SKILL_PATH_RE.finditer(block)]


def action_identity_details(response: str, tokenizer: Any) -> list[dict[str, Any]]:
    """Reconstruct skill-name token spans inside serialized read tool calls."""

    blocks = [match.group(0) for match in JSON_TOOL_CALL_RE.finditer(response)]
    if not blocks:
        blocks = [match.group(0) for match in XML_TOOL_CALL_RE.finditer(response)]
    details = []
    for block in blocks:
        encoded = tokenizer(block, add_special_tokens=False, return_offsets_mapping=True)
        offsets = list(encoded["offset_mapping"])
        for match in SKILL_PATH_RE.finditer(block):
            start, end = match.span(1)
            token_count = sum(
                int(token_end) > start and int(token_start) < end and int(token_end) > int(token_start)
                for token_start, token_end in offsets
            )
            if token_count <= 0:
                token_count = len(tokenizer.encode(match.group(1), add_special_tokens=False))
            details.append({"name": match.group(1), "identity_tokens": int(token_count)})
    return details


def build_advantage_dilution(out: Path) -> dict[str, Any]:
    files = step_files(BASE_TRAIN_SEGMENTS)
    clean_files = step_files(CLEAN_TRAIN_SEGMENTS)
    cache = out / "cache/base_advantage.json"
    current = signature([*files.values(), *clean_files.values(), SFT_MODEL / "tokenizer.json"])
    try:
        payload = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        payload = {}
    if payload.get("signature") == current:
        return payload["result"]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL, trust_remote_code=True)
    categories = task_category_map(TRAIN_MANIFEST)
    observations = []
    cases = {"oracle_failed_negative": None, "misleading_success_positive": None}
    for step, path in files.items():
        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    sample = json.loads(line)
                    groups[int(sample["group_index"])].append(sample)
        for group_index, samples in groups.items():
            rewards = [float((sample.get("reward") or {}).get("raw_score", 0.0)) for sample in samples]
            mean_reward = statistics.mean(rewards)
            std_reward = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
            advantages = [(reward - mean_reward) / (std_reward + 1e-6) for reward in rewards]
            for sample, reward, advantage in zip(samples, rewards, advantages, strict=True):
                meta = sample.get("reward") or {}
                bench = {"seta": "seta_synth", "swe": "swe_lite"}.get(str(meta.get("bench") or ""), str(meta.get("bench") or ""))
                task = str(meta.get("task_id") or "")
                details = action_identity_details(str(sample.get("response") or ""), tokenizer)
                names = [detail["name"] for detail in details]
                action_categories = [categories.get((bench, task), {}).get(name, "unadvertised") for name in names]
                seen = set(action_categories)
                oracle_details = [
                    detail for detail, category in zip(details, action_categories, strict=True)
                    if category == "oracle"
                ]
                misleading_details = [
                    detail for detail, category in zip(details, action_categories, strict=True)
                    if category == "misleading"
                ]
                row = {
                    "step": step, "group_index": group_index, "bench": bench, "task": task,
                    "reward": reward, "advantage": advantage, "read_names": names,
                    "oracle": "oracle" in seen, "misleading": "misleading" in seen,
                    "oracle_actions": len(oracle_details),
                    "oracle_identity_tokens": sum(detail["identity_tokens"] for detail in oracle_details),
                    "misleading_actions": len(misleading_details),
                    "misleading_identity_tokens": sum(detail["identity_tokens"] for detail in misleading_details),
                }
                observations.append(row)
                evidence = {**row, "response_tail": str(sample.get("response") or "")[-1800:]}
                if row["oracle"] and reward <= 0 and advantage < 0 and cases["oracle_failed_negative"] is None:
                    cases["oracle_failed_negative"] = evidence
                if row["misleading"] and not row["oracle"] and reward > 0 and advantage > 0 and cases["misleading_success_positive"] is None:
                    cases["misleading_success_positive"] = evidence

    def stats(rows: list[dict[str, Any]], *, prefix: str) -> dict[str, float | int]:
        values = [row["advantage"] for row in rows]
        token_field = f"{prefix}_identity_tokens"
        action_field = f"{prefix}_actions"
        return {
            "n": len(rows),
            "mean_advantage": float(np.mean(values)) if values else float("nan"),
            "positive_rate": float(np.mean([value > 0 for value in values])) if values else float("nan"),
            "negative_n": sum(value < 0 for value in values),
            "negative_rate": float(np.mean([value < 0 for value in values])) if values else float("nan"),
            "zero_n": sum(abs(value) <= 1e-12 for value in values),
            "positive_n": sum(value > 0 for value in values),
            "mean_reward": float(np.mean([row["reward"] for row in rows])) if values else float("nan"),
            "actions": sum(int(row[action_field]) for row in rows),
            "identity_tokens": sum(int(row[token_field]) for row in rows),
            "negative_identity_tokens": sum(int(row[token_field]) for row in rows if row["advantage"] < 0),
            "positive_identity_tokens": sum(int(row[token_field]) for row in rows if row["advantage"] > 0),
        }

    clean = {
        "steps": len(clean_files),
        "observations": 0,
        "read_actions": 0,
        "identity_tokens": 0,
        "oracle_actions": 0,
        "oracle_identity_tokens": 0,
        "selector_positive_identity_tokens": 0,
        "selector_negative_identity_tokens": 0,
        "oracle_selector_negative_identity_tokens": 0,
        "task_gradient_selector_identity_tokens": 0,
        "task_negative_selector_identity_tokens": 0,
    }
    for path in clean_files.values():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                clean["observations"] += 1
                state = sample.get("selector_action_credit") or {}
                for action in state.get("actions") or []:
                    token_count = len(action.get("identity_token_indices") or [])
                    advantage = float(action.get("selector_advantage") or 0.0)
                    clean["read_actions"] += 1
                    clean["identity_tokens"] += token_count
                    if action.get("category") == "oracle":
                        clean["oracle_actions"] += 1
                        clean["oracle_identity_tokens"] += token_count
                        if advantage < 0:
                            clean["oracle_selector_negative_identity_tokens"] += token_count
                    if advantage > 0:
                        clean["selector_positive_identity_tokens"] += token_count
                    elif advantage < 0:
                        clean["selector_negative_identity_tokens"] += token_count

    result = {
        "steps": len(files),
        "observations": len(observations),
        "normalization": "within 8-sample group, sample standard deviation, epsilon=1e-6",
        "oracle_read": stats([row for row in observations if row["oracle"]], prefix="oracle"),
        "misleading_without_oracle": stats(
            [row for row in observations if row["misleading"] and not row["oracle"]], prefix="misleading"
        ),
        "skillgate_mask_audit": clean,
        "cases": cases,
    }
    write_json(cache, {"signature": current, "result": result})
    return result


STARVATION_BIN_EDGES = (3000, 6000, 10000, 16000)
STARVATION_BIN_LABELS = ("<=3k", "3-6k", "6-10k", "10-16k", ">16k")
_STARVATION_STATE: dict[str, Any] = {}


def starvation_bin(length: int) -> int:
    for index, edge in enumerate(STARVATION_BIN_EDGES):
        if length <= edge:
            return index
    return len(STARVATION_BIN_EDGES)


ASSISTANT_SPAN_RE = re.compile(r"<\|im_start\|>assistant\n?(.*?)(?:<\|im_end\|>|\Z)", re.DOTALL)


def _init_starvation_worker(model_path: str, manifest_path: str) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    _STARVATION_STATE["tokenizer"] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _STARVATION_STATE["categories"] = task_category_map(Path(manifest_path))


def _starvation_read_spans(response: str, tokenizer: Any) -> tuple[int, int, list[str]]:
    """Token counts of full read tool-call blocks and their identity substrings."""

    blocks = [match.group(0) for match in JSON_TOOL_CALL_RE.finditer(response)]
    if not blocks:
        blocks = [match.group(0) for match in XML_TOOL_CALL_RE.finditer(response)]
    read_tokens = 0
    identity_tokens = 0
    names: list[str] = []
    for block in blocks:
        matches = list(SKILL_PATH_RE.finditer(block))
        if not matches:
            continue
        encoded = tokenizer(block, add_special_tokens=False, return_offsets_mapping=True)
        offsets = list(encoded["offset_mapping"])
        read_tokens += len(offsets)
        for match in matches:
            start, end = match.span(1)
            count = sum(
                int(token_end) > start and int(token_start) < end and int(token_end) > int(token_start)
                for token_start, token_end in offsets
            )
            if count <= 0:
                count = len(tokenizer.encode(match.group(1), add_special_tokens=False))
            identity_tokens += int(count)
            names.append(match.group(1))
    return read_tokens, identity_tokens, names


def _starvation_worker(path_str: str) -> list[dict[str, Any]]:
    tokenizer = _STARVATION_STATE["tokenizer"]
    categories = _STARVATION_STATE["categories"]
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with open(path_str, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                sample = json.loads(line)
                groups[int(sample["group_index"])].append(sample)
    step = int(Path(path_str).stem)
    records = []
    for group_index, samples in groups.items():
        rewards = [float((sample.get("reward") or {}).get("raw_score", 0.0)) for sample in samples]
        mean_reward = statistics.mean(rewards)
        std_reward = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        for sample, reward in zip(samples, rewards, strict=True):
            meta = sample.get("reward") or {}
            bench = {"seta": "seta_synth", "swe": "swe_lite"}.get(str(meta.get("bench") or ""), str(meta.get("bench") or ""))
            task = str(meta.get("task_id") or "")
            response = str(sample.get("response") or "")
            first_end = response.find("<|im_end|>")
            spans = [response[: first_end if first_end >= 0 else len(response)]]
            spans += [match.group(1) for match in ASSISTANT_SPAN_RE.finditer(response)]
            assistant_tokens = sum(
                len(tokenizer(span, add_special_tokens=False)["input_ids"]) for span in spans if span
            )
            read_tokens, identity_tokens, names = _starvation_read_spans(response, tokenizer)
            name_categories = {categories.get((bench, task), {}).get(name, "unadvertised") for name in names}
            records.append({
                "step": step,
                "group_index": group_index,
                "bench": bench,
                "task": task,
                "reward": reward,
                "advantage": (reward - mean_reward) / (std_reward + 1e-6),
                "success": reward > 0,
                "assistant_tokens": int(assistant_tokens),
                "response_tokens": int(sample.get("response_length") or 0),
                "read_tokens": int(read_tokens),
                "identity_tokens": int(identity_tokens),
                "oracle": "oracle" in name_categories,
                "any_read": bool(names),
            })
    return records


def _starvation_cluster_ci(
    groups: Sequence[dict[str, Any]], rng: np.random.Generator, samples: int
) -> dict[str, float]:
    per_task: dict[str, list[float]] = defaultdict(list)
    for group in groups:
        per_task[group["task"]].append(group["delta_success"])
    task_means = np.asarray([float(np.mean(values)) for values in per_task.values()])
    if not len(task_means):
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "tasks": 0}
    draws = rng.integers(0, len(task_means), size=(samples, len(task_means)))
    boot = task_means[draws].mean(axis=1)
    return {
        "mean": float(task_means.mean()),
        "low": float(np.percentile(boot, 2.5)),
        "high": float(np.percentile(boot, 97.5)),
        "tasks": int(len(task_means)),
    }


def build_credit_starvation(out: Path, bootstrap_samples: int) -> dict[str, Any]:
    """Section 6.10: loss-weight share, advantage SNR, and importance of the read decision by horizon."""

    files = step_files(BASE_TRAIN_SEGMENTS)
    cache = out / "cache/credit_starvation.json"
    current = signature([*files.values(), TRAIN_MANIFEST, SFT_MODEL / "tokenizer.json"]) + "|starvation-v1"
    try:
        payload = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        payload = {}
    if payload.get("signature") == current:
        result = payload["result"]
    else:
        from concurrent.futures import ProcessPoolExecutor

        records: list[dict[str, Any]] = []
        with ProcessPoolExecutor(
            max_workers=min(24, os.cpu_count() or 4),
            initializer=_init_starvation_worker,
            initargs=(str(SFT_MODEL), str(TRAIN_MANIFEST)),
        ) as pool:
            for chunk in pool.map(_starvation_worker, [str(path) for path in files.values()]):
                records.extend(chunk)

        eligible = [row for row in records if row["assistant_tokens"] > 0]
        for row in eligible:
            row["bin"] = starvation_bin(row["assistant_tokens"])
            row["read_share"] = row["read_tokens"] / row["assistant_tokens"]
            row["identity_share"] = row["identity_tokens"] / row["assistant_tokens"]

        def share_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
            reads = [row["read_share"] for row in rows if row["any_read"]]
            identities = [row["identity_share"] for row in rows if row["any_read"]]
            return {
                "n": len(rows),
                "n_with_read": len(reads),
                "read_share_median": float(np.median(reads)) if reads else float("nan"),
                "read_share_mean": float(np.mean(reads)) if reads else float("nan"),
                "identity_share_median": float(np.median(identities)) if identities else float("nan"),
                "assistant_tokens_median": float(np.median([row["assistant_tokens"] for row in rows])),
                "assistant_over_response_mean": float(np.mean([
                    row["assistant_tokens"] / row["response_tokens"] for row in rows if row["response_tokens"]
                ])),
            }

        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in eligible:
            grouped[(row["step"], row["group_index"])].append(row)
        mixed_groups = []
        for (step, group_index), members in grouped.items():
            oracle_members = [row for row in members if row["oracle"]]
            other_members = [row for row in members if not row["oracle"]]
            if not oracle_members or not other_members:
                continue
            mixed_groups.append({
                "task": f"{members[0]['bench']}::{members[0]['task']}",
                "bin": starvation_bin(int(np.median([row["assistant_tokens"] for row in members]))),
                "delta_advantage": float(np.mean([row["advantage"] for row in oracle_members])
                                         - np.mean([row["advantage"] for row in other_members])),
                "delta_success": float(np.mean([row["success"] for row in oracle_members])
                                       - np.mean([row["success"] for row in other_members])),
            })

        rng = np.random.default_rng(20260724)
        bins = []
        for index, label in enumerate(STARVATION_BIN_LABELS):
            rows = [row for row in eligible if row["bin"] == index]
            oracle_rows = [row for row in rows if row["oracle"]]
            bin_groups = [group for group in mixed_groups if group["bin"] == index]
            deltas = np.asarray([group["delta_advantage"] for group in bin_groups])
            bins.append({
                "label": label,
                **share_stats(rows),
                "oracle_read_n": len(oracle_rows),
                "oracle_negative_rate": float(np.mean([row["advantage"] < 0 for row in oracle_rows]))
                if oracle_rows else float("nan"),
                "mixed_groups": len(bin_groups),
                "delta_advantage_mean": float(deltas.mean()) if len(deltas) else float("nan"),
                "delta_advantage_sd": float(deltas.std(ddof=1)) if len(deltas) > 1 else float("nan"),
                "delta_advantage_snr": float(deltas.mean() / deltas.std(ddof=1))
                if len(deltas) > 1 and deltas.std(ddof=1) > 0 else float("nan"),
                "delta_success": _starvation_cluster_ci(bin_groups, rng, bootstrap_samples),
            })

        all_oracle = [row for row in eligible if row["oracle"]]
        all_deltas = np.asarray([group["delta_advantage"] for group in mixed_groups])
        result = {
            "steps": len(files),
            "observations": len(records),
            "eligible": len(eligible),
            "bin_edges": list(STARVATION_BIN_EDGES),
            "normalization": (
                "per-sequence mean: each trajectory contributes mean of masked-token loss; "
                "loss mask covers assistant-generated tokens only (tool observations are 0)"
            ),
            "overall": {
                **share_stats(eligible),
                "oracle_read_n": len(all_oracle),
                "oracle_negative_rate": float(np.mean([row["advantage"] < 0 for row in all_oracle])),
                "mixed_groups": len(mixed_groups),
                "delta_advantage_mean": float(all_deltas.mean()),
                "delta_advantage_sd": float(all_deltas.std(ddof=1)),
                "delta_advantage_snr": float(all_deltas.mean() / all_deltas.std(ddof=1)),
                "delta_success": _starvation_cluster_ci(mixed_groups, rng, bootstrap_samples),
            },
            "bins": bins,
        }
        write_json(cache, {"signature": current, "result": result})

    import matplotlib.pyplot as plt

    labels = [row["label"] for row in result["bins"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4), constrained_layout=True)

    read_shares = [100 * row["read_share_median"] for row in result["bins"]]
    identity_shares = [100 * row["identity_share_median"] for row in result["bins"]]
    axes[0].plot(x, read_shares, marker="o", color=COLORS["mixed baseRL"], label="full read tool call")
    axes[0].plot(x, identity_shares, marker="s", color="#89959B", label="skill identity tokens")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Median share of trajectory loss weight (%)")
    axes[0].set_title("A. Selection share of loss mass", pad=10)
    axes[0].legend(frameon=False, fontsize=8)

    negative_rates = [100 * row["oracle_negative_rate"] for row in result["bins"]]
    bars = axes[1].bar(x, negative_rates, color=COLORS["misleading"], width=0.6)
    for bar, value in zip(bars, negative_rates, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}", ha="center", fontsize=8)
    axes[1].axhline(50, linestyle="--", color="#5B6770", linewidth=1)
    axes[1].text(-0.35, 51.2, "coin flip", fontsize=8, color="#5B6770")
    twin = axes[1].twinx()
    snrs = [row["delta_advantage_snr"] for row in result["bins"]]
    twin.plot(x, snrs, marker="D", color="#27323A", linewidth=1.4, label="group SNR (right)")
    twin.set_ylabel("Within-group advantage SNR of oracle read")
    twin.spines[["top"]].set_visible(False)
    twin.legend(frameon=False, fontsize=8, loc="center right")
    axes[1].set_ylabel("P(negative advantage | read oracle) (%)")
    axes[1].set_ylim(0, 70)
    axes[1].set_title("B. Sequence-level signal at the decision", pad=10)

    delta_means = [100 * row["delta_success"]["mean"] for row in result["bins"]]
    delta_err = np.asarray([
        [100 * (row["delta_success"]["mean"] - row["delta_success"]["low"]) for row in result["bins"]],
        [100 * (row["delta_success"]["high"] - row["delta_success"]["mean"]) for row in result["bins"]],
    ])
    axes[2].errorbar(x, delta_means, yerr=delta_err, fmt="o-", color=COLORS["SkillGate"], capsize=3)
    axes[2].axhline(0, linestyle="--", color="#5B6770", linewidth=1)
    axes[2].set_ylabel("Within-group success delta of oracle read (pp)")
    axes[2].set_title("C. The decision still matters", pad=10)

    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("Assistant (loss-bearing) tokens per trajectory")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    figure = out / "figures/section6_10_credit_starvation.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    result["figure"] = str(figure.relative_to(ROOT))
    return result


def incremental_metrics(trial: dict[str, Any]) -> dict[str, float]:
    path = Path(trial["leaf"]) / "incremental.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    row = rows[-1]
    return {
        "turns": float(row.get("turns") or 0),
        "input_tokens": float(row.get("input_tokens") or 0),
        "output_tokens": float(row.get("output_tokens") or 0),
        "wall_sec": float(row.get("wall_sec") or row.get("time_sec") or 0),
    }


def build_efficiency(data: dict[str, list[dict[str, Any]]], out: Path) -> dict[str, Any]:
    output = {}
    for label in ("SkillGate", "mixed baseRL"):
        trials = data[label]
        metrics = [incremental_metrics(row) for row in trials]
        output[label] = {
            "trials": len(trials),
            "reads_per_trial": float(np.mean([len(set(row.get("read_names_agent") or [])) for row in trials])),
            **{key: float(np.mean([row[key] for row in metrics])) for key in ("turns", "input_tokens", "output_tokens", "wall_sec")},
        }
    output["relative_delta"] = {
        key: output["SkillGate"][key] / output["mixed baseRL"][key] - 1.0
        for key in ("reads_per_trial", "turns", "input_tokens", "output_tokens")
    }

    import matplotlib.pyplot as plt

    specs = (
        ("reads_per_trial", "Unique skills / trial", 1.0),
        ("turns", "Turns / trial", 1.0),
        ("input_tokens", "Cumulative input tokens (k)", 1000.0),
        ("output_tokens", "Output tokens (k)", 1000.0),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    methods = ("mixed baseRL", "SkillGate")
    for ax, (key, title, scale) in zip(axes.flat, specs, strict=True):
        values = [output[label][key] / scale for label in methods]
        bars = ax.bar(np.arange(2), values, color=[COLORS[label] for label in methods], width=0.58)
        ax.set_xticks(np.arange(2), ("Mixed baseRL", "SkillGate"))
        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.22)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.035,
                    f"{value:.2f}" if value < 100 else f"{value:.0f}", ha="center", fontsize=8)
        delta = 100 * output["relative_delta"][key]
        ax.text(0.98, 0.94, f"SkillGate: {delta:+.1f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#3D4A50")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Observed behavior and context cost", fontsize=12)
    fig.tight_layout()
    figure = out / "figures/section6_6_behavior_cost.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    output["figure"] = str(figure.relative_to(ROOT))
    return output


def build_context_cost(data: dict[str, list[dict[str, Any]]], out: Path) -> dict[str, Any]:
    cache = out / "cache/token_lengths.json"
    rows = manifest_rows(EVAL_MANIFEST)
    skill_paths = [Path(entry["path"]) / "SKILL.md" for row in rows for category in ("oracle", "misleading", "relevant", "irrelevant") for entry in row[category]]
    current = signature(skill_paths) + ":" + signature([SFT_MODEL / "tokenizer.json"])
    try:
        payload = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        payload = {}
    if payload.get("signature") == current:
        task_lengths = payload["task_lengths"]
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL, trust_remote_code=True)
        token_cache: dict[str, int] = {}
        task_lengths = []
        for row in rows:
            skills = {}
            for category in ("oracle", "misleading", "relevant", "irrelevant"):
                for entry in row[category]:
                    path = str(Path(entry["path"]) / "SKILL.md")
                    if path not in token_cache:
                        text = Path(path).read_text(encoding="utf-8", errors="replace")
                        token_cache[path] = len(tokenizer.encode(text, add_special_tokens=False))
                    skills[entry["name"]] = token_cache[path]
            task_lengths.append({"bench": row["bench"], "task": str(row["task_id"]), "skills": skills})
        write_json(cache, {"signature": current, "task_lengths": task_lengths})

    task_map = {(row["bench"], row["task"]): row["skills"] for row in task_lengths}
    full_values = np.asarray([sum(row["skills"].values()) for row in task_lengths], dtype=float)
    preload = {
        count: np.asarray([count * float(np.mean(list(row["skills"].values()))) for row in task_lengths])
        for count in (1, 2, 4, 8, 16)
    }
    actual = []
    unmapped_reads = 0
    for trial in data["SkillGate"]:
        key = (BENCH_TO_MANIFEST.get(str(trial["bench"]), str(trial["bench"])), str(trial["task"]))
        lengths = task_map[key]
        names = set(str(name) for name in (trial.get("read_names_agent") or []))
        unmapped_reads += sum(name not in lengths for name in names)
        actual.append(sum(lengths.get(name, 0) for name in names))
    actual_values = np.asarray(actual, dtype=float)

    strategies = {"SkillGate on-demand": actual_values, **{f"Preload {count}": values for count, values in preload.items()}}
    summaries = {
        label: {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
        }
        for label, values in strategies.items()
    }

    import matplotlib.pyplot as plt

    labels = list(strategies)
    means = [summaries[label]["mean"] for label in labels]
    colors = [COLORS["SkillGate"], "#B8C1C5", "#AAB5BA", "#94A2A8", "#7E9098", "#687E88"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bars = ax.bar(np.arange(len(labels)), means, color=colors, width=0.68)
    ax.set_xticks(np.arange(len(labels)), ("SkillGate\non-demand", "1 full\nskill", "2 full\nskills", "4 full\nskills", "8 full\nskills", "Full slate\n(16)"))
    ax.set_ylabel("Mean loaded skill-body tokens / trial")
    ax.set_title("On-demand reading versus preloading full skill bodies")
    ax.set_ylim(0, max(means) * 1.18)
    for bar, value in zip(bars, means, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(means) * 0.025,
                f"{value / 1000:.1f}k", ha="center", fontsize=8)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    figure = out / "figures/section6_6_context_loading.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "figure": str(figure.relative_to(ROOT)),
        "tasks": len(task_lengths),
        "skillgate_trials": len(actual_values),
        "unmapped_reads": unmapped_reads,
        "strategies": summaries,
        "full_slate": {
            "median": float(np.median(full_values)),
            "p90": float(np.quantile(full_values, 0.9)),
            "max": int(full_values.max()),
            "over_65536": int(np.sum(full_values > 65536)),
        },
        "preload_definition": "For each task, k times the mean full-body token length among its 16 slate candidates.",
    }


def build_gradient_error_figure(out: Path, advantage: dict[str, Any]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    oracle = advantage["oracle_read"]
    clean = advantage["skillgate_mask_audit"]
    total = oracle["n"]
    negative_trajectory_rate = 100 * oracle["negative_n"] / total
    base_negative_token_rate = 100 * oracle["negative_identity_tokens"] / oracle["identity_tokens"]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    trajectory_values = (negative_trajectory_rate, 100.0 - negative_trajectory_rate)
    trajectory_labels = ("Negative task advantage", "Zero or positive")
    bars = axes[0].barh(
        np.arange(2), trajectory_values,
        color=(COLORS["misleading"], "#7A8A93"), height=0.52,
    )
    axes[0].set_yticks(np.arange(2), trajectory_labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Share of oracle-read trajectories (%)")
    axes[0].set_title(f"A. baseRL oracle-read trajectories (N={total:,})", pad=12)
    for bar, value in zip(bars, trajectory_values, strict=True):
        axes[0].text(
            value + 1.5, bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%", va="center", fontsize=9, color="#27323A",
        )

    method_labels = ("mixed baseRL", "SkillGate")
    token_values = (base_negative_token_rate, 0.0)
    bars = axes[1].bar(
        np.arange(2), token_values,
        color=(COLORS["mixed baseRL"], COLORS["SkillGate"]), width=0.58,
    )
    axes[1].set_xticks(np.arange(2), method_labels)
    axes[1].set_ylabel("Oracle identity tokens with\nnegative task gradient (%)")
    axes[1].set_ylim(0, 52)
    axes[1].set_title("B. Does task outcome update the selector?", pad=12)
    for bar, value in zip(bars, token_values, strict=True):
        y = value + 1.2 if value else 1.2
        axes[1].text(bar.get_x() + bar.get_width() / 2, y, f"{value:.1f}%", ha="center", fontsize=9)
    axes[1].text(
        1, 8.0, "read-call task mask", ha="center", va="center",
        fontsize=8, color=COLORS["SkillGate"],
    )

    for ax in axes:
        ax.grid(axis="x" if ax is axes[0] else "y", color="#E5E5E5", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    figure = out / "figures/section6_9_gradient_error.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "figure": str(figure.relative_to(ROOT)),
        "base_negative_identity_token_rate": base_negative_token_rate / 100.0,
        "skillgate_task_gradient_selector_tokens": clean["task_gradient_selector_identity_tokens"],
    }


def relative_figure(path: str) -> str:
    return "../" + path


def render_noskill_route_strata(section: dict[str, Any]) -> list[str]:
    lines = [
        f"这里使用此前已经完成的 no-skill RL final99 标准 mixed-slate 评测（70 题 x 4，共 {section['total_trials']} 条轨迹），不再使用 SFT9B 的另一次 30 题干预。对每条轨迹只看 agent 实际发出的 attributed `read`：只读 oracle 归入 oracle-only，只读 misleading 归入 misleading-only，完全不读归入 no-read。原始 row 为 `{section['source_row']}`。",
        "",
        "| no-skill RL 的自然读取路径 | 成功 / N | trial success | 覆盖任务数 |",
        "|---|---:|---:|---:|",
    ]
    for label in ("oracle-only", "misleading-only", "no-read"):
        row = section["conditions"][label]
        lines.append(
            f"| {label} | {row['success']}/{row['trials']} | {pct(row['rate'])} | {row['tasks_represented']} |"
        )
    lines.extend([
        "",
        f"oracle-only 的成功率最高（{pct(section['conditions']['oracle-only']['rate'])}），misleading-only 较低（{pct(section['conditions']['misleading-only']['rate'])}），no-read 最低（{pct(section['conditions']['no-read']['rate'])}）。另有 {section['excluded_total']} 条轨迹同时读取多类 skill，或只读 relevant / irrelevant，未硬塞进这三个互斥组。这个排序与“正确 skill 更有帮助”的解释一致，但它不是随机干预：任务难度会影响模型选择哪条路径，所以这里只能作为条件相关性证据，不能写成 oracle exposure 的因果提升。",
    ])
    return lines


def render_analysis(result: dict[str, Any]) -> str:
    bootstrap = result["analysis"]["paired_bootstrap"]
    matched = result["analysis"]["matched_routes"]
    route_strata = result["analysis"]["noskill_route_strata"]
    advantage = result["analysis"]["advantage_dilution"]
    efficiency = result["analysis"]["efficiency"]
    context = result["analysis"]["context_cost"]
    gradient = result["analysis"]["gradient_error"]
    lines: list[str] = []

    lines.extend([
        "### 6.1 端到端主结果的不确定性",
        "",
        f"只比较论文方法 SkillGate 与同训练预算的 mixed baseRL。具体做法是：先把每个任务的 4 次 rollout 求平均，得到每个方法每题一个 trial-success 值；然后以任务为抽样单位、有放回地重采样 {bootstrap['bootstrap_samples']:,} 次，并在两种方法中使用同一批任务索引。这就是 task-cluster paired bootstrap；它保留题内 4 repeats 的相关性，也避免把 280 条轨迹误当成 280 个独立任务。图中纵轴是 benchmark（括号为任务数），横轴是 `SkillGate - mixed baseRL` 的成功率百分点差；圆点是原样本差值，横线是 95% bootstrap 区间，竖直虚线 0 表示两者相同。",
        "",
        f"![paired bootstrap]({relative_figure(bootstrap['figure'])})",
        "",
        "| benchmark | trial success 差 (95% CI) | task pass@4 差 (95% CI) |",
        "|---|---:|---:|",
    ])
    labels = {"ALL": "ALL", "claw": "Claw", "sb_ns": "SB", "seta": "SETA", "swe": "SWE", "tb2": "TB2"}
    for row in bootstrap["rows"]:
        trial, task = row["trial_success"], row["task_pass4"]
        lines.append(f"| {labels[row['bench']]} | {pp(trial['delta'])} [{pp(trial['ci_low'])}, {pp(trial['ci_high'])}] | {pp(task['delta'])} [{pp(task['ci_low'])}, {pp(task['ci_high'])}] |")
    lines.extend([
        "",
        "ALL trial-success 的 95% 区间不跨零，而 task pass@4 区间仍跨零；因此正文可以报告 trial-level 的配对 bootstrap 改善，同时不能把更高的 pass@4 写成已经显著。单项 benchmark 的任务数更少、区间普遍更宽。三种旧 credit 设计已放在 Ablation，不再混进主结果的置信区间图。",
        "",
        "### 6.2 同读取路径的 selector / executor 审计",
        "",
        f"这里不再使用“同首读”这种较松条件，只保留 **完整有序读取序列完全相同** 的轨迹。对 SkillGate、mixed baseRL 和 SelSkill DPO 各 280 条自然 rollout，以 `(benchmark, task, ordered read-name sequence)` 建立 stratum；空序列也作为 no-read 路径。只有三种方法都实际出现过的序列才保留，共 {matched['strata']} 个 strata、覆盖 {matched['tasks']} 个任务。各模型的 repeats 独立采样且没有可验证的跨模型 repeat id，因此不把第 1 条和第 1 条硬配对。主统计先求每个 task-sequence 的成功率，再在任务内等权平均 sequence、最后在任务间等权平均；区间按任务 cluster 重采样 {matched['bootstrap_samples']:,} 次。",
        "",
        f"![same full sequence]({relative_figure(matched['figure'])})",
        "",
        "| 方法 | shared-strata raw success/N | task-macro success (95% CI) |",
        "|---|---:|---:|",
    ])
    for label in ("SkillGate", "mixed baseRL", "SelSkill DPO"):
        row = matched["methods"][label]
        lines.append(
            f"| {label} | {row['success']}/{row['n']} ({pct(row['raw_rate'])}) | "
            f"{pct(row['task_macro_rate'])} [{pct(row['ci_low'])}, {pct(row['ci_high'])}] |"
        )
    lines.extend(["", "| task-macro 配对差值 | 差值 (95% CI) |", "|---|---:|"])
    for label, row in matched["comparisons"].items():
        lines.append(f"| {label} | {pp(row['delta'])} [{pp(row['ci_low'])}, {pp(row['ci_high'])}] |")
    lines.extend([
        "",
        "三种方法的 raw N 不同，是因为同一个共享 sequence 在各方法的 4 次 rollout 中出现次数不同；论文比较采用 task-macro，不让重复出现较多的一方获得额外权重。这个结果只描述“观察到完全相同路径以后，executor 做得怎样”，仍然是 post-selection 分析：模型是否进入该路径并非随机分配，所以不能解释成路径的因果效应。",
        "",
        "### 6.3 no-skill RL 读到不同类型 skill 后的结果",
        "",
    ])
    lines.extend(render_noskill_route_strata(route_strata))
    oracle = advantage["oracle_read"]
    lines.extend([
        "",
        "### 6.6 行为成本",
        "",
        "第一张图直接比较 SkillGate 与 mixed baseRL 在同一 280-trial mixed-slate 协议下的实际行为成本。reads 是每条轨迹读取的不同 skill 数；turns 是 agent 回合数；input tokens 是多轮 API 请求的累计输入，因此会把不断增长的 history 重复计入，适合比较推理负担但不等同于单次 context 长度。不同批次的 serving 拓扑会影响 wall time，所以这里不把 wall time 当成方法效率证据。",
        "",
        f"![behavior cost]({relative_figure(efficiency['figure'])})",
        "",
        "| 方法 | unique reads/trial | turns | cumulative input tok | output tok |",
        "|---|---:|---:|---:|---:|",
    ])
    for label in ("SkillGate", "mixed baseRL"):
        row = efficiency[label]
        lines.append(f"| {label} | {row['reads_per_trial']:.2f} | {row['turns']:.1f} | {row['input_tokens']:.0f} | {row['output_tokens']:.0f} |")
    deltas = efficiency["relative_delta"]
    strategy = context["strategies"]
    lines.extend([
        "",
        f"相对 mixed baseRL，SkillGate 的不同 skill 读取数减少 `{abs(100*deltas['reads_per_trial']):.1f}%`，turns 减少 `{abs(100*deltas['turns']):.1f}%`，累计 input tokens 减少 `{abs(100*deltas['input_tokens']):.1f}%`；output tokens 的变化为 `{100*deltas['output_tokens']:+.1f}%`。因此选择纯度不是靠扩大读取量获得。",
        "",
        "第二张图回答另一种部署选择：如果不按需 `read`，而是在开始执行前把 1、2、4、8 或全部 16 个候选的 SKILL.md 正文放进上下文，会产生多少正文 token。对每题的 hypothetical preload-k，取该题 16 个候选正文 token 数的均值乘 k；SkillGate 列则按 280 条真实轨迹实际读取的不同 skill 正文求和。这里只计 skill body，不含共同的 name/description slate、system/tools、任务文本与后续 history。",
        "",
        f"![context loading cost]({relative_figure(context['figure'])})",
        "",
        f"真实 SkillGate 平均加载 `{strategy['SkillGate on-demand']['mean']:.0f}` 个正文 tokens；hypothetical preload-2 / preload-4 / full-slate 分别为 `{strategy['Preload 2']['mean']:.0f}` / `{strategy['Preload 4']['mean']:.0f}` / `{strategy['Preload 16']['mean']:.0f}`。按需读取与理想的单 skill 全文预载成本接近，但不要求外部 selector 事先锁定唯一候选；相比放入多个或整个 slate，正文 context 成本随候选数近似线性增加。",
        "",
        "### 6.9 SkillGate 如何避免 selector 的梯度错误",
        "",
        f"这不是新跑的一轮模型评测，而是对两次已有训练的 loss support 做离线审计。第一步，从 mixed baseRL 的 {advantage['steps']} 个训练 step、{advantage['observations']} 条 on-policy 轨迹中，按训练时的 8-sample prompt group 和 sample-standard-deviation 口径重算 task GRPO advantage。第二步，只保留实际读到 oracle 的轨迹，并用训练所用 Qwen tokenizer 找出 `read(.../<skill-name>/SKILL.md)` 中决定 skill 身份的 token。第三步，检查这些 selector token 最终继承到的是正 task advantage 还是负 task advantage。",
        "",
        f"![selector gradient error]({relative_figure(gradient['figure'])})",
        "",
        f"左图只看 mixed baseRL 中读到 oracle 的 {oracle['n']} 条轨迹：其中 {oracle['negative_n']} 条（{pct(oracle['negative_rate'])}）因为后续 executor 相对同组表现较差而得到负 task advantage。右图把统计单位换成 oracle 身份 token：mixed baseRL 有 {oracle['negative_identity_tokens']}/{oracle['identity_tokens']}（{pct(oracle['negative_identity_tokens']/oracle['identity_tokens'])}）个 token 收到负 task gradient，而 SkillGate 是 0%。这里的 0% 不是经验巧合，而是 task mask 的定义结果。",
        "",
        "| 方法 | 选对 oracle 后会不会因任务失败而惩罚 selector？ | selector 实际由什么信号训练？ |",
        "|---|---|---|",
        f"| mixed baseRL | 会；{pct(oracle['negative_identity_tokens']/oracle['identity_tokens'])} 的 oracle identity tokens 收到负 task gradient | 没有独立 selector 信号，整条 assistant response 共用 task advantage |",
        "| SkillGate | 不会；整个 skill-read tool call 从 task loss 中排除 | 只由 clean-oracle 局部 utility 及组内 read-action 中心化结果更新 |",
        "",
        "白话地说，mixed baseRL 会把“skill 选对了，但后面没做好任务”误解成“刚才不该选这个 skill”；SkillGate 不允许任务成败反向修改这次路径选择。后续 executor tokens 仍照常承接 task GRPO，read identity tokens 则只回答“选了谁、是否只读一次且读对”。因此 selector 辅助项中的负值只表示读错或读取不够 clean，不表示 oracle 内容执行失败。",
    ])
    starvation = result["analysis"]["credit_starvation"]
    overall = starvation["overall"]
    lines.extend([
        "",
        "### 6.10 credit starvation by horizon (generated summary)",
        "",
        f"eligible={starvation['eligible']}/{starvation['observations']}; "
        f"read-share median={pct(overall['read_share_median'])}, identity median={pct(overall['identity_share_median'])}; "
        f"P(neg|oracle)={pct(overall['oracle_negative_rate'])} over {overall['oracle_read_n']} oracle-read trajectories; "
        f"mixed groups={overall['mixed_groups']}, dAdv SNR={overall['delta_advantage_snr']:.3f}, "
        f"dSuccess={pp(100 * overall['delta_success']['mean'])} "
        f"[{pp(100 * overall['delta_success']['low'])}, {pp(100 * overall['delta_success']['high'])}] "
        f"over {overall['delta_success']['tasks']} tasks.",
        "",
        f"![credit starvation]({relative_figure(starvation['figure'])})",
        "",
        "| bin | n | read-share med | P(neg|oracle) | groups | dAdv SNR | dSuccess (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in starvation["bins"]:
        ci = row["delta_success"]
        lines.append(
            f"| {row['label']} | {row['n']} | {pct(row['read_share_median'])} | {pct(row['oracle_negative_rate'])} | "
            f"{row['mixed_groups']} | {row['delta_advantage_snr']:.3f} | "
            f"{pp(100 * ci['mean'])} [{pp(100 * ci['low'])}, {pp(100 * ci['high'])}] |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    args = parser.parse_args()

    out = args.out.resolve()
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(str(EVAL_MANIFEST))
    data = load_paper_trials(cache)

    result = {
        "analysis_schema": OUTPUT_SCHEMA_VERSION,
        "protocol": {
            "eval": "FINAL70 x4 standard mixed-slate",
            "main_comparison": ["SkillGate", "mixed baseRL"],
            "skillgate_claw_merge": "July-19 non-Claw slice plus corrected Claw14 rerun",
            "mixed_baserl_row": str(ROW_ROOTS["mixed baseRL_paper"].relative_to(ROOT)),
            "note": "Execution dates and serving topology are provenance, not evaluation conditions.",
        },
        "main_results": build_main_results_figure(out),
        "unified_claw161": build_unified_claw161_table(),
        "ablation": build_ablation(data, manifest, out),
        "selector_diagnostics": build_selector_diagnostics(cache, manifest, data, out),
        "analysis": {},
    }
    analysis = result["analysis"]
    analysis["paired_bootstrap"] = build_main_bootstrap(data, out, args.bootstrap_samples)
    analysis["matched_routes"] = build_matched_route_audit(data, manifest, cache, out, args.bootstrap_samples)
    analysis["noskill_route_strata"] = build_noskill_route_strata(cache, manifest)
    analysis["advantage_dilution"] = build_advantage_dilution(out)
    analysis["credit_starvation"] = build_credit_starvation(out, args.bootstrap_samples)
    analysis["efficiency"] = build_efficiency(data, out)
    analysis["context_cost"] = build_context_cost(data, out)
    analysis["gradient_error"] = build_gradient_error_figure(out, analysis["advantage_dilution"])

    write_json(out / "skillgate_paper_analysis.json", result)
    fragment = out / "skillgate_paper_analysis_sections_6.md"
    write_text(fragment, render_analysis(result))
    print(f"wrote {out / 'skillgate_paper_analysis.json'}")
    print(f"wrote {fragment}")


if __name__ == "__main__":
    main()
