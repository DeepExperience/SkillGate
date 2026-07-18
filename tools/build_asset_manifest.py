#!/usr/bin/env python3
"""Build deterministic aggregate hashes for the migrated handover assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "assets/migrated-assets.json"
SKIP_PARTS = {".git", ".validation", "__pycache__", ".pytest_cache"}

GROUPS: dict[str, list[str]] = {
    "code.generalagent": [
        "GeneralAgent/eval_scripts",
        "GeneralAgent/sft_data_collection",
        "GeneralAgent/sft_training/configs",
        "GeneralAgent/rl_data_prep",
    ],
    "code.relax": ["Relax/relax", "Relax/examples", "Relax/scripts", "Relax/deps"],
    "code.eval_stack": ["sglang", "slime", "Megatron-LM"],
    "code.operations": ["ops", "tools", "README.md", "docs"],
    "benchmarks": [
        "datasets/terminal-bench-v2",
        "datasets/seta",
        "datasets/seta-env",
        "datasets/skillsbench/tasks",
        "datasets/skillsbench/tasks-no-skills",
        "datasets/skillsbench/libs",
        "datasets/claw-eval",
        "datasets/swe-gym",
        "datasets/swe-bench-verified",
    ],
    "retrieval.skills": ["skill_libraries"],
    "retrieval.indices": [
        "GeneralAgent/eval_scripts/skills_retrieval/skill_index_qwen3emb8b.pkl",
        "GeneralAgent/eval_scripts/skills_retrieval/skill_index_bm25.pkl",
    ],
    "sft.raw": ["experiments/sft_data_use"],
    "sft.split": ["GeneralAgent/sft_data_collection/outputs/splits/default"],
    "sft.intermediate": ["GeneralAgent/sft_training/datasets"],
    "sft.final": ["GeneralAgent/sft_training/llamafactory_data/20260512_sft_campaign_clean_plus_claw_thinkwrap"],
    "rl.base_data": ["datasets/rl/rl_split_v2.json", "datasets/rl/parquet_4bench_base_20260523"],
    "rl.recipe_data": [
        "datasets/rl/parquet_4bench_factual_20260602",
        "datasets/rl/parquet_4bench_factual_noskills_20260617",
        "datasets/rl/parquet_4bench_oracle1_20260612",
        "datasets/rl/parquet_4bench_oracle_promptbc_pair_noskill_grpo_20260623",
        "datasets/rl/parquet_4bench_slate_regret_v8prod_20260708",
        "datasets/rl/parquet_4bench_slate_regret_v8prod_gold_stratified_v2_20260710",
        "datasets/rl/parquet_4bench_slate_regret_hybridv8b0704d_gold_stratified_20260710",
        "datasets/rl/parquet_4bench_mixed_skill_bonus_compare_v8prod_allgold_20260710",
        "datasets/rl/parquet_4bench_mixed_skill_separated_continuous_advantage_v8prod_allgold_20260710",
        "datasets/rl/parquet_4bench_selector_action_credit_hybridv8b0704d_allgold_20260713",
    ],
    "rl.skills": [
        "skill_libraries/snapshots/rl/oracle_top1_skills_20260612",
        "skill_libraries/snapshots/rl/slate_skills_20260704",
        "skill_libraries/snapshots/rl/slate_skills_20260708_hard_negative_v8_production",
        "skill_libraries/snapshots/rl/slate_skills_20260710_hybrid_v8body_0704desc",
    ],
    "eval.frozen_inputs": [
        "experiments/archive_sft_runs/20260424/20260424_v7pipeline_on_2046lib/retrieval_results",
        "skill_libraries/snapshots/rl/oracle_skills_full692_20260612",
        "skill_libraries/snapshots/rl/eval70_oracle_selfread_20260612",
        "ops/workflows/rl_eval/specs/eval70_v1",
    ],
    "experiment.provenance": [
        "experiments/rl/catalog.json",
        "experiments/rl/HANDOVER_MANIFEST.json",
        "experiments/rl/runs",
        "experiments/rl/sample_trajectories",
        "experiments/infra/rl/local_docker_migration",
    ],
    "offline.package_cache": ["ops/cache/pkg"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def files_for(paths: list[str]):
    seen: set[Path] = set()
    for rel in paths:
        path = ROOT / rel
        if path.is_file() or path.is_symlink():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(path.rglob("*"))
        else:
            continue
        for item in candidates:
            if (
                item in seen
                or any(part in SKIP_PARTS for part in item.relative_to(ROOT).parts)
                or (not item.is_file() and not item.is_symlink())
            ):
                continue
            seen.add(item)
            yield item


def summarize(paths: list[str], hash_content: bool) -> dict:
    aggregate = hashlib.sha256()
    files = symlinks = total_bytes = 0
    missing = [rel for rel in paths if not (ROOT / rel).exists()]
    for path in files_for(paths):
        rel = path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            symlinks += 1
            payload = f"L\0{rel}\0{os.readlink(path)}\n".encode()
        else:
            files += 1
            size = path.stat().st_size
            total_bytes += size
            content = sha256_file(path) if hash_content else "metadata-only"
            payload = f"F\0{rel}\0{size}\0{content}\n".encode()
        aggregate.update(payload)
    return {
        "paths": paths,
        "missing": missing,
        "regular_files": files,
        "symlinks": symlinks,
        "bytes": total_bytes,
        "tree_sha256": aggregate.hexdigest(),
        "hash_mode": "content-sha256" if hash_content else "path-size",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fast", action="store_true", help="hash paths and sizes, not file contents")
    args = parser.parse_args()
    groups = {name: summarize(paths, not args.fast) for name, paths in GROUPS.items()}
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hash_mode": "path-size" if args.fast else "content-sha256",
        "groups": groups,
        "totals": {
            "regular_files": sum(v["regular_files"] for v in groups.values()),
            "symlinks": sum(v["symlinks"] for v in groups.values()),
            "bytes": sum(v["bytes"] for v in groups.values()),
            "missing_paths": sum(len(v["missing"]) for v in groups.values()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(args.output)
    print(json.dumps(payload["totals"], sort_keys=True))
    return 1 if payload["totals"]["missing_paths"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
