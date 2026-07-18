#!/usr/bin/env python3
"""Create a small fixed holdout split from the existing sacred test split.

The full test split is intentionally kept for final evaluation. This script
selects a deterministic 20-30 task subset for fast daily regression checks.
It never samples from train.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    display_path,
    dump_json,
    load_json,
    load_retrieval_task_ids,
    stable_even_holdout,
    write_task_lines,
)


DEFAULT_COUNTS = {
    "claw": 8,
    "tb2": 6,
    "sb_ns": 5,
    "seta_synth": 3,
    "swe_lite": 8,
}


def parse_counts(raw: str) -> dict[str, int]:
    if not raw:
        return dict(DEFAULT_COUNTS)
    counts: dict[str, int] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"bad --counts segment {part!r}; expected bench=N")
        bench, value = part.split("=", 1)
        counts[bench.strip()] = int(value)
    return counts


def build_quick_split(
    source: dict[str, Any],
    config: dict[str, Any],
    counts: dict[str, int],
    name: str,
    require_retrieval_coverage: bool,
) -> dict[str, Any]:
    benches: dict[str, dict[str, list[str]]] = {}
    retrieval_files = config["frozen_retrieval"]["files"]
    retrieval_ids = {
        bench: load_retrieval_task_ids(path)
        for bench, path in retrieval_files.items()
    }
    for bench, split in sorted(source["benches"].items()):
        full_test = list(split.get("test", []))
        covered_test = [
            str(task_id) for task_id in full_test
            if str(task_id) in retrieval_ids.get(bench, set())
        ]
        candidate_test = covered_test if require_retrieval_coverage else [str(task_id) for task_id in full_test]
        requested = counts.get(bench, 0)
        quick_test = stable_even_holdout(candidate_test, min(requested, len(candidate_test)))
        train = list(split.get("train", []))
        overlap = sorted(set(train) & set(quick_test))
        if overlap:
            raise RuntimeError(f"quick holdout overlaps train for {bench}: {overlap[:5]}")
        benches[bench] = {
            "train": train,
            "test": quick_test,
            "full_test": full_test,
            "retrieval_covered_test": covered_test,
            "excluded_missing_retrieval": sorted(set(map(str, full_test)) - set(covered_test)),
        }
    return {
        "schema_version": 1,
        "name": name,
        "source_split": source.get("name", "default"),
        "selection_rule": (
            "stable_even_holdout over each bench's retrieval-covered test list"
            if require_retrieval_coverage
            else "stable_even_holdout over each bench's existing test list"
        ),
        "require_retrieval_coverage": require_retrieval_coverage,
        "counts": {bench: len(split["test"]) for bench, split in benches.items()},
        "benches": benches,
        "freeze_manifest": source.get("freeze_manifest", {}),
    }


def write_split(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(output_dir / "holdout_split.json", payload)
    test_dir = output_dir / "test"
    full_test_dir = output_dir / "full_test"
    covered_test_dir = output_dir / "retrieval_covered_test"
    for bench, split in sorted(payload["benches"].items()):
        write_task_lines(test_dir / f"{bench}.txt", split["test"])
        write_task_lines(full_test_dir / f"{bench}.txt", split["full_test"])
        write_task_lines(covered_test_dir / f"{bench}.txt", split["retrieval_covered_test"])

    lines = [
        f"# Quick Holdout: {payload['name']}",
        "",
        f"- source_split: {payload['source_split']}",
        f"- rule: {payload['selection_rule']}",
        f"- require_retrieval_coverage: {payload['require_retrieval_coverage']}",
        "",
        "| bench | quick_test | retrieval_covered_test | full_test | missing_retrieval | tasks |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for bench, split in sorted(payload["benches"].items()):
        lines.append(
            f"| {bench} | {len(split['test'])} | {len(split['retrieval_covered_test'])} | "
            f"{len(split['full_test'])} | {len(split['excluded_missing_retrieval'])} | "
            f"{', '.join(split['test'])} |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default="GeneralAgent/sft_data_collection/outputs/splits/default/holdout_split.json")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--name", default="quick30")
    parser.add_argument("--counts", default="", help="Comma list, e.g. claw=6,tb2=5,sb_ns=4,seta_synth=8,swe_lite=7")
    parser.add_argument("--out-dir", default="", help="Default: outputs/splits/default/quick_test/<name>")
    parser.add_argument("--allow-missing-retrieval", action="store_true",
                        help="Allow quick eval tasks not covered by the frozen retrieval jsonl. Default: filter them out.")
    args = parser.parse_args()

    source = load_json(args.source)
    config = load_json(args.config)
    counts = parse_counts(args.counts)
    payload = build_quick_split(
        source,
        config,
        counts,
        args.name,
        require_retrieval_coverage=not args.allow_missing_retrieval,
    )
    output_dir = (
        Path(args.out_dir)
        if args.out_dir
        else PROJECT_ROOT / "GeneralAgent" / "sft_data_collection" / "outputs" / "splits" / "default" / "quick_test" / args.name
    )
    write_split(payload, output_dir)
    print(f"name={args.name}")
    print(f"total={sum(len(split['test']) for split in payload['benches'].values())}")
    print(f"split={display_path(output_dir / 'holdout_split.json')}")
    print(f"readme={display_path(output_dir / 'README.md')}")


if __name__ == "__main__":
    main()
