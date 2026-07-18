#!/usr/bin/env python3
"""Install the relocation-safe RL input rebuilds into canonical paths."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".validation/rl_rebuild"
BACKUP = ROOT / ".validation/old_rl_inputs"

MAPPING = {
    "factual": "experiments/rl/v2/parquet_4bench_factual_20260602",
    "noskill": "experiments/rl/v2/parquet_4bench_factual_noskills_20260617",
    "oracle1": "experiments/rl/v2/parquet_4bench_oracle1_20260612",
    "pair": "experiments/rl/v2/parquet_4bench_oracle_promptbc_pair_noskill_grpo_20260623",
    "slate_v8": "experiments/rl/v2/parquet_4bench_slate_regret_v8prod_20260708",
    "gold_v8": "experiments/rl/v2/parquet_4bench_slate_regret_v8prod_gold_stratified_v2_20260710",
    "hybrid": "experiments/rl/v2/parquet_4bench_slate_regret_hybridv8b0704d_gold_stratified_20260710",
    "mixed": "experiments/rl/v2/parquet_4bench_mixed_skill_bonus_compare_v8prod_allgold_20260710",
    "separated": "experiments/rl/v2/parquet_4bench_mixed_skill_separated_continuous_advantage_v8prod_allgold_20260710",
}


def rewrite_metadata(path: Path, replacements: list[tuple[str, str]]) -> None:
    for item in path.rglob("*"):
        if not item.is_file() or item.suffix.lower() not in {".json", ".md", ".txt", ".yaml", ".yml"}:
            continue
        text = item.read_text(encoding="utf-8")
        updated = text
        for before, after in replacements:
            updated = updated.replace(before, after)
        if updated != text:
            item.write_text(updated, encoding="utf-8")


def main() -> int:
    missing = [name for name in MAPPING if not (STAGE / name / "train.parquet").is_file()]
    if missing:
        raise SystemExit(f"missing staged rebuilds: {missing}")
    if BACKUP.exists():
        raise SystemExit(f"backup already exists: {BACKUP}")
    BACKUP.mkdir(parents=True)

    for alias, relative in MAPPING.items():
        canonical = ROOT / relative
        backup = BACKUP / alias
        if canonical.exists():
            canonical.rename(backup)
        canonical.parent.mkdir(parents=True, exist_ok=True)
        (STAGE / alias).rename(canonical)

    replacements: list[tuple[str, str]] = []
    for alias, relative in MAPPING.items():
        replacements.extend(
            [
                (str(STAGE / alias), str(ROOT / relative)),
                (f".validation/rl_rebuild/{alias}", relative),
            ]
        )
    for relative in MAPPING.values():
        rewrite_metadata(ROOT / relative, replacements)

    helper_dir = ROOT / "ops/workflows/rl_data_prep"
    sys.path.insert(0, str(helper_dir))
    from portable_fingerprint import portable_file_sha256  # noqa: PLC0415

    separated = ROOT / MAPPING["separated"] / "build_report.json"
    report = json.loads(separated.read_text(encoding="utf-8"))
    mixed_report = ROOT / MAPPING["mixed"] / "build_report.json"
    report["fingerprints"]["input_build_report_sha256"] = portable_file_sha256(mixed_report)
    separated.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"INSTALLED_RL_INPUTS count={len(MAPPING)} backup={BACKUP.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
