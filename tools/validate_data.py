#!/usr/bin/env python3
"""Validate handover data invariants without GPU, Ray, Docker, or network."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Checks:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def expect(self, condition: bool, message: str) -> None:
        if not condition:
            self.failures.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)


def check_skills(c: Checks) -> None:
    root = ROOT / "skill_libraries/merged"
    dirs = [p for p in root.iterdir() if p.is_dir()]
    skills = [p for p in dirs if (p / "SKILL.md").is_file()]
    c.expect(len(dirs) == 2045, f"merged child dirs expected 2045, got {len(dirs)}")
    c.expect(len(skills) == 2043, f"merged SKILL.md dirs expected 2043, got {len(skills)}")
    index_path = ROOT / "GeneralAgent/eval_scripts/skills_retrieval/skill_index_qwen3emb8b.pkl"
    with index_path.open("rb") as fh:
        index = pickle.load(fh)
    c.expect(len(index["skill_names"]) == 2043, "Qwen embedding index must contain 2043 skills")
    c.expect(getattr(index["embeddings"], "shape", None) == (2043, 4096), "embedding shape must be 2043x4096")
    missing = [path for path in index["skill_paths"] if not Path(path).is_dir()]
    c.expect(not missing, f"embedding index contains {len(missing)} missing skill paths")
    c.note(f"skills dirs={len(dirs)} indexed={len(index['skill_names'])}")


def check_sft(c: Checks) -> None:
    path = ROOT / (
        "GeneralAgent/sft_training/llamafactory_data/"
        "20260512_sft_campaign_clean_plus_claw_thinkwrap/"
        "agent_sft_campaign_20260512_clean_plus_claw_thinkwrap.json"
    )
    records = json.loads(path.read_text())
    c.expect(len(records) == 1708, f"final SFT records expected 1708, got {len(records)}")
    task_ids = {(r["metadata"]["bench"], str(r["metadata"]["task_id"])) for r in records}
    c.expect(len(task_ids) == 384, f"final SFT unique tasks expected 384, got {len(task_ids)}")
    bench = Counter(r["metadata"]["bench"] for r in records)
    expected = {"seta_synth": 1014, "swe_lite": 237, "claw": 206, "tb2": 147, "sb_ns": 104}
    c.expect(dict(bench) == expected, f"final SFT bench counts differ: {dict(bench)}")
    prefix = "<think>\n\n</think>\n\n<skill_reasoning>"
    bad_prefix = bad_schema = image_literals = 0
    for record in records:
        system = next((m.get("content", "") for m in record["messages"] if m.get("role") == "system"), "")
        assistant = next((m.get("content", "") for m in record["messages"] if m.get("role") == "assistant"), "")
        bad_schema += "<tools>" not in system
        bad_prefix += not assistant.startswith(prefix)
        image_literals += any("<image>" in str(m.get("content", "")) for m in record["messages"])
    c.expect(bad_schema == 0, f"SFT records missing injected tools schema: {bad_schema}")
    c.expect(bad_prefix == 0, f"SFT records missing think-wrap prefix: {bad_prefix}")
    c.expect(image_literals == 0, f"SFT records contain literal <image>: {image_literals}")
    c.note(f"sft records={len(records)} unique_tasks={len(task_ids)}")


def check_rl_split(c: Checks) -> None:
    split = json.loads((ROOT / "datasets/rl/rl_split_v2.json").read_text())
    train = {name: len(value["rl_train"]) for name, value in split["benches"].items()}
    evaluation = {name: len(value["rl_eval"]) for name, value in split["benches"].items()}
    c.expect(sum(train.values()) == 622, f"RL train tasks expected 622, got {sum(train.values())}")
    c.expect(sum(evaluation.values()) == 70, f"RL eval tasks expected 70, got {sum(evaluation.values())}")
    expected_eval = {"claw": 14, "sb_ns": 8, "seta_synth": 30, "swe_lite": 10, "tb2": 8}
    c.expect(evaluation == expected_eval, f"RL eval bench split differs: {evaluation}")
    task_tsv = ROOT / "ops/workflows/rl_eval/specs/eval70_v1/tasks.tsv"
    rows = [line.split("\t") for line in task_tsv.read_text().splitlines() if line.strip()]
    c.expect(len(rows) == 70, f"eval70 TSV expected 70 rows, got {len(rows)}")
    c.note(f"rl split train={sum(train.values())} eval={sum(evaluation.values())}")


def jsonl(path: Path):
    with path.open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def check_frozen_retrieval(c: Checks) -> None:
    root = ROOT / "skill_libraries/snapshots/rl/eval70_oracle_selfread_20260612"
    total = missing = 0
    for path in sorted(root.glob("*.jsonl")):
        for row in jsonl(path):
            total += 1
            candidates = row.get("reranked_top10") or row.get("reranked_top5") or row.get("coarse_top20") or []
            if not candidates or not Path(candidates[0]["skill_path"]).is_dir():
                missing += 1
    c.expect(total == 692, f"oracle retrieval snapshot expected 692 rows, got {total}")
    c.expect(missing == 0, f"oracle snapshot has {missing} missing skill paths")
    c.note(f"oracle snapshot rows={total}")


def _iter_skill_paths(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"path", "skill_path"} and isinstance(item, str) and item.startswith("/"):
                yield item
            yield from _iter_skill_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_skill_paths(item)


def check_slate_snapshots(c: Checks) -> None:
    families = (
        "slate_skills_20260704",
        "slate_skills_20260708_hard_negative_v8_production",
        "slate_skills_20260710_hybrid_v8body_0704desc",
    )
    total_rows = total_paths = 0
    missing: list[str] = []
    for family in families:
        manifest = ROOT / "skill_libraries/snapshots/rl" / family / "manifest"
        for split, expected in (("train", 491), ("eval70", 70)):
            rows = list(jsonl(manifest / f"slate_manifest_{split}.jsonl"))
            c.expect(len(rows) == expected, f"{family} {split} expected {expected} rows, got {len(rows)}")
            total_rows += len(rows)
            for row in rows:
                for raw in _iter_skill_paths(row):
                    total_paths += 1
                    if not Path(raw).is_dir():
                        missing.append(raw)
    c.expect(not missing, f"slate manifests contain {len(missing)} missing skill paths")
    c.note(f"slate manifests rows={total_rows} skill_paths={total_paths}")


def check_parquets(c: Checks) -> None:
    paths = [
        ROOT / "datasets/rl/parquet_4bench_base_20260523/train.parquet",
        ROOT / "datasets/rl/parquet_4bench_base_20260523/eval.parquet",
    ]
    try:
        import pyarrow.parquet as pq
    except ImportError:
        c.note("pyarrow unavailable; parquet existence checked but row counts skipped")
        c.expect(all(p.stat().st_size > 0 for p in paths), "canonical parquet is empty")
        return
    rows = [pq.ParquetFile(path).metadata.num_rows for path in paths]
    c.expect(rows == [622, 70], f"canonical parquet rows expected [622,70], got {rows}")
    c.note(f"canonical parquet rows={rows}")


def main() -> int:
    argparse.ArgumentParser().parse_args()
    checks = Checks()
    for fn in (
        check_skills,
        check_sft,
        check_rl_split,
        check_frozen_retrieval,
        check_slate_snapshots,
        check_parquets,
    ):
        try:
            fn(checks)
        except Exception as exc:
            checks.failures.append(f"{fn.__name__}: {type(exc).__name__}: {exc}")
    for note in checks.notes:
        print(f"OK: {note}")
    for failure in checks.failures:
        print(f"FAIL: {failure}")
    if checks.failures:
        print(f"DATA_CHECK_FAILED count={len(checks.failures)}")
        return 1
    print("DATA_CHECK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
