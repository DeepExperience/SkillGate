#!/usr/bin/env python3
"""Convert RL train/eval task lists into Relax parquet — v2 (full pool).

Generates a prompt for every task in the v2 split, including those the SFT
campaign never saw. For each task::

    prompt[0] = system  ← per-bench canonical, copied verbatim from one SFT record
                          so chat_template renders bit-for-bit identical to SFT
                          (no double-injection, matches P0.3 contract).
    prompt[1] = user    ← task instruction loaded from the dataset:
                          * sb_ns / tb2 / seta_synth → instruction.md
                          * claw                     → task.yaml prompt.text
                          * swe_lite                 → parquet problem_statement

Run::

    python -m GeneralAgent.rl_data_prep.convert_to_relax_data_v2 \\
        --rl-split  /mnt/.../datasets/rl/rl_split_v2.json \\
        --sft-data  /mnt/.../sft_training/llamafactory_data/.../agent_sft_campaign_*.json \\
        --output-dir /mnt/.../datasets/rl/parquet_4bench_base_20260523
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GeneralAgent.task_exclusions import CONFIRMED_BAD_DOCKER_TASKS, is_bad_task


DEFAULT_RL_SPLIT = PROJECT_ROOT / "datasets/rl/rl_split_v2.json"
DEFAULT_SFT_DATA = PROJECT_ROOT / (
    "GeneralAgent/sft_training/llamafactory_data/"
    "20260512_sft_campaign_clean_plus_claw_thinkwrap/"
    "agent_sft_campaign_20260512_clean_plus_claw_thinkwrap.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets/rl/parquet_4bench_base_20260523"


BENCH_DATASET_DIRS = {
    "sb_ns": PROJECT_ROOT / "datasets/skillsbench/tasks",
    "tb2": PROJECT_ROOT / "datasets/terminal-bench-v2",
    "seta_synth": PROJECT_ROOT / "datasets/seta/dataset/seta_synth_top300",
    "claw": PROJECT_ROOT / "datasets/claw-eval/tasks",
}
SWE_PARQUET = PROJECT_ROOT / "datasets/swe-gym/lite/data/train-00000-of-00001.parquet"

BROKEN_TASKS = set(CONFIRMED_BAD_DOCKER_TASKS)


# ---------------------------------------------------------------------------
# SFT record indexing — used as both ground-truth (for SFT-seen tasks) and
# as the source of per-bench canonical system messages (for SFT-unseen tasks).
# ---------------------------------------------------------------------------
def _index_sft_records(sft_data: list[dict[str, Any]]):
    index: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, rec in enumerate(sft_data):
        meta = rec.get("metadata") or {}
        b, t = meta.get("bench"), meta.get("task_id")
        if b and t:
            index[(b, t)].append(i)
    return index


def _build_canonical_system_per_bench(sft_data, sft_index):
    """For each bench, pick the system message from its first SFT record.

    Used when minting prompts for SFT-unseen tasks: we re-use the exact
    system content so chat_template renders identically to SFT-time.
    """
    canonical: dict[str, str] = {}
    for (bench, _task_id), idxs in sft_index.items():
        if bench in canonical:
            continue
        rec = sft_data[idxs[0]]
        sys_msg = (rec.get("messages") or [{}])[0]
        if sys_msg.get("role") == "system" and sys_msg.get("content"):
            canonical[bench] = sys_msg["content"]
    return canonical


# ---------------------------------------------------------------------------
# Per-bench user-message loaders
# ---------------------------------------------------------------------------
def _load_instruction_md(bench: str, task_id: str) -> str:
    base = BENCH_DATASET_DIRS[bench]
    path = base / task_id / "instruction.md"
    if not path.is_file():
        raise FileNotFoundError(f"{bench}/{task_id}: missing instruction.md at {path}")
    return path.read_text(encoding="utf-8").strip()


def _load_claw_prompt(task_id: str) -> str:
    import yaml  # type: ignore

    base = BENCH_DATASET_DIRS["claw"]
    path = base / task_id / "task.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"claw/{task_id}: missing task.yaml at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompt = (data.get("prompt") or {}).get("text") or ""
    if not prompt:
        raise RuntimeError(f"claw/{task_id}: task.yaml has empty prompt.text")
    return prompt.strip()


def _load_swe_problem_statement(swe_parquet_index: dict[str, str], task_id: str) -> str:
    ps = swe_parquet_index.get(task_id)
    if ps is None:
        raise RuntimeError(f"swe_lite/{task_id}: not in parquet (build index first)")
    return ps.strip()


def _load_user_message(
    bench: str,
    task_id: str,
    swe_parquet_index: dict[str, str],
) -> str:
    if bench in ("sb_ns", "tb2", "seta_synth"):
        return _load_instruction_md(bench, task_id)
    if bench == "claw":
        return _load_claw_prompt(task_id)
    if bench == "swe_lite":
        return _load_swe_problem_statement(swe_parquet_index, task_id)
    raise ValueError(f"unsupported bench: {bench!r}")


def _build_swe_parquet_index() -> dict[str, str]:
    """Map ``instance_id → problem_statement`` once so per-task lookups are O(1)."""
    import pyarrow.parquet as pq  # type: ignore

    if not SWE_PARQUET.is_file():
        return {}
    table = pq.read_table(str(SWE_PARQUET))
    rows = table.select(["instance_id", "problem_statement"]).to_pylist()
    return {r["instance_id"]: r["problem_statement"] for r in rows if r["instance_id"]}


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------
def _build_row_from_sft(
    sft_data: list[dict[str, Any]],
    sft_index_entries: list[int],
    task_id: str,
    bench: str,
) -> dict[str, Any]:
    """Prompt comes verbatim from the earliest SFT record for this task."""
    sft_idx = sft_index_entries[0]
    rec = sft_data[sft_idx]
    msgs = rec.get("messages") or []
    if not msgs or msgs[0].get("role") != "system":
        raise RuntimeError(f"{bench}/{task_id}: SFT record missing system")
    system_msg = {"role": "system", "content": msgs[0]["content"]}
    user_msg = next((m for m in msgs[1:] if m.get("role") == "user"), None)
    if user_msg is None:
        raise RuntimeError(f"{bench}/{task_id}: SFT record has no first user turn")
    prompt = [system_msg, {"role": "user", "content": user_msg["content"]}]
    meta = rec.get("metadata") or {}
    return _wrap_row(
        prompt=prompt,
        task_id=task_id,
        bench=bench,
        source="sft_seen",
        sft_record_idx=sft_idx,
        prompt_profile=meta.get("prompt_profile"),
        injected_skills=meta.get("injected_skill_names") or [],
    )


def _build_row_from_dataset(
    canonical_systems: dict[str, str],
    task_id: str,
    bench: str,
    user_text: str,
) -> dict[str, Any]:
    sys_content = canonical_systems.get(bench)
    if not sys_content:
        raise RuntimeError(
            f"{bench}/{task_id}: no canonical system available "
            f"(no SFT records for bench {bench!r})"
        )
    prompt = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_text},
    ]
    return _wrap_row(
        prompt=prompt,
        task_id=task_id,
        bench=bench,
        source="dataset_loader",
        sft_record_idx=None,
        prompt_profile=None,
        injected_skills=[],
    )


def _wrap_row(
    *,
    prompt,
    task_id,
    bench,
    source,
    sft_record_idx,
    prompt_profile,
    injected_skills,
):
    return {
        "prompt": prompt,
        "reward_model": {
            "task_id": task_id,
            "bench": bench,
            "ground_truth": None,
        },
        "extra_info": {
            "task_id": task_id,
            "bench": bench,
            "task_kwargs": {"task_id": task_id, "bench": bench},
            "retrieval_skills_top_n": list(injected_skills),
            "source": source,
            "sft_record_idx": sft_record_idx,
            "sft_prompt_profile": prompt_profile,
        },
    }


# ---------------------------------------------------------------------------
# Dataset build loop
# ---------------------------------------------------------------------------
def build_split(
    rl_split: dict,
    sft_data: list[dict[str, Any]],
    sft_index,
    canonical_systems: dict[str, str],
    swe_index: dict[str, str],
    *,
    split_name: str,
):
    rows: list[dict] = []
    per_bench_src: dict[str, dict[str, int]] = defaultdict(lambda: {"sft_seen": 0, "dataset_loader": 0})
    failures: list[str] = []

    for bench, split in rl_split["benches"].items():
        for task_id in split[split_name]:
            if is_bad_task(bench, task_id):
                failures.append(f"{bench}/{task_id}: skipped known-broken dockerfile task")
                continue
            sft_idxs = sft_index.get((bench, task_id)) or []
            try:
                if sft_idxs:
                    row = _build_row_from_sft(sft_data, sft_idxs, task_id, bench)
                    per_bench_src[bench]["sft_seen"] += 1
                else:
                    user_text = _load_user_message(bench, task_id, swe_index)
                    row = _build_row_from_dataset(canonical_systems, task_id, bench, user_text)
                    per_bench_src[bench]["dataset_loader"] += 1
            except Exception as exc:
                failures.append(f"{bench}/{task_id}: {type(exc).__name__}: {exc}")
                continue
            rows.append(row)

    return rows, dict(per_bench_src), failures


def write_parquet(rows, output_path: Path):
    if not rows:
        raise RuntimeError(f"refusing to write empty parquet at {output_path}")
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    table = pa.Table.from_pylist(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rl-split", default=DEFAULT_RL_SPLIT)
    parser.add_argument("--sft-data", default=DEFAULT_SFT_DATA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rl_split_path = Path(args.rl_split)
    sft_data_path = Path(args.sft_data)
    output_dir = Path(args.output_dir)
    if not rl_split_path.is_file() or not sft_data_path.is_file():
        print("[FAIL] inputs missing", file=sys.stderr)
        return 2

    print(f"[info] loading {rl_split_path}")
    rl_split = json.loads(rl_split_path.read_text(encoding="utf-8"))
    print(f"[info] loading {sft_data_path}")
    sft_data = json.loads(sft_data_path.read_text(encoding="utf-8"))
    sft_index = _index_sft_records(sft_data)
    canonical_systems = _build_canonical_system_per_bench(sft_data, sft_index)
    print(f"[info] canonical system messages per bench: {sorted(canonical_systems.keys())}")
    swe_index = _build_swe_parquet_index()
    print(f"[info] SWE parquet index size: {len(swe_index)}")

    for split_name, file_name in (("rl_train", "train.parquet"), ("rl_eval", "eval.parquet")):
        rows, per_bench, failures = build_split(
            rl_split, sft_data, sft_index, canonical_systems, swe_index, split_name=split_name
        )
        if failures:
            print(f"[WARN] split={split_name} dropped {len(failures)} task(s):")
            for line in failures[:10]:
                print(f"    {line}")
            if len(failures) > 10:
                print(f"    ... and {len(failures) - 10} more")
        write_parquet(rows, output_dir / file_name)
        print(f"  split={split_name}: wrote {len(rows)} rows; per-bench source counts:")
        for b, c in sorted(per_bench.items()):
            print(f"    {b}: sft_seen={c['sft_seen']}, dataset_loader={c['dataset_loader']}")

    print(f"\nv2 parquet build complete → {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
