#!/usr/bin/env python3
"""Create a quick-eval plan for the fixed holdout split.

Unlike SFT collection plans, quick eval does not add hidden use-skill or
no-skill nudges. It supports two table rows:

  - baseline: no retrieval skills injected.
  - retrieval: retrieval top-k is injected, and the model decides whether to
    read skills.
"""

from __future__ import annotations

import argparse
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_CONFIG,
    display_path,
    experiment_plan_path,
    filter_known_bad_tasks,
    load_json,
    load_retrieval_task_ids,
    repo_path,
)
from make_trial_plan import make_record, write_jsonl


def strip_swe_train_instance_file(record: dict[str, Any]) -> None:
    """Use quick-test SWE instances instead of the 100-task train subset.

    The shared `make_record()` helper is intentionally biased toward SFT data
    collection and always pins SWE to `swe_lite_100.txt`. The quick holdout
    split uses the legacy SWE eval instances, so keeping that train file makes
    heldout ids such as `facebookresearch__hydra-2189` filter down to zero
    runnable instances. Quick eval should therefore omit `--instance-file` and
    let `run_unified_swe.py` use its built-in ALL_IMAGES list.
    """
    if record.get("bench") != "swe_lite":
        return
    argv = list(record.get("argv") or [])
    if "--instance-file" not in argv:
        return
    index = argv.index("--instance-file")
    del argv[index:index + 2]
    record["argv"] = argv
    record["command_preview"] = shlex.join(argv)


def build_eval_plan(
    config: dict[str, Any],
    split_payload: dict[str, Any],
    run_id: str,
    date: str,
    benches: list[str],
    model: str,
    trials: int,
    arm: str,
    max_turns: int,
    max_time: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    if arm not in {"baseline", "retrieval"}:
        raise ValueError(f"unsupported arm {arm!r}; expected baseline or retrieval")

    retrieval_files = config["frozen_retrieval"]["files"]
    retrieval_top_n = int(config["budgets"]["retrieval_top_n"])
    coverage = {
        bench: load_retrieval_task_ids(path)
        for bench, path in retrieval_files.items()
    }

    for bench in benches:
        if bench not in split_payload["benches"]:
            raise ValueError(f"bench {bench!r} not in quick split")
        tasks = filter_known_bad_tasks(bench, list(split_payload["benches"][bench]["test"]))
        retrieval_jsonl = retrieval_files.get(bench)
        for task_id in tasks:
            retrieval_covered = str(task_id) in coverage.get(bench, set())
            if arm == "retrieval" and not retrieval_covered:
                warnings.append(f"{bench}/{task_id}: missing retrieval entry in {retrieval_jsonl}")
            for trial_index in range(trials):
                record = make_record(
                    run_id=run_id,
                    date=date,
                    bench=bench,
                    task_id=str(task_id),
                    split="quick_test",
                    mode=f"eval_{arm}",
                    model_role="eval",
                    model=model,
                    arm=arm,
                    trial_index=trial_index,
                    retrieval_jsonl=retrieval_jsonl if arm == "retrieval" else None,
                    retrieval_top_n=retrieval_top_n,
                    max_turns=max_turns,
                    max_time=max_time,
                    retrieval_covered=retrieval_covered if arm == "retrieval" else True,
                    implicit_mode="",
                )
                record.setdefault("env", {})["UNIFIED_ROLLOUT_WALLCLOCK_CAP_SEC"] = str(max_time)
                record.setdefault("env", {})["UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS"] = "1"
                strip_swe_train_instance_file(record)
                records.append(record)
    return records, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", default="GeneralAgent/sft_data_collection/outputs/splits/default/quick_test/quick30/holdout_split.json")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--arm", choices=["baseline", "retrieval"], default="retrieval")
    parser.add_argument("--benches", nargs="+", default=["claw", "tb2", "sb_ns", "seta_synth", "swe_lite"])
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-time", type=int, default=850)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    run_id = args.run_id or f"{args.date}_quick_holdout_eval"
    config = load_json(args.config)
    split_payload = load_json(args.split)
    records, warnings = build_eval_plan(
        config=config,
        split_payload=split_payload,
        run_id=run_id,
        date=args.date,
        benches=args.benches,
        model=args.model,
        trials=args.trials,
        arm=args.arm,
        max_turns=args.max_turns,
        max_time=args.max_time,
    )
    output_path = repo_path(args.out) if args.out else experiment_plan_path(run_id)
    write_jsonl(output_path, records)
    warning_path = output_path.with_suffix(".warnings.txt")
    warning_path.write_text("\n".join(warnings) + ("\n" if warnings else ""), encoding="utf-8")
    print(f"wrote {len(records)} eval records to {display_path(output_path)}")
    print(f"warnings: {len(warnings)} ({display_path(warning_path)})")


if __name__ == "__main__":
    main()
