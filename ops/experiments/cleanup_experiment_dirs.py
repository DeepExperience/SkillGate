#!/usr/bin/env python3
"""Remove disposable experiment directories while preserving indexed runs.

Typical use:

  python3 ops/experiments/cleanup_experiment_dirs.py \
    --date 20260503 \
    --keep 20260503_0919_9b_baseline_quick30 \
    --keep 20260503_0919_9b_bs_retrieval_quick30 \
    --delete-unindexed \
    --execute

The default is dry-run.  A directory is kept when it is explicitly listed via
``--keep`` or appears in ``experiments/RUN_INDEX.jsonl`` unless
``--delete-unindexed`` is not requested.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REGISTRY_PATH = EXPERIMENTS_DIR / "RUN_INDEX.jsonl"


def indexed_run_ids() -> set[str]:
    if not REGISTRY_PATH.exists():
        return set()
    run_ids: set[str] = set()
    for raw_line in REGISTRY_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        run_ids.add(str(json.loads(line)["run_id"]))
    return run_ids


def looks_like_smoke_or_wrapper(path: Path) -> bool:
    name = path.name.lower()
    if any(token in name for token in ("smoke", "wrapper", "resume", "fixed")):
        return True
    files = {child.name for child in path.iterdir() if child.is_file()}
    dirs = {child.name for child in path.iterdir() if child.is_dir()}
    return bool(files & {"run.log", "run_quick9b.sh", "run_quick9b_w8_resume.sh"}) and not (
        {"results", "reports"} & dirs
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--date", required=True, help="Date folder under experiments/, e.g. 20260503")
    parser.add_argument("--keep", action="append", default=[], help="Run id/folder to preserve; repeatable")
    parser.add_argument("--delete-unindexed", action="store_true",
                        help="Delete folders not in RUN_INDEX.jsonl and not explicitly kept")
    parser.add_argument("--delete-smoke-wrappers", action="store_true", default=True,
                        help="Delete obvious wrapper/smoke folders even if --delete-unindexed is false")
    parser.add_argument("--execute", action="store_true", help="Actually delete; default is dry-run")
    args = parser.parse_args()

    date_dir = EXPERIMENTS_DIR / args.date
    if not date_dir.is_dir():
        raise SystemExit(f"missing date dir: {date_dir}")

    keep = set(args.keep) | indexed_run_ids()
    candidates: list[Path] = []
    for child in sorted(path for path in date_dir.iterdir() if path.is_dir()):
        if child.name in keep:
            continue
        if args.delete_unindexed or (args.delete_smoke_wrappers and looks_like_smoke_or_wrapper(child)):
            candidates.append(child)

    action = "DELETE" if args.execute else "DRY-RUN delete"
    for path in candidates:
        print(f"{action}: {path.relative_to(PROJECT_ROOT)}")
        if args.execute:
            shutil.rmtree(path)
    if not candidates:
        print("nothing to delete")


if __name__ == "__main__":
    main()
