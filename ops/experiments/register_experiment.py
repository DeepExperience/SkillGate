#!/usr/bin/env python3
"""Register experiment runs in a repository-level index.

The dated ``experiments/YYYYMMDD/<run_id>`` layout is good for append-only
artifacts, but it is bad as the only memory of why a run exists.  This helper
keeps a small machine-readable registry and rewrites ``experiments/INDEX.md``
so important runs can be found by intent, launcher, and status.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REGISTRY_PATH = EXPERIMENTS_DIR / "RUN_INDEX.jsonl"
MARKDOWN_PATH = EXPERIMENTS_DIR / "INDEX.md"


def read_registry() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not REGISTRY_PATH.exists():
        return records
    for raw_line in REGISTRY_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        record = json.loads(line)
        records[str(record["run_id"])] = record
    return records


def write_registry(records: dict[str, dict[str, Any]]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        records.values(),
        key=lambda item: (str(item.get("date", "")), str(item.get("run_id", ""))),
    )
    REGISTRY_PATH.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in ordered),
        encoding="utf-8",
    )


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def render_markdown(records: dict[str, dict[str, Any]]) -> str:
    rows = sorted(
        records.values(),
        key=lambda item: (str(item.get("date", "")), str(item.get("run_id", ""))),
        reverse=True,
    )
    lines = [
        "# Experiment Index",
        "",
        "This file is generated from `experiments/RUN_INDEX.jsonl` by `ops/experiments/register_experiment.py`.",
        "Use it to find the dated run folder, launcher, intent, and whether a run is canonical or disposable.",
        "",
        "## Policy",
        "",
        "- Keep one directory per meaningful experiment arm, for example `baseline` and `bs_retrieval`.",
        "- Do not keep wrapper-only directories after they finish; record their launcher in this index instead.",
        "- Smoke runs are disposable unless explicitly promoted; delete their folders after verifying they served their purpose.",
        "- Resume into the original `run_id`/folder; do not create `*_resume`, `*_fixed`, or wrapper folders unless the output semantics change.",
        "- Every meaningful run should record `launcher`, key Python entrypoints, intent, and status here.",
        "",
        "## Runs",
        "",
        "| date | run_id | status | kind | path | launcher | intent | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for record in rows:
        launcher = record.get("launcher", "")
        scripts = ", ".join(record.get("scripts", []))
        launcher_cell = launcher
        if scripts:
            launcher_cell = f"{launcher}<br>scripts: {scripts}" if launcher else f"scripts: {scripts}"
        lines.append(
            "| {date} | `{run_id}` | {status} | {kind} | `{path}` | `{launcher}` | {intent} | {notes} |".format(
                date=record.get("date", ""),
                run_id=record.get("run_id", ""),
                status=record.get("status", ""),
                kind=record.get("kind", ""),
                path=record.get("path", ""),
                launcher=launcher_cell,
                intent=str(record.get("intent", "")).replace("|", "\\|"),
                notes=str(record.get("notes", "")).replace("|", "\\|"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def infer_date(run_id: str, path: str) -> str:
    if len(run_id) >= 8 and run_id[:8].isdigit():
        return run_id[:8]
    path_parts = Path(path).parts
    for part in path_parts:
        if len(part) == 8 and part.isdigit():
            return part
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--date", default="")
    parser.add_argument("--kind", default="eval", help="eval/sft_collection/sft_train/diagnostic/smoke/etc.")
    parser.add_argument("--status", default="running")
    parser.add_argument("--launcher", default="")
    parser.add_argument("--scripts", default="", help="Comma-separated Python/script entrypoints")
    parser.add_argument("--intent", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--tags", default="", help="Comma-separated tags such as quick30,9b,baseline")
    args = parser.parse_args()

    records = read_registry()
    date = args.date or infer_date(args.run_id, args.path)
    now = datetime.now(timezone.utc).isoformat()
    old = records.get(args.run_id, {})
    record = {
        **old,
        "run_id": args.run_id,
        "date": date,
        "path": args.path,
        "kind": args.kind,
        "status": args.status,
        "launcher": args.launcher,
        "scripts": split_csv(args.scripts),
        "intent": args.intent,
        "notes": args.notes,
        "tags": split_csv(args.tags),
        "updated_at": now,
    }
    record.setdefault("created_at", now)
    records[args.run_id] = record
    write_registry(records)
    MARKDOWN_PATH.write_text(render_markdown(records), encoding="utf-8")
    print(f"registered {args.run_id} -> {MARKDOWN_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
