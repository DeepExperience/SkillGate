#!/usr/bin/env python3
"""Assemble a hard-negative slate by selecting per-task candidates from scored roots.

This is intentionally a small ops helper for the hard-negative misleading-skill
workflow. It does not promote or mutate the production slate. It copies selected
misleading SKILL.md directories from already-screened candidate roots into a new
root, then writes a traceable selection report. Use the regular
`slate_hard_negative_misleading.py materialize` command after this script.
"""

from __future__ import annotations

import os

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"empty name in {value!r}")
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return name, p


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def task_score(row: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    """Rank rows by observed hard-negative usefulness in no-skill mixed eval."""
    fail_after = int(row.get("fail_after_misleading_no_oracle") or 0)
    exposure = int(row.get("read_misleading_no_oracle") or 0)
    read_misleading = int(row.get("read_misleading") or 0)
    resolved = int(row.get("resolved") or 0)
    read_oracle = int(row.get("read_oracle") or 0)
    accepted = 1 if fail_after >= 1 and exposure >= 1 else 0
    return (accepted, fail_after, exposure, -resolved, read_misleading, -read_oracle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-slate-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--split", default="eval70")
    parser.add_argument("--source-root", action="append", type=parse_named_path, required=True,
                        help="Candidate source root as NAME=PATH.")
    parser.add_argument("--score", action="append", type=parse_named_path, required=True,
                        help="Acceptance score JSON as NAME=PATH.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    old_root = Path(args.old_slate_root)
    out_root = Path(args.out_root)
    if not old_root.is_absolute():
        old_root = ROOT / old_root
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    if out_root.exists() and not args.force:
        raise SystemExit(f"out root exists; pass --force to overwrite selected skills: {out_root}")

    source_roots = dict(args.source_root)
    score_paths = dict(args.score)
    if set(source_roots) != set(score_paths):
        raise SystemExit(f"source-root names {sorted(source_roots)} != score names {sorted(score_paths)}")

    scores: dict[str, dict[str, dict[str, Any]]] = {}
    for name, path in score_paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        scores[name] = {row["task_key"]: row for row in payload["tasks"]}

    manifest_path = old_root / "manifest" / f"slate_manifest_{args.split}.jsonl"
    manifest = read_jsonl(manifest_path)
    out_skills = out_root / "skills"
    if out_root.exists() and args.force:
        shutil.rmtree(out_root)
    out_skills.mkdir(parents=True, exist_ok=True)

    selection: list[dict[str, Any]] = []
    copied = 0
    fallback_missing = 0
    for row in manifest:
        task_key = row["task_key"]
        candidates = {name: by_task[task_key] for name, by_task in scores.items() if task_key in by_task}
        if not candidates:
            raise SystemExit(f"missing score rows for {task_key}")
        chosen = max(candidates, key=lambda name: task_score(candidates[name]))
        chosen_score = candidates[chosen]
        copied_names: list[str] = []
        missing_names: list[str] = []
        for entry in row["misleading"]:
            src = source_roots[chosen] / "skills" / entry["name"]
            dst = out_skills / entry["name"]
            if (src / "SKILL.md").is_file():
                shutil.copytree(src, dst, dirs_exist_ok=True)
                copied += 1
                copied_names.append(entry["name"])
            else:
                fallback_missing += 1
                missing_names.append(entry["name"])
        selection.append({
            "task_key": task_key,
            "bench": row["bench"],
            "task_id": row["task_id"],
            "chosen": chosen,
            "score": chosen_score,
            "rank_tuple": task_score(chosen_score),
            "copied_names": copied_names,
            "missing_names": missing_names,
        })

    counts = Counter(item["chosen"] for item in selection)
    weak_accept = sum(
        1 for item in selection
        if item["score"].get("fail_after_misleading_no_oracle", 0) >= 1
        and item["score"].get("read_misleading_no_oracle", 0) >= 1
    )
    summary = {
        "old_slate_root": str(old_root),
        "out_root": str(out_root),
        "split": args.split,
        "source_roots": {k: str(v) for k, v in source_roots.items()},
        "score_paths": {k: str(v) for k, v in score_paths.items()},
        "tasks": len(selection),
        "copied_skill_dirs": copied,
        "fallback_missing_skill_dirs": fallback_missing,
        "chosen_counts": dict(counts),
        "projected_weak_accept_tasks": weak_accept,
    }
    reports = out_root / "reports"
    write_json(reports / f"hybrid_selection_{args.split}.json", {
        "summary": summary,
        "selection": selection,
    })
    tsv_lines = ["bench\ttask_id\tchosen\tfail_after_misleading_no_oracle\tread_misleading_no_oracle\tresolved\tread_oracle\n"]
    for item in selection:
        s = item["score"]
        tsv_lines.append(
            f"{item['bench']}\t{item['task_id']}\t{item['chosen']}\t"
            f"{s.get('fail_after_misleading_no_oracle', 0)}\t"
            f"{s.get('read_misleading_no_oracle', 0)}\t"
            f"{s.get('resolved', 0)}\t{s.get('read_oracle', 0)}\n"
        )
    (reports / f"hybrid_selection_{args.split}.tsv").write_text("".join(tsv_lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
