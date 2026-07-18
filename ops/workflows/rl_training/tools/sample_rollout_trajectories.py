#!/usr/bin/env python3
# NOTE: Migrated workflow helper copy. Source: experiments/rl/v2/launch/sample_rollout_trajectories.py
# Original historical script is archived during workflow cleanup; maintain this copy going forward.
"""Sample representative RL rollout trajectories as JSON-only artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SKILL_PATH_RE = re.compile(r"/(?:root|workspace)/\.claude/skills/|\.claude/skills/")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def score(row: dict[str, Any]) -> float:
    reward = row.get("reward") or {}
    value = reward.get("score", reward.get("raw_score", 0.0))
    try:
        return float(value)
    except Exception:
        return 0.0


def text_contains_skill(row: dict[str, Any]) -> bool:
    return bool(SKILL_PATH_RE.search(str(row.get("response", ""))))


def pick_rows(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    picked: list[tuple[str, dict[str, Any]]] = []
    if not rows:
        return picked
    best = max(rows, key=lambda row: (score(row), row.get("total_length") or 0))
    worst = min(rows, key=lambda row: (score(row), row.get("total_length") or 0))
    longest = max(rows, key=lambda row: row.get("total_length") or 0)
    skill_rows = [row for row in rows if text_contains_skill(row)]
    if skill_rows:
        skill = max(skill_rows, key=lambda row: (score(row), row.get("total_length") or 0))
        picked.append(("skill", skill))
    picked.extend([("pass", best), ("fail", worst), ("long", longest)])
    dedup: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple[Any, Any]] = set()
    for label, row in picked:
        key = (row.get("group_index"), row.get("sample_index"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append((label, row))
    return dedup


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rollout_id": row.get("rollout_id"),
        "group_index": row.get("group_index"),
        "sample_index": row.get("sample_index"),
        "status": row.get("status"),
        "label": row.get("label"),
        "reward": row.get("reward"),
        "response_length": row.get("response_length"),
        "total_length": row.get("total_length"),
        "used_skill_path": text_contains_skill(row),
        "prompt": row.get("prompt", ""),
        "response": row.get("response", ""),
    }


def write_samples(step: int, rows: list[dict[str, Any]], out_dir: Path, source: Path) -> None:
    step_dir = out_dir / f"step_{step:04d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    picked = pick_rows(rows)
    sample_records: list[dict[str, Any]] = []
    for label, row in picked:
        name = f"{label}_g{row.get('group_index')}_s{row.get('sample_index')}.json"
        payload = compact_row(row)
        (step_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        bench = (row.get("label") or {}).get("bench") or (row.get("reward") or {}).get("bench")
        task = (row.get("label") or {}).get("task_id") or (row.get("reward") or {}).get("task_id")
        sample_records.append({
            "label": label,
            "file": name,
            "bench": bench,
            "task_id": task,
            "score": score(row),
            "total_length": row.get("total_length"),
            "response_length": row.get("response_length"),
            "used_skill_path": text_contains_skill(row),
        })
    scores = [score(row) for row in rows]
    manifest = {
        "schema_version": 1,
        "step": step,
        "source": str(source),
        "total_rows": len(rows),
        "score": {
            "min": min(scores, default=0.0),
            "mean": sum(scores) / len(scores) if scores else 0.0,
            "max": max(scores, default=0.0),
        },
        "used_skill_path_rows": sum(text_contains_skill(row) for row in rows),
        "samples": sample_records,
    }
    (step_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--step", type=int, action="append", required=True)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    for step in args.step:
        source = checkpoint_dir / "rollout_result" / "train" / f"{step}.jsonl"
        if not source.exists():
            print(f"[sample] skip missing step={step}: {source}")
            continue
        rows = load_jsonl(source)
        write_samples(step, rows, out_dir, source)
        print(f"[sample] wrote step={step} rows={len(rows)} -> {out_dir / f'step_{step:04d}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
