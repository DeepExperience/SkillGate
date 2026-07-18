#!/usr/bin/env python3
"""Build the gold-always mixed-only parquet for bonus-comparison GRPO.

The source is the frozen v8-production SlateRL pair parquet.  Only its
``slate_grpo`` train row for each task is retained, and every prompt is rebuilt
from the frozen production manifest as exactly:

    1 oracle + 5 misleading + 5 relevant + 5 irrelevant = 16 skills

Unlike the paired SlateRL data, there is no no-skill arm and no gold-absent
coin flip.  Category names are written into ``extra_info`` for reward-time
attribution, but are not exposed in the prompt.  Build and validate modes hard
fail unless train is exactly 491/491 unique, gold-present, 16-skill tasks.

Resume behavior: this builder is deterministic.  Existing output is validated
against source/output fingerprints; rebuild into a new output directory if a
frozen input changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TRAIN_KIND = "mixed_bonus_compare_grpo"
EVAL_KIND = "mixed_bonus_compare_eval"
EXPECTED_SLATE_COUNTS = {"oracle": 1, "misleading": 5, "relevant": 5, "irrelevant": 5}
EXPECTED_SLATE_SIZE = sum(EXPECTED_SLATE_COUNTS.values())
AVAILABLE_SKILLS_RE = re.compile(r"<available_skills>.*?</available_skills>", re.DOTALL)
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---", re.DOTALL)
PROMPT_SKILL_NAME_RE = re.compile(r"<name>([^<]+)</name>")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, dict):
            return dict(item)
    raise TypeError(f"extra_info is not a dict: {type(value)!r}")


def _plain_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


def _messages(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if isinstance(prompt, (list, tuple)):
        return [dict(item) if isinstance(item, dict) else {"role": "user", "content": str(item)} for item in prompt]
    return [{"role": "user", "content": "" if prompt is None else str(prompt)}]


def _prompt_text(prompt: Any) -> str:
    return "\n".join(str(message.get("content", "")) for message in _messages(prompt))


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = f"{row['bench']}::{row['task_id']}"
            if key in rows:
                raise SystemExit(f"duplicate manifest task {key} at {path}:{line_number}")
            rows[key] = row
    return rows


def _entry_path(entry: dict[str, Any]) -> Path:
    path = Path(str(entry["path"]))
    return path if path.is_absolute() else ROOT / path


def _frontmatter_description(entry: dict[str, Any]) -> str:
    skill_md = _entry_path(entry) / "SKILL.md"
    if not skill_md.is_file():
        raise SystemExit(f"missing frozen skill file: {skill_md}")
    text = skill_md.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    description = ""
    if match:
        description_match = re.search(
            r"^description:\s*(.+?)(?=\n[a-zA-Z_]+:|\Z)",
            match.group(1),
            re.DOTALL | re.MULTILINE,
        )
        if description_match:
            description = " ".join(description_match.group(1).split()).strip("\"'")
    return (description or f"Skill document {entry['name']}.")[:300]


def _seeded_shuffle(
    entries: list[dict[str, Any]], bench: str, task_id: str, *, evaluation: bool
) -> None:
    # Reuse the canonical v8-production ordering.  Skill position is visible
    # to the model, so a variant-specific seed would add a comparison variable.
    tag = "slate-shuffle" if evaluation else "slate-train-shuffle"
    payload = f"{tag}::{bench}::{task_id}".encode()
    seed = int(hashlib.sha256(payload).hexdigest()[:12], 16)
    random.Random(seed).shuffle(entries)


def _validate_manifest_row(row: dict[str, Any], key: str) -> dict[str, list[str]]:
    category_names: dict[str, list[str]] = {}
    all_names: list[str] = []
    for category, expected_count in EXPECTED_SLATE_COUNTS.items():
        entries = list(row.get(category) or [])
        names = [str(entry.get("name") or "").strip() for entry in entries]
        if len(entries) != expected_count or any(not name for name in names):
            raise SystemExit(
                f"manifest {key} requires {expected_count} {category}, got {len(entries)}"
            )
        if len(set(names)) != expected_count:
            raise SystemExit(f"manifest {key} has duplicate {category} names: {names}")
        for entry in entries:
            skill_md = _entry_path(entry) / "SKILL.md"
            if not skill_md.is_file():
                raise SystemExit(f"manifest {key} missing {category} skill: {skill_md}")
        category_names[category] = names
        all_names.extend(names)
    if len(set(all_names)) != EXPECTED_SLATE_SIZE:
        raise SystemExit(f"manifest {key} categories overlap: {all_names}")
    return category_names


def _ordered_entries(
    row: dict[str, Any], key: str, *, evaluation: bool
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    category_names = _validate_manifest_row(row, key)
    entries = [
        dict(entry)
        for category in ("oracle", "misleading", "relevant", "irrelevant")
        for entry in row[category]
    ]
    _seeded_shuffle(
        entries,
        str(row["bench"]),
        str(row["task_id"]),
        evaluation=evaluation,
    )
    return entries, category_names


def _build_slate_block(entries: list[dict[str, Any]]) -> str:
    parts = ["<available_skills>"]
    for entry in entries:
        parts.append(
            "  <skill>\n"
            f"    <name>{entry['name']}</name>\n"
            f"    <description>{_frontmatter_description(entry)}</description>\n"
            f"    <location>/root/.claude/skills/{entry['name']}/SKILL.md</location>\n"
            "  </skill>"
        )
    parts.append("</available_skills>")
    return "\n".join(parts)


def _replace_or_insert_slate(prompt: Any, block: str, key: str) -> np.ndarray:
    messages = _messages(prompt)
    changed = False
    for message in messages:
        content = message.get("content", "")
        if changed or message.get("role") != "system" or not isinstance(content, str):
            continue
        matches = list(AVAILABLE_SKILLS_RE.finditer(content))
        if len(matches) > 1:
            raise SystemExit(f"prompt {key} contains {len(matches)} available_skills blocks")
        if matches:
            message["content"] = AVAILABLE_SKILLS_RE.sub(lambda _: block, content, count=1)
            changed = True
            continue
        for marker in ("\n## Memory Recall\n", "\n## Runtime\n"):
            index = content.find(marker)
            if index >= 0:
                message["content"] = content[:index].rstrip() + "\n" + block + "\n" + content[index:]
                changed = True
                break
    if not changed:
        raise SystemExit(f"could not replace/insert mixed slate for {key}")
    return np.array(messages, dtype=object)


def _normalize_extra(
    source: dict[str, Any],
    entries: list[dict[str, Any]],
    category_names: dict[str, list[str]],
    *,
    evaluation: bool,
) -> dict[str, Any]:
    extra = dict(source)
    kind = EVAL_KIND if evaluation else TRAIN_KIND
    extra["update_kind"] = kind
    extra["hybrid_update_kind"] = kind
    extra["retrieval_skills_top_n"] = [str(entry["name"]) for entry in entries]
    extra["slate_contains_gold"] = 1.0
    extra["slate_gold_name"] = category_names["oracle"][0]
    extra["slate_misleading_names"] = list(category_names["misleading"])
    extra["slate_relevant_names"] = list(category_names["relevant"])
    extra["slate_irrelevant_names"] = list(category_names["irrelevant"])
    extra["slate_size"] = float(EXPECTED_SLATE_SIZE)
    extra["oracle_skill_mode"] = "slate_path"
    extra["mixed_skill_bonus_compare"] = 1.0
    extra["mixed_skill_bonus_category_version"] = "v8prod_manifest_v1"
    extra["hybrid_is_shadow"] = 0.0
    extra["hybrid_grpo_weight"] = 0.0 if evaluation else 1.0
    extra["hybrid_shadow_weight"] = 0.0
    for field in list(extra):
        if field.startswith("relax_pair_") or field.startswith("hybrid_pair_"):
            extra.pop(field, None)
    return extra


def _task_key(extra: dict[str, Any]) -> str:
    return f"{extra.get('bench')}::{extra.get('task_id')}"


def _build_train(
    source: pd.DataFrame,
    manifest: dict[str, dict[str, Any]],
    expected_tasks: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_rows: dict[str, pd.Series] = {}
    source_kind_counts: Counter[str] = Counter()
    for _, row in source.iterrows():
        extra = _plain_dict(row["extra_info"])
        kind = str(extra.get("update_kind") or "").strip().lower()
        source_kind_counts[kind] += 1
        if kind != "slate_grpo":
            continue
        key = _task_key(extra)
        if key in source_rows:
            raise SystemExit(f"duplicate source slate row: {key}")
        source_rows[key] = row

    if len(manifest) != expected_tasks:
        raise SystemExit(f"train manifest must contain {expected_tasks} tasks, got {len(manifest)}")
    if set(source_rows) != set(manifest):
        raise SystemExit(
            "source slate tasks do not exactly match train manifest: "
            f"missing_source={sorted(set(manifest) - set(source_rows))[:10]}, "
            f"missing_manifest={sorted(set(source_rows) - set(manifest))[:10]}"
        )

    output_rows: list[pd.Series] = []
    bench_counts: Counter[str] = Counter()
    for key in sorted(manifest):
        source_row = source_rows[key].copy()
        source_extra = _plain_dict(source_row["extra_info"])
        manifest_row = manifest[key]
        entries, category_names = _ordered_entries(
            manifest_row, key, evaluation=False
        )
        source_row["prompt"] = _replace_or_insert_slate(
            source_row["prompt"], _build_slate_block(entries), key
        )
        source_row["extra_info"] = _normalize_extra(
            source_extra, entries, category_names, evaluation=False
        )
        output_rows.append(source_row)
        bench_counts[str(source_extra.get("bench"))] += 1

    output = pd.DataFrame(output_rows).reset_index(drop=True)
    if len(output) != expected_tasks:
        raise SystemExit(f"train output must contain {expected_tasks} rows, got {len(output)}")
    return output, {
        "rows": len(output),
        "unique_tasks": len(source_rows),
        "gold_present": len(output),
        "gold_absent": 0,
        "slate_size_16": len(output),
        "source_kind_counts": dict(source_kind_counts),
        "bench_counts": dict(bench_counts),
    }


def _build_eval(
    source: pd.DataFrame,
    manifest: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_rows: list[pd.Series] = []
    seen: set[str] = set()
    bench_counts: Counter[str] = Counter()
    for _, row in source.iterrows():
        source_row = row.copy()
        source_extra = _plain_dict(source_row["extra_info"])
        key = _task_key(source_extra)
        if key in seen:
            raise SystemExit(f"duplicate eval task: {key}")
        seen.add(key)
        manifest_row = manifest.get(key)
        if manifest_row is None:
            raise SystemExit(f"eval task missing from frozen manifest: {key}")
        entries, category_names = _ordered_entries(
            manifest_row, key, evaluation=True
        )
        source_row["prompt"] = _replace_or_insert_slate(
            source_row["prompt"], _build_slate_block(entries), key
        )
        source_row["extra_info"] = _normalize_extra(
            source_extra, entries, category_names, evaluation=True
        )
        output_rows.append(source_row)
        bench_counts[str(source_extra.get("bench"))] += 1
    output = pd.DataFrame(output_rows).reset_index(drop=True)
    return output, {
        "rows": len(output),
        "unique_tasks": len(seen),
        "gold_present": len(output),
        "gold_absent": 0,
        "slate_size_16": len(output),
        "bench_counts": dict(bench_counts),
    }


def _validate_frame(
    frame: pd.DataFrame,
    *,
    split: str,
    expected_kind: str,
    expected_rows: int,
) -> dict[str, Any]:
    if len(frame) != expected_rows:
        raise SystemExit(f"{split}: expected {expected_rows} rows, got {len(frame)}")
    problems: list[str] = []
    keys: set[str] = set()
    benches: Counter[str] = Counter()
    for row_index, row in frame.iterrows():
        extra = _plain_dict(row["extra_info"])
        key = _task_key(extra)
        keys.add(key)
        benches[str(extra.get("bench"))] += 1
        retrieval = _plain_list(extra.get("retrieval_skills_top_n"))
        gold = str(extra.get("slate_gold_name") or "")
        misleading = _plain_list(extra.get("slate_misleading_names"))
        relevant = _plain_list(extra.get("slate_relevant_names"))
        irrelevant = _plain_list(extra.get("slate_irrelevant_names"))
        categories = [gold] + misleading + relevant + irrelevant
        prompt_names = PROMPT_SKILL_NAME_RE.findall(_prompt_text(row["prompt"]))
        if extra.get("update_kind") != expected_kind or extra.get("hybrid_update_kind") != expected_kind:
            problems.append(f"row{row_index}:{key}:bad-kind")
        if float(extra.get("slate_contains_gold") or 0.0) != 1.0:
            problems.append(f"row{row_index}:{key}:gold-absent")
        if len(retrieval) != EXPECTED_SLATE_SIZE or len(set(retrieval)) != EXPECTED_SLATE_SIZE:
            problems.append(f"row{row_index}:{key}:retrieval={len(retrieval)}/{len(set(retrieval))}")
        if not gold or len(misleading) != 5 or len(relevant) != 5 or len(irrelevant) != 5:
            problems.append(f"row{row_index}:{key}:bad-category-counts")
        if set(categories) != set(retrieval) or len(set(categories)) != EXPECTED_SLATE_SIZE:
            problems.append(f"row{row_index}:{key}:category-coverage")
        if prompt_names != retrieval:
            problems.append(f"row{row_index}:{key}:prompt-metadata-order-mismatch")
        if _prompt_text(row["prompt"]).count("<available_skills>") != 1:
            problems.append(f"row{row_index}:{key}:available-skills-block-count")
        if float(extra.get("mixed_skill_bonus_compare") or 0.0) != 1.0:
            problems.append(f"row{row_index}:{key}:missing-mode-marker")
    if len(keys) != expected_rows:
        problems.append(f"unique-tasks={len(keys)} expected={expected_rows}")
    if problems:
        raise SystemExit(f"{split} validation failed ({len(problems)}): {problems[:20]}")
    return {
        "rows": len(frame),
        "unique_tasks": len(keys),
        "gold_present": len(frame),
        "gold_absent": 0,
        "slate_size_16": len(frame),
        "bench_counts": dict(benches),
    }


def _runtime_skill_path(name: str, roots: list[Path]) -> Path | None:
    merged = ROOT / "skill_libraries/merged"
    candidates = [root / name for root in roots]
    candidates.extend((merged / name, merged / f"hw-{name}"))
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if slug and slug != name:
        candidates.extend((merged / slug, merged / f"hw-{slug}"))
        if slug.endswith("-skills"):
            candidates.append(merged / slug[: -len("-skills")])
    for candidate in candidates:
        if (candidate / "SKILL.md").is_file():
            return candidate.resolve()
    return None


def _validate_runtime_resolution(
    manifests: list[dict[str, dict[str, Any]]],
    roots: list[Path],
) -> None:
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for manifest in manifests:
        for key, row in manifest.items():
            _validate_manifest_row(row, key)
            for category in EXPECTED_SLATE_COUNTS:
                for entry in row[category]:
                    expected = _entry_path(entry).resolve()
                    pair = (str(entry["name"]), str(expected))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    actual = _runtime_skill_path(str(entry["name"]), roots)
                    if actual != expected:
                        problems.append(f"{entry['name']}:{actual}!={expected}")
    if problems:
        raise SystemExit(
            f"runtime skill resolution failed ({len(problems)}): {problems[:20]}"
        )


def _skill_content_sha256(
    manifests: list[dict[str, dict[str, Any]]],
) -> str:
    digest = hashlib.sha256()
    paths = {
        str((_entry_path(entry) / "SKILL.md").resolve())
        for manifest in manifests
        for row in manifest.values()
        for category in EXPECTED_SLATE_COUNTS
        for entry in row[category]
    }
    for raw_path in sorted(paths):
        path = Path(raw_path)
        digest.update(raw_path.encode() + b"\0")
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def _fingerprints(
    args: argparse.Namespace,
    manifests: list[dict[str, dict[str, Any]]],
) -> dict[str, str]:
    return {
        "input_train_sha256": _file_sha256(args.input_dir / "train.parquet"),
        "input_eval_sha256": _file_sha256(args.input_dir / "eval.parquet"),
        "train_manifest_sha256": _file_sha256(args.train_manifest),
        "eval_manifest_sha256": _file_sha256(args.eval_manifest),
        "skill_content_sha256": _skill_content_sha256(manifests),
    }


def _validate_output(args: argparse.Namespace, fingerprints: dict[str, str]) -> dict[str, Any]:
    report_path = args.output_dir / "build_report.json"
    if not report_path.is_file():
        raise SystemExit(f"validate-only: missing {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("fingerprints") != fingerprints:
        raise SystemExit("validate-only: frozen source fingerprints changed; rebuild into a new output")
    output_fingerprints = {
        "train_parquet_sha256": _file_sha256(args.output_dir / "train.parquet"),
        "eval_parquet_sha256": _file_sha256(args.output_dir / "eval.parquet"),
    }
    if report.get("output_fingerprints") != output_fingerprints:
        raise SystemExit("validate-only: output parquet fingerprint mismatch")
    train = pd.read_parquet(args.output_dir / "train.parquet")
    evaluation = pd.read_parquet(args.output_dir / "eval.parquet")
    result = {
        "train": _validate_frame(
            train,
            split="train",
            expected_kind=TRAIN_KIND,
            expected_rows=args.expected_train_tasks,
        ),
        "eval": _validate_frame(
            evaluation,
            split="eval",
            expected_kind=EVAL_KIND,
            expected_rows=args.expected_eval_tasks,
        ),
        "output_fingerprints": output_fingerprints,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "datasets/rl/parquet_4bench_slate_regret_v8prod_20260708",
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=ROOT
        / "skill_libraries/snapshots/rl/slate_skills_20260708_hard_negative_v8_production/manifest/slate_manifest_train.jsonl",
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=ROOT
        / "skill_libraries/snapshots/rl/slate_skills_20260708_hard_negative_v8_production/manifest/slate_manifest_eval70.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "datasets/rl/parquet_4bench_mixed_skill_bonus_compare_v8prod_allgold_20260710",
    )
    parser.add_argument("--expected-train-tasks", type=int, default=491)
    parser.add_argument("--expected-eval-tasks", type=int, default=56)
    parser.add_argument(
        "--skill-roots",
        default=os.environ.get(
            "AGENT_BENCH_EXTRA_SKILL_ROOTS",
            ":".join(
                [
                    str(ROOT / "skill_libraries/snapshots/rl/slate_skills_20260708_hard_negative_v8_production/skills"),
                    str(ROOT / "skill_libraries/snapshots/rl/slate_skills_20260704/skills"),
                ]
            ),
        ),
        help="colon-separated runtime roots in first-match order; merged library is appended",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    train_manifest = _load_manifest(args.train_manifest)
    eval_manifest = _load_manifest(args.eval_manifest)
    skill_roots = [Path(value).resolve() for value in args.skill_roots.split(":") if value.strip()]
    _validate_runtime_resolution([train_manifest, eval_manifest], skill_roots)
    fingerprints = _fingerprints(args, [train_manifest, eval_manifest])

    if args.validate_only:
        result = _validate_output(args, fingerprints)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty output directory: {args.output_dir}; "
            "use --validate-only or choose a new frozen output directory"
        )

    train_source = pd.read_parquet(args.input_dir / "train.parquet")
    eval_source = pd.read_parquet(args.input_dir / "eval.parquet")
    train, train_stats = _build_train(
        train_source, train_manifest, args.expected_train_tasks
    )
    evaluation, eval_stats = _build_eval(eval_source, eval_manifest)
    _validate_frame(
        train,
        split="train",
        expected_kind=TRAIN_KIND,
        expected_rows=args.expected_train_tasks,
    )
    _validate_frame(
        evaluation,
        split="eval",
        expected_kind=EVAL_KIND,
        expected_rows=args.expected_eval_tasks,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.output_dir / "train.parquet")
    evaluation.to_parquet(args.output_dir / "eval.parquet")
    output_fingerprints = {
        "train_parquet_sha256": _file_sha256(args.output_dir / "train.parquet"),
        "eval_parquet_sha256": _file_sha256(args.output_dir / "eval.parquet"),
    }
    report = {
        "format_version": 2,
        "objective": "mixed-skill-only GRPO with explicit oracle/misleading/no-read behavior comparison",
        "input_dir": str(args.input_dir.resolve()),
        "train_manifest": str(args.train_manifest.resolve()),
        "eval_manifest": str(args.eval_manifest.resolve()),
        "skill_roots": [str(path) for path in skill_roots],
        "composition": EXPECTED_SLATE_COUNTS,
        "train": train_stats,
        "eval": eval_stats,
        "fingerprints": fingerprints,
        "output_fingerprints": output_fingerprints,
        "notes": [
            "train contains no no-skill arm",
            "all 491 train tasks contain exactly one gold skill and 16 total skills",
            "train/eval order reuses canonical v8prod slate-train-shuffle/slate-shuffle seeds",
            "skill category labels live only in extra_info and are not exposed in prompts",
            "runtime reward attribution must use actual skill-file read tool calls",
        ],
    }
    (args.output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
