#!/usr/bin/env python3
"""Build the SlateRL pair parquet (no_skill_grpo + slate_grpo per task).

Input = the canonical pair parquet (no_skill_grpo + oracle_prompt_bc, built by
make_4bench_m1_hybrid_parquet.py --oracle-mode direct_text). Per task:

  no_skill_grpo row   untouched (byte-identical no-skill arm).
  oracle_prompt_bc row -> slate_grpo row:
    - the "## Preloaded Oracle Skill ... </preloaded_oracle_skill>" prompt
      block is replaced by an <available_skills> block advertising the task's
      slate (self-read form: names + descriptions + /root/.claude/skills
      locations; the agent must read SKILL.md files itself);
    - extra_info: update_kind/hybrid_update_kind=slate_grpo,
      retrieval_skills_top_n=[slate names] (resolved at rollout via
      AGENT_BENCH_EXTRA_SKILL_ROOTS + merged library),
      hybrid_is_shadow=0 / hybrid_grpo_weight=1 / hybrid_shadow_weight=0,
      slate_contains_gold / slate_gold_name / slate_size.

Slate composition per task comes from the slate manifest
(ops/workflows/rl_eval/slate_skill_pipeline.py finalize): oracle x1 (renamed
copy) + misleading x5 + relevant x5 + irrelevant x5. Gold presence is a
deterministic per-task coin with --p-gold (gold-absent slates keep the other
15 skills; that trains negative rejection).

eval.parquet advertises the full 16-skill eval slate for every held-out task so
the internal checkpoint metric measures mixed-skill behavior.  This is still a
small 56-task checkpoint-selection signal; eval70 x4 remains the final metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PRELOADED_RE = re.compile(r"## Preloaded Oracle Skill.*?</preloaded_oracle_skill>\s*", re.S)
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---", re.S)
MIXED_EVAL_KIND = "mixed_eval"


def frontmatter_description(skill_md: Path) -> str:
    text = skill_md.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    desc = ""
    if m:
        dm = re.search(r"^description:\s*(.+?)(?=\n[a-zA-Z_]+:|\Z)", m.group(1), re.S | re.M)
        if dm:
            desc = " ".join(dm.group(1).split()).strip("\"'")
    if not desc:
        desc = f"Skill document {skill_md.parent.name}."
    return desc[:300]


def seeded_rng(tag: str, bench: str, task_id: str) -> random.Random:
    seed = int(hashlib.sha256(f"{tag}::{bench}::{task_id}".encode()).hexdigest()[:12], 16)
    return random.Random(seed)


def load_manifest(path: Path) -> dict[str, dict]:
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[f"{row['bench']}::{row['task_id']}"] = row
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(raw_path: str | Path) -> Path:
    """Resolve repository-relative frozen paths independently of caller CWD."""
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def skill_content_sha256(manifests: list[dict[str, dict]]) -> str:
    digest = hashlib.sha256()
    paths = {
        str((resolve_repo_path(entry["path"]) / "SKILL.md").resolve())
        for manifest in manifests
        for row in manifest.values()
        for category in ("oracle", "misleading", "relevant", "irrelevant")
        for entry in row[category]
    }
    for raw_path in sorted(paths):
        path = Path(raw_path)
        digest.update(raw_path.encode() + b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def build_slate_block(entries: list[dict]) -> str:
    parts = ["<available_skills>"]
    for e in entries:
        parts.append(
            "  <skill>\n"
            f"    <name>{e['name']}</name>\n"
            f"    <description>{e['description']}</description>\n"
            f"    <location>/root/.claude/skills/{e['name']}/SKILL.md</location>\n"
            "  </skill>"
        )
    parts.append("</available_skills>")
    return "\n".join(parts)


def pick_slate(mrow: dict, p_gold: float) -> tuple[list[dict], bool]:
    bench, task_id = mrow["bench"], mrow["task_id"]
    has_gold = seeded_rng("slate-gold", bench, task_id).random() < p_gold
    entries = list(mrow["misleading"]) + list(mrow["relevant"]) + list(mrow["irrelevant"])
    if has_gold:
        entries = list(mrow["oracle"]) + entries
    seeded_rng("slate-train-shuffle", bench, task_id).shuffle(entries)
    return entries, has_gold


def pick_eval_slate(mrow: dict) -> list[dict]:
    """Return the same ordered 16-skill slate used by snapshot_eval70."""
    entries = (
        list(mrow["oracle"])
        + list(mrow["misleading"])
        + list(mrow["relevant"])
        + list(mrow["irrelevant"])
    )
    seeded_rng("slate-shuffle", mrow["bench"], mrow["task_id"]).shuffle(entries)
    return entries


def insert_eval_slate_block(prompt: object, block: str) -> tuple[object, bool]:
    """Insert the slate immediately before Memory Recall, matching train prompts."""
    msgs = list(prompt) if isinstance(prompt, (list, np.ndarray)) else None
    if msgs is None:
        return prompt, False

    inserted = False
    out_msgs = []
    for raw_message in msgs:
        message = dict(raw_message)
        content = message.get("content", "")
        if not inserted and message.get("role") == "system" and isinstance(content, str):
            if "<available_skills>" in content:
                return prompt, False
            for marker in ("\n## Memory Recall\n", "\n## Runtime\n"):
                idx = content.find(marker)
                if idx >= 0:
                    message["content"] = content[:idx].rstrip() + "\n" + block + "\n" + content[idx:]
                    inserted = True
                    break
        out_msgs.append(message)
    return np.array(out_msgs, dtype=object), inserted


def _runtime_skill_path(name: str, skill_roots: list[Path]) -> Path | None:
    candidates = [root / name for root in skill_roots]
    merged = ROOT / "skill_libraries/merged"
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


def validate_runtime_resolution(manifests: list[dict[str, dict]], skill_roots: list[Path]) -> None:
    """Ensure ordered runtime roots resolve every manifest name to its intended file."""
    problems = []
    checked: set[tuple[str, str]] = set()
    for manifest in manifests:
        for row in manifest.values():
            for category in ("oracle", "misleading", "relevant", "irrelevant"):
                for entry in row[category]:
                    key = (entry["name"], entry["path"])
                    if key in checked:
                        continue
                    checked.add(key)
                    expected = resolve_repo_path(entry["path"]).resolve()
                    actual = _runtime_skill_path(entry["name"], skill_roots)
                    if actual is None:
                        problems.append(f"unresolved:{entry['name']}")
                    elif actual != expected:
                        problems.append(f"wrong-first-match:{entry['name']}:{actual}!={expected}")
    if problems:
        raise SystemExit(f"runtime skill resolution failed ({len(problems)}): {problems[:10]}")


def transform(df: pd.DataFrame, manifest: dict[str, dict], p_gold: float,
              validate_only: bool) -> tuple[pd.DataFrame, list[str], dict]:
    problems: list[str] = []
    new_prompts, new_extras = [], []
    stats = {"no_skill": 0, "slate": 0, "gold_present": 0, "gold_absent": 0}
    desc_cache: dict[str, str] = {}

    def desc_of(e: dict) -> str:
        if e["path"] not in desc_cache:
            md = resolve_repo_path(e["path"]) / "SKILL.md"
            if not md.is_file():
                problems.append(f"missing-skill-md:{e['path']}")
                desc_cache[e["path"]] = f"Skill document {e['name']}."
            else:
                desc_cache[e["path"]] = frontmatter_description(md)
        return desc_cache[e["path"]]

    for _, row in df.iterrows():
        extra = dict(row["extra_info"])
        kind = str(extra.get("update_kind") or "").strip().lower()
        bench, task_id = str(extra["bench"]), str(extra["task_id"])
        key = f"{bench}::{task_id}"

        if kind != "oracle_prompt_bc":
            stats["no_skill"] += 1
            new_prompts.append(row["prompt"])
            new_extras.append(extra)
            continue

        mrow = manifest.get(key)
        if mrow is None:
            problems.append(f"missing-manifest:{key}")
            new_prompts.append(row["prompt"])
            new_extras.append(extra)
            continue

        entries, has_gold = pick_slate(mrow, p_gold)
        enriched = [{**e, "description": desc_of(e)} for e in entries]
        block = build_slate_block(enriched)

        prompt = row["prompt"]
        msgs = list(prompt) if isinstance(prompt, (list, np.ndarray)) else None
        replaced = False
        if msgs is not None:
            out_msgs = []
            for m in msgs:
                m = dict(m)
                c = m.get("content", "")
                if isinstance(c, str) and PRELOADED_RE.search(c):
                    m["content"] = PRELOADED_RE.sub(lambda _: block + "\n\n", c, count=1)
                    replaced = True
                out_msgs.append(m)
            new_prompts.append(np.array(out_msgs, dtype=object))
        else:
            text = str(prompt)
            if PRELOADED_RE.search(text):
                text = PRELOADED_RE.sub(lambda _: block + "\n\n", text, count=1)
                replaced = True
            new_prompts.append(text)
        if not replaced:
            problems.append(f"no-preloaded-block:{key}")

        extra["update_kind"] = "slate_grpo"
        extra["hybrid_update_kind"] = "slate_grpo"
        extra["retrieval_skills_top_n"] = [e["name"] for e in entries]
        extra["hybrid_is_shadow"] = 0.0
        extra["hybrid_grpo_weight"] = 1.0
        extra["hybrid_shadow_weight"] = 0.0
        extra["oracle_skill_mode"] = "slate_path"
        extra["slate_contains_gold"] = 1.0 if has_gold else 0.0
        extra["slate_gold_name"] = mrow["oracle"][0]["name"] if has_gold else ""
        extra["slate_size"] = float(len(entries))
        new_extras.append(extra)
        stats["slate"] += 1
        stats["gold_present" if has_gold else "gold_absent"] += 1

    out = df.copy()
    out["prompt"] = new_prompts
    out["extra_info"] = new_extras
    return out, problems, stats


def transform_eval(df: pd.DataFrame, manifest: dict[str, dict]) -> tuple[pd.DataFrame, list[str], dict]:
    problems: list[str] = []
    new_prompts, new_extras = [], []
    stats: dict[str, int] = {"mixed": 0, "gold_present": 0}
    desc_cache: dict[str, str] = {}

    def desc_of(entry: dict) -> str:
        path = entry["path"]
        if path not in desc_cache:
            skill_md = resolve_repo_path(path) / "SKILL.md"
            if not skill_md.is_file():
                problems.append(f"missing-skill-md:{path}")
                desc_cache[path] = f"Skill document {entry['name']}."
            else:
                desc_cache[path] = frontmatter_description(skill_md)
        return desc_cache[path]

    for _, row in df.iterrows():
        extra = dict(row["extra_info"])
        bench, task_id = str(extra["bench"]), str(extra["task_id"])
        key = f"{bench}::{task_id}"
        manifest_row = manifest.get(key)
        if manifest_row is None:
            problems.append(f"missing-eval-manifest:{key}")
            new_prompts.append(row["prompt"])
            new_extras.append(extra)
            continue

        entries = pick_eval_slate(manifest_row)
        if len(entries) != 16 or len({entry["name"] for entry in entries}) != 16:
            problems.append(f"bad-eval-slate:{key}:{len(entries)}")
        enriched = [{**entry, "description": desc_of(entry)} for entry in entries]
        prompt, inserted = insert_eval_slate_block(row["prompt"], build_slate_block(enriched))
        if not inserted:
            problems.append(f"eval-slate-insert-failed:{key}")

        extra["update_kind"] = MIXED_EVAL_KIND
        extra["hybrid_update_kind"] = MIXED_EVAL_KIND
        extra["retrieval_skills_top_n"] = [entry["name"] for entry in entries]
        extra["hybrid_is_shadow"] = 0.0
        extra["hybrid_grpo_weight"] = 0.0
        extra["hybrid_shadow_weight"] = 0.0
        extra["oracle_skill_mode"] = "slate_path"
        extra["slate_contains_gold"] = 1.0
        extra["slate_gold_name"] = manifest_row["oracle"][0]["name"]
        extra["slate_size"] = 16.0
        new_prompts.append(prompt)
        new_extras.append(extra)
        stats["mixed"] += 1
        stats["gold_present"] += 1
        stats[f"bench:{bench}"] = stats.get(f"bench:{bench}", 0) + 1

    out = df.copy()
    out["prompt"] = new_prompts
    out["extra_info"] = new_extras
    return out, problems, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path,
                    default=ROOT / "datasets/rl/parquet_4bench_oracle_promptbc_pair_noskill_grpo_20260623")
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "skill_libraries/snapshots/rl/slate_skills_20260708_hard_negative_v8_production/manifest/slate_manifest_train.jsonl")
    ap.add_argument("--eval-manifest", type=Path,
                    default=ROOT / "skill_libraries/snapshots/rl/slate_skills_20260708_hard_negative_v8_production/manifest/slate_manifest_eval70.jsonl")
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "datasets/rl/parquet_4bench_slate_regret_v8prod_mixedeval_20260710")
    ap.add_argument("--p-gold", type=float, default=0.7)
    ap.add_argument(
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
        help="colon-separated runtime skill roots, in first-match order; merged is appended automatically",
    )
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    train_manifest = load_manifest(args.manifest)
    eval_manifest = load_manifest(args.eval_manifest)
    skill_roots = [Path(item).resolve() for item in args.skill_roots.split(":") if item.strip()]
    validate_runtime_resolution([train_manifest, eval_manifest], skill_roots)
    fingerprints = {
        "input_train_sha256": file_sha256(args.input_dir / "train.parquet"),
        "input_eval_sha256": file_sha256(args.input_dir / "eval.parquet"),
        "train_manifest_sha256": file_sha256(args.manifest),
        "eval_manifest_sha256": file_sha256(args.eval_manifest),
        "skill_content_sha256": skill_content_sha256([train_manifest, eval_manifest]),
    }
    report_path = args.output_dir / "build_report.json"

    if args.validate_only:
        if not report_path.is_file():
            raise SystemExit(f"validate-only: missing {report_path}")
        report = json.loads(report_path.read_text())
        for split in ("train", "eval"):
            if not (args.output_dir / f"{split}.parquet").is_file():
                raise SystemExit(f"validate-only: missing {split}.parquet")
        report_train_manifest = Path(report.get("train_manifest", report.get("manifest", ""))).resolve()
        report_eval_manifest = Path(report.get("eval_manifest", "")).resolve()
        if report_train_manifest != args.manifest.resolve() or report_eval_manifest != args.eval_manifest.resolve():
            raise SystemExit(
                "validate-only manifest mismatch: "
                f"report train/eval={report_train_manifest}/{report_eval_manifest}, "
                f"requested={args.manifest.resolve()}/{args.eval_manifest.resolve()}"
            )
        if report.get("fingerprints") != fingerprints:
            raise SystemExit(
                "validate-only source fingerprint mismatch; rebuild into a new output directory: "
                f"report={report.get('fingerprints')} current={fingerprints}"
            )
        output_fingerprints = {
            "train_parquet_sha256": file_sha256(args.output_dir / "train.parquet"),
            "eval_parquet_sha256": file_sha256(args.output_dir / "eval.parquet"),
        }
        if report.get("output_fingerprints") != output_fingerprints:
            raise SystemExit(
                "validate-only output fingerprint mismatch: "
                f"report={report.get('output_fingerprints')} current={output_fingerprints}"
            )
        df = pd.read_parquet(args.output_dir / "train.parquet", columns=["extra_info"])
        kinds = {}
        n_bad_names = 0
        for extra in df["extra_info"]:
            kinds[extra["update_kind"]] = kinds.get(extra["update_kind"], 0) + 1
            if extra["update_kind"] == "slate_grpo":
                raw_names = extra.get("retrieval_skills_top_n")
                names = [] if raw_names is None else list(raw_names)
                if not (10 <= len(names) <= 20):
                    n_bad_names += 1
        assert kinds.get("no_skill_grpo", 0) > 0 and kinds.get("slate_grpo", 0) > 0, kinds
        assert kinds.get("oracle_prompt_bc", 0) in (0, None), f"leftover oracle rows: {kinds}"
        assert n_bad_names == 0, f"{n_bad_names} slate rows with bad name-list size"
        eval_df = pd.read_parquet(args.output_dir / "eval.parquet", columns=["prompt", "extra_info"])
        eval_problems = []
        for row_index, row in eval_df.iterrows():
            extra = row["extra_info"]
            raw_names = extra.get("retrieval_skills_top_n")
            names = [] if raw_names is None else list(raw_names)
            prompt_text = "\n".join(
                str(message.get("content", "")) for message in list(row["prompt"]) if isinstance(message, dict)
            )
            if extra.get("update_kind") != MIXED_EVAL_KIND:
                eval_problems.append(f"row{row_index}:kind={extra.get('update_kind')}")
            if len(names) != 16 or len(set(names)) != 16:
                eval_problems.append(f"row{row_index}:names={len(names)}/{len(set(names))}")
            if "<available_skills>" not in prompt_text or float(extra.get("slate_contains_gold") or 0.0) != 1.0:
                eval_problems.append(f"row{row_index}:missing-slate-prompt-or-gold")
            if not extra.get("slate_gold_name") or extra["slate_gold_name"] not in names:
                eval_problems.append(f"row{row_index}:bad-gold-name")
        if eval_problems:
            raise SystemExit(f"mixed eval validation failed ({len(eval_problems)}): {eval_problems[:10]}")
        print(
            f"validate OK: train={kinds}, eval={len(eval_df)} {MIXED_EVAL_KIND} rows, "
            f"report={report.get('train', {}).get('stats')}"
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "format_version": 3,
        "p_gold": args.p_gold,
        "input_dir": str(args.input_dir),
        "train_manifest": str(args.manifest),
        "eval_manifest": str(args.eval_manifest),
        "eval_mode": "mixed_16_skill",
        "skill_roots": [str(path) for path in skill_roots],
        "fingerprints": fingerprints,
    }
    for split in ("train", "eval"):
        df = pd.read_parquet(args.input_dir / f"{split}.parquet")
        if split == "train":
            out, problems, stats = transform(df, train_manifest, args.p_gold, args.validate_only)
        else:
            out, problems, stats = transform_eval(df, eval_manifest)
        if problems:
            print(f"[{split}] {len(problems)} problems, first 10: {problems[:10]}")
            raise SystemExit("slate parquet build failed; fix manifest/skills first")
        out.to_parquet(args.output_dir / f"{split}.parquet")
        report[split] = {"rows": len(out), "stats": stats, "problems": problems}
        print(f"[{split}] rows={len(out)} stats={stats}")
    report["output_fingerprints"] = {
        "train_parquet_sha256": file_sha256(args.output_dir / "train.parquet"),
        "eval_parquet_sha256": file_sha256(args.output_dir / "eval.parquet"),
    }
    report_path.write_text(json.dumps(report, indent=1))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
