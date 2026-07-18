#!/usr/bin/env python3
"""Build the hybrid misleading slate: v8-production BODY + 0704 DESCRIPTION.

Motivation (2026-07-10, user decision): decouple the two axes that a
judgment-training slate needs -
  * selection-stage learnability  <- 0704 descriptions (puffery tells /
    visible wrong details; SFT init already has a usable prior on them);
  * consequence-stage poison      <- v8 hard-negative bodies (median 28%
    line divergence vs oracle, anti-recovery screened), so reading the wrong
    skill produces a real outcome contrast for evidence-based credit.

Rule per misleading skill (same names exist in both versions):
  * body   := v8 SKILL.md body (byte-identical below the frontmatter);
  * name   := unchanged;
  * desc   := 0704 description, EXCEPT when the 0704 description is
    (near-)identical to the oracle description (similarity > --desc-sim-threshold,
    default 0.92): those are unseparable at selection stage, so keep the v8
    description (0% identical, always carries a checkable wrong detail).

Oracle / relevant / irrelevant entries are passed through from the v8
manifests untouched (oracle = renamed 0704 copies; relevant/irrelevant =
merged library).

Outputs under --output-root (refuses to overwrite existing skills/):
  skills/<name>/          full copy of the v8 skill dir with SKILL.md desc swapped
  manifest/slate_manifest_train.jsonl / slate_manifest_eval70.jsonl
  manifest/hybrid_build_report.json    per-skill decision audit
  snapshot_eval70/<bench>.jsonl        16-entry rows for eval70 --skill-mode mixed
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V4_ROOT = ROOT / "skill_libraries/snapshots/rl/slate_skills_20260704"
V8_ROOT = ROOT / "skill_libraries/snapshots/rl/slate_skills_20260708_hard_negative_v8_production"
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
DESC_LINE_RE = re.compile(r"^description:\s*(.+?)(?=\n[a-zA-Z_]+:|\Z)", re.S | re.M)
BENCHES = ("claw", "sb_ns", "seta_synth", "swe_lite", "tb2")

import hashlib
import random


def seeded_rng(tag: str, bench: str, task_id: str) -> random.Random:
    seed = int(hashlib.sha256(f"{tag}::{bench}::{task_id}".encode()).hexdigest()[:12], 16)
    return random.Random(seed)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_frontmatter(text: str) -> tuple[str, str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError("missing frontmatter")
    return text[: m.end()], text[m.end():]


def extract_desc(frontmatter_block: str) -> str:
    inner = FRONTMATTER_RE.match(frontmatter_block).group(1)
    dm = DESC_LINE_RE.search(inner)
    if not dm:
        raise ValueError("missing description")
    return " ".join(dm.group(1).split()).strip("\"'")


def replace_desc(frontmatter_block: str, new_desc: str) -> str:
    inner = FRONTMATTER_RE.match(frontmatter_block).group(1)
    new_inner = DESC_LINE_RE.sub(lambda _m: f"description: {new_desc}", inner, count=1)
    return f"---\n{new_inner}\n---\n"


def load_manifest(path: Path) -> dict[str, dict]:
    return {
        f"{r['bench']}::{r['task_id']}": r
        for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())
    }


def build_split(split: str, out_root: Path, threshold: float,
                built_names: dict[str, dict], report: list[dict]) -> list[dict]:
    m4 = load_manifest(V4_ROOT / "manifest" / f"slate_manifest_{split}.jsonl")
    m8 = load_manifest(V8_ROOT / "manifest" / f"slate_manifest_{split}.jsonl")
    if set(m4) != set(m8):
        raise SystemExit(f"[{split}] task keys differ between 0704 and v8 manifests")

    out_rows = []
    skills_dir = out_root / "skills"
    for key in sorted(m8):
        r8, r4 = m8[key], m4[key]
        oracle_entry = r8["oracle"][0]
        oracle_desc = extract_desc(split_frontmatter(read_text(Path(oracle_entry["path"]) / "SKILL.md"))[0])

        names8 = [e["name"] for e in r8["misleading"]]
        names4 = {e["name"]: e for e in r4["misleading"]}

        new_misleading = []
        for e8 in r8["misleading"]:
            name = e8["name"]
            v8_dir = Path(e8["path"])
            v8_text = read_text(v8_dir / "SKILL.md")
            v8_fm, v8_body = split_frontmatter(v8_text)
            v8_desc = extract_desc(v8_fm)

            desc_source = "v8_no_0704_counterpart"
            chosen_desc = v8_desc
            if name in names4:
                v4_text = read_text(Path(names4[name]["path"]) / "SKILL.md")
                v4_desc = extract_desc(split_frontmatter(v4_text)[0])
                sim = difflib.SequenceMatcher(None, oracle_desc, v4_desc).ratio()
                if sim > threshold:
                    desc_source = f"v8_desc_0704_too_oracle_like(sim={sim:.3f})"
                    chosen_desc = v8_desc
                else:
                    desc_source = f"0704_desc(sim={sim:.3f})"
                    chosen_desc = v4_desc

            dst = skills_dir / name
            if name in built_names:
                prev = built_names[name]
                if prev["desc"] != chosen_desc or prev["v8_path"] != str(v8_dir):
                    raise SystemExit(f"name collision with divergent content: {name}")
            else:
                if dst.exists():
                    raise SystemExit(f"refusing to overwrite existing {dst}")
                shutil.copytree(v8_dir, dst)
                new_fm = replace_desc(v8_fm, chosen_desc)
                (dst / "SKILL.md").write_text(new_fm + v8_body, encoding="utf-8")
                # -- self-validation --
                out_fm, out_body = split_frontmatter(read_text(dst / "SKILL.md"))
                assert out_body == v8_body, f"body changed for {name}"
                assert extract_desc(out_fm) == chosen_desc, f"desc not applied for {name}"
                name_line = re.search(r"^name:\s*(.+)$", FRONTMATTER_RE.match(out_fm).group(1), re.M)
                assert name_line and name_line.group(1).strip().strip("\"'") == name, f"frontmatter name mismatch {name}"
                built_names[name] = {"desc": chosen_desc, "v8_path": str(v8_dir)}
                report.append({"split": split, "task_key": key, "name": name,
                               "desc_source": desc_source})

            new_misleading.append({**e8, "path": str(dst)})

        row = dict(r8)
        row["misleading"] = new_misleading
        out_rows.append(row)
    return out_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_eval_snapshot(rows: list[dict], out_root: Path) -> None:
    by_bench: dict[str, list[dict]] = {b: [] for b in BENCHES}
    for row in rows:
        entries = (list(row["oracle"]) + list(row["misleading"])
                   + list(row["relevant"]) + list(row["irrelevant"]))
        seeded_rng("slate-shuffle", row["bench"], row["task_id"]).shuffle(entries)
        by_bench[row["bench"]].append({
            "task_id": row["task_id"],
            "reranked_top10": [
                {"skill_name": e["name"], "skill_path": e["path"], "score": 1.0}
                for e in entries
            ],
            "slate_categories": {e["name"]: e["category"] for e in entries},
        })
    for bench, brows in by_bench.items():
        if brows:
            write_jsonl(out_root / "snapshot_eval70" / f"{bench}.jsonl", brows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root",
                    default=str(ROOT / "skill_libraries/snapshots/rl/slate_skills_20260710_hybrid_v8body_0704desc"))
    ap.add_argument("--desc-sim-threshold", type=float, default=0.92)
    args = ap.parse_args()

    out_root = Path(args.output_root)
    (out_root / "skills").mkdir(parents=True, exist_ok=False)

    built: dict[str, dict] = {}
    report: list[dict] = []
    all_rows = {}
    for split in ("train", "eval70"):
        rows = build_split(split, out_root, args.desc_sim_threshold, built, report)
        write_jsonl(out_root / "manifest" / f"slate_manifest_{split}.jsonl", rows)
        all_rows[split] = rows
        print(f"[{split}] {len(rows)} manifest rows written")

    write_eval_snapshot(all_rows["eval70"], out_root)

    from collections import Counter
    sources = Counter(r["desc_source"].split("(")[0] for r in report)
    summary = {
        "skills_built": len(built),
        "desc_sources": dict(sources),
        "desc_sim_threshold": args.desc_sim_threshold,
        "v8_root": str(V8_ROOT),
        "v4_root": str(V4_ROOT),
    }
    (out_root / "manifest" / "hybrid_build_report.json").write_text(
        json.dumps({"summary": summary, "per_skill": report}, ensure_ascii=False, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
