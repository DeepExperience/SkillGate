#!/usr/bin/env python3
"""Per-category skill-read attribution for mixed-slate eval70 runs.

Joins per-trial read_skill_names (from analyze_eval70_3tables.collect / the
extended detect_skill_use) against the slate manifest's name->category map
(oracle / misleading / relevant / irrelevant) and reports, per run:

  - strict read rate + per-category read rates (any-source and agent-initiated)
  - judgment quality: P(read oracle | read anything), misleading-read rate
  - P(resolved | read oracle), P(resolved | read misleading, no oracle),
    P(resolved | read nothing), etc.

Usage:
  python3 ops/workflows/rl_eval/analyze_slate_reads.py \
      --manifest skill_libraries/snapshots/rl/slate_skills_20260708_hard_negative_v8_production/manifest/slate_manifest_eval70.jsonl \
      [--source-to-frozen path/to/source_to_frozen.jsonl] \
      [--out report.md] <label>=<run_root> [<label>=<run_root> ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_eval70_3tables import collect  # noqa: E402

# analyze bench keys -> manifest bench keys
BENCH_MAP = {"seta": "seta_synth", "swe": "swe_lite", "tb2": "tb2", "sb_ns": "sb_ns", "claw": "claw"}
CATEGORIES = ("oracle", "misleading", "relevant", "irrelevant")


def load_manifest(path: str) -> dict[str, dict[str, str]]:
    """(bench::task_id) -> {skill_name: category}"""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cats = {}
            for cat in CATEGORIES:
                for e in row[cat]:
                    cats[e["name"]] = cat
            out[row["task_key"]] = cats
    return out


def load_frozen_name_aliases(path: str) -> dict[str, str]:
    """Load ``frozen dir basename -> source dir basename`` aliases.

    Older deep-frozen snapshots mounted skills as
    ``<path-hash>__<source-name>``; current selector-faithful snapshots preserve
    the logical name. The materializer mapping supports both formats and is the
    authoritative join to the immutable slate manifest.
    """
    aliases: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_name = str(row.get("logical_name") or Path(row["source_path"]).name)
            frozen_name = Path(row["frozen_path"]).name
            previous = aliases.get(frozen_name)
            if previous is not None and previous != source_name:
                raise ValueError(
                    f"conflicting frozen-name mapping at {path}:{line_number}: "
                    f"{frozen_name!r} -> {previous!r}/{source_name!r}"
                )
            aliases[frozen_name] = source_name
    return aliases


def summarize(trials: list[dict], manifest: dict[str, dict[str, str]],
              names_field: str, name_aliases: dict[str, str] | None = None) -> dict:
    n = len(trials)
    stats = {
        "n": n,
        "resolved": sum(t["resolved"] for t in trials),
        "strict_read": sum(bool(t["read"]) for t in trials),
        "read_any": 0,
        "unattributed_read": 0,
        "unknown_names": 0,
        "avg_names_per_trial": 0.0,
    }
    per_cat_read = {c: 0 for c in CATEGORIES}
    per_cat_resolved = {c: 0 for c in CATEGORIES}
    gold_given_read = 0
    misleading_no_gold = 0
    misleading_no_gold_resolved = 0
    noread_resolved = 0
    n_noread = 0
    no_category = 0
    total_names = 0

    for t in trials:
        key = f"{BENCH_MAP.get(t['bench'], t['bench'])}::{t['task']}"
        cats_map = manifest.get(key, {})
        names = t.get(names_field) or []
        has_attributed_read = bool(names)
        cats_read = set()
        for name in names:
            canonical_name = (name_aliases or {}).get(name, name)
            cat = cats_map.get(canonical_name)
            if cat is None:
                stats["unknown_names"] += 1
            else:
                cats_read.add(cat)
        total_names += len(names)
        if has_attributed_read:
            stats["read_any"] += 1
        else:
            n_noread += 1
            noread_resolved += t["resolved"]
        if not cats_read:
            no_category += 1
            continue
        for c in cats_read:
            per_cat_read[c] += 1
            per_cat_resolved[c] += t["resolved"]
        if "oracle" in cats_read:
            gold_given_read += 1
        elif "misleading" in cats_read:
            misleading_no_gold += 1
            misleading_no_gold_resolved += t["resolved"]

    n_read_attr = n - no_category
    stats["avg_names_per_trial"] = total_names / max(n, 1)
    stats.update({
        "per_cat_read": per_cat_read,
        "per_cat_resolved": per_cat_resolved,
        "n_read_attributed": n_read_attr,
        "no_category": no_category,
        "p_gold_given_read": gold_given_read / max(n_read_attr, 1),
        "misleading_no_gold": misleading_no_gold,
        "p_resolved_misleading_no_gold": misleading_no_gold_resolved / max(misleading_no_gold, 1),
        "n_noread": n_noread,
        "p_resolved_noread": noread_resolved / max(n_noread, 1),
    })
    return stats


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render(label: str, s_any: dict, s_agent: dict) -> str:
    lines = [f"### {label}", ""]
    n = s_any["n"]
    lines.append(f"- trials: {n}, resolved: {s_any['resolved']} ({fmt_pct(s_any['resolved']/max(n,1))}), "
                 f"strict agent tool-call read: {s_agent['strict_read']} ({fmt_pct(s_agent['strict_read']/max(n,1))}), "
                 f"attributed agent reads: {s_agent['read_any']}, "
                 f"unknown agent-read names: {s_agent['unknown_names']}, "
                 f"avg agent-read names/trial: {s_agent['avg_names_per_trial']:.2f}")
    lines.append("")
    lines.append("| category | read% (agent tool-call, primary) | read% (any echoed path, diagnostic) | P(resolved \\| agent read cat) |")
    lines.append("|---|---:|---:|---:|")
    for c in CATEGORIES:
        ra, rg = s_any["per_cat_read"][c], s_agent["per_cat_read"][c]
        pr = s_agent["per_cat_resolved"][c] / max(rg, 1)
        lines.append(f"| {c} | {fmt_pct(rg/max(n,1))} ({rg}/{n}) | {fmt_pct(ra/max(n,1))} ({ra}/{n}) | {fmt_pct(pr)} |")
    lines.append("")
    lines.append(f"- 判断力: P(assistant tool-call 读到 oracle | 读了任意 slate skill) = "
                 f"{fmt_pct(s_agent['p_gold_given_read'])} ({s_agent['n_read_attributed']} trials 读过)")
    lines.append(f"- 误导暴露: assistant tool-call 读了 misleading 且没读 oracle 的 trials = "
                 f"{s_agent['misleading_no_gold']}, 其 resolved 率 = "
                 f"{fmt_pct(s_agent['p_resolved_misleading_no_gold'])}")
    lines.append(f"- 未发起任何 attributed skill read tool-call 的 trials = {s_agent['n_noread']}, "
                 f"其 resolved 率 = {fmt_pct(s_agent['p_resolved_noread'])}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument(
        "--source-to-frozen",
        default="",
        help=(
            "optional source_to_frozen.jsonl emitted by "
            "materialize_frozen_skill_snapshot.py; restores logical skill "
            "names from deep-frozen trajectory mount paths"
        ),
    )
    ap.add_argument("--out", default="")
    ap.add_argument("runs", nargs="+", help="label=run_root")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    name_aliases = load_frozen_name_aliases(args.source_to_frozen) if args.source_to_frozen else {}
    if name_aliases:
        print(f"[mapping] loaded {len(name_aliases)} frozen skill-name aliases", file=sys.stderr)
    chunks = ["# Mixed-slate per-category read attribution", ""]
    for arg in args.runs:
        label, root = arg.split("=", 1)
        trials = collect(root)
        print(f"[{label}] {len(trials)} trials from {root}", file=sys.stderr)
        s_any = summarize(trials, manifest, "read_names", name_aliases)
        s_agent = summarize(trials, manifest, "read_names_agent", name_aliases)
        chunks.append(render(label, s_any, s_agent))
    report = "\n".join(chunks)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\nwritten -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
