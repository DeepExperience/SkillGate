#!/usr/bin/env python3
"""Filter SFT dataset to remove records that would mislead the model away from
the current OpenClaw runner environment.

Drop reasons:
1. **stale_url**: assistant content references the old runner URL pattern
   (`http://localhost:PORT`, `/tmp/claw_pilot`, `fixtures/`) — current runner
   uses `host.docker.internal:PORT` and skill paths under `/root/.claude/skills`.
2. **image_literal**: any message contains the literal `<image>` token.
   LLaMA-Factory treats this as a multimodal placeholder and crashes if the
   sample has no matching image payload.
3. **confirmation_loop**: trajectory has >= 3 consecutive assistant turns whose
   first 80 chars (after strip) are byte-identical — model is stuck in a
   "let me verify once more" loop that we don't want to imitate.

Operates on a sft_messages.jsonl input; writes a filtered jsonl + a per-bench
report to <output_dir>/filter_report.md.

Usage:
    python3 GeneralAgent/sft_data_collection/filter_clean_dataset.py \\
        --input  GeneralAgent/sft_training/datasets/<src>/sft_messages.jsonl \\
        --output GeneralAgent/sft_training/datasets/<dst>/sft_messages.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GeneralAgent.task_exclusions import is_bad_task


STALE_URL_PATTERNS = [
    re.compile(r"http://localhost:\d+"),
    re.compile(r"/tmp/claw_pilot"),
    re.compile(r"fixtures/"),
]


def has_stale_url(content: str) -> str | None:
    for pat in STALE_URL_PATTERNS:
        m = pat.search(content)
        if m:
            return m.group(0)
    return None


def has_confirmation_loop(messages: list[dict], n_consec: int = 3, prefix_len: int = 80) -> str | None:
    """Detect >= n_consec consecutive assistant turns sharing the same first
    `prefix_len` chars (post-lstrip). Returns the offending prefix or None."""
    asst_prefixes = [
        ((m.get("content") or "").lstrip()[:prefix_len])
        for m in messages
        if m.get("role") == "assistant"
    ]
    streak = 1
    for i in range(1, len(asst_prefixes)):
        if asst_prefixes[i] and asst_prefixes[i] == asst_prefixes[i - 1]:
            streak += 1
            if streak >= n_consec:
                return asst_prefixes[i]
        else:
            streak = 1
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default=None,
                        help="Markdown report path (default: <output_dir>/filter_report.md)")
    parser.add_argument("--loop-n-consec", type=int, default=3)
    parser.add_argument("--loop-prefix-len", type=int, default=80)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report) if args.report else (out_path.parent / "filter_report.md")

    n_total = 0
    n_kept = 0
    drop_reasons: Counter[str] = Counter()
    per_bench_total: Counter[str] = Counter()
    per_bench_kept: Counter[str] = Counter()
    per_bench_dropped: dict[str, Counter[str]] = defaultdict(Counter)
    drop_examples: dict[str, list[str]] = defaultdict(list)

    with in_path.open() as inf, out_path.open("w") as outf:
        for line in inf:
            line = line.rstrip("\n")
            if not line:
                continue
            n_total += 1
            r = json.loads(line)
            md = r.get("metadata", {}) or {}
            bench = md.get("bench") or "?"
            task = md.get("task_id") or "?"
            per_bench_total[bench] += 1

            if is_bad_task(bench, task):
                drop_reasons["known_bad_docker_task"] += 1
                per_bench_dropped[bench]["known_bad_docker_task"] += 1
                if len(drop_examples["known_bad_docker_task"]) < 5:
                    drop_examples["known_bad_docker_task"].append(f"{bench}::{task}")
                continue

            asst_text = "\n".join(
                m.get("content", "") or ""
                for m in r.get("messages", [])
                if m.get("role") == "assistant"
            )
            all_text = "\n".join(
                m.get("content", "") or ""
                for m in r.get("messages", [])
            )

            if "<image>" in all_text:
                drop_reasons["image_literal"] += 1
                per_bench_dropped[bench]["image_literal"] += 1
                if len(drop_examples["image_literal"]) < 5:
                    drop_examples["image_literal"].append(f"{bench}::{task}")
                continue

            stale = has_stale_url(asst_text)
            if stale:
                drop_reasons["stale_url"] += 1
                per_bench_dropped[bench]["stale_url"] += 1
                if len(drop_examples["stale_url"]) < 5:
                    drop_examples["stale_url"].append(f"{bench}::{task} (matched={stale})")
                continue

            loop_prefix = has_confirmation_loop(
                r.get("messages", []),
                n_consec=args.loop_n_consec,
                prefix_len=args.loop_prefix_len,
            )
            if loop_prefix:
                drop_reasons["confirmation_loop"] += 1
                per_bench_dropped[bench]["confirmation_loop"] += 1
                if len(drop_examples["confirmation_loop"]) < 5:
                    drop_examples["confirmation_loop"].append(
                        f"{bench}::{task} (prefix={loop_prefix[:50]!r}...)"
                    )
                continue

            outf.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_kept += 1
            per_bench_kept[bench] += 1

    # Report
    lines = [
        f"# SFT Dataset Filter Report",
        "",
        f"- input: `{in_path}`",
        f"- output: `{out_path}`",
        f"- loop_n_consec: {args.loop_n_consec}, loop_prefix_len: {args.loop_prefix_len}",
        f"- total input records: **{n_total}**",
        f"- kept: **{n_kept}** ({100*n_kept/max(n_total,1):.1f}%)",
        f"- dropped: **{n_total - n_kept}**",
        "",
        "## Drop reasons (overall)",
        "",
        "| reason | count | % of input |",
        "| --- | ---: | ---: |",
    ]
    for reason, n in drop_reasons.most_common():
        lines.append(f"| {reason} | {n} | {100*n/n_total:.1f}% |")
    lines += [
        "",
        "## Per-bench breakdown",
        "",
        "| bench | input | kept | dropped (bad_docker) | dropped (stale_url) | dropped (loop) | retention |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for bench in sorted(per_bench_total):
        tot = per_bench_total[bench]
        kept = per_bench_kept[bench]
        d_bad = per_bench_dropped[bench]["known_bad_docker_task"]
        d_stale = per_bench_dropped[bench]["stale_url"]
        d_loop = per_bench_dropped[bench]["confirmation_loop"]
        lines.append(f"| {bench} | {tot} | {kept} | {d_bad} | {d_stale} | {d_loop} | {100*kept/tot:.1f}% |")
    lines += [
        "",
        "## Sample drops",
        "",
    ]
    for reason, exs in drop_examples.items():
        lines.append(f"### {reason}")
        lines.append("")
        for e in exs:
            lines.append(f"- {e}")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"input={n_total} kept={n_kept} dropped={n_total - n_kept}")
    print(f"drop reasons: {dict(drop_reasons)}")
    print(f"per-bench retention:")
    for b in sorted(per_bench_total):
        tot = per_bench_total[b]; kept = per_bench_kept[b]
        print(f"  {b}: {kept}/{tot} ({100*kept/tot:.1f}%)")
    print(f"\nreport: {report_path}")
    print(f"output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
