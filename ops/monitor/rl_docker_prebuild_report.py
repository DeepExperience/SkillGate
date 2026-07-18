#!/usr/bin/env python3
"""Generate the canonical RL Docker prebuild report.

This report intentionally consolidates the operational facts that matter:

* current train/eval image coverage,
* active/prebuild run status,
* deterministic Dockerfile failures that should be removed from RL data before
  the next run, and
* commands for monitoring the current tmux jobs.

It does not preserve every historical log line; those stay in the per-run
prebuild directories.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))
DEFAULT_OUT = PROJECT_ROOT / "docs/rl_docker_prebuild.md"
DEFAULT_PREBUILD_ROOT = PROJECT_ROOT / "experiments/infra/rl/prebuild"
LEGACY_AUDIT_DOCS = [
    PROJECT_ROOT / "docs/rl_image_cache_audit_current.md",
    PROJECT_ROOT / "docs/rl_image_cache_audit_eval_current.md",
    PROJECT_ROOT / "docs/rl_image_cache_audit.md",
]
LEGACY_AUDIT_SNAPSHOTS = [
    {
        "source": "rl_image_cache_audit_current.md",
        "parquet": "`datasets/rl/parquet_4bench_base_20260523/train.parquet`",
        "rows": "626",
        "unique": "496",
        "present": "349",
        "missing": "147",
        "bench": {
            "claw": "`claw_base/present`=1",
            "sb_ns": "`local_build/missing`=21; `local_build/present`=50",
            "seta_synth": "`local_build/missing`=116; `local_build/present`=148",
            "swe_lite": "`swe_official/missing`=10; `swe_official/present`=78",
            "tb2": "`declared_pull/present`=72",
        },
    },
    {
        "source": "rl_image_cache_audit_eval_current.md",
        "parquet": "`datasets/rl/parquet_4bench_base_20260523/eval.parquet`",
        "rows": "70",
        "unique": "57",
        "present": "23",
        "missing": "34",
        "bench": {
            "claw": "`claw_base/present`=1",
            "sb_ns": "`local_build/missing`=5; `local_build/present`=3",
            "seta_synth": "`local_build/missing`=27; `local_build/present`=3",
            "swe_lite": "`swe_official/missing`=2; `swe_official/present`=8",
            "tb2": "`declared_pull/present`=8",
        },
    },
    {
        "source": "rl_image_cache_audit.md",
        "parquet": "`datasets/rl/parquet_4bench_base_20260523/train.parquet`",
        "rows": "629",
        "unique": "499",
        "present": "348",
        "missing": "151",
        "bench": {
            "claw": "`claw_base/present`=1",
            "sb_ns": "`local_build/missing`=21; `local_build/present`=50",
            "seta_synth": "`local_build/missing`=120; `local_build/present`=147",
            "swe_lite": "`swe_official/missing`=10; `swe_official/present`=78",
            "tb2": "`declared_pull/present`=72",
        },
    },
]


def _load_status_rows(prebuild_root: Path) -> list[dict[str, Any]]:
    prebuild_root = prebuild_root.resolve()
    rows: list[dict[str, Any]] = []
    for status_path in sorted(prebuild_root.glob("*/status.jsonl")):
        run_dir = status_path.parent
        for line_no, line in enumerate(status_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            item["_status_path"] = str(status_path.resolve().relative_to(PROJECT_ROOT))
            item["_run_dir"] = str(run_dir.resolve().relative_to(PROJECT_ROOT))
            item["_line_no"] = line_no
            rows.append(item)
    return rows


def _latest_by_tag(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        tag = str(row.get("tag") or "")
        if not tag:
            continue
        previous = latest.get(tag)
        if previous is None or str(row.get("finished_at", "")) >= str(previous.get("finished_at", "")):
            latest[tag] = row
    return latest


def _classify_failure(log_text: str) -> tuple[str, str]:
    lower = log_text.lower()
    if "syntaxerror" in lower and "run set -eux" in lower:
        return (
            "dockerfile_invalid_python_heredoc",
            "Dockerfile writes shell RUN text into a Python heredoc/script; build fails deterministically with SyntaxError.",
        )
    if "copy sample.log" in lower and "not found" in lower:
        return (
            "dockerfile_missing_build_context_file",
            "Dockerfile COPY references sample.log but the file is absent from the Harbor build context.",
        )
    if "failed to calculate checksum" in lower and "not found" in lower:
        return (
            "dockerfile_missing_build_context_file",
            "Dockerfile COPY references a file absent from the Harbor build context.",
        )
    if "ground_truth.json" in lower and "did not complete successfully: exit code: 1" in lower:
        return (
            "dockerfile_setup_script_failure",
            "Dockerfile embedded setup script fails deterministically during image generation.",
        )
    if "astral.sh/uv" in lower or "uv/0.7.13/install.sh" in lower or "release process is not working" in lower:
        return (
            "external_download_uv_failed",
            "Build failed while downloading/installing uv from astral.sh; external-download fragility, not proven task Dockerfile corruption.",
        )
    if "inrelease' is no longer signed" in lower or "repository" in lower and "is no longer signed" in lower:
        return (
            "apt_repository_signature_expired",
            "apt-get update fails because the base image uses an expired/invalid Debian repository signature; retry alone will not reliably fix it.",
        )
    if "no matching distribution found" in lower:
        return (
            "pip_package_unavailable",
            "pip cannot resolve a pinned package from the configured index/proxy; likely mirror/proxy/package-version issue.",
        )
    if "modulenotfounderror: no module named 'pkg_resources'" in lower:
        return (
            "dockerfile_python_package_build_isolation",
            "A pinned Python package fails deterministically under current pip build isolation; patch Dockerfile or exclude the task.",
        )
    if "files.pythonhosted.org" in lower and ("read timed out" in lower or "connection" in lower):
        return (
            "pip_external_download_timeout",
            "pip download from files.pythonhosted.org timed out; external-download fragility, retry/prebuild with mirror may fix it.",
        )
    if "command timed out" in lower or "timed out" in lower:
        return ("build_timeout", "Build timed out; may be transient or a genuinely heavy Dockerfile.")
    if "cannot connect to the docker daemon" in lower:
        return ("docker_daemon_unreachable", "Docker daemon or tunnel was unreachable during build.")
    return ("other_build_failure", "Failure needs manual inspection.")


def _read_log(row: dict[str, Any]) -> str:
    log = row.get("log")
    if not log:
        return ""
    path = PROJECT_ROOT / str(log)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _current_missing_summary() -> tuple[list[dict[str, str]], str]:
    sys.path.insert(0, str(PROJECT_ROOT / "ops/launch"))
    import rl_prebuild_missing_images as prebuild  # type: ignore

    try:
        records = prebuild._load_missing_records(prebuild.DEFAULT_PARQUETS)  # noqa: SLF001
    except Exception as exc:  # Keep the report useful even if Docker is temporarily unreachable.
        return [], f"unavailable: {exc!r}"
    return records, ""


def _tmux_sessions() -> list[str]:
    proc = subprocess.run(
        ["tmux", "list-sessions"],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if "rl-prebuild" in line]


def _extract_bullet_value(text: str, prefix: str) -> str:
    marker = f"- {prefix}:"
    for line in text.splitlines():
        if line.startswith(marker):
            return line[len(marker):].strip()
    return ""


def _extract_table(text: str, heading: str) -> list[str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == heading:
            table: list[str] = []
            for row in lines[index + 1 :]:
                if not row.strip():
                    if table:
                        break
                    continue
                if not row.startswith("|"):
                    if table:
                        break
                    continue
                table.append(row)
            return table
    return []


def _legacy_audit_summary() -> list[str]:
    """Summarize old rl_image*.md snapshots inside the canonical report."""
    rows: list[str] = []
    for path in LEGACY_AUDIT_DOCS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        parquet = _extract_bullet_value(text, "Parquet")
        training_rows = _extract_bullet_value(text, "Training rows")
        unique_entries = _extract_bullet_value(text, "Unique expected image entries")
        present = _extract_bullet_value(text, "Present now")
        missing = _extract_bullet_value(text, "Missing now")
        rows.append(
            f"| `{path.name}` | {parquet or '-'} | {training_rows or '-'} | "
            f"{unique_entries or '-'} | {present or '-'} | {missing or '-'} |"
        )
    if rows:
        return rows
    for item in LEGACY_AUDIT_SNAPSHOTS:
        rows.append(
            f"| `{item['source']}` | {item['parquet']} | {item['rows']} | "
            f"{item['unique']} | {item['present']} | {item['missing']} |"
        )
    return rows


def _legacy_bench_tables() -> list[str]:
    sections: list[str] = []
    for path in LEGACY_AUDIT_DOCS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        table = _extract_table(text, "## Summary By Bench (Unique Tags)")
        if not table:
            continue
        sections += ["", f"### `{path.name}` Bench Summary", ""]
        sections.extend(table)
    if sections:
        return sections
    for item in LEGACY_AUDIT_SNAPSHOTS:
        sections += [
            "",
            f"### `{item['source']}` Bench Summary",
            "",
            "| bench | kind/status counts |",
            "|---|---|",
        ]
        for bench, summary in item["bench"].items():
            sections.append(f"| `{bench}` | {summary} |")
    return sections


def _write_report(out_path: Path, prebuild_root: Path) -> None:
    rows = _load_status_rows(prebuild_root)
    latest = _latest_by_tag(rows)
    latest_rows = list(latest.values())
    status_counts = Counter((r.get("bench"), r.get("kind"), r.get("status")) for r in latest_rows)
    all_status_counts = Counter((r.get("bench"), r.get("kind"), r.get("status")) for r in rows)

    missing_records, missing_error = _current_missing_summary()
    missing_counts = Counter((r.get("bench"), r.get("kind")) for r in missing_records)

    failed_latest = [r for r in latest_rows if r.get("status") != "ok"]
    classified_failures: list[dict[str, Any]] = []
    for row in sorted(failed_latest, key=lambda r: (str(r.get("bench")), str(r.get("task_id")))):
        log_text = _read_log(row)
        failure_class, action = _classify_failure(log_text)
        row = dict(row)
        row["failure_class"] = failure_class
        row["action"] = action
        classified_failures.append(row)

    removal_candidates = [
        row
        for row in classified_failures
        if str(row.get("failure_class", "")).startswith("dockerfile_")
    ]

    lines = [
        "# RL Docker Prebuild Status",
        "",
        f"- updated_at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "- canonical_file: `docs/rl_docker_prebuild.md`",
        "- scope: current RL train/eval parquet Docker image coverage and prebuild failure candidates.",
        "",
        "## Current Image Coverage",
        "",
    ]
    if missing_error:
        lines.append(f"- current audit: `{missing_error}`")
    else:
        lines += [
            f"- missing_unique_images_now: {len(missing_records)}",
            "",
            "| bench | kind | missing |",
            "|---|---|---:|",
        ]
        for (bench, kind), count in sorted(missing_counts.items()):
            lines.append(f"| `{bench}` | `{kind}` | {count} |")
        if not missing_counts:
            lines.append("| - | - | 0 |")

    lines += [
        "",
        "## Prebuild Status",
        "",
        "| bench | kind | latest_status | unique_tags |",
        "|---|---|---|---:|",
    ]
    for (bench, kind, status), count in sorted(status_counts.items()):
        lines.append(f"| `{bench}` | `{kind}` | `{status}` | {count} |")
    if not status_counts:
        lines.append("| - | - | - | 0 |")

    lines += [
        "",
        "## Historical Image Cache Audit Snapshots",
        "",
        "These summarize the old `docs/rl_image*.md` files so Docker-related status has one canonical entry point.",
        "",
        "| source | parquet | rows | unique_images | present_then | missing_then |",
        "|---|---|---:|---:|---:|---:|",
    ]
    legacy_rows = _legacy_audit_summary()
    if legacy_rows:
        lines.extend(legacy_rows)
    else:
        lines.append("| - | - | - | - | - | - |")
    lines.extend(_legacy_bench_tables())

    lines += [
        "",
        "### Raw Status Rows",
        "",
        "| bench | kind | status | rows |",
        "|---|---|---|---:|",
    ]
    for (bench, kind, status), count in sorted(all_status_counts.items()):
        lines.append(f"| `{bench}` | `{kind}` | `{status}` | {count} |")
    if not all_status_counts:
        lines.append("| - | - | - | 0 |")

    lines += [
        "",
        "## Remove From RL Data Before Next Run",
        "",
        "These are deterministic Dockerfile/build-context problems. Do not remove automatically; review this list first, then filter from train/eval parquet/split like the previous bad Seta tasks.",
        "",
        "| bench | task_id | tag | failure_class | evidence |",
        "|---|---|---|---|---|",
    ]
    if removal_candidates:
        for row in removal_candidates:
            lines.append(
                f"| `{row.get('bench')}` | `{row.get('task_id')}` | `{row.get('tag')}` | "
                f"`{row.get('failure_class')}` | `{row.get('log')}` |"
            )
    else:
        lines.append("| - | - | - | - | - |")

    lines += [
        "",
        "## Other Failures To Inspect",
        "",
        "| bench | task_id | status | failure_class | action | log |",
        "|---|---|---|---|---|---|",
    ]
    other_failures = [row for row in classified_failures if row not in removal_candidates]
    if other_failures:
        for row in other_failures:
            lines.append(
                f"| `{row.get('bench')}` | `{row.get('task_id')}` | `{row.get('status')}` | "
                f"`{row.get('failure_class')}` | {row.get('action')} | `{row.get('log')}` |"
            )
    else:
        lines.append("| - | - | - | - | - | - |")

    lines += [
        "",
        "## Active Commands",
        "",
        "```bash",
        "tmux list-sessions | rg 'rl-prebuild'",
        "tail -f experiments/infra/rl/prebuild/prebuild_rl_seta_20260523_133955/status.jsonl",
        "tail -f experiments/infra/rl/prebuild/prebuild_rl_seta_20260523_133955/prebuild.log",
        "python3 ops/monitor/rl_docker_prebuild_report.py",
        "```",
        "",
        "## Active Tmux Sessions",
        "",
    ]
    sessions = _tmux_sessions()
    if sessions:
        lines.extend(f"- `{session}`" for session in sessions)
    else:
        lines.append("- none")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prebuild-root", type=Path, default=DEFAULT_PREBUILD_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if os.environ.get("DOCKER_HOST") in (None, "", "tcp://127.0.0.1:2375", "tcp://127.0.0.1:2376", "unix:///tmp/apex-docker.sock"):
        os.environ["DOCKER_HOST"] = "unix:///tmp/local-docker-overlay2.sock"
    _write_report(args.out, args.prebuild_root)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
