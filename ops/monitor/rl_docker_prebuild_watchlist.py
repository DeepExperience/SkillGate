#!/usr/bin/env python3
"""Summarize RL Docker build/start and verifier timeout hotspots.

This intentionally reads only Relax driver logs. It does not mutate Docker state,
delete images, or touch Ray processes. Use it after a run or during a run to
decide which task images should be prebuilt before increasing env concurrency.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))

LOCAL_BUILD_DATASETS = {
    "skillsbench-no-skills": PROJECT_ROOT / "datasets/skillsbench/tasks",
    "seta-synth": PROJECT_ROOT / "datasets/seta/dataset/seta_synth_top300",
    "seta": PROJECT_ROOT / "datasets/seta/dataset/seta_baseline_30",
}


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ENV_FAIL_RE = re.compile(
    r"\[(?P<task>[^\]]+)\]\s+env setup attempt\s+(?P<attempt>\d+)/(?P<max_attempt>\d+)\s+failed.*?:\s*(?P<error>.*)$"
)
GRADER_FAIL_RE = re.compile(
    r"\[(?P<task>[^\]]+)\]\s+(?P<context>grader infrastructure failure|terminal grade .*? infrastructure failure|.*?VERIFIER TIMEOUT).*?(?P<error>Command timed out|VERIFIER TIMEOUT|LauncherError|RuntimeError|$)"
)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def classify_env_error(error: str) -> str:
    lower = error.lower()
    if "cannot connect to the docker daemon" in lower:
        return "docker_daemon_unreachable"
    if "docker build failed: command timed out" in lower:
        return "build_timeout"
    if "docker build failed" in lower and any(
        marker in lower
        for marker in (
            "curl 56",
            "early eof",
            "invalid index-pack",
            "rpc failed",
            "fetch-pack",
            "gnutls",
            "lake build",
        )
    ):
        return "build_network_or_source_fetch"
    if "docker build failed" in lower:
        return "build_failed_other"
    if "failed to start container: command timed out" in lower:
        return "container_start_timeout"
    if "command timed out" in lower:
        return "setup_command_timeout"
    return "setup_other"


def short_error(error: str, limit: int = 220) -> str:
    error = " ".join(error.replace("\\n", " ").split())
    return error[:limit]


def parse_log(log_path: Path) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "env_total": 0,
            "env_classes": Counter(),
            "verifier_total": 0,
            "env_examples": [],
            "verifier_examples": [],
        }
    )
    for lineno, raw in enumerate(log_path.open("r", encoding="utf-8", errors="replace"), 1):
        line = strip_ansi(raw.rstrip())

        env_match = ENV_FAIL_RE.search(line)
        if env_match:
            task = env_match.group("task")
            error = env_match.group("error")
            klass = classify_env_error(error)
            entry = stats[task]
            entry["env_total"] = int(entry["env_total"]) + 1
            entry["env_classes"][klass] += 1  # type: ignore[index]
            examples = entry["env_examples"]
            if len(examples) < 3:
                examples.append(f"L{lineno} env/{klass}: {short_error(error)}")
            continue

        grader_match = GRADER_FAIL_RE.search(line)
        if grader_match:
            task = grader_match.group("task")
            entry = stats[task]
            entry["verifier_total"] = int(entry["verifier_total"]) + 1
            examples = entry["verifier_examples"]
            if len(examples) < 3:
                examples.append(f"L{lineno} verifier: {short_error(line)}")

    return stats


def resolve_local_build_tags(task: str) -> list[str]:
    tags: list[str] = []
    for dataset_tag, root in LOCAL_BUILD_DATASETS.items():
        dockerfile = root / task / "environment" / "Dockerfile"
        if dockerfile.exists():
            tags.append(f"unified-{dataset_tag}-{task}:latest")
    return tags


def render_markdown(stats: dict[str, dict[str, object]], log_path: Path) -> str:
    rows = []
    for task, entry in stats.items():
        env_classes: Counter[str] = entry["env_classes"]  # type: ignore[assignment]
        env_total = int(entry["env_total"])
        verifier_total = int(entry["verifier_total"])
        if not env_total and not verifier_total:
            continue
        dominant = env_classes.most_common(1)[0][0] if env_classes else "verifier_timeout"
        action = "inspect"
        if dominant.startswith("build"):
            action = "prebuild_image"
        elif dominant == "container_start_timeout" or dominant == "docker_daemon_unreachable":
            action = "docker_start_retry_or_cap"
        elif verifier_total and not env_total:
            action = "verifier_timeout_review"
        rows.append((env_total + verifier_total, task, env_total, verifier_total, dominant, action, entry))

    rows.sort(reverse=True)
    prebuild_rows = [
        (total, task, env_total, dominant, resolve_local_build_tags(task))
        for total, task, env_total, _verifier_total, dominant, action, _entry in rows
        if action == "prebuild_image"
    ]
    non_prebuild_rows = [
        (total, task, env_total, verifier_total, dominant, action)
        for total, task, env_total, verifier_total, dominant, action, _entry in rows
        if action != "prebuild_image"
    ]
    out = [
        "# RL Docker / Verifier Hotspot Watchlist",
        "",
        f"- Source log: `{log_path}`",
        "- Purpose: identify tasks that should be prebuilt, rate-limited, or verifier-reviewed before raising env concurrency.",
        "",
        "## Prebuild Candidates",
        "",
        "| rank | task | env failures | dominant class | expected local image tags |",
        "|---:|---|---:|---|---|",
    ]
    if prebuild_rows:
        for rank, (_total, task, env_total, dominant, tags) in enumerate(prebuild_rows, 1):
            tag_text = "<br>".join(f"`{tag}`" for tag in tags) if tags else "`unresolved_tag`"
            out.append(f"| {rank} | `{task}` | {env_total} | `{dominant}` | {tag_text} |")
    else:
        out.append("| - | - | 0 | - | - |")
    out.extend(
        [
            "",
            "## Non-Prebuild Bottlenecks",
            "",
            "| rank | task | env failures | verifier failures | dominant class | suggested action |",
            "|---:|---|---:|---:|---|---|",
        ]
    )
    if non_prebuild_rows:
        for rank, (_total, task, env_total, verifier_total, dominant, action) in enumerate(non_prebuild_rows, 1):
            out.append(f"| {rank} | `{task}` | {env_total} | {verifier_total} | `{dominant}` | `{action}` |")
    else:
        out.append("| - | - | 0 | 0 | - | - |")
    out.extend(
        [
            "",
            "## Full Evidence",
            "",
        "| rank | task | env failures | verifier failures | dominant class | suggested action | examples |",
        "|---:|---|---:|---:|---|---|---|",
        ]
    )
    for rank, (_, task, env_total, verifier_total, dominant, action, entry) in enumerate(rows, 1):
        examples = "<br>".join(
            list(entry["env_examples"])[:2] + list(entry["verifier_examples"])[:1]  # type: ignore[index]
        )
        out.append(
            f"| {rank} | `{task}` | {env_total} | {verifier_total} | `{dominant}` | `{action}` | {examples} |"
        )
    out.append("")
    return "\n".join(out)


def default_log_path() -> Path:
    pointer = Path("experiments/rl/current/latest.txt")
    if pointer.exists():
        run = Path(pointer.read_text(encoding="utf-8").strip())
        return run / "driver.log"
    raise SystemExit("No --log provided and current_rl_resume_run.txt not found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=None, help="Relax driver.log path")
    parser.add_argument("--out", type=Path, default=None, help="Write markdown report")
    args = parser.parse_args()

    log_path = args.log or default_log_path()
    if not log_path.exists():
        raise SystemExit(f"log not found: {log_path}")

    report = render_markdown(parse_log(log_path), log_path)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])
    main()
