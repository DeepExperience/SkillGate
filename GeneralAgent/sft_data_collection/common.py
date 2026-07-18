#!/usr/bin/env python3
"""Shared helpers for SFT data collection scripts.

Naming conventions in this package:
- "bench" = internal short name (claw, tb2, sb_ns, seta_synth, swe_lite). Used
  in plan files and split files because it's stable.
- "output_bench_name" = the dir name a runner actually writes to under
  experiments/<date>/<run_id>/results/<output_bench>/. Differs from bench because
  some runners adopt different naming (sb_ns → skillsbench-no-skills).
- "retrieval_bench_name" = the bench name in the retrieval jsonl directory
  (sb_ns → skillsbench, seta_synth → seta).

Three names exist because (a) we want stable bench keys in plans, (b) the
runners write results under historical dir names we don't want to rename,
(c) the retrieval pipeline produced jsonl files keyed by yet a third naming.
Code that crosses these boundaries MUST go through the helpers below.
"""

from __future__ import annotations

import ast
import hashlib
import os
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GeneralAgent.task_exclusions import filter_bad_tasks, is_bad_task

DEFAULT_CONFIG = MODULE_DIR / "configs" / "default_collection_config.json"
EXPERIMENTS_DIRNAME = "experiments"


# ---------------------------------------------------------------------------
# Path / IO helpers — all repo-relative paths get resolved against PROJECT_ROOT
# ---------------------------------------------------------------------------

def repo_path(path_value: str | Path) -> Path:
    """Resolve a path relative to project root (no-op for absolute paths)."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def display_path(path_value: str | Path) -> str:
    """Return a project-relative path when possible."""
    path = repo_path(path_value)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def experiments_root() -> Path:
    """Canonical root for experiment artifacts."""
    return repo_path(os.environ.get("EXPERIMENTS_ROOT", EXPERIMENTS_DIRNAME))


def infer_experiment_date(run_id: str, default: str | None = None) -> str:
    """Infer the date folder for a run id."""
    for pattern in (r"^(20\d{6})", r"(20\d{6})"):
        match = re.search(pattern, run_id or "")
        if match:
            return match.group(1)
    for env_name in ("EXPERIMENT_DATE", "DATE"):
        value = os.environ.get(env_name, "").strip()
        if re.fullmatch(r"20\d{6}", value):
            return value
    if default:
        return default
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def experiment_root(run_id: str) -> Path:
    """Canonical per-run root.

    Resolution order:
      1. EXPERIMENT_ROOT/RUN_ROOT env override (tests, wrapper-nested arms).
      2. Owner-aware RL paths when explicitly requested:
         train -> experiments/rl/runs/<experiment>/segments/<segment>/
         eval  -> experiments/rl/runs/<owner>/eval/<eval>/rows/<row>/
      3. Existing legacy roots, so non-RL runs started under the old layout
         keep resuming in place.
      4. New non-RL/general evaluation keeps the date-based layout.

    A central experiments/rl_eval directory is intentionally never created.
    """
    override = os.environ.get("EXPERIMENT_ROOT") or os.environ.get("RUN_ROOT")
    if override:
        return repo_path(override)
    kind = os.environ.get("UNIFIED_EXPERIMENT_KIND", "eval").strip().lower()
    if kind == "train":
        experiment_id = os.environ.get("EXPERIMENT_ID", "").strip()
        segment_id = os.environ.get("RUN_NAME", "").strip() or run_id
        if not experiment_id:
            raise RuntimeError("RL train output requires EXPERIMENT_ID")
        return experiments_root() / "rl" / "runs" / experiment_id / "segments" / segment_id
    owner = os.environ.get("OWNER_EXPERIMENT_ID", "").strip()
    eval_id = os.environ.get("EVAL_ID", "").strip()
    row_id = os.environ.get("EVAL_ROW_ID", "").strip() or run_id
    if owner or eval_id:
        if not owner or not eval_id:
            raise RuntimeError("owner-local RL eval requires OWNER_EXPERIMENT_ID and EVAL_ID")
        return experiments_root() / "rl" / "runs" / owner / "eval" / eval_id / "rows" / row_id
    dated_root = experiments_root() / infer_experiment_date(run_id) / run_id
    if dated_root.exists():
        return dated_root
    legacy_root = experiments_root() / run_id
    if legacy_root.exists():
        return legacy_root
    return dated_root


def experiment_plan_path(run_id: str) -> Path:
    return experiment_root(run_id) / "plans" / f"{run_id}.jsonl"


def experiment_combined_plan_path(run_id: str) -> Path:
    return experiment_root(run_id) / "plans" / f"{run_id}.combined.jsonl"


def experiment_chunks_root(run_id: str) -> Path:
    return experiment_root(run_id) / "plans" / "chunks"


def experiment_results_root(run_id: str) -> Path:
    return experiment_root(run_id) / "results"


def experiment_runner_log_root(run_id: str) -> Path:
    return experiment_root(run_id) / "logs" / "runner"


def experiment_sft_log_root(run_id: str) -> Path:
    return experiment_root(run_id) / "logs" / "sft_collection"


def experiment_status_path(run_id: str) -> Path:
    return experiment_sft_log_root(run_id) / "status.jsonl"


def experiment_collected_dir(run_id: str) -> Path:
    return experiment_root(run_id) / "collected"


def experiment_llamafactory_dir(run_id: str) -> Path:
    return experiment_root(run_id) / "llamafactory_data"


def secrets_path() -> Path:
    return repo_path(os.environ.get("PROJECT_SECRETS_FILE", "secrets/.env.secrets"))


def load_json(path_value: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path_value).read_text(encoding="utf-8"))


def dump_json(path_value: str | Path, payload: Any) -> None:
    """Pretty-print JSON with stable key ordering (good for diffs)."""
    path = repo_path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_task_lines(path_value: str | Path) -> list[str]:
    """Read a `tasks.txt` file (one task_id per line, # comments allowed)."""
    path = repo_path(path_value)
    tasks: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tasks.append(line)
    return tasks


def write_task_lines(path_value: str | Path, tasks: list[str]) -> None:
    path = repo_path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tasks) + ("\n" if tasks else ""), encoding="utf-8")


def filter_known_bad_tasks(bench: str, tasks: list[str | int]) -> list[str]:
    """Drop task ids with confirmed broken Docker environments."""
    return filter_bad_tasks(bench, tasks)


def read_jsonl(path_value: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file. Returns [] if file missing (callers handle this)."""
    path = repo_path(path_value)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path_value: str | Path, payload: dict[str, Any]) -> None:
    path = repo_path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# Freeze manifest helpers — fingerprint inputs so a re-run sees if anything
# the data depends on (skill library, retrieval jsonl) changed. Important
# because the whole point of "frozen" is reproducibility.
# ---------------------------------------------------------------------------

def sha256_file(path_value: str | Path) -> str | None:
    """Strong content hash for a single file. None if file missing."""
    path = repo_path(path_value)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(path_value: str | Path) -> dict[str, Any]:
    """Fingerprint a skill library directory.

    What we hash (intentional 80/20 trade-off):
      - sorted list of skill dir names (catches skill add/remove)
      - SHA-256 of each SKILL.md content (catches manifest edits — the
        main file users actually change when a skill is updated)

    What we do NOT hash:
      - supporting files (scripts/, references/, assets/). Walking ~2000
        skill dirs with rglob over NFS takes 5-15 minutes; SKILL.md-only
        finishes in ~10 seconds. SKILL.md changes whenever a skill is
        meaningfully revised, so this misses only "scripts changed without
        manifest changing", which is rare in practice.

    Earlier version hashed SKILL.md mtime only — silently missed any
    content change that preserved mtime (cp -p, git checkout, NFS skew).
    Now we hash content, so byte-identical SKILL.md → same fingerprint.
    """
    root = repo_path(path_value)
    skill_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d / "SKILL.md").is_file()
    )
    overall = hashlib.sha256()
    for skill_dir in skill_dirs:
        # name first (so add/remove changes fingerprint even if all
        # remaining SKILL.md are identical)
        overall.update(skill_dir.name.encode("utf-8"))
        overall.update(b"\0")
        try:
            overall.update(hashlib.sha256((skill_dir / "SKILL.md").read_bytes()).digest())
        except OSError:
            overall.update(b"<unreadable>")
        overall.update(b"\0")
    relative_root = (
        root.relative_to(PROJECT_ROOT) if root.is_relative_to(PROJECT_ROOT) else root
    )
    return {
        "path": str(relative_root),
        "hashed_files": "sorted skill dir names + SHA-256 of each SKILL.md content",
        "skill_count": len(skill_dirs),
        "fingerprint": overall.hexdigest(),
    }


# ---------------------------------------------------------------------------
# Holdout split helper
# ---------------------------------------------------------------------------

def stable_even_holdout(tasks: list[str], holdout_count: int) -> list[str]:
    """Pick `holdout_count` task_ids spread evenly across sorted(tasks).

    Deterministic: same input always yields same holdout. This matters because
    the holdout set MUST be reproducible across re-runs (otherwise SFT data
    can leak into test).

    Implementation: divide sorted tasks into N equal-sized buckets and take
    the midpoint of each. Earlier version used a "shift on collision" loop
    that could spin if both directions were already taken; the explicit
    bucket-midpoint approach can't collide because midpoints of N disjoint
    buckets are distinct by construction.
    """
    if holdout_count <= 0:
        return []
    if holdout_count >= len(tasks):
        return sorted(tasks)
    sorted_tasks = sorted(tasks)
    n = len(sorted_tasks)
    selected: list[str] = []
    for i in range(holdout_count):
        # Midpoint of the i-th bucket: floor((i + 0.5) * n / holdout_count).
        position = int((i + 0.5) * n / holdout_count)
        position = min(n - 1, max(0, position))
        selected.append(sorted_tasks[position])
    # Buckets are disjoint so positions are distinct, but de-dupe defensively
    # in case of off-by-one with very small holdout_count.
    seen: set[str] = set()
    deduped: list[str] = []
    for task in selected:
        if task not in seen:
            seen.add(task)
            deduped.append(task)
    return deduped


# ---------------------------------------------------------------------------
# SWE bench-specific helpers
# ---------------------------------------------------------------------------

def parse_swe_all_images(runner_path: str | Path) -> list[str]:
    """Extract ALL_IMAGES list from run_unified_swe.py via AST.

    We parse the AST instead of importing the module because the runner has
    heavy import-time side effects (it imports docker, sglang clients, etc.)
    and we just want the literal list.
    """
    path = repo_path(runner_path)
    module_ast = ast.parse(path.read_text(encoding="utf-8"))
    image_names: list[str] = []
    for node in module_ast.body:
        if isinstance(node, ast.Assign):
            target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "ALL_IMAGES" in target_names:
                image_names = list(ast.literal_eval(node.value))
                break
    return [swe_image_to_instance_id(name) for name in image_names]


def swe_image_to_instance_id(image_name: str) -> str:
    """Convert SWE-gym docker image tag to instance_id format.

    Example:
      xingyaoww/sweb.eval.x86_64.modin-project_s_modin-6298:latest
        → modin-project__modin-6298
    """
    return image_name.split(".")[-1].split(":")[0].replace("_s_", "__")


# ---------------------------------------------------------------------------
# Slug / model name normalization
# ---------------------------------------------------------------------------

def safe_slug(value: str, max_len: int = 56) -> str:
    """Make a string safe for filesystem paths. Long values get a hash suffix
    so two long-but-different names don't collide after truncation."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    if len(slug) <= max_len:
        return slug
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{slug[: max_len - 11]}-{digest}"


def model_short(model: str) -> str:
    """Short tag for filenames: 'qwen3.5-9b' → '9b'."""
    lowered = model.lower()
    if "9b" in lowered:
        return "9b"
    if "27b" in lowered:
        return "27b"
    return safe_slug(model, max_len=20)


# ---------------------------------------------------------------------------
# Bench naming (see top-of-file note explaining why these exist)
# ---------------------------------------------------------------------------

# Internal bench key → directory under experiments/<date>/<run_id>/results/.
# Has to match what the runners actually write (dataset_tag in harbor.py:710 etc).
_OUTPUT_BENCH_NAMES = {
    "sb_ns": "skillsbench-no-skills",
    "seta_synth": "seta-synth",
    "swe_lite": "swe",
    "claw": "claw",
    "tb2": "tb2",
}

# Internal bench key → bench name in the retrieval jsonl directory.
# These were chosen by the retrieval pipeline (retrieve_v6_3stage.py).
_RETRIEVAL_BENCH_NAMES = {
    "sb_ns": "skillsbench",
    "seta_synth": "seta",
    "swe_lite": "swe",
    "claw": "claw",
    "tb2": "tb2",
}


def output_bench_name(bench: str) -> str:
    if bench not in _OUTPUT_BENCH_NAMES:
        raise KeyError(f"unknown bench {bench!r}; expected one of {sorted(_OUTPUT_BENCH_NAMES)}")
    return _OUTPUT_BENCH_NAMES[bench]


def retrieval_bench_name(bench: str) -> str:
    if bench not in _RETRIEVAL_BENCH_NAMES:
        raise KeyError(f"unknown bench {bench!r}; expected one of {sorted(_RETRIEVAL_BENCH_NAMES)}")
    return _RETRIEVAL_BENCH_NAMES[bench]


def trajectory_file_name(bench: str, task_id: str) -> str:
    """Filename the runner uses for trajectory JSON.

    SWE instance_ids contain '/' (e.g. owner/repo); the SWE runner replaces
    them with '__'. Other benches use the raw task_id.
    """
    if bench == "swe_lite":
        return task_id.replace("/", "__") + ".json"
    return task_id + ".json"


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def load_retrieval_task_ids(path_value: str | Path) -> set[str]:
    """All task_ids covered by a retrieval jsonl. Returns set of str (we
    coerce because some benches use int task_ids)."""
    path = repo_path(path_value)
    if not path.exists():
        return set()
    task_ids: set[str] = set()
    with path.open(encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            if raw_line.strip():
                payload = json.loads(raw_line)
                task_ids.add(str(payload.get("task_id", "")))
    return task_ids


def ceiling_fraction_count(total: int, fraction: float) -> int:
    return int(math.ceil(total * fraction))
