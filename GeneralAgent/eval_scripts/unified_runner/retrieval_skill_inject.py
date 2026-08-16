"""Inject retrieval-selected skills into a task's container.

Context:
    Retrieval pipeline produces per-task top-N skills from a 574-skill public
    library at /mnt/.../Projects/skill_libraries/merged/. This module loads
    that jsonl + docker-cp's the top-N skill dirs into the agent container at
    /root/.claude/skills/ and siblings (7 paths) mirroring SkillsBench's
    Dockerfile convention.

2026-04-20 v6 changes:
  - Default top_n now 10 (was 3) — richer context so agent can pick among
    semantically-related options instead of being limited to 3.
  - Prompt hint now includes each skill's `description:` (YAML frontmatter
    summary) so agent can decide whether to read SKILL.md without paying the
    full file-read token cost.
  - JSON schema reads `reranked_top10` first (v6 pipeline), falls back to
    `reranked_top5` (v5 pipeline) for backward compatibility.

Call from the runner after docker_run / start_container, before running the agent.

Usage:
    from unified_runner.retrieval_skill_inject import (
        load_retrieval_mapping, inject_retrieval_skills,
    )
    mapping = load_retrieval_mapping("experiments/<date>/<run_id>/retrieval_results/<bench>.jsonl")
    inject_retrieval_skills(docker_run_fn, cname, task_id, mapping, top_n=10)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shlex
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple

from unified_runner.openclaw_compat import (
    SkillPromptEntry,
    format_skills_for_openclaw,
)

SKILL_LIB_ROOT = Path("/path/to/skillRL/skill_libraries/merged")

# Agent skill paths, mirrors SkillsBench Dockerfile COPY pattern.
AGENT_SKILL_DIRS = [
    "/root/.claude/skills",
    "/root/.codex/skills",
    "/root/.opencode/skill",
    "/root/.goose/skills",
    "/root/.factory/skills",
    "/root/.agents/skills",
    "/root/.gemini/skills",
]

# Default top_n for retrieval/irrelevant arms (2026-04-20 v6: 10 per user ask).
DEFAULT_TOP_N = 10


# ---------------------------------------------------------------------------
# Skill-metadata helpers
# ---------------------------------------------------------------------------

# Module-level cache for parsed skill descriptions (SKILL.md YAML frontmatter).
_SKILL_DESC_CACHE: Dict[str, str] = {}


def _source_skill_dir(skill_dir: Path) -> Path | None:
    """Return the original library skill dir recorded by merged/_source.json."""
    source_meta = skill_dir / "_source.json"
    if not source_meta.exists():
        return None
    try:
        meta = json.loads(source_meta.read_text(encoding="utf-8"))
    except Exception:
        return None
    source_repo = str(meta.get("source_repo") or "").strip()
    source_path = str(meta.get("source_path") or "").strip()
    if not source_repo or not source_path:
        return None
    root = SKILL_LIB_ROOT.parent / source_repo / source_path
    return root if root.exists() else None


def _copy_plugin_companion_resources(src: Path, dst: Path) -> None:
    """Copy plugin-level resources that merged skills can reference indirectly.

    Some public-library plugins keep large reference material outside
    `skills/<name>/`, e.g. `plugins/linux-sysadmin/guides/...`, while the skill
    text tells agents to read `guides/...`. The merged library stores only the
    skill directory, so copying the merged dir alone makes otherwise-valid reads
    fail inside the benchmark container. When `_source.json` points to an
    original `.../skills/<name>` dir, mirror selected sibling resource dirs into
    the staged skill directory to preserve the paths agents learned to use.
    """
    original = _source_skill_dir(src)
    if original is None:
        return
    parts = original.parts
    if "skills" not in parts:
        return
    skills_index = len(parts) - 1 - parts[::-1].index("skills")
    plugin_root = Path(*parts[:skills_index])
    for resource_name in ("guides",):
        resource_src = plugin_root / resource_name
        resource_dst = dst / resource_name
        if resource_src.is_dir() and not resource_dst.exists():
            shutil.copytree(resource_src, resource_dst)


def _read_skill_description(skill_dir: Path) -> str:
    """Read SKILL.md YAML frontmatter and return the `description` field.

    Falls back to:
      - First non-empty prose line after frontmatter if description is missing.
      - "(no summary available)" if nothing parseable.

    Result is cached across calls (skill_libraries/merged/ is immutable).
    """
    key = str(skill_dir.expanduser().resolve()) if skill_dir.exists() else str(skill_dir)
    if key in _SKILL_DESC_CACHE:
        return _SKILL_DESC_CACHE[key]

    skm = skill_dir / "SKILL.md"
    if not skm.exists():
        _SKILL_DESC_CACHE[key] = "(no SKILL.md)"
        return _SKILL_DESC_CACHE[key]

    desc = ""
    try:
        txt = skm.read_text(encoding="utf-8", errors="replace")
        if txt.startswith("---\n"):
            end = txt.find("\n---", 4)
            if end > 0:
                # Parse YAML frontmatter for `description`
                try:
                    import yaml
                    data = yaml.safe_load(txt[4:end])
                    if isinstance(data, dict):
                        raw_desc = data.get("description")
                        if raw_desc:
                            desc = str(raw_desc).strip().replace("\n", " ")
                except Exception:
                    # YAML parse error (~30 skills have this). Fall through to body fallback.
                    pass
                if not desc:
                    # Fallback: first non-empty prose line after `---`
                    body = txt[end + 4:].lstrip()
                    for line in body.splitlines():
                        line = line.strip()
                        if line and not line.startswith("#") and not line.startswith("```"):
                            desc = line
                            break
        if not desc:
            desc = "(no summary available)"
    except Exception:
        desc = "(read error)"

    # Trim to reasonable length for prompt injection (avoid bloating system prompt)
    if len(desc) > 300:
        desc = desc[:297].rstrip() + "..."
    _SKILL_DESC_CACHE[key] = desc
    return desc


def _prompt_skill_dir(skill_path: str, skill_lib_root: Path) -> Path:
    """Return the directory whose SKILL.md should be summarized in the prompt.

    Retrieval JSONL normally stores absolute skill directories. Older public
    library mappings also work with the historical merged-library fallback.
    """
    candidate = Path(skill_path)
    if candidate.is_file() and candidate.name == "SKILL.md":
        candidate = candidate.parent
    if candidate.is_dir():
        return candidate
    return skill_lib_root / os.path.basename(skill_path.rstrip("/"))


# ---------------------------------------------------------------------------
# Mapping loaders
# ---------------------------------------------------------------------------

def load_retrieval_mapping(retrieval_jsonl: str | Path) -> Dict[str, List[str]]:
    """Read retrieval jsonl, return {task_id: [skill_path_1, ...]}.

    Order of fields checked (v6 → v5 fallback):
      reranked_top10  → v6 pipeline (embedding + reranker, no LLM rerank)
      reranked_top5   → v5 pipeline (embedding + LLM rerank)
      coarse_top20    → last resort (embedding only)
    """
    mapping: Dict[str, List[str]] = {}
    with open(retrieval_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tid = d["task_id"]
            # v6 schema first, fall back gracefully
            skills = (d.get("reranked_top10")
                      or d.get("reranked_top5")
                      or d.get("coarse_top20")
                      or [])
            paths = []
            for s in skills:
                if not isinstance(s, dict):
                    continue
                p = s.get("skill_path")
                if p and Path(p).is_dir():
                    paths.append(p)
            mapping[tid] = paths
    return mapping


def _coarse_top_names_per_task(retrieval_jsonl: str | Path) -> Dict[str, Set[str]]:
    """Return {task_id: set(names_in_coarse_top)} — the exclusion set for
    irrelevant-arm negative control.

    Uses `coarse_top50` (v6) or `coarse_top20` (v5) — whichever is available.
    Also adds reranked_top10/top5 names (they're subset of coarse but cheap to add).
    """
    out: Dict[str, Set[str]] = {}
    with open(retrieval_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tid = d["task_id"]
            coarse = (d.get("coarse_top50")
                      or d.get("coarse_top20")
                      or [])
            rerank = (d.get("reranked_top10")
                      or d.get("reranked_top5")
                      or [])
            names: Set[str] = set()
            for s in coarse + rerank:
                if isinstance(s, dict):
                    nm = s.get("skill_name")
                    if not nm and s.get("skill_path"):
                        nm = os.path.basename(s["skill_path"].rstrip("/"))
                    if nm:
                        names.add(nm)
            out[tid] = names
    return out


def build_irrelevant_mapping(
    retrieval_jsonl: str | Path,
    top_n: int = DEFAULT_TOP_N,
    skill_lib_root: Path | None = None,
) -> Dict[str, List[str]]:
    """Return {task_id: [N irrelevant skill paths]} — negative-control arm.

    Excludes anything appearing in coarse_top50 (or coarse_top20 fallback).
    Deterministic: seeded by SHA-256(task_id).
    """
    root = Path(skill_lib_root) if skill_lib_root else SKILL_LIB_ROOT
    all_skill_dirs = sorted(
        str(p) for p in root.iterdir()
        if p.is_dir() and (p / "SKILL.md").exists()
    )

    coarse = _coarse_top_names_per_task(retrieval_jsonl)
    out: Dict[str, List[str]] = {}

    for tid, excluded in coarse.items():
        candidates = [p for p in all_skill_dirs if os.path.basename(p) not in excluded]
        seed = int.from_bytes(hashlib.sha256(tid.encode()).digest()[:8], "big")
        rng = random.Random(seed)
        picked = rng.sample(candidates, min(top_n, len(candidates)))
        out[tid] = picked
    return out


# ---------------------------------------------------------------------------
# Docker injection
# ---------------------------------------------------------------------------

def inject_retrieval_skills(
    docker_run_fn: Callable,
    container_name: str,
    task_id: str,
    mapping: Dict[str, List[str]],
    top_n: int = DEFAULT_TOP_N,
    verbose: bool = True,
) -> int:
    """Copy top-N retrieval skills into the container's agent skill dirs (7 paths)."""
    skills = mapping.get(task_id, [])
    if not skills:
        if verbose:
            print(f"    [retrieval-skills] task {task_id}: no retrieval entry; skipping")
        return 0
    selected = skills[:top_n]

    staged_names: list[str] = []
    with tempfile.TemporaryDirectory(prefix=f"retrieval-skills-{task_id}-") as tmp:
        stage_dir = Path(tmp) / "skills"
        stage_dir.mkdir()
        for skill_path in selected:
            src = Path(skill_path)
            if not src.is_dir():
                if verbose:
                    print(f"    [retrieval-skills] missing skill dir: {src}")
                continue
            skill_name = src.name
            staged_skill_dir = stage_dir / skill_name
            shutil.copytree(src, staged_skill_dir, dirs_exist_ok=True)
            _copy_plugin_companion_resources(src, staged_skill_dir)
            staged_names.append(skill_name)

        if not staged_names:
            if verbose:
                print(f"    [retrieval-skills] task {task_id}: 0/{len(selected)} skills staged")
            return 0

        def run_retry(cmd: list[str], timeout: int, attempts: int = 3) -> tuple[str, str, int]:
            last = ("", "", -1)
            for attempt in range(attempts):
                last = docker_run_fn(cmd, timeout=timeout)
                if last[2] == 0:
                    return last
                if attempt < attempts - 1:
                    time.sleep(1.5 * (attempt + 1))
            return last

        remote_stage = "/tmp/retrieval-skills"
        quoted_dirs = " ".join(shlex.quote(d) for d in AGENT_SKILL_DIRS)
        # 2026-04-26: bumped 60→120s to match the other inject steps. Under
        # your-docker-host dockerd contention from concurrent workloads, a 60s ceiling
        # was tight enough to occasionally fail; 120s covers the same worst-
        # case as the cp/fan-out steps below.
        _, stderr, rc = run_retry(
            ["docker", "exec", container_name, "sh", "-lc",
             f"rm -rf {shlex.quote(remote_stage)} && mkdir -p {shlex.quote(remote_stage)} {quoted_dirs}"],
            timeout=120,
        )
        if rc != 0:
            if verbose:
                print(f"    [retrieval-skills] mkdir stage/targets fail: {stderr[:160]}")
            return 0

        _, stderr, rc = run_retry(
            ["docker", "cp", f"{stage_dir}/.", f"{container_name}:{remote_stage}/"],
            timeout=120,
        )
        if rc != 0:
            if verbose:
                print(f"    [retrieval-skills] bulk docker cp fail: {stderr[:160]}")
            return 0

        # Some existing SFT/RL trajectories learned to inspect README.md even
        # though OpenClaw advertises SKILL.md. Keep SKILL.md as the canonical
        # file, but provide README.md as a byte-identical compatibility alias so
        # those reads do not become false environment failures.
        copy_script = (
            "set -e; "
            f"for s in {shlex.quote(remote_stage)}/*; do "
            "[ -d \"$s\" ] || continue; "
            "[ -f \"$s/SKILL.md\" ] || continue; "
            "[ -f \"$s/README.md\" ] || cp \"$s/SKILL.md\" \"$s/README.md\"; "
            "done; "
            f"for d in {quoted_dirs}; do "
            f"mkdir -p \"$d\"; cp -a {shlex.quote(remote_stage)}/. \"$d\"/; "
            "done"
        )
        _, stderr, rc = run_retry(
            ["docker", "exec", container_name, "sh", "-lc", copy_script],
            timeout=120,
        )
        if rc != 0:
            if verbose:
                print(f"    [retrieval-skills] in-container fanout fail: {stderr[:160]}")
            return 0

    injected = len(staged_names)
    if verbose:
        for skill_name in staged_names:
            print(f"    [retrieval-skills] injected {skill_name}")
    if verbose:
        print(f"    [retrieval-skills] task {task_id}: {injected}/{len(selected)} skills placed "
              f"(top_n={top_n} of {len(skills)} retrieved)")
    return injected


# ---------------------------------------------------------------------------
# Prompt hint (v6: includes summaries)
# ---------------------------------------------------------------------------

def build_retrieval_prompt_hint(
    task_id: str,
    mapping: Dict[str, List[str]],
    top_n: int = DEFAULT_TOP_N,
    arm: str = "retrieval",
    skill_lib_root: Path | None = None,
) -> str:
    """Return an OpenClaw-compatible <available_skills> block.

    v6: summary pulled from each skill's SKILL.md YAML frontmatter `description:`.
    OpenClaw expects <location> to be the exact SKILL.md path, not the skill
    directory. The runner may replace /root/.claude/skills with a host workdir
    after this function returns.
    Empty string if no skills are injected.

    `arm` is "retrieval" or "irrelevant"; phrased identically so model cannot
    distinguish positive from negative control at prompt level.
    """
    skills = mapping.get(task_id, [])[:top_n]
    if not skills:
        return ""
    root = Path(skill_lib_root) if skill_lib_root else SKILL_LIB_ROOT

    entries: list[SkillPromptEntry] = []
    for p in skills:
        name = os.path.basename(p.rstrip("/"))
        summary = _read_skill_description(_prompt_skill_dir(p, root))
        entries.append(
            SkillPromptEntry(
                name=name,
                description=summary,
                location=f"/root/.claude/skills/{name}/SKILL.md",
            )
        )
    hint = format_skills_for_openclaw(entries)
    selection_instruction = os.environ.get("UNIFIED_SKILL_SELECTION_INSTRUCTION", "").strip()
    if not selection_instruction:
        return hint
    return "\n".join(
        [
            hint,
            "<skill_selection_instruction>",
            selection_instruction,
            "</skill_selection_instruction>",
        ]
    )


def build_top1_skill_text_prompt(
    task_id: str,
    mapping: Dict[str, List[str]],
    *,
    max_chars: int = 0,
) -> tuple[str, str]:
    """Return (prompt, skill_name) with the full top-1 SKILL.md content.

    This arm intentionally bypasses the "should I read a skill?" decision:
    the selected skill's text is preloaded into the system prompt. Companion
    resources are not inlined; this tests direct text usefulness, not runtime
    file access.
    """
    skills = mapping.get(task_id, [])
    if not skills:
        return "", ""
    skill_dir = Path(skills[0])
    skill_name = skill_dir.name
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        return "", skill_name
    try:
        content = skill_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "", skill_name
    truncated = False
    if max_chars > 0 and len(content) > max_chars:
        content = content[:max_chars].rstrip() + "\n\n[SKILL.md truncated by max_chars]"
        truncated = True
    source = _source_skill_dir(skill_dir)
    source_line = f"Source skill directory: {source}" if source else f"Source skill directory: {skill_dir}"
    prompt = "\n".join(
        [
            "<preloaded_top1_skill>",
            f"<name>{skill_name}</name>",
            f"<location>/root/.claude/skills/{skill_name}/SKILL.md</location>",
            f"<truncated>{str(truncated).lower()}</truncated>",
            source_line,
            "--- BEGIN SKILL.md ---",
            content,
            "--- END SKILL.md ---",
            "</preloaded_top1_skill>",
        ]
    )
    return prompt, skill_name


# ---------------------------------------------------------------------------
# Host-mode injection (copy-tree into workdir/.claude/skills) — claw host mode.
# ---------------------------------------------------------------------------

def inject_retrieval_skills_host(
    workdir: Path,
    task_id: str,
    mapping: Dict[str, List[str]],
    top_n: int = DEFAULT_TOP_N,
    verbose: bool = True,
) -> int:
    """Host-mode variant: shutil.copytree into workdir/.claude/skills/<name>.

    Used by run_unified_claw.py when the agent runs on the host FS (not in container).
    """
    import shutil
    skills = mapping.get(task_id, [])
    if not skills:
        return 0
    selected = skills[:top_n]
    dst_base = workdir / ".claude" / "skills"
    dst_base.mkdir(parents=True, exist_ok=True)

    injected = 0
    for skill_path in selected:
        src = Path(skill_path)
        if not src.is_dir():
            continue
        dst = dst_base / src.name
        try:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            _copy_plugin_companion_resources(src, dst)
            injected += 1
            if verbose:
                print(f"    [retrieval-skills-host] {src.name} → {dst}")
        except Exception as e:
            if verbose:
                print(f"    [retrieval-skills-host] {src.name} failed: {e}")
    if verbose:
        print(f"    [retrieval-skills-host] task {task_id}: {injected}/{len(selected)}")
    return injected


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

def main():
    """Dry-run sanity: load a retrieval jsonl and print coverage stats + sample prompt."""
    import argparse
    ap = argparse.ArgumentParser(description="Dry-run retrieval / irrelevant skill injector (v6)")
    ap.add_argument("jsonl", help="Path to retrieval jsonl")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--sample-task", help="Show injection plan for this task_id")
    ap.add_argument("--irrelevant", action="store_true",
                    help="Show irrelevant-arm (negative-control) mapping instead of retrieval")
    args = ap.parse_args()

    if args.irrelevant:
        mapping = build_irrelevant_mapping(args.jsonl, top_n=args.top_n)
        arm = "irrelevant"
    else:
        mapping = load_retrieval_mapping(args.jsonl)
        arm = "retrieval"
    n_with = sum(1 for v in mapping.values() if v)
    n_avg = sum(len(v) for v in mapping.values()) / max(1, n_with)
    print(f"[{arm}] {len(mapping)} tasks, {n_with} have ≥1 skill, avg {n_avg:.1f} skills/task")

    if args.sample_task:
        hint = build_retrieval_prompt_hint(args.sample_task, mapping, args.top_n, arm=arm)
        print(f"\n=== {arm} prompt hint for task {args.sample_task} (top_n={args.top_n}) ===")
        print(hint or "(no skills for this task)")
        skills = mapping.get(args.sample_task, [])[: args.top_n]
        print(f"\nSkill paths (would docker cp):")
        for p in skills:
            exists = Path(p).is_dir()
            print(f"  {'OK' if exists else 'MISS'}: {p}")


if __name__ == "__main__":
    main()
