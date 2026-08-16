#!/usr/bin/env python3
"""Rewrite embedded checkout roots and internal absolute symlinks in place."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAMP = ROOT / ".skillrl-root"
# The original absolute roots are deliberately not hard-coded: pass the root
# this tree (or a restored asset bundle) was exported from via --from-root,
# or set the SKILLRL_FORMER_* environment variables. The values for a given
# asset bundle are recorded alongside the bundle, not in Git.
_env = os.environ.get
FORMER_PROJECTS = Path(_env("SKILLRL_FORMER_PROJECTS", "/nonexistent/former/Projects"))
FORMER_HANDOVER = Path(_env("SKILLRL_FORMER_HANDOVER", "/nonexistent/former/skillRL_handover"))
FORMER_MODELS = Path(_env("SKILLRL_FORMER_MODELS", "/nonexistent/former/LLMWeights"))
TEXT_SUFFIXES = {
    ".py", ".sh", ".md", ".toml", ".yaml", ".yml", ".json", ".jsonl",
    ".txt", ".tsv", ".csv", ".env", ".ini", ".cfg",
}
SKIP_PARTS = {".git", "__pycache__", "wandb"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--from-root", default="")
    return p.parse_args()


def candidate_files(needles: list[str]):
    rg = shutil.which("rg")
    if rg:
        command = [rg, "-l", "-0", "--hidden", "--no-ignore", "--text", "--fixed-strings"]
        for needle in needles:
            command.extend(["-e", needle])
        for suffix in sorted(TEXT_SUFFIXES):
            command.extend(["-g", f"*{suffix}"])
        command.extend(["-g", "skillrl", "-g", ".skillrl-root", str(ROOT)])
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode not in (0, 1):
            raise RuntimeError(proc.stderr.decode(errors="replace"))
        for raw in proc.stdout.split(b"\0"):
            if raw:
                yield Path(os.fsdecode(raw))
        return
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"skillrl", ".skillrl-root"}:
            yield path


def rewrite_file(path: Path, replacements: list[tuple[bytes, bytes]]) -> bool:
    try:
        data = path.read_bytes()
    except OSError:
        return False
    updated = data
    for before, after in replacements:
        updated = updated.replace(before, after)
    if updated == data:
        return False
    path.write_bytes(updated)
    return True


def rewrite_symlinks(old_roots: list[Path]) -> tuple[int, int]:
    changed = 0
    unresolved = 0
    for path in ROOT.rglob("*"):
        if not path.is_symlink():
            continue
        raw = os.readlink(path)
        if not os.path.isabs(raw):
            continue
        target = Path(raw)
        # The CUDA fast-home shim is intentionally an environment link, not a
        # checkout dependency. It is rebuilt/validated on each worker.
        if "/cache/cuda_fast_home/" in path.as_posix() and str(target).startswith("/usr/local/cuda/"):
            continue
        replacement: Path | None = None
        for old in old_roots:
            try:
                replacement = ROOT / target.relative_to(old)
                break
            except ValueError:
                pass
        if replacement is None:
            unresolved += 1
            continue
        relative = os.path.relpath(replacement, start=path.parent)
        path.unlink()
        path.symlink_to(relative)
        changed += 1
    return changed, unresolved


def replace_object(value, replacements: list[tuple[str, str]]):
    if isinstance(value, str):
        for before, after in replacements:
            value = value.replace(before, after)
        return value
    if isinstance(value, list):
        return [replace_object(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(replace_object(item, replacements) for item in value)
    if isinstance(value, dict):
        return {
            replace_object(key, replacements): replace_object(item, replacements)
            for key, item in value.items()
        }
    # Embedding arrays, BM25 objects, stemmers, and numeric values are kept by
    # identity; only the companion skill_paths strings need relocation.
    return value


def rewrite_pickles(replacements: list[tuple[str, str]]) -> tuple[int, int]:
    changed = 0
    unresolved = 0
    paths = [
        ROOT / "GeneralAgent/eval_scripts/skills_retrieval/skill_index_qwen3emb8b.pkl",
        ROOT / "GeneralAgent/eval_scripts/skills_retrieval/skill_index_bm25.pkl",
    ]
    for path in paths:
        if not path.exists():
            continue
        try:
            with path.open("rb") as fh:
                payload = pickle.load(fh)
        except ModuleNotFoundError as exc:
            print(f"WARN: cannot relocate {path.relative_to(ROOT)} without dependency: {exc}")
            unresolved += 1
            continue
        updated = replace_object(payload, replacements)
        old_paths = payload.get("skill_paths", []) if isinstance(payload, dict) else []
        new_paths = updated.get("skill_paths", []) if isinstance(updated, dict) else []
        if old_paths == new_paths:
            continue
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(updated, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        changed += 1
    return changed, unresolved


def parquet_candidates(needles: list[str]):
    rg = shutil.which("rg")
    if not rg:
        return []
    command = [
        rg, "-l", "-0", "--hidden", "--no-ignore", "--text", "--fixed-strings",
    ]
    for needle in needles:
        command.extend(["-e", needle])
    command.extend(["-g", "*.parquet", "-g", "!.validation/**", str(ROOT)])
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode not in (0, 1):
        raise RuntimeError(proc.stderr.decode(errors="replace"))
    return [Path(os.fsdecode(raw)) for raw in proc.stdout.split(b"\0") if raw]


def rewrite_parquets(replacements: list[tuple[str, str]]) -> tuple[int, int]:
    paths = parquet_candidates([before for before, _ in replacements])
    if not paths:
        return 0, 0
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        print(f"WARN: cannot relocate parquet prompts without pyarrow: {exc}")
        return 0, len(paths)
    changed = 0
    for path in paths:
        table = pq.read_table(path)
        rows = table.to_pylist()
        updated_rows = replace_object(rows, replacements)
        if updated_rows == rows:
            continue
        updated = pa.Table.from_pylist(updated_rows, schema=table.schema)
        tmp = path.with_name(path.name + ".relocate.tmp")
        pq.write_table(updated, tmp, compression="snappy")
        check = pq.ParquetFile(tmp).metadata
        if check.num_rows != table.num_rows or pq.read_schema(tmp) != table.schema:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"parquet relocation shape mismatch: {path}")
        tmp.replace(path)
        changed += 1
    return changed, 0


_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}
_SKILL_FINGERPRINT_CACHE: dict[tuple[tuple[str, int, int], ...], str] = {}


def file_sha256(path: Path) -> str:
    stat = path.stat()
    cache_key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    cached = _FILE_HASH_CACHE.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[cache_key] = value
    return value


def rooted_path(raw_path: str | os.PathLike[str]) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def manifest_skill_content_sha256(manifest_paths: list[Path]) -> str:
    """Match the immutable RL builders' path-and-content skill fingerprint."""
    cache_key = tuple(
        (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)
        for path in manifest_paths
    )
    cached = _SKILL_FINGERPRINT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    skill_files: set[Path] = set()
    for manifest_path in manifest_paths:
        with manifest_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                for category in ("oracle", "misleading", "relevant", "irrelevant"):
                    for entry in row.get(category) or []:
                        skill_files.add(
                            (rooted_path(str(entry["path"])) / "SKILL.md").resolve()
                        )
    digest = hashlib.sha256()
    for skill_file in sorted(skill_files, key=str):
        digest.update(str(skill_file).encode() + b"\0")
        digest.update(bytes.fromhex(file_sha256(skill_file)))
    value = digest.hexdigest()
    _SKILL_FINGERPRINT_CACHE[cache_key] = value
    return value


def refresh_rl_build_reports() -> tuple[int, int]:
    """Refresh path-sensitive frozen hashes after an intentional relocation.

    The actual train/eval parquet content is never rewritten here.  Existing
    report fields are recomputed from their declared inputs, manifests, skill
    files, and outputs.  Multiple passes settle downstream reports that hash an
    upstream ``build_report.json``.
    """
    report_paths = sorted((ROOT / "datasets/rl").glob("*/build_report.json"))
    changed_paths: set[Path] = set()
    unresolved: set[str] = set()

    for _ in range(4):
        pass_changed = False
        for report_path in report_paths:
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                fingerprints = report.get("fingerprints")
                if not isinstance(fingerprints, dict):
                    continue
                updated = dict(fingerprints)
                input_dir = rooted_path(str(report["input_dir"]))
                direct_inputs = {
                    "input_train_sha256": input_dir / "train.parquet",
                    "input_eval_sha256": input_dir / "eval.parquet",
                    "input_build_report_sha256": input_dir / "build_report.json",
                }
                for key, source in direct_inputs.items():
                    if key in updated:
                        updated[key] = file_sha256(source)

                train_manifest_raw = report.get("train_manifest") or report.get("manifest")
                eval_manifest_raw = report.get("eval_manifest")
                manifest_paths: list[Path] = []
                if train_manifest_raw:
                    train_manifest = rooted_path(str(train_manifest_raw))
                    manifest_paths.append(train_manifest)
                    if "train_manifest_sha256" in updated:
                        updated["train_manifest_sha256"] = file_sha256(train_manifest)
                if eval_manifest_raw:
                    eval_manifest = rooted_path(str(eval_manifest_raw))
                    manifest_paths.append(eval_manifest)
                    if "eval_manifest_sha256" in updated:
                        updated["eval_manifest_sha256"] = file_sha256(eval_manifest)
                if "skill_content_sha256" in updated:
                    if not manifest_paths:
                        raise FileNotFoundError("skill fingerprint has no declared manifest")
                    updated["skill_content_sha256"] = manifest_skill_content_sha256(
                        manifest_paths
                    )
                report["fingerprints"] = updated

                output_fingerprints = report.get("output_fingerprints")
                if isinstance(output_fingerprints, dict):
                    refreshed_outputs = dict(output_fingerprints)
                    if "train_parquet_sha256" in refreshed_outputs:
                        refreshed_outputs["train_parquet_sha256"] = file_sha256(
                            report_path.parent / "train.parquet"
                        )
                    if "eval_parquet_sha256" in refreshed_outputs:
                        refreshed_outputs["eval_parquet_sha256"] = file_sha256(
                            report_path.parent / "eval.parquet"
                        )
                    report["output_fingerprints"] = refreshed_outputs

                original = json.loads(report_path.read_text(encoding="utf-8"))
                if report != original:
                    report_path.write_text(
                        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    changed_paths.add(report_path)
                    pass_changed = True
            except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
                unresolved.add(f"{report_path.relative_to(ROOT)}: {exc}")
        if not pass_changed:
            break

    for problem in sorted(unresolved):
        print(f"WARN: could not refresh RL build report: {problem}")
    return len(changed_paths), len(unresolved)


def main() -> int:
    args = parse_args()
    old_roots: list[Path] = [FORMER_HANDOVER, FORMER_PROJECTS]
    if args.from_root:
        old_roots.insert(0, Path(args.from_root).resolve())
    elif STAMP.exists():
        stamped = Path(STAMP.read_text().strip()).resolve()
        if stamped != ROOT:
            old_roots.insert(0, stamped)

    string_replacements = [(str(old), str(ROOT)) for old in old_roots if old != ROOT]
    string_replacements.append((str(FORMER_MODELS), str(ROOT / "models")))
    byte_replacements = [(before.encode(), after.encode()) for before, after in string_replacements]
    changed = sum(
        rewrite_file(path, byte_replacements)
        for path in candidate_files([before for before, _ in string_replacements])
    )
    symlinks, unresolved = rewrite_symlinks(old_roots)
    pickles, unresolved_pickles = rewrite_pickles(string_replacements)
    parquets, unresolved_parquets = rewrite_parquets(string_replacements)
    rl_reports, unresolved_rl_reports = refresh_rl_build_reports()
    STAMP.write_text(str(ROOT) + "\n")
    print(
        f"RELOCATE_OK root={ROOT} files={changed} symlinks={symlinks} "
        f"pickles={pickles} parquets={parquets} rl_build_reports={rl_reports} "
        f"unresolved_absolute_symlinks={unresolved} "
        f"unresolved_pickles={unresolved_pickles} unresolved_parquets={unresolved_parquets} "
        f"unresolved_rl_build_reports={unresolved_rl_reports}"
    )
    return 1 if (
        unresolved
        or unresolved_pickles
        or unresolved_parquets
        or unresolved_rl_reports
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
