#!/usr/bin/env python3
"""Materialize a self-contained, content-verified skill snapshot.

The retrieval JSONLs used by eval contain absolute ``skill_path`` pointers.
Copy every referenced directory into an immutable eval input root and rewrite
the JSONLs so cross-day retries cannot silently read changed skill content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


PARTS = ("claw", "sb_ns", "seta_synth", "swe_lite", "tb2")


def tree_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            digest.update(b"L\0" + rel.encode() + b"\0" + os.readlink(path).encode() + b"\0")
            continue
        if not path.is_file():
            continue
        digest.update(b"F\0" + rel.encode() + b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                total_bytes += len(chunk)
        digest.update(b"\0")
        files += 1
    return digest.hexdigest(), files, total_bytes


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def verify(output_root: Path, expected_preserve_skill_names: bool | None = None) -> None:
    mapping_path = output_root / "source_to_frozen.jsonl"
    complete = output_root / "COMPLETE"
    if not complete.is_file() or not mapping_path.is_file():
        raise RuntimeError(f"incomplete frozen snapshot: {output_root}")
    try:
        complete_record = json.loads(complete.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid frozen snapshot completion marker: {complete}") from exc
    preserve_skill_names = bool(complete_record.get("preserve_skill_names", False))
    if expected_preserve_skill_names is not None and preserve_skill_names != expected_preserve_skill_names:
        raise RuntimeError(
            f"frozen snapshot mode mismatch: preserve_skill_names={preserve_skill_names}, "
            f"requested={expected_preserve_skill_names}"
        )
    mappings = read_jsonl(mapping_path)
    if not mappings:
        raise RuntimeError(f"empty mapping: {mapping_path}")
    for row in mappings:
        frozen = Path(row["frozen_path"])
        if not frozen.is_dir():
            raise RuntimeError(f"missing frozen skill directory: {frozen}")
        if preserve_skill_names and frozen.name != row.get("logical_name"):
            raise RuntimeError(f"frozen logical name mismatch: {frozen.name} != {row.get('logical_name')}")
        digest, files, total_bytes = tree_digest(frozen)
        expected = (row["tree_sha256"], int(row["files"]), int(row["bytes"]))
        if (digest, files, total_bytes) != expected:
            raise RuntimeError(f"frozen skill content mismatch: {frozen}")
    for part in PARTS:
        path = output_root / "snapshot_eval70" / f"{part}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"missing frozen snapshot part: {path}")
        for row in read_jsonl(path):
            for item in row.get("reranked_top10", []):
                skill_path = Path(item["skill_path"])
                if not skill_path.is_dir() or output_root not in skill_path.parents:
                    raise RuntimeError(f"snapshot contains non-frozen skill path: {skill_path}")
    print(f"[deep-freeze-ok] root={output_root} skill_dirs={len(mappings)}")


def materialize(
    source_snapshot: Path,
    source_manifest: Path,
    output_root: Path,
    *,
    preserve_skill_names: bool = False,
) -> None:
    if (output_root / "COMPLETE").is_file():
        verify(output_root, expected_preserve_skill_names=preserve_skill_names)
        return
    if output_root.exists():
        raise RuntimeError(
            f"partial output exists: {output_root}; remove only this incomplete eval-owned directory before retry"
        )

    temp_root = output_root.with_name(f"{output_root.name}.tmp-{os.getpid()}")
    if temp_root.exists():
        shutil.rmtree(temp_root)
    snapshot_out = temp_root / "snapshot_eval70"
    skills_out = temp_root / "skills"
    snapshot_out.mkdir(parents=True)
    skills_out.mkdir()

    frozen_entries: dict[tuple[str, str], dict[str, str]] = {}
    name_to_source: dict[str, str] = {}
    rewritten: dict[str, list[dict]] = {}
    try:
        for part in PARTS:
            source_part = source_snapshot / f"{part}.jsonl"
            rows = read_jsonl(source_part)
            rewritten[part] = rows
            for row in rows:
                for item in row.get("reranked_top10", []):
                    source = Path(item["skill_path"]).resolve()
                    if not source.is_dir():
                        raise RuntimeError(f"missing source skill directory: {source}")
                    logical_name = str(item.get("skill_name") or source.name).strip()
                    if not logical_name or Path(logical_name).name != logical_name:
                        raise RuntimeError(f"invalid logical skill name {logical_name!r} for {source}")
                    source_key = str(source)
                    name = logical_name if preserve_skill_names else (
                        f"{hashlib.sha256(source_key.encode()).hexdigest()[:16]}__{source.name}"
                    )
                    previous_source = name_to_source.get(name)
                    if previous_source is not None and previous_source != source_key:
                        raise RuntimeError(
                            f"frozen skill-name collision: {name!r} maps to {previous_source} and {source_key}"
                        )
                    name_to_source[name] = source_key
                    key = (source_key, name)
                    if key not in frozen_entries:
                        shutil.copytree(source, skills_out / name, symlinks=True)
                        frozen_entries[key] = {
                            "source_path": source_key,
                            "frozen_name": name,
                            "logical_name": logical_name,
                        }
                    item["skill_path"] = str((output_root / "skills" / name).resolve())

        mappings: list[dict] = []
        for entry in sorted(frozen_entries.values(), key=lambda item: (item["frozen_name"], item["source_path"])):
            source = entry["source_path"]
            name = entry["frozen_name"]
            temp_frozen = skills_out / name
            final_frozen = (output_root / "skills" / name).resolve()
            digest, files, total_bytes = tree_digest(temp_frozen)
            mappings.append(
                {
                    "source_path": source,
                    "frozen_path": str(final_frozen),
                    "logical_name": entry["logical_name"],
                    "tree_sha256": digest,
                    "files": files,
                    "bytes": total_bytes,
                }
            )

        for part, rows in rewritten.items():
            with (snapshot_out / f"{part}.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        shutil.copy2(source_manifest, temp_root / "slate_manifest_eval70.jsonl")
        with (temp_root / "source_to_frozen.jsonl").open("w", encoding="utf-8") as handle:
            for row in mappings:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        (temp_root / "COMPLETE").write_text(
            json.dumps(
                    {
                        "skill_dirs": len(mappings),
                        "files": sum(row["files"] for row in mappings),
                        "bytes": sum(row["bytes"] for row in mappings),
                        "preserve_skill_names": preserve_skill_names,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temp_root.rename(output_root)
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    verify(output_root, expected_preserve_skill_names=preserve_skill_names)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--preserve-skill-names",
        action="store_true",
        help="freeze each skill under its advertised skill_name so selector prompts remain unchanged",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if args.verify_only:
        verify(output_root, expected_preserve_skill_names=True if args.preserve_skill_names else None)
    else:
        materialize(
            args.source_snapshot.resolve(),
            args.source_manifest.resolve(),
            output_root,
            preserve_skill_names=args.preserve_skill_names,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise
