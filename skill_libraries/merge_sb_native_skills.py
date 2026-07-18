#!/usr/bin/env python3
"""
Merge SkillsBench native skills (datasets/skillsbench/tasks/*/environment/skills/)
into skill_libraries/merged/.

Strategy (Plan E):
  For each unique (skill_name, content_manifest_hash) seen across all SB task
  environments:
    1. If skill_name does NOT exist in merged/  → add as `<skill_name>`
    2. If it exists AND the SB version is >=90% similar to merged version
       → skip (the merged version is good enough)
    3. Otherwise → add as `sb-<skill_name>-v<hash6>` to preserve the SB variant

Produces:
  - New dirs under skill_libraries/merged/
  - A manifest file skill_libraries/merged_sb_manifest.json for audit
"""
import os
import hashlib
import json
import shutil
import difflib
from pathlib import Path
from collections import defaultdict

PROJ = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))
SB_TASKS = PROJ / "datasets/skillsbench/tasks"
MERGED = PROJ / "skill_libraries/merged"
MANIFEST = PROJ / "skill_libraries/merged_sb_manifest.json"

SIM_SKIP_THRESHOLD = 0.90  # if SB SKILL.md similar enough to merged, skip


def manifest_hash(skill_dir: Path) -> tuple[str, dict]:
    """MD5 over (relpath, file_md5) for every .md file, sorted."""
    files = {}
    for f in sorted(skill_dir.rglob("*.md")):
        rel = str(f.relative_to(skill_dir))
        h = hashlib.md5(f.read_bytes()).hexdigest()[:8]
        files[rel] = h
    summary = "|".join(f"{k}:{v}" for k, v in sorted(files.items()))
    mhash = hashlib.md5(summary.encode()).hexdigest()[:8]
    return mhash, files


def skill_md_similarity(dir_a: Path, dir_b: Path) -> float:
    """Rough similarity of main SKILL.md only."""
    a = dir_a / "SKILL.md"
    b = dir_b / "SKILL.md"
    if not a.exists() or not b.exists():
        return 0.0
    return difflib.SequenceMatcher(
        None, a.read_text(errors="replace"), b.read_text(errors="replace")
    ).ratio()


def main():
    # 1. Scan SB skills: (skill_name) -> {manifest_hash: (skill_dir, used_by_tasks)}
    sb_versions = defaultdict(dict)
    for tdir in sorted(SB_TASKS.iterdir()):
        if not tdir.is_dir() or tdir.name.startswith("."):
            continue
        sdir = tdir / "environment" / "skills"
        if not sdir.is_dir():
            continue
        for skill in sdir.iterdir():
            if not skill.is_dir():
                continue
            mh, _ = manifest_hash(skill)
            entry = sb_versions[skill.name].setdefault(mh, {"dir": skill, "tasks": []})
            entry["tasks"].append(tdir.name)

    # 2. For each skill_name + version, decide name
    actions = []  # list of dict: {sb_name, version_hash, sb_dir, action, new_name, used_tasks, note}
    for skill_name, versions in sb_versions.items():
        exists_in_merged = (MERGED / skill_name).is_dir()
        merged_dir = MERGED / skill_name if exists_in_merged else None

        for vhash, ent in versions.items():
            sb_dir = ent["dir"]
            tasks = ent["tasks"]
            if not exists_in_merged:
                actions.append({
                    "sb_name": skill_name, "version_hash": vhash,
                    "sb_dir": str(sb_dir), "action": "add_original_name",
                    "new_name": skill_name, "used_tasks": tasks,
                    "note": "skill not in merged; using original name",
                })
            else:
                sim = skill_md_similarity(merged_dir, sb_dir)
                if sim >= SIM_SKIP_THRESHOLD:
                    actions.append({
                        "sb_name": skill_name, "version_hash": vhash,
                        "sb_dir": str(sb_dir), "action": "skip_similar",
                        "new_name": None, "used_tasks": tasks,
                        "note": f"SKILL.md similarity {sim:.0%} >= {SIM_SKIP_THRESHOLD:.0%}, merged is sufficient",
                    })
                else:
                    new_name = f"sb-{skill_name}-v{vhash}"
                    actions.append({
                        "sb_name": skill_name, "version_hash": vhash,
                        "sb_dir": str(sb_dir), "action": "add_renamed",
                        "new_name": new_name, "used_tasks": tasks,
                        "note": f"SKILL.md similarity {sim:.0%} < {SIM_SKIP_THRESHOLD:.0%}, preserving SB variant",
                    })

    # 3. Dedupe: if same (new_name) has multiple original versions (rare), prefix with hash
    name_count = defaultdict(list)
    for a in actions:
        if a["action"] in ("add_original_name", "add_renamed"):
            name_count[a["new_name"]].append(a)
    for new_name, entries in name_count.items():
        if len(entries) > 1:
            # Add version hash suffix to distinguish
            for e in entries:
                e["new_name"] = f"{new_name}-v{e['version_hash']}"
                e["note"] += " [name collision → added -v<hash> suffix]"

    # 4. Execute
    added = 0
    skipped = 0
    name_collisions = 0
    for a in actions:
        if a["action"] == "skip_similar":
            skipped += 1
            continue
        dst = MERGED / a["new_name"]
        if dst.exists():
            a["note"] += " [dst already exists in merged, skipping copy]"
            name_collisions += 1
            continue
        shutil.copytree(a["sb_dir"], dst)
        added += 1

    # 5. Write manifest
    summary = {
        "total_sb_skill_versions_found": len(actions),
        "added": added,
        "skipped_similar": skipped,
        "name_collisions": name_collisions,
        "final_skill_count": len(list(MERGED.iterdir())),
        "actions": actions,
    }
    MANIFEST.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # Print summary
    print(f"SB skill versions scanned: {len(actions)}")
    print(f"  → added (new name):      {added}")
    print(f"  → skipped (>=90% sim):   {skipped}")
    print(f"  → name collision skip:   {name_collisions}")
    print(f"Final merged/ skill count: {summary['final_skill_count']}")
    print(f"Manifest written to:       {MANIFEST}")

    # Show samples
    print("\nSample added:")
    for a in [x for x in actions if x["action"] != "skip_similar"][:10]:
        print(f"  [{a['action']}] {a['sb_name']:<30} → {a['new_name']}  (used by {len(a['used_tasks'])} task)")
    print("\nSample skipped:")
    for a in [x for x in actions if x["action"] == "skip_similar"][:5]:
        print(f"  [skip] {a['sb_name']:<30} — {a['note']}")


if __name__ == "__main__":
    main()
