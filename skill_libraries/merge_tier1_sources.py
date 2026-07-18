#!/usr/bin/env python3
"""
Incremental merge: add skills from _sources_v2/<repo>/ into merged/.

Strategy:
  For each SKILL.md in source repos:
    - name not in merged → add with original slug
    - name exists, manifest hash == existing → skip (same content)
    - name exists, similar SKILL.md (>0.90) → skip
    - otherwise → add as <repo_prefix>-<slug>

Writes audit manifest to merged_tier1_manifest.json.
"""
import os
import hashlib
import json
import shutil
import difflib
from pathlib import Path
from collections import defaultdict

PROJ = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))
SRC_DIR = PROJ / "skill_libraries/_sources_v2"
MERGED = PROJ / "skill_libraries/merged"
MANIFEST = PROJ / "skill_libraries/merged_tier1_manifest.json"
SIM_SKIP = 0.90

REPO_PREFIX = {
    "bioSkills": "bio",
    "SciAgent-Skills": "sci",
    "dev-skills": "dev",
    "superpowers": "sp",
}


def manifest_hash(skill_dir: Path):
    files = {}
    for f in sorted(skill_dir.rglob("*.md")):
        try:
            h = hashlib.md5(f.read_bytes()).hexdigest()[:8]
        except Exception:
            continue
        files[str(f.relative_to(skill_dir))] = h
    blob = "|".join(f"{k}:{v}" for k, v in sorted(files.items()))
    return hashlib.md5(blob.encode()).hexdigest()[:8]


def skill_md_sim(a: Path, b: Path) -> float:
    fa = a / "SKILL.md"; fb = b / "SKILL.md"
    if not fa.exists() or not fb.exists(): return 0.0
    try:
        return difflib.SequenceMatcher(None,
            fa.read_text(errors="replace"),
            fb.read_text(errors="replace")).ratio()
    except Exception:
        return 0.0


def find_skills(repo_dir: Path):
    skills = []
    for skill_md in repo_dir.rglob("SKILL.md"):
        sdir = skill_md.parent
        if any(p.name in (".git", ".github", "__pycache__", "node_modules") for p in sdir.parents):
            continue
        skills.append((sdir.name, sdir))
    return skills


def main():
    actions = []
    added = skipped_similar = skipped_exists = collisions = 0

    for repo_dir in sorted(SRC_DIR.iterdir()):
        if not repo_dir.is_dir():
            continue
        repo = repo_dir.name
        prefix = REPO_PREFIX.get(repo, repo.lower())
        skills = find_skills(repo_dir)
        print(f"\n=== {repo} ({len(skills)} skills) ===")

        for slug, sdir in skills:
            mh = manifest_hash(sdir)
            existing = MERGED / slug
            if existing.is_dir():
                sim = skill_md_sim(existing, sdir)
                if sim >= SIM_SKIP:
                    actions.append({"repo": repo, "slug": slug, "action": "skip_similar",
                                    "sim": round(sim,2), "new_name": None})
                    skipped_similar += 1
                    continue
                # Different → use prefix
                new_name = f"{prefix}-{slug}"
                if (MERGED / new_name).exists():
                    new_name = f"{prefix}-{slug}-v{mh}"
                    if (MERGED / new_name).exists():
                        actions.append({"repo": repo, "slug": slug, "action": "collision_skip",
                                        "new_name": new_name, "note": "3-way collision"})
                        collisions += 1
                        continue
            else:
                new_name = slug
                if (MERGED / new_name).exists():
                    # same name from 2 sources in this run
                    new_name = f"{prefix}-{slug}"
                    if (MERGED / new_name).exists():
                        new_name = f"{prefix}-{slug}-v{mh}"

            try:
                shutil.copytree(sdir, MERGED / new_name)
                added += 1
                actions.append({"repo": repo, "slug": slug, "action": "add",
                                "new_name": new_name,
                                "used_prefix": (new_name != slug)})
            except Exception as e:
                actions.append({"repo": repo, "slug": slug, "action": "error",
                                "error": str(e)})

    summary = {
        "total_scanned": len(actions),
        "added": added,
        "skipped_similar": skipped_similar,
        "skipped_exists": skipped_exists,
        "collisions": collisions,
        "final_merged_count": len(list(MERGED.iterdir())),
        "actions": actions,
    }
    MANIFEST.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n{'='*60}")
    print(f"TOTAL: {len(actions)} skills scanned")
    print(f"  added: {added}")
    print(f"  skipped (>={SIM_SKIP*100:.0f}% similar to existing): {skipped_similar}")
    print(f"  collisions: {collisions}")
    print(f"Final merged/ count: {summary['final_merged_count']}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
