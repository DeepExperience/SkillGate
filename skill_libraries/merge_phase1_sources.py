#!/usr/bin/env python3
"""Phase 1 merge: 7 cloned repos → skill_libraries/merged/.

Same incremental policy as merge_tier1_sources.py:
  - skill_name not in merged → add as-is
  - skill_name exists, similarity >=0.90 → skip (existing is sufficient)
  - else → add as <prefix>-<slug> with hash suffix on collision

These are external-source clones, not handwritten — no `hw-` prefix.
"""
import os
import hashlib
import json
import shutil
import difflib
from pathlib import Path
from collections import defaultdict

PROJ = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))
SRC_DIR = PROJ / "skill_libraries/_sources_v3"
MERGED = PROJ / "skill_libraries/merged"
MANIFEST = PROJ / "skill_libraries/merged_phase1_manifest.json"
SIM_SKIP = 0.90

REPO_PREFIX = {
    "skills":                                       "octa",  # OctagonAI
    "jira":                                         "spw-jira",
    "mastering-confluence-agent-skill":             "spw-conf",
    "hubspot-admin-skills":                         "hub",
    "sf-skills":                                    "sf",
    "Claude-Skills-Governance-Risk-and-Compliance": "grc",
    "ctf-skills":                                   "ctf",
}


def manifest_hash(d: Path):
    files = {}
    for f in sorted(d.rglob("*.md")):
        try:
            h = hashlib.md5(f.read_bytes()).hexdigest()[:8]
        except Exception:
            continue
        files[str(f.relative_to(d))] = h
    blob = "|".join(f"{k}:{v}" for k, v in sorted(files.items()))
    return hashlib.md5(blob.encode()).hexdigest()[:8]


def sim(a: Path, b: Path) -> float:
    fa = a / "SKILL.md"; fb = b / "SKILL.md"
    if not fa.exists() or not fb.exists(): return 0.0
    try:
        return difflib.SequenceMatcher(None,
            fa.read_text(errors="replace"),
            fb.read_text(errors="replace")).ratio()
    except Exception:
        return 0.0


def find_skills(repo_dir: Path):
    out = []
    for skill_md in repo_dir.rglob("SKILL.md"):
        sd = skill_md.parent
        if any(p.name in (".git", ".github", "__pycache__", "node_modules") for p in sd.parents):
            continue
        out.append((sd.name, sd))
    return out


def main():
    actions = []
    added = skipped = collisions = 0
    for repo_dir in sorted(SRC_DIR.iterdir()):
        if not repo_dir.is_dir(): continue
        repo = repo_dir.name
        prefix = REPO_PREFIX.get(repo, repo.lower()[:6])
        skills = find_skills(repo_dir)
        print(f"\n=== {repo} ({len(skills)} skills, prefix={prefix}) ===")

        for slug, sdir in skills:
            mh = manifest_hash(sdir)
            existing = MERGED / slug
            if existing.is_dir():
                s = sim(existing, sdir)
                if s >= SIM_SKIP:
                    actions.append({"repo": repo, "slug": slug, "action": "skip_similar",
                                    "sim": round(s,2)})
                    skipped += 1
                    continue
                new_name = f"{prefix}-{slug}"
                if (MERGED / new_name).exists():
                    new_name = f"{prefix}-{slug}-v{mh}"
                    if (MERGED / new_name).exists():
                        actions.append({"repo": repo, "slug": slug, "action": "collision_skip",
                                        "new_name": new_name})
                        collisions += 1
                        continue
            else:
                new_name = slug
                if (MERGED / new_name).exists():
                    new_name = f"{prefix}-{slug}"
                    if (MERGED / new_name).exists():
                        new_name = f"{prefix}-{slug}-v{mh}"

            try:
                shutil.copytree(sdir, MERGED / new_name)
                added += 1
                actions.append({"repo": repo, "slug": slug, "action": "add",
                                "new_name": new_name,
                                "renamed": (new_name != slug)})
            except Exception as e:
                actions.append({"repo": repo, "slug": slug, "action": "error", "error": str(e)})

    summary = {
        "total_scanned": len(actions),
        "added": added,
        "skipped_similar": skipped,
        "collisions": collisions,
        "final_merged_count": len(list(MERGED.iterdir())),
        "actions": actions,
    }
    MANIFEST.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n{'='*60}")
    print(f"  added: {added}")
    print(f"  skipped (sim≥{SIM_SKIP}): {skipped}")
    print(f"  collisions: {collisions}")
    print(f"  final merged/ count: {summary['final_merged_count']}")
    print(f"  manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
