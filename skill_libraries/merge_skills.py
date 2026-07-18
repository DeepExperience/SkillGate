#!/usr/bin/env python3
"""
Merge skills from multiple skill library repos into a single flat directory.

Each skill becomes a folder named by its skill slug, containing SKILL.md and
any supporting files (scripts/, references/, assets/, etc.).

Conflict resolution (2026-04-23 update):
  - When the same skill name exists in multiple repos, ask DeepSeek-v3.2
    (MAAS) whether each pair describes the SAME skill concept. Skills that the
    LLM judges "same" are merged (longest SKILL.md wins). Skills judged
    "different" are kept separately under `<name>__<repo>` suffix names.
  - Disable LLM with `--no-llm-judge` → fall back to longest-wins merge.
  - Pairwise equivalence results are cached in `_merge_llm_cache.json`.

Usage:
    python merge_skills.py                        # merge all repos + LLM judge
    python merge_skills.py --repos a b            # subset of repos
    python merge_skills.py --dry-run              # preview
    python merge_skills.py --no-llm-judge         # old behavior (longest wins)
    python merge_skills.py --llm-concurrency 16   # parallel LLM calls
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "merged"
LLM_CACHE_PATH = SCRIPT_DIR / "_merge_llm_cache.json"

# Directories / patterns to skip when scanning for skills
SKIP_DIRS = {".git", ".github", "__pycache__", "node_modules",
             ".venv", "pending", "template", "spec", "packages"}

# --- LLM judge ---
LLM_MODEL = "deepseek-v3.2"
LLM_SYSTEM = (
    "You compare two skill cards that share the same short name and judge "
    "whether they describe the SAME skill concept (same tool, same workflow, "
    "same domain). They are considered 'same' if reading both would be "
    "redundant; 'different' if they're about genuinely different things that "
    "happen to share a name. Output STRICT JSON only."
)
LLM_USER_TMPL = """Same-name skill conflict — are these the SAME skill concept?

Skill A  (source: {repo_a})
name: {name_a}
---
{body_a}
---

Skill B  (source: {repo_b})
name: {name_b}
---
{body_b}
---

Output STRICT JSON (no preamble, no markdown fence):
{{"same": <true|false>, "reason": "<one concise sentence>"}}"""


# ---------------------------------------------------------------------------
# Skill discovery (unchanged)
# ---------------------------------------------------------------------------

def find_skills(repo_dir: Path) -> list[dict]:
    repo_name = repo_dir.name
    skills = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        if "SKILL.md" not in files:
            continue
        skill_path = Path(root)
        skill_md = skill_path / "SKILL.md"
        depth = len(skill_path.relative_to(repo_dir).parts)
        if skill_path == repo_dir:
            continue
        skills.append({
            "name": skill_path.name,
            "path": skill_path,
            "repo": repo_name,
            "skill_md_size": skill_md.stat().st_size,
            "depth": depth,
            "rel_path": str(skill_path.relative_to(repo_dir)),
        })
    return skills


def is_sub_skill(skill: dict, all_skills_in_repo: list[dict]) -> bool:
    parent_skill_paths = {s["path"] for s in all_skills_in_repo if s is not skill}
    for ancestor in skill["path"].parents:
        if ancestor in parent_skill_paths:
            return True
    return False


def read_skill_body(path: Path, max_chars: int = 2500) -> str:
    """SKILL.md head — keep frontmatter + description; truncate long body."""
    md = path / "SKILL.md"
    if not md.exists():
        return ""
    try:
        txt = md.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return txt[:max_chars]


# ---------------------------------------------------------------------------
# LLM judge (DeepSeek via MAAS)
# ---------------------------------------------------------------------------

class MergeJudge:
    def __init__(self, model: str, max_parallel: int):
        self.model = model
        self.max_parallel = max_parallel
        self.api_base = os.environ.get("MAAS_API_BASE") or os.environ.get("DEEPSEEK_BASE_URL")
        self.api_key  = os.environ.get("MAAS_API_KEY")  or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_base or not self.api_key:
            raise RuntimeError(
                "MAAS_API_BASE / MAAS_API_KEY not set. Source secrets/.env.secrets "
                "first (it has MAAS keys) OR pass --no-llm-judge for old behavior."
            )
        import requests
        self.sess = requests.Session()
        self.sess.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        # cache
        self.cache: dict = {}
        if LLM_CACHE_PATH.exists():
            try:
                self.cache = json.loads(LLM_CACHE_PATH.read_text())
            except Exception:
                self.cache = {}
        self._cache_lock_misses = 0

    def _cache_key(self, a: dict, b: dict) -> str:
        # canonical: name + sorted(repo|rel_path) pair — order-independent
        pair = sorted([f'{a["repo"]}|{a["rel_path"]}', f'{b["repo"]}|{b["rel_path"]}'])
        return f'{a["name"]}||{pair[0]}||{pair[1]}'

    def _score_pair(self, a: dict, b: dict) -> dict:
        """Return {'same': bool, 'reason': str, 'cached': bool}."""
        k = self._cache_key(a, b)
        if k in self.cache:
            return {**self.cache[k], "cached": True}

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM},
                {"role": "user", "content": LLM_USER_TMPL.format(
                    name_a=a["name"], repo_a=a["repo"], body_a=read_skill_body(a["path"]),
                    name_b=b["name"], repo_b=b["repo"], body_b=read_skill_body(b["path"]),
                )},
            ],
            "temperature": 0.0,
            "max_tokens": 200,
        }
        url = self.api_base.rstrip("/") + "/chat/completions"
        for attempt in range(3):
            try:
                r = self.sess.post(url, json=payload, timeout=60)
                r.raise_for_status()
                d = r.json()
                content = d["choices"][0]["message"]["content"].strip()
                m = re.search(r"```(?:json)?\s*(.+?)```", content, re.DOTALL)
                if m: content = m.group(1).strip()
                try:
                    parsed = json.loads(content)
                    same = bool(parsed.get("same", True))   # default: assume same on uncertainty
                    reason = str(parsed.get("reason", ""))[:250]
                except Exception:
                    # loose parse
                    lm = re.search(r'"same"\s*:\s*(true|false)', content, re.I)
                    if lm:
                        same = lm.group(1).lower() == "true"
                        reason = "(loose parse) " + content[:200]
                    else:
                        same = True   # fallback: dedup (safer for lib size)
                        reason = "(unparseable) " + content[:200]
                self.cache[k] = {"same": same, "reason": reason}
                return {**self.cache[k], "cached": False}
            except Exception as e:
                if attempt == 2:
                    self.cache[k] = {"same": True, "reason": f"(error: {e})"}
                    return {**self.cache[k], "cached": False}
                time.sleep(2 ** attempt)

    def cluster(self, candidates: list[dict]) -> list[list[dict]]:
        """Partition candidates into equivalence clusters using LLM judge.

        Uses incremental O(n*k) clustering: new candidate joins first cluster
        whose representative it matches; else opens its own cluster.
        """
        if len(candidates) <= 1:
            return [candidates]
        clusters = [[candidates[0]]]
        for c in candidates[1:]:
            placed = False
            for cl in clusters:
                verdict = self._score_pair(cl[0], c)
                if verdict["same"]:
                    cl.append(c)
                    placed = True
                    break
            if not placed:
                clusters.append([c])
        return clusters

    def cluster_many_parallel(self, conflict_groups: list[tuple[str, list[dict]]]
                              ) -> dict[str, list[list[dict]]]:
        """Cluster multiple same-name groups in parallel (speeds up 500+ conflicts)."""
        result: dict[str, list[list[dict]]] = {}
        # Each group is small (usually 2-5 candidates); do each on its own thread
        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            fut2name = {pool.submit(self.cluster, cands): name
                        for name, cands in conflict_groups}
            for fut in as_completed(fut2name):
                name = fut2name[fut]
                try:
                    result[name] = fut.result()
                except Exception as e:
                    # fallback: one cluster = keep longest
                    cands = dict(conflict_groups)[name]
                    print(f"  [judge ERR] {name}: {e} — fallback to single cluster")
                    result[name] = [cands]
        return result

    def persist_cache(self):
        LLM_CACHE_PATH.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------

def pick_best_in_cluster(cluster: list[dict]) -> dict:
    """Within a cluster of equivalent skills, pick longest SKILL.md (prefer top-level)."""
    top_level = [c for c in cluster if not c.get("is_sub_skill")]
    pool = top_level if top_level else cluster
    return max(pool, key=lambda c: c["skill_md_size"])


def resolve_conflicts(by_name: dict[str, list[dict]], judge: Optional[MergeJudge],
                      verbose: bool = False) -> dict[str, dict]:
    """Return {final_skill_name: chosen_skill_dict}.

    Naming rules:
      - Non-conflict (n=1): keep original name.
      - Conflict cluster 0 (largest, or first if tie): keep original name.
      - Conflict cluster 1+: rename to `<name>__<repo>` (append source repo).
    """
    selected: dict[str, dict] = {}

    # Separate conflicts vs non-conflicts
    conflicts = [(n, c) for n, c in by_name.items() if len(c) > 1]
    singletons = [(n, c) for n, c in by_name.items() if len(c) == 1]

    # Singletons go straight through
    for n, c in singletons:
        selected[n] = c[0]

    if not conflicts:
        return selected

    # Cluster conflicts
    if judge is not None:
        print(f"\n  Judging {len(conflicts)} conflicts via {judge.model} "
              f"(parallel={judge.max_parallel})...")
        t0 = time.time()
        clustered = judge.cluster_many_parallel(conflicts)
        judge.persist_cache()
        elapsed = time.time() - t0
        total_pairs = sum(len(c)*(len(c)+1)//2 for _, c in conflicts)
        cache_hits = sum(1 for k in judge.cache if k.startswith(tuple(n+"||" for n,_ in conflicts)))
        print(f"  Judge done in {elapsed:.0f}s  (cache size: {len(judge.cache)})")
    else:
        # No-LLM: treat whole conflict group as one cluster (old behavior)
        clustered = {n: [cands] for n, cands in conflicts}

    n_split = 0
    for name, clusters in clustered.items():
        if len(clusters) == 1:
            # Single equivalence class → normal merge
            winner = pick_best_in_cluster(clusters[0])
            selected[name] = winner
        else:
            # Multiple distinct skills sharing the name → keep all with suffix
            n_split += 1
            # Sort clusters by (size desc, max skill_md_size desc) — biggest/most-detailed first
            clusters.sort(key=lambda cl: (-len(cl), -max(c["skill_md_size"] for c in cl)))
            for i, cl in enumerate(clusters):
                winner = pick_best_in_cluster(cl)
                if i == 0:
                    # First cluster keeps the plain name
                    selected[name] = winner
                else:
                    # Subsequent clusters get suffixed names: <name>__<repo>
                    suffix = re.sub(r'[^A-Za-z0-9_-]+', '-', winner["repo"])
                    new_name = f"{name}__{suffix}"
                    # collision check (rare)
                    if new_name in selected:
                        new_name = f"{name}__{suffix}_{i}"
                    winner_copy = dict(winner)
                    winner_copy["final_name"] = new_name
                    selected[new_name] = winner_copy
            if verbose:
                reps = [(cl[0]["repo"], cl[0].get("skill_md_size", 0)) for cl in clusters]
                print(f"    SPLIT {name}: {len(clusters)} clusters (reps: {reps})")

    print(f"  Conflict clusters: {len(conflicts)} name-groups → "
          f"{n_split} split into multiple (kept under suffixes), "
          f"{len(conflicts)-n_split} merged as equivalent (longest wins)")
    return selected


# ---------------------------------------------------------------------------
# Copy (unchanged except accept final_name override)
# ---------------------------------------------------------------------------

def copy_skill(final_name: str, skill: dict, output_dir: Path, dry_run: bool = False):
    dest = output_dir / final_name
    src = skill["path"]
    if dry_run:
        return
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        src, dest,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "node_modules", ".DS_Store"),
    )
    source_info = {
        "source_repo": skill["repo"],
        "source_path": skill["rel_path"],
        "skill_md_size": skill["skill_md_size"],
        "original_name": skill["name"],
        "renamed_to": final_name if final_name != skill["name"] else None,
    }
    (dest / "_source.json").write_text(json.dumps(source_info, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Merge skill libraries into one directory")
    parser.add_argument("--repos", nargs="*", default=None)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-sub-skills", action="store_true")
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="Skip DeepSeek equivalence check; old longest-wins behavior.")
    parser.add_argument("--llm-model", default=LLM_MODEL)
    parser.add_argument("--llm-concurrency", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Discover repos
    all_repos = sorted([
        d for d in SCRIPT_DIR.iterdir()
        if d.is_dir() and (d / ".git").exists()
    ], key=lambda d: d.name)
    if args.repos:
        all_repos = [r for r in all_repos if r.name in args.repos]
    if not all_repos:
        print("No repos found."); sys.exit(1)
    print(f"Scanning {len(all_repos)} repos: {[r.name for r in all_repos]}")

    all_skills = []
    for repo_dir in all_repos:
        skills = find_skills(repo_dir)
        for s in skills:
            s["is_sub_skill"] = is_sub_skill(s, skills)
        all_skills.extend(skills)
        sub_count = sum(1 for s in skills if s["is_sub_skill"])
        print(f"  {repo_dir.name}: {len(skills)} skills ({sub_count} sub-skills)")

    if not args.include_sub_skills:
        before = len(all_skills)
        all_skills = [s for s in all_skills if not s["is_sub_skill"]]
        print(f"Filtered out {before - len(all_skills)} sub-skills")

    by_name: dict[str, list[dict]] = {}
    for s in all_skills:
        by_name.setdefault(s["name"], []).append(s)

    conflicts = {k: v for k, v in by_name.items() if len(v) > 1}
    print(f"\nTotal unique skill names: {len(by_name)}")
    print(f"Name conflicts (>1 candidate): {len(conflicts)}")

    judge = None
    if not args.no_llm_judge and conflicts:
        try:
            judge = MergeJudge(args.llm_model, args.llm_concurrency)
        except Exception as e:
            print(f"WARN: LLM judge unavailable ({e}); falling back to longest-wins.")
            judge = None

    selected = resolve_conflicts(by_name, judge, verbose=args.verbose)

    # Show first 10 splits as a sample
    splits = [(k, v) for k, v in selected.items() if "__" in k and k.split("__")[0] in by_name]
    if splits:
        print(f"\nFirst 10 split-renames:")
        for name, skill in splits[:10]:
            orig = skill["name"]
            print(f"  {orig}  →  {name}   (from {skill['repo']})")

    if not args.dry_run:
        if output_dir.exists():
            print(f"\nClearing existing output: {output_dir}")
            import subprocess
            subprocess.run(["rm", "-rf", str(output_dir)], check=False)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Merging {len(selected)} skills → {output_dir}")
    for final_name in sorted(selected.keys()):
        copy_skill(final_name, selected[final_name], output_dir, dry_run=args.dry_run)

    if not args.dry_run:
        index = {}
        for final_name, skill in sorted(selected.items()):
            index[final_name] = {
                "source_repo": skill["repo"],
                "source_path": skill["rel_path"],
                "skill_md_size": skill["skill_md_size"],
                "original_name": skill["name"],
            }
        (output_dir / "_index.json").write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n")
        print(f"\nDone. {len(selected)} skills merged.")
        print(f"Index: {output_dir / '_index.json'}")
    else:
        print(f"\n[DRY RUN] Would merge {len(selected)} skills.")


if __name__ == "__main__":
    main()
