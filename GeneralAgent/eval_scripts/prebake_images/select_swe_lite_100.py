#!/usr/bin/env python3
"""Select 100 SWE-Gym-lite instances via stratified sampling (proportional to repo).

Rationale:
  - lite has 230 instances across 11 repos; naive random sample could
    over-represent the largest (moto 59, mypy 40) and miss small repos.
  - Stratified: each repo contributes ceil(100 * repo_count / 230) instances,
    then trim to exactly 100 by removing one from the largest repos.
  - Sorted by instance_id inside each repo for determinism (no randomness).

Output: swe_lite_100.txt (one instance_id per line, matches parquet `instance_id`).

Usage:
    python3 select_swe_lite_100.py             # writes swe_lite_100.txt
    python3 select_swe_lite_100.py --n 150     # select 150 instead
    python3 select_swe_lite_100.py --preview   # show per-repo split, don't write
"""

import os
import argparse
import math
from collections import Counter
from pathlib import Path

import pandas as pd

PARQUET = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL")) / "datasets/swe-gym/lite/data/train-00000-of-00001.parquet"
IMAGE_PREFIX = "xingyaoww/sweb.eval.x86_64."


def stratified_select(df: pd.DataFrame, n: int) -> list[str]:
    """Return n instance_ids, proportionally across repos, deterministic."""
    repo_counts = df["repo"].value_counts().to_dict()
    total = sum(repo_counts.values())
    # Initial allocation by ceil proportional
    quota = {
        r: max(1, math.ceil(n * c / total)) for r, c in repo_counts.items()
    }
    # Adjust down to exact n (trim from largest quotas until sum==n)
    while sum(quota.values()) > n:
        biggest = max(quota, key=quota.get)
        quota[biggest] -= 1
    # If sum < n (unlikely with ceil), pad from largest repos
    while sum(quota.values()) < n:
        biggest_by_count = max(repo_counts, key=lambda r: repo_counts[r] - quota[r])
        quota[biggest_by_count] += 1

    selected = []
    for repo, k in quota.items():
        subset = sorted(df[df["repo"] == repo]["instance_id"].tolist())
        selected.extend(subset[:k])
    return selected


def instance_to_image(iid: str) -> str:
    return IMAGE_PREFIX + iid.replace("__", "_s_") + ":latest"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, help="Total to select")
    ap.add_argument("--out", default=str(Path(__file__).parent / "swe_lite_100.txt"))
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET)
    print(f"Loaded {len(df)} swe-gym-lite instances across {df['repo'].nunique()} repos")

    selected = stratified_select(df, args.n)
    repo_c = Counter(df[df["instance_id"].isin(selected)]["repo"])
    print(f"\nSelected {len(selected)} instances, per-repo split:")
    for repo, cnt in repo_c.most_common():
        orig = (df["repo"] == repo).sum()
        print(f"  {cnt:3d}/{orig:3d}  {repo}")

    if args.preview:
        print(f"\nSample (first 5 + last 5):")
        for iid in selected[:5] + ["..."] + selected[-5:]:
            print(f"  {iid}")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# SWE-Gym-lite {len(selected)} instances selected by stratified-by-repo sampling\n")
        f.write(f"# From {PARQUET}\n")
        f.write(f"# Per-repo split: {dict(repo_c)}\n")
        f.write(f"# Corresponding image names: {IMAGE_PREFIX}<instance_id with __ → _s_>:latest\n\n")
        for iid in selected:
            f.write(f"{iid}\n")

    # Also emit a parallel image-name list (used by prebake script)
    img_out = out.with_name(out.stem + "_images.txt")
    with open(img_out, "w") as f:
        f.write("# Docker image tags for the selected swe-gym-lite instances\n")
        for iid in selected:
            f.write(f"{instance_to_image(iid)}\n")

    print(f"\nWrote {len(selected)} instance_ids → {out}")
    print(f"Wrote {len(selected)} image tags   → {img_out}")


if __name__ == "__main__":
    main()
