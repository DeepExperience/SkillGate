"""Pick a small stratified subset of SWE-Gym-Lite + SWE-bench Verified instances
   for docker image pre-pull. Emits two image-tag lists under /tmp.

Usage: python pick_subset.py
"""
import os
import pandas as pd
from pathlib import Path

DATA = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL")) / "datasets"
LITE_PQ = DATA / "swe-gym/lite/data/train-00000-of-00001.parquet"
VER_PQ  = DATA / "swe-bench-verified/data/data/test-00000-of-00001.parquet"


def to_image(instance_id: str) -> str:
    """parquet instance_id -> docker image tag (SWE-bench convention):
        '/' -> '_',  '__' -> '_s_',  lowercase."""
    tag = instance_id.replace("__", "_s_").replace("/", "_").lower()
    return f"xingyaoww/sweb.eval.x86_64.{tag}:latest"


def pick(df: pd.DataFrame, plan: list[tuple[str, int]]) -> list[str]:
    out = []
    for repo, n in plan:
        out += df[df["repo"] == repo].head(n)["instance_id"].tolist()
    return out


# SWE-Gym-Lite: 10 across 7 repos (rollout sanity)
lite = pd.read_parquet(LITE_PQ)
lite_plan = [("getmoto/moto", 2), ("python/mypy", 2), ("iterative/dvc", 2),
             ("Project-MONAI/MONAI", 1), ("pydantic/pydantic", 1),
             ("dask/dask", 1), ("facebookresearch/hydra", 1)]
lite_ids = pick(lite, lite_plan)

# SWE-bench Verified: 10 from smallest repos (eval sanity; keep disk small)
ver = pd.read_parquet(VER_PQ)
ver_plan = [("pallets/flask", 1), ("mwaskom/seaborn", 2), ("psf/requests", 2),
            ("pylint-dev/pylint", 2), ("pytest-dev/pytest", 2),
            ("pydata/xarray", 1)]
ver_ids = pick(ver, ver_plan)

Path("/tmp/pull_lite.txt").write_text("\n".join(to_image(i) for i in lite_ids) + "\n")
Path("/tmp/pull_verified.txt").write_text("\n".join(to_image(i) for i in ver_ids) + "\n")

print(f"Lite ({len(lite_ids)}):")
for i in lite_ids: print(f"  {i}  ->  {to_image(i)}")
print(f"\nVerified ({len(ver_ids)}):")
for i in ver_ids: print(f"  {i}  ->  {to_image(i)}")
print(f"\nwrote /tmp/pull_lite.txt, /tmp/pull_verified.txt")
