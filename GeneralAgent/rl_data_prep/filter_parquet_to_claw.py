#!/usr/bin/env python3
"""Filter v2 train/eval parquet down to the claw bench only.

v1.1 ships only the Claw real launcher; harbor/swe launchers are still stubs
so any RL run that mixes those benches would fail at ``env.reset()``. This
helper produces a parquet pair that only references claw tasks, letting the
first GRPO pilot get real reward signal end-to-end.

Run::

    python -m GeneralAgent.rl_data_prep.filter_parquet_to_claw \\
        --input-dir  /mnt/.../datasets/rl/parquet_4bench_base_20260523 \\
        --output-dir /mnt/.../datasets/rl/parquet_claw_only_20260525
"""
from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "datasets/rl/parquet_4bench_base_20260523"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets/rl/parquet_claw"


def filter_one(in_path: Path, out_path: Path) -> int:
    import pyarrow.parquet as pq  # type: ignore
    import pyarrow as pa  # type: ignore

    table = pq.read_table(str(in_path))
    rows = table.to_pylist()
    claw_rows = [r for r in rows if r["extra_info"]["bench"] == "claw"]
    if not claw_rows:
        raise RuntimeError(f"{in_path}: no claw rows after filtering")
    out_table = pa.Table.from_pylist(claw_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, str(out_path))
    return len(claw_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    for fname in ("train.parquet", "eval.parquet"):
        n = filter_one(in_dir / fname, out_dir / fname)
        print(f"  {fname}: {n} claw rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
