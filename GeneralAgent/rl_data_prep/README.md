# RL Data Preparation

This directory owns the code that turns cleaned SFT/evaluation metadata into
Relax-compatible RL split and parquet inputs.

## Current Line

- `build_rl_split_v2.py`: builds the canonical v2 RL split, applying holdout and
  structural task exclusions.
- `convert_to_relax_data_v2.py`: materializes the v2 split into Relax training
  and evaluation parquet files.
- `filter_rl_split_exclusions.py`: applies maintained exclusion lists.
- `filter_parquet_to_claw.py`: creates claw-only ablation/training subsets.

The maintained workflow entrypoint is:

- `ops/workflows/rl_data_prep/build_rl_data_v2.sh`

## Archived Line

The older v1 split/conversion flow was archived because current training and
audits use v2:

- `archive/generalagent_cleanup_20260526/originals/GeneralAgent/rl_data_prep/`
- `archive/generalagent_cleanup_20260526/originals/experiments/rl/`

