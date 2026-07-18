# Central Workflows

This directory is the maintained orchestration layer for reproducible project runs.

Policy:
- Domain code stays in its owning package, e.g. `GeneralAgent/sft_data_collection`, `GeneralAgent/rl_data_prep`, and `Relax/examples/agent_bench`.
- Reproducible multi-step launch scripts live here.
- `experiments/` stores run outputs, manifests, logs, reports, trajectories, and checkpoints; it should not be the maintained source of launch scripts.
- Historical scripts migrated into this directory are archived under `archive/workflow_migration_20260526/`.
- Obsolete GeneralAgent/RL experiment helpers archived during the follow-up
  cleanup live under `archive/generalagent_cleanup_20260526/`.

Workflow groups:
- `sft_data_collection/`: collect phase1/phase2 skill-use trajectories and export SFT candidates.
- `sft_training/`: train/merge/evaluate SFT models from the current clean-plus-claw data line.
- `rl_data_prep/`: build RL split/parquet inputs from SFT data and exclusions.
- `rl_training/`: reproduce the 5-bench GRPO run and the claw-only GRPO run.

If a workflow needs benchmark-specific implementation code, keep that code in
the owning package and call it from a short script here. Do not put maintained
launch scripts under `experiments/`.
