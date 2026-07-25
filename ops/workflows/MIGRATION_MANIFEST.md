# Workflow Migration Manifest — 2026-05-26

Goal: centralize maintained orchestration scripts under `ops/workflows` while preserving domain code in-place and archiving superseded historical script locations.

## Canonical workflow entries

### SFT data collection
- `ops/workflows/sft_data_collection/run_qwen27b_campaign_pipeline.sh` — copied from `ops/launch/run_sft_qwen27b_campaign_pipeline.sh`; now calls workflow-local `run_sft_pipeline.sh`.
- `ops/workflows/sft_data_collection/run_sft_pipeline.sh` — copied from `ops/launch/run_sft_pipeline.sh`; main phase1/phase2 collection pipeline.
- `ops/workflows/sft_data_collection/run_phase2_teacher_worker.sh` — copied from `ops/launch/run_phase2_teacher_worker.sh`; standalone teacher fallback worker.
- `ops/workflows/sft_data_collection/collect_and_export.sh` — copied from `GeneralAgent/sft_data_collection/scripts/collect_and_export.sh`; converts successful trajectories to SFT/LLaMA-Factory data.

### SFT training
- `ops/workflows/sft_training/run_27b_clean_plus_claw_sft_eval_chain.sh` — copied from `ops/launch/run_claw_collect_to_27b_sft_eval_chain.sh`; clean-plus-claw 27B train/export/eval chain.
- `ops/workflows/sft_training/run_9b_clean_plus_claw_lora.sh` — copied from current clean-plus-claw 9B LLaMA-Factory launcher.
- `ops/workflows/sft_training/run_27b_clean_lora_legacy.sh` — retained as legacy clean-only 27B LLaMA-Factory launcher for historical reproduction.
- `ops/workflows/sft_training/run_sft_v2_serve_and_eval_chain.sh` — copied from `ops/launch/run_sft_v2_serve_and_eval_chain.sh`; merge/serve/quick-eval helper for older SFT runs.

### RL data prep
- `ops/workflows/rl_data_prep/build_rl_data_v2.sh` — new thin wrapper that calls `GeneralAgent/rl_data_prep/build_rl_split_v2.py` and `convert_to_relax_data_v2.py`.
- `ops/workflows/rl_data_prep/filter_parquet_tasks.py` — copied from `experiments/rl/v2/launch/filter_parquet_tasks.py`.

### RL training
- `ops/workflows/rl_training/run_5bench_from_sft_initial_active32_lr5e6_20260526.sh` — initial 5-bench GRPO launch from the current SFT model, using the same active32/tiny-KL/lr5e-6 runtime shape as the successful 20260525 iter99→159 segment.
- `ops/workflows/rl_training/run_resume_prebuilt_generic.sh` — copied from the generic prebuilt-image resume launcher.
- `ops/workflows/rl_training/run_5bench_resume_iter99_active32_tinykl_lr5e6.sh` — copied from the final active32/tiny-KL/lr5e-6 5-bench resume launcher.
- `ops/workflows/rl_training/run_claw_only_from_sft_active32_100rollouts.sh` — copied from the current claw-only from-SFT GRPO launcher.
- `ops/workflows/rl_training/tools/sample_rollout_trajectories.py` — copied from the old experiment launch helper.

## Validation performed

- `bash -n` passed for all shell scripts in `ops/workflows`.
- Python syntax compile passed for copied Python helpers with `PYTHONDONTWRITEBYTECODE=1`.
- Workflow copies no longer call `experiments/rl/v2/launch/...` as maintained entrypoints.

## Archive location

Superseded original scripts are moved to `archive/workflow_migration_20260526/` with their original relative directory structure.

## Follow-up cleanup — 2026-05-26

Additional non-canonical helpers were conservatively archived under
`archive/generalagent_cleanup_20260526/`:

- SFT one-off analysis/conversion helpers and obsolete provider configs.
- RL v1 split/conversion code and old cached experiment outputs.
- Historical benchmark direct runners/watchdogs superseded by
  `GeneralAgent/eval_scripts/unified_runner/`.
- `GeneralAgent/rl_training/` deploy-check helpers, now archived because the
  maintained RL implementation is in `Relax/` and launch orchestration is in
  `ops/workflows/rl_training/`.
- Older SFT `datasets/` and `llamafactory_data/` directories that are not needed
  to inspect/regenerate the current 20260512 clean-plus-claw-thinkwrap SFT data.

The maintained boundary after cleanup is:

- `GeneralAgent/*`: domain code and small package-local utilities.
- `ops/workflows/*`: reproducible orchestration scripts.
- `experiments/*`: run outputs only.
