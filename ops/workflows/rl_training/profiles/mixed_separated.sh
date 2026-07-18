#!/usr/bin/env bash
# Pure mixed-skill GRPO with a separate selector advantage.

rl_profile_configure() {
  export RL_PROFILE=mixed_separated
  export EXPERIMENT_BASENAME="${EXPERIMENT_BASENAME:-mixed-skills-separated-v8prod}"
  export RL_RUN_PURPOSE="${RL_RUN_PURPOSE:-Mixed 16-skill GRPO with factual task advantage plus outcome-stratified selector behavior advantage.}"
  export RELAX_CONTEXT_DECISION="${RELAX_CONTEXT_DECISION:-mixed_separated_actor8_tp4_cp2_70k_lr1e6_active128_v8prod_gold}"

  export SLATE_ROOT="${SLATE_ROOT:-${ROOT}/skill_libraries/snapshots/rl/slate_skills_20260708_hard_negative_v8_production}"
  export SLATE_FALLBACK_ROOT="${SLATE_FALLBACK_ROOT:-${ROOT}/skill_libraries/snapshots/rl/slate_skills_20260704}"
  export AGENT_BENCH_EXTRA_SKILL_ROOTS="${AGENT_BENCH_EXTRA_SKILL_ROOTS:-${SLATE_ROOT}/skills:${SLATE_FALLBACK_ROOT}/skills}"
  export AGENT_BENCH_RETRIEVAL_TOP_N=16
  export SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-${ROOT}/datasets/rl/parquet_4bench_mixed_skill_bonus_compare_v8prod_allgold_20260710}"
  export DATA_DIR="${DATA_DIR:-${ROOT}/datasets/rl/parquet_4bench_mixed_skill_separated_continuous_advantage_v8prod_allgold_20260710}"

  export RELAX_MIXED_SKILL_BONUS_ENABLED=0
  export RELAX_MIXED_SEPARATED_ADV_ENABLED=1
  export RELAX_MIXED_SEPARATED_BEHAVIOR_COEF="${RELAX_MIXED_SEPARATED_BEHAVIOR_COEF:-0.30}"
  export RELAX_MIXED_SEPARATED_BEHAVIOR_CLIP="${RELAX_MIXED_SEPARATED_BEHAVIOR_CLIP:-0.40}"
  export SETA_CONTINUOUS_REWARD=0
  export CUSTOM_REWARD_POST_PROCESS_PATH=examples.agent_bench.mixed_skill_separated_advantage.post_process_rewards
  export DYNAMIC_SAMPLING_FILTER_PATH=examples.agent_bench.mixed_skill_separated_advantage.keep_raw_task_reward_nonzero_std
  export LOSS_TYPE=policy_loss
  export RELAX_DISABLE_TIS=1

  export ROLLOUT_SHUFFLE=0
  export OVER_SAMPLING_BATCH_SIZE=32
  export RELAX_DYNAMIC_FILTER_MAX_REJECTS_PER_ROLLOUT=0
  export RELAX_DYNAMIC_FILTER_MAX_REJECT_SAMPLES_PER_ROLLOUT=0
  export RELAX_DYNAMIC_FILTER_MIN_SKILL_READ_FRAC=0
  export RELAX_DYNAMIC_FILTER_MIN_NO_SKILL_READ_FRAC=0
  export RELAX_DYNAMIC_FILTER_SKILL_READ_MAX_SAMPLES=0
  export DISABLE_EVAL=0
  export EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
  export SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-1}"
  export KEEP_BEST_ACTOR_CKPT="${KEEP_BEST_ACTOR_CKPT:-1}"
}

rl_profile_prepare() {
  local source_builder="${ROOT}/ops/workflows/rl_data_prep/make_4bench_mixed_skill_bonus_compare_parquet.py"
  local builder="${ROOT}/ops/workflows/rl_data_prep/make_4bench_mixed_skill_separated_advantage_parquet.py"
  local -a source_args=(
    --output-dir "${SOURCE_DATA_DIR}"
    --train-manifest "${SLATE_ROOT}/manifest/slate_manifest_train.jsonl"
    --eval-manifest "${SLATE_ROOT}/manifest/slate_manifest_eval70.jsonl"
    --skill-roots "${AGENT_BENCH_EXTRA_SKILL_ROOTS}"
    --expected-train-tasks 491
    --expected-eval-tasks 56
  )
  local -a args=(
    --input-dir "${SOURCE_DATA_DIR}"
    --output-dir "${DATA_DIR}"
    --expected-train-tasks 491
    --expected-eval-tasks 56
  )

  if [[ ! -f "${SOURCE_DATA_DIR}/train.parquet" || ! -f "${SOURCE_DATA_DIR}/eval.parquet" || ! -f "${SOURCE_DATA_DIR}/build_report.json" ]]; then
    [[ "${DRY_RUN:-0}" != "1" ]] || { echo "FATAL: mixed source parquet missing during dry-run" >&2; return 2; }
    "${PYTHON_BIN:-python3}" "${source_builder}" "${source_args[@]}"
  fi
  "${PYTHON_BIN:-python3}" "${source_builder}" "${source_args[@]}" --validate-only \
    >"/tmp/rl_profile_mixed_source_validate.json"

  if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/eval.parquet" || ! -f "${DATA_DIR}/build_report.json" ]]; then
    [[ "${DRY_RUN:-0}" != "1" ]] || { echo "FATAL: mixed separated parquet missing during dry-run" >&2; return 2; }
    "${PYTHON_BIN:-python3}" "${builder}" "${args[@]}"
  fi
  "${PYTHON_BIN:-python3}" "${builder}" "${args[@]}" --validate-only \
    >"/tmp/rl_profile_mixed_separated_validate.json"
  "${PYTHON_BIN:-python3}" - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("/tmp/rl_profile_mixed_separated_validate.json").read_text())
train, evaluation = report["train"], report["eval"]
assert train["rows"] == train["unique_tasks"] == train["gold_present"] == train["slate_size_16"] == 491, train
assert evaluation["rows"] == evaluation["unique_tasks"] == evaluation["gold_present"] == evaluation["slate_size_16"] == 56, evaluation
assert train["gold_absent"] == evaluation["gold_absent"] == 0, report
assert train["prompt_equal_to_source"] == 491 and evaluation["prompt_equal_to_source"] == 56, report
assert train["schema"] == "continuous_task_grpo_plus_adaptive_outcome_stratified_behavior_v3", train
print("PROFILE_DATA_OK mixed_separated train=491 eval=56 slate16 all_gold")
PY
  "${PYTHON_BIN:-python3}" "${ROOT}/ops/workflows/rl_training/tools/smoke_mixed_skill_separated_advantage.py" \
    >"/tmp/rl_profile_mixed_separated_smoke.json"
}
