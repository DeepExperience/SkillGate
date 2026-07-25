#!/usr/bin/env bash
# FINAL Hybrid mixed-slate GRPO ablation: keep the selector action/task masks,
# but multiply selector advantage loss by zero.  Historical clean-oracle and
# task-only profiles are intentionally untouched.

rl_profile_configure() {
  export RL_PROFILE=selector_clean_oracle_masked_task_only
  export EXPERIMENT_BASENAME="${EXPERIMENT_BASENAME:-selector-clean-oracle-maskedtaskonly-sft9b-finalhybrid-lr1e6}"
  export RL_RUN_PURPOSE="${RL_RUN_PURPOSE:-Masked-task-only ablation on FINAL Hybrid train/V8-fixed4 eval: read-call tokens are removed from task PG, selector loss coefficient is exactly zero, and all other clean-oracle run hyperparameters remain fixed.}"
  export RELAX_CONTEXT_DECISION="${RELAX_CONTEXT_DECISION:-selector_clean_oracle_masked_task_only_sft9b_actor8_tp4_cp2_70k_lr1e6_active128_finalhybrid}"

  export SLATE_ROOT="${SLATE_ROOT:-${ROOT}/skill_libraries/snapshots/rl/slate_skills_20260708_hard_negative_v8_production}"
  export SLATE_FALLBACK_ROOT="${SLATE_FALLBACK_ROOT:-${ROOT}/skill_libraries/snapshots/rl/slate_skills_20260704}"
  export AGENT_BENCH_EXTRA_SKILL_ROOTS="${AGENT_BENCH_EXTRA_SKILL_ROOTS:-${SLATE_ROOT}/skills:${SLATE_FALLBACK_ROOT}/skills}"
  export AGENT_BENCH_RETRIEVAL_TOP_N=16
  export DATA_DIR="${DATA_DIR:-${ROOT}/datasets/rl/parquet_4bench_final_hybridtrain_v8prodfixed4eval_20260720}"

  export MODEL_DIR="${MODEL_DIR:-${ROOT}/GeneralAgent/sft_training/merged_models}"
  export QWEN35_9B_SFT_SUBDIR="${QWEN35_9B_SFT_SUBDIR:-qwen35_9b_sft_campaign_20260512_clean_plus_claw_thinkwrap_4gpu_49k_5epoch_r32_liger}"

  # The feature remains enabled because it constructs disjoint task/action
  # masks.  coef=0 removes clean-oracle credit while retaining the task mask.
  export RELAX_SELECTOR_ACTION_CREDIT=1
  export RELAX_SELECTOR_ACTION_LOSS_COEF=0
  export LOSS_TYPE=custom_loss
  export CUSTOM_LOSS_FUNCTION_PATH=examples.agent_bench.selector_action_grpo_loss.selector_action_grpo_loss
  export CUSTOM_REWARD_POST_PROCESS_PATH=examples.agent_bench.selector_clean_oracle_action_credit.post_process_rewards
  export DYNAMIC_SAMPLING_FILTER_PATH=examples.agent_bench.selector_clean_oracle_action_credit.keep_raw_task_reward_nonzero_std
  export RELAX_DISABLE_TIS=1

  # Every other historical selector/skill-credit path stays off.
  export RELAX_MIXED_SKILL_BONUS_ENABLED=0
  export RELAX_MIXED_SEPARATED_ADV_ENABLED=0
  export RELAX_MIXED_SEPARATED_BEHAVIOR_COEF=0
  export RELAX_SKILL_GROUP_REWARD=0
  export RELAX_SKILL_GROUP_BONUS_COEF=0
  export RELAX_SKILL_GROUP_BONUS_MAX=0
  export RELAX_SKILL_GROUP_MARGIN=0
  export RELAX_SKILL_GROUP_SUBGROUP_ADV_COEF=0
  export RELAX_SKILL_GROUP_REQUIRE_BOTH=0
  export RELAX_SKILL_GROUP_NO_READ_SUCCESS_BONUS=0
  export RELAX_SLATE_REGRET_GRPO=0
  export RELAX_SLATE_STRATIFIED_ADVANTAGE=0
  export RELAX_PAIR_ATOMIC_SAMPLING=0
  export RELAX_PAIR_SPECULATIVE_EXTRA_GROUPS=0
  export RELAX_PAIR_ORACLE_GRPO=0
  export RELAX_OPSD_MODE=0
  export RELAX_M1_CLEAN=0
  export RELAX_PROMPT_ONLY_SHADOW_CLEAN=0
  export RELAX_SHADOW_BC_ACTION_MASK=0
  export RELAX_SHADOW_BC_HARD_SPAN_MASK=0
  export SETA_CONTINUOUS_REWARD=0

  export NUM_ROLLOUT=100
  export ROLLOUT_BATCH_SIZE=16
  export N_SAMPLES_PER_PROMPT=8
  export GLOBAL_BATCH_SIZE=128
  export NUM_ITERS_PER_TRAIN_UPDATE=4
  export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
  export CALCULATE_PER_TOKEN_LOSS=1
  export ROLLOUT_SHUFFLE=0
  export OVER_SAMPLING_BATCH_SIZE=32
  export RELAX_DYNAMIC_FILTER_MAX_REJECTS_PER_ROLLOUT=0
  export RELAX_DYNAMIC_FILTER_MAX_REJECT_SAMPLES_PER_ROLLOUT=0
  export RELAX_DYNAMIC_FILTER_MIN_SKILL_READ_FRAC=0
  export RELAX_DYNAMIC_FILTER_MIN_NO_SKILL_READ_FRAC=0
  export RELAX_DYNAMIC_FILTER_SKILL_READ_MAX_SAMPLES=0

  export DISABLE_EVAL=0
  export EVAL_INTERVAL=10
  export SKIP_EVAL_BEFORE_TRAIN=1
  export SAVE_INTERVAL=5
  export MAX_ACTOR_CKPT_TO_KEEP=1
  export KEEP_BEST_ACTOR_CKPT=1
}

rl_profile_prepare() {
  local train="${DATA_DIR}/train.parquet" evaluation="${DATA_DIR}/eval.parquet"
  local expected_train="6dd2350879c6337fc0304f6ea08973ee9d8697ed6a72c70467aab9ae41f30732"
  local expected_eval="4d6ebedecc0c9d730f0c6800d68a674b1fe7699978f06e13170ab766de346b35"
  [[ -f "${train}" && -f "${evaluation}" ]] || {
    echo "FATAL: FINAL train/eval parquet missing: ${DATA_DIR}" >&2
    return 2
  }
  [[ "$(sha256sum "${train}" | awk '{print $1}')" == "${expected_train}" ]] || {
    echo "FATAL: FINAL train parquet hash drifted: ${train}" >&2
    return 2
  }
  [[ "$(sha256sum "${evaluation}" | awk '{print $1}')" == "${expected_eval}" ]] || {
    echo "FATAL: FINAL eval parquet hash drifted: ${evaluation}" >&2
    return 2
  }
  "${PYTHON_BIN:-python3}" - "${train}" "${evaluation}" <<'PY'
import sys
import pandas as pd

for path, expected in ((sys.argv[1], 491), (sys.argv[2], 56)):
    frame = pd.read_parquet(path, columns=["reward_model", "extra_info"])
    keys = {
        (str(row["reward_model"]["bench"]), str(row["reward_model"]["task_id"]))
        for _, row in frame.iterrows()
    }
    assert len(frame) == len(keys) == expected, (path, len(frame), len(keys))
    assert all(float(extra["slate_contains_gold"]) == 1 for extra in frame["extra_info"]), path
    assert all(int(extra["slate_size"]) == 16 for extra in frame["extra_info"]), path
print("PROFILE_DATA_OK selector_clean_oracle_masked_task_only train=491 Hybrid eval=56 V8-fixed4 slate16 all_gold FINAL hashes")
PY
  "${PYTHON_BIN:-python3}" "${ROOT}/ops/workflows/rl_training/tools/smoke_selector_clean_oracle_action_credit.py" \
    >"/tmp/rl_profile_selector_clean_oracle_masked_task_only_smoke.json"
}

rl_profile_validate_algorithm() {
  [[ "${AGENT_BENCH_RETRIEVAL_TOP_N}" == 16 && "${N_SAMPLES_PER_PROMPT}" == 8 \
     && "${ROLLOUT_BATCH_SIZE}" == 16 && "${GLOBAL_BATCH_SIZE}" == 128 ]] || {
    echo "FATAL: masked-task-only requires slate16, n=8, rollout_batch=16, global_batch=128" >&2
    return 2
  }
  [[ "${NUM_ROLLOUT}" == 100 && "${LEARNING_RATE}" == 1e-6 \
     && "${RELAX_SELECTOR_ACTION_LOSS_COEF}" == 0 ]] || {
    echo "FATAL: masked-task-only freezes num_rollout=100, lr=1e-6, selector_coef=0" >&2
    return 2
  }
  [[ "${LOSS_TYPE}" == custom_loss \
     && "${CUSTOM_LOSS_FUNCTION_PATH}" == examples.agent_bench.selector_action_grpo_loss.selector_action_grpo_loss \
     && "${CUSTOM_REWARD_POST_PROCESS_PATH}" == examples.agent_bench.selector_clean_oracle_action_credit.post_process_rewards \
     && "${DYNAMIC_SAMPLING_FILTER_PATH}" == examples.agent_bench.selector_clean_oracle_action_credit.keep_raw_task_reward_nonzero_std ]] || {
    echo "FATAL: masked-task-only loss/reward/filter wiring drifted" >&2
    return 2
  }
  [[ "${RELAX_SELECTOR_ACTION_CREDIT}" == 1 && "${RELAX_DISABLE_TIS}" == 1 \
     && "${CALCULATE_PER_TOKEN_LOSS}" == 1 ]] || {
    echo "FATAL: masked-task-only requires selector masks, TIS disabled, per-token loss" >&2
    return 2
  }
  [[ "${SKIP_EVAL_BEFORE_TRAIN}" == 1 && "${KEEP_BEST_ACTOR_CKPT}" == 1 ]] || {
    echo "FATAL: masked-task-only checkpoint/eval contract drifted" >&2
    return 2
  }
  local key
  for key in \
    RELAX_MIXED_SKILL_BONUS_ENABLED RELAX_MIXED_SEPARATED_ADV_ENABLED \
    RELAX_MIXED_SEPARATED_BEHAVIOR_COEF RELAX_SKILL_GROUP_REWARD \
    RELAX_SKILL_GROUP_BONUS_COEF RELAX_SKILL_GROUP_BONUS_MAX \
    RELAX_SKILL_GROUP_SUBGROUP_ADV_COEF RELAX_SKILL_GROUP_NO_READ_SUCCESS_BONUS \
    RELAX_SLATE_REGRET_GRPO RELAX_SLATE_STRATIFIED_ADVANTAGE \
    RELAX_PAIR_ATOMIC_SAMPLING RELAX_PAIR_SPECULATIVE_EXTRA_GROUPS \
    RELAX_PAIR_ORACLE_GRPO RELAX_OPSD_MODE RELAX_M1_CLEAN \
    RELAX_PROMPT_ONLY_SHADOW_CLEAN RELAX_SHADOW_BC_ACTION_MASK \
    RELAX_SHADOW_BC_HARD_SPAN_MASK; do
    [[ "${!key:-0}" == 0 ]] || {
      echo "FATAL: masked-task-only requires ${key}=0, got ${!key}" >&2
      return 2
    }
  done
  [[ "${RELAX_DYNAMIC_FILTER_MAX_REJECTS_PER_ROLLOUT}" == 0 \
     && "${RELAX_DYNAMIC_FILTER_MAX_REJECT_SAMPLES_PER_ROLLOUT}" == 0 ]] || {
    echo "FATAL: masked-task-only forbids force-accept fallback" >&2
    return 2
  }
  echo "PROFILE_ALGORITHM_OK selector_clean_oracle_masked_task_only task_raw_grpo read_action_task_mask selector_coef_zero"
}
