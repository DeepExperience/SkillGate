#!/usr/bin/env bash
# Normal mixed-slate GRPO with disjoint task and token-local selector credit.

rl_profile_configure() {
  export RL_PROFILE=selector_action_credit
  export EXPERIMENT_BASENAME="${EXPERIMENT_BASENAME:-selector-action-credit-v1-oraclegrpo60}"
  export RL_RUN_PURPOSE="${RL_RUN_PURPOSE:-Normal on-policy all-gold mixed-slate GRPO from oracle-GRPO60 final99; task outcome excludes skill-read calls and oracle-vs-distractor selector credit updates only skill identity tokens.}"
  export RELAX_CONTEXT_DECISION="${RELAX_CONTEXT_DECISION:-selector_action_credit_actor8_tp4_cp2_70k_lr5e7_active128_hybridv8b0704d}"

  export SLATE_ROOT="${SLATE_ROOT:-${ROOT}/skill_libraries/snapshots/rl/slate_skills_20260710_hybrid_v8body_0704desc}"
  export SLATE_FALLBACK_ROOT="${SLATE_FALLBACK_ROOT:-${ROOT}/skill_libraries/snapshots/rl/slate_skills_20260704}"
  export AGENT_BENCH_EXTRA_SKILL_ROOTS="${AGENT_BENCH_EXTRA_SKILL_ROOTS:-${SLATE_ROOT}/skills:${SLATE_FALLBACK_ROOT}/skills}"
  export AGENT_BENCH_RETRIEVAL_TOP_N=16
  export SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-${ROOT}/datasets/rl/parquet_4bench_slate_regret_v8prod_20260708}"
  export DATA_DIR="${DATA_DIR:-${ROOT}/datasets/rl/parquet_4bench_selector_action_credit_hybridv8b0704d_allgold_20260713}"

  export MODEL_DIR="${MODEL_DIR:-${ROOT}/experiments/rl/runs/oraclegrpo60-hardspanv4params-pair-spec8-eval0/model/exports}"
  export QWEN35_9B_SFT_SUBDIR="${QWEN35_9B_SFT_SUBDIR:-iter-0000099-07932a57}"

  export RELAX_SELECTOR_ACTION_CREDIT=1
  export RELAX_SELECTOR_ACTION_LOSS_COEF="${RELAX_SELECTOR_ACTION_LOSS_COEF:-0.20}"
  export LOSS_TYPE=custom_loss
  export CUSTOM_LOSS_FUNCTION_PATH=examples.agent_bench.selector_action_grpo_loss.selector_action_grpo_loss
  export CUSTOM_REWARD_POST_PROCESS_PATH=examples.agent_bench.selector_action_credit.post_process_rewards
  export DYNAMIC_SAMPLING_FILTER_PATH=examples.agent_bench.selector_action_credit.keep_raw_task_reward_nonzero_std
  export RELAX_DISABLE_TIS=1

  # Every other historical selector/skill credit path is explicitly off.
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
  export LEARNING_RATE="${LEARNING_RATE:-5e-7}"
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
  # Rollout 0 is the exact pre-update behavior baseline. Relax's durable
  # keep-best association requires periodic evals to start after an update.
  export SKIP_EVAL_BEFORE_TRAIN=1
  export SAVE_INTERVAL=5
  export MAX_ACTOR_CKPT_TO_KEEP=1
  export KEEP_BEST_ACTOR_CKPT=1
}

rl_profile_prepare() {
  local builder="${ROOT}/ops/workflows/rl_data_prep/make_4bench_mixed_skill_bonus_compare_parquet.py"
  local -a args=(
    --input-dir "${SOURCE_DATA_DIR}"
    --output-dir "${DATA_DIR}"
    --train-manifest "${SLATE_ROOT}/manifest/slate_manifest_train.jsonl"
    --eval-manifest "${SLATE_ROOT}/manifest/slate_manifest_eval70.jsonl"
    --skill-roots "${AGENT_BENCH_EXTRA_SKILL_ROOTS}"
    --expected-train-tasks 491
    --expected-eval-tasks 56
  )

  if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/eval.parquet" || ! -f "${DATA_DIR}/build_report.json" ]]; then
    [[ "${DRY_RUN:-0}" != "1" ]] || {
      echo "FATAL: selector action-credit parquet missing during dry-run: ${DATA_DIR}" >&2
      return 2
    }
    "${PYTHON_BIN:-python3}" "${builder}" "${args[@]}"
  fi
  "${PYTHON_BIN:-python3}" "${builder}" "${args[@]}" --validate-only \
    >"/tmp/rl_profile_selector_action_credit_validate.json"
  "${PYTHON_BIN:-python3}" - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("/tmp/rl_profile_selector_action_credit_validate.json").read_text())
train, evaluation = report["train"], report["eval"]
assert train["rows"] == train["unique_tasks"] == train["gold_present"] == train["slate_size_16"] == 491, train
assert evaluation["rows"] == evaluation["unique_tasks"] == evaluation["gold_present"] == evaluation["slate_size_16"] == 56, evaluation
assert train["gold_absent"] == evaluation["gold_absent"] == 0, report
print("PROFILE_DATA_OK selector_action_credit train=491 eval=56 hybrid_v8body_0704desc slate16 all_gold")
PY
  "${PYTHON_BIN:-python3}" "${ROOT}/ops/workflows/rl_training/tools/smoke_selector_action_credit.py" \
    >"/tmp/rl_profile_selector_action_credit_smoke.json"
}

rl_profile_validate_algorithm() {
  [[ "${AGENT_BENCH_RETRIEVAL_TOP_N}" == 16 && "${N_SAMPLES_PER_PROMPT}" == 8 \
     && "${ROLLOUT_BATCH_SIZE}" == 16 && "${GLOBAL_BATCH_SIZE}" == 128 ]] || {
    echo "FATAL: selector action credit requires mixed slate16, n=8, rollout_batch=16, global_batch=128" >&2
    return 2
  }
  [[ "${NUM_ROLLOUT}" == 100 && "${LEARNING_RATE}" == 5e-7 \
     && "${RELAX_SELECTOR_ACTION_LOSS_COEF}" == 0.20 ]] || {
    echo "FATAL: selector action credit freezes num_rollout=100, lr=5e-7, selector_coef=0.20" >&2
    return 2
  }
  [[ "${LOSS_TYPE}" == custom_loss \
     && "${CUSTOM_LOSS_FUNCTION_PATH}" == examples.agent_bench.selector_action_grpo_loss.selector_action_grpo_loss \
     && "${CUSTOM_REWARD_POST_PROCESS_PATH}" == examples.agent_bench.selector_action_credit.post_process_rewards \
     && "${DYNAMIC_SAMPLING_FILTER_PATH}" == examples.agent_bench.selector_action_credit.keep_raw_task_reward_nonzero_std ]] || {
    echo "FATAL: selector action-credit loss/reward/filter wiring drifted" >&2
    return 2
  }
  [[ "${RELAX_SELECTOR_ACTION_CREDIT}" == 1 && "${RELAX_DISABLE_TIS}" == 1 \
     && "${CALCULATE_PER_TOKEN_LOSS}" == 1 ]] || {
    echo "FATAL: selector action credit requires feature=1, TIS disabled, per-token loss enabled" >&2
    return 2
  }
  [[ "${SKIP_EVAL_BEFORE_TRAIN}" == 1 && "${KEEP_BEST_ACTOR_CKPT}" == 1 ]] || {
    echo "FATAL: selector action credit requires pre-update rollout0 audit plus durable post-update keep-best" >&2
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
      echo "FATAL: selector action credit requires ${key}=0, got ${!key}" >&2
      return 2
    }
  done
  [[ "${RELAX_DYNAMIC_FILTER_MAX_REJECTS_PER_ROLLOUT}" == 0 \
     && "${RELAX_DYNAMIC_FILTER_MAX_REJECT_SAMPLES_PER_ROLLOUT}" == 0 ]] || {
    echo "FATAL: selector action credit forbids force-accept fallback" >&2
    return 2
  }
  echo "PROFILE_ALGORITHM_OK selector_action_credit normal_onpolicy task_raw_grpo identity_token_selector"
}
