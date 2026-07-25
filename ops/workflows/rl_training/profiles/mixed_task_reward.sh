#!/usr/bin/env bash
# Mixed-skill control: ordinary GRPO on verifier task reward only.  The
# historical V8-production dataset remains the default.  Point DATA_DIR at the
# frozen FINAL Hybrid-train/V8-fixed4-eval directory to run the paper-final data
# protocol; that branch is accepted only when both frozen parquet hashes match.

rl_profile_configure() {
  export RL_PROFILE=mixed_task_reward
  export EXPERIMENT_BASENAME="${EXPERIMENT_BASENAME:-mixed-skills-task-reward-v8prod}"
  export RL_RUN_PURPOSE="${RL_RUN_PURPOSE:-V8-production all-gold 16-skill mixed-prompt control using only final verifier task reward and ordinary GRPO advantage.}"
  export RELAX_CONTEXT_DECISION="${RELAX_CONTEXT_DECISION:-mixed_task_reward_actor8_tp4_cp2_70k_lr1e6_active128_v8prod_allgold}"

  export SLATE_ROOT="${SLATE_ROOT:-${ROOT}/skill_libraries/snapshots/rl/slate_skills_20260708_hard_negative_v8_production}"
  export SLATE_FALLBACK_ROOT="${SLATE_FALLBACK_ROOT:-${ROOT}/skill_libraries/snapshots/rl/slate_skills_20260704}"
  export AGENT_BENCH_EXTRA_SKILL_ROOTS="${AGENT_BENCH_EXTRA_SKILL_ROOTS:-${SLATE_ROOT}/skills:${SLATE_FALLBACK_ROOT}/skills}"
  export AGENT_BENCH_RETRIEVAL_TOP_N=16
  export DATA_DIR="${DATA_DIR:-${ROOT}/datasets/rl/parquet_4bench_mixed_skill_bonus_compare_v8prod_allgold_20260710}"

  # Preserve behavior observability in trajectories/W&B, but keep every
  # behavior-dependent training coefficient and advantage path disabled.
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
  export SETA_CONTINUOUS_REWARD=0

  export LOSS_TYPE=policy_loss
  export CUSTOM_REWARD_POST_PROCESS_PATH=examples.agent_bench.skill_group_reward.post_process_rewards
  export DYNAMIC_SAMPLING_FILTER_PATH=relax.engine.filters.dynamic_sampling_filters.check_reward_nonzero_std
  export RELAX_DISABLE_TIS=1

  export ROLLOUT_SHUFFLE=0
  export OVER_SAMPLING_BATCH_SIZE=32
  export RELAX_DYNAMIC_FILTER_MAX_REJECTS_PER_ROLLOUT="${RELAX_DYNAMIC_FILTER_MAX_REJECTS_PER_ROLLOUT:-64}"
  export RELAX_DYNAMIC_FILTER_MAX_REJECT_SAMPLES_PER_ROLLOUT="${RELAX_DYNAMIC_FILTER_MAX_REJECT_SAMPLES_PER_ROLLOUT:-300}"
  export RELAX_DYNAMIC_FILTER_MIN_SKILL_READ_FRAC=0
  export RELAX_DYNAMIC_FILTER_MIN_NO_SKILL_READ_FRAC=0
  export RELAX_DYNAMIC_FILTER_SKILL_READ_MAX_SAMPLES=0

  export DISABLE_EVAL=0
  export EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
  export SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-1}"
  export KEEP_BEST_ACTOR_CKPT="${KEEP_BEST_ACTOR_CKPT:-1}"
}

rl_profile_prepare() {
  local final_data_dir="${ROOT}/datasets/rl/parquet_4bench_final_hybridtrain_v8prodfixed4eval_20260720"
  if [[ "${DATA_DIR}" == "${final_data_dir}" ]]; then
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
    frame = pd.read_parquet(path, columns=["prompt", "reward_model", "extra_info"])
    keys = {
        (str(row["reward_model"]["bench"]), str(row["reward_model"]["task_id"]))
        for _, row in frame.iterrows()
    }
    assert len(frame) == len(keys) == expected, (path, len(frame), len(keys))
    assert all(float(extra["slate_contains_gold"]) == 1 for extra in frame["extra_info"]), path
    assert all(int(extra["slate_size"]) == 16 for extra in frame["extra_info"]), path
    assert all(len(extra["retrieval_skills_top_n"]) == 16 for extra in frame["extra_info"]), path
    assert all(
        sum(str(message.get("content", "")).count("<skill>") for message in prompt) == 16
        for prompt in frame["prompt"]
    ), path
print("PROFILE_DATA_OK mixed_task_reward FINAL train=491 Hybrid eval=56 V8-fixed4 slate16 all_gold frozen_hashes")
PY
    return
  fi

  local builder="${ROOT}/ops/workflows/rl_data_prep/make_4bench_mixed_skill_bonus_compare_parquet.py"
  local -a args=(
    --output-dir "${DATA_DIR}"
    --train-manifest "${SLATE_ROOT}/manifest/slate_manifest_train.jsonl"
    --eval-manifest "${SLATE_ROOT}/manifest/slate_manifest_eval70.jsonl"
    --skill-roots "${AGENT_BENCH_EXTRA_SKILL_ROOTS}"
    --expected-train-tasks 491
    --expected-eval-tasks 56
  )

  if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/eval.parquet" || ! -f "${DATA_DIR}/build_report.json" ]]; then
    [[ "${DRY_RUN:-0}" != "1" ]] || { echo "FATAL: mixed task-reward parquet missing during dry-run" >&2; return 2; }
    "${PYTHON_BIN:-python3}" "${builder}" "${args[@]}"
  fi
  "${PYTHON_BIN:-python3}" "${builder}" "${args[@]}" --validate-only \
    >"/tmp/rl_profile_mixed_task_reward_validate.json"
  "${PYTHON_BIN:-python3}" - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("/tmp/rl_profile_mixed_task_reward_validate.json").read_text())
train, evaluation = report["train"], report["eval"]
assert train["rows"] == train["unique_tasks"] == train["gold_present"] == train["slate_size_16"] == 491, train
assert evaluation["rows"] == evaluation["unique_tasks"] == evaluation["gold_present"] == evaluation["slate_size_16"] == 56, evaluation
assert train["gold_absent"] == evaluation["gold_absent"] == 0, report
print("PROFILE_DATA_OK mixed_task_reward train=491 eval=56 slate16 all_gold task_reward_only")
PY
}

rl_profile_validate_algorithm() {
  [[ "${AGENT_BENCH_RETRIEVAL_TOP_N}" == "16" && "${N_SAMPLES_PER_PROMPT}" == "8" ]] || {
    echo "FATAL: mixed task-reward control requires one 16-skill prompt and 8 rollouts per task" >&2
    return 2
  }
  [[ "${CUSTOM_REWARD_POST_PROCESS_PATH}" == "examples.agent_bench.skill_group_reward.post_process_rewards" \
     && "${DYNAMIC_SAMPLING_FILTER_PATH}" == "relax.engine.filters.dynamic_sampling_filters.check_reward_nonzero_std" ]] || {
    echo "FATAL: mixed task-reward control must use ordinary GRPO normalization and raw task-reward dynamic sampling" >&2
    return 2
  }
  local key
  for key in \
    RELAX_MIXED_SKILL_BONUS_ENABLED RELAX_MIXED_SEPARATED_ADV_ENABLED \
    RELAX_MIXED_SEPARATED_BEHAVIOR_COEF RELAX_SKILL_GROUP_REWARD \
    RELAX_SKILL_GROUP_BONUS_COEF RELAX_SKILL_GROUP_BONUS_MAX \
    RELAX_SKILL_GROUP_SUBGROUP_ADV_COEF RELAX_SKILL_GROUP_NO_READ_SUCCESS_BONUS \
    RELAX_SLATE_REGRET_GRPO RELAX_SLATE_STRATIFIED_ADVANTAGE \
    RELAX_PAIR_ATOMIC_SAMPLING RELAX_PAIR_SPECULATIVE_EXTRA_GROUPS; do
    [[ "${!key:-0}" == "0" ]] || {
      echo "FATAL: task-reward-only control requires ${key}=0, got ${!key}" >&2
      return 2
    }
  done
  [[ -z "${CUSTOM_LOSS_FUNCTION_PATH:-}" ]] || {
    echo "FATAL: task-reward-only control must not use a custom loss" >&2
    return 2
  }
  echo "PROFILE_ALGORITHM_OK mixed_task_reward ordinary_task_grpo no_behavior_signal"
}
