#!/usr/bin/env bash
# No-skill 4bench GRPO. Training/eval parquet contains no skill prompt or files.

rl_profile_configure() {
  export RL_PROFILE=no_skill
  export EXPERIMENT_BASENAME="${EXPERIMENT_BASENAME:-no-skill-rl}"
  export RL_RUN_PURPOSE="${RL_RUN_PURPOSE:-No-skill 4bench GRPO from the 9B SFT model; no advertised or injected skills and no skill-specific reward.}"
  export RELAX_CONTEXT_DECISION="${RELAX_CONTEXT_DECISION:-no_skill_actor8_tp4_cp2_70k_lr1e6_active128_localdocker}"
  export DATA_DIR="${DATA_DIR:-${ROOT}/datasets/rl/parquet_4bench_factual_noskills_20260617}"

  export AGENT_BENCH_RETRIEVAL_TOP_N=0
  unset AGENT_BENCH_EXTRA_SKILL_ROOTS 2>/dev/null || true
  export LOSS_TYPE=policy_loss
  export CUSTOM_REWARD_POST_PROCESS_PATH=examples.agent_bench.skill_group_reward.post_process_rewards
  export DYNAMIC_SAMPLING_FILTER_PATH=relax.engine.filters.dynamic_sampling_filters.check_reward_nonzero_std
  export RELAX_DISABLE_TIS=0
  export RELAX_SKILL_GROUP_REWARD=0
  export RELAX_SKILL_GROUP_BONUS_COEF=0
  export RELAX_SKILL_GROUP_BONUS_MAX=0
  export RELAX_SKILL_GROUP_MARGIN=0.0
  export RELAX_SKILL_GROUP_SUBGROUP_ADV_COEF=0
  export RELAX_SKILL_GROUP_REQUIRE_BOTH=0
  export RELAX_SKILL_GROUP_NO_READ_SUCCESS_BONUS=0
  export RELAX_DYNAMIC_FILTER_MIN_SKILL_READ_FRAC=0
  export RELAX_DYNAMIC_FILTER_MIN_NO_SKILL_READ_FRAC=0
  export RELAX_DYNAMIC_FILTER_SKILL_READ_MAX_SAMPLES=0
  export RELAX_DYNAMIC_FILTER_MAX_REJECTS_PER_ROLLOUT="${RELAX_DYNAMIC_FILTER_MAX_REJECTS_PER_ROLLOUT:-64}"
  export RELAX_DYNAMIC_FILTER_MAX_REJECT_SAMPLES_PER_ROLLOUT="${RELAX_DYNAMIC_FILTER_MAX_REJECT_SAMPLES_PER_ROLLOUT:-300}"

  # The training recipe is unchanged; the maintained checkpoint policy adds a
  # no-skill internal eval so both the last and the best checkpoint survive.
  export DISABLE_EVAL=0
  export EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
  export SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-1}"
  export KEEP_BEST_ACTOR_CKPT="${KEEP_BEST_ACTOR_CKPT:-1}"
}

rl_profile_prepare() {
  local builder="${ROOT}/ops/workflows/rl_data_prep/make_4bench_factual_noskill_parquet.py"
  if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/eval.parquet" ]]; then
    [[ "${DRY_RUN:-0}" != "1" ]] || { echo "FATAL: no-skill parquet missing during dry-run: ${DATA_DIR}" >&2; return 2; }
    "${PYTHON_BIN:-python3}" "${builder}" \
      --input-dir "${ROOT}/datasets/rl/parquet_4bench_factual_20260602" \
      --output-dir "${DATA_DIR}"
  fi
  "${PYTHON_BIN:-python3}" "${builder}" --output-dir "${DATA_DIR}" --validate-only \
    >"/tmp/rl_profile_no_skill_validate.json"
  "${PYTHON_BIN:-python3}" - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("/tmp/rl_profile_no_skill_validate.json").read_text())
assert report["files"]["train"]["rows"] == 491, report
assert report["files"]["eval"]["rows"] == 56, report
for split in ("train", "eval"):
    item = report["files"][split]
    assert item["nonempty_retrieval_after"] == 0, item
    assert item["residual_skill_sections"] == 0, item
    assert item["residual_available_skill_blocks"] == 0, item
    assert not item["problems"], item
print("PROFILE_DATA_OK no_skill train=491 eval=56 no_skill_payloads")
PY
}
