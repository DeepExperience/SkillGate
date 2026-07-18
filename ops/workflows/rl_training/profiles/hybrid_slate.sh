#!/usr/bin/env bash
# Hybrid SlateRL v2: paired no-skill/mixed arms with regret and selector strata.

rl_profile_configure() {
  export RL_PROFILE=hybrid_slate
  export EXPERIMENT_BASENAME="${EXPERIMENT_BASENAME:-hybrid-slate-v2}"
  export RL_RUN_PURPOSE="${RL_RUN_PURPOSE:-Hybrid SlateRL v2 using v8 misleading bodies and separable 0704 descriptions; paired regret plus strict-read stratified advantage.}"
  export RELAX_CONTEXT_DECISION="${RELAX_CONTEXT_DECISION:-hybrid_slate_actor8_tp4_cp2_70k_lr1e6_active128_pair_spec8}"

  export HYBRID_ROOT="${HYBRID_ROOT:-${ROOT}/skill_libraries/snapshots/rl/slate_skills_20260710_hybrid_v8body_0704desc}"
  export SLATE_FALLBACK_ROOT="${SLATE_FALLBACK_ROOT:-${ROOT}/skill_libraries/snapshots/rl/slate_skills_20260704}"
  export SLATE_TRAIN_MANIFEST="${SLATE_TRAIN_MANIFEST:-${HYBRID_ROOT}/manifest/slate_manifest_train.jsonl}"
  export SLATE_EVAL_MANIFEST="${SLATE_EVAL_MANIFEST:-${HYBRID_ROOT}/manifest/slate_manifest_eval70.jsonl}"
  export AGENT_BENCH_EXTRA_SKILL_ROOTS="${AGENT_BENCH_EXTRA_SKILL_ROOTS:-${HYBRID_ROOT}/skills:${SLATE_FALLBACK_ROOT}/skills}"
  export AGENT_BENCH_RETRIEVAL_TOP_N=16
  export DATA_DIR="${DATA_DIR:-${ROOT}/datasets/rl/parquet_4bench_slate_regret_hybridv8b0704d_gold_stratified_20260710}"

  export RELAX_SLATE_REGRET_GRPO=1
  export RELAX_SLATE_UNIFORM_MIN_DELTA="${RELAX_SLATE_UNIFORM_MIN_DELTA:-0.25}"
  export RELAX_SLATE_REGRET_COEF="${RELAX_SLATE_REGRET_COEF:-0.5}"
  export RELAX_SLATE_STRATIFIED_ADVANTAGE=1
  export RELAX_SLATE_STRATIFIED_ADV_COEF="${RELAX_SLATE_STRATIFIED_ADV_COEF:-1.0}"
  export RELAX_SLATE_STRATIFIED_SHRINKAGE="${RELAX_SLATE_STRATIFIED_SHRINKAGE:-1.0}"
  export RELAX_SLATE_STRATIFIED_ADV_CLIP="${RELAX_SLATE_STRATIFIED_ADV_CLIP:-0.5}"
  export LOSS_TYPE=custom_loss
  export CUSTOM_LOSS_FUNCTION_PATH=examples.agent_bench.hybrid_shadow_grpo_loss.hybrid_shadow_grpo_loss
  export DYNAMIC_SAMPLING_FILTER_PATH=examples.agent_bench.hybrid_pair_gating.keep_pair_candidate_groups
  export CUSTOM_REWARD_POST_PROCESS_PATH=examples.agent_bench.slate_regret_stratified_gating.post_process_rewards
  export RELAX_DISABLE_TIS=1
  export RELAX_PAIR_ATOMIC_SAMPLING=1
  export RELAX_PAIR_SPECULATIVE_EXTRA_GROUPS="${RL_PAIR_SPECULATIVE_EXTRA_GROUPS:-8}"
  export RELAX_PAIR_BC_PASS_THRESHOLD="${RELAX_PAIR_BC_PASS_THRESHOLD:-1.0}"
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
  local required
  for required in \
    "${SLATE_TRAIN_MANIFEST}" \
    "${SLATE_EVAL_MANIFEST}" \
    "${HYBRID_ROOT}/manifest/hybrid_build_report.json"; do
    [[ -f "${required}" ]] || { echo "FATAL: hybrid slate artifact missing: ${required}" >&2; return 2; }
  done

  local builder="${ROOT}/ops/workflows/rl_data_prep/make_4bench_slate_regret_gold_v2_parquet.py"
  local -a args=(
    --manifest "${SLATE_TRAIN_MANIFEST}"
    --eval-manifest "${SLATE_EVAL_MANIFEST}"
    --skill-roots "${AGENT_BENCH_EXTRA_SKILL_ROOTS}"
    --output-dir "${DATA_DIR}"
  )
  if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/eval.parquet" ]]; then
    [[ "${DRY_RUN:-0}" != "1" ]] || { echo "FATAL: hybrid parquet missing during dry-run: ${DATA_DIR}" >&2; return 2; }
    "${PYTHON_BIN:-python3}" "${builder}" "${args[@]}"
  fi
  "${PYTHON_BIN:-python3}" "${builder}" "${args[@]}" --validate-only \
    >"/tmp/rl_profile_hybrid_slate_validate.json"
  "${PYTHON_BIN:-python3}" "${ROOT}/ops/workflows/rl_training/tools/smoke_slate_regret_stratified_v2.py" \
    >"/tmp/rl_profile_hybrid_slate_smoke.json"
}
