#!/usr/bin/env bash
# NOTE: Migrated canonical workflow copy. Source: ops/launch/run_sft_qwen27b_campaign_pipeline.sh
# Original historical script is archived during workflow cleanup; maintain this copy going forward.
set -Eeuo pipefail

PROJECT_ROOT="${SKILLRL_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

export CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-experiments/sft_skill_use_campaign}"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M)_qwen27b_campaign_run}"
export RUN_ROOT="${RUN_ROOT:-${CAMPAIGN_ROOT}/runs/${RUN_ID}}"
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${RUN_ROOT}}"

export CONFIG="${CONFIG:-GeneralAgent/sft_data_collection/configs/qwen27b_use_skill_collection_config.json}"
export MODEL="${MODEL:-qwen3.5-27b}"
export TEACHER_MODEL="${TEACHER_MODEL:-qwen3.5-27b}"

export STUDENT_OPENAI_API_BASE="${STUDENT_OPENAI_API_BASE:-http://127.0.0.1:30000/v1}"
export TEACHER_OPENAI_API_BASE="${TEACHER_OPENAI_API_BASE:-http://127.0.0.1:30001/v1}"
export STUDENT_OPENAI_API_KEY="${STUDENT_OPENAI_API_KEY:-sk-local-anything}"
export TEACHER_OPENAI_API_KEY="${TEACHER_OPENAI_API_KEY:-sk-local-anything}"

export CHUNK_TASKS="${CHUNK_TASKS:-8}"
export PHASE1_WORKERS="${PHASE1_WORKERS:-10}"
export PHASE2_WORKERS="${PHASE2_WORKERS:-6}"
export PHASE1_STREAMING="${PHASE1_STREAMING:-1}"
export PHASE1_TASK_WINDOW="${PHASE1_TASK_WINDOW:-24}"
export TIMEOUT="${TIMEOUT:-3000}"
export TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-3000}"
export TEACHER_TRIALS="${TEACHER_TRIALS:-8}"
export SKIP_STALE_CLEANUP="${SKIP_STALE_CLEANUP:-0}"

export PHASE1_BENCH_CAPS="${PHASE1_BENCH_CAPS:-claw=1,tb2=1,sb_ns=3,seta_synth=4,swe_lite=2}"
export PHASE2_BENCH_CAPS="${PHASE2_BENCH_CAPS:-claw=1,tb2=1,sb_ns=2,seta_synth=3,swe_lite=1}"

export UNIFIED_LLM_MAX_RETRIES="${UNIFIED_LLM_MAX_RETRIES:-4}"
export UNIFIED_LLM_RETRY_BACKOFF_SEC="${UNIFIED_LLM_RETRY_BACKOFF_SEC:-3}"
export UNIFIED_LLM_RETRY_MAX_BACKOFF_SEC="${UNIFIED_LLM_RETRY_MAX_BACKOFF_SEC:-30}"
export UNIFIED_LLM_REQUEST_TIMEOUT_SEC="${UNIFIED_LLM_REQUEST_TIMEOUT_SEC:-300}"
export UNIFIED_LLM_RETRY_HTTP_STATUSES="${UNIFIED_LLM_RETRY_HTTP_STATUSES:-408,409,425,429,500,502,503,504,529}"

export CAMPAIGN_SUCCESS_POLICY="${CAMPAIGN_SUCCESS_POLICY:-strict_used_non_meta}"

if [[ -z "${CAMPAIGN_SOURCE_RUNS:-}" ]]; then
  source_runs=(
    "experiments/20260427/20260427_0847_sft_pipeline_full"
    "experiments/20260429/20260429_0837_sft_glm51_strict_prompt_streaming_full"
  )
  if [[ -d "${CAMPAIGN_ROOT}/runs" ]]; then
    while IFS= read -r previous_run; do
      [[ "${previous_run}" == "${RUN_ROOT}" ]] && continue
      source_runs+=("${previous_run}")
    done < <(find "${CAMPAIGN_ROOT}/runs" -mindepth 1 -maxdepth 1 -type d | sort)
  fi
  export CAMPAIGN_SOURCE_RUNS="${source_runs[*]}"
fi

exec bash ops/workflows/sft_data_collection/run_sft_pipeline.sh
