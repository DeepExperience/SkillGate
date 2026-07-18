#!/usr/bin/env bash
# NOTE: Migrated canonical workflow copy. Source: ops/launch/run_phase2_teacher_worker.sh
# Original historical script is archived during workflow cleanup; maintain this copy going forward.
# Standalone phase2 teacher-reflection worker.
#
# Pulls *.jsonl chunks from queue/, launches launch_trials.py against the
# configured teacher endpoint, and moves chunks to done/failed.
#
# Why standalone: the in-wrapper phase2_worker() in run_sft_pipeline.sh dies
# whenever the wrapper itself exits. This script lets phase2 keep consuming
# queue independently without restarting phase1.
#
# Usage:
#   bash ops/workflows/sft_data_collection/run_phase2_teacher_worker.sh \
#     experiments/sft_skill_use_campaign/runs/<RUN_ID>
#
# Env overrides:
#   TEACHER_MODEL, TEACHER_OPENAI_API_BASE, PHASE2_WORKERS, TEACHER_TIMEOUT,
#   PHASE2_BENCH_CAPS, DOCKER_WAIT_SEC

set -uo pipefail

PROJECT_ROOT="${SKILLRL_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

RUN_ROOT="${1:?run_root path required (e.g. experiments/sft_skill_use_campaign/runs/<id>)}"
[[ -d "${RUN_ROOT}" ]] || { echo "RUN_ROOT not found: ${RUN_ROOT}"; exit 2; }

TEACHER_MODEL="${TEACHER_MODEL:-qwen3.5-27b}"
TEACHER_OPENAI_API_BASE="${TEACHER_OPENAI_API_BASE:-http://127.0.0.1:30000/v1}"
PHASE2_WORKERS="${PHASE2_WORKERS:-6}"
TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-3000}"
PHASE2_BENCH_CAPS="${PHASE2_BENCH_CAPS:-claw=1,tb2=2,sb_ns=2,seta_synth=3,swe_lite=1}"
DOCKER_WAIT_SEC="${DOCKER_WAIT_SEC:-54000}"
DOCKER_CHECK_INTERVAL_SEC="${DOCKER_CHECK_INTERVAL_SEC:-60}"

QUEUE_DIR="${RUN_ROOT}/plans/chunks/queue"
RUNNING_DIR="${RUN_ROOT}/plans/chunks/running"
DONE_DIR="${RUN_ROOT}/plans/chunks/done"
FAILED_DIR="${RUN_ROOT}/plans/chunks/failed"
LOG_DIR="${RUN_ROOT}/logs/runner"
mkdir -p "${RUNNING_DIR}" "${DONE_DIR}" "${FAILED_DIR}" "${LOG_DIR}"

CAP_ARGS=()
IFS=',' read -r -a _caps <<< "${PHASE2_BENCH_CAPS}"
for c in "${_caps[@]}"; do
  [[ -n "${c}" ]] && CAP_ARGS+=(--bench-cap "${c}")
done

if [[ -f "secrets/.env.secrets" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "secrets/.env.secrets"
  set +a
fi

echo "[phase2] start RUN_ROOT=${RUN_ROOT}"
echo "[phase2] teacher=${TEACHER_MODEL} api=${TEACHER_OPENAI_API_BASE}"
echo "[phase2] workers=${PHASE2_WORKERS} timeout=${TEACHER_TIMEOUT} caps=${PHASE2_BENCH_CAPS}"
echo "[phase2] queue=${QUEUE_DIR}"

idle_loops=0
while true; do
  next="$(find "${QUEUE_DIR}" -maxdepth 1 -type f -name '*.jsonl' | sort | head -1 || true)"
  if [[ -z "${next}" ]]; then
    if [[ -f "${QUEUE_DIR}/STOP" ]]; then
      echo "[phase2] queue empty + STOP marker; exit cleanly"
      exit 0
    fi
    idle_loops=$((idle_loops + 1))
    if (( idle_loops % 10 == 1 )); then
      echo "[phase2] idle (no chunks in queue) loop=${idle_loops}"
    fi
    sleep 30
    continue
  fi
  idle_loops=0

  base="$(basename "${next}")"
  running="${RUNNING_DIR}/${base}"
  done="${DONE_DIR}/${base}"
  failed="${FAILED_DIR}/${base}"

  if [[ -e "${done}" ]]; then
    echo "[phase2] skip ${base}: already in done/"
    rm -f "${next}"
    continue
  fi

  echo "[phase2] start chunk: ${base}"
  mv "${next}" "${running}"

  set +e
  EXPERIMENT_ROOT="${RUN_ROOT}" \
    OPENAI_API_BASE="${TEACHER_OPENAI_API_BASE}" \
    python3 GeneralAgent/sft_data_collection/launch_trials.py \
      --plan "${running}" \
      --model "${TEACHER_MODEL}" \
      --api-base-override "${TEACHER_OPENAI_API_BASE}" \
      --workers "${PHASE2_WORKERS}" \
      "${CAP_ARGS[@]}" \
      --per-trial-timeout-sec "${TEACHER_TIMEOUT}" \
      --docker-wait-sec "${DOCKER_WAIT_SEC}" \
      --docker-check-interval-sec "${DOCKER_CHECK_INTERVAL_SEC}" \
      --skip-mysql-cleanup \
      --execute 2>&1 | tee -a "${LOG_DIR}/phase2_standalone.log"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ ${rc} -eq 0 ]]; then
    mv "${running}" "${done}"
    echo "[phase2] done chunk: ${base}"
  else
    mv "${running}" "${failed}"
    echo "[phase2] FAILED chunk: ${base} rc=${rc}; continuing to next"
  fi
done
