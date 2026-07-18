#!/usr/bin/env bash
# NOTE: Migrated canonical workflow copy. Source: ops/launch/run_sft_pipeline.sh
# Original historical script is archived during workflow cleanup; maintain this copy going forward.
set -Eeuo pipefail

PROJECT_ROOT="${SKILLRL_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"
if [[ -f "secrets/.env.secrets" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "secrets/.env.secrets"
  set +a
fi

run_date_from_id() {
  if [[ "$1" =~ (20[0-9]{6}) ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    date -u +%Y%m%d
  fi
}

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M)_sft_pipeline}"
DATE="${DATE:-$(run_date_from_id "${RUN_ID}")}"
RUN_ROOT="${EXPERIMENT_ROOT:-${RUN_ROOT:-experiments/${DATE}/${RUN_ID}}}"
CONFIG="${CONFIG:-GeneralAgent/sft_data_collection/configs/default_collection_config.json}"
SPLITS="${SPLITS:-GeneralAgent/sft_data_collection/outputs/splits/default/holdout_split.json}"
MODEL="${MODEL:-qwen3.5-9b}"
TEACHER_MODEL="${TEACHER_MODEL:-glm-5.1}"
STUDENT_OPENAI_API_BASE="${STUDENT_OPENAI_API_BASE:-${OPENAI_API_BASE:-http://127.0.0.1:30000/v1}}"
STUDENT_OPENAI_API_KEY="${STUDENT_OPENAI_API_KEY:-${OPENAI_API_KEY:-}}"
TEACHER_OPENAI_API_BASE="${TEACHER_OPENAI_API_BASE:-${MAAS_API_BASE:-}}"
TEACHER_OPENAI_API_KEY="${TEACHER_OPENAI_API_KEY:-${MAAS_API_KEY:-}}"
if [[ -z "${STUDENT_OPENAI_API_KEY}" && -n "${MAAS_API_BASE:-}" && "${STUDENT_OPENAI_API_BASE%/}" == "${MAAS_API_BASE%/}" ]]; then
  STUDENT_OPENAI_API_KEY="${MAAS_API_KEY:-}"
fi
TEACHER_LLM_MAX_RETRIES="${TEACHER_LLM_MAX_RETRIES:-8}"
TEACHER_LLM_RETRY_BACKOFF_SEC="${TEACHER_LLM_RETRY_BACKOFF_SEC:-5}"
TEACHER_LLM_RETRY_MAX_BACKOFF_SEC="${TEACHER_LLM_RETRY_MAX_BACKOFF_SEC:-60}"
TEACHER_LLM_REQUEST_TIMEOUT_SEC="${TEACHER_LLM_REQUEST_TIMEOUT_SEC:-300}"
TEACHER_LLM_RETRY_HTTP_STATUSES="${TEACHER_LLM_RETRY_HTTP_STATUSES:-408,409,425,429,500,502,503,504,529}"
export TEACHER_LLM_MAX_RETRIES TEACHER_LLM_RETRY_BACKOFF_SEC \
  TEACHER_LLM_RETRY_MAX_BACKOFF_SEC TEACHER_LLM_REQUEST_TIMEOUT_SEC \
  TEACHER_LLM_RETRY_HTTP_STATUSES
PILOT_PER_BENCH="${PILOT_PER_BENCH:-0}"
CHUNK_TASKS="${CHUNK_TASKS:-10}"
PHASE1_WORKERS="${PHASE1_WORKERS:-8}"
PHASE2_WORKERS="${PHASE2_WORKERS:-2}"
PHASE2_BATCH_TASKS="${PHASE2_BATCH_TASKS:-1}"
PHASE1_BENCH_CAPS="${PHASE1_BENCH_CAPS:-}"
PHASE2_BENCH_CAPS="${PHASE2_BENCH_CAPS:-}"
PHASE1_TASK_WINDOW="${PHASE1_TASK_WINDOW:-32}"
TIMEOUT="${TIMEOUT:-2400}"
TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-3000}"
TEACHER_TRIALS="${TEACHER_TRIALS:-4}"
DOCKER_WAIT_SEC="${DOCKER_WAIT_SEC:-54000}"
DOCKER_CHECK_INTERVAL_SEC="${DOCKER_CHECK_INTERVAL_SEC:-60}"
PHASE1_STREAMING="${PHASE1_STREAMING:-0}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-}"
CAMPAIGN_SOURCE_RUNS="${CAMPAIGN_SOURCE_RUNS:-}"
CAMPAIGN_SUCCESS_POLICY="${CAMPAIGN_SUCCESS_POLICY:-strict_used_non_meta}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${BENCHES:-}" ]]; then
  # shellcheck disable=SC2206
  BENCH_ARGS=(${BENCHES})
else
  BENCH_ARGS=(claw tb2 sb_ns seta_synth swe_lite)
fi

PLAN="${RUN_ROOT}/plans/${RUN_ID}.jsonl"
CHUNK_ROOT="${RUN_ROOT}/plans/chunks"
PHASE1_CHUNK_DIR="${CHUNK_ROOT}/phase1"
REFLECTION_DIR="${CHUNK_ROOT}/teacher"
QUEUE_DIR="${CHUNK_ROOT}/queue"
RUNNING_DIR="${CHUNK_ROOT}/running"
DONE_DIR="${CHUNK_ROOT}/done"
FAILED_DIR="${CHUNK_ROOT}/failed"
COMBINED_PLAN="${RUN_ROOT}/plans/${RUN_ID}.combined.jsonl"

LOG_ROOT="${RUN_ROOT}/logs/runner"
REPORT_PATH="${RUN_ROOT}/reports/${RUN_ID}_sft_pipeline.md"
MASTER_LOG="${LOG_ROOT}/run.log"
EVENTS_JSONL="${LOG_ROOT}/events.jsonl"

mkdir -p "${LOG_ROOT}" "${PHASE1_CHUNK_DIR}" "${REFLECTION_DIR}" \
  "${QUEUE_DIR}" "${RUNNING_DIR}" "${DONE_DIR}" "${FAILED_DIR}" "$(dirname "${REPORT_PATH}")"
exec > >(tee -a "${MASTER_LOG}") 2>&1

PHASE1_BENCH_CAP_ARGS=()
if [[ -n "${PHASE1_BENCH_CAPS}" ]]; then
  IFS=',' read -r -a _phase1_caps <<< "${PHASE1_BENCH_CAPS}"
  for cap in "${_phase1_caps[@]}"; do
    [[ -n "${cap}" ]] && PHASE1_BENCH_CAP_ARGS+=(--bench-cap "${cap}")
  done
fi

PHASE2_BENCH_CAP_ARGS=()
if [[ -n "${PHASE2_BENCH_CAPS}" ]]; then
  IFS=',' read -r -a _phase2_caps <<< "${PHASE2_BENCH_CAPS}"
  for cap in "${_phase2_caps[@]}"; do
    [[ -n "${cap}" ]] && PHASE2_BENCH_CAP_ARGS+=(--bench-cap "${cap}")
  done
fi

json_event() {
  local event="$1"
  local detail="${2:-}"
  python3 - "$EVENTS_JSONL" "$event" "$detail" <<'PY'
import json
import sys
from datetime import datetime, timezone
path, event, detail = sys.argv[1:4]
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "detail": detail,
    }, ensure_ascii=False) + "\n")
PY
}

line_count() {
  local path="$1"
  if [[ -s "${path}" ]]; then
    wc -l < "${path}" | tr -d ' '
  else
    echo 0
  fi
}

run_logged() {
  local name="$1"
  shift
  local log_file="${LOG_ROOT}/${name}.log"
  json_event "step_start" "${name}"
  echo
  echo "===== ${name} ====="
  echo "command: $*"
  set +e
  "$@" 2>&1 | tee "${log_file}"
  local rc=${PIPESTATUS[0]}
  set -e
  json_event "step_finish" "${name} rc=${rc}"
  if [[ "${rc}" -ne 0 ]]; then
    echo "step failed: ${name} rc=${rc}"
    return "${rc}"
  fi
}

served_model() {
  local api_base="$1"
  local api_key="${2:-}"
  if [[ -n "${api_key}" ]]; then
    curl -sS --max-time 10 -H "Authorization: Bearer ${api_key}" "${api_base%/}/models" 2>/dev/null
  else
    curl -sS --max-time 10 "${api_base%/}/models" 2>/dev/null
  fi \
    | python3 -c 'import json,sys; print("\n".join(str(x.get("id","")) for x in (json.load(sys.stdin).get("data") or []) if isinstance(x, dict)))' \
      2>/dev/null || true
}

assert_endpoint_model() {
  local api_base="$1"
  local expected="$2"
  local api_key="${3:-}"
  local models
  models="$(served_model "${api_base}" "${api_key}")"
  if ! grep -Fxq "${expected}" <<< "${models}"; then
    echo "endpoint ${api_base} does not list expected model ${expected}"
    echo "available models:"
    sed -n '1,20p' <<< "${models}"
    return 1
  fi
}

refresh_existing_teacher_queue() {
  local queue_file base phase1_file tmp
  for queue_file in "${QUEUE_DIR}"/*.teacher.jsonl "${RUNNING_DIR}"/*.teacher.jsonl; do
    [[ -e "${queue_file}" ]] || continue
    if python3 - "$queue_file" "$TEACHER_MODEL" "$TEACHER_OPENAI_API_BASE" \
      "$TEACHER_LLM_MAX_RETRIES" "$TEACHER_LLM_RETRY_BACKOFF_SEC" \
      "$TEACHER_LLM_RETRY_MAX_BACKOFF_SEC" "$TEACHER_LLM_REQUEST_TIMEOUT_SEC" \
      "$TEACHER_LLM_RETRY_HTTP_STATUSES" <<'PY'
import json, sys
path, expected_model, expected_base = sys.argv[1:4]
expected_env = {
    "UNIFIED_LLM_MAX_RETRIES": sys.argv[4],
    "UNIFIED_LLM_RETRY_BACKOFF_SEC": sys.argv[5],
    "UNIFIED_LLM_RETRY_MAX_BACKOFF_SEC": sys.argv[6],
    "UNIFIED_LLM_REQUEST_TIMEOUT_SEC": sys.argv[7],
    "UNIFIED_LLM_RETRY_HTTP_STATUSES": sys.argv[8],
}
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
if not rows:
    raise SystemExit(0)
bad = [
    row.get("model") != expected_model
    or (row.get("env") or {}).get("OPENAI_API_BASE") != expected_base
    or any((row.get("env") or {}).get(key) != value for key, value in expected_env.items())
    for row in rows
]
raise SystemExit(1 if any(bad) else 0)
PY
    then
      continue
    fi
    base="$(basename "${queue_file}" .teacher.jsonl)"
    phase1_file="${PHASE1_CHUNK_DIR}/${base}.jsonl"
    if [[ ! -f "${phase1_file}" ]]; then
      echo "cannot refresh ${queue_file}: missing ${phase1_file}"
      return 1
    fi
    tmp="${queue_file}.tmp"
    echo "refresh existing teacher queue ${queue_file}: model/base -> ${TEACHER_MODEL} @ ${TEACHER_OPENAI_API_BASE}"
    env "TEACHER_OPENAI_API_BASE=${TEACHER_OPENAI_API_BASE}" "EXPERIMENT_ROOT=${RUN_ROOT}" \
      python3 GeneralAgent/sft_data_collection/make_teacher_fallback_plan.py \
        --config "${CONFIG}" \
        --plan "${phase1_file}" \
        --out "${tmp}" \
        --teacher-model "${TEACHER_MODEL}" \
        --teacher-trials "${TEACHER_TRIALS}"
    mv "${tmp}" "${queue_file}"
  done
}

phase2_worker() {
  json_event "phase2_worker_start" "workers=${PHASE2_WORKERS}"
  local batch_dir="${RUNNING_DIR}/.batches"
  local batch_seq=0
  mkdir -p "${batch_dir}"
  while true; do
    local claimed_files=()
    local next base running
    while IFS= read -r next; do
      [[ -n "${next}" ]] || continue
      base="$(basename "${next}")"
      running="${RUNNING_DIR}/${base}"
      if mv "${next}" "${running}" 2>/dev/null; then
        claimed_files+=("${running}")
      fi
      if [[ "${#claimed_files[@]}" -ge "${PHASE2_BATCH_TASKS}" ]]; then
        break
      fi
    done < <(find "${QUEUE_DIR}" -maxdepth 1 -type f -name '*.jsonl' | sort)

    if [[ "${#claimed_files[@]}" -eq 0 ]]; then
      if [[ -f "${QUEUE_DIR}/STOP" ]]; then
        json_event "phase2_worker_stop" "queue_empty"
        return 0
      fi
      sleep 10
      continue
    fi

    batch_seq=$((batch_seq + 1))
    local batch_base="phase2_batch_${batch_seq}_$(date -u +%Y%m%dT%H%M%SZ)"
    local batch_plan="${batch_dir}/${batch_base}.jsonl"
    : > "${batch_plan}"
    local claimed_file
    for claimed_file in "${claimed_files[@]}"; do
      cat "${claimed_file}" >> "${batch_plan}"
    done

    json_event "phase2_chunk_start" "${batch_base} files=${#claimed_files[@]} records=$(line_count "${batch_plan}")"
    echo
    echo "===== phase2_${batch_base} files=${#claimed_files[@]} ====="
    set +e
    EXPERIMENT_ROOT="${RUN_ROOT}" python3 GeneralAgent/sft_data_collection/launch_trials.py \
      --plan "${batch_plan}" \
      --model "${TEACHER_MODEL}" \
      --workers "${PHASE2_WORKERS}" \
      "${PHASE2_BENCH_CAP_ARGS[@]}" \
      --per-trial-timeout-sec "${TEACHER_TIMEOUT}" \
      --docker-wait-sec "${DOCKER_WAIT_SEC}" \
      --docker-check-interval-sec "${DOCKER_CHECK_INTERVAL_SEC}" \
      --execute \
      2>&1 | tee "${LOG_ROOT}/phase2_${batch_base}.log"
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ "${rc}" -eq 0 ]]; then
      for claimed_file in "${claimed_files[@]}"; do
        mv "${claimed_file}" "${DONE_DIR}/$(basename "${claimed_file}")"
      done
      rm -f "${batch_plan}"
      json_event "phase2_chunk_finish" "${batch_base} rc=0 files=${#claimed_files[@]}"
    else
      for claimed_file in "${claimed_files[@]}"; do
        mv "${claimed_file}" "${FAILED_DIR}/$(basename "${claimed_file}")"
      done
      rm -f "${batch_plan}"
      json_event "phase2_chunk_finish" "${batch_base} rc=${rc} files=${#claimed_files[@]}"
      return "${rc}"
    fi
  done
}

write_report() {
  python3 - "$RUN_ID" "$PLAN" "$COMBINED_PLAN" "$REPORT_PATH" "$RUN_ROOT" <<'PY'
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone

run_id, plan_path, combined_path, report_path, run_root_path = sys.argv[1:6]
root = Path(os.environ.get("SKILLRL_ROOT") or os.environ.get("PROJECT_ROOT") or "/path/to/skillRL")
run_root = Path(run_root_path)
if not run_root.is_absolute():
    run_root = root / run_root
status_path = run_root / "logs/sft_collection/status.jsonl"
collected = run_root / "collected"

def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]

plan_rows = read_jsonl(root / plan_path)
combined_rows = read_jsonl(root / combined_path)
status_rows = read_jsonl(status_path)
by = defaultdict(Counter)
for row in status_rows:
    by[(row.get("bench", ""), row.get("mode", ""), row.get("model", ""))]["n"] += 1
    by[(row.get("bench", ""), row.get("mode", ""), row.get("model", ""))][str(row.get("returncode"))] += 1

lines = [
    f"# SFT Pipelined Report: {run_id}",
    "",
    f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
    f"- phase1_plan_records: {len(plan_rows)}",
    f"- combined_plan_records: {len(combined_rows)}",
    f"- status_rows: {len(status_rows)}",
    f"- sft_messages: {sum(1 for _ in (collected / 'sft_messages.jsonl').open(encoding='utf-8')) if (collected / 'sft_messages.jsonl').exists() else 0}",
    "",
    "## Status by bench/mode/model",
    "",
    "| bench | mode | model | completed | rc=0 | rc=-9 |",
    "|---|---|---|---:|---:|---:|",
]
for (bench, mode, model), counter in sorted(by.items()):
    lines.append(f"| {bench} | {mode} | {model} | {counter['n']} | {counter['0']} | {counter['-9']} |")
summary = collected / "summary.md"
if summary.exists():
    lines += ["", "## Collector Summary Head", ""]
    lines += summary.read_text(encoding="utf-8", errors="ignore").splitlines()[:80]
Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"report={report_path}")
PY
}

on_error() {
  local rc=$?
  json_event "wrapper_error" "rc=${rc}"
  touch "${QUEUE_DIR}/STOP" 2>/dev/null || true
  if [[ -n "${PHASE2_PID:-}" ]]; then
    wait "${PHASE2_PID}" 2>/dev/null || true
  fi
  write_report || true
  echo "pipeline failed rc=${rc}; report: ${REPORT_PATH}"
  exit "${rc}"
}
trap on_error ERR

json_event "wrapper_start" "run_id=${RUN_ID}"

echo "RUN_ID=${RUN_ID}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "CONFIG=${CONFIG}"
echo "SPLITS=${SPLITS}"
echo "BENCHES=${BENCH_ARGS[*]}"
echo "STUDENT_OPENAI_API_BASE=${STUDENT_OPENAI_API_BASE}"
echo "TEACHER_OPENAI_API_BASE=${TEACHER_OPENAI_API_BASE}"
echo "CHUNK_TASKS=${CHUNK_TASKS} PHASE1_WORKERS=${PHASE1_WORKERS} PHASE2_WORKERS=${PHASE2_WORKERS}"
echo "PHASE2_BATCH_TASKS=${PHASE2_BATCH_TASKS}"
echo "PHASE1_BENCH_CAPS=${PHASE1_BENCH_CAPS:-<none>} PHASE2_BENCH_CAPS=${PHASE2_BENCH_CAPS:-<none>}"
echo "PHASE1_TASK_WINDOW=${PHASE1_TASK_WINDOW}"
echo "TIMEOUT=${TIMEOUT} TEACHER_TIMEOUT=${TEACHER_TIMEOUT} TEACHER_TRIALS=${TEACHER_TRIALS}"
echo "DOCKER_WAIT_SEC=${DOCKER_WAIT_SEC} DOCKER_CHECK_INTERVAL_SEC=${DOCKER_CHECK_INTERVAL_SEC}"
echo "PHASE1_STREAMING=${PHASE1_STREAMING}"
if [[ -n "${CAMPAIGN_ROOT}" ]]; then
  echo "CAMPAIGN_ROOT=${CAMPAIGN_ROOT}"
  echo "CAMPAIGN_SUCCESS_POLICY=${CAMPAIGN_SUCCESS_POLICY}"
  echo "CAMPAIGN_SOURCE_RUNS=${CAMPAIGN_SOURCE_RUNS:-<none>}"
fi
echo "TEACHER_LLM_MAX_RETRIES=${TEACHER_LLM_MAX_RETRIES} TEACHER_LLM_RETRY_BACKOFF_SEC=${TEACHER_LLM_RETRY_BACKOFF_SEC}"
echo "REPORT_PATH=${REPORT_PATH}"

if [[ -z "${TEACHER_OPENAI_API_BASE}" ]]; then
  echo "TEACHER_OPENAI_API_BASE is empty; set it or provide MAAS_API_BASE in secrets/.env.secrets"
  exit 2
fi
assert_endpoint_model "${STUDENT_OPENAI_API_BASE}" "${MODEL}" "${STUDENT_OPENAI_API_KEY:-}"
assert_endpoint_model "${TEACHER_OPENAI_API_BASE}" "${TEACHER_MODEL}" "${TEACHER_OPENAI_API_KEY:-}"

if [[ "${SKIP_STALE_CLEANUP:-0}" == "1" ]]; then
  echo "skip stale collection container cleanup (SKIP_STALE_CLEANUP=1)"
  json_event "cleanup_skipped" "SKIP_STALE_CLEANUP=1"
else
  run_logged "00_cleanup_stale_collection_containers" \
    timeout "${STALE_CLEANUP_TIMEOUT:-180}" \
    bash ops/cleanup/cleanup_stale_collection_containers.sh
fi

run_logged "01_make_phase1_plan" \
  env "OPENAI_API_BASE=${STUDENT_OPENAI_API_BASE}" "EXPERIMENT_ROOT=${RUN_ROOT}" \
  python3 GeneralAgent/sft_data_collection/make_trial_plan.py \
    --config "${CONFIG}" \
    --splits "${SPLITS}" \
    --run-id "${RUN_ID}" \
    --date "${DATE}" \
    --pilot-per-bench "${PILOT_PER_BENCH}" \
    --benches "${BENCH_ARGS[@]}" \
    --out "${PLAN}"

if [[ -n "${CAMPAIGN_ROOT}" ]]; then
  UNFILTERED_PLAN="${RUN_ROOT}/plans/${RUN_ID}.unfiltered.jsonl"
  CAMPAIGN_FILTER_MANIFEST="${RUN_ROOT}/reports/campaign_filter_manifest.json"
  mv "${PLAN}" "${UNFILTERED_PLAN}"
  run_logged "01b_filter_campaign_plan" \
    python3 GeneralAgent/sft_data_collection/filter_plan_for_campaign.py \
      --plan "${UNFILTERED_PLAN}" \
      --out "${PLAN}" \
      --campaign-root "${CAMPAIGN_ROOT}" \
      --source-runs "${CAMPAIGN_SOURCE_RUNS}" \
      --success-policy "${CAMPAIGN_SUCCESS_POLICY}" \
      --manifest "${CAMPAIGN_FILTER_MANIFEST}"
  if [[ "$(line_count "${PLAN}")" -eq 0 ]]; then
    echo "campaign filter selected 0 records; nothing to run"
    write_report
    json_event "wrapper_finish" "campaign_filter_empty"
    exit 0
  fi
fi

if [[ "${PHASE1_STREAMING}" == "1" ]]; then
  echo "phase1 streaming mode: one global dynamic launcher; teacher fallback is queued per completed task"
  CHUNKS=()
else
  run_logged "02_split_phase1_plan" \
    python3 GeneralAgent/sft_data_collection/split_plan_chunks.py \
      --plan "${PLAN}" \
      --out-dir "${PHASE1_CHUNK_DIR}" \
      --tasks-per-chunk "${CHUNK_TASKS}"

  mapfile -t CHUNKS < <(find "${PHASE1_CHUNK_DIR}" -maxdepth 1 -type f -name 'chunk_*.jsonl' | sort)
  echo "chunks=${#CHUNKS[@]}"
fi

rm -f "${QUEUE_DIR}/STOP"
if compgen -G "${RUNNING_DIR}/*.jsonl" > /dev/null; then
  for running in "${RUNNING_DIR}"/*.jsonl; do
    base="$(basename "${running}")"
    if [[ ! -e "${DONE_DIR}/${base}" && ! -e "${QUEUE_DIR}/${base}" ]]; then
      mv "${running}" "${QUEUE_DIR}/${base}"
      json_event "phase2_requeue_running" "${base}"
    fi
  done
fi
refresh_existing_teacher_queue

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "dry run only; not launching phase1/phase2"
  write_report
  json_event "wrapper_finish" "dry_run"
  exit 0
fi

phase2_worker &
PHASE2_PID=$!

if [[ "${PHASE1_STREAMING}" == "1" ]]; then
  PHASE1_DONE_FILE="${CHUNK_ROOT}/PHASE1_DONE"
  rm -f "${PHASE1_DONE_FILE}"
  json_event "phase1_watcher_start" "plan=${PLAN}"
  env "TEACHER_OPENAI_API_BASE=${TEACHER_OPENAI_API_BASE}" "EXPERIMENT_ROOT=${RUN_ROOT}" \
    python3 GeneralAgent/sft_data_collection/watch_phase1_teacher_queue.py \
      --plan "${PLAN}" \
      --config "${CONFIG}" \
      --queue-dir "${QUEUE_DIR}" \
      --reflection-dir "${REFLECTION_DIR}" \
      --running-dir "${RUNNING_DIR}" \
      --done-dir "${DONE_DIR}" \
      --failed-dir "${FAILED_DIR}" \
      --stop-file "${PHASE1_DONE_FILE}" \
      --teacher-model "${TEACHER_MODEL}" \
      --teacher-trials "${TEACHER_TRIALS}" \
      2>&1 | tee "${LOG_ROOT}/phase1_teacher_watcher.log" &
  WATCHER_PID=$!

  set +e
  run_logged "phase1_streaming_all" \
    env "EXPERIMENT_ROOT=${RUN_ROOT}" python3 GeneralAgent/sft_data_collection/launch_trials.py \
      --plan "${PLAN}" \
      --model "${MODEL}" \
      --workers "${PHASE1_WORKERS}" \
      --task-window "${PHASE1_TASK_WINDOW}" \
      "${PHASE1_BENCH_CAP_ARGS[@]}" \
      --per-trial-timeout-sec "${TIMEOUT}" \
      --docker-wait-sec "${DOCKER_WAIT_SEC}" \
      --docker-check-interval-sec "${DOCKER_CHECK_INTERVAL_SEC}" \
      --execute
  PHASE1_RC=$?
  set -e
  touch "${PHASE1_DONE_FILE}"
  wait "${WATCHER_PID}"
  if [[ "${PHASE1_RC}" -ne 0 ]]; then
    echo "phase1_streaming_all failed rc=${PHASE1_RC}"
    touch "${QUEUE_DIR}/STOP"
    wait "${PHASE2_PID}" || true
    exit "${PHASE1_RC}"
  fi
else
  for chunk_path in "${CHUNKS[@]}"; do
    chunk_base="$(basename "${chunk_path}" .jsonl)"
    teacher_queue_base="${chunk_base}.teacher.jsonl"
    if [[ -e "${DONE_DIR}/${teacher_queue_base}" || -e "${QUEUE_DIR}/${teacher_queue_base}" || -e "${RUNNING_DIR}/${teacher_queue_base}" ]]; then
      json_event "chunk_skip_existing_teacher" "${teacher_queue_base}"
      echo "skip ${chunk_base}: teacher fallback already done/queued/running"
      continue
    fi

    run_logged "phase1_${chunk_base}" \
      env "EXPERIMENT_ROOT=${RUN_ROOT}" python3 GeneralAgent/sft_data_collection/launch_trials.py \
        --plan "${chunk_path}" \
        --model "${MODEL}" \
        --workers "${PHASE1_WORKERS}" \
        "${PHASE1_BENCH_CAP_ARGS[@]}" \
        --per-trial-timeout-sec "${TIMEOUT}" \
        --docker-wait-sec "${DOCKER_WAIT_SEC}" \
        --docker-check-interval-sec "${DOCKER_CHECK_INTERVAL_SEC}" \
        --execute

    teacher_plan="${REFLECTION_DIR}/${chunk_base}.teacher.jsonl"
    run_logged "make_teacher_${chunk_base}" \
      env "TEACHER_OPENAI_API_BASE=${TEACHER_OPENAI_API_BASE}" "EXPERIMENT_ROOT=${RUN_ROOT}" \
      python3 GeneralAgent/sft_data_collection/make_teacher_fallback_plan.py \
        --config "${CONFIG}" \
        --plan "${chunk_path}" \
        --out "${teacher_plan}" \
        --teacher-model "${TEACHER_MODEL}" \
        --teacher-trials "${TEACHER_TRIALS}"

    if [[ -s "${teacher_plan}" ]]; then
      cp "${teacher_plan}" "${QUEUE_DIR}/${chunk_base}.teacher.jsonl.tmp"
      mv "${QUEUE_DIR}/${chunk_base}.teacher.jsonl.tmp" "${QUEUE_DIR}/${teacher_queue_base}"
      json_event "phase2_enqueue" "${chunk_base} records=$(line_count "${teacher_plan}")"
    else
      json_event "phase2_skip_empty" "${chunk_base}"
    fi
  done
fi

touch "${QUEUE_DIR}/STOP"
wait "${PHASE2_PID}"

{
  cat "${PLAN}"
  find "${REFLECTION_DIR}" -maxdepth 1 -type f -name '*.teacher.jsonl' -size +0c -print0 \
    | sort -z \
    | xargs -0 --no-run-if-empty cat
} > "${COMBINED_PLAN}"
json_event "combined_plan_written" "records=$(line_count "${COMBINED_PLAN}")"

run_logged "collect_and_export" \
  env "EXPERIMENT_ROOT=${RUN_ROOT}" bash ops/workflows/sft_data_collection/collect_and_export.sh "${RUN_ID}"

run_logged "summarize_final" \
  env "EXPERIMENT_ROOT=${RUN_ROOT}" python3 GeneralAgent/sft_data_collection/scripts/summarize_run.py "${RUN_ID}"

write_report
json_event "wrapper_finish" "success"
echo "DONE"
echo "RUN_ID=${RUN_ID}"
echo "REPORT_PATH=${REPORT_PATH}"
echo "COMBINED_PLAN=${COMBINED_PLAN}"
