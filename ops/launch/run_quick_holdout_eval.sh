#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${SKILLRL_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"
if [[ -f "secrets/.env.secrets" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "secrets/.env.secrets"
  set +a
fi

DATE="${DATE:-$(date -u +%Y%m%d)}"
MODEL="${MODEL:-qwen3.5-9b}"
ARM="${ARM:-retrieval}"
RUN_ID="${RUN_ID:-${DATE}_quick_holdout_eval_${ARM}_${MODEL//[^A-Za-z0-9]/_}}"
RUN_ROOT="${EXPERIMENT_ROOT:-${RUN_ROOT:-experiments/${DATE}/${RUN_ID}}}"
SPLIT="${SPLIT:-GeneralAgent/sft_data_collection/outputs/splits/default/quick_test/quick30/holdout_split.json}"
TRIALS="${TRIALS:-1}"
WORKERS="${WORKERS:-4}"
TIMEOUT="${TIMEOUT:-2400}"
MAX_TURNS="${MAX_TURNS:-30}"
MAX_TIME="${MAX_TIME:-850}"

if [[ -n "${BENCHES:-}" ]]; then
  # shellcheck disable=SC2206
  BENCH_ARGS=(${BENCHES})
else
  BENCH_ARGS=(claw tb2 sb_ns seta_synth swe_lite)
fi

PLAN="${RUN_ROOT}/plans/${RUN_ID}.jsonl"
mkdir -p "${RUN_ROOT}/plans" "${RUN_ROOT}/logs/runner" "${RUN_ROOT}/reports"

echo "RUN_ID=${RUN_ID}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "MODEL=${MODEL}"
echo "ARM=${ARM}"
echo "OPENAI_API_BASE=${OPENAI_API_BASE:-http://127.0.0.1:30000/v1}"
echo "SPLIT=${SPLIT}"
echo "BENCHES=${BENCH_ARGS[*]}"
echo "TRIALS=${TRIALS} WORKERS=${WORKERS} TIMEOUT=${TIMEOUT}"
echo "MAX_TURNS=${MAX_TURNS} MAX_TIME=${MAX_TIME}"

env "EXPERIMENT_ROOT=${RUN_ROOT}" \
  python3 GeneralAgent/sft_data_collection/make_quick_eval_plan.py \
    --run-id "${RUN_ID}" \
    --date "${DATE}" \
    --model "${MODEL}" \
    --arm "${ARM}" \
    --split "${SPLIT}" \
    --trials "${TRIALS}" \
    --max-turns "${MAX_TURNS}" \
    --max-time "${MAX_TIME}" \
    --benches "${BENCH_ARGS[@]}" \
    --out "${PLAN}"

env "EXPERIMENT_ROOT=${RUN_ROOT}" \
  python3 GeneralAgent/sft_data_collection/launch_trials.py \
    --plan "${PLAN}" \
    --model "${MODEL}" \
    --workers "${WORKERS}" \
    --per-trial-timeout-sec "${TIMEOUT}" \
    --execute \
    2>&1 | tee "${RUN_ROOT}/logs/runner/quick_holdout_eval.log"

env "EXPERIMENT_ROOT=${RUN_ROOT}" \
  python3 GeneralAgent/sft_data_collection/data_quality_dashboard.py "${RUN_ID}" \
    --run-root "${RUN_ROOT}"

python3 ops/experiments/register_experiment.py \
  --run-id "${RUN_ID}" \
  --path "${RUN_ROOT}" \
  --date "${DATE}" \
  --kind "eval" \
  --status "completed" \
  --launcher "ops/launch/run_quick_holdout_eval.sh" \
  --scripts "GeneralAgent/sft_data_collection/make_quick_eval_plan.py,GeneralAgent/sft_data_collection/launch_trials.py,GeneralAgent/sft_data_collection/data_quality_dashboard.py" \
  --intent "quick holdout eval: ${MODEL} ${ARM} on fixed quick30 split" \
  --notes "resume by reusing RUN_ID/RUN_ROOT; do not keep wrapper-only folders" \
  --tags "quick30,${MODEL},${ARM}"
