#!/usr/bin/env bash
# NOTE: Migrated canonical workflow copy. Source: GeneralAgent/sft_data_collection/scripts/collect_and_export.sh
# Original historical script is archived during workflow cleanup; maintain this copy going forward.
set -euo pipefail

RUN_ID="${1:?usage: $0 RUN_ID}"
PROJECT_ROOT="${SKILLRL_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

if [[ "${RUN_ID}" =~ (20[0-9]{6}) ]]; then
  RUN_DATE="${BASH_REMATCH[1]}"
else
  RUN_DATE="${DATE:-$(date -u +%Y%m%d)}"
fi
RUN_ROOT="${EXPERIMENT_ROOT:-${RUN_ROOT:-experiments/${RUN_DATE}/${RUN_ID}}}"
PLAN="${RUN_ROOT}/plans/${RUN_ID}.jsonl"
COMBINED="${RUN_ROOT}/plans/${RUN_ID}.combined.jsonl"
if [[ -f "${COMBINED}" ]]; then
  PLAN="${COMBINED}"
fi

OUT_DIR="${RUN_ROOT}/collected"
DATASET_NAME="agent_${RUN_ID//[^A-Za-z0-9_]/_}"
LF_DIR="${RUN_ROOT}/llamafactory_data"

python3 GeneralAgent/sft_data_collection/collect_successes.py \
  --plan "${PLAN}" \
  --out-dir "${OUT_DIR}" \
  --max-successes-per-task "${MAX_SUCCESSES_PER_TASK:-2}" \
  --max-successes-per-use-skill-task "${MAX_SUCCESSES_PER_USE_SKILL_TASK:-4}"

set +e
python3 GeneralAgent/sft_training/export_llamafactory.py \
  --input "${OUT_DIR}/sft_messages.jsonl" \
  --out-dir "${LF_DIR}" \
  --dataset-name "${DATASET_NAME}"
export_rc=$?
set -e
if [[ "${export_rc}" -ne 0 ]]; then
  if [[ "${ALLOW_EMPTY_SFT_EXPORT:-0}" == "1" ]] && [[ -f "${OUT_DIR}/sft_messages.jsonl" ]] && [[ ! -s "${OUT_DIR}/sft_messages.jsonl" ]]; then
    echo "ALLOW_EMPTY_SFT_EXPORT=1 and no SFT messages were collected; keeping empty export for smoke/test run"
  else
    exit "${export_rc}"
  fi
fi

echo "COLLECTED=${OUT_DIR}"
echo "LLAMAFACTORY_DIR=${LF_DIR}"
echo "LLAMAFACTORY_DATASET=${DATASET_NAME}"
