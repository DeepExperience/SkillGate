#!/usr/bin/env bash
# NOTE: Migrated canonical workflow copy. Source: ops/launch/run_sft_v2_serve_and_eval_chain.sh
# Original historical script is archived during workflow cleanup; maintain this copy going forward.
# Chain: wait LoRA merge done → launch SGLang on merged model → wait endpoint ready → run quick30 holdout eval.
# Use for hands-off overnight evaluation after `llamafactory-cli export` is kicked off in another tmux.
set -uo pipefail

PROJECT_ROOT="${SKILLRL_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

MERGED="${MERGED:-${PROJECT_ROOT}/GeneralAgent/sft_training/merged_models/qwen35_9b_sft_campaign_20260503_1015_hindsight_49k_5epoch_r32_liger}"
MERGE_TMUX="${MERGE_TMUX:-sft-v2-merge}"
SGLANG_TMUX="${SGLANG_TMUX:-sglang-qwen9b-sft-v2}"
EVAL_TMUX="${EVAL_TMUX:-quick-holdout-eval-v2}"
EVAL_DATE="${EVAL_DATE:-$(date -u +%Y%m%d)}"
EVAL_RUN_ID="${EVAL_RUN_ID:-${EVAL_DATE}_quick_holdout_eval_v2_retrieval}"
SERVED_NAME="${SERVED_NAME:-qwen3.5-9b-sft-v2}"
SGLANG_PORT="${SGLANG_PORT:-30001}"
SGLANG_TP="${SGLANG_TP:-4}"
SGLANG_GPUS="${SGLANG_GPUS:-4,5,6,7}"
SGLANG_CTX="${SGLANG_CTX:-262144}"
SGLANG_MEM="${SGLANG_MEM:-0.90}"
SGLANG_WAIT_SEC="${SGLANG_WAIT_SEC:-1800}"
CONDA_SH="${CONDA_SH:-${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-slime}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
SGLANG_LOG="/tmp/${SGLANG_TMUX}-${STAMP}.log"

echo "[chain] start $(date -Iseconds)"
echo "[chain] MERGED=${MERGED}"
echo "[chain] EVAL_RUN_ID=${EVAL_RUN_ID}"

# 1. Wait for merge
echo "[chain] waiting for merge tmux=${MERGE_TMUX}"
while [[ ! -f "${MERGED}/config.json" || ! -f "${MERGED}/tokenizer.json" ]]; do
  if ! tmux has-session -t "${MERGE_TMUX}" 2>/dev/null; then
    echo "[chain] FATAL: merge tmux exited before merged checkpoint appeared. tail of merge log:"
    tail -40 /tmp/sft_v2_merge_*.log 2>/dev/null | tail -40
    exit 1
  fi
  sleep 30
done
echo "[chain] merge done $(date -Iseconds): $(du -sh "${MERGED}" | awk '{print $1}')"

# 2. Start SGLang on merged model
if tmux has-session -t "${SGLANG_TMUX}" 2>/dev/null; then
  echo "[chain] sglang tmux already exists, killing first"
  tmux kill-session -t "${SGLANG_TMUX}" || true
  sleep 3
fi

echo "[chain] launching sglang tmux=${SGLANG_TMUX} port=${SGLANG_PORT} gpus=${SGLANG_GPUS}"
tmux new-session -d -s "${SGLANG_TMUX}" \
  "cd '${PROJECT_ROOT}' && \
   CUDA_VISIBLE_DEVICES='${SGLANG_GPUS}' \
   bash -lc 'source \"${CONDA_SH}\" && conda activate \"${CONDA_ENV}\" && \
     export NO_PROXY=\"127.0.0.1,localhost,0.0.0.0\" no_proxy=\"127.0.0.1,localhost,0.0.0.0\" && \
     export SGLANG_DISABLE_CUDNN_CHECK=1 && \
     export CUDA_HOME=\"\$CONDA_PREFIX\" && \
     nvidia-smi --query-gpu=index,memory.used --format=csv,noheader && \
     exec python -m sglang.launch_server \
       --model-path \"${MERGED}\" \
       --served-model-name \"${SERVED_NAME}\" \
       --tensor-parallel-size '${SGLANG_TP}' \
       --host 0.0.0.0 \
       --port '${SGLANG_PORT}' \
       --context-length '${SGLANG_CTX}' \
       --mem-fraction-static '${SGLANG_MEM}' \
       --random-seed 1063810697 \
       --enable-metrics \
       --log-level info' 2>&1 | tee '${SGLANG_LOG}'"

# 3. Wait until SGLang answers /v1/models with our served name
deadline=$((SECONDS + SGLANG_WAIT_SEC))
echo "[chain] waiting SGLang to serve ${SERVED_NAME} on :${SGLANG_PORT} (max ${SGLANG_WAIT_SEC}s)"
while (( SECONDS < deadline )); do
  current="$(curl -sS --max-time 3 "http://127.0.0.1:${SGLANG_PORT}/v1/models" 2>/dev/null \
    | python3 -c 'import json,sys; d=(json.load(sys.stdin).get("data") or [{}])[0]; print(d.get("id",""))' 2>/dev/null || true)"
  if [[ "${current}" == "${SERVED_NAME}" ]]; then
    echo "[chain] sglang ready $(date -Iseconds): ${SERVED_NAME} @ :${SGLANG_PORT}"
    break
  fi
  if ! tmux has-session -t "${SGLANG_TMUX}" 2>/dev/null; then
    echo "[chain] FATAL: sglang tmux died before readiness. tail:"
    tail -50 "${SGLANG_LOG}"
    exit 1
  fi
  sleep 20
done
if [[ "$(curl -sS --max-time 3 "http://127.0.0.1:${SGLANG_PORT}/v1/models" 2>/dev/null \
  | python3 -c 'import json,sys; d=(json.load(sys.stdin).get("data") or [{}])[0]; print(d.get("id",""))' 2>/dev/null || true)" != "${SERVED_NAME}" ]]; then
  echo "[chain] FATAL: sglang readiness timeout"
  exit 1
fi

# 4. Run quick30 holdout eval against the new endpoint
EVAL_RUN_ROOT="experiments/${EVAL_DATE}/${EVAL_RUN_ID}"
mkdir -p "${EVAL_RUN_ROOT}/plans" "${EVAL_RUN_ROOT}/logs/runner" "${EVAL_RUN_ROOT}/reports"

echo "[chain] launching eval tmux=${EVAL_TMUX} run_id=${EVAL_RUN_ID}"
tmux new-session -d -s "${EVAL_TMUX}" "cd '${PROJECT_ROOT}' && \
  RUN_ID='${EVAL_RUN_ID}' \
  RUN_ROOT='${EVAL_RUN_ROOT}' \
  EXPERIMENT_ROOT='${EVAL_RUN_ROOT}' \
  DATE='${EVAL_DATE}' \
  MODEL='${SERVED_NAME}' \
	  ARM='retrieval' \
	  OPENAI_API_BASE='http://127.0.0.1:${SGLANG_PORT}/v1' \
	  UNIFIED_TOOLS_SCHEMA_MODE='manual_schema' \
	  UNIFIED_INJECT_SCHEMA_TOKENIZER_PATH='${PROJECT_ROOT}/models/Qwen3.5-9B' \
	  TRIALS=1 \
	  WORKERS=4 \
	  TIMEOUT=2400 \
  bash ops/launch/run_quick_holdout_eval.sh 2>&1 | tee '/tmp/quick_holdout_eval_v2_${STAMP}.log'"

echo "[chain] all steps fired $(date -Iseconds)"
echo "[chain]   merged: ${MERGED}"
echo "[chain]   sglang tmux: ${SGLANG_TMUX} log=${SGLANG_LOG}"
echo "[chain]   eval tmux: ${EVAL_TMUX} run_root=${EVAL_RUN_ROOT}"
