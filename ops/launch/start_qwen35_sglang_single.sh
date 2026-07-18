#!/usr/bin/env bash
set -Eeuo pipefail

# Start one Qwen-family SGLang endpoint in tmux and wait until /v1/models is ready.
#
# This is a reusable low-level launcher. Workflow scripts should set MODEL_PATH,
# SERVED_NAME, PORT, GPU_IDS, and resource knobs through env vars.

PROJECT_ROOT="${PROJECT_ROOT:-${SKILLRL_ROOT:-$(pwd)}}"
CONDA_SH="${CONDA_SH:-${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-slime}"
MODEL_ROOT="${MODEL_ROOT:-${SKILLRL_ROOT:-$(pwd)}/models}"

TMUX_SESSION="${TMUX_SESSION:-sglang-qwen-single}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/Qwen3.5-27B}"
SERVED_NAME="${SERVED_NAME:-qwen3.5-27b}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-4}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
MEM_FRACTION="${MEM_FRACTION:-0.88}"
RANDOM_SEED="${RANDOM_SEED:-1063810697}"
WAIT_SEC="${WAIT_SEC:-1200}"
RESTART="${RESTART:-1}"
LOG_DIR="${LOG_DIR:-/tmp}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${TMUX_SESSION}-${STAMP}.log}"

cd "${PROJECT_ROOT}"

served_model() {
  curl -sS --max-time 3 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
    | python3 -c 'import json,sys; print((json.load(sys.stdin).get("data") or [{}])[0].get("id",""))' \
      2>/dev/null || true
}

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 2
fi

current="$(served_model)"
if [[ "${current}" == "${SERVED_NAME}" ]]; then
  echo "ready already: ${SERVED_NAME} at http://127.0.0.1:${PORT}/v1"
  exit 0
fi

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  if [[ "${RESTART}" == "1" ]]; then
    tmux kill-session -t "${TMUX_SESSION}"
    sleep 2
  else
    echo "tmux session already exists: ${TMUX_SESSION}"
    echo "view: tmux attach -t ${TMUX_SESSION}"
    exit 0
  fi
fi

mkdir -p "$(dirname "${LOG_PATH}")"

echo "starting ${TMUX_SESSION}: ${SERVED_NAME} GPUs=${GPU_IDS} port=${PORT} log=${LOG_PATH}"
echo "MODEL_PATH=${MODEL_PATH}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

tmux new-session -d -s "${TMUX_SESSION}" \
  "cd '${PROJECT_ROOT}' && \
   CUDA_VISIBLE_DEVICES='${GPU_IDS}' \
   CONDA_SH='${CONDA_SH}' \
   CONDA_ENV='${CONDA_ENV}' \
   MODEL_PATH='${MODEL_PATH}' \
   SERVED_NAME='${SERVED_NAME}' \
   PORT='${PORT}' \
   TP_SIZE='${TP_SIZE}' \
   CONTEXT_LENGTH='${CONTEXT_LENGTH}' \
   MEM_FRACTION='${MEM_FRACTION}' \
   RANDOM_SEED='${RANDOM_SEED}' \
   bash ops/launch/run_qwen35_sglang_server.sh 2>&1 | tee '${LOG_PATH}'"

deadline=$((SECONDS + WAIT_SEC))
while (( SECONDS < deadline )); do
  current="$(served_model)"
  if [[ "${current}" == "${SERVED_NAME}" ]]; then
    echo "ready: ${SERVED_NAME} at http://127.0.0.1:${PORT}/v1"
    echo "tmux: ${TMUX_SESSION}"
    echo "log: ${LOG_PATH}"
    exit 0
  fi
  echo "waiting: ${SERVED_NAME} on ${PORT}; current=${current:-<not-ready>}"
  sleep 15
done

echo "timeout waiting for ${SERVED_NAME} on port ${PORT}" >&2
echo "view: tmux attach -t ${TMUX_SESSION}" >&2
echo "log: ${LOG_PATH}" >&2
exit 1
