#!/usr/bin/env bash
# Foreground SGLang server used by eval workflows and tmux launch helpers.
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SKILLRL_ROOT:-$(pwd)}}"
CONDA_SH="${CONDA_SH:-${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-slime}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
SERVED_NAME="${SERVED_NAME:?set SERVED_NAME}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-4}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
MEM_FRACTION="${MEM_FRACTION:-0.88}"
RANDOM_SEED="${RANDOM_SEED:-1063810697}"

[[ -f "${MODEL_PATH}/config.json" ]] || {
  echo "FATAL: model config missing: ${MODEL_PATH}/config.json" >&2
  exit 2
}

# shellcheck source=/dev/null
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,0.0.0.0}"
export no_proxy="${NO_PROXY}"
export SGLANG_DISABLE_CUDNN_CHECK=1
export CUDA_HOME="${SGLANG_CUDA_HOME:-${CONDA_PREFIX}}"
export CUDA_PATH="${CUDA_HOME}"

echo "SGLang model=${MODEL_PATH} served=${SERVED_NAME} GPUs=${CUDA_VISIBLE_DEVICES:-all} port=${PORT} TP=${TP_SIZE}"
exec python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --context-length "${CONTEXT_LENGTH}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --random-seed "${RANDOM_SEED}" \
  --enable-metrics \
  --log-level info
