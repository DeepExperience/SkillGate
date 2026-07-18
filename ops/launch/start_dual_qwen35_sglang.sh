#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SKILLRL_ROOT:-$(pwd)}}"
CONDA_SH="${CONDA_SH:-${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-slime}"
MODEL_ROOT="${MODEL_ROOT:-${SKILLRL_ROOT:-$(pwd)}/models}"

STUDENT_SESSION="${STUDENT_SESSION:-sglang-9b}"
TEACHER_SESSION="${TEACHER_SESSION:-sglang-27b}"
STUDENT_GPUS="${STUDENT_GPUS:-0,1,2,3}"
TEACHER_GPUS="${TEACHER_GPUS:-4,5,6,7}"
STUDENT_PORT="${STUDENT_PORT:-30000}"
TEACHER_PORT="${TEACHER_PORT:-30001}"
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-${MODEL_ROOT}/Qwen3.5-9B}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${MODEL_ROOT}/Qwen3.5-27B}"
STUDENT_SERVED_NAME="${STUDENT_SERVED_NAME:-qwen3.5-9b}"
TEACHER_SERVED_NAME="${TEACHER_SERVED_NAME:-qwen3.5-27b}"
TP_SIZE="${TP_SIZE:-4}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
RANDOM_SEED="${RANDOM_SEED:-1063810697}"
WAIT_SEC="${WAIT_SEC:-1200}"
RESTART="${RESTART:-1}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"

cd "${PROJECT_ROOT}"

launch_server() {
  local session="$1"
  local gpus="$2"
  local model_path="$3"
  local served_name="$4"
  local port="$5"
  local log_path="/tmp/${session}-${STAMP}.log"

  if tmux has-session -t "${session}" 2>/dev/null; then
    if [[ "${RESTART}" == "1" ]]; then
      tmux kill-session -t "${session}"
      sleep 2
    else
      echo "tmux session already exists: ${session}"
      return 0
    fi
  fi

  echo "starting ${session}: ${served_name} GPUs=${gpus} port=${port} log=${log_path}"
  tmux new-session -d -s "${session}" \
    "cd '${PROJECT_ROOT}' && \
     CUDA_VISIBLE_DEVICES='${gpus}' \
     MODEL_PATH='${model_path}' \
     SERVED_NAME='${served_name}' \
     PORT='${port}' \
     TP_SIZE='${TP_SIZE}' \
     CONTEXT_LENGTH='${CONTEXT_LENGTH}' \
     MEM_FRACTION='${MEM_FRACTION}' \
     RANDOM_SEED='${RANDOM_SEED}' \
     bash -lc 'source \"${CONDA_SH}\" && conda activate \"${CONDA_ENV}\" && \
       export NO_PROXY=\"127.0.0.1,localhost,0.0.0.0\" no_proxy=\"127.0.0.1,localhost,0.0.0.0\" && \
       export SGLANG_DISABLE_CUDNN_CHECK=1 && \
       export CUDA_HOME=\"\$CONDA_PREFIX\" && \
       echo \"CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES\" && \
       echo \"CUDA_HOME=\$CUDA_HOME\" && \
       nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader && \
       exec python -m sglang.launch_server \
         --model-path \"\$MODEL_PATH\" \
         --served-model-name \"\$SERVED_NAME\" \
         --tensor-parallel-size \"\$TP_SIZE\" \
         --host 0.0.0.0 \
         --port \"\$PORT\" \
         --context-length \"\$CONTEXT_LENGTH\" \
         --mem-fraction-static \"\$MEM_FRACTION\" \
         --random-seed \"\$RANDOM_SEED\" \
         --enable-metrics \
         --log-level info' 2>&1 | tee '${log_path}'"
}

served_model() {
  local port="$1"
  curl -sS --max-time 3 "http://127.0.0.1:${port}/v1/models" 2>/dev/null \
    | python3 -c 'import json,sys; print((json.load(sys.stdin).get("data") or [{}])[0].get("id",""))' \
      2>/dev/null || true
}

wait_for_model() {
  local served_name="$1"
  local port="$2"
  local deadline=$((SECONDS + WAIT_SEC))
  local current=""
  while (( SECONDS < deadline )); do
    current="$(served_model "${port}")"
    if [[ "${current}" == "${served_name}" ]]; then
      echo "ready: ${served_name} at http://127.0.0.1:${port}/v1"
      return 0
    fi
    echo "waiting: ${served_name} on ${port}; current=${current:-<not-ready>}"
    sleep 15
  done
  echo "timeout waiting for ${served_name} on port ${port}"
  return 1
}

echo "GPU inventory:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

launch_server "${STUDENT_SESSION}" "${STUDENT_GPUS}" "${STUDENT_MODEL_PATH}" "${STUDENT_SERVED_NAME}" "${STUDENT_PORT}"
launch_server "${TEACHER_SESSION}" "${TEACHER_GPUS}" "${TEACHER_MODEL_PATH}" "${TEACHER_SERVED_NAME}" "${TEACHER_PORT}"

wait_for_model "${STUDENT_SERVED_NAME}" "${STUDENT_PORT}"
wait_for_model "${TEACHER_SERVED_NAME}" "${TEACHER_PORT}"

echo "dual SGLang ready"
echo "student endpoint: http://127.0.0.1:${STUDENT_PORT}/v1"
echo "teacher endpoint: http://127.0.0.1:${TEACHER_PORT}/v1"
