#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SKILLRL_ROOT:-$(pwd)}}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-experiments/sft_skill_use_campaign}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/${CAMPAIGN_ROOT}/logs/monitor}"
PYTHON_BIN="${GPU_GUARD_PYTHON:-${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/envs/slime/bin/python}"

IDLE_HOURS="${GPU_GUARD_IDLE_HOURS:-0.01}"
KEEPALIVE_SEC="${GPU_GUARD_KEEPALIVE_SEC:-60}"
SAMPLE_SEC="${GPU_GUARD_SAMPLE_SEC:-30}"
GPU_PROBE_SEC="${GPU_GUARD_GPU_PROBE_SEC:-25}"
GPU_PROBE_SIZE="${GPU_GUARD_GPU_PROBE_SIZE:-4096}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"

start_guard() {
  local name="$1"
  local api_base="$2"
  local model="$3"
  local gpu_indices="$4"
  local log_path="${LOG_DIR}/gpu_keepalive_${name}.log"

  GPU_GUARD_TMUX_SESSION="gpu-keepalive-${name}" \
  GPU_GUARD_API_BASE="${api_base}" \
  GPU_GUARD_MODEL="${model}" \
  GPU_GUARD_GPU_INDICES="${gpu_indices}" \
  GPU_GUARD_IDLE_HOURS="${IDLE_HOURS}" \
  GPU_GUARD_KEEPALIVE_SEC="${KEEPALIVE_SEC}" \
  GPU_GUARD_SAMPLE_SEC="${SAMPLE_SEC}" \
  GPU_GUARD_BUSY_THRESHOLD=20 \
  GPU_GUARD_MAX_TOKENS=512 \
  GPU_GUARD_GPU_PROBE_SEC="${GPU_PROBE_SEC}" \
  GPU_GUARD_GPU_PROBE_SIZE="${GPU_PROBE_SIZE}" \
  GPU_GUARD_ALWAYS_GPU_PROBE=1 \
  GPU_GUARD_RESTART=1 \
  GPU_GUARD_RESTART_SEC=10 \
  GPU_GUARD_PYTHON="${PYTHON_BIN}" \
  GPU_GUARD_LOG_PATH="${log_path}" \
    bash ops/monitor/gpu_idle_keepalive/start_tmux.sh
}

start_guard "qwen9b" "http://127.0.0.1:30000/v1" "qwen3.5-9b" "0,1,2,3"
start_guard "qwen27b" "http://127.0.0.1:30001/v1" "qwen3.5-27b" "4,5,6,7"

echo "qwen3.5 9B/27B keepalive guards requested"
echo "python: ${PYTHON_BIN}"
echo "logs: ${LOG_DIR}/gpu_keepalive_qwen{9b,27b}.log"
