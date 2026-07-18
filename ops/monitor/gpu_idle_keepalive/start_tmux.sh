#!/usr/bin/env bash
set -euo pipefail

PROJ=${SKILLRL_ROOT:-$(pwd)}
SESSION=${GPU_GUARD_TMUX_SESSION:-gpu-idle-keepalive}
API_BASE=${GPU_GUARD_API_BASE:-http://127.0.0.1:30000/v1}
MODEL=${GPU_GUARD_MODEL:-}
GPU_INDICES=${GPU_GUARD_GPU_INDICES:-${CUDA_VISIBLE_DEVICES:-}}
IDLE_HOURS=${GPU_GUARD_IDLE_HOURS:-0.25}
KEEPALIVE_SEC=${GPU_GUARD_KEEPALIVE_SEC:-60}
SAMPLE_SEC=${GPU_GUARD_SAMPLE_SEC:-60}
BUSY_THRESHOLD=${GPU_GUARD_BUSY_THRESHOLD:-20}
MAX_TOKENS=${GPU_GUARD_MAX_TOKENS:-512}
GPU_PROBE_SEC=${GPU_GUARD_GPU_PROBE_SEC:-20}
GPU_PROBE_SIZE=${GPU_GUARD_GPU_PROBE_SIZE:-4096}
ALWAYS_GPU_PROBE=${GPU_GUARD_ALWAYS_GPU_PROBE:-0}
RESTART=${GPU_GUARD_RESTART:-1}
RESTART_SEC=${GPU_GUARD_RESTART_SEC:-10}
LOG_PATH=${GPU_GUARD_LOG_PATH:-}
PYTHON_BIN=${GPU_GUARD_PYTHON:-python3}
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-${GPU_INDICES}}
RUNNER="/tmp/${SESSION}.runner.sh"

cd "$PROJ"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION"
    echo "view: tmux attach -t $SESSION"
    exit 0
fi

{
    echo '#!/usr/bin/env bash'
    echo 'set -u'
    printf 'PROJ=%q\n' "$PROJ"
    printf 'API_BASE=%q\n' "$API_BASE"
    printf 'MODEL=%q\n' "$MODEL"
    printf 'GPU_INDICES=%q\n' "$GPU_INDICES"
    printf 'CUDA_DEVICES=%q\n' "$CUDA_DEVICES"
    printf 'IDLE_HOURS=%q\n' "$IDLE_HOURS"
    printf 'KEEPALIVE_SEC=%q\n' "$KEEPALIVE_SEC"
    printf 'SAMPLE_SEC=%q\n' "$SAMPLE_SEC"
    printf 'BUSY_THRESHOLD=%q\n' "$BUSY_THRESHOLD"
    printf 'MAX_TOKENS=%q\n' "$MAX_TOKENS"
    printf 'GPU_PROBE_SEC=%q\n' "$GPU_PROBE_SEC"
    printf 'GPU_PROBE_SIZE=%q\n' "$GPU_PROBE_SIZE"
    printf 'ALWAYS_GPU_PROBE=%q\n' "$ALWAYS_GPU_PROBE"
    printf 'RESTART=%q\n' "$RESTART"
    printf 'RESTART_SEC=%q\n' "$RESTART_SEC"
    printf 'LOG_PATH=%q\n' "$LOG_PATH"
    printf 'PYTHON_BIN=%q\n' "$PYTHON_BIN"
    cat <<'SH'
cd "$PROJ"
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "$CUDA_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
fi
if [[ -n "$LOG_PATH" ]]; then
  mkdir -p "$(dirname "$LOG_PATH")"
  exec > >(tee -a "$LOG_PATH") 2>&1
fi
while true; do
  echo "[$(date -Iseconds)] supervisor starting gpu_idle_keepalive session=${TMUX_PANE:-unknown} gpu_indices=${GPU_INDICES:-all} cuda_visible=${CUDA_VISIBLE_DEVICES:-all}"
  args=(
    --api-base "$API_BASE"
    --model "$MODEL"
    --gpu-indices "$GPU_INDICES"
    --idle-hours "$IDLE_HOURS"
    --sample-sec "$SAMPLE_SEC"
    --keepalive-sec "$KEEPALIVE_SEC"
    --busy-threshold "$BUSY_THRESHOLD"
    --max-tokens "$MAX_TOKENS"
    --gpu-probe-sec "$GPU_PROBE_SEC"
    --gpu-probe-size "$GPU_PROBE_SIZE"
  )
  if [[ "$ALWAYS_GPU_PROBE" == "1" ]]; then
    args+=(--always-gpu-probe)
  fi
  "$PYTHON_BIN" -B -u ops/monitor/gpu_idle_keepalive/gpu_idle_keepalive.py "${args[@]}"
  rc=$?
  echo "[$(date -Iseconds)] supervisor gpu_idle_keepalive exited rc=$rc"
  if [[ "$RESTART" != "1" ]]; then
    exit "$rc"
  fi
  sleep "$RESTART_SEC"
done
SH
} > "$RUNNER"
chmod +x "$RUNNER"

tmux new-session -d -s "$SESSION" "bash '$RUNNER'"

echo "started tmux session: $SESSION"
echo "view: tmux attach -t $SESSION"
if [[ -n "$LOG_PATH" ]]; then
    echo "log: $LOG_PATH"
fi
echo "stop: bash ops/monitor/gpu_idle_keepalive/stop_tmux.sh"
