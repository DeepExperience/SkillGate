#!/usr/bin/env bash
set -Eeuo pipefail

# Build the standalone 147-task Claw mixed-skill snapshot.
#
# Resume behavior:
# - The task universe and names are deterministic and stored under eval_claw_147.
# - Accepted oracle/misleading SKILL.md files are skipped on rerun.
# - Independent and frozen-instance audit rejects are regenerated in the same skill path.
# - No dated candidate roots or v2/v3 skill copies are created.
#
# Canonical output:
#   skill_libraries/snapshots/rl/eval_claw_147
#
# Canonical entrypoint:
#   bash ops/workflows/rl_eval/run_eval_claw_147_slate.sh

ROOT="${ROOT:-/path/to/skillRL}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/skill_libraries/snapshots/rl/eval_claw_147}"
WORK_ROOT="${WORK_ROOT:-${ROOT}/experiments/skill_slate_build/eval_claw_147}"
MODEL_PATH="${MODEL_PATH:-/path/to/skillRL/models/Qwen3.5-27B}"
SERVED_NAME="${SERVED_NAME:-qwen3.5-27b}"
REMOTE_NODE="${REMOTE_NODE:-}"
FLEET_MODE="${FLEET_MODE:-auto}"
ROUTER_PORT="${ROUTER_PORT:-30100}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-39147}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
MEM_FRACTION="${MEM_FRACTION:-0.88}"
SEED="${SEED:-1063810697}"
GEN_WORKERS="${GEN_WORKERS:-}"
GEN_ATTEMPTS="${GEN_ATTEMPTS:-6}"
KEEP_SERVERS="${KEEP_SERVERS:-1}"
DRY_RUN="${DRY_RUN:-0}"

LOCAL_SESSION_0="claw147-27b-local0"
LOCAL_SESSION_1="claw147-27b-local1"
ROUTER_SESSION="claw147-27b-router"
BUILDER="ops/workflows/rl_eval/build_eval_claw_147_slate.py"

cd "${ROOT}"
mkdir -p "${WORK_ROOT}/logs" "${OUTPUT_ROOT}"
exec 9>"${WORK_ROOT}/pipeline.lock"
if ! flock -n 9; then
  echo "ERROR: another eval_claw_147 slate pipeline holds ${WORK_ROOT}/pipeline.lock" >&2
  exit 2
fi

LOG="${WORK_ROOT}/logs/pipeline.log"
export PYTHONUNBUFFERED=1
export SGLANG_DISABLE_CUDNN_CHECK=1

resolve_fleet() {
  local discovered=""
  case "${FLEET_MODE}" in
    single)
      REMOTE_NODE=""
      ;;
    dual|auto)
      if [[ -z "${REMOTE_NODE}" ]]; then
        discovered=$(timeout 15 /usr/bin/python3 - <<'PY' 2>/dev/null || true
import ray
ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
local = ray.util.get_node_ip_address()
nodes = sorted({
    item["NodeManagerAddress"]
    for item in ray.nodes()
    if item.get("Alive") and float((item.get("Resources") or {}).get("GPU", 0) or 0) > 0
})
other = [node for node in nodes if node != local]
if len(nodes) == 2 and len(other) == 1:
    print(other[0])
ray.shutdown()
PY
        )
        REMOTE_NODE="${discovered}"
      fi
      if [[ "${FLEET_MODE}" == dual && -z "${REMOTE_NODE}" ]]; then
        echo "ERROR: FLEET_MODE=dual requires a second live Ray GPU node or REMOTE_NODE" >&2
        return 2
      fi
      if [[ -z "${REMOTE_NODE}" ]]; then
        FLEET_MODE=single
      else
        FLEET_MODE=dual
      fi
      ;;
    *)
      echo "ERROR: FLEET_MODE must be auto, single, or dual" >&2
      return 2
      ;;
  esac
  GEN_WORKERS="${GEN_WORKERS:-$([[ "${FLEET_MODE}" == dual ]] && echo 48 || echo 24)}"
  export NO_PROXY="127.0.0.1,localhost,0.0.0.0${REMOTE_NODE:+,${REMOTE_NODE}}"
  export no_proxy="${NO_PROXY}"
  echo "[fleet] mode=${FLEET_MODE} remote=${REMOTE_NODE:-none} generation_workers=${GEN_WORKERS}"
}

api_ready() {
  local host="$1" port="$2"
  curl -sS --max-time 3 "http://${host}:${port}/v1/models" 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("data") or [{}])[0].get("id", ""))' 2>/dev/null \
    | grep -Fxq "${SERVED_NAME}"
}

start_local() {
  local session="$1" gpus="$2" port="$3"
  if api_ready 127.0.0.1 "${port}"; then
    echo "[fleet] local ${port} already ready"
    return
  fi
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[fleet] local ${session} is already starting"
    return
  fi
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && CUDA_VISIBLE_DEVICES='${gpus}' MODEL_PATH='${MODEL_PATH}' \
     SERVED_NAME='${SERVED_NAME}' PORT='${port}' TP_SIZE=4 \
     CONTEXT_LENGTH='${CONTEXT_LENGTH}' MEM_FRACTION='${MEM_FRACTION}' \
     RANDOM_SEED='${SEED}' bash ops/launch/run_qwen35_sglang_server.sh \
     2>&1 | tee '${WORK_ROOT}/logs/local_${port}.log'"
  echo "[fleet] launched local ${session} GPUs=${gpus} port=${port}"
}

start_remote() {
  [[ "${FLEET_MODE}" == dual ]] || return 0
  if api_ready "${REMOTE_NODE}" 30000 && api_ready "${REMOTE_NODE}" 30001; then
    echo "[fleet] both remote endpoints already ready"
    return
  fi
  local status
  status=$(/usr/bin/python3 ops/workflows/rl_eval/ray_remote_sglang.py \
    --action status --target-node "${REMOTE_NODE}" --model-path "${MODEL_PATH}" \
    --served-name "${SERVED_NAME}" 2>/dev/null || true)
  if grep -q "sglang.launch_server.*Qwen3.5-27B" <<<"${status}"; then
    echo "[fleet] remote endpoints are already starting"
    return
  fi
  /usr/bin/python3 ops/workflows/rl_eval/ray_remote_sglang.py \
    --action launch --target-node "${REMOTE_NODE}" \
    --model-path "${MODEL_PATH}" --served-name "${SERVED_NAME}" \
    --tp-size 4 --context-length "${CONTEXT_LENGTH}" --mem-fraction "${MEM_FRACTION}" \
    --seed "${SEED}" --engine 0,1,2,3:30000 --engine 4,5,6,7:30001 \
    --log-dir "${WORK_ROOT}/logs"
}

wait_endpoint() {
  local host="$1" port="$2" deadline=$((SECONDS + 2400))
  until api_ready "${host}" "${port}"; do
    if (( SECONDS > deadline )); then
      echo "ERROR: ${host}:${port} not ready after 40 minutes" >&2
      return 1
    fi
    sleep 15
  done
  echo "[fleet] ready http://${host}:${port}/v1"
}

start_router() {
  local worker_urls="http://127.0.0.1:30000 http://127.0.0.1:30001"
  if [[ "${FLEET_MODE}" == dual ]]; then
    worker_urls+=" http://${REMOTE_NODE}:30000 http://${REMOTE_NODE}:30001"
  fi
  if api_ready 127.0.0.1 "${ROUTER_PORT}"; then
    echo "[fleet] router already ready"
    return
  fi
  if tmux has-session -t "${ROUTER_SESSION}" 2>/dev/null; then
    tmux kill-session -t "${ROUTER_SESSION}"
  fi
  tmux new-session -d -s "${ROUTER_SESSION}" \
    "cd '${ROOT}' && source /path/to/conda/etc/profile.d/conda.sh && \
     conda activate slime && NO_PROXY='${NO_PROXY}' no_proxy='${NO_PROXY}' \
     python -m sglang_router.launch_router --host 0.0.0.0 --port '${ROUTER_PORT}' \
       --worker-urls ${worker_urls} \
       --policy round_robin --prometheus-port '${PROMETHEUS_PORT}' \
       2>&1 | tee '${WORK_ROOT}/logs/router.log'"
}

builder() {
  python3 "${BUILDER}" "$@" --output-root "${OUTPUT_ROOT}" --work-root "${WORK_ROOT}"
}

fill_stage() {
  local command="$1" max_rounds="$2" round
  shift 2
  for ((round=1; round<=max_rounds; round++)); do
    echo "[pipeline] ${command} round ${round}/${max_rounds}"
    if builder "${command}" \
      --api-base "http://127.0.0.1:${ROUTER_PORT}/v1" --model "${SERVED_NAME}" \
      --workers "${GEN_WORKERS}" --attempts "${GEN_ATTEMPTS}" "$@"; then
      return 0
    fi
    sleep 5
  done
  echo "ERROR: ${command} did not converge after ${max_rounds} rounds" >&2
  return 1
}

teardown() {
  if [[ "${KEEP_SERVERS}" == "1" ]]; then
    if [[ "${FLEET_MODE}" == dual ]]; then
      echo "[fleet] KEEP_SERVERS=1; four 27B endpoints remain mounted"
    else
      echo "[fleet] KEEP_SERVERS=1; two local 27B endpoints remain mounted"
    fi
    return
  fi
  tmux kill-session -t "${ROUTER_SESSION}" 2>/dev/null || true
  tmux kill-session -t "${LOCAL_SESSION_0}" 2>/dev/null || true
  tmux kill-session -t "${LOCAL_SESSION_1}" 2>/dev/null || true
  if [[ "${FLEET_MODE}" == dual ]]; then
    /usr/bin/python3 ops/workflows/rl_eval/ray_remote_sglang.py \
      --action stop --target-node "${REMOTE_NODE}" --model-path "${MODEL_PATH}" \
      --served-name unused --log-dir "${WORK_ROOT}/logs" || true
  fi
}

{
  echo "[pipeline-start] $(date -Iseconds)"
  [[ -f "${MODEL_PATH}/config.json" ]] || { echo "ERROR: missing ${MODEL_PATH}/config.json"; exit 2; }

  resolve_fleet
  if [[ "${DRY_RUN}" == 1 ]]; then
    builder status
    echo "[dry-run] fleet and canonical snapshot preflight passed"
    exit 0
  fi
  builder prepare
  start_remote
  start_local "${LOCAL_SESSION_0}" 0,1,2,3 30000
  start_local "${LOCAL_SESSION_1}" 4,5,6,7 30001
  wait_endpoint 127.0.0.1 30000
  wait_endpoint 127.0.0.1 30001
  if [[ "${FLEET_MODE}" == dual ]]; then
    wait_endpoint "${REMOTE_NODE}" 30000
    wait_endpoint "${REMOTE_NODE}" 30001
  fi
  start_router
  wait_endpoint 127.0.0.1 "${ROUTER_PORT}"

  fill_stage generate-names 4
  oracle_audit_ok=0
  for cycle in 1 2 3 4 5 6; do
    echo "[pipeline] oracle generation/deep-audit cycle ${cycle}/6"
    fill_stage generate-oracles 6
    if builder audit-oracles \
      --api-base "http://127.0.0.1:${ROUTER_PORT}/v1" --model "${SERVED_NAME}" \
      --workers "${GEN_WORKERS}" --attempts "${GEN_ATTEMPTS}"; then
      oracle_audit_ok=1
      break
    fi
  done
  [[ "${oracle_audit_ok}" == "1" ]] || { echo "ERROR: oracle deep audit did not converge"; exit 1; }

  audit_ok=0
  for cycle in 1 2 3 4 5 6 7 8; do
    echo "[pipeline] misleading generation/audit cycle ${cycle}/8"
    fill_stage generate-misleading 8 --temperature 0.55
    if builder audit-misleading \
         --api-base "http://127.0.0.1:${ROUTER_PORT}/v1" --model "${SERVED_NAME}" \
         --workers "${GEN_WORKERS}" --attempts "${GEN_ATTEMPTS}"; then
      if builder audit-outcomes \
           --api-base "http://127.0.0.1:${ROUTER_PORT}/v1" --model "${SERVED_NAME}" \
           --workers "${GEN_WORKERS}" --attempts "${GEN_ATTEMPTS}"; then
        audit_ok=1
        break
      fi
    fi
  done
  [[ "${audit_ok}" == "1" ]] || { echo "ERROR: misleading audit did not converge"; exit 1; }

  builder finalize
  builder check
  builder status
  echo "[pipeline-done] $(date -Iseconds)"
  teardown
} 2>&1 | tee -a "${LOG}"
