#!/usr/bin/env bash
# One-command preparation for multi-node Relax RL after a fresh container reset.
#
# Assumption: run this from a node where your per-node init script has already been applied and
# the Ray cluster is visible. The script initializes all GPU Ray workers,
# resolves the RL Docker endpoint, and runs the local prelaunch health gate.
# The old remote-Docker tunnel bootstrap is opt-in via PREPARE_REMOTE_DOCKER_TUNNEL=1.
set -Eeuo pipefail

ROOT="${ROOT:-${SKILLRL_ROOT:-$(pwd)}}"
PERSIST_DIR="${PERSIST_DIR:-$HOME}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
RAY_WORKER_INIT="${RAY_WORKER_INIT:-${PERSIST_DIR}/ray_worker_init.sh}"
RUN_ON_EACH="${RUN_ON_EACH:-${ROOT}/Relax/scripts/tools/run_on_each_ray_node.py}"
DOCKER_HOST_VALUE="${DOCKER_HOST_VALUE:-tcp://127.0.0.1:2376}"
RAY_CMD_TIMEOUT="${RAY_CMD_TIMEOUT:-900}"
FORCE_RESTART_TUNNELS="${FORCE_RESTART_TUNNELS:-0}"
PREPARE_REMOTE_DOCKER_TUNNEL="${PREPARE_REMOTE_DOCKER_TUNNEL:-0}"

cd "${ROOT}"

echo "[prepare_multinode_rl] initializing GPU workers with ${RAY_WORKER_INIT}"
RAY_CMD_TIMEOUT="${RAY_CMD_TIMEOUT}" "${PYTHON_BIN}" "${RUN_ON_EACH}" -t "${RAY_CMD_TIMEOUT}" "bash '${RAY_WORKER_INIT}'"

if [[ "${PREPARE_REMOTE_DOCKER_TUNNEL}" == "1" ]]; then
  echo "[prepare_multinode_rl] bootstrapping legacy remote Docker tunnel and worker smoke tests"
  DOCKER_HOST_VALUE="${DOCKER_HOST_VALUE}" \
  FORCE_RESTART_TUNNELS="${FORCE_RESTART_TUNNELS}" \
  RAY_CMD_TIMEOUT=180 \
  bash "${ROOT}/ops/launch/bootstrap_multinode.sh"
else
  echo "[prepare_multinode_rl] skipping legacy remote Docker tunnel bootstrap; use PREPARE_REMOTE_DOCKER_TUNNEL=1 for the old remote-Docker path"
fi

echo "[prepare_multinode_rl] resolving Docker endpoint"
source "${ROOT}/ops/launch/resolve_rl_docker_host.sh"
echo "[prepare_multinode_rl] resolved Docker endpoint: ${RELAX_DOCKER_HOST}"

echo "[prepare_multinode_rl] local health gate"
python3 "${ROOT}/ops/health/rl_prelaunch_check.py" --strict

echo "[prepare_multinode_rl] done"
