#!/usr/bin/env bash
set -Eeuo pipefail

# Supplemental multi-node bootstrap for Relax training.
# This does not replace your per-node init scripts. Run those first on the
# fresh machines, then use this from the Ray head to make the shared runtime
# assumptions explicit and smoke-testable.

PROJECTS_DIR="${PROJECTS_DIR:-${SKILLRL_ROOT:-$(pwd)}}"
PERSIST_DIR="${PERSIST_DIR:-$HOME}"
RELAX_DIR="${RELAX_DIR:-${PROJECTS_DIR}/Relax}"
RUN_ON_EACH="${RUN_ON_EACH:-${RELAX_DIR}/scripts/tools/run_on_each_ray_node.py}"
TUNNEL_SCRIPT="${TUNNEL_SCRIPT:-${PROJECTS_DIR}/ops/monitor/docker_tunnel_watch.sh}"
CUDA_FAST_HOME="${CUDA_FAST_HOME:-${PROJECTS_DIR}/ops/cache/cuda_fast_home}"

PYTHON_BIN="${PYTHON_BIN:-${RELAX_PYTHON:-/usr/bin/python3}}"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="$(command -v python3)"
fi

DOCKER_HOST_VALUE="${DOCKER_HOST_VALUE:-ssh://your-docker-host}"
LOCAL_BIND="${LOCAL_BIND:-127.0.0.1:2376}"
LOCAL_PORT="${LOCAL_BIND##*:}"
REMOTE_BIND="${REMOTE_BIND:-/var/run/docker.sock}"
SSH_TARGET="${SSH_TARGET:-your-docker-host}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-15}"
HEAD_TMUX_SESSION="${HEAD_TMUX_SESSION:-docker-tunnel-2376-head}"
WORKER_TMUX_SESSION="${WORKER_TMUX_SESSION:-docker-tunnel-2376-worker}"
FORCE_RESTART_TUNNELS="${FORCE_RESTART_TUNNELS:-0}"

log() {
  printf '[bootstrap_multinode][%s] %s\n' "$(date -Is)" "$*"
}

die() {
  printf '[bootstrap_multinode] ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  [ -f "$1" ] || die "missing file: $1"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

docker_server_version() {
  DOCKER_HOST="${DOCKER_HOST_VALUE}" timeout 10 docker version --format '{{.Server.Version}}'
}

start_head_tunnel_watchdog() {
  if [[ "${DOCKER_HOST_VALUE}" == ssh://* ]]; then
    log "using Docker-over-SSH on head: ${DOCKER_HOST_VALUE}; no local tunnel watchdog needed"
    if docker_server_version >/tmp/bootstrap_multinode_head_docker_version.out 2>&1; then
      log "head docker server version: $(cat /tmp/bootstrap_multinode_head_docker_version.out)"
      return 0
    fi
    cat /tmp/bootstrap_multinode_head_docker_version.out >&2 || true
    die "head Docker-over-SSH check failed"
  fi

  log "checking head docker tunnel ${DOCKER_HOST_VALUE}"
  if [ "${FORCE_RESTART_TUNNELS}" = "1" ]; then
    tmux kill-session -t "${HEAD_TMUX_SESSION}" 2>/dev/null || true
  fi

  if tmux has-session -t "${HEAD_TMUX_SESSION}" 2>/dev/null; then
    log "head tmux session already exists: ${HEAD_TMUX_SESSION}"
  else
    log "starting head tmux session: ${HEAD_TMUX_SESSION}"
    tmux new -ds "${HEAD_TMUX_SESSION}" \
      "DOCKER_HOST_VALUE='${DOCKER_HOST_VALUE}' LOCAL_BIND='${LOCAL_BIND}' REMOTE_BIND='${REMOTE_BIND}' SSH_TARGET='${SSH_TARGET}' CHECK_INTERVAL_SEC='${CHECK_INTERVAL_SEC}' SSH_LOG='/tmp/skillrl_docker_tunnel_2376_head/ssh.log' bash '${TUNNEL_SCRIPT}'"
  fi

  log "waiting for head docker tunnel"
  for _ in $(seq 1 20); do
    if docker_server_version >/tmp/bootstrap_multinode_head_docker_version.out 2>&1; then
      log "head docker server version: $(cat /tmp/bootstrap_multinode_head_docker_version.out)"
      return 0
    fi
    sleep 1
  done

  tmux capture-pane -pt "${HEAD_TMUX_SESSION}" -S -80 2>/dev/null || true
  die "head docker tunnel did not become healthy"
}

discover_worker_ips() {
  if [ -n "${RAY_WORKER_IPS:-}" ]; then
    printf '%s\n' "${RAY_WORKER_IPS}" | tr ',' '\n' | awk 'NF {print $1}'
    return 0
  fi

  RAY_ADDRESS="${RAY_ADDRESS:-auto}" "${PYTHON_BIN}" - <<'PY'
import os
import ray

ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), ignore_reinit_error=True, logging_level="ERROR")
for node in ray.nodes():
    resources = node.get("Resources", {})
    if node.get("Alive") and resources.get("GPU", 0) > 0:
        print(node["NodeManagerAddress"])
ray.shutdown()
PY
}

run_on_worker() {
  local worker_ip="$1"
  local command="$2"
  RAY_CMD_TIMEOUT="${RAY_CMD_TIMEOUT:-120}" "${PYTHON_BIN}" "${RUN_ON_EACH}" -t "${RAY_CMD_TIMEOUT:-120}" -n "${worker_ip}" "${command}"
}

start_worker_tunnel_watchdog() {
  local worker_ip="$1"
  local remote_cmd

  if [[ "${DOCKER_HOST_VALUE}" == ssh://* ]]; then
    read -r -d '' remote_cmd <<EOF || true
set -Eeuo pipefail
DOCKER_HOST='${DOCKER_HOST_VALUE}' timeout 15 docker version --format '{{.Server.Version}}'
EOF

    log "checking worker Docker-over-SSH on ${worker_ip}: ${DOCKER_HOST_VALUE}"
    run_on_worker "${worker_ip}" "${remote_cmd}"
    return 0
  fi

  read -r -d '' remote_cmd <<EOF || true
set -Eeuo pipefail
session='${WORKER_TMUX_SESSION}'
if [ '${FORCE_RESTART_TUNNELS}' = '1' ]; then
  tmux kill-session -t "\${session}" 2>/dev/null || true
fi
if tmux has-session -t "\${session}" 2>/dev/null; then
  echo "worker tmux session already exists: \${session}"
else
  echo "starting worker tmux session: \${session}"
  tmux new -ds "\${session}" "DOCKER_HOST_VALUE='${DOCKER_HOST_VALUE}' LOCAL_BIND='${LOCAL_BIND}' REMOTE_BIND='${REMOTE_BIND}' SSH_TARGET='${SSH_TARGET}' CHECK_INTERVAL_SEC='${CHECK_INTERVAL_SEC}' SSH_LOG='/tmp/skillrl_docker_tunnel_2376_worker/ssh.log' bash '${TUNNEL_SCRIPT}'"
fi
for i in \$(seq 1 20); do
  if DOCKER_HOST='${DOCKER_HOST_VALUE}' timeout 10 docker version --format '{{.Server.Version}}'; then
    ss -tlnp 2>/dev/null | grep ':${LOCAL_PORT}' || true
    exit 0
  fi
  sleep 1
done
tmux capture-pane -pt "\${session}" -S -80 2>/dev/null || true
exit 1
EOF

  log "checking worker docker tunnel on ${worker_ip}"
  run_on_worker "${worker_ip}" "${remote_cmd}"
}

check_cuda_fast_home() {
  log "checking CUDA fast shim: ${CUDA_FAST_HOME}"
  [ -d "${CUDA_FAST_HOME}" ] || die "missing CUDA fast shim dir: ${CUDA_FAST_HOME}"

  local name
  for name in bin include lib64 nvvm targets; do
    [ -L "${CUDA_FAST_HOME}/${name}" ] || die "missing symlink: ${CUDA_FAST_HOME}/${name}"
  done
  ls -la "${CUDA_FAST_HOME}"
}

smoke_worker() {
  local worker_ip="$1"
  local remote_cmd

  read -r -d '' remote_cmd <<EOF || true
set -Eeuo pipefail
echo "host=\$(hostname)"
echo "docker=\$(DOCKER_HOST='${DOCKER_HOST_VALUE}' timeout 10 docker version --format '{{.Server.Version}}')"
echo "harbor=\$(harbor --version)"
echo "cuda_fast_home:"
ls -la '${CUDA_FAST_HOME}'
EOF

  log "smoke testing worker ${worker_ip}"
  run_on_worker "${worker_ip}" "${remote_cmd}"
}

main() {
  cd "${PROJECTS_DIR}"

  require_cmd tmux
  require_cmd docker
  require_cmd ray
  require_file "${RUN_ON_EACH}"
  require_file "${TUNNEL_SCRIPT}"

  log "projects dir: ${PROJECTS_DIR}"
  log "python: ${PYTHON_BIN} ($("${PYTHON_BIN}" --version 2>&1))"

  local worker_ips=()
  mapfile -t worker_ips < <(discover_worker_ips)
  [ "${#worker_ips[@]}" -gt 0 ] || die "no GPU worker IPs found; set RAY_WORKER_IPS or check ray.nodes()"
  log "worker IPs: ${worker_ips[*]}"

  log "ray status"
  ray status

  start_head_tunnel_watchdog

  local worker_ip
  for worker_ip in "${worker_ips[@]}"; do
    start_worker_tunnel_watchdog "${worker_ip}"
  done

  check_cuda_fast_home

  log "head smoke tests"
  docker_server_version
  harbor --version

  for worker_ip in "${worker_ips[@]}"; do
    smoke_worker "${worker_ip}"
  done

  log "done; use CUDA_HOME=${CUDA_FAST_HOME} and DOCKER_HOST=${DOCKER_HOST_VALUE} in the Relax launcher/runtime env"
}

main "$@"
