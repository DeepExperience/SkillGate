#!/usr/bin/env bash
set -Eeuo pipefail

# Start a local Docker daemon backed by an ext4 path, suitable for nested
# Docker eval/RL environments on QS containers.
#
# Why this exists:
# - `/` and ordinary `/tmp` in these containers are overlay-backed; nested
#   Docker overlay2 fails there.
# - `/data/cache` and `/data/temp` are ext4 mounts on this machine, so overlay2
#   works there and is much faster than the vfs fallback.
#
# Resume / reuse:
# - Re-running this script is idempotent if the socket is healthy.
# - The daemon root is intentionally under /data/cache by default. It is fast
#   but may not survive container recreation. Persist image tarballs separately
#   under a networked-storage path and restore with migrate_apex_images_to_local.py.

PROJECT_ROOT="${PROJECT_ROOT:-${SKILLRL_ROOT:-$(pwd)}}"
DATA_ROOT="${LOCAL_DOCKER_DATA_ROOT:-/data/cache/local-docker-overlay2-root}"
EXEC_ROOT="${LOCAL_DOCKER_EXEC_ROOT:-/data/cache/local-docker-overlay2-exec}"
SOCKET="${LOCAL_DOCKER_SOCKET:-/tmp/local-docker-overlay2.sock}"
ACTIVE_SOCKET_FILE="${LOCAL_DOCKER_ACTIVE_SOCKET_FILE:-/tmp/local-docker-active.sock}"
LOG="${LOCAL_DOCKER_LOG:-/tmp/local-dockerd-overlay2.log}"
TMUX_SESSION="${LOCAL_DOCKER_TMUX_SESSION:-local_dockerd_overlay2}"

mkdir -p "${DATA_ROOT}" "${EXEC_ROOT}" "$(dirname "${SOCKET}")"

if [[ -S "${SOCKET}" ]] && DOCKER_HOST="unix://${SOCKET}" docker info >/dev/null 2>&1; then
  echo "${SOCKET}" > "${ACTIVE_SOCKET_FILE}"
  DOCKER_HOST="unix://${SOCKET}" docker info --format \
    'local docker already healthy: Server={{.ServerVersion}} Driver={{.Driver}} Root={{.DockerRootDir}}'
  exit 0
fi

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${TMUX_SESSION}" || true
fi

rm -f "${SOCKET}"

for var in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy NO_PROXY no_proxy; do
  if [[ -n "${!var:-}" ]]; then
    tmux set-environment -g "${var}" "${!var}" || true
  fi
done

# --log-opt caps each container's dockerd-side json-log (a container flooding
# stdout grows that file OUTSIDE its writable layer -> invisible to `docker ps -s`
# and to in-container ulimits -> can fill the node disk and trip kubelet's
# ephemeral-storage eviction; see docs/rl_log 2026-06-10 20:25 post-mortem).
DOCKERD_CMD="dockerd --host=unix://${SOCKET} --data-root=${DATA_ROOT} --exec-root=${EXEC_ROOT} --pidfile=${EXEC_ROOT}/dockerd.pid --storage-driver=overlay2 --log-opt max-size=${LOCAL_DOCKER_LOG_MAX_SIZE:-64m} --log-opt max-file=${LOCAL_DOCKER_LOG_MAX_FILE:-2}"

# Optional: run dockerd under a PR_SET_CHILD_SUBREAPER wrapper so orphaned
# containerd-shim processes get reaped here instead of piling up under a
# non-reaping PID1 (KubeRay's `ray start --block`). Prevents the shim-accumulation
# control-plane meltdown documented in docs/rl_log 2026-06-08. Opt-in, no-op by default.
if [[ "${LOCAL_DOCKER_USE_SUBREAPER:-0}" == "1" ]]; then
  SUBREAPER_PY="${LOCAL_DOCKER_SUBREAPER_PY:-${PROJECT_ROOT}/ops/launch/subreaper_exec.py}"
  SUBREAPER_PYBIN="${LOCAL_DOCKER_SUBREAPER_PYBIN:-/usr/bin/python3}"
  echo "starting dockerd under subreaper: ${SUBREAPER_PYBIN} ${SUBREAPER_PY}"
  tmux new-session -d -s "${TMUX_SESSION}" \
    "exec ${SUBREAPER_PYBIN} ${SUBREAPER_PY} ${DOCKERD_CMD} >${LOG} 2>&1"
else
  tmux new-session -d -s "${TMUX_SESSION}" \
    "exec ${DOCKERD_CMD} >${LOG} 2>&1"
fi

for _ in $(seq 1 80); do
  if [[ -S "${SOCKET}" ]] && DOCKER_HOST="unix://${SOCKET}" docker info >/dev/null 2>&1; then
    echo "${SOCKET}" > "${ACTIVE_SOCKET_FILE}"
    DOCKER_HOST="unix://${SOCKET}" docker info --format \
      'local docker ready: Server={{.ServerVersion}} Driver={{.Driver}} Root={{.DockerRootDir}}'
    exit 0
  fi
  sleep 0.5
done

echo "ERROR: local overlay2 dockerd did not become healthy" >&2
tail -120 "${LOG}" >&2 || true
exit 1
