#!/usr/bin/env bash
set -Eeuo pipefail

DOCKER_HOST_VALUE="${DOCKER_HOST_VALUE:-tcp://127.0.0.1:2375}"
SSH_TARGET="${SSH_TARGET:-your-docker-host}"
LOCAL_BIND="${LOCAL_BIND:-127.0.0.1:2375}"
REMOTE_BIND="${REMOTE_BIND:-127.0.0.1:2375}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-30}"
DOCKER_CHECK_TIMEOUT_SEC="${DOCKER_CHECK_TIMEOUT_SEC:-30}"
TUNNEL_RESTART_CONSECUTIVE_FAILURES="${TUNNEL_RESTART_CONSECUTIVE_FAILURES:-5}"
SSH_CONNECT_TIMEOUT_SEC="${SSH_CONNECT_TIMEOUT_SEC:-10}"
SSH_SERVER_ALIVE_INTERVAL_SEC="${SSH_SERVER_ALIVE_INTERVAL_SEC:-15}"
SSH_SERVER_ALIVE_COUNT_MAX="${SSH_SERVER_ALIVE_COUNT_MAX:-6}"
SSH_STALE_FORWARD_CLEANUP="${SSH_STALE_FORWARD_CLEANUP:-1}"
SSH_FORWARD_PORT_FREE_TIMEOUT_SEC="${SSH_FORWARD_PORT_FREE_TIMEOUT_SEC:-10}"
SSH_FORWARD_TERM_GRACE_SEC="${SSH_FORWARD_TERM_GRACE_SEC:-3}"
SSH_LOG="${SSH_LOG:-/tmp/docker_tunnel_watch_ssh.log}"
CURRENT_SSH_PID=""
CONSECUTIVE_FAILURES=0

export DOCKER_HOST="${DOCKER_HOST_VALUE}"
mkdir -p "$(dirname "${SSH_LOG}")"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

docker_ok() {
  DOCKER_HOST="${DOCKER_HOST_VALUE}" timeout "${DOCKER_CHECK_TIMEOUT_SEC}" docker version --format '{{.Server.Version}}' >/dev/null 2>&1
}

stop_forward() {
  if [ -n "${CURRENT_SSH_PID}" ]; then
    if kill -0 "${CURRENT_SSH_PID}" >/dev/null 2>&1; then
      kill -- "-${CURRENT_SSH_PID}" >/dev/null 2>&1 || kill "${CURRENT_SSH_PID}" >/dev/null 2>&1 || true
      sleep "${SSH_FORWARD_TERM_GRACE_SEC}"
      if kill -0 "${CURRENT_SSH_PID}" >/dev/null 2>&1; then
        kill -KILL -- "-${CURRENT_SSH_PID}" >/dev/null 2>&1 || kill -KILL "${CURRENT_SSH_PID}" >/dev/null 2>&1 || true
      fi
    fi
    wait "${CURRENT_SSH_PID}" >/dev/null 2>&1 || true
    CURRENT_SSH_PID=""
  fi
}

local_bind_port() {
  printf '%s\n' "${LOCAL_BIND##*:}"
}

local_bind_is_free() {
  local port
  port="$(local_bind_port)"
  ! ss -H -ltn "sport = :${port}" 2>/dev/null | grep -q .
}

wait_local_bind_free() {
  local timeout_sec="${1:-10}"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if local_bind_is_free; then
      return 0
    fi
    if [ $(( $(date +%s) - start_ts )) -ge "${timeout_sec}" ]; then
      return 1
    fi
    sleep 1
  done
}

find_stale_forward_pids() {
  ps -eo pid=,args= \
    | awk -v local_bind="${LOCAL_BIND}" -v remote_bind="${REMOTE_BIND}" -v target="${SSH_TARGET}" '
        $0 ~ /[s]sh / &&
        index($0, "-L " local_bind ":" remote_bind) &&
        index($0, target) {
          print $1
        }
      '
}

local_bind_listener_pids() {
  local port
  port="$(local_bind_port)"
  ss -H -ltnp "sport = :${port}" 2>/dev/null \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
    | sort -u
}

ssh_forward_cmd_matches() {
  local pid="$1"
  local args
  args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
  [ -n "${args}" ] || return 1
  printf '%s\n' "${args}" | grep -Eq '(^|/| )ssh( |$)' || return 1
  printf '%s\n' "${args}" | grep -F -- "-L ${LOCAL_BIND}:${REMOTE_BIND}" >/dev/null || return 1
}

kill_stale_forward_pid() {
  local pid="$1"
  if ! ssh_forward_cmd_matches "${pid}"; then
    log "not killing listener pid=${pid}; command does not match expected ssh forward"
    ps -p "${pid}" -o pid=,ppid=,stat=,etime=,args= 2>/dev/null || true
    return 1
  fi
  log "stopping stale ssh forward pid=${pid} for ${LOCAL_BIND}:${REMOTE_BIND}"
  kill "${pid}" >/dev/null 2>&1 || true
  sleep "${SSH_FORWARD_TERM_GRACE_SEC}"
  if kill -0 "${pid}" >/dev/null 2>&1; then
    log "stale ssh forward pid=${pid} survived TERM; sending KILL"
    kill -KILL "${pid}" >/dev/null 2>&1 || true
  fi
}

stop_stale_forwards() {
  if [ "${SSH_STALE_FORWARD_CLEANUP}" != "1" ]; then
    return 0
  fi
  local pid
  local found=0
  while read -r pid; do
    [ -n "${pid}" ] || continue
    if [ -n "${CURRENT_SSH_PID}" ] && [ "${pid}" = "${CURRENT_SSH_PID}" ]; then
      continue
    fi
    found=1
    kill_stale_forward_pid "${pid}" || true
  done < <(find_stale_forward_pids)
  if [ "${found}" = "1" ]; then
    wait_local_bind_free "${SSH_FORWARD_PORT_FREE_TIMEOUT_SEC}" || true
  fi
}

clear_local_bind_if_owned_by_forward() {
  local pid
  local found=0
  while read -r pid; do
    [ -n "${pid}" ] || continue
    found=1
    kill_stale_forward_pid "${pid}" || true
  done < <(local_bind_listener_pids)
  if [ "${found}" = "1" ]; then
    wait_local_bind_free "${SSH_FORWARD_PORT_FREE_TIMEOUT_SEC}" || true
  fi
}

cleanup() {
  stop_forward
}
trap cleanup EXIT INT TERM

start_forward() {
  stop_forward
  stop_stale_forwards
  if ! wait_local_bind_free "${SSH_FORWARD_PORT_FREE_TIMEOUT_SEC}"; then
    log "local bind ${LOCAL_BIND} still occupied before starting ssh forward; recent listeners:"
    ss -ltnp 2>/dev/null | grep "$(local_bind_port)" || true
    clear_local_bind_if_owned_by_forward
  fi
  if ! wait_local_bind_free "${SSH_FORWARD_PORT_FREE_TIMEOUT_SEC}"; then
    log "local bind ${LOCAL_BIND} still occupied after cleanup; refusing to start duplicate ssh forward"
    ss -ltnp 2>/dev/null | grep "$(local_bind_port)" || true
    return 1
  fi
  log "starting ssh forward ${LOCAL_BIND}:${REMOTE_BIND} via ${SSH_TARGET}"
  setsid bash -c '
    ssh "$@" 2>&1 | while IFS= read -r line; do
      printf "[%s] %s\n" "$(date -Is)" "${line}"
    done
  ' -- \
    -n \
    -o BatchMode=yes \
    -o ControlMaster=no \
    -o ControlPath=none \
    -o ConnectTimeout="${SSH_CONNECT_TIMEOUT_SEC}" \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval="${SSH_SERVER_ALIVE_INTERVAL_SEC}" \
    -o ServerAliveCountMax="${SSH_SERVER_ALIVE_COUNT_MAX}" \
    -N \
    -L "${LOCAL_BIND}:${REMOTE_BIND}" \
    "${SSH_TARGET}" >>"${SSH_LOG}" 2>&1 &
  local ssh_pid=$!
  CURRENT_SSH_PID="${ssh_pid}"
  log "ssh forward launched pid=${ssh_pid} log=${SSH_LOG}"
  for _ in $(seq 1 10); do
    if docker_ok; then
      return 0
    fi
    if ! kill -0 "${ssh_pid}" >/dev/null 2>&1; then
      wait "${ssh_pid}" >/dev/null 2>&1 || true
      CURRENT_SSH_PID=""
      log "ssh forward exited early pid=${ssh_pid}; recent ssh log:"
      tail -20 "${SSH_LOG}" 2>/dev/null || true
      return 1
    fi
    sleep 1
  done
  return 0
}

log "watching docker endpoint ${DOCKER_HOST_VALUE} via ${SSH_TARGET} every ${CHECK_INTERVAL_SEC}s (check_timeout=${DOCKER_CHECK_TIMEOUT_SEC}s restart_after=${TUNNEL_RESTART_CONSECUTIVE_FAILURES} failures server_alive=${SSH_SERVER_ALIVE_INTERVAL_SEC}x${SSH_SERVER_ALIVE_COUNT_MAX} stale_cleanup=${SSH_STALE_FORWARD_CLEANUP})"

while true; do
  if docker_ok; then
    if [ "${CONSECUTIVE_FAILURES}" -gt 0 ]; then
      log "docker endpoint healthy again after ${CONSECUTIVE_FAILURES} failed check(s)"
    fi
    CONSECUTIVE_FAILURES=0
    sleep "${CHECK_INTERVAL_SEC}"
    continue
  fi
  CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
  if [ "${CONSECUTIVE_FAILURES}" -lt "${TUNNEL_RESTART_CONSECUTIVE_FAILURES}" ]; then
    log "docker endpoint check failed ${CONSECUTIVE_FAILURES}/${TUNNEL_RESTART_CONSECUTIVE_FAILURES}; not restarting tunnel yet"
    sleep "${CHECK_INTERVAL_SEC}"
    continue
  fi
  log "docker endpoint unavailable after ${CONSECUTIVE_FAILURES} consecutive failed checks; restarting tunnel"
  if start_forward; then
    sleep 5
    if docker_ok; then
      log "docker endpoint restored"
      CONSECUTIVE_FAILURES=0
    else
      log "docker endpoint still unavailable after tunnel restart"
    fi
  else
    log "ssh forward start failed"
  fi
  sleep "${CHECK_INTERVAL_SEC}"
done
