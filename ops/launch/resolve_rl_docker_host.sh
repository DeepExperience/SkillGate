#!/usr/bin/env bash
# Resolve the Docker endpoint used by Relax/Unified Runner.
#
# Default policy:
#   1. Prefer the maintained local overlay2 Docker socket. It avoids
#      Docker-over-SSH sessions and keeps high-concurrency RL/eval traffic off
#      the shared remote Docker daemon.
#   2. If local Docker is absent or unhealthy, fall back to the old tunnel path.
#   3. If the tunnel is still unhealthy, fall back to ssh://your-docker-host quickly.
#
# Strict local mode:
#   Set RL_DOCKER_REQUIRE_LOCAL=1 to fail fast instead of silently falling back
#   to slower remote endpoints.
#
# The chosen endpoint is exported as both DOCKER_HOST and RELAX_DOCKER_HOST.
set -Eeuo pipefail

ROOT="${ROOT:-${SKILLRL_ROOT:-$(pwd)}}"
RL_DOCKER_LOCAL_HOST="${RL_DOCKER_LOCAL_HOST:-unix:///tmp/local-docker-overlay2.sock}"
RL_DOCKER_TUNNEL_HOST="${RL_DOCKER_TUNNEL_HOST:-tcp://127.0.0.1:2376}"
RL_DOCKER_SSH_HOST="${RL_DOCKER_SSH_HOST:-ssh://your-docker-host}"
RL_DOCKER_PREFERRED="${RL_DOCKER_PREFERRED:-auto}"   # auto | local | tunnel | ssh
RL_DOCKER_REQUIRE_LOCAL="${RL_DOCKER_REQUIRE_LOCAL:-0}"
RL_DOCKER_AUTO_START_TUNNEL="${RL_DOCKER_AUTO_START_TUNNEL:-1}"
RL_DOCKER_REQUIRE_TUNNEL="${RL_DOCKER_REQUIRE_TUNNEL:-0}"
RL_DOCKER_TUNNEL_START_SCRIPT="${RL_DOCKER_TUNNEL_START_SCRIPT:-${ROOT}/ops/monitor/start_worker_docker_tunnel_2376.sh}"

rl_docker_log() {
  printf '[resolve_rl_docker_host] %s\n' "$*" >&2
}

rl_docker_ok() {
  local host="$1"
  DOCKER_HOST="${host}" timeout 10 docker info --format '{{.ServerVersion}} {{.DockerRootDir}}' >/tmp/resolve_rl_docker_host.out 2>&1
}

rl_choose_docker_host() {
  local chosen=""

  if [ "${RL_DOCKER_PREFERRED}" = "local" ] || [ "${RL_DOCKER_PREFERRED}" = "auto" ]; then
    if rl_docker_ok "${RL_DOCKER_LOCAL_HOST}"; then
      chosen="${RL_DOCKER_LOCAL_HOST}"
      rl_docker_log "using local Docker endpoint ${chosen}: $(cat /tmp/resolve_rl_docker_host.out)"
    elif [ "${RL_DOCKER_REQUIRE_LOCAL}" = "1" ]; then
      rl_docker_log "ERROR: local Docker required but unavailable: ${RL_DOCKER_LOCAL_HOST}"
      cat /tmp/resolve_rl_docker_host.out >&2 || true
      return 1
    else
      rl_docker_log "local Docker unavailable; falling back to tunnel/ssh: ${RL_DOCKER_LOCAL_HOST}"
      cat /tmp/resolve_rl_docker_host.out >&2 || true
    fi
  fi

  if [ -z "${chosen}" ] && [ "${RL_DOCKER_PREFERRED}" != "ssh" ] && [ "${RL_DOCKER_PREFERRED}" != "local" ]; then
    if rl_docker_ok "${RL_DOCKER_TUNNEL_HOST}"; then
      chosen="${RL_DOCKER_TUNNEL_HOST}"
      rl_docker_log "using tunnel Docker endpoint ${chosen}: $(cat /tmp/resolve_rl_docker_host.out)"
    elif [ "${RL_DOCKER_AUTO_START_TUNNEL}" = "1" ] && [ -x "${RL_DOCKER_TUNNEL_START_SCRIPT}" ]; then
      rl_docker_log "tunnel ${RL_DOCKER_TUNNEL_HOST} unhealthy; starting watchdog via ${RL_DOCKER_TUNNEL_START_SCRIPT}"
      if timeout 45 bash "${RL_DOCKER_TUNNEL_START_SCRIPT}" >/tmp/resolve_rl_docker_tunnel_start.out 2>&1; then
        if rl_docker_ok "${RL_DOCKER_TUNNEL_HOST}"; then
          chosen="${RL_DOCKER_TUNNEL_HOST}"
          rl_docker_log "using restored tunnel Docker endpoint ${chosen}: $(cat /tmp/resolve_rl_docker_host.out)"
        fi
      else
        rl_docker_log "tunnel start failed; recent output:"
        tail -40 /tmp/resolve_rl_docker_tunnel_start.out >&2 || true
      fi
    fi

    if [ -z "${chosen}" ] && [ "${RL_DOCKER_REQUIRE_TUNNEL}" = "1" ]; then
      rl_docker_log "ERROR: tunnel required but unavailable: ${RL_DOCKER_TUNNEL_HOST}"
      return 1
    fi
  fi

  if [ -z "${chosen}" ]; then
    if rl_docker_ok "${RL_DOCKER_SSH_HOST}"; then
      chosen="${RL_DOCKER_SSH_HOST}"
      rl_docker_log "using fallback Docker-over-SSH endpoint ${chosen}: $(cat /tmp/resolve_rl_docker_host.out)"
    else
      rl_docker_log "ERROR: fallback Docker-over-SSH endpoint failed: ${RL_DOCKER_SSH_HOST}"
      cat /tmp/resolve_rl_docker_host.out >&2 || true
      return 1
    fi
  fi

  export DOCKER_HOST="${chosen}"
  export RELAX_DOCKER_HOST="${chosen}"
  export RL_DOCKER_HOST_RESOLVED="${chosen}"
}

rl_choose_docker_host
