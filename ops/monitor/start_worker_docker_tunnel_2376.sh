#!/usr/bin/env bash
set -Eeuo pipefail
session=docker-tunnel-2376-worker
mkdir -p /tmp/skillrl_docker_tunnel_2376_worker
tmux kill-session -t "$session" 2>/dev/null || true
tmux new -ds "$session" "DOCKER_HOST_VALUE='tcp://127.0.0.1:2376' LOCAL_BIND='127.0.0.1:2376' REMOTE_BIND='/var/run/docker.sock' SSH_TARGET='your-docker-host' CHECK_INTERVAL_SEC='15' DOCKER_CHECK_TIMEOUT_SEC='30' TUNNEL_RESTART_CONSECUTIVE_FAILURES='5' SSH_SERVER_ALIVE_INTERVAL_SEC='15' SSH_SERVER_ALIVE_COUNT_MAX='6' SSH_STALE_FORWARD_CLEANUP='1' SSH_FORWARD_PORT_FREE_TIMEOUT_SEC='10' SSH_LOG='/tmp/skillrl_docker_tunnel_2376_worker/ssh.log' bash '${SKILLRL_ROOT:-$(pwd)}/ops/monitor/docker_tunnel_watch.sh'"
for i in $(seq 1 30); do
  if DOCKER_HOST=tcp://127.0.0.1:2376 timeout 10 docker info --format 'root={{.DockerRootDir}} version={{.ServerVersion}}'; then
    ss -tlnp 2>/dev/null | grep ':2376' || true
    tmux capture-pane -pt "$session" -S -30 2>/dev/null || true
    exit 0
  fi
  sleep 1
done
echo 'FAILED tunnel smoke' >&2
tmux capture-pane -pt "$session" -S -100 2>/dev/null || true
tail -80 /tmp/skillrl_docker_tunnel_2376_worker/ssh.log 2>/dev/null || true
exit 1
