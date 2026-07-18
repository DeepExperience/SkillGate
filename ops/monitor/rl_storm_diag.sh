#!/usr/bin/env bash
# RL storm root-cause monitor (local overlay2 docker).
# Every INTERVAL s logs a lightweight heartbeat. When trouble is detected
# (D-state spike / docker-ps latency / fork-bomb / disk near-full) it dumps a
# DEEP SNAPSHOT that classifies the storm WITHOUT guessing:
#   - disk-flush        -> D-state wchan flush_workqueue/rq_qos_wait + disk% high
#   - control-plane      -> docker ps slow + D-state runc/bridge/iptables, disk ok
#   - stale containers   -> many containers whose embedded RolloutMgr PID != current
#   - heavy containers   -> few current containers with high PID/RSS (dd/qemu/pip)
#   - fork bomb          -> docker_cleanup.sh process count > 0
# Usage: rl_storm_diag.sh <driver_log> [interval_s]
set -uo pipefail
export PATH="${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/envs/slime/bin:/root/.local/bin:$PATH"
DH="${RL_DOCKER_LOCAL_HOST:-unix:///tmp/local-docker-overlay2.sock}"
DRIVER_LOG="${1:-}"
INTERVAL="${2:-25}"
DOCK_DISK="${DOCK_DISK:-/tmp/ray}"
ROOT="${ROOT:-${SKILLRL_ROOT:-$(pwd)}}"
OUT="${RL_STORM_OUT:-}"
if [[ -z "$OUT" && -n "$DRIVER_LOG" ]]; then
  OUT="$(dirname "$DRIVER_LOG")/diagnostics/storm"
elif [[ -z "$OUT" && -n "${EXPERIMENT_ID:-}" && -n "${RUN_NAME:-}" ]]; then
  OUT="${ROOT}/experiments/rl/runs/${EXPERIMENT_ID}/segments/${RUN_NAME}/diagnostics/storm"
elif [[ -z "$OUT" ]]; then
  echo "FATAL: set RL_STORM_OUT, pass a driver log, or export EXPERIMENT_ID and RUN_NAME" >&2
  exit 2
fi
mkdir -p "$OUT"
HEART="$OUT/heartbeat.log"
DEEP="$OUT/deep_snapshots.log"
DSTATE_TRIG="${DSTATE_TRIG:-40}"     # D-state count to trigger deep snapshot
DPS_TRIG="${DPS_TRIG:-8}"            # docker ps latency (s) to trigger
DISK_TRIG="${DISK_TRIG:-90}"        # disk % to trigger

dcur() { timeout 20 docker -H "$DH" "$@" 2>/dev/null; }
rollout_pid() { ps -eo pid,cmd --no-headers 2>/dev/null | grep -E "RolloutManager|sglang_rollout" | grep -v grep | awk '{print $1}' | head -1; }

deep_snapshot() {
  local reason="$1" ts; ts=$(date '+%F %T')
  { echo "================ DEEP SNAPSHOT @ $ts  reason=$reason ================"
    echo "--- load/procs ---"; cat /proc/loadavg; echo "procs=$(ps -e --no-headers|wc -l) Z=$(ps -eo stat --no-headers|grep -c Z)"
    echo "--- D-state wchan histogram (flush_workqueue/rq_qos=disk; runc/bridge/iptables=ctrl-plane; rwsem=lock) ---"
    ps -eo stat,wchan:34,comm --no-headers 2>/dev/null | awk '$1 ~ /^D/{print $2}' | sort | uniq -c | sort -rn | head -12
    echo "--- D-state processes (top 25, cmd) ---"
    ps -eo stat,pid,wchan:28,args --no-headers 2>/dev/null | awk '$1 ~ /^D/' | cut -c1-160 | head -25
    echo "--- docker disk ---"; df -h "$DOCK_DISK" 2>/dev/null | tail -1
    echo "--- fork-bomb (docker_cleanup.sh) count ---"; ps -e --no-headers 2>/dev/null | grep -c docker_cleanup
    local cur; cur=$(rollout_pid); echo "--- current RolloutManager pid=$cur ; container current-vs-stale by embedded p<PID> ---"
    dcur ps --format '{{.Names}}' | grep -oE 'p[0-9]+' | sort | uniq -c | sort -rn | head -8
    echo "  (a p<PID> != $cur with many containers => STALE leftovers)"
    echo "--- container count + statuses ---"; dcur ps -a --format '{{.Status}}' | awk '{print $1}' | sort | uniq -c | sort -rn | head
    echo "--- heaviest running containers (PIDs, top 8) ---"
    for c in $(dcur ps -q | head -60); do echo "$(timeout 5 docker -H "$DH" top "$c" 2>/dev/null | tail -n +2 | wc -l) $c"; done 2>/dev/null | sort -rn | head -8
    echo "--- docker info disk ---"; dcur system df 2>/dev/null | head
    echo "--- dmesg tail (overlay/oom/space) ---"; dmesg 2>/dev/null | tail -8 | grep -iE "overlay|oom|space|nvme|ext4|blocked|hung" | tail -6
    echo "================ END SNAPSHOT ================"; echo
  } >> "$DEEP" 2>&1
}

last_deep=0
echo "# storm_diag start $(date '+%F %T') driver=$DRIVER_LOG interval=${INTERVAL}s disk=$DOCK_DISK" | tee -a "$HEART"
while true; do
  ts=$(date '+%F %T')
  load=$(cut -d' ' -f1 /proc/loadavg)
  dcount=$(ps -eo stat --no-headers 2>/dev/null | grep -c '^D')
  fb=$(ps -e --no-headers 2>/dev/null | grep -c docker_cleanup)
  disk=$(df --output=pcent "$DOCK_DISK" 2>/dev/null | tail -1 | tr -dc '0-9')
  t0=$(date +%s); nrun=$(dcur ps -q | grep -c .); t1=$(date +%s); dps=$((t1-t0))
  step=""
  [ -n "$DRIVER_LOG" ] && [ -f "$DRIVER_LOG" ] && step=$(perl -pe 's/\e\[[0-9;]*m//g' "$DRIVER_LOG" 2>/dev/null | grep -aoE "training (completed )?step [0-9]+|Start rollout [0-9]+|Total yielded: [0-9]+/16 for step: [0-9]+" | tail -1)
  echo "$ts load=$load Dstate=$dcount run_ctr=$nrun dps=${dps}s disk=${disk}% forkbomb=$fb | $step" | tee -a "$HEART"
  now=$(date +%s)
  if { [ "${dcount:-0}" -ge "$DSTATE_TRIG" ] || [ "${dps:-0}" -ge "$DPS_TRIG" ] || [ "${fb:-0}" -gt 0 ] || [ "${disk:-0}" -ge "$DISK_TRIG" ]; } && [ $((now-last_deep)) -ge 60 ]; then
    deep_snapshot "Dstate=$dcount dps=${dps}s disk=${disk}% forkbomb=$fb"
    last_deep=$now
    echo "  >>> DEEP SNAPSHOT taken -> $DEEP" | tee -a "$HEART"
  fi
  sleep "$INTERVAL"
done
