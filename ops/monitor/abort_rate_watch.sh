#!/usr/bin/env bash
set -Eeuo pipefail

RUN_DIR="${RUN_DIR:-${1:-}}"
[[ -n "${RUN_DIR}" ]] || { echo "usage: RUN_DIR=/path/to/run $0" >&2; exit 2; }

DRIVER_LOG="${DRIVER_LOG:-${RUN_DIR%/}/driver.log}"
RUN_NAME="${RUN_NAME:-$(basename "${RUN_DIR%/}")}"
LOG_FILE="${LOG_FILE:-/tmp/${RUN_NAME}_abort_watch.log}"
ALERT_FILE="${ALERT_FILE:-/tmp/${RUN_NAME}_abort_watch.alert}"
TARGET_GROUPS="${TARGET_GROUPS:-4}"
ALERT_PCT="${ALERT_PCT:-30}"
ALERT_STREAK="${ALERT_STREAK:-3}"

declare -A yielded drops done task_counts
current=""
total_drops=0
bad_streak=0

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

top_tasks() {
  local r="$1"
  for k in "${!task_counts[@]}"; do
    [[ "$k" == "$r|"* ]] || continue
    printf '%s %s\n' "${task_counts[$k]}" "${k#"$r|"}"
  done | sort -nr | head -3 | awk '{printf "%s%s:%s", sep, $2, $1; sep=", "} END{if(NR==0) printf "-"}'
}

task_from_line() {
  local line="$1" msg path id
  msg="${line#*env_agent_bench:}"
  if [[ "$line" =~ \[(harbor/[^]]+|swe/[^]]+)\] ]]; then
    path="${BASH_REMATCH[1]}"
    case "$path" in
      harbor/seta_synth/*) echo "seta:${path##*/}" ;;
      harbor/sb_ns/*) echo "sb_ns:${path##*/}" ;;
      harbor/tb2/*) echo "tb2:${path##*/}" ;;
      harbor/claw/*) echo "claw:${path##*/}" ;;
      swe/*) echo "swe_lite:${path#swe/}" ;;
      *) echo "unknown" ;;
    esac
  elif [[ "$msg" =~ \[([^]]+)\] ]]; then
    id="${BASH_REMATCH[1]}"
    [[ "$id" =~ ^[0-9]+$ ]] && echo "seta:${id}" || echo "$id"
  else
    echo "unknown"
  fi
}

exec > >(tee -a "$LOG_FILE") 2>&1
[[ -r "$DRIVER_LOG" ]] || { log "cannot read ${DRIVER_LOG}"; exit 1; }
log "watching ${DRIVER_LOG}; log=${LOG_FILE}; alert=${ALERT_FILE}"

tail -n +1 -F "$DRIVER_LOG" |
grep --line-buffered -aE 'Starting rollout step|Start rollout [0-9]+/|Total yielded:|Dropping ABORTED group|examples.agent_bench.env_agent_bench:.*sample will be ABORTED' |
while IFS= read -r line; do
  if [[ "$line" =~ Starting\ rollout\ step\ ([0-9]+) || "$line" =~ Start\ rollout\ ([0-9]+)/ ]]; then
    current="${BASH_REMATCH[1]}"
  elif [[ "$line" =~ Dropping\ ABORTED\ group\ for\ rollout_id=([0-9]+).*dropped=([0-9]+) ]]; then
    r="${BASH_REMATCH[1]}"; d="${BASH_REMATCH[2]}"; old="${drops[$r]:-0}"
    if (( d > old )); then total_drops=$((total_drops + d - old)); drops[$r]="$d"; fi
  elif [[ "$line" =~ sample\ will\ be\ ABORTED ]]; then
    r="${current:-unknown}"; task="$(task_from_line "$line")"
    key="${r}|${task}"; task_counts[$key]=$((${task_counts[$key]:-0} + 1))
  elif [[ "$line" =~ Total\ yielded:\ ([0-9]+)/([0-9]+)\ for\ step:\ ([0-9]+) ]]; then
    y="${BASH_REMATCH[1]}"; den="${BASH_REMATCH[2]}"; r="${BASH_REMATCH[3]}"
    current="$r"; yielded[$r]="$y"
    log "current rollout ${r}: Total yielded ${y}/${den}; cumulative abort drops ${total_drops}"
    if (( y == den )) && [[ -z "${done[$r]:-}" ]]; then
      d="${drops[$r]:-0}"; done[$r]=1
      rate="$(awk -v d="$d" -v y="$den" 'BEGIN{printf "%.1f", d*100/(y+d)}')"
      log "rollout ${r}: yielded ${y}/${den}, dropped ${d} groups, top task aborts: $(top_tasks "$r"), abort_rate=${rate}%"
      if (( d * 100 > ALERT_PCT * (den + d) )); then bad_streak=$((bad_streak + 1)); else bad_streak=0; fi
      if (( bad_streak >= ALERT_STREAK )); then
        msg="HIGH ALERT: abort rate > ${ALERT_PCT}% for ${bad_streak} rollouts; latest=${r}"
        log "$msg"; printf '[%s] %s\n' "$(date -Is)" "$msg" >> "$ALERT_FILE"
      fi
    fi
  fi
done
