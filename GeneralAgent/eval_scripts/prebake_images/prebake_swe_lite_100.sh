#!/bin/bash
# Prebake SWE-Gym-lite 100 instance docker images (pull-through registry-mirror → local cache).
#
# Idempotent: skips any image already in local cache.
# Parallel: up to $CONCURRENCY pulls at once (default 4).
# Retry: each pull retries up to 2 times on transient errors.
# Progress: live counter + per-attempt log at /tmp/prebake_swe_lite_100.log
# Flow summary printed at end (success / cached / failed).
#
# Usage:
#   bash prebake_swe_lite_100.sh --dry-run           # just list what would pull
#   bash prebake_swe_lite_100.sh                      # default: concurrency=4
#   CONCURRENCY=8 bash prebake_swe_lite_100.sh        # more aggressive (watches clash)
#
# Cost estimate (2.5 GB avg/image × 100 − already-cached):
#   When called cold → ~220-250 GB clash traffic, ~40-60 min at clash speed.
#   Re-runs are fast (cache check only).

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR=${SKILLRL_ROOT:-$(pwd)}
IMAGES_FILE="$SCRIPT_DIR/swe_lite_100_images.txt"
LOG="/tmp/prebake_swe_lite_100.log"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_RETRIES=2
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

export DOCKER_HOST=tcp://127.0.0.1:2375

if [[ ! -f "$IMAGES_FILE" ]]; then
  echo "ERROR: $IMAGES_FILE not found. Run: python3 $SCRIPT_DIR/select_swe_lite_100.py" >&2
  exit 1
fi

mapfile -t ALL_IMAGES < <(grep -v '^#' "$IMAGES_FILE" | grep -v '^$')
TOTAL=${#ALL_IMAGES[@]}

# Identify already-cached vs need-pull
CACHED=()
NEED_PULL=()
for img in "${ALL_IMAGES[@]}"; do
  if [[ -n "$(docker images -q "$img" 2>/dev/null)" ]]; then
    CACHED+=("$img")
  else
    NEED_PULL+=("$img")
  fi
done

echo "=== SWE-Gym-lite prebake ==="
echo "Target total: $TOTAL images"
echo "Already cached: ${#CACHED[@]}"
echo "Need to pull:   ${#NEED_PULL[@]}"
echo "Concurrency:    $CONCURRENCY"
echo "Retries:        $MAX_RETRIES"
echo "Log file:       $LOG"
echo "Traffic estimate: ~$(awk "BEGIN{printf \"%.0f\", ${#NEED_PULL[@]} * 2.5}") GB (avg 2.5 GB/image)"
echo

if [[ $DRY_RUN -eq 1 ]]; then
  echo "=== DRY RUN (first 10 to pull) ==="
  for img in "${NEED_PULL[@]:0:10}"; do echo "  pull: $img"; done
  [[ ${#NEED_PULL[@]} -gt 10 ]] && echo "  ... and $((${#NEED_PULL[@]} - 10)) more"
  exit 0
fi

if [[ ${#NEED_PULL[@]} -eq 0 ]]; then
  echo "All ${TOTAL} images already cached. Nothing to do."
  exit 0
fi

# Pull with retry worker
pull_one() {
  local img="$1"
  for attempt in 1 2 3; do
    if docker pull "$img" >>"$LOG" 2>&1; then
      echo "OK $img"
      return 0
    fi
    if [[ $attempt -le $MAX_RETRIES ]]; then
      echo "RETRY $attempt $img" >>"$LOG"
      sleep $((attempt * 10))
    fi
  done
  echo "FAIL $img"
  return 1
}
export -f pull_one
export LOG MAX_RETRIES

: > "$LOG"
echo "[$(date -u)] starting pulls..." | tee -a "$LOG"

# Run in parallel using xargs
SUCCESS=0
FAILED=0
FAIL_LOG=/tmp/prebake_swe_failures.txt
: > "$FAIL_LOG"

# Spawn parallel pulls
printf '%s\n' "${NEED_PULL[@]}" | \
  xargs -n1 -P"$CONCURRENCY" -I{} bash -c 'pull_one "$@"' _ {} | \
  while IFS=' ' read -r status img; do
    if [[ "$status" == "OK" ]]; then
      SUCCESS=$((SUCCESS+1))
      DONE=$((SUCCESS + FAILED))
      printf '\r  [%d/%d] pulled %-80s' "$DONE" "${#NEED_PULL[@]}" "$(basename "$img")"
    else
      FAILED=$((FAILED+1))
      echo "$img" >> "$FAIL_LOG"
      printf '\r  [FAIL] %-80s\n' "$img"
    fi
  done
echo

echo
echo "[$(date -u)] done."
echo "  cached (pre-existing): ${#CACHED[@]}"
# Re-scan for final tally (xargs subshell counts are lost by design)
FINAL_HAVE=0
FINAL_MISS=()
for img in "${ALL_IMAGES[@]}"; do
  if [[ -n "$(docker images -q "$img" 2>/dev/null)" ]]; then
    FINAL_HAVE=$((FINAL_HAVE+1))
  else
    FINAL_MISS+=("$img")
  fi
done
echo "  final cached:          $FINAL_HAVE / $TOTAL"
echo "  still missing:         ${#FINAL_MISS[@]}"
if [[ ${#FINAL_MISS[@]} -gt 0 ]]; then
  echo
  echo "Missing images (see $FAIL_LOG):"
  for img in "${FINAL_MISS[@]:0:5}"; do echo "  $img"; done
  [[ ${#FINAL_MISS[@]} -gt 5 ]] && echo "  ... and $((${#FINAL_MISS[@]} - 5)) more"
  echo
  echo "To retry only the failed ones:"
  echo "  while read img; do docker pull \"\$img\"; done < $FAIL_LOG"
  exit 1
fi

echo
echo "All ${TOTAL} SWE-Gym-lite images ready in local cache."
