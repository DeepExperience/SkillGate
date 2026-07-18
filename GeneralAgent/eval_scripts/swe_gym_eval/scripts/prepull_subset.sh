#!/usr/bin/env bash
# Pre-pull a small subset of SWE-Gym-Lite + SWE-bench Verified instance images
# onto the remote Docker host's dockerd. Serial + exponential backoff (same pattern as TB 2.0).
#
# Run in tmux:  tmux new -s swe-prepull 'bash GeneralAgent/eval_scripts/swe_gym_eval/scripts/prepull_subset.sh'

set -uo pipefail

export DOCKER_HOST="tcp://127.0.0.1:2375"
export NO_PROXY="127.0.0.1,localhost"

LOG=/tmp/swe_prepull.log
: > "$LOG"

# Dedupe + concatenate two image lists produced by pick_subset.py
{ cat /tmp/pull_lite.txt; cat /tmp/pull_verified.txt; } | awk 'NF && !seen[$0]++' > /tmp/swe_pull_all.txt
TOTAL=$(wc -l < /tmp/swe_pull_all.txt)
echo "[$(date -Iseconds)] pulling $TOTAL SWE images (lite + verified subset)" | tee -a "$LOG"

ok=0; fail=0; skip=0
declare -a fail_list=()
i=0
while IFS= read -r img; do
    [ -z "$img" ] && continue
    i=$((i+1))
    if docker image inspect "$img" >/dev/null 2>&1; then
        echo "[$i/$TOTAL] $img ... already cached" | tee -a "$LOG"
        skip=$((skip+1))
        continue
    fi
    printf "[%2d/%2d] %s ... " "$i" "$TOTAL" "$img"
    success=0
    for attempt in 1 2 3 4 5 6; do
        err=$(timeout 300 docker pull "$img" 2>&1 >/dev/null) && { success=1; break; }
        echo "  [attempt $attempt err] ${err##*$'\n'}" >> "$LOG"
        # exponential backoff: 3, 6, 12, 24, 48 s
        sleep $((3 * 2 ** (attempt - 1)))
    done
    if [ $success -eq 1 ]; then
        echo "ok (attempt $attempt)" | tee -a "$LOG"
        ok=$((ok+1))
    else
        echo "FAIL" | tee -a "$LOG"
        fail=$((fail+1))
        fail_list+=("$img")
    fi
done < /tmp/swe_pull_all.txt

echo | tee -a "$LOG"
echo "=== summary ===" | tee -a "$LOG"
echo "  pulled  : $ok" | tee -a "$LOG"
echo "  cached  : $skip" | tee -a "$LOG"
echo "  failed  : $fail" | tee -a "$LOG"
if [ ${#fail_list[@]} -gt 0 ]; then
    echo "  fail list:" | tee -a "$LOG"
    for f in "${fail_list[@]}"; do echo "    $f" | tee -a "$LOG"; done
fi
echo "Log: $LOG"
