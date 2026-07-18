#!/usr/bin/env bash
# Pre-pull the FIRST N alexgshaw/* TB 2.0 images. Defaults to 10 to conserve
# proxy bandwidth. All pulls go through the remote dockerd + HTTPS_PROXY=proxy:8888.
#
# Usage:
#   bash prepull_partial.sh           # first 10
#   bash prepull_partial.sh 20        # first 20
#
# Run inside tmux:
#   tmux new -s tb2-prepull 'bash prepull_partial.sh 10'

set -uo pipefail

N="${1:-10}"
PARALLEL="${PARALLEL:-4}"

cd ${SKILLRL_ROOT:-$(pwd)}/datasets/terminal-bench-v2

export DOCKER_HOST="tcp://127.0.0.1:2375"
export NO_PROXY="127.0.0.1,localhost"

LOG=/tmp/tb2_prepull_partial.log
: > "$LOG"

IMAGES=$(${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/envs/slime/bin/python - <<'PY'
from pathlib import Path
import tomllib
imgs = []
for d in sorted(Path('.').iterdir()):
    if not d.is_dir() or d.name.startswith('.'): continue
    tt = d / 'task.toml'
    if not tt.exists(): continue
    try:
        meta = tomllib.loads(tt.read_text())
        img = meta.get('environment', {}).get('docker_image')
        if img:
            imgs.append(img)
    except Exception:
        pass
for i in imgs:
    print(i)
PY
)

SELECTED=$(echo "$IMAGES" | head -n "$N")
TOTAL=$(echo "$SELECTED" | wc -l)
echo "[$(date -Iseconds)] pulling $TOTAL / $(echo "$IMAGES" | wc -l) TB2 images (parallel=$PARALLEL)" | tee -a "$LOG"
echo "--- list ---" | tee -a "$LOG"
echo "$SELECTED" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

echo "$SELECTED" | xargs -n 1 -P "$PARALLEL" -I {} bash -c '
    img="{}"
    if docker image inspect "$img" >/dev/null 2>&1; then
        echo "[cached] $img"
    else
        if timeout 300 docker pull "$img" >/dev/null 2>&1; then
            echo "[ok]     $img"
        else
            sleep 3
            if timeout 300 docker pull "$img" >/dev/null 2>&1; then
                echo "[ok*2]   $img"
            else
                echo "[FAIL]   $img"
            fi
        fi
    fi
' 2>&1 | tee -a "$LOG"

echo "[$(date -Iseconds)] done" | tee -a "$LOG"
echo "=== summary ==="      | tee -a "$LOG"
grep -c '^\[ok'     "$LOG" | awk '{print "  ok     : "$1}' | tee -a "$LOG"
grep -c '^\[cached\]' "$LOG" | awk '{print "  cached : "$1}' | tee -a "$LOG"
grep -c '^\[FAIL\]' "$LOG" | awk '{print "  failed : "$1}' | tee -a "$LOG"
grep '^\[FAIL\]' "$LOG" | tee -a "$LOG" || true
