#!/usr/bin/env bash
# Pre-pull all 89 alexgshaw/* prebuilt images for TB 2.0 onto the remote Docker host.
# Run inside tmux: `tmux new -s tb2-prepull 'bash scripts/prepull_images.sh'`.
#
# Why: Harbor's `docker compose up --wait` has a proxy-handling bug that makes
# its pull path EOF through the remote host's proxy. Classic `docker pull` works fine.
# Pre-pulling once avoids `--force-build` entirely for subsequent runs.

set -uo pipefail

cd ${SKILLRL_ROOT:-$(pwd)}/datasets/terminal-bench-v2

export DOCKER_HOST="tcp://127.0.0.1:2375"
export NO_PROXY="127.0.0.1,localhost"

LOG=/tmp/tb2_prepull.log
: > "$LOG"

# Gather (task, image) pairs from task.toml
IMAGES=$(${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/envs/slime/bin/python - <<'PY'
from pathlib import Path
import tomllib
for d in sorted(Path('.').iterdir()):
    if not d.is_dir() or d.name.startswith('.'): continue
    tt = d / 'task.toml'
    if not tt.exists(): continue
    try:
        meta = tomllib.loads(tt.read_text())
        img = meta.get('environment', {}).get('docker_image')
        if img:
            print(img)
    except Exception:
        pass
PY
)

TOTAL=$(echo "$IMAGES" | wc -l)
echo "[$(date -Iseconds)] pre-pulling $TOTAL images on the remote docker host" | tee -a "$LOG"

i=0
ok=0
fail=0
fail_list=()
# parallel pulls: 6 at a time (the remote host has 255 CPU / 500 GB RAM, network is the only bottleneck)
echo "$IMAGES" | xargs -n 1 -P 6 -I {} bash -c '
    img="{}"
    if docker image inspect "$img" >/dev/null 2>&1; then
        echo "[cached] $img"
    else
        if timeout 180 docker pull "$img" >/dev/null 2>&1; then
            echo "[ok]     $img"
        else
            # retry once — auth.docker.io EOFs are often transient
            sleep 3
            if timeout 180 docker pull "$img" >/dev/null 2>&1; then
                echo "[ok*2]   $img"
            else
                echo "[FAIL]   $img"
            fi
        fi
    fi
' 2>&1 | tee -a "$LOG"

echo "[$(date -Iseconds)] done" | tee -a "$LOG"
echo | tee -a "$LOG"
echo "=== summary ===" | tee -a "$LOG"
grep -c '^\[ok' "$LOG" | awk '{print "  ok     : "$1}' | tee -a "$LOG"
grep -c '^\[cached\]' "$LOG" | awk '{print "  cached : "$1}' | tee -a "$LOG"
grep -c '^\[FAIL\]' "$LOG" | awk '{print "  failed : "$1}' | tee -a "$LOG"
echo | tee -a "$LOG"
grep '^\[FAIL\]' "$LOG" | tee -a "$LOG" || true
echo
echo "Log: $LOG"
