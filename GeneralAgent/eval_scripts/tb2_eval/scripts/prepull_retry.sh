#!/usr/bin/env bash
# Serial retry pull for failed TB 2.0 images.
# the remote host's proxy has intermittent EOFs on auth.docker.io; parallel pulls hit
# the issue much harder. Serial + exponential backoff is the robust path.

set -uo pipefail

cd ${SKILLRL_ROOT:-$(pwd)}/datasets/terminal-bench-v2
export DOCKER_HOST="tcp://127.0.0.1:2375"
export NO_PROXY="127.0.0.1,localhost"

LOG=/tmp/tb2_prepull_retry.log
: > "$LOG"

# Which images still need pulling? Compare task.toml-declared images against
# what's already on the remote host's dockerd.
MISSING=$(${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/envs/slime/bin/python - <<'PY'
from pathlib import Path
import tomllib, subprocess
have = set(subprocess.check_output(["docker","images","--format","{{.Repository}}:{{.Tag}}"]).decode().split())
for d in sorted(Path('.').iterdir()):
    if not d.is_dir() or d.name.startswith('.'): continue
    tt = d / 'task.toml'
    if not tt.exists(): continue
    try:
        img = tomllib.loads(tt.read_text()).get('environment',{}).get('docker_image')
        if img and img not in have:
            print(img)
    except Exception:
        pass
PY
)

TOTAL=$(echo "$MISSING" | grep -c . || echo 0)
echo "[$(date -Iseconds)] $TOTAL images still missing" | tee -a "$LOG"

ok=0; fail=0
fail_list=()
i=0
while IFS= read -r img; do
    [ -z "$img" ] && continue
    i=$((i+1))
    printf "[%3d/%3d] %s ... " "$i" "$TOTAL" "$img"
    success=0
    for attempt in 1 2 3 4 5; do
        err=$(timeout 300 docker pull "$img" 2>&1 >/dev/null) && { success=1; break; }
        echo "  [attempt $attempt err] ${err##*$'\n'}" >> "$LOG"
        # backoff: 2, 4, 8, 16s
        sleep $((2 ** (attempt - 1) * 2))
    done
    if [ $success -eq 1 ]; then
        echo "ok (attempt $attempt)" | tee -a "$LOG"
        ok=$((ok+1))
    else
        echo "FAIL after 5 attempts" | tee -a "$LOG"
        fail=$((fail+1))
        fail_list+=("$img")
    fi
done <<< "$MISSING"

echo | tee -a "$LOG"
echo "=== summary ===" | tee -a "$LOG"
echo "  pulled : $ok / $TOTAL" | tee -a "$LOG"
echo "  failed : $fail" | tee -a "$LOG"
if [ ${#fail_list[@]} -gt 0 ]; then
    echo "  fail list:" | tee -a "$LOG"
    for f in "${fail_list[@]}"; do echo "    $f" | tee -a "$LOG"; done
fi
echo "Log: $LOG"
