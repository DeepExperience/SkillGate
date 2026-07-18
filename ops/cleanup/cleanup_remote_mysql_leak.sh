#!/usr/bin/env bash
# Clean up STALE leaked mysql_<pid> containers on the shared remote Docker host.
#
# Background: another tenant on the shared Docker host runs LiveMCP simulations
# that spawn mysql:8.0 sidecars and sometimes leave them in Created/Exited
# state when their tests crash. Each stale container slows down our
# `docker exec` (boltdb metadata + global dockerd lock contention), which
# can push our SHELL_INIT_TIMEOUT (30s) over the edge.
#
# Safety contract:
#   1. ONLY removes mysql:8.0 containers (filter by ancestor image).
#   2. ONLY removes containers in NON-Up state (Created/Exited/Dead).
#      Running mysql containers (the other tenant's active workload) are untouched.
#   3. ONLY removes containers stalled >= STALE_MIN_SEC (default 300s).
#      Anything younger might be the other tenant's in-flight container that's about
#      to be `docker start`'d; killing it would race them.
#
# Idempotent and safe to run repeatedly. Designed to be auto-invoked from
# launch_trials.py preflight, but also runnable standalone.
set -euo pipefail

STALE_MIN_SEC="${STALE_MIN_SEC:-300}"
REMOTE_DOCKER_HOST="${REMOTE_DOCKER_HOST:-your-docker-host}"

echo "=== remote mysql cleanup $(date -Iseconds) (stale_threshold=${STALE_MIN_SEC}s) ==="

ssh -o ConnectTimeout=5 "$REMOTE_DOCKER_HOST" "STALE_MIN_SEC=$STALE_MIN_SEC bash -s" <<'REMOTE'
set -euo pipefail

before_total=$(docker ps -a -q | wc -l)
before_mysql=$(docker ps -a --filter name=mysql -q | wc -l)
now_epoch=$(date +%s)
removed=0

# For each non-Up mysql:8.0 container, inspect to get FinishedAt/StartedAt/CreatedAt.
# We pick the most recent of these as the container's "last state-change" time
# and only remove if it's older than STALE_MIN_SEC.
for cname in $(docker ps -a --filter ancestor=mysql:8.0 \
                  --filter status=created \
                  --filter status=exited \
                  --filter status=dead \
                  --format '{{.Names}}'); do
    # docker inspect gives precise ISO timestamps for State.{StartedAt,FinishedAt}
    # and Created. Format: 2026-04-25T14:51:11.123456789Z
    info=$(docker inspect "$cname" \
            --format '{{.State.FinishedAt}}|{{.State.StartedAt}}|{{.Created}}' \
            2>/dev/null) || continue

    # Pick newest non-zero timestamp as last state change.
    last=""
    IFS='|' read -r finished started created <<< "$info"
    for ts in "$finished" "$started" "$created"; do
        # Docker uses 0001-01-01T00:00:00Z as "never set"
        if [ -n "$ts" ] && [ "$ts" != "0001-01-01T00:00:00Z" ]; then
            last="$ts"
            break
        fi
    done
    [ -z "$last" ] && continue

    last_epoch=$(date -d "$last" +%s 2>/dev/null || echo 0)
    [ "$last_epoch" -eq 0 ] && continue
    age=$(( now_epoch - last_epoch ))

    if [ "$age" -ge "$STALE_MIN_SEC" ]; then
        if docker rm -f "$cname" >/dev/null 2>&1; then
            removed=$(( removed + 1 ))
        fi
    fi
done

after_mysql=$(docker ps -a --filter name=mysql -q | wc -l)
after_total=$(docker ps -a -q | wc -l)
echo "  mysql: $before_mysql → $after_mysql (removed $removed stale)"
echo "  total: $before_total → $after_total"
REMOTE
