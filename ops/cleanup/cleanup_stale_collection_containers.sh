#!/usr/bin/env bash
set -euo pipefail

# Remove stale per-trial containers whose owner PID no longer exists.
# Names created by unified runners end with "-p<PID>". We only touch known
# SFT/eval prefixes and skip any container whose PID is still alive.

removed=0
scanned=0

while IFS= read -r name; do
  [[ -n "${name}" ]] || continue
  if [[ ! "${name}" =~ ^(u-|swe-unified-) ]]; then
    continue
  fi
  if [[ ! "${name}" =~ -p([0-9]+)(-|$) ]]; then
    continue
  fi
  scanned=$((scanned + 1))
  pid="${BASH_REMATCH[1]}"
  if [[ -e "/proc/${pid}" ]]; then
    state="$(awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null || true)"
    if [[ "${state}" != "Z" ]]; then
      echo "keep alive-pid container: ${name} (pid=${pid}, state=${state:-unknown})"
      continue
    fi
    echo "remove zombie-pid container: ${name} (pid=${pid})"
  else
    echo "remove stale container: ${name} (dead pid=${pid})"
  fi
  if timeout 30 docker rm -f "${name}" >/dev/null 2>&1; then
    removed=$((removed + 1))
  else
    echo "warn: docker rm timed out/failed for ${name}" >&2
  fi
done < <(docker ps -a --format '{{.Names}}')

echo "stale collection container cleanup: scanned=${scanned} removed=${removed}"
