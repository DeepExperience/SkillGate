#!/bin/bash
# Prebake SETA synth_data_harbor 300 tasks — docker build each task's image.
#
# Reads task ids from seta_300.txt (produced by select_seta_300.py).
# Each task lives at datasets/seta/dataset/synth_data_harbor/<id>/
#   with environment/Dockerfile (harbor-compat format).
# Builds as `unified-seta-synth-<id>:latest` into local Docker cache.
# Skip already-cached images; retry transient failures up to 2 times.
#
# Concurrency: default 2 (docker daemon serializes some layer work; more isn't faster).
# Proxy: injects clash + tsinghua as build-args, same as run_unified_harbor.build_image().
#
# Usage:
#   bash prebake_seta_300.sh --dry-run                # list what would build
#   bash prebake_seta_300.sh                           # build all
#   CONCURRENCY=4 bash prebake_seta_300.sh             # more parallel
#   bash prebake_seta_300.sh --from-list OTHER.txt     # use another list
#
# Cost estimate:
#   Each SETA task's Dockerfile mostly installs python + bash tools + sometimes
#   extra packages via pip/apt. Average build time 60-180s, traffic 100-500 MB
#   (most via tsinghua apt/pip — won't use clash). HF model downloads and
#   github raw files will use clash.
#   300 × ~300 MB avg = ~90 GB traffic (upper bound),
#   of which ~20-30 GB via clash, rest via tsinghua.
#   Runtime: ~3-6 hours at concurrency=2.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR=${SKILLRL_ROOT:-$(pwd)}
SYNTH_HARBOR="$BASE_DIR/datasets/seta/dataset/synth_data_harbor"
DEFAULT_LIST="$SCRIPT_DIR/seta_300.txt"
LOG="/tmp/prebake_seta_300.log"
FAIL_LOG="/tmp/prebake_seta_failures.txt"
CONCURRENCY="${CONCURRENCY:-2}"
MAX_RETRIES=2
DRY_RUN=0
LIST_FILE="$DEFAULT_LIST"

# Build-args (same as run_unified_harbor.build_image for consistency)
PROXY_URL="http://your-docker-gateway:8888"
NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1,mirrors.tuna.tsinghua.edu.cn,pypi.tuna.tsinghua.edu.cn,mirrors.aliyun.com"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)    DRY_RUN=1; shift ;;
    --from-list)  LIST_FILE="$2"; shift 2 ;;
    *)            echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

export DOCKER_HOST=tcp://127.0.0.1:2375

if [[ ! -f "$LIST_FILE" ]]; then
  echo "ERROR: $LIST_FILE not found. Run: python3 $SCRIPT_DIR/select_seta_300.py" >&2
  exit 1
fi

mapfile -t TASK_IDS < <(grep -v '^#' "$LIST_FILE" | grep -v '^$' | sed 's/\s.*$//')
TOTAL=${#TASK_IDS[@]}

CACHED=()
NEED_BUILD=()
MISSING_DOCKERFILE=()
for tid in "${TASK_IDS[@]}"; do
  tag="unified-seta-synth-${tid}:latest"
  dockerfile="$SYNTH_HARBOR/$tid/environment/Dockerfile"
  if [[ ! -f "$dockerfile" ]]; then
    MISSING_DOCKERFILE+=("$tid")
    continue
  fi
  if [[ -n "$(docker images -q "$tag" 2>/dev/null)" ]]; then
    CACHED+=("$tid")
  else
    NEED_BUILD+=("$tid")
  fi
done

echo "=== SETA synth_data_harbor prebake ==="
echo "List file:         $LIST_FILE"
echo "Target total:      $TOTAL"
echo "Already cached:    ${#CACHED[@]}"
echo "Need to build:     ${#NEED_BUILD[@]}"
echo "Missing Dockerfile: ${#MISSING_DOCKERFILE[@]}"
echo "Concurrency:       $CONCURRENCY (docker build parallelism; higher ≠ faster due to layer lock)"
echo "Retries:           $MAX_RETRIES"
echo "Log file:          $LOG"
echo

if [[ ${#MISSING_DOCKERFILE[@]} -gt 0 ]]; then
  echo "⚠ Tasks missing environment/Dockerfile (skipped): ${MISSING_DOCKERFILE[*]:0:10}..."
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  echo "=== DRY RUN (first 10 to build) ==="
  for tid in "${NEED_BUILD[@]:0:10}"; do
    echo "  build: unified-seta-synth-${tid}:latest  (from $SYNTH_HARBOR/$tid/environment/Dockerfile)"
  done
  [[ ${#NEED_BUILD[@]} -gt 10 ]] && echo "  ... and $((${#NEED_BUILD[@]} - 10)) more"
  exit 0
fi

if [[ ${#NEED_BUILD[@]} -eq 0 ]]; then
  echo "All ${TOTAL} images already cached. Nothing to do."
  exit 0
fi

HEAVY_PATTERN="python3|libreoffice|calibre|ffmpeg|imagemagick|pandoc|openjdk|poppler-utils|ghostscript|rustc|cargo|nodejs"
export HEAVY_PATTERN

build_one() {
  local tid="$1"
  local tag="unified-seta-synth-${tid}:latest"
  local env_dir="$SYNTH_HARBOR/$tid/environment"
  local dockerfile="$env_dir/Dockerfile"

  # Pick base image: heavy for tasks needing libreoffice/calibre/ffmpeg/etc.
  local use_heavy=0
  if grep -iqE "$HEAVY_PATTERN" "$dockerfile" 2>/dev/null; then
    use_heavy=1
  fi

  for attempt in 1 2 3; do
    local build_rc=1
    if [ $use_heavy -eq 1 ]; then
      # Stream modified Dockerfile via stdin: replace first FROM ubuntu:24.04 with ubuntu:24.04-heavy
      sed '0,/^FROM ubuntu:24\.04/s||FROM ubuntu:24.04-heavy|' "$dockerfile" | \
        docker build \
          --build-arg "HTTP_PROXY=$PROXY_URL" \
          --build-arg "HTTPS_PROXY=$PROXY_URL" \
          --build-arg "http_proxy=$PROXY_URL" \
          --build-arg "https_proxy=$PROXY_URL" \
          --build-arg "NO_PROXY=$NO_PROXY" \
          --build-arg "no_proxy=$NO_PROXY" \
          -t "$tag" -f - "$env_dir" >>"$LOG" 2>&1
      build_rc=$?
    else
      docker build \
        --build-arg "HTTP_PROXY=$PROXY_URL" \
        --build-arg "HTTPS_PROXY=$PROXY_URL" \
        --build-arg "http_proxy=$PROXY_URL" \
        --build-arg "https_proxy=$PROXY_URL" \
        --build-arg "NO_PROXY=$NO_PROXY" \
        --build-arg "no_proxy=$NO_PROXY" \
        -t "$tag" -f "$dockerfile" "$env_dir" >>"$LOG" 2>&1
      build_rc=$?
    fi
    if [ $build_rc -eq 0 ]; then
      echo "OK $tid"
      return 0
    fi
    if [[ $attempt -le $MAX_RETRIES ]]; then
      echo "[retry $attempt] $tid" >> "$LOG"
      sleep $((attempt * 15))
    fi
  done
  echo "FAIL $tid"
  return 1
}
export -f build_one
export SYNTH_HARBOR PROXY_URL NO_PROXY LOG MAX_RETRIES

: > "$LOG"
: > "$FAIL_LOG"
echo "[$(date -u)] starting builds (concurrency=$CONCURRENCY)..." | tee -a "$LOG"

printf '%s\n' "${NEED_BUILD[@]}" | \
  xargs -n1 -P"$CONCURRENCY" -I{} bash -c 'build_one "$@"' _ {} | \
  while IFS=' ' read -r status tid; do
    DONE=$(wc -l < <(grep -c . "$LOG"))  # not perfect, just vibes
    if [[ "$status" == "OK" ]]; then
      printf '\r  built task %-6s' "$tid"
    else
      echo "$tid" >> "$FAIL_LOG"
      printf '\r  [FAIL] task %s\n' "$tid"
    fi
  done
echo

echo
echo "[$(date -u)] done."
FINAL_HAVE=0
FINAL_MISS=()
for tid in "${TASK_IDS[@]}"; do
  tag="unified-seta-synth-${tid}:latest"
  [[ -f "$SYNTH_HARBOR/$tid/environment/Dockerfile" ]] || continue
  if [[ -n "$(docker images -q "$tag" 2>/dev/null)" ]]; then
    FINAL_HAVE=$((FINAL_HAVE+1))
  else
    FINAL_MISS+=("$tid")
  fi
done
echo "  final cached: $FINAL_HAVE / ${TOTAL} (skipped ${#MISSING_DOCKERFILE[@]} missing-Dockerfile)"
echo "  still missing: ${#FINAL_MISS[@]}"
if [[ ${#FINAL_MISS[@]} -gt 0 ]]; then
  echo
  echo "Failed task ids (see $FAIL_LOG):"
  for tid in "${FINAL_MISS[@]:0:5}"; do echo "  $tid"; done
  [[ ${#FINAL_MISS[@]} -gt 5 ]] && echo "  ... +$((${#FINAL_MISS[@]} - 5)) more"
  echo
  echo "Retry failed with: bash $0 --from-list $FAIL_LOG"
  exit 1
fi

echo
echo "All ${TOTAL} SETA synth_data images ready in local cache."
