#!/bin/bash
# Prebake TB 2.0 test.sh dependencies so verifier doesn't re-download every task.
#
# Root cause: TB 2.0's stock test.sh does
#   1. apt-get install curl
#   2. curl astral.sh/uv/install.sh | sh        (~20MB via clash)
#   3. uvx -p 3.13 -w pytest==X -w <deps> ...   (cpython-3.13.9 = 32MB + deps 20-300MB)
# That's 50-400MB *per task*, all via clash → runs out of quota, and verifier
# times out at 1200-3600s waiting for downloads.
#
# This script bakes:
#   - uv 0.9.5 binary
#   - cpython-3.13.9-linux-x86_64-gnu
#   - common verifier deps (pytest 8.4.1, numpy, scipy, beautifulsoup4, selenium,
#     pytest-json-ctrf, requests, pyyaml, etc)
# into a tarball at ops/cache/pkg/tb2_uv_cache.tar.gz.
#
# Runtime: ~15-20 min (clash-limited); ~500MB tarball. Run ONCE.
# Output: ops/cache/pkg/tb2_uv_cache.tar.gz
#
# Usage:
#   bash prebake_uv_cache.sh              # build tarball (starts fresh container)
#   bash prebake_uv_cache.sh --dry-run    # print plan only, don't execute

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

BASE_DIR=${SKILLRL_ROOT:-$(pwd)}
CACHE_OUT="$BASE_DIR/ops/cache/pkg/tb2_uv_cache.tar.gz"
CNAME=tb2-uv-prebake
BASE_IMAGE=ubuntu:24.04
CLASH_PROXY=http://your-docker-gateway:8888

echo "=== Prebake TB 2.0 uv+python+deps cache ==="
echo "Output: $CACHE_OUT"
echo "Temp container: $CNAME"
echo "Base image: $BASE_IMAGE"
echo "Clash proxy: $CLASH_PROXY"
echo

# Common verifier deps observed in 52 tb2 test.sh files (union):
# pytest, pytest-json-ctrf, numpy, scipy, beautifulsoup4, selenium, requests,
# pyyaml, pandas, matplotlib, pillow, pyopengl, mujoco, pygments, lxml
# We preinstall the top-tier ones; rare deps still fall through to clash.
PREINSTALL_DEPS=(
  # --- Core test runners (every TB2 task) ---
  "pytest==8.4.1"
  "pytest==8.3.4"                # 1x task uses older
  "pytest-json-ctrf==0.3.5"
  # --- numpy (13x across 6 versions; preload the 3 most common) ---
  "numpy==2.3.3"
  "numpy==2.3.1"                 # 8x — most common mismatch source
  "numpy==2.3.2"                 # 1x
  # --- scipy (3 versions seen) ---
  "scipy==1.16.2"
  "scipy==1.16.1"                # 1x
  "scipy==1.16.3"                # 1x
  # --- pandas ---
  "pandas==2.2.3"
  "pandas==2.3.3"                # 3x
  # --- pillow ---
  "pillow==11.0.0"
  "pillow==11.2.1"               # 4x
  # --- HTTP / scraping (shared across many tasks) ---
  "beautifulsoup4==4.13.5"
  "requests==2.32.3"
  "requests==2.32.4"             # 1x newer
  "selenium==4.35.0"
  "selenium==4.38.0"             # 1x
  # --- utilities ---
  "pyyaml==6.0.2"
  "lxml==5.3.0"
  "pygments==2.18.0"
  "setuptools==80.9.0"           # 1x explicit
  "setuptools==78.1.1"           # 1x
  # --- media / image (2x tasks use cv2) ---
  "opencv-python==4.11.0.86"
  "scikit-image==0.25.0"         # 1x
  "matplotlib==3.10.7"           # 1x
  "pytesseract==0.3.13"          # 1x
)

# v3 (2026-04-20): add torch CPU wheel to cache (5 tasks use it: pytorch-model-cli/
# pytorch-model-recovery/sam-cell-seg/torch-pipeline-parallelism/torch-tensor-parallelism).
# Each runtime download was ~300MB × 3 arms × 5 tasks = ~4.5 GB/run wasted.
# Installed separately because torch needs pytorch's own --index (CPU-only wheel).
# Tsinghua has a pytorch mirror at https://pypi.tuna.tsinghua.edu.cn/simple (metadata
# only, wheels redirect to pytorch.org which is still foreign); fallback to pytorch.org.
TORCH_DEPS=(
  "torch==2.7.1"
  "torchvision==0.22.1"
)
TORCH_INDEX="https://download.pytorch.org/whl/cpu"

# Build the shell script to run inside the container. Kept separate for dry-run printing.
read -r -d '' INSIDE_SCRIPT <<'EOF' || true
set -euo pipefail
apt-get update -qq
apt-get install -y --no-install-recommends curl ca-certificates tar gzip >/dev/null
# Install uv 0.9.5 (matches the version 52 test.sh files pin)
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
source "$HOME/.local/bin/env"
# Download cpython 3.13.9 (this is the expensive 32MB fetch to amortize)
uv python install 3.13.9
# Preinstall deps into uv cache by creating a throwaway env
DEPS="__DEPS_PLACEHOLDER__"
# Pre-warm uvx cache (this is the pattern tb2 test.sh uses, so it's the right cache to fill).
# `uvx -p 3.13 --with <deps> ... pytest --version` downloads each dep into uv cache.
# Do it one dep at a time + `|| true` so one broken dep doesn't kill the rest.
for d in $DEPS; do
  echo "[prebake] preinstalling $d via uvx --with"
  uvx -p 3.13 --with "$d" python -c "import sys" 2>&1 | tail -3 || \
    echo "[prebake] WARN: $d install failed; skipping"
done
# Also warm combined uvx: exactly mirrors the common tb2 test.sh invocation pattern
echo "[prebake] warming composite uvx (pytest + common deps combined)"
uvx -p 3.13 \
  --with pytest==8.4.1 --with pytest-json-ctrf==0.3.5 \
  --with numpy==2.3.3 --with scipy==1.16.2 \
  pytest --version 2>&1 | tail -3 || echo "[prebake] WARN: composite uvx warmup failed"

# --- torch CPU bake (v3, 2026-04-20) ---
# 5 tb2 task use torch==2.7.1 + torchvision==0.22.1 via pytorch's --index.
# Wheel is ~200MB each. Bake so test.sh hits cache.
TORCH_DEPS_STR="__TORCH_DEPS_PLACEHOLDER__"
TORCH_IDX="__TORCH_INDEX_PLACEHOLDER__"
for d in $TORCH_DEPS_STR; do
  echo "[prebake v3] preinstalling $d via --index $TORCH_IDX"
  uvx -p 3.13 --index "$TORCH_IDX" --with "$d" python -c "import sys" 2>&1 | tail -3 || \
    echo "[prebake v3] WARN: $d install failed; skipping"
done
# Composite: the exact combo used by pytorch-model-cli/test.sh etc.
echo "[prebake v3] warming torch composite (torch+torchvision+numpy)"
uvx -p 3.13 --index "$TORCH_IDX" --index-strategy unsafe-best-match \
  --with torch==2.7.1 --with torchvision==0.22.1 \
  --with numpy==2.3.1 \
  python -c "import torch; print('torch', torch.__version__)" 2>&1 | tail -3 || \
  echo "[prebake v3] WARN: torch composite warmup failed"

# Assemble /opt/tb2-uv/ layout the runner will use
mkdir -p /opt/tb2-uv
cp "$HOME/.local/bin/uv" /opt/tb2-uv/uv
cp "$HOME/.local/bin/uvx" /opt/tb2-uv/uvx
# TWO cache-related dirs exist; we need both:
#   ~/.local/share/uv/  — python installations + shared envs (UV_PYTHON_INSTALL_DIR lives here)
#   ~/.cache/uv/        — wheel downloads / simple index cache (UV_CACHE_DIR, default $HOME/.cache/uv)
# Without the second, `uvx --with pytest` re-downloads all wheels.
cp -a "$HOME/.local/share/uv" /opt/tb2-uv/data
cp -a "$HOME/.cache/uv"       /opt/tb2-uv/cache
du -sh /opt/tb2-uv/data /opt/tb2-uv/cache
# Create tarball
cd /opt
tar czf /tb2_uv_cache.tar.gz tb2-uv
ls -lh /tb2_uv_cache.tar.gz
EOF

# Substitute the deps list into the script (bash heredoc expands $VAR but we want literal
# control for the dry-run print; use placeholder+sed here).
DEPS_LINE="${PREINSTALL_DEPS[*]}"
TORCH_LINE="${TORCH_DEPS[*]}"
# shellcheck disable=SC2001
INSIDE_SCRIPT_FINAL=$(echo "$INSIDE_SCRIPT" \
  | sed "s|__DEPS_PLACEHOLDER__|${DEPS_LINE}|" \
  | sed "s|__TORCH_DEPS_PLACEHOLDER__|${TORCH_LINE}|" \
  | sed "s|__TORCH_INDEX_PLACEHOLDER__|${TORCH_INDEX}|")

if [[ $DRY_RUN -eq 1 ]]; then
  echo "--- DRY RUN ---"
  echo "Would start container '$CNAME' from '$BASE_IMAGE' and run this script inside:"
  echo
  echo "$INSIDE_SCRIPT_FINAL"
  echo
  echo "Then docker cp /tb2_uv_cache.tar.gz → $CACHE_OUT and docker rm -f $CNAME."
  echo "--- END DRY RUN ---"
  exit 0
fi

mkdir -p "$(dirname "$CACHE_OUT")"

echo "[1/3] Starting prebake container..."
docker rm -f "$CNAME" 2>/dev/null || true
# --add-host: ~/.docker/config.json's default proxies use host.docker.internal
#   so container needs host-gateway mapping. Otherwise DNS fails → pip/curl
#   go to dead proxy → silent install failures (seen 2026-04-19 first attempt).
docker run -d --name "$CNAME" \
  --add-host host.docker.internal:host-gateway \
  -e HTTP_PROXY="$CLASH_PROXY" -e HTTPS_PROXY="$CLASH_PROXY" \
  -e http_proxy="$CLASH_PROXY" -e https_proxy="$CLASH_PROXY" \
  -e NO_PROXY="localhost,127.0.0.1,mirrors.tuna.tsinghua.edu.cn,pypi.tuna.tsinghua.edu.cn,mirrors.aliyun.com" \
  -e no_proxy="localhost,127.0.0.1,mirrors.tuna.tsinghua.edu.cn,pypi.tuna.tsinghua.edu.cn,mirrors.aliyun.com" \
  -e PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple" \
  -e PIP_EXTRA_INDEX_URL="https://mirrors.aliyun.com/pypi/simple" \
  -e UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple" \
  "$BASE_IMAGE" sleep infinity

# Inject Tsinghua apt sources (saves a few MB via clash even for tiny apt packages)
APT_SRC="$BASE_DIR/ops/cache/pkg/apt_sources/ubuntu.sources.noble"
if [[ -f "$APT_SRC" ]]; then
  docker exec "$CNAME" sh -c "true > /etc/apt/sources.list"
  docker cp "$APT_SRC" "$CNAME:/etc/apt/sources.list.d/ubuntu.sources"
  echo "  [inject] Tsinghua apt sources → /etc/apt/sources.list.d/ubuntu.sources"
fi
# Inject Tsinghua pip config
PIP_CONF="$BASE_DIR/ops/cache/pkg/pip.conf"
if [[ -f "$PIP_CONF" ]]; then
  docker cp "$PIP_CONF" "$CNAME:/etc/pip.conf"
  docker exec "$CNAME" mkdir -p /root/.config/pip
  docker cp "$PIP_CONF" "$CNAME:/root/.config/pip/pip.conf"
  echo "  [inject] Tsinghua pip config → /etc/pip.conf + /root/.config/pip/pip.conf"
fi

echo "[2/3] Running uv+python prebake inside container (this takes ~15-20 min)..."
docker exec "$CNAME" bash -c "$INSIDE_SCRIPT_FINAL"

echo "[3/3] Copying tarball out..."
docker cp "$CNAME:/tb2_uv_cache.tar.gz" "$CACHE_OUT"
docker rm -f "$CNAME"

echo
echo "DONE. Tarball: $(ls -lh "$CACHE_OUT" | awk '{print $5}')"
echo "Next: run python3 $BASE_DIR/GeneralAgent/eval_scripts/tb2_eval/patch_test_sh.py --dry-run"
echo "      to see what will be patched in the 52 tb2 test.sh files."
