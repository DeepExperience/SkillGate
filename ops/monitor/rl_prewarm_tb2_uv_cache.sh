#!/usr/bin/env bash
set -euo pipefail
cd ${SKILLRL_ROOT:-$(pwd)}
WORK="${1:-ops/cache/pkg/tb2_uv_cache_work}"
TARBALL="ops/cache/pkg/tb2_uv_cache.tar.gz"
mkdir -p "$WORK"
if [ ! -d "$WORK/tb2-uv" ]; then
  echo "[tb2-uv-prewarm] extracting $TARBALL -> $WORK"
  tar xzf "$TARBALL" -C "$WORK"
fi
export PATH="$PWD/$WORK/tb2-uv:$PATH"
export UV_PYTHON_INSTALL_DIR="$PWD/$WORK/tb2-uv/data/python"
export UV_CACHE_DIR="$PWD/$WORK/tb2-uv/cache"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export HTTP_PROXY="${HTTP_PROXY:-http://your-proxy:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://your-proxy:3128}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,0.0.0.0,10.0.0.0/8,mirrors.tuna.tsinghua.edu.cn,pypi.tuna.tsinghua.edu.cn,hf-mirror.com}"
export no_proxy="$NO_PROXY"

echo "[tb2-uv-prewarm] uv=$(command -v uv) uvx=$(command -v uvx) cache=$UV_CACHE_DIR"

echo "[tb2-uv-prewarm] common pytest/request stack"
uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 -w requests==2.31.0 python - <<'PY'
import pytest, requests
print('ok common', pytest.__version__, requests.__version__)
PY

echo "[tb2-uv-prewarm] playwright/python package stack (browser binaries are task/image-specific)"
uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 -w playwright==1.57.0 -w pillow==10.4.0 python - <<'PY'
import PIL, playwright
print('ok playwright package')
PY

echo "[tb2-uv-prewarm] torch==2.7.0 stack; may download large CUDA wheels if mirror lacks cpu wheel"
uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 -w torch==2.7.0 python - <<'PY'
import torch
print('ok torch', torch.__version__)
PY

echo "[tb2-uv-prewarm] repacking $TARBALL"
cp "$TARBALL" "${TARBALL}.bak_$(date +%Y%m%d_%H%M%S)"
( cd "$WORK" && tar czf "$PWD/$TARBALL" tb2-uv )
ls -lh "$TARBALL"
