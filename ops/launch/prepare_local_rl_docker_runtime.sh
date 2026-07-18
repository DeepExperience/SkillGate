#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare the local Docker runtime used for RL/eval after migrating remote-Docker
# images and external verifier caches.
#
# This script intentionally keeps fast mutable state on /data/cache and durable
# restore artifacts under experiments/infra/rl/local_docker_migration.

PROJECT_ROOT="${PROJECT_ROOT:-${SKILLRL_ROOT:-$(pwd)}}"
TB2_CACHE_TAR="${TB2_CACHE_TAR:-${PROJECT_ROOT}/experiments/infra/rl/local_docker_migration/external_caches/tb2_uv_cache_tb2-uv.tar}"
TB2_CACHE_PARENT="${TB2_CACHE_PARENT:-/data/cache/tb2_uv_cache}"
TB2_CACHE_DIR="${TB2_CACHE_DIR:-${TB2_CACHE_PARENT}/tb2-uv}"
RESTORE_LOCAL_IMAGES="${RESTORE_LOCAL_IMAGES:-0}"
RESTORE_LOCAL_IMAGE_WORKERS="${RESTORE_LOCAL_IMAGE_WORKERS:-4}"
RESTORE_LOCAL_IMAGE_RUN_ROOT="${RESTORE_LOCAL_IMAGE_RUN_ROOT:-${PROJECT_ROOT}/experiments/infra/rl/local_docker_migration/restore_$(date +%Y%m%d_%H%M%S)}"

cd "${PROJECT_ROOT}"

bash ops/launch/start_local_overlay2_docker.sh

if [[ ! -d "${TB2_CACHE_DIR}" || ! -x "${TB2_CACHE_DIR}/uv" || ! -x "${TB2_CACHE_DIR}/uvx" ]]; then
  if [[ ! -s "${TB2_CACHE_TAR}" ]]; then
    echo "ERROR: missing TB2 cache tar: ${TB2_CACHE_TAR}" >&2
    exit 2
  fi
  rm -rf "${TB2_CACHE_DIR}"
  mkdir -p "${TB2_CACHE_PARENT}"
  tar -xf "${TB2_CACHE_TAR}" -C "${TB2_CACHE_PARENT}"
fi

SOCKET="$(cat /tmp/local-docker-active.sock)"

if [[ "${RESTORE_LOCAL_IMAGES}" == "1" ]]; then
  python3 ops/launch/restore_local_docker_images_from_cache.py \
    --run-root "${RESTORE_LOCAL_IMAGE_RUN_ROOT}" \
    --workers "${RESTORE_LOCAL_IMAGE_WORKERS}" \
    --local-docker-host "unix://${SOCKET}"
fi

DOCKER_HOST="unix://${SOCKET}" docker info --format \
  'local docker: Server={{.ServerVersion}} Driver={{.Driver}} Root={{.DockerRootDir}} Images={{.Images}}'

echo
echo "Use these for local RL/eval Docker:"
echo "export DOCKER_HOST=unix://${SOCKET}"
echo "export DOCKER_HOST_VALUE=unix://${SOCKET}"
echo "export TB2_UV_CACHE_BIND_MOUNT=1"
echo "export TB2_UV_CACHE_REMOTE_DIR=${TB2_CACHE_DIR}"
echo "export UNIFIED_HARBOR_REQUIRE_PREBUILT_LOCAL=1"
echo "export UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS=1"
echo
echo "Optional image restore after container recreation:"
echo "RESTORE_LOCAL_IMAGES=1 RESTORE_LOCAL_IMAGE_WORKERS=${RESTORE_LOCAL_IMAGE_WORKERS} bash ops/launch/prepare_local_rl_docker_runtime.sh"
