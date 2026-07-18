#!/usr/bin/env bash
# NOTE: Migrated canonical workflow copy. Source: GeneralAgent/sft_training/scripts/run_qwen35_9b_lora_4gpu_49k_5epoch_r32_clean_plus_claw_thinkwrap_20260512.sh
# Original historical script is archived during workflow cleanup; maintain this copy going forward.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${PROJECT_ROOT}"

CFG="GeneralAgent/sft_training/configs/qwen35_9b_lora_campaign_20260512_clean_plus_claw_thinkwrap_4gpu_49k_5epoch_r32_liger.yaml"
GPUS="${GPUS:-4,5,6,7}"
LOCAL_HF_CACHE="${LOCAL_HF_CACHE:-/tmp/${USER:-root}_20260512_9b_clean_plus_claw_hf_cache}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export NPROC_PER_NODE=4
export NNODES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export LLAMAFACTORY_ALLOW_TORCH29_CONV3D="${LLAMAFACTORY_ALLOW_TORCH29_CONV3D:-1}"

export CUDA_HOME="${CUDA_HOME:-${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/envs/slime}"
export CUDA_PATH="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${LOCAL_HF_CACHE}"
export TRANSFORMERS_CACHE="${LOCAL_HF_CACHE}/transformers"
export HF_DATASETS_CACHE="${LOCAL_HF_CACHE}/datasets"

mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

source GeneralAgent/sft_training/activate_llamafactory.sh

export CUDA_HOME="${CUDA_HOME:-${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/envs/slime}"
export CUDA_PATH="${CUDA_HOME}"
export LLAMAFACTORY_VENV="${PROJECT_ROOT}/GeneralAgent/.venvs/llamafactory"
export PATH="${LLAMAFACTORY_VENV}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "HF_HOME=${HF_HOME}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
which python || true
python -c 'import sys; print(sys.executable)' || true
which nvcc || true
nvcc -V | tail -n 1 || true
python - <<'PY'
import os
print("preflight CUDA_HOME", os.environ.get("CUDA_HOME"))
import deepspeed
print("preflight deepspeed", deepspeed.__version__)
import liger_kernel
print("preflight liger_kernel module imported")
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
print("preflight liger qwen3_5 patch ready")
PY

exec llamafactory-cli train "${CFG}"
