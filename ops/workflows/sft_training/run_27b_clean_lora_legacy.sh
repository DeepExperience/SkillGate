#!/usr/bin/env bash
# NOTE: Migrated canonical workflow copy. Source: GeneralAgent/sft_training/scripts/run_qwen35_27b_lora_8gpu_64k_5epoch_r32_clean.sh
# Original historical script is archived during workflow cleanup; maintain this copy going forward.
set -Eeuo pipefail

PROJECT_ROOT="${SKILLRL_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

CFG="GeneralAgent/sft_training/configs/qwen35_27b_lora_campaign_20260509_clean_8gpu_64k_5epoch_r32_liger.yaml"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export NPROC_PER_NODE=8
export NNODES=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export LLAMAFACTORY_ALLOW_TORCH29_CONV3D="${LLAMAFACTORY_ALLOW_TORCH29_CONV3D:-1}"

export CUDA_HOME="${CUDA_HOME:-${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/envs/slime}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

source GeneralAgent/sft_training/activate_llamafactory.sh

export CUDA_HOME="${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/envs/slime"
export CUDA_PATH="${CUDA_HOME}"
export LLAMAFACTORY_VENV="${PROJECT_ROOT}/GeneralAgent/.venvs/llamafactory"
export PATH="${LLAMAFACTORY_VENV}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "CUDA_HOME=${CUDA_HOME}"
which python || true
python -c 'import sys; print(sys.executable)' || true
which nvcc || true
nvcc -V | tail -n 1 || true
python - <<'PY'
import os
print('preflight CUDA_HOME', os.environ.get('CUDA_HOME'))
import deepspeed
print('preflight deepspeed', deepspeed.__version__)
import liger_kernel
print('preflight liger_kernel module imported')
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
print('preflight liger qwen3_5 patch ready')
PY

exec llamafactory-cli train "${CFG}"
