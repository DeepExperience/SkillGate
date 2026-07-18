#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${SKILLRL_ROOT:-$(pwd)}"
export LLAMAFACTORY_ROOT="${PROJECT_ROOT}/GeneralAgent/third_party/LLaMA-Factory"

source "${PROJECT_ROOT}/GeneralAgent/.venvs/llamafactory/bin/activate"

export PYTHONPATH="${LLAMAFACTORY_ROOT}/src:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export NNODES="${NNODES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
