#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Entrypoint / source helper for Ray Job tasks.
# The Ray cluster is already running. This script MUST NOT kill ray or stop the
# cluster. It only cleans up residual python/sglang processes and then sets up
# the environment for running training against an existing Ray cluster.
#
# Two usage modes:
#   1) Entry-point mode — first argument is a .sh script path:
#        bash scripts/entrypoint/ray-job.sh <run-script> [extra-args...]
#      Sets up env, cleans residual processes, then execs the run script.
#
#      Example:
#        bash scripts/entrypoint/ray-job.sh scripts/training/text/run-qwen35-9B-8xgpu-async.sh
#        bash scripts/entrypoint/ray-job.sh scripts/training/text/run-qwen35-9B-8xgpu-async.sh --lr 5e-7
#
#   2) Source mode — no .sh script arg (like local.sh):
#        source scripts/entrypoint/ray-job.sh
#      Sets up env only, so the caller can continue execution.
#
# Environment variables (optional):
#   MEGATRON      - Path to Megatron-LM (default: <Relax>/deps/Megatron-LM)
#   RELAX         - Path to Relax project (default: ../../)

# Guard: skip if already sourced by another entrypoint
if [ -n "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    return 0 2>/dev/null || exit 0
fi

# ── mode detection ──────────────────────────────────────────────────────────
# Entry-point mode: directly executed AND first arg is an existing .sh file.
# Otherwise act as a sourced setup script.
_RAY_JOB_RUN_SCRIPT=""
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    _RAY_JOB_FIRST_ARG="${1:-}"
    if [ -n "$_RAY_JOB_FIRST_ARG" ] && [ -f "$_RAY_JOB_FIRST_ARG" ] && [[ "$_RAY_JOB_FIRST_ARG" == *.sh ]]; then
        _RAY_JOB_RUN_SCRIPT="$_RAY_JOB_FIRST_ARG"
        shift
    else
        echo "Usage: $0 <run-script.sh> [extra-args...]" >&2
        exit 1
    fi
fi

set -eo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# ── clean up residual python/sglang processes (NOT ray) ─────────────────────
# IMPORTANT: Do NOT pkill ray or run ray stop — the cluster is managed externally.
echo "=== Cleaning up residual python/sglang processes ==="
_RAY_SERVE_SHUTDOWN_ADDRESS="${RAY_SERVE_DASHBOARD_ADDRESS:-${RAY_DASHBOARD_ADDRESS:-}}"
if [ -z "${_RAY_SERVE_SHUTDOWN_ADDRESS}" ]; then
    _RAY_SERVE_HEAD_IP=$((timeout "${RELAX_RAY_LIST_NODES_TIMEOUT_SEC:-20}" ray list nodes --format json || true) | jq -r '
      map(select(.is_head_node == true)) |
      .[0].node_ip // empty
    ')
    if [ -n "${_RAY_SERVE_HEAD_IP}" ] && [ "${_RAY_SERVE_HEAD_IP}" != "null" ]; then
        _RAY_SERVE_SHUTDOWN_ADDRESS="http://${_RAY_SERVE_HEAD_IP}:8265"
    fi
fi
if [ -n "${_RAY_SERVE_SHUTDOWN_ADDRESS}" ]; then
    timeout "${RELAX_RAY_SERVE_SHUTDOWN_TIMEOUT_SEC:-30}" ray serve shutdown --address "${_RAY_SERVE_SHUTDOWN_ADDRESS}" -y || echo "ray serve shutdown failed at ${_RAY_SERVE_SHUTDOWN_ADDRESS}; continuing with process cleanup"
else
    timeout "${RELAX_RAY_SERVE_SHUTDOWN_TIMEOUT_SEC:-30}" ray serve shutdown -y || echo "ray serve shutdown failed; continuing with process cleanup"
fi
if [[ "${RELAX_SKIP_RAY_NODE_CLEANUP:-0}" == "1" ]]; then
    echo "RELAX_SKIP_RAY_NODE_CLEANUP=1; skipping run_on_each_ray_node cleanup"
else
    "${RELAX_PYTHON:-python}" ${DIR}/../tools/run_on_each_ray_node.py ${DIR}/../tools/kill_for_ray.sh || echo "failed"
fi

# kill old tasks. Ray dashboard/state API can hang on large clusters; never let this
# best-effort cleanup block a training launch.
if [[ "${RELAX_SKIP_RAY_JOB_STOP:-0}" == "1" ]]; then
    echo "RELAX_SKIP_RAY_JOB_STOP=1; skipping broad Ray job stop"
else
    (timeout "${RELAX_RAY_JOB_LIST_TIMEOUT_SEC:-20}" ray job list || true) \
      | grep RUNNING \
      | grep -v job_id=None \
      | grep -oP "submission_id='\\K[^']+" \
      | xargs -r ray job stop || true
fi

set -x

# ── environment setup ───────────────────────────────────────────────────────
# Use the first GPU node as MASTER_ADDR (prefer head node).
# NOTE: assignment is split from `export` on purpose — `export VAR=$(...)`
# always returns 0 (export's own exit code), which would mask failures of
# the command substitution and defeat `set -eo pipefail` set above.
if [ -n "${RAY_MASTER_ADDR_OVERRIDE:-}" ]; then
    MASTER_ADDR="${RAY_MASTER_ADDR_OVERRIDE}"
else
    MASTER_ADDR=$((timeout "${RELAX_RAY_LIST_NODES_TIMEOUT_SEC:-20}" ray list nodes --format json || true) | jq -r '
      map(select(.state == "ALIVE" and (.resources_total.GPU // 0) > 0)) |
      sort_by(.is_head_node | not) |
      .[0].node_ip
    ')
fi
if { [ -z "$MASTER_ADDR" ] || [ "$MASTER_ADDR" = "null" ]; } && [ -n "${SLIME_HOST_IP:-}" ]; then
    MASTER_ADDR="${SLIME_HOST_IP}"
fi
if [ -z "$MASTER_ADDR" ] || [ "$MASTER_ADDR" = "null" ]; then
    echo "ERROR: failed to resolve MASTER_ADDR (no ALIVE GPU node returned by 'ray list nodes')." >&2
    exit 1
fi
export MASTER_ADDR

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON=${MEGATRON:-${DIR}/../../deps/Megatron-LM}
export RELAX=${RELAX:-${DIR}/../../}
export PYTHONPATH=${RELAX}:$MEGATRON:$RELAX:${PYTHONPATH:-}
export MODEL_CONFIG_DIR="${DIR}/../models"
RELAX_FAST_CUDA_HOME="${RELAX_FAST_CUDA_HOME:-${ROOT:-${RELAX}/..}/ops/cache/cuda_fast_home}"
export CUDNN_HOME="${CUDNN_HOME:-/usr/local/lib/python3.12/dist-packages/nvidia/cudnn}"
if [ -d "${RELAX_FAST_CUDA_HOME}" ] && { [ -z "${CUDA_HOME:-}" ] || [ "${CUDA_HOME}" = "/usr/local/cuda" ] || [[ "${CUDA_HOME}" == */anaconda3 ]]; }; then
  export CUDA_HOME="${RELAX_FAST_CUDA_HOME}"
fi
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
if [ -d "${RELAX_FAST_CUDA_HOME}" ] && { [ "${CUDA_PATH:-}" = "/usr/local/cuda" ] || [[ "${CUDA_PATH:-}" == */anaconda3 ]]; }; then
  export CUDA_PATH="${RELAX_FAST_CUDA_HOME}"
fi
if [ -n "${RELAX_DOCKER_HOST:-}" ]; then
  export DOCKER_HOST="${RELAX_DOCKER_HOST}"
elif [ -z "${DOCKER_HOST:-}" ] \
  || [ "${DOCKER_HOST}" = "tcp://127.0.0.1:2375" ] \
  || [ "${DOCKER_HOST}" = "tcp://127.0.0.1:2376" ] \
  || [ "${DOCKER_HOST}" = "unix:///tmp/apex-docker.sock" ]; then
  export DOCKER_HOST="${RL_DOCKER_LOCAL_HOST:-unix:///tmp/local-docker-overlay2.sock}"
fi
RELAX_NO_PROXY_DEFAULT="127.0.0.1,localhost,0.0.0.0,10.0.0.0/8,172.16.0.0/12,mirrors.tuna.tsinghua.edu.cn,pypi.tuna.tsinghua.edu.cn,hf-mirror.com"
export NO_PROXY="${RELAX_NO_PROXY:-${NO_PROXY:-${RELAX_NO_PROXY_DEFAULT}}}"
case ",${NO_PROXY}," in
  *",10.0.0.0/8,"*) ;;
  *) export NO_PROXY="${NO_PROXY},10.0.0.0/8,172.16.0.0/12,mirrors.tuna.tsinghua.edu.cn,pypi.tuna.tsinghua.edu.cn,hf-mirror.com" ;;
esac
export no_proxy="${NO_PROXY}"

# ── NVLink detection ────────────────────────────────────────────────────────
if nvidia-smi 2>&1 > /dev/null; then
    NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
else
    NVLINK_COUNT=0
fi
if [ "$NVLINK_COUNT" -gt 0 ]; then
    export HAS_NVLINK=1
else
    export HAS_NVLINK=0
fi
if [ -n "${NCCL_NVLS_ENABLE:-}" ] && [ "${NCCL_NVLS_ENABLE}" -eq 0 ]; then
    export HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ── entrypoint mode & runtime env ──────────────────────────────────────────
export RELAX_ENTRYPOINT_MODE="ray-job"
RAY_DEBUG=${RAY_DEBUG:-"0"}
RAY_DEBUG_POST_MORTEM=${RAY_DEBUG_POST_MORTEM:-"0"}
if [ "${RELAX_PIN_SERVE_CONTROLLER_TO_ROLLOUT:-0}" = "1" ] && [ -n "${RELAX_PIN_NODE_ROLLOUT:-}" ]; then
    export RAY_SERVE_CONTROLLER_NODE_RESOURCE="node:${RELAX_PIN_NODE_ROLLOUT}"
fi

# Runtime env for ray-job mode (env inherited from Ray cluster)
NVSHMEM_LIB_PATH="${NVSHMEM_LIB_PATH:-/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib}"
CURRENT_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${NVSHMEM_LIB_PATH}"

export RUNTIME_ENV_JSON="{
\"worker_process_setup_hook\": \"relax.utils.logging_utils.install_asyncio_noise_filter\",
\"env_vars\": {
   \"PYTHONUNBUFFERED\": \"1\",
   \"TOKENIZERS_PARALLELISM\": \"${TOKENIZERS_PARALLELISM:-false}\",
   \"RAYON_NUM_THREADS\": \"${RAYON_NUM_THREADS:-1}\",
   \"OMP_NUM_THREADS\": \"${OMP_NUM_THREADS:-1}\",
   \"MKL_NUM_THREADS\": \"${MKL_NUM_THREADS:-1}\",
   \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF:-}\",
   \"TORCH_COMPILE_DISABLE\": \"${TORCH_COMPILE_DISABLE:-0}\",
   \"TORCHDYNAMO_DISABLE\": \"${TORCHDYNAMO_DISABLE:-0}\",
   \"RELAX_TRAIN_STEP_DIAG\": \"${RELAX_TRAIN_STEP_DIAG:-0}\",
   \"RELAX_TRAIN_SUBSTEP_DIAG\": \"${RELAX_TRAIN_SUBSTEP_DIAG:-0}\",
   \"PYTHONPATH\": \"${PYTHONPATH}\",
   \"PATH\": \"${PATH:-/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}\",
	   \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
	   \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\",
	   \"RAY_SERVE_CONTROLLER_NODE_RESOURCE\": \"${RAY_SERVE_CONTROLLER_NODE_RESOURCE:-}\",
	   \"RAY_SERVE_HTTP_PROXY_TIMEOUT_S\": \"${RAY_SERVE_HTTP_PROXY_TIMEOUT_S:-60}\",
	   \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
   \"MASTER_ADDR\": \"${MASTER_ADDR}\",
	   \"RAY_DEBUG\": \"${RAY_DEBUG}\",
	   \"RAY_DEBUG_POST_MORTEM\": \"${RAY_DEBUG_POST_MORTEM}\",
	   \"DOCKER_HOST\": \"${DOCKER_HOST:-unix:///tmp/local-docker-overlay2.sock}\",
	   \"HTTP_PROXY\": \"${HTTP_PROXY:-}\",
	   \"HTTPS_PROXY\": \"${HTTPS_PROXY:-}\",
	   \"ALL_PROXY\": \"${ALL_PROXY:-}\",
	   \"http_proxy\": \"${http_proxy:-}\",
	   \"https_proxy\": \"${https_proxy:-}\",
	   \"all_proxy\": \"${all_proxy:-}\",
	   \"NO_PROXY\": \"${NO_PROXY:-127.0.0.1,localhost,0.0.0.0,10.0.0.0/8,172.16.0.0/12}\",
	   \"no_proxy\": \"${no_proxy:-127.0.0.1,localhost,0.0.0.0,10.0.0.0/8,172.16.0.0/12}\",
	   \"NCCL_IB_DISABLE\": \"${NCCL_IB_DISABLE:-0}\",
	   \"NCCL_ASYNC_ERROR_HANDLING\": \"${NCCL_ASYNC_ERROR_HANDLING:-1}\",
	   \"TORCH_NCCL_ASYNC_ERROR_HANDLING\": \"${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}\",
	   \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME:-eth0}\",
	   \"UNIFIED_LAUNCHER_MODE\": \"${UNIFIED_LAUNCHER_MODE:-real}\",
	   \"UNIFIED_CLAW_USE_DOCKER_SANDBOX\": \"${UNIFIED_CLAW_USE_DOCKER_SANDBOX:-1}\",
	   \"UNIFIED_CLAW_SANDBOX_FAIL_HARD\": \"${UNIFIED_CLAW_SANDBOX_FAIL_HARD:-1}\",
	   \"UNIFIED_DISABLE_THINKING\": \"${UNIFIED_DISABLE_THINKING:-1}\",
	   \"UNIFIED_ROLLOUT_WALLCLOCK_CAP_SEC\": \"${UNIFIED_ROLLOUT_WALLCLOCK_CAP_SEC:-900}\",
	   \"AGENT_BENCH_DOCKER_START_CONCURRENCY\": \"${AGENT_BENCH_DOCKER_START_CONCURRENCY:-4}\",
	   \"AGENT_BENCH_ACTIVE_ENV_CONCURRENCY\": \"${AGENT_BENCH_ACTIVE_ENV_CONCURRENCY:-12}\",
	   \"AGENT_BENCH_SETUP_ATTEMPTS\": \"${AGENT_BENCH_SETUP_ATTEMPTS:-3}\",
	   \"AGENT_BENCH_SETUP_TOTAL_TIMEOUT_SEC\": \"${AGENT_BENCH_SETUP_TOTAL_TIMEOUT_SEC:-600}\",
	   \"AGENT_BENCH_RETRIEVAL_TOP_N\": \"${AGENT_BENCH_RETRIEVAL_TOP_N:-10}\",
	   \"AGENT_BENCH_EXTRA_SKILL_ROOTS\": \"${AGENT_BENCH_EXTRA_SKILL_ROOTS:-}\",
	   \"UNIFIED_HARBOR_BUILD_TIMEOUT_SEC\": \"${UNIFIED_HARBOR_BUILD_TIMEOUT_SEC:-300}\",
	   \"UNIFIED_HARBOR_REQUIRE_PREBUILT_LOCAL\": \"${UNIFIED_HARBOR_REQUIRE_PREBUILT_LOCAL:-0}\",
	   \"UNIFIED_VERIFIER_TIMEOUT_CAP_SEC\": \"${UNIFIED_VERIFIER_TIMEOUT_CAP_SEC:-900}\",
	   \"UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS\": \"${UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS:-1}\",
	   \"UNIFIED_DOCKER_PIDS_LIMIT\": \"${UNIFIED_DOCKER_PIDS_LIMIT:-1024}\",
	   \"UNIFIED_DOCKER_ULIMIT_FSIZE_GB\": \"${UNIFIED_DOCKER_ULIMIT_FSIZE_GB:-}\",
	   \"UNIFIED_DOCKER_CPUSET\": \"${UNIFIED_DOCKER_CPUSET:-}\",
	   \"UNIFIED_DOCKER_BUILD_JOBS\": \"${UNIFIED_DOCKER_BUILD_JOBS:-}\",
	   \"UNIFIED_DOCKER_NETWORK_HOST\": \"${UNIFIED_DOCKER_NETWORK_HOST:-1}\",
	   \"UNIFIED_CONTAINER_PROXY\": \"${UNIFIED_CONTAINER_PROXY:-}\",
	   \"UNIFIED_TOOL_TIMEOUT_CHILD_CLEANUP\": \"${UNIFIED_TOOL_TIMEOUT_CHILD_CLEANUP:-1}\",
	   \"SETA_CONTINUOUS_REWARD\": \"${SETA_CONTINUOUS_REWARD:-0}\",
	   \"TB2_UV_CACHE_BIND_MOUNT\": \"${TB2_UV_CACHE_BIND_MOUNT:-1}\",
	   \"TB2_UV_CACHE_REMOTE_DIR\": \"${TB2_UV_CACHE_REMOTE_DIR:-/data/cache/tb2_uv_cache/tb2-uv}\",
	   \"UNIFIED_SWE_VERIFIER_TIMEOUT_SEC\": \"${UNIFIED_SWE_VERIFIER_TIMEOUT_SEC:-300}\",
	   \"AGENT_BENCH_SKIP_CLOSE_GRADING_ON_ABORT\": \"${AGENT_BENCH_SKIP_CLOSE_GRADING_ON_ABORT:-0}\",
	   \"RELAX_REQUEUE_ABORTED_GROUPS\": \"${RELAX_REQUEUE_ABORTED_GROUPS:-0}\",
	   \"RELAX_MAX_DROPPED_ABORT_GROUPS_PER_ROLLOUT\": \"${RELAX_MAX_DROPPED_ABORT_GROUPS_PER_ROLLOUT:-64}\",
	   \"RELAX_DYNAMIC_FILTER_MAX_REJECTS_PER_ROLLOUT\": \"${RELAX_DYNAMIC_FILTER_MAX_REJECTS_PER_ROLLOUT:-0}\",
	   \"RELAX_DYNAMIC_FILTER_MAX_REJECT_SAMPLES_PER_ROLLOUT\": \"${RELAX_DYNAMIC_FILTER_MAX_REJECT_SAMPLES_PER_ROLLOUT:-0}\",
	   \"RELAX_DYNAMIC_FILTER_MIN_SKILL_READ_FRAC\": \"${RELAX_DYNAMIC_FILTER_MIN_SKILL_READ_FRAC:-0}\",
	   \"RELAX_DYNAMIC_FILTER_MIN_NO_SKILL_READ_FRAC\": \"${RELAX_DYNAMIC_FILTER_MIN_NO_SKILL_READ_FRAC:-0}\",
	   \"RELAX_DYNAMIC_FILTER_SKILL_READ_MAX_SAMPLES\": \"${RELAX_DYNAMIC_FILTER_SKILL_READ_MAX_SAMPLES:-0}\",
	   \"RELAX_SKILL_GROUP_REWARD\": \"${RELAX_SKILL_GROUP_REWARD:-0}\",
	   \"RELAX_SKILL_GROUP_BONUS_COEF\": \"${RELAX_SKILL_GROUP_BONUS_COEF:-0.1}\",
	   \"RELAX_SKILL_GROUP_BONUS_MAX\": \"${RELAX_SKILL_GROUP_BONUS_MAX:-0.2}\",
	   \"RELAX_SKILL_GROUP_MARGIN\": \"${RELAX_SKILL_GROUP_MARGIN:-0.0}\",
	   \"RELAX_SKILL_GROUP_SUBGROUP_ADV_COEF\": \"${RELAX_SKILL_GROUP_SUBGROUP_ADV_COEF:-0.0}\",
	   \"RELAX_SKILL_GROUP_REQUIRE_BOTH\": \"${RELAX_SKILL_GROUP_REQUIRE_BOTH:-1}\",
	   \"RELAX_SKILL_GROUP_NO_READ_SUCCESS_BONUS\": \"${RELAX_SKILL_GROUP_NO_READ_SUCCESS_BONUS:-0.0}\",
	   \"RELAX_SKILL_GROUP_NO_READ_SUCCESS_THRESHOLD\": \"${RELAX_SKILL_GROUP_NO_READ_SUCCESS_THRESHOLD:-1.0}\",
	   \"RELAX_MIXED_SKILL_BONUS_ENABLED\": \"${RELAX_MIXED_SKILL_BONUS_ENABLED:-0}\",
		   \"RELAX_MIXED_SKILL_BONUS_ORACLE\": \"${RELAX_MIXED_SKILL_BONUS_ORACLE:-0.30}\",
		   \"RELAX_MIXED_SKILL_BONUS_MISLEADING\": \"${RELAX_MIXED_SKILL_BONUS_MISLEADING:-0.30}\",
		   \"RELAX_MIXED_SKILL_BONUS_NO_READ_SUCCESS\": \"${RELAX_MIXED_SKILL_BONUS_NO_READ_SUCCESS:-0.35}\",
		   \"RELAX_MIXED_SEPARATED_ADV_ENABLED\": \"${RELAX_MIXED_SEPARATED_ADV_ENABLED:-0}\",
		   \"RELAX_MIXED_SEPARATED_BEHAVIOR_COEF\": \"${RELAX_MIXED_SEPARATED_BEHAVIOR_COEF:-0.30}\",
		   \"RELAX_MIXED_SEPARATED_BEHAVIOR_CLIP\": \"${RELAX_MIXED_SEPARATED_BEHAVIOR_CLIP:-0.40}\",
		   \"RELAX_SELECTOR_ACTION_CREDIT\": \"${RELAX_SELECTOR_ACTION_CREDIT:-0}\",
		   \"RELAX_SELECTOR_ACTION_LOSS_COEF\": \"${RELAX_SELECTOR_ACTION_LOSS_COEF:-0.20}\",
		   \"RELAX_ABORT_PENDING_TIMEOUT_SEC\": \"${RELAX_ABORT_PENDING_TIMEOUT_SEC:-180}\",
		   \"RELAX_ABORT_PROTECTED_TIMEOUT_SEC\": \"${RELAX_ABORT_PROTECTED_TIMEOUT_SEC:-180}\",
		   \"RELAX_ABORT_CANCEL_WAIT_SEC\": \"${RELAX_ABORT_CANCEL_WAIT_SEC:-5}\",
		   \"RELAX_SHADOW_BC_HARD_SPAN_MASK\": \"${RELAX_SHADOW_BC_HARD_SPAN_MASK:-0}\",
		   \"RELAX_HARD_SPAN_VERSION\": \"${RELAX_HARD_SPAN_VERSION:-v1}\",
		   \"RELAX_HARD_SPAN_ACTION_MASK_MODE\": \"${RELAX_HARD_SPAN_ACTION_MASK_MODE:-tool_call}\",
		   \"RELAX_HARD_SPAN_MASK_RENORMALIZE\": \"${RELAX_HARD_SPAN_MASK_RENORMALIZE:-1}\",
		   \"RELAX_HARD_SPAN_REASONING_MAX_CHARS\": \"${RELAX_HARD_SPAN_REASONING_MAX_CHARS:-4096}\",
		   \"RELAX_HARD_SPAN_FINAL_MAX_CHARS\": \"${RELAX_HARD_SPAN_FINAL_MAX_CHARS:-4096}\",
		   \"RELAX_HARD_SPAN_MAX_RESPONSE_TOKENS\": \"${RELAX_HARD_SPAN_MAX_RESPONSE_TOKENS:-0}\",
		   \"RELAX_HARD_SPAN_KEEP_FINAL\": \"${RELAX_HARD_SPAN_KEEP_FINAL:-1}\",
		   \"RELAX_HARD_SPAN_REQUIRE_USEFUL_REASONING\": \"${RELAX_HARD_SPAN_REQUIRE_USEFUL_REASONING:-1}\",
		   \"RELAX_PAIR_ORACLE_GRPO\": \"${RELAX_PAIR_ORACLE_GRPO:-0}\",
		   \"RELAX_PAIR_ORACLE_GRPO_CROSS_ARM_ADV\": \"${RELAX_PAIR_ORACLE_GRPO_CROSS_ARM_ADV:-0}\",
		   \"RELAX_PAIR_ORACLE_GRPO_CROSS_ARM_ADV_CLIP\": \"${RELAX_PAIR_ORACLE_GRPO_CROSS_ARM_ADV_CLIP:-}\",
		   \"RELAX_PAIR_ORACLE_GRPO_DROP_ALL_PASS\": \"${RELAX_PAIR_ORACLE_GRPO_DROP_ALL_PASS:-0}\",
		   \"RELAX_SLATE_REGRET_GRPO\": \"${RELAX_SLATE_REGRET_GRPO:-0}\",
		   \"RELAX_SLATE_UNIFORM_MIN_DELTA\": \"${RELAX_SLATE_UNIFORM_MIN_DELTA:-0.25}\",
		   \"RELAX_SLATE_REGRET_COEF\": \"${RELAX_SLATE_REGRET_COEF:-0.5}\",
		   \"RELAX_SLATE_REGRET_COEF_NOGOLD\": \"${RELAX_SLATE_REGRET_COEF_NOGOLD:-}\",
		   \"RELAX_SLATE_STRATIFIED_ADVANTAGE\": \"${RELAX_SLATE_STRATIFIED_ADVANTAGE:-0}\",
		   \"RELAX_SLATE_STRATIFIED_ADV_COEF\": \"${RELAX_SLATE_STRATIFIED_ADV_COEF:-1.0}\",
		   \"RELAX_SLATE_STRATIFIED_SHRINKAGE\": \"${RELAX_SLATE_STRATIFIED_SHRINKAGE:-1.0}\",
		   \"RELAX_SLATE_STRATIFIED_ADV_CLIP\": \"${RELAX_SLATE_STRATIFIED_ADV_CLIP:-0.5}\",
		   \"RELAX_OPSD_MODE\": \"${RELAX_OPSD_MODE:-0}\",
		   \"RELAX_OPSD_FORM\": \"${RELAX_OPSD_FORM:-k1adv}\",
		   \"RELAX_OPSD_K3_COEF\": \"${RELAX_OPSD_K3_COEF:-}\",
		   \"RELAX_OPSD_K3_ELL_CLAMP\": \"${RELAX_OPSD_K3_ELL_CLAMP:-}\",
		   \"RELAX_OPSD_K3_RHO_CLAMP\": \"${RELAX_OPSD_K3_RHO_CLAMP:-}\",
		   \"RELAX_OPSD_SKILL_TOKEN_MASK\": \"${RELAX_OPSD_SKILL_TOKEN_MASK:-0}\",
		   \"RELAX_OPSD_KL_COEF\": \"${RELAX_OPSD_KL_COEF:-}\",
		   \"RELAX_OPSD_SCOPE\": \"${RELAX_OPSD_SCOPE:-mixed}\",
		   \"RELAX_OPSD_TEACHER_SELF_ROUTER\": \"${RELAX_OPSD_TEACHER_SELF_ROUTER:-0}\",
		   \"RELAX_OPSD_TEACHER_TIMEOUT_S\": \"${RELAX_OPSD_TEACHER_TIMEOUT_S:-}\",
		   \"RELAX_PIN_NODE_ACTOR\": \"${RELAX_PIN_NODE_ACTOR:-}\",
	   \"RELAX_PIN_NODE_ACTOR_FWD\": \"${RELAX_PIN_NODE_ACTOR_FWD:-}\",
	   \"RELAX_PIN_NODE_REFERENCE\": \"${RELAX_PIN_NODE_REFERENCE:-}\",
	   \"RELAX_PIN_NODE_ROLLOUT\": \"${RELAX_PIN_NODE_ROLLOUT:-}\",
	   \"RELAX_DCS_MASTER_PORT_MIN\": \"${RELAX_DCS_MASTER_PORT_MIN:-24000}\",
	   \"RELAX_DCS_MASTER_PORT_MAX\": \"${RELAX_DCS_MASTER_PORT_MAX:-24999}\",
	   \"RELAX_DCS_ACTOR_FWD_REF_PORT_MIN\": \"${RELAX_DCS_ACTOR_FWD_REF_PORT_MIN:-25000}\",
	   \"RELAX_DCS_ACTOR_FWD_REF_PORT_MAX\": \"${RELAX_DCS_ACTOR_FWD_REF_PORT_MAX:-25999}\",
	   \"RELAX_TRAIN_MASTER_PORT_ACTOR_MIN\": \"${RELAX_TRAIN_MASTER_PORT_ACTOR_MIN:-20000}\",
	   \"RELAX_TRAIN_MASTER_PORT_ACTOR_MAX\": \"${RELAX_TRAIN_MASTER_PORT_ACTOR_MAX:-20999}\",
	   \"RELAX_TRAIN_MASTER_PORT_ACTOR_FWD_MIN\": \"${RELAX_TRAIN_MASTER_PORT_ACTOR_FWD_MIN:-21000}\",
	   \"RELAX_TRAIN_MASTER_PORT_ACTOR_FWD_MAX\": \"${RELAX_TRAIN_MASTER_PORT_ACTOR_FWD_MAX:-21999}\",
	   \"RELAX_TRAIN_MASTER_PORT_REFERENCE_MIN\": \"${RELAX_TRAIN_MASTER_PORT_REFERENCE_MIN:-22000}\",
	   \"RELAX_TRAIN_MASTER_PORT_REFERENCE_MAX\": \"${RELAX_TRAIN_MASTER_PORT_REFERENCE_MAX:-22999}\",
	   \"RELAX_TRAIN_MASTER_PORT_CRITIC_MIN\": \"${RELAX_TRAIN_MASTER_PORT_CRITIC_MIN:-26000}\",
	   \"RELAX_TRAIN_MASTER_PORT_CRITIC_MAX\": \"${RELAX_TRAIN_MASTER_PORT_CRITIC_MAX:-26999}\",
	   \"RELAX_TRAIN_MASTER_PORT_DEFAULT_MIN\": \"${RELAX_TRAIN_MASTER_PORT_DEFAULT_MIN:-27000}\",
	   \"RELAX_TRAIN_MASTER_PORT_DEFAULT_MAX\": \"${RELAX_TRAIN_MASTER_PORT_DEFAULT_MAX:-29999}\",
	   \"CUDNN_HOME\": \"${CUDNN_HOME}\",
	   \"CUDA_HOME\": \"${CUDA_HOME}\",
	   \"CUDA_PATH\": \"${CUDA_PATH}\",
	   \"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK\": \"${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-32}\",
	   \"NVSHMEM_DISABLE_NCCL\": \"${NVSHMEM_DISABLE_NCCL:-1}\",
   \"SGLANG_HEALTH_CHECK_TIMEOUT\": \"${SGLANG_HEALTH_CHECK_TIMEOUT:-180}\",
   \"INDEXER_ROPE_NEOX_STYLE\": \"${INDEXER_ROPE_NEOX_STYLE:-0}\",
   \"NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME\": \"${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-${NCCL_SOCKET_IFNAME}}\",
   \"LD_LIBRARY_PATH\": \"${CURRENT_LD_LIBRARY_PATH}\"
}
}"

echo "=== Ray-job environment ready ==="

# ── delegate to run script (entry-point mode only) ─────────────────────────
if [ -n "$_RAY_JOB_RUN_SCRIPT" ]; then
    echo "=== Launching training script: $_RAY_JOB_RUN_SCRIPT ==="
    exec bash "$_RAY_JOB_RUN_SCRIPT" "$@"
fi
