#!/usr/bin/env bash
# Shared 16-GPU Relax runtime for the maintained RL profiles.
# This file defines functions only; run_rl.sh is the user-facing entrypoint.

rl_reset_algorithm_env() {
  unset CUSTOM_LOSS_FUNCTION_PATH CUSTOM_REWARD_POST_PROCESS_PATH \
    DYNAMIC_SAMPLING_FILTER_PATH RELAX_PAIR_ORACLE_BC_UNTIL_STEP \
    RELAX_SLATE_REGRET_COEF_NOGOLD RELAX_SLATE_UNIFORM_MIN_DELTA \
    RELAX_SLATE_REGRET_COEF RELAX_SLATE_STRATIFIED_ADV_COEF \
    RELAX_SLATE_STRATIFIED_SHRINKAGE RELAX_SLATE_STRATIFIED_ADV_CLIP \
    RELAX_MIXED_SEPARATED_BEHAVIOR_COEF RELAX_MIXED_SEPARATED_BEHAVIOR_CLIP \
    RELAX_SELECTOR_ACTION_LOSS_COEF RELAX_PAIR_BC_PASS_THRESHOLD \
    AGENT_BENCH_EXTRA_SKILL_ROOTS 2>/dev/null || true

  local key
  for key in \
    RELAX_M1_CLEAN RELAX_PROMPT_ONLY_SHADOW_CLEAN \
    RELAX_SHADOW_BC_ACTION_MASK RELAX_SHADOW_BC_HARD_SPAN_MASK \
    RELAX_SHADOW_BC_COMPAT_WEIGHTS RELAX_COMPAT_ACTION_MONITOR \
    RELAX_MIXED_SKILL_BONUS_ENABLED RELAX_MIXED_SEPARATED_ADV_ENABLED \
    RELAX_SELECTOR_ACTION_CREDIT \
    RELAX_SLATE_REGRET_GRPO RELAX_SLATE_STRATIFIED_ADVANTAGE \
    RELAX_PAIR_ATOMIC_SAMPLING RELAX_PAIR_SPECULATIVE_EXTRA_GROUPS \
    RELAX_PAIR_ORACLE_GRPO RELAX_OPSD_MODE RELAX_OPSD_TEACHER_SELF_ROUTER \
    RELAX_SKILL_GROUP_REWARD RELAX_SKILL_GROUP_BONUS_COEF \
    RELAX_SKILL_GROUP_BONUS_MAX RELAX_SKILL_GROUP_SUBGROUP_ADV_COEF; do
    export "${key}=0"
  done
}

rl_apply_common_defaults() {
  export PYTHON_BIN="${PYTHON_BIN:-${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/envs/relax/bin/python}"
  export RAY_PYTHON_BIN="${RAY_PYTHON_BIN:-/usr/bin/python3}"
  export RELAX_PYTHON="${RELAX_PYTHON:-/usr/bin/python3}"

  export MODEL_DIR="${MODEL_DIR:-${ROOT}/GeneralAgent/sft_training/merged_models}"
  export QWEN35_9B_SFT_SUBDIR="${QWEN35_9B_SFT_SUBDIR:-qwen35_9b_sft_campaign_20260512_clean_plus_claw_thinkwrap_4gpu_49k_5epoch_r32_liger}"
  export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${ROOT}/experiments/rl/runs/${EXPERIMENT_ID}}"
  export SAVE_DIR="${SAVE_DIR:-${EXPERIMENT_DIR}/segments}"

  export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
  export START_ROLLOUT_ID="${START_ROLLOUT_ID:-0}"
  export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
  export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
  export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
  export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
  export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
  export NUM_ITERS_PER_TRAIN_UPDATE="${NUM_ITERS_PER_TRAIN_UPDATE:-4}"
  export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-32}"

  export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-71680}"
  export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-32768}"
  export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-55296}"
  export RELAX_SOFT_OVERLONG_PENALTY="${RELAX_SOFT_OVERLONG_PENALTY:-0}"
  export RELAX_SOFT_OVERLONG_LMAX="${RELAX_SOFT_OVERLONG_LMAX:-55296}"
  export RELAX_SOFT_OVERLONG_CACHE="${RELAX_SOFT_OVERLONG_CACHE:-4096}"

  export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
  export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-2}"
  export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-1}"
  export ACTOR_MAX_TOKENS_PER_GPU="${ACTOR_MAX_TOKENS_PER_GPU:-4096}"
  export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-4096}"
  export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"
  export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
  export USE_KL_LOSS="${USE_KL_LOSS:-1}"
  export KL_LOSS_COEF="${KL_LOSS_COEF:-0.00003}"
  export OVERRIDE_OPT_PARAM_SCHEDULER="${OVERRIDE_OPT_PARAM_SCHEDULER:-1}"
  export RELAX_FORCE_REFERENCE_ROLE="${RELAX_FORCE_REFERENCE_ROLE:-1}"
  export RELAX_SKIP_REFERENCE_ROLE=0
  if [[ -z "${RELAX_RESOURCE_OVERRIDE:-}" ]]; then
    RELAX_RESOURCE_OVERRIDE='{"actor": [1, 8], "rollout": [1, 4], "reference": [1, 2], "actor_fwd": [1, 2], "advantages": [1, 0]}'
  fi
  if [[ -z "${REF_ACTOR_CONFIG:-}" ]]; then
    REF_ACTOR_CONFIG='{"tensor_model_parallel_size": 2, "context_parallel_size": 1, "max_tokens_per_gpu": 4096, "sequence_parallel": true, "only_load_weight": true}'
  fi
  export RELAX_RESOURCE_OVERRIDE REF_ACTOR_CONFIG

  export AGENT_BENCH_ACTIVE_ENV_CONCURRENCY="${AGENT_BENCH_ACTIVE_ENV_CONCURRENCY:-128}"
  export AGENT_BENCH_DOCKER_START_CONCURRENCY="${AGENT_BENCH_DOCKER_START_CONCURRENCY:-128}"
  export AGENT_BENCH_SETUP_ATTEMPTS="${AGENT_BENCH_SETUP_ATTEMPTS:-3}"
  export AGENT_BENCH_SETUP_TOTAL_TIMEOUT_SEC="${AGENT_BENCH_SETUP_TOTAL_TIMEOUT_SEC:-600}"
  export UNIFIED_HARBOR_BUILD_TIMEOUT_SEC="${UNIFIED_HARBOR_BUILD_TIMEOUT_SEC:-300}"
  export UNIFIED_HARBOR_REQUIRE_PREBUILT_LOCAL="${UNIFIED_HARBOR_REQUIRE_PREBUILT_LOCAL:-1}"
  export UNIFIED_ROLLOUT_WALLCLOCK_CAP_SEC="${UNIFIED_ROLLOUT_WALLCLOCK_CAP_SEC:-850}"
  export UNIFIED_VERIFIER_TIMEOUT_CAP_SEC="${UNIFIED_VERIFIER_TIMEOUT_CAP_SEC:-300}"
  export UNIFIED_SWE_VERIFIER_TIMEOUT_SEC="${UNIFIED_SWE_VERIFIER_TIMEOUT_SEC:-300}"
  export UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS="${UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS:-1}"
  export UNIFIED_LAUNCHER_MODE=real
  export UNIFIED_DISABLE_THINKING=1
  export UNIFIED_DOCKER_NETWORK_HOST="${UNIFIED_DOCKER_NETWORK_HOST:-1}"
  export UNIFIED_CONTAINER_PROXY="${UNIFIED_CONTAINER_PROXY:-http://your-proxy:3128}"
  export UNIFIED_DOCKER_PIDS_LIMIT="${UNIFIED_DOCKER_PIDS_LIMIT:-1024}"
  export UNIFIED_TOOL_TIMEOUT_CHILD_CLEANUP="${UNIFIED_TOOL_TIMEOUT_CHILD_CLEANUP:-0}"
  export UNIFIED_DOCKER_ULIMIT_FSIZE_GB="${UNIFIED_DOCKER_ULIMIT_FSIZE_GB:-32}"
  export UNIFIED_DOCKER_CPUSET="${UNIFIED_DOCKER_CPUSET:-24-179}"

  export RL_DOCKER_PREFERRED="${RL_DOCKER_PREFERRED:-local}"
  export RL_DOCKER_LOCAL_HOST="${RL_DOCKER_LOCAL_HOST:-unix:///tmp/local-docker-overlay2.sock}"
  export RELAX_DOCKER_HOST="${RELAX_DOCKER_HOST:-${RL_DOCKER_LOCAL_HOST}}"
  export RL_DOCKER_REQUIRE_LOCAL="${RL_DOCKER_REQUIRE_LOCAL:-1}"
  export LOCAL_DOCKER_DATA_ROOT="${LOCAL_DOCKER_DATA_ROOT:-/data/cache/local-docker-overlay2-root}"
  export LOCAL_DOCKER_EXEC_ROOT="${LOCAL_DOCKER_EXEC_ROOT:-/data/cache/local-docker-overlay2-exec}"
  export LOCAL_DOCKER_USE_SUBREAPER="${LOCAL_DOCKER_USE_SUBREAPER:-1}"
  export RL_DOCKER_DISK_PATH="${RL_DOCKER_DISK_PATH:-$(dirname "${LOCAL_DOCKER_DATA_ROOT}")}"
  export DISK_REAP_WATERMARK_GB="${DISK_REAP_WATERMARK_GB:-800}"

  export RELAX_SKIP_RAY_JOB_STOP="${RELAX_SKIP_RAY_JOB_STOP:-1}"
  export RELAX_SKIP_RAY_NODE_CLEANUP="${RELAX_SKIP_RAY_NODE_CLEANUP:-1}"
  export SKIP_RL_PRELAUNCH_CHECK="${SKIP_RL_PRELAUNCH_CHECK:-1}"
  export RAY_SERVE_HTTP_PROXY_TIMEOUT_S="${RAY_SERVE_HTTP_PROXY_TIMEOUT_S:-300}"
  export RELAX_PIN_SERVE_CONTROLLER_TO_ROLLOUT=1

  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
  export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
  export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
  export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
  export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
  export NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
  export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

  export RELAX_TRAIN_MASTER_PORT_ACTOR_MIN="${RELAX_TRAIN_MASTER_PORT_ACTOR_MIN:-20000}"
  export RELAX_TRAIN_MASTER_PORT_ACTOR_MAX="${RELAX_TRAIN_MASTER_PORT_ACTOR_MAX:-20999}"
  export RELAX_TRAIN_MASTER_PORT_ACTOR_FWD_MIN="${RELAX_TRAIN_MASTER_PORT_ACTOR_FWD_MIN:-21000}"
  export RELAX_TRAIN_MASTER_PORT_ACTOR_FWD_MAX="${RELAX_TRAIN_MASTER_PORT_ACTOR_FWD_MAX:-21999}"
  export RELAX_TRAIN_MASTER_PORT_REFERENCE_MIN="${RELAX_TRAIN_MASTER_PORT_REFERENCE_MIN:-22000}"
  export RELAX_TRAIN_MASTER_PORT_REFERENCE_MAX="${RELAX_TRAIN_MASTER_PORT_REFERENCE_MAX:-22999}"
  export RELAX_DCS_MASTER_PORT_MIN="${RELAX_DCS_MASTER_PORT_MIN:-24000}"
  export RELAX_DCS_MASTER_PORT_MAX="${RELAX_DCS_MASTER_PORT_MAX:-24999}"
  export RELAX_DCS_ACTOR_FWD_REF_PORT_MIN="${RELAX_DCS_ACTOR_FWD_REF_PORT_MIN:-25000}"
  export RELAX_DCS_ACTOR_FWD_REF_PORT_MAX="${RELAX_DCS_ACTOR_FWD_REF_PORT_MAX:-25999}"
  export RELAX_TRAIN_MASTER_PORT_CRITIC_MIN="${RELAX_TRAIN_MASTER_PORT_CRITIC_MIN:-26000}"
  export RELAX_TRAIN_MASTER_PORT_CRITIC_MAX="${RELAX_TRAIN_MASTER_PORT_CRITIC_MAX:-26999}"
  export RELAX_TRAIN_MASTER_PORT_DEFAULT_MIN="${RELAX_TRAIN_MASTER_PORT_DEFAULT_MIN:-27000}"
  export RELAX_TRAIN_MASTER_PORT_DEFAULT_MAX="${RELAX_TRAIN_MASTER_PORT_DEFAULT_MAX:-29999}"

  export DISABLE_HEALTH_CHECK="${DISABLE_HEALTH_CHECK:-1}"
  export DISABLE_METRICS_SERVICE="${DISABLE_METRICS_SERVICE:-0}"
  export PASS_REWARD_THRESHOLD="${PASS_REWARD_THRESHOLD:-1.0}"
  export WANDB_PROJECT="${WANDB_PROJECT:-relax-rl-agent}"
  export PROJECT_NAME="${PROJECT_NAME:-Relax/dev/agent_bench}"
  export USE_CLEARML=0
  export TOKENIZERS_PARALLELISM=false
  export RAYON_NUM_THREADS=1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1

  export RELAX_REQUEUE_ABORTED_GROUPS="${RELAX_REQUEUE_ABORTED_GROUPS:-0}"
  export RELAX_MAX_DROPPED_ABORT_GROUPS_PER_ROLLOUT="${RELAX_MAX_DROPPED_ABORT_GROUPS_PER_ROLLOUT:-64}"
  export RELAX_ABORT_PENDING_TIMEOUT_SEC="${RELAX_ABORT_PENDING_TIMEOUT_SEC:-180}"
  export RELAX_ABORT_PROTECTED_TIMEOUT_SEC="${RELAX_ABORT_PROTECTED_TIMEOUT_SEC:-180}"
  export RELAX_ABORT_CANCEL_WAIT_SEC="${RELAX_ABORT_CANCEL_WAIT_SEC:-5}"

  export EXP_NAME="${RUN_NAME}"
  export RELAX_RL_RUN_ID="${RUN_NAME}"
  export TRAIN_PARQUET="${DATA_DIR}/train.parquet"
  export EVAL_PARQUET="${DATA_DIR}/eval.parquet"
  export RUN_DIR="${SAVE_DIR}/${RUN_NAME}"
  export CHECKPOINT_DIR="${SAVE_DIR}/${RUN_NAME}"
}

rl_configure_identity() {
  local basename="${EXPERIMENT_BASENAME:-${RL_PROFILE}}"
  export EXPERIMENT_ID="${EXPERIMENT_ID:-${basename}-${RL_LAUNCH_STAMP}}"
  if [[ -z "${RUN_NAME:-}" ]]; then
    if [[ -n "${LOAD_DIR:-}" ]]; then
      export RUN_NAME="${RL_LAUNCH_STAMP}-resume-${EXPECTED_LATEST_CKPT:-unknown}"
    else
      export RUN_NAME="${RL_LAUNCH_STAMP}-initial"
    fi
  fi
  [[ "${EXPERIMENT_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$ ]] || {
    echo "FATAL: EXPERIMENT_ID must be a safe single path component: ${EXPERIMENT_ID}" >&2
    return 2
  }
  [[ "${RUN_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$ ]] || {
    echo "FATAL: RUN_NAME/segment id must be a safe single path component: ${RUN_NAME}" >&2
    return 2
  }
  if [[ -n "${LOAD_DIR:-}" && "${RL_EXPERIMENT_ID_WAS_EXPLICIT:-0}" != 1 ]]; then
    echo "FATAL: resume requires an explicit existing EXPERIMENT_ID; do not create a parallel experiment accidentally" >&2
    return 2
  fi
}

rl_load_wandb_credentials() {
  if [[ -z "${WANDB_API_KEY:-}" && -f "${ROOT}/secrets/.env.secrets" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${ROOT}/secrets/.env.secrets"
    set +a
  fi
  # Optional extra fallback: point SKILLRL_WANDB_ENV_FILE at a shell file that
  # contains `export WANDB_API_KEY=...` (and optionally WANDB_SILENT).
  if [[ -z "${WANDB_API_KEY:-}" && -n "${SKILLRL_WANDB_ENV_FILE:-}" && -f "${SKILLRL_WANDB_ENV_FILE}" ]]; then
    eval "$(grep -E '^export WANDB_(API_KEY|SILENT)=' "${SKILLRL_WANDB_ENV_FILE}" || true)"
  fi
  [[ -n "${WANDB_API_KEY:-}" ]] || {
    echo "FATAL: WANDB_API_KEY is unavailable; refusing to launch an untracked RL run." >&2
    return 2
  }
}

rl_validate_resume() {
  if [[ -n "${LOAD_DIR:-}" ]]; then
    : "${EXPECTED_LATEST_CKPT:?LOAD_DIR requires EXPECTED_LATEST_CKPT}"
    : "${START_ROLLOUT_ID:?LOAD_DIR requires START_ROLLOUT_ID}"
    [[ "${EXPECTED_LATEST_CKPT}" =~ ^[0-9]+$ && "${START_ROLLOUT_ID}" =~ ^[0-9]+$ ]] || {
      echo "FATAL: resume checkpoint and rollout ids must be non-negative integers" >&2
      return 2
    }
    [[ -d "${LOAD_DIR}" ]] || { echo "FATAL: LOAD_DIR not found: ${LOAD_DIR}" >&2; return 2; }
    [[ -f "${EXPERIMENT_DIR}/experiment.json" ]] || {
      echo "FATAL: resume owner has no experiment.json: ${EXPERIMENT_DIR}" >&2
      return 2
    }
    local latest
    latest=$(<"${LOAD_DIR}/latest_checkpointed_iteration.txt") || {
      echo "FATAL: missing ${LOAD_DIR}/latest_checkpointed_iteration.txt" >&2
      return 2
    }
    [[ "${latest}" == "${EXPECTED_LATEST_CKPT}" ]] || {
      echo "FATAL: expected checkpoint ${EXPECTED_LATEST_CKPT}, found ${latest} in ${LOAD_DIR}" >&2
      return 2
    }
    (( START_ROLLOUT_ID == EXPECTED_LATEST_CKPT + 1 )) || {
      echo "FATAL: START_ROLLOUT_ID=${START_ROLLOUT_ID} must equal checkpoint+1=$((EXPECTED_LATEST_CKPT + 1))" >&2
      return 2
    }
    [[ "$(realpath -m "${LOAD_DIR}")" != "$(realpath -m "${CHECKPOINT_DIR}")" ]] || {
      echo "FATAL: resume output must use a new RUN_NAME instead of overwriting LOAD_DIR" >&2
      return 2
    }
    if [[ "${ALLOW_CROSS_EXPERIMENT_RESUME:-0}" != 1 ]]; then
      case "$(realpath -m "${LOAD_DIR}")" in
        "$(realpath -m "${EXPERIMENT_DIR}/segments")"/*) ;;
        *)
          echo "FATAL: LOAD_DIR is outside experiment ${EXPERIMENT_ID}; set a new scientific experiment explicitly or ALLOW_CROSS_EXPERIMENT_RESUME=1" >&2
          return 2
          ;;
      esac
    fi
  elif [[ "${START_ROLLOUT_ID}" != "0" ]]; then
    echo "FATAL: START_ROLLOUT_ID=${START_ROLLOUT_ID} requires LOAD_DIR and EXPECTED_LATEST_CKPT" >&2
    return 2
  fi
}

rl_validate_common_config() {
  [[ -f "${DATA_DIR}/train.parquet" && -f "${DATA_DIR}/eval.parquet" ]] || {
    echo "FATAL: DATA_DIR must contain train.parquet and eval.parquet: ${DATA_DIR}" >&2
    return 2
  }
  [[ -d "${MODEL_DIR}/${QWEN35_9B_SFT_SUBDIR}" ]] || {
    echo "FATAL: merged SFT model is missing: ${MODEL_DIR}/${QWEN35_9B_SFT_SUBDIR}" >&2
    return 2
  }
  (( GLOBAL_BATCH_SIZE % NUM_ITERS_PER_TRAIN_UPDATE == 0 )) || {
    echo "FATAL: GLOBAL_BATCH_SIZE must be divisible by NUM_ITERS_PER_TRAIN_UPDATE" >&2
    return 2
  }
  (( (GLOBAL_BATCH_SIZE / NUM_ITERS_PER_TRAIN_UPDATE) % N_SAMPLES_PER_PROMPT == 0 )) || {
    echo "FATAL: per-update advantage batch must be divisible by N_SAMPLES_PER_PROMPT" >&2
    return 2
  }
  (( MAX_PROMPT_LEN + MAX_RESPONSE_LEN >= MAX_CONTEXT_LEN )) || {
    echo "FATAL: prompt+response budgets do not cover MAX_CONTEXT_LEN" >&2
    return 2
  }
  (( NUM_ROLLOUT > START_ROLLOUT_ID )) || {
    echo "FATAL: NUM_ROLLOUT=${NUM_ROLLOUT} must be greater than START_ROLLOUT_ID=${START_ROLLOUT_ID}" >&2
    return 2
  }
  rl_validate_resume
}

rl_resolve_nodes() {
  local gpu_nodes local_node rollout_node actor_node node
  gpu_nodes=$("${RAY_PYTHON_BIN}" - <<'PY' 2>/dev/null || true
import ray
try:
    ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
    nodes = []
    for item in ray.nodes():
        resources = item.get("Resources") or {}
        if item.get("Alive") and float(resources.get("GPU", 0) or 0) > 0:
            nodes.append(item["NodeManagerAddress"])
    print(" ".join(sorted(set(nodes))))
    ray.shutdown()
except Exception:
    pass
PY
  )
  local_node=$("${RAY_PYTHON_BIN}" - <<'PY' 2>/dev/null || true
import ray
try:
    ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
    print(ray.util.get_node_ip_address())
    ray.shutdown()
except Exception:
    pass
PY
  )
  [[ $(wc -w <<<"${gpu_nodes}") -eq 2 ]] || {
    echo "FATAL: expected exactly two live GPU nodes, got '${gpu_nodes:-none}'" >&2
    return 2
  }

  rollout_node="${RELAX_PIN_NODE_ROLLOUT:-}"
  if [[ -z "${rollout_node}" ]]; then
    for node in ${gpu_nodes}; do
      [[ "${node}" == "${local_node}" ]] && rollout_node="${node}"
    done
    rollout_node="${rollout_node:-${gpu_nodes%% *}}"
  fi
  actor_node="${RELAX_PIN_NODE_ACTOR:-}"
  if [[ -z "${actor_node}" ]]; then
    for node in ${gpu_nodes}; do
      [[ "${node}" != "${rollout_node}" ]] && actor_node="${node}"
    done
  fi
  [[ -n "${actor_node}" && "${actor_node}" != "${rollout_node}" ]] || {
    echo "FATAL: actor and rollout must resolve to distinct GPU nodes" >&2
    return 2
  }
  for node in "${rollout_node}" "${actor_node}"; do
    case " ${gpu_nodes} " in
      *" ${node} "*) ;;
      *) echo "FATAL: pinned node ${node} is not in live GPU nodes: ${gpu_nodes}" >&2; return 2 ;;
    esac
  done

  export RELAX_PIN_NODE_ROLLOUT="${rollout_node}"
  export RELAX_PIN_NODE_ACTOR="${actor_node}"
  export RELAX_PIN_NODE_REFERENCE="${RELAX_PIN_NODE_REFERENCE:-${rollout_node}}"
  export RELAX_PIN_NODE_ACTOR_FWD="${RELAX_PIN_NODE_ACTOR_FWD:-${rollout_node}}"
  [[ "${RELAX_PIN_NODE_REFERENCE}" == "${rollout_node}" && "${RELAX_PIN_NODE_ACTOR_FWD}" == "${rollout_node}" ]] || {
    echo "FATAL: this actor8/rollout4/ref2/fwd2 topology requires reference and actor_fwd on rollout node ${rollout_node}" >&2
    return 2
  }
  export RAY_MASTER_ADDR_OVERRIDE="${rollout_node}"
  export MASTER_ADDR="${rollout_node}"
  export SLIME_HOST_IP="${rollout_node}"
  export RAY_SERVE_CONTROLLER_NODE_RESOURCE="node:${rollout_node}"

  local local_ips ray_ips
  local_ips=$(hostname -I | tr ' ' ',' | sed 's/,,*/,/g;s/,$//')
  ray_ips=$(tr ' ' ',' <<<"${gpu_nodes}")
  export RELAX_NO_PROXY="127.0.0.1,localhost,0.0.0.0,10.0.0.0/8,172.16.0.0/12,mirrors.tuna.tsinghua.edu.cn,pypi.tuna.tsinghua.edu.cn,hf-mirror.com,${local_ips},${ray_ips}"
  export NO_PROXY="${RELAX_NO_PROXY}"
  export no_proxy="${NO_PROXY}"
}

rl_preflight() {
  bash "${ROOT}/ops/launch/start_local_overlay2_docker.sh"
  pgrep -f 'ops/launch/subreaper_exec.py.*dockerd' >/dev/null || {
    echo "FATAL: local dockerd is not running under subreaper_exec.py" >&2
    return 2
  }

  local image_count image cid out err_file
  image_count=$(DOCKER_HOST="${RL_DOCKER_LOCAL_HOST}" docker images -q 2>/dev/null | wc -l)
  (( image_count >= ${RL_MIN_DOCKER_IMAGES:-500} )) || {
    echo "FATAL: local dockerd has ${image_count} images; expected at least ${RL_MIN_DOCKER_IMAGES:-500}" >&2
    return 2
  }
  image=$(DOCKER_HOST="${RL_DOCKER_LOCAL_HOST}" docker images --format '{{.Repository}}:{{.Tag}}' | rg -m1 'unified-seta-synth' || true)
  [[ -n "${image}" ]] || { echo "FATAL: no unified-seta-synth image available for Docker exec smoke" >&2; return 2; }
  err_file=$(mktemp /tmp/rl-docker-exec-smoke.XXXXXX)
  cid=$(DOCKER_HOST="${RL_DOCKER_LOCAL_HOST}" docker run -d --network host --entrypoint sleep "${image}" 120)
  out=$(DOCKER_HOST="${RL_DOCKER_LOCAL_HOST}" docker exec "${cid}" sh -c 'echo D5OUT; echo D5ERR >&2; exit 3' 2>"${err_file}" || echo 'rc=3')
  DOCKER_HOST="${RL_DOCKER_LOCAL_HOST}" docker rm -f "${cid}" >/dev/null 2>&1 || true
  [[ "${out}" == $'D5OUT\nrc=3' && "$(<"${err_file}")" == 'D5ERR' ]] || {
    rm -f "${err_file}"
    echo "FATAL: Docker exec stdout/stderr smoke failed" >&2
    return 2
  }
  rm -f "${err_file}"

  "${RAY_PYTHON_BIN}" - <<'PY'
import os
import time
import ray

nodes = [os.environ["RELAX_PIN_NODE_ROLLOUT"], os.environ["RELAX_PIN_NODE_ACTOR"]]
ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")

def make_probe(ip):
    @ray.remote(num_cpus=0.1, resources={f"node:{ip}": 0.01})
    def probe():
        import socket
        import subprocess
        count = subprocess.check_output(
            ["bash", "-lc", "nvidia-smi --query-gpu=index --format=csv,noheader | wc -l"],
            text=True,
            timeout=10,
        ).strip()
        return {"host": socket.gethostname(), "ip": ray.util.get_node_ip_address(), "gpus": int(count)}
    return probe

probes = {ip: make_probe(ip) for ip in nodes}
for attempt in range(int(os.environ.get("RL_HEALTH_GATE_REPEATS", "4"))):
    total = int(ray.cluster_resources().get("GPU", 0))
    expected = int(os.environ.get("RL_EXPECTED_TOTAL_GPUS", "16"))
    if total != expected:
        raise SystemExit(f"FATAL: Ray reports {total} GPUs, expected {expected}")
    got = ray.get([probes[ip].remote() for ip in nodes], timeout=30)
    if sorted(item["ip"] for item in got) != sorted(nodes):
        raise SystemExit(f"FATAL: health probes landed on {got}, expected {nodes}")
    print(f"HEALTH_GATE_OK {attempt + 1} total_gpus={total} nodes={got}", flush=True)
    time.sleep(5)
ray.shutdown()
PY
  echo "PREFLIGHT OK: profile=${RL_PROFILE} docker_images=${image_count} actor=${RELAX_PIN_NODE_ACTOR} rollout=${RELAX_PIN_NODE_ROLLOUT}"
}

declare -a RL_GUARD_SESSIONS=()

rl_start_guards() {
  mkdir -p "${RUN_DIR}/cleanup" "${RUN_DIR}/docker_lifecycle"
  export AGENT_BENCH_DOCKER_LIFECYCLE_DIR="${RUN_DIR}/docker_lifecycle"
  local guard_id
  guard_id=$(printf '%s' "${RUN_NAME}" | sha256sum | cut -c1-8)

  rl_add_guard "dmesg" \
    "exec stdbuf -oL dmesg -w -T 2>/dev/null | stdbuf -oL grep -iE 'unregister_netdevice|waiting for .* to become free|kobject_uevent' >>'${RUN_DIR}/cleanup/dmesg_unregister_netdevice.log' 2>&1" \
    "${guard_id}"
  rl_add_guard "shim" \
    "cd '${ROOT}' && export DOCKER_HOST='${RL_DOCKER_LOCAL_HOST}' REAP_INTERVAL_SEC=120; exec '${RAY_PYTHON_BIN}' ops/cleanup/reap_orphan_shims.py >>'${RUN_DIR}/cleanup/shim_reaper.log' 2>&1" \
    "${guard_id}"
  rl_add_guard "disk" \
    "cd '${ROOT}' && export DOCKER_HOST='${RL_DOCKER_LOCAL_HOST}' DISK_REAP_PATH='${RL_DOCKER_DISK_PATH}' DISK_REAP_INTERVAL_SEC=5 DISK_REAP_WATERMARK_GB='${DISK_REAP_WATERMARK_GB}'; exec '${RAY_PYTHON_BIN}' ops/cleanup/reap_disk_bombs.py >>'${RUN_DIR}/cleanup/disk_reaper.log' 2>&1" \
    "${guard_id}"
  rl_add_guard "stale" \
    "cd '${ROOT}' && export DOCKER_HOST='${RL_DOCKER_LOCAL_HOST}' RELAX_RL_RUN_ID='${RUN_NAME}'; exec '${RAY_PYTHON_BIN}' ops/cleanup/watch_rl_stale_containers.py --run-id '${RUN_NAME}' --driver-log '${RUN_DIR}/driver.log' --loop --interval-sec '${STALE_CLEANER_INTERVAL_SEC:-60}' --max-remove '${STALE_CLEANER_MAX_REMOVE:-64}' --max-running-remove '${STALE_CLEANER_MAX_RUNNING_REMOVE:-64}' --remove-running-after-sec '${STALE_CLEANER_REMOVE_RUNNING_AFTER_SEC:-120}' --keep-recent-steps '${STALE_CLEANER_KEEP_RECENT_STEPS:-1}' >>'${RUN_DIR}/cleanup/rl_stale_cleaner.log' 2>&1" \
    "${guard_id}"
  rl_add_guard "flight" \
    "while true; do echo \"=== \$(date -Is) load=\$(cut -d' ' -f1-3 /proc/loadavg) memavail=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo)kB dirty=\$(awk '/^Dirty/{print \$2}' /proc/meminfo)kB dstate=\$(ps -eo stat 2>/dev/null | grep -c '^D' || true) containers=\$(DOCKER_HOST='${RL_DOCKER_LOCAL_HOST}' timeout 10 docker ps -q 2>/dev/null | wc -l) dockerdisk=\$(df -BG --output=avail '${RL_DOCKER_DISK_PATH}' 2>/dev/null | tail -1 | tr -d ' ')\" >>'${RUN_DIR}/cleanup/flight_recorder.log'; sleep 30; done" \
    "${guard_id}"
}

rl_add_guard() {
  local kind body guard_id session
  kind="$1"
  body="$2"
  guard_id="$3"
  session="rl-${kind}-${guard_id}"
  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" "${body}"
  RL_GUARD_SESSIONS+=("${session}")
}

rl_stop_guards() {
  local session
  for session in "${RL_GUARD_SESSIONS[@]:-}"; do
    [[ -n "${session}" ]] && tmux kill-session -t "${session}" 2>/dev/null || true
  done
}

RL_RESOLVED_KEYS=(
  RL_PROFILE EXPERIMENT_ID EXPERIMENT_DIR RUN_NAME RUN_DIR CHECKPOINT_DIR
  RL_RUN_PURPOSE RELAX_CONTEXT_DECISION DATA_DIR TRAIN_PARQUET EVAL_PARQUET
  MODEL_DIR QWEN35_9B_SFT_SUBDIR SAVE_DIR LOAD_DIR EXPECTED_LATEST_CKPT START_ROLLOUT_ID
  NUM_ROLLOUT SAVE_INTERVAL MAX_ACTOR_CKPT_TO_KEEP KEEP_BEST_ACTOR_CKPT
  DISABLE_EVAL EVAL_INTERVAL SKIP_EVAL_BEFORE_TRAIN
  ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE NUM_ITERS_PER_TRAIN_UPDATE
  MAX_CONTEXT_LEN MAX_PROMPT_LEN MAX_RESPONSE_LEN ACTOR_MAX_TOKENS_PER_GPU LOG_PROBS_CHUNK_SIZE
  TENSOR_MODEL_PARALLEL_SIZE CONTEXT_PARALLEL_SIZE LEARNING_RATE USE_KL_LOSS KL_LOSS_COEF
  RELAX_RESOURCE_OVERRIDE REF_ACTOR_CONFIG ROLLOUT_NUM_GPUS_PER_ENGINE
  AGENT_BENCH_ACTIVE_ENV_CONCURRENCY AGENT_BENCH_DOCKER_START_CONCURRENCY
  AGENT_BENCH_RETRIEVAL_TOP_N AGENT_BENCH_EXTRA_SKILL_ROOTS
  LOSS_TYPE CUSTOM_LOSS_FUNCTION_PATH DYNAMIC_SAMPLING_FILTER_PATH CUSTOM_REWARD_POST_PROCESS_PATH
  RELAX_DISABLE_TIS RELAX_M1_CLEAN RELAX_PROMPT_ONLY_SHADOW_CLEAN
  RELAX_MIXED_SKILL_BONUS_ENABLED SETA_CONTINUOUS_REWARD ROLLOUT_SHUFFLE OVER_SAMPLING_BATCH_SIZE
  RELAX_MIXED_SEPARATED_ADV_ENABLED RELAX_MIXED_SEPARATED_BEHAVIOR_COEF RELAX_MIXED_SEPARATED_BEHAVIOR_CLIP
  RELAX_SELECTOR_ACTION_CREDIT RELAX_SELECTOR_ACTION_LOSS_COEF
  RELAX_SLATE_REGRET_GRPO RELAX_SLATE_REGRET_COEF RELAX_SLATE_STRATIFIED_ADVANTAGE
  RELAX_SLATE_STRATIFIED_ADV_COEF RELAX_SLATE_STRATIFIED_SHRINKAGE RELAX_SLATE_STRATIFIED_ADV_CLIP
  RELAX_PAIR_ATOMIC_SAMPLING RELAX_PAIR_SPECULATIVE_EXTRA_GROUPS RELAX_PAIR_ORACLE_GRPO
  RELAX_OPSD_MODE RELAX_OPSD_TEACHER_SELF_ROUTER RELAX_SKILL_GROUP_REWARD
  RELAX_SKILL_GROUP_BONUS_COEF RELAX_SKILL_GROUP_BONUS_MAX RELAX_SKILL_GROUP_MARGIN
  RELAX_SKILL_GROUP_SUBGROUP_ADV_COEF RELAX_SKILL_GROUP_REQUIRE_BOTH RELAX_SKILL_GROUP_NO_READ_SUCCESS_BONUS
  RELAX_DYNAMIC_FILTER_MAX_REJECTS_PER_ROLLOUT RELAX_DYNAMIC_FILTER_MAX_REJECT_SAMPLES_PER_ROLLOUT
  RELAX_DYNAMIC_FILTER_MIN_SKILL_READ_FRAC RELAX_DYNAMIC_FILTER_MIN_NO_SKILL_READ_FRAC
  RELAX_DYNAMIC_FILTER_SKILL_READ_MAX_SAMPLES
  RELAX_PIN_NODE_ACTOR RELAX_PIN_NODE_ROLLOUT RELAX_PIN_NODE_REFERENCE RELAX_PIN_NODE_ACTOR_FWD
  RL_DOCKER_LOCAL_HOST LOCAL_DOCKER_DATA_ROOT LOCAL_DOCKER_EXEC_ROOT RL_DOCKER_DISK_PATH
)

rl_dump_resolved_config() {
  local key value
  echo "# resolved by ops/workflows/rl_training/run_rl.sh"
  for key in "${RL_RESOLVED_KEYS[@]}"; do
    value="${!key-}"
    printf 'export %s=%q\n' "${key}" "${value}"
  done
}

rl_init_run_dir() {
  if [[ -e "${RUN_DIR}" ]] && find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null | grep -q .; then
    echo "FATAL: segment already exists and will not be overwritten: ${RUN_DIR}" >&2
    return 2
  fi
  mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}" "${RUN_DIR}/cleanup"
  : >"${RUN_DIR}/driver.log"
  rl_dump_resolved_config >"${RUN_DIR}/resolved_config.env"
}

rl_prepare_relax_entrypoint() {
  cd "${ROOT}/Relax"
  "${PYTHON_BIN}" "${ROOT}/ops/launch/patch_ray_serve_controller_pin.py"
  export MEGATRON="${MEGATRON:-${ROOT}/Relax/deps/Megatron-LM}"
  export PYTHONPATH="${ROOT}/Relax:${MEGATRON}:${ROOT}/GeneralAgent/eval_scripts:${PYTHONPATH:-}"
  # Optional egress proxy for driver-side downloads. Leave unset for direct
  # networking; set http_proxy (and UNIFIED_CONTAINER_PROXY for rollout
  # containers) in your environment when your cluster requires one.
  if [[ -n "${http_proxy:-}" ]]; then
    export https_proxy="${https_proxy:-${http_proxy}}"
    export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
    export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
    export ALL_PROXY="${ALL_PROXY:-${http_proxy}}"
    export all_proxy="${all_proxy:-${ALL_PROXY}}"
  fi
  # shellcheck source=/dev/null
  source scripts/entrypoint/ray-job.sh
  set +x
  cd "${ROOT}"
}

rl_record_launch() {
  env | sort | sed -E '/(^|_)(API_KEY|TOKEN|SECRET|PASSWORD)=/d' >"${RUN_DIR}/launch_env.redacted.txt"
  "${PYTHON_BIN}" "${ROOT}/GeneralAgent/sft_data_collection/run_manifest.py" train \
    --experiment-id "${EXPERIMENT_ID}" --experiment-dir "${EXPERIMENT_DIR}" \
    --segment-id "${RUN_NAME}" --run-dir "${RUN_DIR}"
  RL_MANIFEST_RECORDED=1
  mkdir -p "${ROOT}/experiments/rl/current"
  printf '%s\n' "${RUN_DIR}" >"${ROOT}/experiments/rl/current/latest.txt"
  printf '%s\n' "${RUN_DIR}" >"${ROOT}/experiments/rl/current/${RL_PROFILE}.txt"
}

rl_record_finish() {
  local rc="$1"
  [[ "${RL_MANIFEST_RECORDED:-0}" == 1 && "${RL_MANIFEST_FINALIZED:-0}" != 1 ]] || return 0
  "${PYTHON_BIN}" "${ROOT}/GeneralAgent/sft_data_collection/run_manifest.py" train-finalize \
    --experiment-id "${EXPERIMENT_ID}" --experiment-dir "${EXPERIMENT_DIR}" \
    --segment-id "${RUN_NAME}" --run-dir "${RUN_DIR}" --return-code "${rc}"
  RL_MANIFEST_FINALIZED=1
}

rl_launch_training() {
  unset AGENT_BENCH_DOCKER_EXEC_CONCURRENCY AGENT_BENCH_DOCKER_TEARDOWN_CONCURRENCY \
    AGENT_BENCH_DOCKER_HOSTS UNIFIED_DOCKER_NPROC_LIMIT UNIFIED_DOCKER_MEMORY_LIMIT \
    UNIFIED_DOCKER_RM_TIMEOUT_SEC LD_LIBRARY_PATH 2>/dev/null || true
  cd "${ROOT}/Relax"
  /usr/bin/env bash examples/agent_bench/run_agent_grpo_9B.sh
}
