#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-9B GRPO training on the 5-bench agent campaign.
#
# Resource layout (8 H800 GPUs on a single node, fully-async):
#   actor:      4 GPUs (TP=4)
#   rollout:    2 GPUs (1 engine × 2 GPUs)
#   reference:  1 GPU  (TP=1, weight-only)
#   actor_fwd:  1 GPU
#
# Prerequisites:
#   * conda env `relax` active (run `unset LD_LIBRARY_PATH` first — TE import
#     hangs on networked-storage FUSE if LD_LIBRARY_PATH is polluted; see
#     `memory/feedback_working_style.md` / Phase 0 notes).
#   * Phase-0 deploy_check scripts have already passed.
#   * train.parquet built via `python -m GeneralAgent.rl_data_prep.convert_to_relax_data`.
#   * UNIFIED_LAUNCHER_MODE in {mock, real}: 'mock' for first end-to-end smoke,
#     'real' once Phase B v1.1 launchers are wired up.
#
# Usage:
#   MODEL_DIR=/mnt/.../GeneralAgent/sft_training/merged_models \
#   DATA_DIR=/mnt/.../experiments/rl/v1/parquet \
#   SAVE_DIR=/mnt/.../experiments/rl/v1/checkpoints \
#   UNIFIED_LAUNCHER_MODE=mock \
#       bash examples/agent_bench/run_agent_grpo_9B.sh

set -ex
set -o pipefail

TIMESTAMP=$(date "+%Y-%m-%d-%H:%M:%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RELAX_PYTHON="${RELAX_PYTHON:-/usr/bin/python3}"

# Load project secrets before Ray runtime_env is built.  The script runs with
# xtrace enabled, so source secrets with xtrace temporarily disabled to avoid
# leaking API keys into driver logs.
if [ -f "${PROJECT_ROOT}/secrets/.env.secrets" ]; then
    _RELAX_XTRACE_WAS_ON=0
    case "$-" in
        *x*) _RELAX_XTRACE_WAS_ON=1; set +x ;;
    esac
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/secrets/.env.secrets"
    set +a
    if [ "${_RELAX_XTRACE_WAS_ON}" = "1" ]; then
        set -x
    fi
    unset _RELAX_XTRACE_WAS_ON
fi

# Auto-source local environment (Ray cluster + ports).
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi

# The user's long-lived shell can contain stale values from older single-node
# runs (notably CUDA_HOME=/.../anaconda3 and DOCKER_HOST=tcp://127.0.0.1:2375).
# Restore the validated multi-node values *after* sourcing the entrypoint so the
# driver process and Ray runtime env see the same working setup.
RELAX_FAST_CUDA_HOME="${RELAX_FAST_CUDA_HOME:-${PROJECT_ROOT}/ops/cache/cuda_fast_home}"
if [ -d "${RELAX_FAST_CUDA_HOME}" ]; then
    export CUDA_HOME="${RELAX_FAST_CUDA_HOME}"
    export CUDA_PATH="${RELAX_FAST_CUDA_HOME}"
fi
export CUDNN_HOME="${CUDNN_HOME:-/usr/local/lib/python3.12/dist-packages/nvidia/cudnn}"
if [ -n "${RELAX_DOCKER_HOST:-}" ]; then
    export DOCKER_HOST="${RELAX_DOCKER_HOST}"
elif [ -z "${DOCKER_HOST:-}" ] || [ "${DOCKER_HOST}" = "tcp://127.0.0.1:2375" ]; then
    export DOCKER_HOST="${RL_DOCKER_LOCAL_HOST:-unix:///tmp/local-docker-overlay2.sock}"
fi
RELAX_NO_PROXY_DEFAULT="127.0.0.1,localhost,0.0.0.0,10.0.0.0/8,172.16.0.0/12,mirrors.tuna.tsinghua.edu.cn,pypi.tuna.tsinghua.edu.cn,hf-mirror.com"
export NO_PROXY="${RELAX_NO_PROXY:-${NO_PROXY:-${RELAX_NO_PROXY_DEFAULT}}}"
case ",${NO_PROXY}," in
    *",10.0.0.0/8,"*) ;;
    *) export NO_PROXY="${NO_PROXY},10.0.0.0/8,172.16.0.0/12,mirrors.tuna.tsinghua.edu.cn,pypi.tuna.tsinghua.edu.cn,hf-mirror.com" ;;
esac
export no_proxy="${NO_PROXY}"

source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"

###############################################################################
#                                  PATHS                                      #
###############################################################################

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/agent_bench}"
EXP_NAME="${EXP_NAME:-qwen35-9B-agent-bench-${UNIFIED_LAUNCHER_MODE:-mock}-${TIMESTAMP}}"

if [ -z "${MODEL_DIR:-}" ] || [ -z "${DATA_DIR:-}" ] || [ -z "${SAVE_DIR:-}" ]; then
    echo "ERROR: MODEL_DIR, DATA_DIR, and SAVE_DIR must be set." >&2
    exit 1
fi
mkdir -p "${SAVE_DIR}"

QWEN35_9B_SFT_SUBDIR="${QWEN35_9B_SFT_SUBDIR:-qwen35_9b_sft_campaign_20260512_clean_plus_claw_thinkwrap_4gpu_49k_5epoch_r32_liger}"
SFT_MODEL_DIR="${MODEL_DIR}/${QWEN35_9B_SFT_SUBDIR}"

if [ ! -d "${SFT_MODEL_DIR}" ]; then
    echo "ERROR: SFT merged model not found: ${SFT_MODEL_DIR}" >&2
    exit 1
fi

# RL parquet
TRAIN_PARQUET="${TRAIN_PARQUET:-${DATA_DIR}/train.parquet}"
if [ ! -f "${TRAIN_PARQUET}" ]; then
    echo "ERROR: train.parquet not found: ${TRAIN_PARQUET}" >&2
    exit 1
fi
# Prefer eval.parquet if it exists (v2 mixed-bench has it). Fall back to
# train.parquet slice for legacy claw-only configs that lack eval.parquet.
if [ -f "${DATA_DIR}/eval.parquet" ]; then
    TEST_PARQUET="${DATA_DIR}/eval.parquet"
    TEST_SLICE="${TEST_SLICE:-[0:70]}"
else
    TEST_PARQUET="${DATA_DIR}/train.parquet"
    TEST_SLICE="${TEST_SLICE:-[0:30]}"
fi

BEST_ACTOR_CKPT_EVAL_FINGERPRINT="${BEST_ACTOR_CKPT_EVAL_FINGERPRINT:-}"
if [ "${KEEP_BEST_ACTOR_CKPT:-0}" = "1" ] && [ "${DISABLE_EVAL:-0}" != "1" ] \
    && [ -z "${BEST_ACTOR_CKPT_EVAL_FINGERPRINT}" ]; then
    BEST_ACTOR_CKPT_EVAL_FINGERPRINT=$(
        {
            printf 'dataset=agent_eval\npath=%s\nslice=%s\nreward_key=score\n' \
                "$(readlink -f "${TEST_PARQUET}")" "${TEST_SLICE}"
            printf 'n_samples=%s\nmax_prompt=%s\nmax_response=%s\ntop_p=0.7\n' \
                "${N_SAMPLES_PER_EVAL_PROMPT:-1}" "${MAX_PROMPT_LEN:-24576}" "${MAX_RESPONSE_LEN:-16384}"
            printf 'skill_roots=%s\n' "${AGENT_BENCH_EXTRA_SKILL_ROOTS:-}"
            sha256sum "${TEST_PARQUET}"
            if [ -f "${DATA_DIR}/build_report.json" ]; then
                sha256sum "${DATA_DIR}/build_report.json"
            fi
        } | sha256sum | awk '{print $1}'
    )
fi

###############################################################################
#                                  MODEL                                      #
###############################################################################

CKPT_ARGS=(
    --hf-checkpoint "${SFT_MODEL_DIR}"
    --ref-load      "${SFT_MODEL_DIR}"
    --save          "${SAVE_DIR}/${EXP_NAME}"
    --megatron-to-hf-mode bridge
    --save-interval "${SAVE_INTERVAL:-10}"
    --max-actor-ckpt-to-keep "${MAX_ACTOR_CKPT_TO_KEEP:-1}"
)
if [ "${KEEP_BEST_ACTOR_CKPT:-0}" = "1" ] && [ "${DISABLE_EVAL:-0}" != "1" ]; then
    CKPT_ARGS+=(
        --keep-best-actor-ckpt
        --best-actor-ckpt-eval-dataset "${BEST_ACTOR_CKPT_EVAL_DATASET:-agent_eval}"
        --best-actor-ckpt-eval-fingerprint "${BEST_ACTOR_CKPT_EVAL_FINGERPRINT}"
    )
fi
if [ -n "${LOAD_DIR:-}" ]; then
    if [ ! -d "${LOAD_DIR}" ]; then
        echo "ERROR: LOAD_DIR not found: ${LOAD_DIR}" >&2
        exit 1
    fi
    CKPT_ARGS+=(--load "${LOAD_DIR}")
fi

###############################################################################
#                                 DATASETS                                    #
###############################################################################

TRAIN_FILES=("${TRAIN_PARQUET}")
TEST_FILES=("${TEST_PARQUET}@${TEST_SLICE}")
PROMPT_SET="['$(IFS=,; echo "${TRAIN_FILES[*]}")']"

###############################################################################
#                                 ROLLOUT                                     #
###############################################################################

NUM_ROLLOUT="${NUM_ROLLOUT:=100}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
NUM_ITERS_PER_TRAIN_UPDATE="${NUM_ITERS_PER_TRAIN_UPDATE:-4}"

if (( GLOBAL_BATCH_SIZE % NUM_ITERS_PER_TRAIN_UPDATE != 0 )); then
    echo "[run_agent_grpo_9B] ERROR: GLOBAL_BATCH_SIZE (${GLOBAL_BATCH_SIZE}) must be divisible by NUM_ITERS_PER_TRAIN_UPDATE (${NUM_ITERS_PER_TRAIN_UPDATE})." >&2
    exit 2
fi
ADVANTAGE_BATCH_SIZE=$(( GLOBAL_BATCH_SIZE / NUM_ITERS_PER_TRAIN_UPDATE ))
if (( ADVANTAGE_BATCH_SIZE % N_SAMPLES_PER_PROMPT != 0 )); then
    echo "[run_agent_grpo_9B] ERROR: GLOBAL_BATCH_SIZE / NUM_ITERS_PER_TRAIN_UPDATE = ${ADVANTAGE_BATCH_SIZE} must be a multiple of N_SAMPLES_PER_PROMPT (${N_SAMPLES_PER_PROMPT})." >&2
    echo "[run_agent_grpo_9B] Hint: for the default GLOBAL_BATCH_SIZE=32 and N_SAMPLES_PER_PROMPT=8, use NUM_ITERS_PER_TRAIN_UPDATE=4." >&2
    exit 2
fi

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}"
    --input-key prompt                # parquet column: list[dict] messages
    --label-key reward_model          # parquet column: dict (task_id, bench, ground_truth)
    --metadata-key extra_info         # parquet column: dict — passed to env via sample.metadata
    --reward-key score                # reward_func returns dict; advantage uses the "score" key
    --apply-chat-template
    # IMPORTANT (P0.3): never pass `tools=` here, otherwise chat_template will
    # double-inject the OpenClaw schema on top of what the SFT system message
    # already contains. We never set `--apply-chat-template-kwargs '{"tools":...}'`.
    # IMPORTANT (post-mortem dispatches=0): SFT data was generated with empty
    # `<think>\n\n</think>\n\n` thinking blocks (UNIFIED_DISABLE_THINKING=1).
    # The Qwen3.5 chat_template defaults to opening `<think>\n` for
    # generation, which confuses the SFT model (it never learned to emit
    # non-empty thinking). Pass enable_thinking=false so the template emits
    # the empty closed-think prefix the model expects.
    --apply-chat-template-kwargs '{"enable_thinking": false}'
    --custom-generate-function-path examples.agent_bench.rollout.generate
    --custom-rm-path                examples.agent_bench.reward_agent_bench.reward_func
    --custom-config-path            examples/agent_bench/agent_bench_config.yaml
    --rollout-interaction-env-path  examples.agent_bench.env_agent_bench
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"             # tasks per rollout batch
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"         # GRPO group size (8 → richer reward variance)
    # Schema-injected SFT prompts measured at 17-18K tokens (max 18033 from
    # tokenizer check). Defaults below cover 17K prompt + 16K response = ~33K
    # context. Override via env for memory-budget experiments.
    --rollout-max-response-len ${MAX_RESPONSE_LEN:-16384}
    --rollout-max-prompt-len ${MAX_PROMPT_LEN:-24576}
    --rollout-max-context-len ${MAX_CONTEXT_LEN:-40960}
    --rollout-temperature 1.0
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --num-rollout "${NUM_ROLLOUT}"
    --use-fault-tolerance
    --use-streaming-dataset
)
if [[ "${ROLLOUT_SHUFFLE:-1}" != "0" ]]; then
    ROLLOUT_ARGS+=(--rollout-shuffle)
fi
if [[ -n "${DYNAMIC_SAMPLING_FILTER_PATH:-}" ]]; then
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}")
fi
if [[ -n "${OVER_SAMPLING_BATCH_SIZE:-}" ]]; then
    ROLLOUT_ARGS+=(--over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}")
fi
if [[ -n "${CUSTOM_REWARD_POST_PROCESS_PATH:-}" ]]; then
    ROLLOUT_ARGS+=(--custom-reward-post-process-path "${CUSTOM_REWARD_POST_PROCESS_PATH}")
fi

if [ -n "${START_ROLLOUT_ID:-}" ]; then
    ROLLOUT_ARGS+=(--start-rollout-id "${START_ROLLOUT_ID}")
fi

# Rollout-engine health-check tolerance (opt-in via env; unset => argparse defaults 30s/2).
# Hardened on crash-prone pods: a transient control-plane wedge (D-state storm from a
# container-teardown burst) can starve the SGLang engine's /health_generate for ~3min,
# tripping the default 2x30s threshold => _kill_engine => engine None => driver exit,
# even though the node self-recovers ~1min later. Raising timeout (precedent: glm5-744B
# recipe uses 120) + consecutive-failure budget lets the run ride out the transient.
# Training-neutral: only changes fault-tolerance sensitivity, not gradients/rollout content.
if [ -n "${ROLLOUT_HEALTH_CHECK_TIMEOUT:-}" ]; then
    ROLLOUT_ARGS+=(--rollout-health-check-timeout "${ROLLOUT_HEALTH_CHECK_TIMEOUT}")
fi
if [ -n "${ROLLOUT_HEALTH_CHECK_MAX_FAILS:-}" ]; then
    ROLLOUT_ARGS+=(--rollout-health-check-max-consecutive-failures "${ROLLOUT_HEALTH_CHECK_MAX_FAILS}")
fi
if [ -n "${ROLLOUT_HEALTH_CHECK_INTERVAL:-}" ]; then
    ROLLOUT_ARGS+=(--rollout-health-check-interval "${ROLLOUT_HEALTH_CHECK_INTERVAL}")
fi

###############################################################################
#                                 EVAL                                        #
###############################################################################

EVAL_ARGS=(
    --eval-interval ${EVAL_INTERVAL:-25}
    --eval-prompt-data agent_eval "${TEST_FILES[@]}"
    # n_samples_per_eval_prompt=1 to mirror native eval scale (1 sample/task
    # like the 20260514 base9b retrieval eval that we benchmark against).
    --n-samples-per-eval-prompt ${N_SAMPLES_PER_EVAL_PROMPT:-1}
    # Eval prompts are the same 17-18K-token agent prompts as training.
    --eval-max-prompt-len ${MAX_PROMPT_LEN:-24576}
    --eval-max-response-len ${MAX_RESPONSE_LEN:-16384}
    --eval-top-p 0.7
)
# Opt-in skip of the step-0 eval. We already have run19's step-0 baseline
# (12.3% pass / per-bench breakdown saved); re-running it just delays the
# first training step.
if [[ "${SKIP_EVAL_BEFORE_TRAIN:-0}" == "1" ]]; then
    EVAL_ARGS+=(--skip-eval-before-train)
fi
if [[ "${DISABLE_EVAL:-0}" == "1" ]]; then
    EVAL_ARGS=()
fi

DEBUG_ARGS=()
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA:-}" ]]; then
    DEBUG_ARGS+=(--load-debug-rollout-data "${LOAD_DEBUG_ROLLOUT_DATA}")
fi
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE:-}" ]]; then
    DEBUG_ARGS+=(--load-debug-rollout-data-subsample "${LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE}")
fi
if [[ -n "${DUMP_DETAILS:-}" ]]; then
    DEBUG_ARGS+=(--dump-details "${DUMP_DETAILS}")
fi
if [[ "${DISABLE_COMPUTE_ADVANTAGES_AND_RETURNS:-0}" == "1" ]]; then
    DEBUG_ARGS+=(--disable-compute-advantages-and-returns)
fi
if [[ "${RELAX_SKIP_HF_VALIDATE:-0}" == "1" ]]; then
    DEBUG_ARGS+=(--skip-hf-validate)
fi

###############################################################################
#                              ALGORITHM (GRPO)                               #
###############################################################################

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef "${KL_LOSS_COEF:-0}"
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
    --eps-clip-c 3
)
# TIS (truncated importance sampling) needs per-token rollout_log_probs aligned
# to the trained tokens. The M1 skill-free clean rebuilds the token sequence
# (synthetic, no true rollout logprobs), so TIS must be OFF there. Default ON to
# preserve every existing run; set RELAX_DISABLE_TIS=1 (M1) to drop it.
if [[ "${RELAX_DISABLE_TIS:-0}" != "1" ]]; then
    GRPO_ARGS+=(--use-tis)
fi
if [[ "${USE_KL_LOSS:-0}" == "1" ]]; then
    GRPO_ARGS+=(--use-kl-loss)
fi

LOSS_ARGS=()
if [[ -n "${LOSS_TYPE:-}" ]]; then
    LOSS_ARGS+=(--loss-type "${LOSS_TYPE}")
fi
if [[ -n "${CUSTOM_LOSS_FUNCTION_PATH:-}" ]]; then
    LOSS_ARGS+=(--custom-loss-function-path "${CUSTOM_LOSS_FUNCTION_PATH}")
fi

###############################################################################
#                                OPTIMIZER                                    #
###############################################################################

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LEARNING_RATE:-1e-6}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)
if [[ "${OVERRIDE_OPT_PARAM_SCHEDULER:-0}" == "1" ]]; then
    OPTIMIZER_ARGS+=(--override-opt-param-scheduler)
fi

###############################################################################
#                                 SGLANG                                      #
###############################################################################

SGLANG_ARGS=(
    # Match the rollout GPU allocation in --resource (rollout: [1, N]).
    --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}
    --sglang-mem-fraction-static 0.6
    # NOTE: Tried --sglang-config sglang_config_textonly.yaml to force
    # Qwen3_5ForCausalLM/Qwen3NextForCausalLM via json_model_override_args
    # — both failed at scheduler init (Qwen3_5ForCausalLM not a SGLang
    # EntryClass; Qwen3NextForCausalLM expects top-level num_hidden_layers
    # but our config nests it under text_config). The VL class loads but
    # produces broken inference. See _status_20260517_overnight/STATUS.md.
)

###############################################################################
#                              MEGATRON CONFIG                                #
###############################################################################

MEGATRON_ARGS=(
    --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE:-4}"
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE:-1}"
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    # Per-GPU token cap (TP=4 SP=true): one full sample is ~32768 / 4 = 8192/GPU.
    # Bump to 12288 to give the dynamic batcher headroom for 1+ sample.
    --max-tokens-per-gpu "${ACTOR_MAX_TOKENS_PER_GPU:-12288}"
    --no-rope-fusion
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)
if [[ "${CALCULATE_PER_TOKEN_LOSS:-0}" == "1" ]]; then
    MEGATRON_ARGS+=(--calculate-per-token-loss)
fi
if [[ -n "${LOG_PROBS_CHUNK_SIZE:-}" ]]; then
    MEGATRON_ARGS+=(--log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}")
fi
if [[ "${OVERLAP_GRAD_REDUCE:-0}" == "1" ]]; then
    MEGATRON_ARGS+=(--overlap-grad-reduce)
fi
if [[ "${GRAD_REDUCE_IN_BF16:-0}" == "1" ]]; then
    MEGATRON_ARGS+=(--grad-reduce-in-bf16)
fi
if [[ -n "${DDP_BUCKET_SIZE:-}" ]]; then
    MEGATRON_ARGS+=(--ddp-bucket-size "${DDP_BUCKET_SIZE}")
fi
if [[ -n "${UPDATE_WEIGHT_BUFFER_SIZE:-}" ]]; then
    MEGATRON_ARGS+=(--update-weight-buffer-size "${UPDATE_WEIGHT_BUFFER_SIZE}")
fi

###############################################################################
#                                LOGGING                                      #
###############################################################################

LOG_ARGS=(
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}"
    # wandb: read key from $WANDB_API_KEY env var (user already logged in)
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-relax-rl-agent}"
    --wandb-exp-name "${EXP_NAME}"
    # Direct task-capability curves: actor-side pass@n and rollout-side score
    # metrics are logged every step for W&B monitoring.
    --log-passrate
    --pass-reward-threshold "${PASS_REWARD_THRESHOLD:-1.0}"
)
if [[ "${DISABLE_METRICS_SERVICE:-0}" != "1" ]]; then
    LOG_ARGS=(--use-metrics-service "${LOG_ARGS[@]}")
fi
if [[ "${USE_CLEARML:-0}" == "1" ]]; then
    LOG_ARGS=(--use-clearml "${LOG_ARGS[@]}")
fi

###############################################################################
#                             RESOURCE (8 GPU)                                #
###############################################################################

# Layout (8 H800 single-node default): actor TP=4 / rollout TP=2 / reference TP=1 / actor_fwd TP=1.
# Multi-node long-context runs can override both RELAX_RESOURCE_OVERRIDE and
# REF_ACTOR_CONFIG. ref/actor_fwd share the same --ref-actor-config (Relax
# applies it to BOTH services in process_args), so tensor_model_parallel_size
# must be <= the GPU count assigned to each of reference and actor_fwd.
# Override via RELAX_RESOURCE_OVERRIDE for multi-node (e.g. rollout: [2, 2]).
RELAX_RESOURCE_DEFAULT='{"actor": [1, 4], "rollout": [1, 2], "reference": [1, 1], "actor_fwd": [1, 1], "advantages": [1, 0]}'
if [ -z "${REF_ACTOR_CONFIG:-}" ]; then
    REF_ACTOR_CONFIG='{"tensor_model_parallel_size": 1, "max_tokens_per_gpu": 4096, "sequence_parallel": false, "only_load_weight": true}'
fi
RAY_RESOURCE_ARGS=(
    --resource "${RELAX_RESOURCE_OVERRIDE:-${RELAX_RESOURCE_DEFAULT}}"
    --max-staleness 2
    --num-data-storage-units 1
    --num-iters-per-train-update "${NUM_ITERS_PER_TRAIN_UPDATE}"
    # max_tokens_per_gpu=4096 forces dynamic batching to chop 17K samples
    # into ~5 microbatches of 4K tokens each. 4K × vocab 248K = ~2GB logits
    # which fits comfortably on H800. Original 16384 OOM'd at run5 step 0.
    --ref-actor-config "${REF_ACTOR_CONFIG}"
)
if [[ "${RELAX_DISABLE_FULLY_ASYNC:-0}" != "1" ]]; then
    RAY_RESOURCE_ARGS+=(--fully-async)
fi
if [[ "${RELAX_ENABLE_COLOCATE:-0}" == "1" ]]; then
    RAY_RESOURCE_ARGS+=(--colocate)
fi
if [[ "${RELAX_DISABLE_ROLLOUT_OFFLOAD:-0}" == "1" ]]; then
    RAY_RESOURCE_ARGS+=(--no-offload-rollout)
fi
if [[ "${DISABLE_HEALTH_CHECK:-0}" != "1" ]]; then
    RAY_RESOURCE_ARGS+=(--use-health-check)
fi

###############################################################################
#                            ENV VAR HYGIENE                                  #
###############################################################################

# TE import hangs on FUSE readdir if LD_LIBRARY_PATH is polluted; clearing it
# here is cheap insurance even if the operator already unset it.
unset LD_LIBRARY_PATH

# PyTorch allocator: use expandable segments to mitigate fragmentation. The
# OOM error message in run5/run6 explicitly suggested this. Without it, peak
# alloc on ref/actor_fwd saw ~65GB allocated + 3GB reserved+unallocated which
# blocks new 14GB allocations on 80GB H800.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[run_agent_grpo_9B] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
echo "[run_agent_grpo_9B] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"

# Default to mock launcher unless caller explicitly opted in to real sandboxes.
export UNIFIED_LAUNCHER_MODE="${UNIFIED_LAUNCHER_MODE:-mock}"
echo "[run_agent_grpo_9B] UNIFIED_LAUNCHER_MODE=${UNIFIED_LAUNCHER_MODE}"

###############################################################################
#                                LAUNCH                                       #
###############################################################################

RUN_AGENT_GRPO_LOG_DIR="${RUN_AGENT_GRPO_LOG_DIR:-logs}"
mkdir -p "${RUN_AGENT_GRPO_LOG_DIR}"
# Driver-mode launch: connect train.py directly to the running cluster
# via RAY_ADDRESS=auto (resolves to GCS port from /tmp/ray/ray_current_cluster).
# Bypasses ray job submit + dashboard JobHead which has been hanging on POST.
export RAY_ADDRESS="auto"
"${RELAX_PYTHON}" -m relax.entrypoints.train \
    "${RAY_RESOURCE_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${LOSS_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${LOG_ARGS[@]}" \
    "${MEGATRON_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${DEBUG_ARGS[@]}" \
    2>&1 | tee "${RUN_AGENT_GRPO_LOG_DIR}/${EXP_NAME}.log"
