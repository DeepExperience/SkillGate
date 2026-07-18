#!/usr/bin/env bash
# NOTE: Migrated canonical workflow copy. Source: ops/launch/run_claw_collect_to_27b_sft_eval_chain.sh
# Original historical script is archived during workflow cleanup; maintain this copy going forward.
# Wait for the clean Claw supplement collection, merge it with the current clean
# campaign data, train a new 27B LoRA, export it, and run OpenClaw-aligned
# retrieval evals. This is intentionally a plain bash chain instead of a Codex
# goal so that failures are explicit and resumable from logged paths.
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SKILLRL_ROOT:-$(pwd)}}"
cd "${PROJECT_ROOT}"

DATE="${DATE:-$(date -u +%Y%m%d)}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"

PY="${PY:-${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/envs/slime/bin/python3}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${SKILLRL_ROOT:-$(pwd)}/models/Qwen3.5-27B}"
BASE_SERVED="${BASE_SERVED:-qwen3.5-27b}"
SFT_SERVED="${SFT_SERVED:-qwen3.5-27b-sft-${DATE}-clean-plus-claw}"
BASE_PORT="${BASE_PORT:-30000}"
SFT_PORT="${SFT_PORT:-30001}"
TP_SIZE="${TP_SIZE:-4}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
RANDOM_SEED="${RANDOM_SEED:-1063810697}"

CLAW_RUN_ID="${CLAW_RUN_ID:-20260510_claw_phase1_qwen27b_openclaw_docker_clean_supplement}"
CLAW_RUN_ROOT="${CLAW_RUN_ROOT:-experiments/20260510/${CLAW_RUN_ID}}"
CLAW_PLAN="${CLAW_PLAN:-${CLAW_RUN_ROOT}/plans/${CLAW_RUN_ID}.missing_clean_claw.jsonl}"
CLAW_TMUX="${CLAW_TMUX:-claw-parallel-docker-v3}"
CLAW_LAUNCH_LOG="${CLAW_LAUNCH_LOG:-/tmp/claw_parallel_docker_v3_154124.log}"
CLAW_SAFE_CUTOFF_UTC="${CLAW_SAFE_CUTOFF_UTC:-2026-05-11T15:33:00+00:00}"

OLD_CLEAN_LF="${OLD_CLEAN_LF:-GeneralAgent/sft_training/llamafactory_data/20260509_sft_campaign_clean_thinkwrap/agent_sft_campaign_20260509_clean_thinkwrap.json}"
OLD_TRAIN_CFG="${OLD_TRAIN_CFG:-GeneralAgent/sft_training/configs/qwen35_27b_lora_campaign_20260509_clean_8gpu_64k_5epoch_r32_liger.yaml}"
OLD_EXPORT_CFG="${OLD_EXPORT_CFG:-GeneralAgent/sft_training/configs/qwen35_27b_lora_campaign_20260509_clean_export.yaml}"

CHAIN_ID="${CHAIN_ID:-${DATE}_27b_clean_plus_claw}"
CHAIN_ROOT="${CHAIN_ROOT:-experiments/${DATE}/${CHAIN_ID}}"
REPORT_DIR="${CHAIN_ROOT}/reports"
DATA_ROOT="${CHAIN_ROOT}/sft_build"
LOG_DIR="${CHAIN_ROOT}/logs"
mkdir -p "${REPORT_DIR}" "${DATA_ROOT}" "${LOG_DIR}"
CHAIN_LOG="${CHAIN_LOG:-${LOG_DIR}/${CHAIN_ID}_${STAMP}.log}"
LIVE_LOG="${REPORT_DIR}/live.log"
LOCAL_HF_CACHE="${LOCAL_HF_CACHE:-/tmp/${USER:-root}_${CHAIN_ID}_hf_cache}"

NEW_DATASET_NAME="${NEW_DATASET_NAME:-agent_sft_campaign_${DATE}_clean_plus_claw_thinkwrap}"
LF_OUT_DIR="${LF_OUT_DIR:-GeneralAgent/sft_training/llamafactory_data/${DATE}_sft_campaign_clean_plus_claw_thinkwrap}"
TRAIN_CFG="${TRAIN_CFG:-GeneralAgent/sft_training/configs/qwen35_27b_lora_campaign_${DATE}_clean_plus_claw_8gpu_49k_5epoch_r32_liger.yaml}"
EXPORT_CFG="${EXPORT_CFG:-GeneralAgent/sft_training/configs/qwen35_27b_lora_campaign_${DATE}_clean_plus_claw_export.yaml}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-${PROJECT_ROOT}/GeneralAgent/sft_training/outputs/qwen35_27b_lora_campaign_${DATE}_clean_plus_claw_8gpu_49k_5epoch_r32_liger}"
MERGED="${MERGED:-${PROJECT_ROOT}/GeneralAgent/sft_training/merged_models/qwen35_27b_sft_campaign_${DATE}_clean_plus_claw_8gpu_49k_5epoch_r32_liger}"
TRAIN_TMUX="${TRAIN_TMUX:-sft-27b-clean-plus-claw-${DATE}}"
TRAIN_STDOUT_LOG="${TRAIN_STDOUT_LOG:-/tmp/${TRAIN_TMUX}.log}"
TRAINER_LOG="${TRAINER_LOG:-${TRAIN_OUTPUT}/trainer_log.jsonl}"

POLL_SEC="${POLL_SEC:-300}"
QUICK_WORKERS="${QUICK_WORKERS:-6}"
QUICK_TIMEOUT="${QUICK_TIMEOUT:-2400}"
FULL_BENCHES="${FULL_BENCHES:-seta swe claw sb_ns tb2}"
FULL_PAR_seta="${FULL_PAR_seta:-2}"
FULL_PAR_swe="${FULL_PAR_swe:-1}"
FULL_PAR_claw="${FULL_PAR_claw:-2}"
FULL_PAR_sb_ns="${FULL_PAR_sb_ns:-2}"
FULL_PAR_tb2="${FULL_PAR_tb2:-2}"
RETRIEVAL_DATE="${RETRIEVAL_DATE:-20260424}"
RETR_SUFFIX="${RETR_SUFFIX:-v7pipeline_on_2046lib}"

log() {
  local msg="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
  echo "${msg}" | tee -a "${CHAIN_LOG}" "${LIVE_LOG}" >/dev/null
}

fail() {
  log "FATAL: $*"
  exit 1
}

served_model() {
  local port="$1"
  curl -sS --max-time 3 --noproxy '*' "http://127.0.0.1:${port}/v1/models" 2>/dev/null \
    | "${PY}" -c 'import json,sys; print((json.load(sys.stdin).get("data") or [{}])[0].get("id",""))' \
      2>/dev/null || true
}

start_base_27b_if_needed() {
  local current
  current="$(served_model "${BASE_PORT}")"
  if [[ "${current}" == "${BASE_SERVED}" ]]; then
    log "base 27B already serving on :${BASE_PORT}"
    return
  fi
  local session="${BASE_TMUX:-sglang-qwen27b-base-chain-0to3}"
  local slog="${LOG_DIR}/${session}_${STAMP}.log"
  if tmux has-session -t "${session}" 2>/dev/null; then
    tmux kill-session -t "${session}" || true
    sleep 3
  fi
  log "starting base 27B for hindsight augmentation: port=${BASE_PORT}"
  tmux new-session -d -s "${session}" \
    "cd '${PROJECT_ROOT}' && CUDA_VISIBLE_DEVICES='0,1,2,3' bash -lc 'source ${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/etc/profile.d/conda.sh && conda activate slime && export NO_PROXY=\"127.0.0.1,localhost,0.0.0.0\" no_proxy=\"127.0.0.1,localhost,0.0.0.0\" && export SGLANG_DISABLE_CUDNN_CHECK=1 CUDA_HOME=\"\$CONDA_PREFIX\" && exec python -m sglang.launch_server --model-path \"${BASE_MODEL_PATH}\" --served-model-name \"${BASE_SERVED}\" --tensor-parallel-size \"${TP_SIZE}\" --host 0.0.0.0 --port \"${BASE_PORT}\" --context-length \"${CONTEXT_LENGTH}\" --mem-fraction-static \"${MEM_FRACTION}\" --random-seed \"${RANDOM_SEED}\" --enable-metrics --log-level info' 2>&1 | tee '${slog}'"
  local deadline=$((SECONDS + 2400))
  while (( SECONDS < deadline )); do
    current="$(served_model "${BASE_PORT}")"
    [[ "${current}" == "${BASE_SERVED}" ]] && { log "base 27B ready"; return; }
    tmux has-session -t "${session}" 2>/dev/null || { tail -120 "${slog}" || true; fail "base 27B SGLang died"; }
    sleep 20
  done
  tail -120 "${slog}" || true
  fail "timeout waiting for base 27B"
}

stop_sglang_before_training() {
  log "stopping known SGLang sessions before 8GPU training"
  for session in \
    sglang-qwen27b-base-chain-0to3 \
    sglang-qwen27b-base-goal-0to3 \
    sglang-qwen27b-base-eval-0to3 \
    sglang-qwen27b-sft-clean-goal-4to7 \
    sglang-qwen27b-sft-clean-plus-claw-0to3 \
    sglang-qwen27b-sft-clean-plus-claw-4to7; do
    tmux kill-session -t "${session}" 2>/dev/null || true
  done
  sleep 5
}

wait_for_claw_collection() {
  [[ -f "${CLAW_PLAN}" ]] || fail "missing claw plan: ${CLAW_PLAN}"
  : > "${LIVE_LOG}"
  log "chain start; waiting for Claw collection run=${CLAW_RUN_ID}"
  local last=0
  while true; do
    local proc_count tmux_live
    proc_count="$(pgrep -fc "launch_claw_trials_parallel.py.*${CLAW_RUN_ID}" || true)"
    if tmux has-session -t "${CLAW_TMUX}" 2>/dev/null; then tmux_live=1; else tmux_live=0; fi
    if [[ "${proc_count}" == "0" && "${tmux_live}" == "0" ]]; then
      log "Claw collection launcher is no longer running"
      break
    fi
    if (( SECONDS - last >= POLL_SEC )); then
      last="${SECONDS}"
      local done_line tail_line
      done_line="$(grep -E 'done=[0-9]+/[0-9]+' "${CLAW_LAUNCH_LOG}" 2>/dev/null | tail -1 || true)"
      tail_line="$(tail -1 "${CLAW_LAUNCH_LOG}" 2>/dev/null || true)"
      log "Claw collection still running: tmux=${tmux_live} proc=${proc_count} progress='${done_line:-${tail_line}}'"
    fi
    sleep 30
  done
}

audit_claw_collection() {
  log "auditing Claw docker-sandbox traces after cutoff ${CLAW_SAFE_CUTOFF_UTC}"
  "${PY}" - "${CLAW_RUN_ROOT}" "${CLAW_SAFE_CUTOFF_UTC}" "${REPORT_DIR}/claw_docker_audit_${STAMP}.json" <<'PY'
import datetime as dt
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
cutoff = dt.datetime.fromisoformat(sys.argv[2].replace("Z", "+00:00")).timestamp()
out = pathlib.Path(sys.argv[3])
total = host = docker = recent = 0
bad_recent = []
host_markers = ["/tmp/claw_pilot", "http://localhost:91", "localhost:9100", "localhost:9200"]
docker_markers = ["/workspace", "host.docker.internal"]
for path in root.glob("results/claw/**/trajectories/*.json"):
    total += 1
    text = path.read_text(encoding="utf-8", errors="ignore")
    is_host = any(m in text for m in host_markers)
    is_docker = any(m in text for m in docker_markers)
    host += int(is_host)
    docker += int(is_docker)
    if path.stat().st_mtime >= cutoff:
        recent += 1
        if is_host:
            bad_recent.append(str(path))
payload = {
    "root": str(root),
    "cutoff": sys.argv[2],
    "total_trajectories": total,
    "host_like_total": host,
    "docker_like_total": docker,
    "recent_trajectories": recent,
    "recent_host_like": bad_recent,
    "passed": not bad_recent and recent > 0,
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
if not payload["passed"]:
    raise SystemExit(2)
PY
}

collect_and_process_new_claw() {
  local collected="${DATA_ROOT}/new_claw_collected"
  local augmented="${DATA_ROOT}/new_claw_augmented/sft_messages.jsonl"
  local filtered="${DATA_ROOT}/new_claw_filtered/sft_messages.jsonl"
  local thinkwrap="${DATA_ROOT}/new_claw_thinkwrap/sft_messages.jsonl"
  local lf_new="${DATA_ROOT}/new_claw_lf"
  local lf_new_name="agent_${DATE}_new_claw_thinkwrap"

  log "collecting Claw successes from ${CLAW_PLAN}"
  "${PY}" GeneralAgent/sft_data_collection/collect_successes.py \
    --plan "${CLAW_PLAN}" \
    --out-dir "${collected}" \
    --max-successes-per-task "${MAX_SUCCESSES_PER_TASK:-2}" \
    --max-successes-per-use-skill-task "${MAX_SUCCESSES_PER_USE_SKILL_TASK:-4}" \
    2>&1 | tee -a "${CHAIN_LOG}"
  [[ -s "${collected}/sft_messages.jsonl" ]] || fail "no new Claw SFT messages collected"

  start_base_27b_if_needed
  log "augmenting new Claw data with hindsight reasoning and schema block"
  mkdir -p "$(dirname "${augmented}")"
  "${PY}" GeneralAgent/sft_data_collection/augment_hindsight.py \
    --input "${collected}/sft_messages.jsonl" \
    --output "${augmented}" \
    --api-base "http://127.0.0.1:${BASE_PORT}/v1" \
    --model "${BASE_SERVED}" \
    --workers "${AUGMENT_WORKERS:-8}" \
    --max-tokens "${AUGMENT_MAX_TOKENS:-550}" \
    --timeout-sec "${AUGMENT_TIMEOUT_SEC:-240}" \
    --inject-tools-schema \
    --tokenizer-path "${BASE_MODEL_PATH}" \
    2>&1 | tee -a "${CHAIN_LOG}"
  [[ -s "${augmented}" ]] || fail "augment output is empty"

  log "filtering stale host-mode or looped trajectories"
  mkdir -p "$(dirname "${filtered}")"
  "${PY}" GeneralAgent/sft_data_collection/filter_clean_dataset.py \
    --input "${augmented}" \
    --output "${filtered}" \
    --report "${DATA_ROOT}/new_claw_filtered/filter_report.md" \
    2>&1 | tee -a "${CHAIN_LOG}"
  [[ -s "${filtered}" ]] || fail "filtered new Claw data is empty"

  log "applying think-wrap normalization"
  mkdir -p "$(dirname "${thinkwrap}")"
  "${PY}" GeneralAgent/sft_data_collection/apply_think_wrap.py \
    --input "${filtered}" \
    --output "${thinkwrap}" \
    2>&1 | tee -a "${CHAIN_LOG}"

  log "exporting new Claw data to LLaMA-Factory format"
  mkdir -p "${lf_new}"
  "${PY}" GeneralAgent/sft_training/export_llamafactory.py \
    --input "${thinkwrap}" \
    --out-dir "${lf_new}" \
    --dataset-name "${lf_new_name}" \
    2>&1 | tee -a "${CHAIN_LOG}"

  echo "${lf_new}/${lf_new_name}.json" > "${DATA_ROOT}/new_claw_lf_path.txt"
}

merge_lf_data_and_write_configs() {
  local new_lf
  new_lf="$(cat "${DATA_ROOT}/new_claw_lf_path.txt")"
  log "merging old clean LF data with new Claw LF data"
  mkdir -p "${LF_OUT_DIR}" "$(dirname "${TRAIN_CFG}")" "$(dirname "${EXPORT_CFG}")"
  "${PY}" - \
    "${OLD_CLEAN_LF}" \
    "${new_lf}" \
    "${LF_OUT_DIR}/${NEW_DATASET_NAME}.json" \
    "${LF_OUT_DIR}/dataset_info.json" \
    "${NEW_DATASET_NAME}" \
    "${REPORT_DIR}/merged_dataset_stats_${STAMP}.json" \
    "${REPORT_DIR}/merged_dataset_stats_live.md" <<'PY'
import collections
import hashlib
import json
import pathlib
import sys

old_path, new_path, out_data, out_info, name, out_stats, out_md = map(pathlib.Path, sys.argv[1:8])
old = json.loads(old_path.read_text(encoding="utf-8"))
new = json.loads(new_path.read_text(encoding="utf-8"))
merged = []
seen = set()
image_literal_dropped = collections.Counter()
for source, rows in [("old_clean", old), ("new_claw", new)]:
    for row in rows:
        row_text = json.dumps(row.get("messages", []), ensure_ascii=False)
        if "<image>" in row_text:
            image_literal_dropped[source] += 1
            continue
        md = row.get("metadata") or {}
        key = md.get("trajectory_path") or hashlib.sha256(json.dumps(row.get("messages", []), ensure_ascii=False).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        row = dict(row)
        md = dict(md)
        md.setdefault("merged_source", source)
        row["metadata"] = md
        merged.append(row)

out_data.parent.mkdir(parents=True, exist_ok=True)
out_data.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
dataset_info = {
    name.name: {
        "file_name": out_data.name,
        "formatting": "openai",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "observation_tag": "tool",
            "function_tag": "function",
            "system_tag": "system",
        },
    }
}
out_info.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

bench = collections.Counter()
used = collections.Counter()
source_counts = collections.Counter()
reasoning = 0
schema = 0
for row in merged:
    md = row.get("metadata") or {}
    b = md.get("bench") or "?"
    u = bool(md.get("used_skill"))
    bench[b] += 1
    used[(b, "used" if u else "not_used")] += 1
    source_counts[md.get("merged_source") or "?"] += 1
    assistant_text = "\n".join((m.get("content") or "") for m in row.get("messages", []) if m.get("role") == "assistant")
    reasoning += int("<skill_reasoning>" in assistant_text)
    schema += int("# Tools" in (row.get("messages", [{}])[0].get("content") or "")[:200])
stats = {
    "old_records": len(old),
    "new_records": len(new),
    "merged_records": len(merged),
    "source_counts": dict(source_counts),
    "image_literal_dropped": dict(image_literal_dropped),
    "bench_counts": dict(bench),
    "used_skill_by_bench": {f"{b}:{k}": v for (b, k), v in used.items()},
    "records_with_skill_reasoning_prefix": reasoning,
    "records_with_tools_schema_prefix": schema,
    "dataset_file": str(out_data),
    "dataset_info": str(out_info),
}
out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
lines = ["# Merged SFT Dataset", "", f"- total: {len(merged)}", f"- old_clean: {len(old)}", f"- new_claw: {len(new)}", f"- skill_reasoning_prefix: {reasoning}", f"- tools_schema_prefix: {schema}", "", "| bench | total | used_skill | not_used |", "|---|---:|---:|---:|"]
for b in sorted(bench):
    lines.append(f"| {b} | {bench[b]} | {used[(b, 'used')]} | {used[(b, 'not_used')]} |")
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(stats, ensure_ascii=False))
PY

  log "writing generated 27B train/export configs"
  "${PY}" - \
    "${OLD_TRAIN_CFG}" \
    "${TRAIN_CFG}" \
    "${NEW_DATASET_NAME}" \
    "${PROJECT_ROOT}/${LF_OUT_DIR}" \
    "${TRAIN_OUTPUT}" \
    "${OLD_EXPORT_CFG}" \
    "${EXPORT_CFG}" \
    "${MERGED}" <<'PY'
import pathlib
import sys

old_train, train_cfg, dataset, dataset_dir, output_dir, old_export, export_cfg, merged = sys.argv[1:9]
old_train = pathlib.Path(old_train)
train_cfg = pathlib.Path(train_cfg)
text = old_train.read_text(encoding="utf-8")
out = []
for line in text.splitlines():
    if line.startswith("dataset:"):
        out.append(f"dataset: {dataset}")
    elif line.startswith("dataset_dir:"):
        out.append(f"dataset_dir: {dataset_dir}")
    elif line.startswith("output_dir:"):
        out.append(f"output_dir: {output_dir}")
    elif line.startswith("# 20260509 clean variant:"):
        out.append(f"# {dataset}: old clean campaign plus docker-sandbox Claw supplement.")
    else:
        out.append(line)
train_cfg.write_text("\n".join(out) + "\n", encoding="utf-8")

old_export = pathlib.Path(old_export)
export_cfg = pathlib.Path(export_cfg)
text = old_export.read_text(encoding="utf-8")
out = []
for line in text.splitlines():
    if line.startswith("adapter_name_or_path:"):
        out.append(f"adapter_name_or_path: {output_dir}")
    elif line.startswith("export_dir:"):
        out.append(f"export_dir: {merged}")
    else:
        out.append(line)
export_cfg.write_text("\n".join(out) + "\n", encoding="utf-8")
print(train_cfg)
print(export_cfg)
PY
}

training_complete() {
  "${PY}" - "${TRAINER_LOG}" "${TRAIN_OUTPUT}/trainer_state.json" <<'PY'
import json
import pathlib
import sys
log_p, state_p = map(pathlib.Path, sys.argv[1:])
if state_p.exists():
    try:
        d = json.loads(state_p.read_text())
        if int(d.get("global_step") or 0) >= int(d.get("max_steps") or 10**18):
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        pass
if not log_p.exists():
    raise SystemExit(1)
rows = [json.loads(x) for x in log_p.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
if not rows:
    raise SystemExit(1)
d = rows[-1]
cur, total = int(d.get("current_steps") or 0), int(d.get("total_steps") or 0)
raise SystemExit(0 if total and cur >= total else 1)
PY
}

trainer_snapshot() {
  "${PY}" - "${TRAINER_LOG}" <<'PY' || true
import json
import pathlib
import sys
p = pathlib.Path(sys.argv[1])
if not p.exists():
    print("trainer_log=missing")
    raise SystemExit
rows = [json.loads(x) for x in p.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
if not rows:
    print("trainer_log=empty")
    raise SystemExit
d = rows[-1]
print("step={}/{} pct={} epoch={} loss={} elapsed={} remaining={}".format(
    d.get("current_steps", "?"), d.get("total_steps", "?"), d.get("percentage", "?"),
    d.get("epoch", "?"), d.get("loss", "?"), d.get("elapsed_time", "?"), d.get("remaining_time", "?")))
PY
}

run_training() {
  if training_complete; then
    log "training already complete: $(trainer_snapshot)"
    return
  fi
  stop_sglang_before_training
  log "launching 8GPU 27B SFT: cfg=${TRAIN_CFG}"
  tmux kill-session -t "${TRAIN_TMUX}" 2>/dev/null || true
  mkdir -p "${LOCAL_HF_CACHE}/datasets" "${LOCAL_HF_CACHE}/transformers"
  tmux new-session -d -s "${TRAIN_TMUX}" \
    "cd '${PROJECT_ROOT}' && bash -lc 'source GeneralAgent/sft_training/activate_llamafactory.sh && export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 NNODES=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DISABLE_VERSION_CHECK=1 LLAMAFACTORY_ALLOW_TORCH29_CONV3D=1 CUDA_HOME=${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/envs/slime CUDA_PATH=${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/envs/slime HF_ENDPOINT=https://hf-mirror.com HF_HOME=${LOCAL_HF_CACHE} TRANSFORMERS_CACHE=${LOCAL_HF_CACHE}/transformers HF_DATASETS_CACHE=${LOCAL_HF_CACHE}/datasets && llamafactory-cli train ${TRAIN_CFG}' 2>&1 | tee '${TRAIN_STDOUT_LOG}'"
  while ! training_complete; do
    if ! tmux has-session -t "${TRAIN_TMUX}" 2>/dev/null; then
      tail -160 "${TRAIN_STDOUT_LOG}" || true
      fail "training tmux exited before completion"
    fi
    log "training progress: $(trainer_snapshot)"
    sleep "${POLL_SEC}"
  done
  log "training complete: $(trainer_snapshot)"
}

verify_training_loss() {
  log "verifying training loss decrease"
  "${PY}" - "${TRAINER_LOG}" "${REPORT_DIR}/training_loss_check_${STAMP}.json" <<'PY'
import json
import pathlib
import statistics
import sys
log_p, out_p = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
rows = [json.loads(x) for x in log_p.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
losses = [float(r["loss"]) for r in rows if r.get("loss") is not None]
if len(losses) < 10:
    raise SystemExit("not enough loss rows")
first = statistics.mean(losses[:min(10, len(losses))])
last = statistics.mean(losses[-min(10, len(losses)):])
payload = {"num_loss_rows": len(losses), "first_window_loss": first, "last_window_loss": last, "passed": last < first}
out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
if not payload["passed"]:
    raise SystemExit("loss gate failed")
PY
}

export_model() {
  if [[ -f "${MERGED}/config.json" && -f "${MERGED}/tokenizer.json" ]]; then
    log "merged model already exists: ${MERGED}"
    return
  fi
  log "exporting merged model to ${MERGED}"
  export CUDA_VISIBLE_DEVICES="${EXPORT_GPUS:-0}"
  export CUDA_HOME="${CUDA_HOME:-${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/envs/slime}"
  export CUDA_PATH="${CUDA_HOME}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/hf_cache}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
  export HF_DATASETS_CACHE="${HF_HOME}/datasets"
  export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
  export LLAMAFACTORY_ALLOW_TORCH29_CONV3D="${LLAMAFACTORY_ALLOW_TORCH29_CONV3D:-1}"
  source GeneralAgent/sft_training/activate_llamafactory.sh
  llamafactory-cli export "${EXPORT_CFG}" 2>&1 | tee "${LOG_DIR}/export_${STAMP}.log"
  [[ -f "${MERGED}/config.json" && -f "${MERGED}/tokenizer.json" ]] || fail "export incomplete: ${MERGED}"
}

start_sft_27b() {
  local session="${SFT_TMUX:-sglang-qwen27b-sft-clean-plus-claw-0to3}"
  local current slog
  current="$(served_model "${SFT_PORT}")"
  if [[ "${current}" == "${SFT_SERVED}" ]]; then
    log "SFT 27B already serving on :${SFT_PORT}"
    return
  fi
  tmux kill-session -t "${session}" 2>/dev/null || true
  slog="${LOG_DIR}/${session}_${STAMP}.log"
  log "starting SFT 27B: port=${SFT_PORT} merged=${MERGED}"
  tmux new-session -d -s "${session}" \
    "cd '${PROJECT_ROOT}' && CUDA_VISIBLE_DEVICES='0,1,2,3' bash -lc 'source ${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/etc/profile.d/conda.sh && conda activate slime && export NO_PROXY=\"127.0.0.1,localhost,0.0.0.0\" no_proxy=\"127.0.0.1,localhost,0.0.0.0\" && export SGLANG_DISABLE_CUDNN_CHECK=1 CUDA_HOME=\"\$CONDA_PREFIX\" && exec python -m sglang.launch_server --model-path \"${MERGED}\" --served-model-name \"${SFT_SERVED}\" --tensor-parallel-size \"${TP_SIZE}\" --host 0.0.0.0 --port \"${SFT_PORT}\" --context-length \"${CONTEXT_LENGTH}\" --mem-fraction-static \"${MEM_FRACTION}\" --random-seed \"${RANDOM_SEED}\" --enable-metrics --log-level info' 2>&1 | tee '${slog}'"
  local deadline=$((SECONDS + 2400))
  while (( SECONDS < deadline )); do
    current="$(served_model "${SFT_PORT}")"
    [[ "${current}" == "${SFT_SERVED}" ]] && { log "SFT 27B ready"; return; }
    tmux has-session -t "${session}" 2>/dev/null || { tail -120 "${slog}" || true; fail "SFT 27B SGLang died"; }
    sleep 20
  done
  tail -120 "${slog}" || true
  fail "timeout waiting for SFT 27B"
}

register_run() {
  local run_id="$1" run_root="$2" status="$3" intent="$4" tags="$5" launcher="$6"
  "${PY}" ops/experiments/register_experiment.py \
    --run-id "${run_id}" \
    --path "${run_root}" \
    --date "${DATE}" \
    --kind "eval" \
    --status "${status}" \
    --launcher "${launcher}" \
    --scripts "ops/launch/run_claw_collect_to_27b_sft_eval_chain.sh,ops/launch/run_dynamic_bench.sh,GeneralAgent/sft_data_collection/data_quality_dashboard.py" \
    --intent "${intent}" \
    --notes "OpenClaw-full profile; frozen retrieval ${RETRIEVAL_DATE}_${RETR_SUFFIX}; dataset=${NEW_DATASET_NAME}" \
    --tags "${tags}" || true
}

run_quick_holdout() {
  local run_id="${DATE}_quick30_sft27b_clean_plus_claw_openclaw_full_retrieval"
  local run_root="experiments/${DATE}/${run_id}"
  log "running SFT 27B quick30 retrieval: ${run_id}"
  RUN_ID="${run_id}" RUN_ROOT="${run_root}" EXPERIMENT_ROOT="${run_root}" DATE="${DATE}" \
  MODEL="${SFT_SERVED}" ARM="retrieval" OPENAI_API_BASE="http://127.0.0.1:${SFT_PORT}/v1" \
  OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local-anything}" UNIFIED_PROMPT_PROFILE="openclaw_full" \
  UNIFIED_TOOLS_SCHEMA_MODE="openai_tools" UNIFIED_CLAW_USE_DOCKER_SANDBOX=1 \
  TRIALS=1 WORKERS="${QUICK_WORKERS}" TIMEOUT="${QUICK_TIMEOUT}" \
  bash ops/launch/run_quick_holdout_eval.sh
  "${PY}" GeneralAgent/sft_data_collection/data_quality_dashboard.py "${run_id}" --run-root "${run_root}" || true
}

behavior_gate_quick() {
  local run_id="${DATE}_quick30_sft27b_clean_plus_claw_openclaw_full_retrieval"
  local run_root="experiments/${DATE}/${run_id}"
  log "checking quick30 hindsight/skill-use behavior"
  "${PY}" - "${run_root}" "${REPORT_DIR}/quick30_behavior_gate_${STAMP}.json" <<'PY'
import json
import pathlib
import sys

root, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
traj = list(root.glob("results/**/trajectories/*.json"))
reason = 0
skill_reads = 0
resolved = 0
for p in traj:
    text = p.read_text(encoding="utf-8", errors="ignore")
    if "<skill_reasoning>" in text:
        reason += 1
    if "SKILL.md" in text:
        skill_reads += 1
    try:
        data = json.loads(text)
        md = data.get("metadata") or {}
        resolved += int(bool(md.get("resolved")))
    except Exception:
        pass
n = len(traj)
payload = {
    "trajectories": n,
    "resolved": resolved,
    "with_skill_reasoning": reason,
    "strict_skill_read_text_contains_SKILL_md": skill_reads,
    "reasoning_rate": reason / n if n else 0,
    "skill_read_rate": skill_reads / n if n else 0,
    "passed": n > 0 and reason / n >= 0.50 and skill_reads / n >= 0.30,
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
if not payload["passed"]:
    raise SystemExit(2)
PY
}

cleanup_eval_tmux() {
  local prefix="$1"
  while read -r session; do
    [[ -n "${session}" ]] || continue
    tmux kill-session -t "${session}" 2>/dev/null || true
  done < <(tmux ls 2>/dev/null | awk -F: -v p="${prefix}" 'index($1, p "_") == 1 {print $1}')
}

sessions_remaining() {
  local prefix="$1"
  tmux ls 2>/dev/null | awk -F: -v p="${prefix}" 'index($1, p "_") == 1 {print $1}' | wc -l
}

pending_summary() {
  local namespace="$1" arm="$2"
  local qdir="/tmp/v9_queue/${namespace}"
  for bench in seta swe claw sb_ns tb2; do
    local q="${qdir}/${bench}_${arm}_pending.txt"
    [[ -f "${q}" ]] && printf "%s=%s " "${bench}" "$(wc -l < "${q}")"
  done
  echo
}

launch_full_bench() {
  local bench="$1" arm="$2" par="$3" namespace="$4" prefix="$5" run_id="$6" run_root="$7"
  RESULT_DATE="${DATE}" RETRIEVAL_DATE="${RETRIEVAL_DATE}" RETR_SUFFIX="${RETR_SUFFIX}" \
  RUN_ID="${run_id}" RUN_ROOT="${run_root}" EXPERIMENT_ROOT="${run_root}" UNIFIED_RUN_ID="${run_id}" \
  UNIFIED_MODEL="${SFT_SERVED}" PHASE_B_MODEL="${SFT_SERVED}" UNIFIED_EXP_VERSION="sft27b_clean_plus_claw_openclaw_full" \
  OPENAI_API_BASE="http://127.0.0.1:${SFT_PORT}/v1" OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local-anything}" \
  UNIFIED_PROMPT_PROFILE="openclaw_full" UNIFIED_TOOLS_SCHEMA_MODE="openai_tools" UNIFIED_CLAW_USE_DOCKER_SANDBOX=1 \
  DYN_QUEUE_NAMESPACE="${namespace}" DYN_TMUX_PREFIX="${prefix}" \
  bash ops/launch/run_dynamic_bench.sh "${bench}" "${arm}" "${par}" 2>&1 | tee -a "${CHAIN_LOG}"
}

run_full_sft_retrieval() {
  local arm="retrieval"
  local run_id="${DATE}_full_sft27b_clean_plus_claw_retrieval_openclaw_full"
  local run_root="experiments/${DATE}/${run_id}"
  local prefix="g27sftplus_retr"
  local namespace="g27sftplus_retr"
  log "starting full SFT 27B retrieval eval: ${run_id}"
  mkdir -p "${run_root}/logs/runner" "${run_root}/reports"
  register_run "${run_id}" "${run_root}" "running" "full retrieval eval for 27B SFT clean-plus-claw under openclaw_full" "full,sft27b,retrieval,openclaw_full" "ops/launch/run_claw_collect_to_27b_sft_eval_chain.sh"
  cleanup_eval_tmux "${prefix}"
  for bench in ${FULL_BENCHES}; do
    local par_var="FULL_PAR_${bench}"
    launch_full_bench "${bench}" "${arm}" "${!par_var}" "${namespace}" "${prefix}" "${run_id}" "${run_root}"
  done
  local last=0
  while (( "$(sessions_remaining "${prefix}")" > 0 )); do
    if (( SECONDS - last >= 300 )); then
      last="${SECONDS}"
      log "waiting full SFT retrieval: sessions=$(sessions_remaining "${prefix}") pending=$(pending_summary "${namespace}" "${arm}")"
    fi
    sleep 30
  done
  log "full SFT retrieval workers done; building dashboard"
  "${PY}" GeneralAgent/sft_data_collection/data_quality_dashboard.py "${run_id}" --run-root "${run_root}" || true
  register_run "${run_id}" "${run_root}" "completed" "full retrieval eval for 27B SFT clean-plus-claw under openclaw_full" "full,sft27b,retrieval,openclaw_full" "ops/launch/run_claw_collect_to_27b_sft_eval_chain.sh"
}

write_final_summary() {
  log "writing final chain summary"
  "${PY}" - "${DATE}" "${CHAIN_ID}" "${REPORT_DIR}/final_summary.md" <<'PY'
import json
import pathlib
import sys

date, chain_id, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
runs = [
    ("quick30_sft27b", f"{date}_quick30_sft27b_clean_plus_claw_openclaw_full_retrieval"),
    ("full_sft27b", f"{date}_full_sft27b_clean_plus_claw_retrieval_openclaw_full"),
]
lines = [f"# {chain_id}", ""]
for label, rid in runs:
    root = pathlib.Path("experiments") / date / rid
    dash = root / "reports" / "data_quality_dashboard.json"
    lines += [f"## {label}", "", f"- run: `{rid}`", f"- path: `{root}`"]
    if dash.exists():
        d = json.loads(dash.read_text())
        t = d.get("totals") or {}
        n = t.get("trajectories_loaded", 0)
        ok = t.get("resolved", 0)
        lines += [
            f"- resolved: {ok}/{n} ({(100*ok/n if n else 0):.1f}%)",
            f"- strict_used_skill: {t.get('strict_used_skill', 0)}",
            f"- success_strict_used_skill: {t.get('success_strict_used_skill', 0)}",
        ]
    else:
        lines.append("- dashboard: missing")
    lines.append("")
out.write_text("\n".join(lines), encoding="utf-8")
print(out)
PY
}

main() {
  : > "${LIVE_LOG}"
  wait_for_claw_collection
  audit_claw_collection
  collect_and_process_new_claw
  merge_lf_data_and_write_configs
  run_training
  verify_training_loss
  export_model
  start_sft_27b
  run_quick_holdout
  behavior_gate_quick | tee -a "${CHAIN_LOG}"
  run_full_sft_retrieval
  write_final_summary
  log "chain complete"
}

main "$@"
