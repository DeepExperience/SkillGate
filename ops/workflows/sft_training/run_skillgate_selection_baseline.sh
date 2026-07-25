#!/usr/bin/env bash
# Canonical, restartable trainer/exporter for the SkillGate paper's selection
# baselines. Usage: run_skillgate_selection_baseline.sh bc|dpo [all|train|export]
#
# Both arms start from the frozen merged SFT9B model. LLaMA-Factory resumes from
# the latest checkpoint in the fixed owner segment when this entrypoint is
# rerun. Export is staged in a temporary directory, validated, then atomically
# renamed into the owner experiment's model/exports directory.
set -Eeuo pipefail

ROOT="${ROOT:-/path/to/skillRL}"
cd "${ROOT}"

ARM="${1:-}"
MODE="${2:-all}"
case "${ARM}" in
  bc)
    OWNER="goldbc-selection-sft9b-20260721"
    METHOD="selection_turn_bc"
    OBJECTIVE="Gold Selector BC: teacher-force a first-turn read of the advertised oracle skill on all 491 FINAL Hybrid training tasks."
    TRAIN_CONFIG="${ROOT}/GeneralAgent/sft_training/configs/skillgate_gold_selector_bc_lora_20260721.yaml"
    EXPORT_CONFIG="${ROOT}/GeneralAgent/sft_training/configs/skillgate_gold_selector_bc_export_20260721.yaml"
    DATA_FILE="${ROOT}/GeneralAgent/sft_training/llamafactory_data/20260721_selection_bc/skillgate_gold_selector_bc_20260721.json"
    EXPORT_ID="bc-lora-merged"
    ;;
  dpo)
    OWNER="selskill-dpo-selection-sft9b-20260721"
    METHOD="selskill_style_dpo"
    OBJECTIVE="SelSkill-style preference adaptation: prefer first-turn read(gold) over each of five advertised misleading candidates on all 491 FINAL Hybrid training tasks."
    TRAIN_CONFIG="${ROOT}/GeneralAgent/sft_training/configs/skillgate_selskill_dpo_lora_20260721.yaml"
    EXPORT_CONFIG="${ROOT}/GeneralAgent/sft_training/configs/skillgate_selskill_dpo_export_20260721.yaml"
    DATA_FILE="${ROOT}/GeneralAgent/sft_training/llamafactory_data/20260721_selection_dpo/skillgate_selskill_dpo_20260721.json"
    EXPORT_ID="dpo-lora-merged"
    ;;
  *)
    echo "Usage: $0 bc|dpo [all|train|export]" >&2
    exit 2
    ;;
esac
case "${MODE}" in all|train|export) ;; *) echo "invalid mode: ${MODE}" >&2; exit 2 ;; esac

BASE_MODEL="${ROOT}/GeneralAgent/sft_training/merged_models/qwen35_9b_sft_campaign_20260512_clean_plus_claw_thinkwrap_4gpu_49k_5epoch_r32_liger"
OWNER_DIR="${ROOT}/experiments/rl/runs/${OWNER}"
SEGMENT_ID="20260721-initial"
SEGMENT_DIR="${OWNER_DIR}/segments/${SEGMENT_ID}"
CHECKPOINT_DIR="${SEGMENT_DIR}/checkpoints"
EXPORT_DIR="${OWNER_DIR}/model/exports/${EXPORT_ID}"
TRAIN_MARKER="${SEGMENT_DIR}/TRAIN_COMPLETE"
DRIVER_LOG="${SEGMENT_DIR}/driver.log"
DATA_REPORT="${ROOT}/GeneralAgent/sft_training/llamafactory_data/20260721_selection_bc/build_report.json"

for path in "${TRAIN_CONFIG}" "${EXPORT_CONFIG}" "${DATA_FILE}" "${DATA_REPORT}"; do
  [[ -f "${path}" ]] || { echo "FATAL: required file missing: ${path}" >&2; exit 2; }
done
[[ -d "${BASE_MODEL}" ]] || { echo "FATAL: base model missing: ${BASE_MODEL}" >&2; exit 2; }

mkdir -p "${SEGMENT_DIR}" "${OWNER_DIR}/model/exports"
exec {owner_lock_fd}>"${OWNER_DIR}/.selection_baseline.lock"
flock "${owner_lock_fd}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NNODES=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export LLAMAFACTORY_ALLOW_TORCH29_CONV3D="${LLAMAFACTORY_ALLOW_TORCH29_CONV3D:-1}"
export CUDA_HOME="${CUDA_HOME:-/path/to/conda/envs/slime}"
export CUDA_PATH="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/data/cache/skillgate_lf_hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

# shellcheck source=/dev/null
source "${ROOT}/GeneralAgent/sft_training/activate_llamafactory.sh"

python "${ROOT}/GeneralAgent/rl_data_prep/build_selector_bc_dpo_data.py" --validate-only \
  >"${SEGMENT_DIR}/data_validation.json"
cp "${TRAIN_CONFIG}" "${SEGMENT_DIR}/resolved_train_config.yaml"
cp "${EXPORT_CONFIG}" "${SEGMENT_DIR}/resolved_export_config.yaml"

manifest_update() {
  local status="$1" stage="$2" return_code="${3:-0}"
  python - \
    "${OWNER_DIR}" "${OWNER}" "${SEGMENT_ID}" "${SEGMENT_DIR}" \
    "${OBJECTIVE}" "${METHOD}" "${status}" "${stage}" "${return_code}" \
    "${TRAIN_CONFIG}" "${EXPORT_CONFIG}" "${DATA_FILE}" "${DATA_REPORT}" \
    "${BASE_MODEL}" "${CHECKPOINT_DIR}" "${EXPORT_ID}" "${EXPORT_DIR}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(owner_dir, owner, segment_id, segment_dir, objective, method, status, stage,
 return_code, train_config, export_config, data_file, data_report, base_model,
 checkpoint_dir, export_id, export_dir) = sys.argv[1:]
owner_dir, segment_dir = Path(owner_dir), Path(segment_dir)
path = owner_dir / "experiment.json"
now = datetime.now(timezone.utc).isoformat()

def read_json(candidate):
    try:
        return json.loads(Path(candidate).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}

def file_hash(candidate):
    digest = hashlib.sha256()
    with Path(candidate).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()

experiment = read_json(path)
if experiment and experiment.get("experiment_id") != owner:
    raise SystemExit(f"owner collision: {path}")
segments = {
    item.get("segment_id"): item
    for item in experiment.get("segments", [])
    if isinstance(item, dict) and item.get("segment_id")
}
segment = segments.get(segment_id, {})
segment.update({
    "segment_id": segment_id,
    "path": str(segment_dir),
    "created_at": segment.get("created_at") or now,
    "updated_at": now,
    "status": status,
    "stage": stage,
    "return_code": int(return_code),
})
if status in {"completed", "failed", "trained"}:
    segment["completed_at"] = now
segments[segment_id] = segment

report = read_json(data_report)
experiment.update({
    "schema_version": 1,
    "kind": "selection_baseline_training",
    "experiment_id": owner,
    "experiment_dir": str(owner_dir),
    "objective": objective,
    "created_at": experiment.get("created_at") or now,
    "updated_at": now,
    "status": status,
    "segments": sorted(segments.values(), key=lambda item: item["segment_id"]),
    "inputs": {
        "data": data_file,
        "data_sha256": file_hash(data_file),
        "data_protocol": report.get("protocol", ""),
        "source_train_parquet": report.get("inputs", {}).get("train_parquet", ""),
        "source_train_parquet_sha256": report.get("inputs", {}).get("train_parquet_sha256", ""),
        "base_model": base_model,
    },
    "training": {
        "method": method,
        "train_config": train_config,
        "export_config": export_config,
        "checkpoint_dir": checkpoint_dir,
    },
    "evals": experiment.get("evals", []),
})
model = experiment.setdefault("model", {})
model["adapter"] = checkpoint_dir
if Path(export_dir).is_dir() and (Path(export_dir) / "COMPLETE").is_file():
    model["exports"] = sorted(set(model.get("exports", [])) | {export_id})
    model["selected"] = {"final_hf": export_id, "exports": model["exports"], "updated_at": now}

tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
tmp.write_text(json.dumps(experiment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(tmp, path)
PY
}

validate_adapter() {
  [[ -f "${CHECKPOINT_DIR}/adapter_config.json" ]] || return 1
  [[ -s "${CHECKPOINT_DIR}/adapter_model.safetensors" || -s "${CHECKPOINT_DIR}/adapter_model.bin" ]] || return 1
  [[ -f "${CHECKPOINT_DIR}/trainer_state.json" ]] || return 1
}

validate_hf_export() {
  local root="$1" require_complete="${2:-1}"
  python - "${root}" "${require_complete}" <<'PY'
import json
import sys
from pathlib import Path
from safetensors import safe_open

root = Path(sys.argv[1])
require_complete = int(sys.argv[2])
if require_complete and not (root / "COMPLETE").is_file():
    raise SystemExit(1)
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
if config.get("model_type") != "qwen3_5":
    raise SystemExit(f"unexpected model_type={config.get('model_type')!r}")
index_path = root / "model.safetensors.index.json"
if index_path.is_file():
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") or {}
    shards = sorted(set(weight_map.values()))
else:
    shards = ["model.safetensors"]
if not shards:
    raise SystemExit("no model shards")
for shard in shards:
    path = root / shard
    if not path.is_file() or path.stat().st_size <= 8:
        raise SystemExit(f"missing/truncated shard: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        if not list(handle.keys()):
            raise SystemExit(f"empty shard: {path}")
for name in ("tokenizer.json", "tokenizer_config.json"):
    if not (root / name).is_file():
        raise SystemExit(f"missing {name}")
print(f"HF_EXPORT_OK root={root} shards={len(shards)}")
PY
}

FAILED_STAGE="preflight"
SUCCESS=0
on_exit() {
  local rc=$?
  if ((SUCCESS == 0)); then
    manifest_update failed "${FAILED_STAGE}" "${rc}" || true
  fi
}
trap on_exit EXIT

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "DRY_RUN_OK arm=${ARM} mode=${MODE} owner=${OWNER}"
  echo "train_config=${TRAIN_CONFIG}"
  echo "data_file=${DATA_FILE}"
  echo "export_dir=${EXPORT_DIR}"
  SUCCESS=1
  exit 0
fi

manifest_update running preflight 0

exec > >(tee -a "${DRIVER_LOG}") 2>&1
echo "[$(date -Is)] START arm=${ARM} mode=${MODE} owner=${OWNER}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC_PER_NODE=${NPROC_PER_NODE}"
python - <<'PY'
from importlib.metadata import version
for package in ("torch", "transformers", "trl", "peft", "deepspeed", "liger-kernel", "pandas", "pyarrow", "llamafactory"):
    try:
        print(f"package {package}={version(package)}")
    except Exception as exc:
        print(f"package {package}=UNAVAILABLE ({exc})")
PY

if [[ "${MODE}" == all || "${MODE}" == train ]]; then
  FAILED_STAGE="train"
  if [[ -f "${TRAIN_MARKER}" ]] && validate_adapter; then
    echo "[$(date -Is)] training already complete: ${CHECKPOINT_DIR}"
  else
    manifest_update running train 0
    if [[ "${ARM}" == dpo ]]; then
      compat_dir="${ROOT}/GeneralAgent/sft_training/compat/liger_qwen35_dpo"
      PYTHONPATH="${compat_dir}:${PYTHONPATH}" python -c \
        'from liger_kernel.transformers.cross_entropy import liger_cross_entropy; from llamafactory.train.dpo.trainer import CustomDPOTrainer; assert getattr(CustomDPOTrainer.concatenated_forward, "_skillgate_completion_logits", False)'
      PYTHONPATH="${compat_dir}:${PYTHONPATH}" llamafactory-cli train "${TRAIN_CONFIG}"
    else
      llamafactory-cli train "${TRAIN_CONFIG}"
    fi
    validate_adapter
    printf 'validated %s\n' "$(date -Is)" >"${TRAIN_MARKER}"
    echo "[$(date -Is)] training complete: ${CHECKPOINT_DIR}"
  fi
  if [[ "${MODE}" == train ]]; then
    manifest_update trained train 0
    SUCCESS=1
    exit 0
  fi
fi

if [[ "${MODE}" == all || "${MODE}" == export ]]; then
  FAILED_STAGE="export"
  [[ -f "${TRAIN_MARKER}" ]] && validate_adapter || {
    echo "FATAL: adapter training is not complete: ${CHECKPOINT_DIR}" >&2
    exit 2
  }
  manifest_update running export 0
  if validate_hf_export "${EXPORT_DIR}" 1; then
    echo "[$(date -Is)] export already complete: ${EXPORT_DIR}"
  else
    if [[ -e "${EXPORT_DIR}" ]]; then
      stale="${EXPORT_DIR}.incomplete-$(date +%Y%m%d_%H%M%S)"
      mv "${EXPORT_DIR}" "${stale}"
      echo "preserved incomplete export at ${stale}"
    fi
    tmp="$(dirname "${EXPORT_DIR}")/.${EXPORT_ID}.tmp-${BASHPID}"
    [[ ! -e "${tmp}" ]] || { echo "FATAL: stale export temp exists: ${tmp}" >&2; exit 2; }
    CUDA_VISIBLE_DEVICES="${EXPORT_GPUS:-0}" NPROC_PER_NODE=1 \
      llamafactory-cli export "${EXPORT_CONFIG}" "export_dir=${tmp}"
    validate_hf_export "${tmp}" 0
    python - "${tmp}" "${OWNER}" "${EXPORT_ID}" "${METHOD}" "${CHECKPOINT_DIR}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root, owner, export_id, method, adapter = sys.argv[1:]
root = Path(root)
(root / "export.json").write_text(json.dumps({
    "schema_version": 1,
    "experiment_id": owner,
    "export_id": export_id,
    "method": method,
    "adapter": adapter,
    "created_at": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
(root / "COMPLETE").write_text("validated\n")
PY
    mv "${tmp}" "${EXPORT_DIR}"
    validate_hf_export "${EXPORT_DIR}" 1
    echo "[$(date -Is)] export complete: ${EXPORT_DIR}"
  fi

  python - "${OWNER_DIR}" "${OWNER}" "${EXPORT_ID}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

owner_dir, owner, export_id = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
model_dir = owner_dir / "model"
selected_path = model_dir / "selected.json"
now = datetime.now(timezone.utc).isoformat()
payload = {
    "schema_version": 1,
    "experiment_id": owner,
    "exports": [export_id],
    "final_hf": export_id,
    "updated_at": now,
}
tmp = selected_path.with_name(f".{selected_path.name}.tmp-{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2) + "\n")
os.replace(tmp, selected_path)
alias = model_dir / "final_hf"
alias_tmp = model_dir / f".final_hf.tmp-{os.getpid()}"
try:
    alias_tmp.unlink()
except FileNotFoundError:
    pass
alias_tmp.symlink_to(Path("exports") / export_id)
os.replace(alias_tmp, alias)
PY
fi

manifest_update completed complete 0
echo "[$(date -Is)] COMPLETE arm=${ARM} export=${EXPORT_DIR}"
SUCCESS=1
