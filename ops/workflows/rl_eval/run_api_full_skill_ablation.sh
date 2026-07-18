#!/usr/bin/env bash
set -Eeuo pipefail

# Full RL train+eval skill ablation with an OpenAI-compatible API model.
#
# Purpose:
#   Run the same OpenClaw-aligned unified-runner evaluation on the clean RL
#   task universe twice, sequentially:
#     1. baseline  — no retrieval skills injected
#     2. retrieval — frozen v7 retrieval top-10 skills injected
#
# Safe default:
#   EXECUTE=0 only generates plans and prints launch dry-runs. Set EXECUTE=1
#   after stopping RL or moving eval to a separate remote Docker endpoint.
#
# Resume:
#   Reuse the same RUN_GROUP/RUN_ID; launch_trials.py skips completed
#   trajectories unless RERUN_COMPLETED=1 is set.

PROJECT_ROOT="${SKILLRL_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

if [[ -f "secrets/.env.secrets" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "secrets/.env.secrets"
  set +a
fi

DATE="${DATE:-$(date +%Y%m%d)}"
MODEL="${MODEL:-glm-5.1}"
RUN_GROUP="${RUN_GROUP:-${DATE}_api_strong_skill_ablation_full_${MODEL//[^A-Za-z0-9]/_}}"
: "${OWNER_EXPERIMENT_ID:?set OWNER_EXPERIMENT_ID to the reference-model owner experiment}"
OWNER_ROOT="experiments/rl/runs/${OWNER_EXPERIMENT_ID}"
[[ -f "${OWNER_ROOT}/experiment.json" ]] || {
  echo "ERROR: owner experiment is missing: ${OWNER_ROOT}/experiment.json" >&2
  exit 2
}
EVAL_ROOT="${EVAL_ROOT:-${OWNER_ROOT}/eval}"
CONFIG="${CONFIG:-GeneralAgent/sft_data_collection/configs/default_collection_config.json}"
TRAIN_PARQUET="${TRAIN_PARQUET:-datasets/rl/parquet_4bench_base_20260523/train.parquet}"
EVAL_PARQUET="${EVAL_PARQUET:-datasets/rl/parquet_4bench_base_20260523/eval.parquet}"

API_BASE="${API_BASE:-${MAAS_API_BASE:-}}"
API_KEY="${API_KEY:-${MAAS_API_KEY:-}}"
if [[ -z "${API_BASE}" || -z "${API_KEY}" ]]; then
  echo "ERROR: API_BASE/API_KEY are empty. Source secrets/.env.secrets or set API_BASE/API_KEY." >&2
  exit 2
fi

DOCKER_HOST_VALUE="${DOCKER_HOST_VALUE:-unix:///tmp/local-docker-overlay2.sock}"
WORKERS="${WORKERS:-64}"
DOCKER_START_CAP="${DOCKER_START_CAP:-8}"
ALLOW_CONCURRENT_CLAW="${ALLOW_CONCURRENT_CLAW:-1}"
TRIALS="${TRIALS:-1}"
MAX_TURNS="${MAX_TURNS:-30}"
MAX_TIME="${MAX_TIME:-850}"
TIMEOUT="${TIMEOUT:-2700}"
DOCKER_WAIT_SEC="${DOCKER_WAIT_SEC:-54000}"
DOCKER_CHECK_INTERVAL_SEC="${DOCKER_CHECK_INTERVAL_SEC:-30}"
TASK_WINDOW="${TASK_WINDOW:-0}"
ARMS="${ARMS:-baseline retrieval}"
EXECUTE="${EXECUTE:-0}"
RERUN_COMPLETED="${RERUN_COMPLETED:-0}"

export OPENAI_API_BASE="${API_BASE%/}"
export OPENAI_API_KEY="${API_KEY}"
export DOCKER_HOST="${DOCKER_HOST_VALUE}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,0.0.0.0,10.0.0.0/8}"
export no_proxy="${no_proxy:-${NO_PROXY}}"
export UNIFIED_PROMPT_PROFILE="${UNIFIED_PROMPT_PROFILE:-openclaw_full}"
export UNIFIED_TOOLS_SCHEMA_MODE="${UNIFIED_TOOLS_SCHEMA_MODE:-openai_tools}"
export UNIFIED_CLAW_USE_DOCKER_SANDBOX="${UNIFIED_CLAW_USE_DOCKER_SANDBOX:-1}"
export UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS="${UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS:-1}"
export UNIFIED_HARBOR_REQUIRE_PREBUILT_LOCAL="${UNIFIED_HARBOR_REQUIRE_PREBUILT_LOCAL:-1}"
export UNIFIED_ROLLOUT_WALLCLOCK_CAP_SEC="${UNIFIED_ROLLOUT_WALLCLOCK_CAP_SEC:-${MAX_TIME}}"
export UNIFIED_VERIFIER_TIMEOUT_CAP_SEC="${UNIFIED_VERIFIER_TIMEOUT_CAP_SEC:-300}"
export AGENT_BENCH_DOCKER_START_CONCURRENCY="${DOCKER_START_CAP}"
if [[ "${DOCKER_HOST_VALUE}" == unix://* ]]; then
  export TB2_UV_CACHE_BIND_MOUNT="${TB2_UV_CACHE_BIND_MOUNT:-1}"
  export TB2_UV_CACHE_REMOTE_DIR="${TB2_UV_CACHE_REMOTE_DIR:-/data/cache/tb2_uv_cache/tb2-uv}"
fi

echo "RUN_GROUP=${RUN_GROUP}"
echo "EVAL_ROOT=${EVAL_ROOT}"
echo "MODEL=${MODEL}"
echo "API_BASE=${OPENAI_API_BASE}"
echo "DOCKER_HOST=${DOCKER_HOST}"
echo "WORKERS=${WORKERS} DOCKER_START_CAP=${DOCKER_START_CAP}"
echo "ALLOW_CONCURRENT_CLAW=${ALLOW_CONCURRENT_CLAW}"
echo "TRAIN_PARQUET=${TRAIN_PARQUET}"
echo "EVAL_PARQUET=${EVAL_PARQUET}"
echo "TB2_UV_CACHE_BIND_MOUNT=${TB2_UV_CACHE_BIND_MOUNT:-}"
echo "TB2_UV_CACHE_REMOTE_DIR=${TB2_UV_CACHE_REMOTE_DIR:-}"
echo "ARMS=${ARMS}"
echo "EXECUTE=${EXECUTE}"

for ARM in ${ARMS}; do
  EVAL_FINGERPRINT=$(python3 - \
    "${PROJECT_ROOT}" "${CONFIG}" "${TRAIN_PARQUET}" "${EVAL_PARQUET}" "${ARM}" \
    "${TRIALS}" "${MAX_TURNS}" "${MAX_TIME}" "${UNIFIED_PROMPT_PROFILE}" \
    "${UNIFIED_TOOLS_SCHEMA_MODE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, config, train, evaluation, arm, trials, turns, max_time, prompt, tools = sys.argv[1:]
root = Path(root)


def resolve(raw):
    path = Path(raw)
    return path if path.is_absolute() else root / path


def digest(path):
    path = resolve(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


config_path = resolve(config)
config_data = json.loads(config_path.read_text())
retrieval = {
    bench: digest(path)
    for bench, path in sorted((config_data.get("frozen_retrieval", {}).get("files") or {}).items())
} if arm != "baseline" else {}
payload = {
    "kind": "api_full_skill_ablation_v1",
    "arm": arm,
    "trials": int(trials),
    "max_turns": int(turns),
    "max_time": int(max_time),
    "prompt_profile": prompt,
    "tools_schema": tools,
    "config": digest(config),
    "train_parquet": digest(train),
    "eval_parquet": digest(evaluation),
    "retrieval_files": retrieval,
}
print(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
PY
  )
  EVAL_ID="api-full-${ARM}-r${TRIALS}-${EVAL_FINGERPRINT:0:10}"
  ROW_ID="${MODEL//[^A-Za-z0-9_.-]/-}"
  RUN_ROOT="${EVAL_ROOT}/${EVAL_ID}/rows/${ROW_ID}"
  RUN_ID="${OWNER_EXPERIMENT_ID}_${EVAL_ID}_${ROW_ID}"
  PLAN="${RUN_ROOT}/plans/${RUN_ID}.jsonl"
  mkdir -p "${RUN_ROOT}/plans" "${RUN_ROOT}/logs/runner" "${RUN_ROOT}/reports"

  python3 - \
    "${PROJECT_ROOT}" "${OWNER_EXPERIMENT_ID}" "${EVAL_ID}" "${EVAL_FINGERPRINT}" \
    "${ROW_ID}" "${RUN_ROOT}" "${MODEL}" "${ARM}" "${TRIALS}" \
    "${TRAIN_PARQUET}" "${EVAL_PARQUET}" "${UNIFIED_PROMPT_PROFILE}" \
    "${UNIFIED_TOOLS_SCHEMA_MODE}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(root, owner, eval_id, fingerprint, row_id, row_root, model, arm, trials,
 train_parquet, eval_parquet, prompt, tools) = sys.argv[1:]
root, row_root = Path(root), Path(row_root)
owner_root = root / "experiments/rl/runs" / owner
eval_root = owner_root / "eval" / eval_id


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


now = datetime.now(timezone.utc).isoformat()
eval_path = eval_root / "eval.json"
evaluation = read(eval_path)
if evaluation and evaluation.get("eval_spec_fingerprint") != fingerprint:
    raise SystemExit(f"eval id collision: {eval_path}")
rows = {item["row_id"]: item for item in evaluation.get("rows", []) if item.get("row_id")}
rows[row_id] = {
    **rows.get(row_id, {}), "row_id": row_id, "model_ref": model,
    "path": str(row_root), "status": rows.get(row_id, {}).get("status", "planned"),
}
write(eval_path, {
    "schema_version": 1, "kind": "evaluation", "experiment_id": owner,
    "eval_id": eval_id, "eval_spec_id": "api_full_skill_ablation_v1",
    "eval_spec_fingerprint": fingerprint,
    "created_at": evaluation.get("created_at") or now, "updated_at": now,
    "protocol": {
        "arm": arm, "trials": int(trials), "train_parquet": train_parquet,
        "eval_parquet": eval_parquet, "prompt_profile": prompt, "tools_schema": tools,
    },
    "rows": sorted(rows.values(), key=lambda item: item["row_id"]),
})
row_path = row_root / "row.json"
row = read(row_path)
row.update({
    "schema_version": 1, "kind": "eval_row", "experiment_id": owner,
    "eval_id": eval_id, "eval_spec_fingerprint": fingerprint, "row_id": row_id,
    "model_ref": model, "status": row.get("status", "planned"),
    "created_at": row.get("created_at") or now,
})
write(row_path, row)
experiment_path = owner_root / "experiment.json"
experiment = read(experiment_path)
if experiment.get("experiment_id") != owner:
    raise SystemExit(f"invalid owner manifest: {experiment_path}")
experiment["evals"] = sorted(set(experiment.get("evals", [])) | {eval_id})
experiment["updated_at"] = now
write(experiment_path, experiment)
PY

  echo
  echo "=== arm=${ARM} run_id=${RUN_ID} ==="
  env "EXPERIMENT_ROOT=${RUN_ROOT}" \
    python3 ops/workflows/rl_eval/make_full_parquet_eval_plan.py \
      --run-id "${RUN_ID}" \
      --date "${DATE}" \
      --model "${MODEL}" \
      --config "${CONFIG}" \
      --arm "${ARM}" \
      --trials "${TRIALS}" \
      --max-turns "${MAX_TURNS}" \
      --max-time "${MAX_TIME}" \
      --docker-host "${DOCKER_HOST_VALUE}" \
      --train-parquet "${TRAIN_PARQUET}" \
      --eval-parquet "${EVAL_PARQUET}" \
      --out "${PLAN}"

  LAUNCH_ARGS=(
    --plan "${PLAN}"
    --model "${MODEL}"
    --workers "${WORKERS}"
    --per-trial-timeout-sec "${TIMEOUT}"
    --docker-wait-sec "${DOCKER_WAIT_SEC}"
    --docker-check-interval-sec "${DOCKER_CHECK_INTERVAL_SEC}"
    --api-base-override "${OPENAI_API_BASE}"
  )
  if [[ "${TASK_WINDOW}" != "0" ]]; then
    LAUNCH_ARGS+=(--task-window "${TASK_WINDOW}")
  fi
  if [[ "${RERUN_COMPLETED}" == "1" ]]; then
    LAUNCH_ARGS+=(--rerun-completed)
  fi
  if [[ "${ALLOW_CONCURRENT_CLAW}" == "1" ]]; then
    LAUNCH_ARGS+=(--allow-concurrent-claw)
  fi
  if [[ "${EXECUTE}" == "1" ]]; then
    LAUNCH_ARGS+=(--execute)
  fi

  env "EXPERIMENT_ROOT=${RUN_ROOT}" \
    python3 GeneralAgent/sft_data_collection/launch_trials.py "${LAUNCH_ARGS[@]}" \
    2>&1 | tee "${RUN_ROOT}/logs/runner/launch_${ARM}.log"

  if [[ "${EXECUTE}" == "1" ]]; then
    env "EXPERIMENT_ROOT=${RUN_ROOT}" \
      python3 GeneralAgent/sft_data_collection/data_quality_dashboard.py "${RUN_ID}" \
        --run-root "${RUN_ROOT}" \
      2>&1 | tee "${RUN_ROOT}/reports/data_quality_dashboard.log" || true
    python3 - "${RUN_ROOT}" "${EVAL_ROOT}/${EVAL_ID}/eval.json" "${ROW_ID}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

row_root, eval_path, row_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
now = datetime.now(timezone.utc).isoformat()


def update(path, callback):
    payload = json.loads(path.read_text())
    callback(payload)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


update(row_root / "row.json", lambda row: row.update(status="completed", completed_at=now))
def finish_eval(evaluation):
    for row in evaluation.get("rows", []):
        if row.get("row_id") == row_id:
            row.update(status="completed", completed_at=now)
    evaluation["updated_at"] = now
update(eval_path, finish_eval)
PY
  fi
done

echo
echo "Done. Results root: ${EVAL_ROOT}"
