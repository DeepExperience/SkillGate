#!/usr/bin/env bash
# Canonical post-training eval70 workflow. Each row is written to the evaluated
# model's owner experiment; z_cc_terminal_imgs receives only the derived
# cross-model report and small orchestration state.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
export ROOT
cd "${ROOT}"

usage() {
  cat <<'EOF'
Usage:
  run_eval70_checkpoint_set.sh --group NAME [options] ROW...

Rows:
  --model OWNER_EXPERIMENT LABEL HF_DIR
  --checkpoint OWNER_EXPERIMENT LABEL CHECKPOINT_ROOT ITER [best|final|none]

Core options:
  --skill-mode mixed|noskill|oracle|retrieve   Default: mixed
  --snapshot DIR                               Required for mixed/retrieve
  --manifest JSONL                             Enables oracle/misleading/read tables
  --eval-id ID                                  Optional explicit protocol id
  --task-list TSV                               Default: canonical eval70_v1 split
  --report PATH                                Default: z_cc_terminal_imgs/NAME_results.md
  --report-style full|main-only                Default: full
  --dry-run

The default topology is two local TP4 engines plus two TP4 engines on the other
live Ray GPU node. Every row is resume-safe under
experiments/rl/runs/OWNER/eval/EVAL_ID/rows/ROW_ID.
EOF
}

GROUP=""
EVAL_ID="${EVAL_ID:-}"
SKILL_MODE="${SKILL_MODE:-mixed}"
SNAPSHOT="${SNAPSHOT:-}"
MANIFEST="${MANIFEST:-}"
REPORT="${REPORT:-}"
REPORT_STYLE="${REPORT_STYLE:-full}"
REMOTE_NODE="${REMOTE_NODE:-}"
WORKERS="${WORKERS:-64}"
DOCKER_START_CAP="${DOCKER_START_CAP:-128}"
REPEATS="${REPEATS:-4}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
MEM_FRACTION="${MEM_FRACTION:-0.88}"
SEED="${SEED:-1063810697}"
DOCKER_HOST_VALUE="${DOCKER_HOST_VALUE:-unix:///tmp/local-docker-overlay2.sock}"
EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC:-43200}"
GPU_IDLE_MEM_MB="${GPU_IDLE_MEM_MB:-5000}"
CHECKPOINT_WAIT_SEC="${CHECKPOINT_WAIT_SEC:-120}"
DRY_RUN="${DRY_RUN:-0}"
TASK_LIST="${TASK_LIST:-${ROOT}/ops/workflows/rl_eval/specs/eval70_v1/tasks.tsv}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${ROOT}/datasets/rl/parquet_4bench_base_20260523/train.parquet}"
EVAL_PARQUET="${EVAL_PARQUET:-${ROOT}/datasets/rl/parquet_4bench_base_20260523/eval.parquet}"
TOOLS_SCHEMA="${TOOLS_SCHEMA:-manual_schema}"
PROMPT_PROFILE="${PROMPT_PROFILE:-openclaw_full}"

declare -a ROW_KIND=() ROW_OWNER=() ROW_LABEL=() ROW_PATH=() ROW_ITER=() ROW_ROLE=()

while (($#)); do
  case "$1" in
    --group) GROUP="$2"; shift 2 ;;
    --eval-id) EVAL_ID="$2"; shift 2 ;;
    --skill-mode) SKILL_MODE="$2"; shift 2 ;;
    --snapshot) SNAPSHOT="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --task-list) TASK_LIST="$2"; shift 2 ;;
    --train-parquet) TRAIN_PARQUET="$2"; shift 2 ;;
    --eval-parquet) EVAL_PARQUET="$2"; shift 2 ;;
    --tools-schema) TOOLS_SCHEMA="$2"; shift 2 ;;
    --prompt-profile) PROMPT_PROFILE="$2"; shift 2 ;;
    --report) REPORT="$2"; shift 2 ;;
    --report-style) REPORT_STYLE="$2"; shift 2 ;;
    --remote-node) REMOTE_NODE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --model)
      ROW_KIND+=(model); ROW_OWNER+=("$2"); ROW_LABEL+=("$3"); ROW_PATH+=("$4"); ROW_ITER+=(""); ROW_ROLE+=(none)
      shift 4
      ;;
    --checkpoint)
      role=none
      shift_count=5
      if [[ "${6:-}" == best || "${6:-}" == final || "${6:-}" == none ]]; then
        role="$6"
        shift_count=6
      elif [[ "${3,,}" == *best* ]]; then
        role=best
      elif [[ "${3,,}" == *final* || "${3,,}" == *last* ]]; then
        role=final
      fi
      ROW_KIND+=(checkpoint); ROW_OWNER+=("$2"); ROW_LABEL+=("$3"); ROW_PATH+=("$4"); ROW_ITER+=("$5"); ROW_ROLE+=("${role}")
      shift "${shift_count}"
      ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "${GROUP}" ]] || { echo "FATAL: --group is required" >&2; exit 2; }
((${#ROW_KIND[@]} > 0)) || { echo "FATAL: at least one --model or --checkpoint row is required" >&2; exit 2; }
case "${SKILL_MODE}" in mixed|noskill|oracle|retrieve) ;; *) echo "FATAL: invalid --skill-mode ${SKILL_MODE}" >&2; exit 2 ;; esac
case "${REPORT_STYLE}" in full|main-only) ;; *) echo "FATAL: invalid --report-style ${REPORT_STYLE}" >&2; exit 2 ;; esac
if [[ "${SKILL_MODE}" == mixed || "${SKILL_MODE}" == retrieve ]]; then
  [[ -n "${SNAPSHOT}" && -d "${SNAPSHOT}" ]] || { echo "FATAL: --snapshot directory is required for ${SKILL_MODE}" >&2; exit 2; }
fi
if [[ -n "${MANIFEST}" ]]; then
  [[ -f "${MANIFEST}" ]] || { echo "FATAL: manifest missing: ${MANIFEST}" >&2; exit 2; }
fi
[[ -f "${TASK_LIST}" ]] || { echo "FATAL: task list missing: ${TASK_LIST}" >&2; exit 2; }
[[ -f "${TRAIN_PARQUET}" && -f "${EVAL_PARQUET}" ]] || {
  echo "FATAL: canonical train/eval parquet missing: ${TRAIN_PARQUET}, ${EVAL_PARQUET}" >&2
  exit 2
}
for owner in "${ROW_OWNER[@]}"; do
  [[ "${owner}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$ ]] || {
    echo "FATAL: unsafe owner experiment id: ${owner}" >&2; exit 2;
  }
  owner_manifest="${ROOT}/experiments/rl/runs/${owner}/experiment.json"
  [[ -f "${owner_manifest}" ]] || {
    echo "FATAL: owner experiment does not exist: ${owner_manifest}" >&2; exit 2;
  }
done

REPORT="${REPORT:-${ROOT}/z_cc_terminal_imgs/${GROUP}_results.md}"
CONTROL_ROOT="${CONTROL_ROOT:-${ROOT}/z_cc_terminal_imgs/.eval_queues/${GROUP}}"
ORIGIN_HF_DIR="${ORIGIN_HF_DIR:-${ROOT}/models/Qwen3.5-9B}"
LOG="${CONTROL_ROOT}/logs/queue_$(date -u +%Y%m%d_%H%M%S).log"

abs_path() {
  [[ "$1" = /* ]] && printf '%s\n' "$1" || printf '%s/%s\n' "${ROOT}" "$1"
}

safe_name() {
  printf '%s' "$1" | tr -c '[:alnum:]_.-' '-' | sed 's/^-*//;s/-*$//' | cut -c1-72
}

compute_eval_identity() {
  EVAL_SPEC_FINGERPRINT=$(python3 - \
    "${SKILL_MODE}" "${TASK_LIST}" "${SNAPSHOT}" "${MANIFEST}" \
    "${REPEATS}" "${SEED}" "${TOOLS_SCHEMA}" "${PROMPT_PROFILE}" \
    "${CONTEXT_LENGTH}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def hash_path(raw):
    if not raw:
        return ""
    root = Path(raw)
    h = hashlib.sha256()
    if root.is_file():
        paths = [root]
        base = root.parent
    elif root.is_dir():
        paths = sorted(path for path in root.rglob("*") if path.is_file())
        base = root
    else:
        raise SystemExit(f"fingerprint input missing: {root}")
    for path in paths:
        h.update(str(path.relative_to(base)).encode())
        h.update(str(path.stat().st_size).encode())
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()


mode, tasks, snapshot, manifest, repeats, seed, schema, prompt, context = sys.argv[1:]
payload = {
    "eval_spec": "eval70_v1",
    "skill_mode": mode,
    "task_list_sha256": hash_path(tasks),
    "snapshot_sha256": hash_path(snapshot),
    "manifest_sha256": hash_path(manifest),
    "repeats": int(repeats),
    "seed": str(seed),
    "tools_schema": schema,
    "prompt_profile": prompt,
    "context_length": int(context),
}
print(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
PY
  )
  EVAL_ID="${EVAL_ID:-eval70-${SKILL_MODE}-r${REPEATS}-${EVAL_SPEC_FINGERPRINT:0:10}}"
  export EVAL_ID EVAL_SPEC_FINGERPRINT
}

row_root_for() {
  local owner="$1" row_id="$2"
  printf '%s/experiments/rl/runs/%s/eval/%s/rows/%s\n' \
    "${ROOT}" "${owner}" "${EVAL_ID}" "${row_id}"
}

register_eval_row() {
  local owner="$1" row_id="$2" label="$3" model_ref="$4" run_root="$5"
  python3 - \
    "${ROOT}" "${owner}" "${EVAL_ID}" "${EVAL_SPEC_FINGERPRINT}" \
    "${row_id}" "${label}" "${model_ref}" "${run_root}" \
    "${SKILL_MODE}" "${TASK_LIST}" "${SNAPSHOT}" "${MANIFEST}" \
    "${REPEATS}" "${SEED}" "${TOOLS_SCHEMA}" "${PROMPT_PROFILE}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(root, owner, eval_id, fingerprint, row_id, label, model_ref, run_root,
 mode, task_list, snapshot, skill_manifest, repeats, seed, tools_schema,
 prompt_profile) = sys.argv[1:]
root = Path(root)
owner_dir = root / "experiments/rl/runs" / owner
eval_dir = owner_dir / "eval" / eval_id
row_dir = Path(run_root)
row_dir.mkdir(parents=True, exist_ok=True)


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


now = datetime.now(timezone.utc).isoformat()
eval_path = eval_dir / "eval.json"
evaluation = read(eval_path)
if evaluation and evaluation.get("eval_spec_fingerprint") != fingerprint:
    raise SystemExit(f"eval id collision with different fingerprint: {eval_path}")
rows = {item["row_id"]: item for item in evaluation.get("rows", []) if isinstance(item, dict) and item.get("row_id")}
rows[row_id] = {"row_id": row_id, "label": label, "model_ref": model_ref, "path": str(row_dir), "status": "running"}
evaluation = {
    "schema_version": 1,
    "kind": "evaluation",
    "experiment_id": owner,
    "eval_id": eval_id,
    "eval_spec_id": "eval70_v1",
    "eval_spec_fingerprint": fingerprint,
    "created_at": evaluation.get("created_at") or now,
    "updated_at": now,
    "protocol": {
        "skill_mode": mode,
        "task_list": task_list,
        "snapshot": snapshot,
        "skill_manifest": skill_manifest,
        "repeats": int(repeats),
        "seed": seed,
        "tools_schema": tools_schema,
        "prompt_profile": prompt_profile,
    },
    "rows": sorted(rows.values(), key=lambda item: item["row_id"]),
}
write(eval_path, evaluation)
write(row_dir / "row.json", {
    "schema_version": 1,
    "kind": "eval_row",
    "experiment_id": owner,
    "eval_id": eval_id,
    "eval_spec_fingerprint": fingerprint,
    "row_id": row_id,
    "label": label,
    "model_ref": model_ref,
    "status": "running",
    "created_at": now,
})

experiment_path = owner_dir / "experiment.json"
experiment = read(experiment_path)
if experiment.get("experiment_id") != owner:
    raise SystemExit(f"invalid owner experiment manifest: {experiment_path}")
experiment["evals"] = sorted(set(experiment.get("evals", [])) | {eval_id})
experiment["updated_at"] = now
write(experiment_path, experiment)
PY
}

finalize_eval_row() {
  local owner="$1" row_id="$2" run_root="$3"
  python3 - "${ROOT}" "${owner}" "${EVAL_ID}" "${row_id}" "${run_root}" "${REPEATS}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root, owner, eval_id, row_id, run_root, repeats = sys.argv[1:]
root, row_dir = Path(root), Path(run_root)
repeats = int(repeats)
sys.path.insert(0, str(root / "ops/workflows/rl_eval"))
from analyze_eval70_3tables import analyze, collect

trials = collect(str(row_dir))
metrics = analyze(trials)
metrics.update({"records": len(trials), "complete": len(trials) == 70 * repeats})
status = "completed" if metrics["complete"] else "incomplete"


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def write(path, payload):
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


write(row_dir / "metrics.json", metrics)
row_path = row_dir / "row.json"
row = read(row_path)
row.update({"status": status, "completed_at": datetime.now(timezone.utc).isoformat(), "records": len(trials)})
write(row_path, row)
eval_path = root / "experiments/rl/runs" / owner / "eval" / eval_id / "eval.json"
evaluation = read(eval_path)
for item in evaluation.get("rows", []):
    if item.get("row_id") == row_id:
        item.update({"status": status, "records": len(trials), "metrics": str(row_dir / "metrics.json")})
evaluation["updated_at"] = datetime.now(timezone.utc).isoformat()
write(eval_path, evaluation)
if not metrics["complete"]:
    raise SystemExit(
        f"eval row incomplete after finalization: records={len(trials)} expected={70 * repeats}"
    )
PY
}

mark_eval_row_failed() {
  local owner="$1" row_id="$2" run_root="$3" rc="$4"
  python3 - "${ROOT}" "${owner}" "${EVAL_ID}" "${row_id}" "${run_root}" "${rc}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root, owner, eval_id, row_id, run_root, return_code = sys.argv[1:]
root, row_dir = Path(root), Path(run_root)
now = datetime.now(timezone.utc).isoformat()


def update(path, callback):
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        return
    callback(payload)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


update(
    row_dir / "row.json",
    lambda row: row.update(status="failed", failed_at=now, return_code=int(return_code)),
)
eval_path = root / "experiments/rl/runs" / owner / "eval" / eval_id / "eval.json"
def fail_eval(evaluation):
    for row in evaluation.get("rows", []):
        if row.get("row_id") == row_id:
            row.update(status="failed", failed_at=now, return_code=int(return_code))
    evaluation["updated_at"] = now
update(eval_path, fail_eval)
PY
}

resolve_remote_node() {
  [[ -n "${REMOTE_NODE}" ]] && return 0
  REMOTE_NODE=$(/usr/bin/python3 - <<'PY'
import ray
ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
local = ray.util.get_node_ip_address()
nodes = sorted({
    item["NodeManagerAddress"]
    for item in ray.nodes()
    if item.get("Alive") and float((item.get("Resources") or {}).get("GPU", 0) or 0) > 0
})
other = [node for node in nodes if node != local]
if len(nodes) != 2 or len(other) != 1:
    raise SystemExit(f"expected two GPU nodes with one remote peer; local={local}, nodes={nodes}")
print(other[0])
ray.shutdown()
PY
  )
}

checkpoint_dir() {
  printf '%s/iter_%07d\n' "$(abs_path "$1")" "$2"
}

wait_for_checkpoints() {
  local idx dir
  for idx in "${!ROW_KIND[@]}"; do
    [[ "${ROW_KIND[$idx]}" == checkpoint ]] || continue
    dir=$(checkpoint_dir "${ROW_PATH[$idx]}" "${ROW_ITER[$idx]}")
    until validate_checkpoint "${dir}" "${ROW_ITER[$idx]}"; do
      echo "[wait-checkpoint] ${ROW_LABEL[$idx]} waiting for complete checkpoint ${dir}"
      sleep "${CHECKPOINT_WAIT_SEC}"
    done
    echo "[wait-checkpoint] ready ${dir}"
  done
}

validate_checkpoint() {
  local dir="$1" expected_iter="$2"
  python3 - "${dir}" "${expected_iter}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_iter = int(sys.argv[2])
marker = root / ".relax_complete.json"
try:
    record = json.loads(marker.read_text(encoding="utf-8"))
except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"[checkpoint-incomplete] invalid or missing {marker}: {exc}")
if int(record.get("iteration", -1)) != expected_iter:
    raise SystemExit(
        f"[checkpoint-incomplete] marker iteration={record.get('iteration')} expected={expected_iter}: {marker}"
    )
files = record.get("files")
if not isinstance(files, dict) or not files:
    raise SystemExit(f"[checkpoint-incomplete] empty files map: {marker}")
required = {".metadata", "common.pt", *(f"__{idx}_0.distcp" for idx in range(8))}
missing_names = sorted(required - set(files))
if missing_names:
    raise SystemExit(f"[checkpoint-incomplete] marker lacks required files: {missing_names}")
for name, expected_size in files.items():
    if Path(name).name != name:
        raise SystemExit(f"[checkpoint-incomplete] unsafe marker filename: {name!r}")
    path = root / name
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise SystemExit(f"[checkpoint-incomplete] missing {path}: {exc}")
    if actual_size != int(expected_size):
        raise SystemExit(
            f"[checkpoint-incomplete] size mismatch {path}: actual={actual_size} expected={expected_size}"
        )
metadata_sha = record.get("metadata_sha256")
if metadata_sha:
    actual_sha = hashlib.sha256((root / ".metadata").read_bytes()).hexdigest()
    if actual_sha != metadata_sha:
        raise SystemExit(
            f"[checkpoint-incomplete] .metadata sha256 mismatch: actual={actual_sha} expected={metadata_sha}"
        )
print(f"[checkpoint-ok] iter={expected_iter} files={len(files)} root={root}")
PY
}

validate_hf_model() {
  local model_dir="$1" expected_input="${2:-}"
  python - "${model_dir}" "${expected_input}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
expected_input = sys.argv[2]
try:
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"[hf-invalid] invalid config/index under {root}: {exc}")
if config.get("model_type") != "qwen3_5":
    raise SystemExit(f"[hf-invalid] unexpected model_type={config.get('model_type')!r}: {root}")
weight_map = index.get("weight_map")
if not isinstance(weight_map, dict) or not weight_map:
    raise SystemExit(f"[hf-invalid] empty weight_map: {root}")
shards = sorted(set(weight_map.values()))
for shard in shards:
    path = root / shard
    if Path(shard).name != shard or not path.is_file() or path.stat().st_size <= 8:
        raise SystemExit(f"[hf-invalid] missing/truncated shard: {path}")
try:
    from safetensors import safe_open
except ImportError as exc:
    raise SystemExit(f"[hf-invalid] safetensors unavailable: {exc}")
for shard in shards:
    expected_keys = {key for key, filename in weight_map.items() if filename == shard}
    try:
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
    except Exception as exc:
        raise SystemExit(f"[hf-invalid] cannot open {root / shard}: {exc}")
    if actual_keys != expected_keys:
        raise SystemExit(
            f"[hf-invalid] key mismatch {shard}: actual={len(actual_keys)} expected={len(expected_keys)}"
        )
if expected_input:
    source_path = root / "export_source.json"
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[hf-invalid] missing/invalid export lineage {source_path}: {exc}")
    actual_input = Path(str(source.get("input_ckpt_dir", ""))).resolve()
    if actual_input != Path(expected_input).resolve() or source.get("mode") != "bridge":
        raise SystemExit(
            f"[hf-invalid] lineage mismatch: input={actual_input} mode={source.get('mode')!r}; "
            f"expected_input={Path(expected_input).resolve()} mode='bridge'"
        )
print(f"[hf-ok] weights={len(weight_map)} shards={len(shards)} root={root}")
PY
}

cluster_gpu_max_mem() {
  /usr/bin/python3 - <<'PY'
import ray
ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
nodes = sorted({
    item["NodeManagerAddress"]
    for item in ray.nodes()
    if item.get("Alive") and float((item.get("Resources") or {}).get("GPU", 0) or 0) > 0
})

def probe_for(ip):
    @ray.remote(num_cpus=0.01, resources={f"node:{ip}": 0.001})
    def probe():
        import subprocess
        text = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=15,
        )
        return max(int(line.strip()) for line in text.splitlines() if line.strip())
    return probe

values = ray.get([probe_for(ip).remote() for ip in nodes], timeout=30)
print(max(values))
ray.shutdown()
PY
}

wait_gpu_idle() {
  local used
  echo "[wait-gpu] waiting for all Ray GPU nodes <= ${GPU_IDLE_MEM_MB} MB"
  while true; do
    used=$(cluster_gpu_max_mem || echo 999999)
    if (( used <= GPU_IDLE_MEM_MB )); then
      echo "[wait-gpu] cluster idle; max_mem=${used}MB"
      return 0
    fi
    echo "[wait-gpu] max_mem=${used}MB; sleep ${CHECKPOINT_WAIT_SEC}"
    sleep "${CHECKPOINT_WAIT_SEC}"
  done
}

export_hf() {
  local owner="$1" label="$2" root="$3" iter="$4" role="$5" input out export_id tmp lock
  input=$(checkpoint_dir "${root}" "${iter}")
  export_id="iter-$(printf '%07d' "${iter}")-$(printf '%s' "$(realpath -m "${input}")" | sha256sum | cut -c1-8)"
  out="${ROOT}/experiments/rl/runs/${owner}/model/exports/${export_id}"
  lock="${ROOT}/experiments/rl/runs/${owner}/model/.export-${export_id}.lock"
  mkdir -p "$(dirname "${out}")"
  if ! validate_hf_model "${out}" "${input}" >&2; then
    exec {export_lock_fd}>"${lock}"
    flock "${export_lock_fd}"
    if ! validate_hf_model "${out}" "${input}" >&2; then
      tmp="$(dirname "${out}")/.${export_id}.tmp-${BASHPID}"
      if [[ -e "${tmp}" ]]; then
        echo "FATAL: stale export temp exists: ${tmp}" >&2
        return 2
      fi
      echo "[export] ${label}: ${input} -> ${out}" >&2
      PYTHONPATH="${ROOT}/Relax${PYTHONPATH:+:${PYTHONPATH}}" \
        python "${ROOT}/ops/workflows/rl_eval/convert_cp2_qwen35_hf.py" \
        --input-dir "${input}" --origin-hf-dir "${ORIGIN_HF_DIR}" \
        --output-dir "${tmp}" --mode bridge >&2
      validate_hf_model "${tmp}" "${input}" >&2
      python3 - "${tmp}" "${owner}" "${export_id}" "${iter}" "${input}" "${label}" "${role}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path, owner, export_id, iteration, checkpoint, label, role = sys.argv[1:]
path = Path(path)
(path / "export.json").write_text(json.dumps({
    "schema_version": 1,
    "experiment_id": owner,
    "export_id": export_id,
    "iteration": int(iteration),
    "input_checkpoint": checkpoint,
    "label": label,
    "selection_role": role,
    "created_at": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
(path / "COMPLETE").write_text("validated\n")
PY
      mv "${tmp}" "${out}"
    fi
    flock -u "${export_lock_fd}"
  fi
  validate_hf_model "${out}" "${input}" >&2
  python3 - "${ROOT}" "${owner}" "${export_id}" "${out}" "${role}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root, owner, export_id, out, role = sys.argv[1:]
root = Path(root)
model_dir = root / "experiments/rl/runs" / owner / "model"
selected_path = model_dir / "selected.json"
try:
    selected = json.loads(selected_path.read_text())
except (OSError, ValueError, TypeError, json.JSONDecodeError):
    selected = {}
selected.update({
    "schema_version": 1,
    "experiment_id": owner,
    "exports": sorted(set(selected.get("exports", [])) | {export_id}),
    "updated_at": datetime.now(timezone.utc).isoformat(),
})
if role in {"best", "final"}:
    selected[f"{role}_hf"] = export_id
    alias = model_dir / f"{role}_hf"
    alias_tmp = model_dir / f".{role}_hf.tmp-{os.getpid()}"
    try:
        alias_tmp.unlink()
    except FileNotFoundError:
        pass
    alias_tmp.symlink_to(Path("exports") / export_id)
    os.replace(alias_tmp, alias)
tmp = selected_path.with_name(f".{selected_path.name}.tmp-{os.getpid()}")
tmp.write_text(json.dumps(selected, indent=2) + "\n")
os.replace(tmp, selected_path)
PY
  printf '%s\n' "${out}"
}

remote_stop() {
  [[ -n "${REMOTE_NODE}" ]] || return 0
  /usr/bin/python3 "${ROOT}/ops/workflows/rl_eval/ray_remote_sglang.py" --action stop \
    --target-node "${REMOTE_NODE}" --model-path "" --served-name unused \
    --log-dir "${CONTROL_ROOT}/logs" || true
  REMOTE_OWNED=0
}

REMOTE_OWNED=0

cleanup_remote() {
  ((REMOTE_OWNED == 0)) || remote_stop
}

remote_launch() {
  local model_path="$1" served="$2" current port deadline ready=0
  for port in 30000 30001; do
    current=$(curl -s --max-time 5 "http://${REMOTE_NODE}:${port}/v1/models" 2>/dev/null \
      | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true)
    [[ "${current}" == "${served}" ]] && ready=$((ready + 1))
  done
  if ((ready != 2)); then
    remote_stop
    sleep 10
    REMOTE_OWNED=1
    /usr/bin/python3 "${ROOT}/ops/workflows/rl_eval/ray_remote_sglang.py" --action launch \
      --target-node "${REMOTE_NODE}" --model-path "${model_path}" --served-name "${served}" \
      --tp-size 4 --context-length "${CONTEXT_LENGTH}" --mem-fraction "${MEM_FRACTION}" \
      --seed "${SEED}" --engine 0,1,2,3:30000 --engine 4,5,6,7:30001 \
      --log-dir "${CONTROL_ROOT}/logs"
  fi
  REMOTE_OWNED=1
  deadline=$((SECONDS + 1800))
  for port in 30000 30001; do
    until curl -s --max-time 5 "http://${REMOTE_NODE}:${port}/v1/models" | grep -q "${served}"; do
      ((SECONDS <= deadline)) || { echo "FATAL: remote SGLang port ${port} did not become ready" >&2; return 2; }
      sleep 20
    done
  done
}

declare -a TABLE_ARGS=()

validate_eval_row() {
  local run_root="$1"
  python3 - "${ROOT}" "${run_root}" "${REPEATS}" <<'PY'
import sys
from collections import Counter
from pathlib import Path

repo, run_root, repeats = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
sys.path.insert(0, str(repo / "ops" / "workflows" / "rl_eval"))
from analyze_eval70_3tables import collect

trials = collect(str(run_root))
expected_records = 70 * repeats
if len(trials) != expected_records:
    raise SystemExit(f"[row-incomplete] records={len(trials)} expected={expected_records}: {run_root}")
counts = Counter((trial["bench"], trial["task"]) for trial in trials)
if len(counts) != 70 or set(counts.values()) != {repeats}:
    bad = sorted((bench, task, count) for (bench, task), count in counts.items() if count != repeats)
    raise SystemExit(f"[row-incomplete] tasks={len(counts)} expected=70 bad_repeat_counts={bad[:8]}")
expected_benches = {
    "claw": 14 * repeats,
    "sb_ns": 8 * repeats,
    "seta": 30 * repeats,
    "swe": 10 * repeats,
    "tb2": 8 * repeats,
}
actual_benches = Counter(trial["bench"] for trial in trials)
if dict(actual_benches) != expected_benches:
    raise SystemExit(f"[row-incomplete] bench counts={dict(actual_benches)} expected={expected_benches}")
missing_trajectories = sum(not trial["has_traj"] for trial in trials)
if missing_trajectories:
    raise SystemExit(f"[row-incomplete] parsed trajectories missing={missing_trajectories}: {run_root}")
print(f"[row-ok] records={len(trials)} tasks={len(counts)} repeats={repeats} root={run_root}")
PY
}

render_report() {
  local tables="${CONTROL_ROOT}/reports_combined.md"
  local reads="${CONTROL_ROOT}/reports_slate_reads.md"
  local table_context="${SKILL_MODE} skills"
  [[ "${SKILL_MODE}" == noskill ]] && table_context="no skill"
  mkdir -p "${CONTROL_ROOT}"
  EVAL70_TABLE_CONTEXT="${table_context}" EVAL70_TABLE_STYLE="${REPORT_STYLE}" \
    python3 "${ROOT}/ops/workflows/rl_eval/format_eval70_zcc.py" "${TABLE_ARGS[@]}" >"${tables}"
  if [[ -n "${MANIFEST}" && "${REPORT_STYLE}" == full ]]; then
    EVAL70_DUMP_TRIALS_DIR="${CONTROL_ROOT}" \
      python3 "${ROOT}/ops/workflows/rl_eval/analyze_slate_reads.py" \
        --manifest "${MANIFEST}" --out "${reads}" "${TABLE_ARGS[@]}"
  fi
  {
    echo "# ${GROUP}"
    echo
    echo "- condition: ${SKILL_MODE}; repeats: ${REPEATS}; workers: ${WORKERS}"
    echo "- eval id: \`${EVAL_ID}\`; protocol fingerprint: \`${EVAL_SPEC_FINGERPRINT}\`"
    [[ -n "${SNAPSHOT}" ]] && echo "- skill snapshot: \`${SNAPSHOT}\`"
    echo "- serving: 4 x TP4 across the two Ray GPU nodes"
    echo
    echo "## Outcome tables"
    echo
    cat "${tables}" 2>/dev/null || true
    if [[ "${REPORT_STYLE}" == full && -f "${reads}" ]]; then
      echo
      echo "## Per-method skill-read behavior"
      echo
      sed '1{/^# Mixed-slate per-category read attribution$/d;}' "${reads}"
    fi
  } >"${REPORT}"
  echo "[report] ${REPORT}"
}

run_row() {
  local owner="$1" label="$2" model_path="$3" model_ref="$4" safe served row_id run_root
  safe=$(safe_name "${label}")
  row_id="${safe}-$(printf '%s' "${model_ref}" | sha256sum | cut -c1-8)"
  served="qwen3.5-9b-${safe}"
  run_root=$(row_root_for "${owner}" "${row_id}")
  register_eval_row "${owner}" "${row_id}" "${label}" "${model_ref}" "${run_root}"
  TABLE_ARGS+=("${label}=${run_root}")
  if [[ -f "${run_root}/ROW_DONE" ]]; then
    if validate_eval_row "${run_root}"; then
      echo "[row] ${label} already complete"
      finalize_eval_row "${owner}" "${row_id}" "${run_root}"
      return 0
    fi
    echo "[row] removing stale incomplete marker: ${run_root}/ROW_DONE" >&2
    rm -f "${run_root}/ROW_DONE"
  fi
  if validate_eval_row "${run_root}" >/dev/null 2>&1; then
    echo "[row] ${label} artifacts are complete; recovering ROW_DONE without rerunning"
    render_report
    touch "${run_root}/ROW_DONE"
    finalize_eval_row "${owner}" "${row_id}" "${run_root}"
    return 0
  fi
  remote_launch "${model_path}" "${served}"
  set +e
  python3 "${ROOT}/ops/workflows/rl_eval/run_eval70_model.py" \
    --run-id "${owner}_${EVAL_ID}_${row_id}" --label "${label}-${SKILL_MODE}" --run-root "${run_root}" \
    --owner-experiment "${owner}" --eval-id "${EVAL_ID}" --row-id "${row_id}" \
    --intent "canonical checkpoint-set eval70 under identical ${SKILL_MODE} skill condition" \
    --model-path "${model_path}" --served-name "${served}" --tools-schema "${TOOLS_SCHEMA}" \
    --prompt-profile "${PROMPT_PROFILE}" --task-list "${TASK_LIST}" \
    --train-parquet "${TRAIN_PARQUET}" --eval-parquet "${EVAL_PARQUET}" \
    --skill-mode "${SKILL_MODE}" ${SNAPSHOT:+--retrieval-root "${SNAPSHOT}"} \
    --context-length "${CONTEXT_LENGTH}" --mem-fraction "${MEM_FRACTION}" --seed "${SEED}" \
    --workers "${WORKERS}" --docker-start-cap "${DOCKER_START_CAP}" --repeats "${REPEATS}" \
    --concurrent-trials --expected-records $((70 * REPEATS)) \
    --engine 0,1,2,3:30000 --engine 4,5,6,7:30001 \
    --router-worker-url "http://${REMOTE_NODE}:30000" --router-worker-url "http://${REMOTE_NODE}:30001" \
    --router-policy round_robin --docker-host "${DOCKER_HOST_VALUE}" --min-images 500 \
    --no-proxy "127.0.0.1,localhost,0.0.0.0,${REMOTE_NODE}" \
    --start-guards --eval-timeout-sec "${EVAL_TIMEOUT_SEC}" --execute
  eval_rc=$?
  set -e
  remote_stop
  if ((eval_rc != 0)); then
    mark_eval_row_failed "${owner}" "${row_id}" "${run_root}" "${eval_rc}"
    return "${eval_rc}"
  fi
  validate_eval_row "${run_root}"
  if [[ -n "${MANIFEST}" ]]; then
    EVAL70_DUMP_TRIALS_DIR="${run_root}/reports" \
      python3 "${ROOT}/ops/workflows/rl_eval/analyze_slate_reads.py" \
        --manifest "${MANIFEST}" --out "${run_root}/reports/slate_reads.md" \
        "${label}=${run_root}"
  fi
  render_report
  touch "${run_root}/ROW_DONE"
  finalize_eval_row "${owner}" "${row_id}" "${run_root}"
}

compute_eval_identity
echo "[config] group=${GROUP} eval_id=${EVAL_ID} condition=${SKILL_MODE} rows=${#ROW_KIND[@]}"
for idx in "${!ROW_KIND[@]}"; do
  echo "[row-config] owner=${ROW_OWNER[$idx]} ${ROW_KIND[$idx]} label=${ROW_LABEL[$idx]} path=${ROW_PATH[$idx]} iter=${ROW_ITER[$idx]:-n/a} role=${ROW_ROLE[$idx]}"
done

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_OK eval70 checkpoint set"
  exit 0
fi

resolve_remote_node
echo "[topology] remote=${REMOTE_NODE}"

mkdir -p "${CONTROL_ROOT}/logs" "$(dirname "${REPORT}")"
# shellcheck source=/dev/null
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source ${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/etc/profile.d/conda.sh
conda activate slime
export CUDA_HOME=/usr/local/cuda-12.9 CUDA_PATH=/usr/local/cuda-12.9 PYTHONUNBUFFERED=1
# Relax checkpoints pickle launcher/config classes from the local package.  The
# HF converter must be able to import them while loading common.pt.
export PYTHONPATH="${ROOT}/Relax:/root/Megatron-LM:${PYTHONPATH:-}"
export DOCKER_HOST="${DOCKER_HOST_VALUE}" DOCKER_HOST_VALUE
export NO_PROXY="127.0.0.1,localhost,0.0.0.0,${REMOTE_NODE}"
export no_proxy="${NO_PROXY}"
trap cleanup_remote EXIT INT TERM

{
  echo "[queue-start] $(date -u +%FT%TZ)"
  wait_for_checkpoints
  wait_gpu_idle
  for idx in "${!ROW_KIND[@]}"; do
    if [[ "${ROW_KIND[$idx]}" == checkpoint ]]; then
      model=$(export_hf "${ROW_OWNER[$idx]}" "${ROW_LABEL[$idx]}" "${ROW_PATH[$idx]}" "${ROW_ITER[$idx]}" "${ROW_ROLE[$idx]}")
      model_ref="$(basename "${model}")"
    else
      model=$(abs_path "${ROW_PATH[$idx]}")
      validate_hf_model "${model}"
      model_ref="$(basename "$(realpath -m "${model}")")"
    fi
    run_row "${ROW_OWNER[$idx]}" "${ROW_LABEL[$idx]}" "${model}" "${model_ref}"
  done
  render_report
  echo "[queue-done] $(date -u +%FT%TZ)"
} 2>&1 | tee -a "${LOG}"
