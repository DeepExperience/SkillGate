#!/usr/bin/env bash
# Usage: bash run_dynamic_bench.sh <bench> <arm> [parallel]
#   bench    ∈ {seta, swe, tb2, sb_ns, claw}
#   arm      ∈ {baseline, retrieval, irrelevant}
#   parallel default 8
#
# Dynamic worker pool (work-stealing) — fixes the "N static shards, tail worker
# runs alone" problem. N workers pull from a shared task queue file via flock;
# when one worker finishes a task it atomically pops the next pending task.
# No idle time until <N tasks remain.
#
# For harbor (tb2/seta/sb_ns) + swe: launches N tmux sessions, each running a
# steal-loop that calls the runner with --task / --instance (singular) per pop.
# For claw: the runner already has internal --parallel N; we just pass that.
#
# Retrieval jsonl lives at `experiments/<DATE>/<DATE>_<suffix>/retrieval_results/<bench>.jsonl`
# Env override: RETR_SUFFIX (default v8_4stage). Or pass your own via RETR_JSONL_OVERRIDE.
#
# Output: `experiments/<DATE>/<RUN_ID>/results/<bench>/<EXP>_<arm>/`
# EXP slug from UNIFIED_EXP_VERSION (default v9).

set -uo pipefail
BENCH="${1:-}"
ARM="${2:-retrieval}"
PAR="${3:-8}"
if [[ -z "$BENCH" || ! "$ARM" =~ ^(baseline|retrieval|irrelevant)$ ]]; then
    echo "Usage: $0 <seta|swe|tb2|sb_ns|claw> <baseline|retrieval|irrelevant> [parallel=8]"
    exit 1
fi

PROJ=${SKILLRL_ROOT:-$(pwd)}
DATE=$(date +%Y%m%d)
OUT_DATE="${RESULT_DATE:-$DATE}"
RETR_DATE="${RETRIEVAL_DATE:-$DATE}"
RETR_SUFFIX=${RETR_SUFFIX:-v8_4stage}
RETR_RUN_ID="${RETR_DATE}_${RETR_SUFFIX}"
RETR_DIR="${PROJ}/experiments/${RETR_DATE}/${RETR_RUN_ID}/retrieval_results"
[[ "$BENCH" == "sb_ns" ]] && RETR_BENCH="skillsbench" || RETR_BENCH="$BENCH"
RETR="${RETR_JSONL_OVERRIDE:-${RETR_DIR}/${RETR_BENCH}.jsonl}"
EXP_VERSION="${UNIFIED_EXP_VERSION:-${PHASE_B_EXP:-}}"
MODEL_NAME="${UNIFIED_MODEL:-${PHASE_B_MODEL:-}}"
if [[ -z "$MODEL_NAME" ]]; then
    MODEL_HINT="${EXP_VERSION:-}"
    case "$MODEL_HINT" in
        *9b*|*9B*) MODEL_NAME="qwen3.5-9b" ;;
        *27b*|*27B*) MODEL_NAME="qwen3.5-27b" ;;
    esac
fi
if [[ "$BENCH" == "sb_ns" ]]; then
    OUTPUT_BENCH="skillsbench-no-skills"
else
    OUTPUT_BENCH="$BENCH"
fi
API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:30000/v1}"
API_MODELS_URL="${API_BASE%/}/models"
EVAL_MAX_TURNS="${UNIFIED_DEFAULT_MAX_TURNS:-30}"
EVAL_MAX_TIME="${UNIFIED_ROLLOUT_WALLCLOCK_CAP_SEC:-850}"

# Preflight
SERVED_MODEL="$(curl -sS --max-time 5 --noproxy '*' "${API_MODELS_URL}" 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["data"][0]["id"])' 2>/dev/null || true)"
[[ -n "$SERVED_MODEL" ]] \
    || { echo "ABORT: SGLang not reachable"; exit 1; }
if [[ -z "$MODEL_NAME" ]]; then
    MODEL_NAME="$SERVED_MODEL"
fi
if [[ -z "$EXP_VERSION" ]]; then
    case "$MODEL_NAME" in
        *9b*|*9B*) EXP_VERSION="v9_9b" ;;
        *) EXP_VERSION="v9" ;;
    esac
fi
RUN_ID="${UNIFIED_RUN_ID:-${RUN_ID:-${OUT_DATE}_${EXP_VERSION}}}"
RUN_ROOT="${EXPERIMENT_ROOT:-${RUN_ROOT:-experiments/${OUT_DATE}/${RUN_ID}}}"
if [[ "$SERVED_MODEL" != "$MODEL_NAME" ]]; then
    echo "ABORT: model label mismatch: script would use --model ${MODEL_NAME}, but SGLang serves ${SERVED_MODEL}"
    echo "       Set UNIFIED_MODEL/PHASE_B_MODEL correctly, or restart SGLang with the intended model."
    exit 1
fi
if [[ "$ARM" != "baseline" ]]; then
    [[ -f "$RETR" ]] || { echo "ABORT: retrieval jsonl missing at $RETR (run pipeline first)"; exit 1; }
fi

# v8 jsonl format compat: injector expects reranked_top10; rewrite from stage3_top10_llm_judge if needed
RETR_FOR_INJECT="$RETR"
if [[ "$ARM" != "baseline" ]]; then
    python3 - "$RETR" <<'PY'
import json, sys
src = sys.argv[1]
try:
    with open(src) as f:
        first = json.loads(next(f))
    if "reranked_top10" not in first and ("stage3_top10_llm_judge" in first or "reranked_top20" in first):
        compat = src.replace(".jsonl", ".compat_top10.jsonl")
        with open(src) as fin, open(compat, "w") as fout:
            for l in fin:
                r = json.loads(l)
                s3 = r.get("stage3_top10_llm_judge") or r.get("reranked_top20") or []
                r["reranked_top10"] = [
                    {"rank": e.get("rank_final", e.get("rank", i+1)),
                     "skill_name": e["skill_name"],
                     "skill_path": e["skill_path"],
                     "rerank_score": e.get("rerank_score", 0.0),
                     "llm_score": e.get("llm_score")} for i, e in enumerate(s3[:10])
                ]
                r["coarse_top50"] = r.get("stage1_embedding_top50") or r.get("coarse_embed_top50") or []
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(compat)
    else:
        print(src)
except Exception as e:
    print(f"ERR: {e}", file=sys.stderr); print(src)
PY
    RETR_FOR_INJECT=$(python3 -c "
import json
try:
    with open('$RETR') as f: d=json.loads(next(f))
    print('$RETR'.replace('.jsonl','.compat_top10.jsonl') if 'reranked_top10' not in d and ('stage3_top10_llm_judge' in d or 'reranked_top20' in d) else '$RETR')
except: print('$RETR')
")
fi

case "$ARM" in
    retrieval)  INJECT_FLAG="--inject-retrieval-skills ${RETR_FOR_INJECT} --retrieval-top-n 10" ;;
    irrelevant) INJECT_FLAG="--inject-irrelevant-skills ${RETR_FOR_INJECT} --retrieval-top-n 10" ;;
    baseline)   INJECT_FLAG="" ;;
esac

# Common env for worker scripts
read -r -d '' COMMON_ENV <<EOF || true
set -uo pipefail
export PATH="${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/envs/slime/bin:/root/.local/bin:\$PATH"
export DOCKER_HOST="\${DOCKER_HOST:-unix:///tmp/local-docker-overlay2.sock}"
export HTTP_PROXY="\${HTTP_PROXY:-http://your-proxy:3128}"
export HTTPS_PROXY="\${HTTPS_PROXY:-http://your-proxy:3128}"
export ALL_PROXY="\${ALL_PROXY:-http://your-proxy:3128}"
export http_proxy="\${http_proxy:-\${HTTP_PROXY}}"
export https_proxy="\${https_proxy:-\${HTTPS_PROXY}}"
export all_proxy="\${all_proxy:-\${ALL_PROXY}}"
export NO_PROXY="127.0.0.1,localhost,0.0.0.0"
export no_proxy="\${NO_PROXY}"
export OPENAI_API_KEY="sk-local-anything"
export OPENAI_API_BASE="${API_BASE}"
export HF_ENDPOINT="https://hf-mirror.com"
export UNIFIED_DISABLE_THINKING=1
export UNIFIED_PRESENCE_PENALTY=1.5
export UNIFIED_EARLY_STOP_N=3
# 2026-05-09 P0 fixes: bigger turn budget for SFT students + prefix-based loop
# detection (see agent_loop.py + run_unified_*.py). Override per run if needed.
export UNIFIED_MIN_MAX_TURNS="\${UNIFIED_MIN_MAX_TURNS:-30}"
export UNIFIED_DEFAULT_MAX_TURNS="\${UNIFIED_DEFAULT_MAX_TURNS:-30}"
export UNIFIED_ROLLOUT_WALLCLOCK_CAP_SEC="\${UNIFIED_ROLLOUT_WALLCLOCK_CAP_SEC:-850}"
export UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS="\${UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS:-1}"
export UNIFIED_PREFIX_STOP_N="\${UNIFIED_PREFIX_STOP_N:-3}"
export UNIFIED_PREFIX_CONTENT_LEN="\${UNIFIED_PREFIX_CONTENT_LEN:-120}"
export UNIFIED_PREFIX_ARGS_LEN="\${UNIFIED_PREFIX_ARGS_LEN:-80}"
export UNIFIED_LLM_REQUEST_TIMEOUT_SEC="${UNIFIED_LLM_REQUEST_TIMEOUT_SEC:-600}"
export UNIFIED_EXP_VERSION=${EXP_VERSION}
export UNIFIED_RUN_ID=${RUN_ID}
export EXPERIMENT_ROOT=${RUN_ROOT}
export UNIFIED_MODEL=${MODEL_NAME}
export PHASE_B_MODEL=${MODEL_NAME}
export UNIFIED_RESULTS_DATE=${OUT_DATE}
[ -f ${PROJ}/secrets/.env.secrets ] && source ${PROJ}/secrets/.env.secrets
cd ${PROJ}

wait_for_docker() {
    local wait_sec="\${DOCKER_WAIT_SEC:-54000}"
    local deadline=\$((SECONDS + wait_sec))
    while ! docker info >/dev/null 2>&1; do
        if (( SECONDS >= deadline )); then
            echo "[docker-wait] Docker unavailable after \${wait_sec}s; exiting without popping a task" >&2
            return 1
        fi
        echo "[docker-wait] Docker unavailable at \${DOCKER_HOST:-default}; sleeping 30s before task dispatch" >&2
        sleep 30
    done
}
EOF

QUEUE_NAMESPACE="${DYN_QUEUE_NAMESPACE:-default}"
TMUX_PREFIX="${DYN_TMUX_PREFIX:-v9}"
QDIR="/tmp/v9_queue/${QUEUE_NAMESPACE}"
mkdir -p "$QDIR"
QUEUE="${QDIR}/${BENCH}_${ARM}_pending.txt"
LOCK="${QDIR}/${BENCH}_${ARM}_pending.lock"
TAG="${BENCH}_${ARM}"     # tmux session / script suffix — harbor benches all prefixed

filter_completed_queue() {
    [[ "${DYN_SKIP_COMPLETED:-0}" == "1" ]] || return 0
    local queue="$1"
    local tmp="${queue}.tmp.$$"
    python3 - "$queue" "$RUN_ROOT" "$OUTPUT_BENCH" <<'PY'
import json
import pathlib
import sys

queue = pathlib.Path(sys.argv[1])
run_root = pathlib.Path(sys.argv[2])
bench = sys.argv[3]
completed = set()
for inc in (run_root / "results" / bench).glob("**/incremental.jsonl"):
    try:
        lines = inc.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        continue
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        task_id = row.get("task_id") or row.get("instance_id") or row.get("id") or row.get("task")
        if task_id:
            completed.add(str(task_id))
tasks = [line.strip() for line in queue.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
remaining = [task for task in tasks if task not in completed]
queue.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
print(f"[resume-skip] {bench}: completed={len(completed)} total={len(tasks)} remaining={len(remaining)}")
PY
}

filter_known_bad_queue() {
    local bench="$1"
    local queue="$2"
    local before
    before=$(wc -l < "${queue}" 2>/dev/null || echo 0)
    python3 - "${bench}" "${queue}" "${PROJ}" <<'PY'
import pathlib
import sys

bench, queue_path, project_root = sys.argv[1:4]
sys.path.insert(0, project_root)
from GeneralAgent.task_exclusions import filter_bad_tasks

bench_map = {
    "seta": "seta_synth",
    "sb_ns": "sb_ns",
    "tb2": "tb2",
    "swe": "swe_lite",
    "claw": "claw",
}
path = pathlib.Path(queue_path)
tasks = [
    line.strip()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if line.strip()
]
kept = filter_bad_tasks(bench_map.get(bench, bench), tasks)
path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
print(f"[exclude-known-bad] {bench}: removed={len(tasks) - len(kept)} kept={len(kept)}")
PY
    local after
    after=$(wc -l < "${queue}" 2>/dev/null || echo 0)
    if [[ "${before}" != "${after}" ]]; then
        echo "[exclude-known-bad] ${bench}: queue ${before} -> ${after}"
    fi
}

# Build task queue
case "$BENCH" in
    seta)
        python3 - > "$QUEUE" <<PY
from pathlib import Path
p = Path("${PROJ}/datasets/seta/dataset/seta_baseline_30")
for t in sorted(d.name for d in p.iterdir() if d.is_dir()):
    print(t)
PY
        RUNNER="python -u GeneralAgent/eval_scripts/unified_runner/run_unified_harbor.py --dataset seta --model ${MODEL_NAME} --api-base ${API_BASE} --max-turns ${EVAL_MAX_TURNS} --max-time ${EVAL_MAX_TIME} --task \$T ${INJECT_FLAG}"
        ;;
    tb2)
        python3 - > "$QUEUE" <<PY
from pathlib import Path
p = Path("${PROJ}/datasets/terminal-bench-v2")
for t in sorted(d.name for d in p.iterdir() if d.is_dir() and (d/"tests"/"test.sh").exists()):
    print(t)
PY
        RUNNER="python -u GeneralAgent/eval_scripts/unified_runner/run_unified_harbor.py --dataset tb2 --model ${MODEL_NAME} --api-base ${API_BASE} --max-turns ${EVAL_MAX_TURNS} --max-time ${EVAL_MAX_TIME} --task \$T ${INJECT_FLAG}"
        ;;
    sb_ns)
        python3 - > "$QUEUE" <<PY
from pathlib import Path
p = Path("${PROJ}/datasets/skillsbench/tasks-no-skills")
for t in sorted(d.name for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")):
    print(t)
PY
        RUNNER="python -u GeneralAgent/eval_scripts/unified_runner/run_unified_harbor.py --dataset skillsbench --variant no-skills --model ${MODEL_NAME} --api-base ${API_BASE} --max-turns ${EVAL_MAX_TURNS} --max-time ${EVAL_MAX_TIME} --task \$T ${INJECT_FLAG}"
        ;;
    swe)
        python3 - > "$QUEUE" <<PY
import importlib.util
spec = importlib.util.spec_from_file_location("m", "${PROJ}/GeneralAgent/eval_scripts/unified_runner/run_unified_swe.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for img in getattr(m, "ALL_IMAGES", []):
    iid = img.split(".")[-1].split(":")[0].replace("_s_", "__")
    print(iid)
PY
        RUNNER="python -u GeneralAgent/eval_scripts/unified_runner/run_unified_swe.py --model ${MODEL_NAME} --api-base ${API_BASE} --max-turns ${EVAL_MAX_TURNS} --max-time ${EVAL_MAX_TIME} --instance \$T ${INJECT_FLAG}"
        ;;
    claw)
        # Claw already has internal --parallel N; single tmux, pass parallel=PAR
        CLAW_TF=${DYN_CLAW_TASKS_FILE:-${PROJ}/GeneralAgent/eval_scripts/prebake_images/claw_161_t_series.txt}
        if [[ "${DYN_SKIP_COMPLETED:-0}" == "1" ]]; then
            CLAW_TF_FILTERED="/tmp/run_dyn_${TMUX_PREFIX}_claw_${ARM}_missing.txt"
            python3 - "${CLAW_TF}" "${RUN_ROOT}" "${OUTPUT_BENCH}" "${CLAW_TF_FILTERED}" <<'PY'
import json
import pathlib
import sys

src, run_root, bench, out = map(pathlib.Path, sys.argv[1:])
completed = set()
for inc in (run_root / "results" / str(bench)).glob("**/incremental.jsonl"):
    try:
        lines = inc.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        continue
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        task_id = row.get("task_id") or row.get("id") or row.get("task")
        if task_id:
            completed.add(str(task_id))
tasks = [
    line.strip()
    for line in src.read_text(encoding="utf-8", errors="ignore").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
remaining = [task for task in tasks if task not in completed]
out.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
print(f"[resume-skip] claw: completed={len(completed)} total={len(tasks)} remaining={len(remaining)}")
PY
            CLAW_TF="${CLAW_TF_FILTERED}"
        fi
        CLAW_SCRIPT="/tmp/run_dyn_${TMUX_PREFIX}_claw_${ARM}.sh"
        CLAW_SESSION="${TMUX_PREFIX}_claw_${ARM}"
cat > "${CLAW_SCRIPT}" <<CLAW_EOF
${COMMON_ENV}
export UNIFIED_CLAW_USE_DOCKER_SANDBOX=1
echo "=== ${EXP_VERSION}_claw_${ARM} START \$(date -Iseconds) (claw internal --parallel ${PAR}) ==="
wait_for_docker || exit 75
python -u GeneralAgent/eval_scripts/unified_runner/run_unified_claw.py \\
    --model ${MODEL_NAME} \\
    --api-base ${API_BASE} \\
    --tasks-file ${CLAW_TF} \\
    ${INJECT_FLAG} \\
    --parallel ${PAR} \\
    2>&1 | tee /tmp/${TMUX_PREFIX}_claw_${ARM}.log
echo "=== ${EXP_VERSION}_claw_${ARM} DONE \$(date -Iseconds) ==="
CLAW_EOF
        chmod +x "${CLAW_SCRIPT}"
        tmux new-session -d -s "${CLAW_SESSION}" "bash ${CLAW_SCRIPT}"
        sleep 3
        tmux has-session -t "${CLAW_SESSION}" && echo "claw (${ARM}) ALIVE (internal parallel=${PAR}, session=${CLAW_SESSION})" || echo "claw DEAD"
        exit 0
        ;;
    *) echo "Unknown bench: $BENCH"; exit 1 ;;
esac

filter_known_bad_queue "$BENCH" "$QUEUE"
filter_completed_queue "$QUEUE"

# Show queue size
n_tasks=$(wc -l < "$QUEUE")
echo "=== ${BENCH} ${ARM}: ${n_tasks} tasks queued; launching ${PAR} workers (work-stealing via flock ${LOCK}) ==="

# Worker script template: pop-loop via flock
# Each worker reads 1 task at a time; atomic via flock so no dup dispatches.
for w in $(seq 0 $((PAR - 1))); do
WORKER_SCRIPT="/tmp/run_dyn_${TMUX_PREFIX}_${TAG}_w${w}.sh"
WORKER_SESSION="${TMUX_PREFIX}_${TAG}_w${w}"
cat > "${WORKER_SCRIPT}" <<WORKER_EOF
${COMMON_ENV}
echo "=== ${TMUX_PREFIX}_${TAG}_w${w} START \$(date -Iseconds) ==="
while true; do
    wait_for_docker || exit 75
    # Atomically pop first line from queue
    T=\$(flock -x "${LOCK}" -c 'head -1 "${QUEUE}" 2>/dev/null; sed -i "1d" "${QUEUE}" 2>/dev/null')
    if [ -z "\$T" ]; then
        echo "=== w${w} queue empty, exiting \$(date -Iseconds) ==="
        break
    fi
    echo "--- w${w} task=\$T \$(date -Iseconds) ---"
    ${RUNNER} 2>&1
done
echo "=== ${TMUX_PREFIX}_${TAG}_w${w} DONE \$(date -Iseconds) ==="
WORKER_EOF
    chmod +x "${WORKER_SCRIPT}"
    tmux new-session -d -s "${WORKER_SESSION}" "bash ${WORKER_SCRIPT}"
done
sleep 3
alive=$(tmux ls 2>&1 | grep -c "^${TMUX_PREFIX}_${TAG}_")
echo "=== ${BENCH} ${ARM} launched: ${alive}/${PAR} tmux sessions alive ==="
echo "  Queue:  ${QUEUE}  (${n_tasks} tasks)"
echo "  Model:  ${MODEL_NAME} (served=${SERVED_MODEL})"
echo "  Retrieval: experiments/${RETR_DATE}/${RETR_RUN_ID}/retrieval_results/${RETR_BENCH}.jsonl"
echo "  Output: ${RUN_ROOT}/results/${OUTPUT_BENCH}/${EXP_VERSION}_${ARM}/"
echo "  Monitor: tmux capture-pane -t ${TMUX_PREFIX}_${TAG}_w0 -pS -20"
echo "  Remaining: wc -l \"${QUEUE}\"  (live view of pending tasks)"
