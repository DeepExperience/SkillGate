#!/usr/bin/env bash
# Lightweight monitor for the 20260508 9B base eval + SFT handoff goal.
#
# It records status snapshots and performs only conservative Docker cleanup:
# running containers older than STALE_HOURS are removed, excluding registry-mirror.
# This is intentionally separate from the SFT2093 goal chain, which performs the
# post-training export/serve/eval handoff.
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SKILLRL_ROOT:-$(pwd)}}"
cd "${PROJECT_ROOT}"

DATE="${DATE:-$(date -u +%Y%m%d)}"
OUT_DIR="${OUT_DIR:-experiments/${DATE}/monitor_9b_goal/reports}"
LOG="${LOG:-${OUT_DIR}/health_monitor.log}"
SNAP_JSONL="${SNAP_JSONL:-${OUT_DIR}/health_snapshots.jsonl}"
SLEEP_SEC="${SLEEP_SEC:-600}"
STALE_HOURS="${STALE_HOURS:-12}"
DOCKER_HOST_URL="${DOCKER_HOST_URL:-unix:///tmp/local-docker-overlay2.sock}"

mkdir -p "${OUT_DIR}"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG}"
}

queue_counts_json() {
  python3 - <<'PY'
import json, pathlib
out = {}
for p in sorted(pathlib.Path("/tmp/v9_queue").glob("*_pending.txt")):
    out[p.name] = sum(1 for _ in p.open(errors="ignore")) if p.exists() else 0
print(json.dumps(out, sort_keys=True))
PY
}

result_counts_json() {
  python3 - <<'PY'
import json, pathlib
root = pathlib.Path("experiments/20260508/20260508_full_base9b_baseline_openclaw_full/results")
out = {}
if root.exists():
    for p in sorted(root.glob("**/incremental.jsonl")):
        out[str(p)] = sum(1 for line in p.open(errors="ignore") if line.strip())
print(json.dumps(out, sort_keys=True))
PY
}

sft_snapshot_json() {
  python3 - <<'PY'
import json, pathlib
p = pathlib.Path("GeneralAgent/sft_training/outputs/qwen35_9b_lora_campaign_20260508_2093_thinkwrap_4gpu_82k_5epoch_r32_liger/trainer_log.jsonl")
if not p.exists():
    print("{}")
    raise SystemExit
rows = [json.loads(x) for x in p.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
print(json.dumps(rows[-1] if rows else {}, sort_keys=True))
PY
}

tmux_has() {
  tmux has-session -t "$1" 2>/dev/null && echo true || echo false
}

docker_cleanup_stale() {
  docker -H "${DOCKER_HOST_URL}" ps --format '{{.Names}}\t{{.Status}}' \
    | python3 - "${STALE_HOURS}" <<'PY' > /tmp/monitor_9b_stale_containers.txt
import re, sys
limit = int(sys.argv[1])
for line in sys.stdin:
    if not line.strip():
        continue
    name, status = line.rstrip("\n").split("\t", 1)
    if name == "registry-mirror":
        continue
    days = re.search(r"Up (\d+) days?", status)
    hours = re.search(r"Up (\d+) hours?", status)
    if days or (hours and int(hours.group(1)) >= limit):
        print(name)
PY
  local count
  count="$(wc -l < /tmp/monitor_9b_stale_containers.txt)"
  if [[ "${count}" != "0" ]]; then
    log "cleanup stale docker containers count=${count}"
    xargs -r -a /tmp/monitor_9b_stale_containers.txt docker -H "${DOCKER_HOST_URL}" rm -f \
      >> "${OUT_DIR}/health_monitor_cleanup.log" 2>&1 || true
  fi
}

write_snapshot() {
  local ts docker_running docker_time_ms gpu_raw gpu_json queue_json result_json sft_json
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  docker_running="$(docker -H "${DOCKER_HOST_URL}" ps -q 2>/dev/null | wc -l || echo -1)"
  docker_time_ms="$(python3 - "${DOCKER_HOST_URL}" <<'PY'
import subprocess, sys, time
host = sys.argv[1]
t0 = time.time()
try:
    subprocess.run(["docker", "-H", host, "ps", "-q"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)
    print(int((time.time() - t0) * 1000))
except Exception:
    print(-1)
PY
)"
  gpu_raw="$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
  gpu_json="$(python3 - "${gpu_raw}" <<'PY'
import json
import sys

rows = []
for line in sys.argv[1].splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) == 4:
        rows.append({
            "index": int(parts[0]),
            "util": int(parts[1]),
            "mem_used": int(parts[2]),
            "mem_total": int(parts[3]),
        })
print(json.dumps(rows))
PY
  )"
  queue_json="$(queue_counts_json)"
  result_json="$(result_counts_json)"
  sft_json="$(sft_snapshot_json)"
  python3 - "${SNAP_JSONL}" "${ts}" "${docker_running}" "${docker_time_ms}" "${gpu_json}" "${queue_json}" "${result_json}" "${sft_json}" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
row = {
    "ts": sys.argv[2],
    "docker_running": int(sys.argv[3]),
    "docker_ps_ms": int(sys.argv[4]),
    "gpu": json.loads(sys.argv[5]),
    "queues": json.loads(sys.argv[6]),
    "result_rows": json.loads(sys.argv[7]),
    "sft": json.loads(sys.argv[8]),
    "tmux": {
        "base9b_suite": None,
        "sft_train": None,
        "sft2093_goal_chain": None,
    },
}
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
print(json.dumps({
    "ts": row["ts"],
    "docker_running": row["docker_running"],
    "docker_ps_ms": row["docker_ps_ms"],
    "sft_step": row["sft"].get("current_steps"),
    "sft_total": row["sft"].get("total_steps"),
    "queues": row["queues"],
}, ensure_ascii=False, sort_keys=True))
PY
}

log "monitor started sleep=${SLEEP_SEC}s stale_hours=${STALE_HOURS}"
while true; do
  docker_cleanup_stale || log "warning: docker cleanup check failed"
  summary="$(write_snapshot || true)"
  python3 ops/monitor/build_20260508_eval_table.py --date "${DATE}" >/dev/null 2>&1 || true
  log "snapshot ${summary}"
  sleep "${SLEEP_SEC}"
done
