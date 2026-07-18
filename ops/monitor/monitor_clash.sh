#!/usr/bin/env bash
# 监控远程 Docker 主机代理流量消耗（适合长实验）。
#
# 用法:
#   bash ops/monitor/monitor_clash.sh              # 默认 60s 采样
#   bash ops/monitor/monitor_clash.sh 30           # 30s 采样
#   bash ops/monitor/monitor_clash.sh 60 mylog.log # 自定义 log 路径
#
# 两个输出:
#   1. log 文件: 每个 sample 一行 (JSON + 人类可读 metrics)
#   2. stdout: rate + delta + top-3 hosts (tail -f 时看这个)
#
# 总结:
#   - 每 30 min 自动 emit 一条 "=== SUMMARY ===" 显示近 30min 消耗 + top hosts
#   - 偵测异常: rate > 20 MB/s 持续 > 3 sample / 单 host 单连接 > 500MB / CN 镜像漏水 > 100MB
#
# 死法:
#   - Ctrl+C 干净退出 (trap)
#   - 启在 tmux 里最稳: tmux new -d -s clash-mon "bash ops/monitor/monitor_clash.sh"

set -uo pipefail

SAMPLE_SEC=${1:-60}
PROJECT_ROOT="${SKILLRL_ROOT:-$(pwd)}"
DATE=${DATE:-$(date +%Y%m%d)}
if [[ -n "${EXPERIMENT_ROOT:-${RUN_ROOT:-}}" ]]; then
    ROOT="${EXPERIMENT_ROOT:-${RUN_ROOT:-}}"
    if [[ "$ROOT" = /* ]]; then
        DEFAULT_LOG="${ROOT}/logs/monitor/traffic/${DATE}_clash.log"
    else
        DEFAULT_LOG="${PROJECT_ROOT}/${ROOT}/logs/monitor/traffic/${DATE}_clash.log"
    fi
else
    RUN_ID="${RUN_ID:-${DATE}_ops_monitor}"
    DEFAULT_LOG="${PROJECT_ROOT}/experiments/${DATE}/${RUN_ID}/logs/monitor/traffic/${DATE}_clash.log"
fi
LOG=${2:-$DEFAULT_LOG}
mkdir -p "$(dirname "$LOG")"

SUMMARY_INTERVAL_SEC=1800  # 30 min
WARN_RATE_MB_S=20
WARN_HOST_MB=500
WARN_CN_LEAK_MB=100

CN_MIRROR_PATTERN='aliyun|tsinghua|ustc|nju\.edu|bfsu|cernet|163\.com|huaweicloud|hf-mirror'
DOCKER_PATTERN='docker\.com|docker\.io|ghcr|quay|gcr\.io|registry'

echo "[monitor] sample=${SAMPLE_SEC}s  log=$LOG"
echo "[monitor] Ctrl+C to stop"
echo

cleanup() {
    echo
    echo "[monitor] stopping at $(date -Iseconds)"
    exit 0
}
trap cleanup INT TERM

fetch_clash() {
    ssh "${REMOTE_HOST:-your-docker-host}" "timeout 3 curl -sS -H 'Authorization: Bearer ${CLASH_API_SECRET:-}' http://127.0.0.1:56789/connections" 2>/dev/null
}

# State across samples
prev_total=""
prev_ts=""
last_summary_ts=$(date +%s)
window_start_total=""
window_start_ts=""
warn_high_rate_streak=0
sample_idx=0

# Init with a first read (retry up to 3 times for transient failures)
prev_total=""
for i in 1 2 3; do
    first_json=$(fetch_clash)
    if [[ -n "$first_json" ]]; then
        prev_total=$(echo "$first_json" | python3 -c "
import sys,json
try:
    d = json.loads(sys.stdin.read())
    print(d.get('downloadTotal', 0))
except:
    print(0)
" 2>/dev/null)
        [[ -n "$prev_total" && "$prev_total" != "0" ]] && break
    fi
    echo "[monitor] init fetch attempt $i failed, retry in 3s"
    sleep 3
done
if [[ -z "$prev_total" || "$prev_total" == "0" ]]; then
    echo "[monitor] ERROR: cannot reach clash API after 3 retries. Check the remote host / port 56789" >&2
    exit 1
fi
prev_ts=$(date +%s)
window_start_total=$prev_total
window_start_ts=$prev_ts

echo "[monitor] initial downloadTotal=$((prev_total/1000000)) MB at $(date +%H:%M:%S)"
echo "====================================" >> "$LOG"
echo "# monitor started $(date -Iseconds)" >> "$LOG"
echo "# sample_sec=$SAMPLE_SEC" >> "$LOG"
echo "# initial_downloadTotal_bytes=$prev_total" >> "$LOG"

while :; do
    sleep "$SAMPLE_SEC"
    now_ts=$(date +%s)
    json=$(fetch_clash)
    if [[ -z "$json" ]]; then
        echo "[$(date +%H:%M:%S)] ⚠ fetch failed, retrying" | tee -a "$LOG"
        continue
    fi

    # Parse + analyze: save JSON to tmp then read by python
    # (heredoc-into-python3 conflicts with echo|python3 pipe, so use file)
    SAMPLE_TMP=$(mktemp /tmp/clash_sample.XXXXXX)
    echo "$json" > "$SAMPLE_TMP"
    parsed=$(CN_PAT="$CN_MIRROR_PATTERN" DOCKER_PAT="$DOCKER_PATTERN" \
             SAMPLE_TMP="$SAMPLE_TMP" python3 - <<'PY'
import json, os, re, sys
from collections import Counter

with open(os.environ['SAMPLE_TMP']) as f:
    raw = f.read()
try:
    d = json.loads(raw) if raw.strip() else {}
except Exception as e:
    sys.stderr.write(f"parse_err: {e}\n")
    d = {}
total = d.get('downloadTotal', 0)
conns = d.get('connections') or []

CN = re.compile(os.environ.get('CN_PAT', ''), re.I) if os.environ.get('CN_PAT') else None

host_bytes = Counter()
big_conns = []
cn_leak_bytes = 0
for c in conns:
    m = c.get('metadata', {}) or {}
    host = m.get('host') or m.get('destinationIP') or '?'
    src = m.get('sourceIP', '')
    dl = c.get('download', 0)
    host_bytes[host] += dl
    if dl > 100 * 1024 * 1024:
        big_conns.append((dl, host, src))
    if CN and CN.search(host or ''):
        cn_leak_bytes += dl

big_conns.sort(reverse=True)
top_hosts = host_bytes.most_common(5)

print(json.dumps({
    "downloadTotal": total,
    "active_conns": len(conns),
    "top_hosts": [[h, b] for h, b in top_hosts],
    "big_conns": big_conns[:5],
    "cn_leak_bytes": cn_leak_bytes,
}))
PY
)
    rm -f "$SAMPLE_TMP"

    if [[ -z "$parsed" ]]; then
        echo "[$(date +%H:%M:%S)] ⚠ parse failed"
        continue
    fi

    # Extract metrics (tolerate missing fields)
    cur_total=$(echo "$parsed" | python3 -c "import sys,json; print(json.load(sys.stdin).get('downloadTotal', 0))")
    if [[ -z "$cur_total" || "$cur_total" == "0" ]]; then
        echo "[$(date +%H:%M:%S)] ⚠ parse empty, skipping sample"
        continue
    fi
    active_conns=$(echo "$parsed" | python3 -c "import sys,json; print(json.load(sys.stdin).get('active_conns', 0))")
    top3=$(echo "$parsed" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(' '.join(f'{h[:30]}={int(b/1e6)}MB' for h,b in d['top_hosts'][:3]))
")
    cn_leak_mb=$(echo "$parsed" | python3 -c "import sys,json; print(int(json.load(sys.stdin)['cn_leak_bytes']/1e6))")
    # Biggest single-connection warning (top 1 only; persistent big pulls stay shown as info not alarm)
    big_conn_flag=$(echo "$parsed" | WARN_HOST_MB=$WARN_HOST_MB python3 -c "
import sys, json, os
d = json.load(sys.stdin)
warn = int(os.environ.get('WARN_HOST_MB', '500'))
bigs = [(dl, host) for dl, host, src in d['big_conns'] if dl > warn * 1024 * 1024]
if bigs:
    # Show only top-1 to keep stdout clean
    dl, host = bigs[0]
    print(f'⚠BIG={int(dl/1e6)}MB@{host[:30]}')
")

    dt=$(( now_ts - prev_ts ))
    delta_bytes=$(( cur_total - prev_total ))
    rate_mbs=$(python3 -c "print(f'{$delta_bytes/$dt/1e6:.2f}')")
    cum_mb=$(( cur_total / 1000000 ))
    window_delta_mb=$(( (cur_total - window_start_total) / 1000000 ))
    window_dt_min=$(( (now_ts - window_start_ts) / 60 ))

    # Compact stdout line
    ts_short=$(date +%H:%M:%S)
    line="[$ts_short] cum=${cum_mb}MB +${delta_bytes}B rate=${rate_mbs}MB/s conns=$active_conns cn_leak=${cn_leak_mb}MB | $top3"

    # Append anomaly warnings (use python for float compare to avoid bc dep)
    warnings=""
    rate_over=$(python3 -c "print(int(float('$rate_mbs') > $WARN_RATE_MB_S))")
    if [[ "$rate_over" == "1" ]]; then
        warn_high_rate_streak=$(( warn_high_rate_streak + 1 ))
        [[ $warn_high_rate_streak -ge 3 ]] && warnings+=" ⚠HIGH_RATE_${warn_high_rate_streak}x"
    else
        warn_high_rate_streak=0
    fi
    [[ $cn_leak_mb -gt $WARN_CN_LEAK_MB ]] && warnings+=" ⚠ CN_LEAK"
    [[ -n "$big_conn_flag" ]] && warnings+=" $big_conn_flag"

    full_line="$line$warnings"
    echo "$full_line"
    # Log JSON + human line
    {
        echo "$ts_short | $full_line"
        echo "  _json: $parsed"
    } >> "$LOG"

    # Periodic summary every SUMMARY_INTERVAL_SEC
    if (( now_ts - last_summary_ts >= SUMMARY_INTERVAL_SEC )); then
        summary="=== SUMMARY ${ts_short} | window=${window_dt_min}min total=+${window_delta_mb}MB avg=$(python3 -c "print(f'{$window_delta_mb/$window_dt_min/60*3600/1024:.1f}')")GB/h ==="
        echo
        echo "$summary"
        echo
        echo "$summary" >> "$LOG"
        last_summary_ts=$now_ts
        window_start_total=$cur_total
        window_start_ts=$now_ts
    fi

    prev_total=$cur_total
    prev_ts=$now_ts
    sample_idx=$(( sample_idx + 1 ))
done
