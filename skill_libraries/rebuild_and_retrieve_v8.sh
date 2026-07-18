#!/usr/bin/env bash
# One-shot: kill SGLang → rebuild skill_index_qwen3emb8b.pkl → rerun v8_3stage retrieval on all benches → restart SGLang.
# Run AFTER v7 eval completes (no tmux sb_ws_retr_* / v7_* active).
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ="${PROJ:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATE=$(date +%Y%m%d)
SUFFIX=v8_3stage

export PATH="${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/envs/slime/bin:/root/.local/bin:$PATH"
export NO_PROXY=127.0.0.1,localhost,0.0.0.0
export no_proxy=127.0.0.1,localhost,0.0.0.0
export HF_ENDPOINT="https://hf-mirror.com"
export SGLANG_DISABLE_CUDNN_CHECK=1

cd $PROJ

# Pre-flight
echo "================ pre-flight ================"
alive_v7=$(tmux ls 2>/dev/null | grep -cE "^(v7_|sb_ws_retr_)")
if [[ $alive_v7 -gt 0 ]]; then
    echo "ABORT: $alive_v7 active v7/eval tmux sessions — wait for them to finish"
    tmux ls | grep -E "^(v7_|sb_ws_retr_)" | head
    exit 1
fi
echo "No active v7/eval tmux — safe to kill SGLang"
n_skills=$(ls $PROJ/skill_libraries/merged/ | wc -l)
echo "Current skill library: $n_skills skills"

# Step 1: kill SGLang
echo "================ step 1: kill SGLang ================"
tmux kill-session -t sglang 2>/dev/null || true
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
sleep 5
for i in 1 2 3 4 5; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1{print $1}')
    [[ $free -gt 60000 ]] && { echo "GPU0 free $free MB ok"; break; }
    echo "  wait GPU release iter $i, free=$free MB"
    sleep 3
done

# Step 2: rebuild skill_index_qwen3emb8b.pkl
echo "================ step 2: rebuild index ================"
cd $PROJ/GeneralAgent/eval_scripts/skills_retrieval
python build_index.py \
    --model Qwen/Qwen3-Embedding-8B \
    --output skill_index_qwen3emb8b.pkl 2>&1 | tail -15
cd $PROJ

# Step 3: rerun v8_3stage retrieval on all 5 benches
echo "================ step 3: rerun v8 retrieval (all 5 benches) ================"
OUT_DIR=$PROJ/experiments
mkdir -p $OUT_DIR
python GeneralAgent/eval_scripts/skills_retrieval/retrieve_v6_3stage.py \
    --bench skillsbench tb2 seta claw swe \
    --coarse-k 50 --rerank-k 10 \
    --out-dir $OUT_DIR \
    --date $DATE \
    --suffix $SUFFIX \
    --skip-gpu-check 2>&1 | tail -30

# Step 4: compute improvement table vs v7
echo "================ step 4: compare v7 vs v8 score distribution ================"
python3 <<'PY'
import json
from datetime import datetime
from pathlib import Path
from collections import Counter

V7_NEW = "experiments/20260422/20260422_v7_3stage/retrieval_results/{bench}.jsonl"
V7_LEGACY = "archive/overnight/logs/migrated_20260428/logs/results/20260422/20260422_retrieval_{bench}_v7_3stage.jsonl"
date = datetime.utcnow().strftime('%Y%m%d')
V8 = f"experiments/{date}/{date}_v8_3stage/retrieval_results/{{bench}}.jsonl"
BENCHES = ["skillsbench","tb2","seta","claw","swe"]

def low_high(path):
    if not Path(path).exists(): return None, None, 0
    low = high = 0; n = 0
    for l in open(path):
        r = json.loads(l)
        top = r.get("reranked_top10") or []
        s = top[0].get("rerank_score", 0) if top else 0
        if s < 0.2: low += 1
        if s > 0.5: high += 1
        n += 1
    return low, high, n

print(f"{'bench':<12} {'v7_low':>7} {'v8_low':>7} {'Δlow':>6} | {'v7_high':>8} {'v8_high':>8} {'Δhi':>5} | N")
print("-"*70)
for b in BENCHES:
    v7 = V7_NEW.format(bench=b)
    if not Path(v7).exists():
        v7 = V7_LEGACY.format(bench=b)
    v8 = V8.format(bench=b)
    l7, h7, n7 = low_high(v7)
    l8, h8, n8 = low_high(v8)
    if l7 is None or l8 is None:
        print(f"{b:<12} SKIP missing"); continue
    n = max(n7, n8)
    print(f"{b:<12} {l7:>7} {l8:>7} {l8-l7:>+6} | {h7:>8} {h8:>8} {h8-h7:>+5} | {n}")
PY

# Step 5: restart SGLang
echo "================ step 5: restart SGLang ================"
tmux new-session -d -s sglang "bash /tmp/restart_sglang.sh 2>&1 | tee /tmp/sglang_restart_v8.log"
for i in $(seq 1 60); do
    if curl -sS --max-time 2 http://localhost:30000/v1/models 2>&1 | grep -q '"id"'; then
        echo "SGLang UP (${i}0s)"
        break
    fi
    sleep 10
done

echo "================ done ================"
echo "Compare distribution: python3 skill_libraries/compare_v7_v8_retrieval.py"
