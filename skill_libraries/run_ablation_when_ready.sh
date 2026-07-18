#!/usr/bin/env bash
# Chain: wait for tb2 workers → kill SGLang → run v7-pipeline ablation → compare.
set -uo pipefail
PROJ=${SKILLRL_ROOT:-$(pwd)}
DATE=$(date +%Y%m%d)
SUFFIX=v7pipeline_on_v8lib
LOG=/tmp/ablation_when_ready.log

echo "[STEP 1/5] wait for v8_tb2_* tmux to exit..."
until [ "$(tmux ls 2>/dev/null | grep -c '^v8_tb2_')" = "0" ]; do
    sleep 60
done
echo "[STEP 1/5 DONE] tb2 workers all exited"

echo "[STEP 2/5] kill SGLang + wait GPU release"
tmux kill-session -t sglang 2>/dev/null || true
pkill -9 -f 'sglang.launch_server' 2>/dev/null || true
sleep 6
for i in 1 2 3 4 5 6 7 8 9 10; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1{print $1}')
    [ "$free" -gt 60000 ] && { echo "[STEP 2/5 DONE] GPU0 free=$free MB"; break; }
    echo "  waiting GPU release iter $i, free=$free MB"
    sleep 4
done

echo "[STEP 3/5] run v7 pipeline on 1651-skill library"
export PATH="${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/envs/slime/bin:$PATH"
export NO_PROXY=127.0.0.1,localhost,0.0.0.0
export HF_ENDPOINT="https://hf-mirror.com"
cd $PROJ
python GeneralAgent/eval_scripts/skills_retrieval/retrieve_v6_3stage.py \
    --bench skillsbench tb2 seta claw swe \
    --coarse-k 50 --rerank-k 10 \
    --out-dir $PROJ/experiments \
    --date $DATE \
    --suffix $SUFFIX \
    --skip-gpu-check 2>&1 | tail -40
echo "[STEP 3/5 DONE]"

echo "[STEP 4/5] compare old v7 (1143) vs new (1651) reranker dist"
REPORT_DIR="$PROJ/experiments/${DATE}/${DATE}_${SUFFIX}/reports"
mkdir -p "$REPORT_DIR"
python $PROJ/skill_libraries/compare_skill_lib_ablation.py $DATE $SUFFIX | tee "$REPORT_DIR/skill_lib_ablation.md"
echo "[STEP 4/5 DONE] report at experiments/${DATE}/${DATE}_${SUFFIX}/reports/skill_lib_ablation.md"

echo "[STEP 5/5] restart SGLang"
cp /tmp/restart_sglang.sh /tmp/restart_sglang_ablation.sh 2>/dev/null || true
if [ -f /tmp/restart_sglang.sh ]; then
    tmux new-session -d -s sglang "bash /tmp/restart_sglang.sh 2>&1 | tee /tmp/sglang_restart.log"
    echo "[STEP 5/5 DONE] SGLang launching in tmux"
else
    echo "[STEP 5/5 WARN] /tmp/restart_sglang.sh not found; SGLang NOT restarted"
fi

echo "[ALL DONE $(date -Iseconds)]"
