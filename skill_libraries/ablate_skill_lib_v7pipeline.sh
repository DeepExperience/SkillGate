#!/usr/bin/env bash
# Ablate: fixed V7 pipeline (embed-8b coarse top50 → Qwen3-Reranker-8B top10)
# applied to current 1651-skill library. Compare resulting reranker top-10 score
# distribution against v7 historical (1143-skill).
#
# Prereqs: SGLang must be killed (pipeline needs GPU).
# Output: experiments/<date>/<date>_v7pipeline_on_v8lib/retrieval_results/<bench>.jsonl

set -uo pipefail
PROJ=${SKILLRL_ROOT:-$(pwd)}
DATE=${DATE:-$(date +%Y%m%d)}
SUFFIX=${SUFFIX:-v7pipeline_on_v8lib}

export PATH="${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/envs/slime/bin:$PATH"
export NO_PROXY=127.0.0.1,localhost,0.0.0.0
export HF_ENDPOINT="https://hf-mirror.com"

echo "=== Preflight ==="
if tmux ls 2>/dev/null | grep -qE '^v[78]_(tb2|claw|sb|seta|swe)'; then
    echo "ABORT: eval tmux still active (kill them first)"
    tmux ls | grep -E '^v[78]_'
    exit 1
fi

free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1{print $1}')
if [[ $free -lt 60000 ]]; then
    echo "ABORT: GPU0 only $free MB free; need ≥60GB. Kill SGLang."
    exit 1
fi

cd $PROJ

echo "=== Running v7 pipeline (retrieve_v6_3stage.py) on current skill library ==="
python GeneralAgent/eval_scripts/skills_retrieval/retrieve_v6_3stage.py \
    --bench skillsbench tb2 seta claw swe \
    --coarse-k 50 --rerank-k 10 \
    --out-dir $PROJ/experiments \
    --date $DATE \
    --suffix $SUFFIX \
    --skip-gpu-check 2>&1 | tee /tmp/v7pipeline_on_v8lib.log

echo
echo "=== Comparison: old v7 (1143 skill) vs new (1651 skill, same pipeline) ==="
python $PROJ/skill_libraries/compare_skill_lib_ablation.py $DATE $SUFFIX
