#!/usr/bin/env bash
# Ablation v2: 1849-skill library (1651 + 160 clone + 27 hw) with V7 pipeline.
# Compares against 1651-skill (yesterday's v7pipeline_on_v8lib) and 1143-skill (v7_3stage).
set -uo pipefail
PROJ=${SKILLRL_ROOT:-$(pwd)}
DATE=$(date +%Y%m%d)
SUFFIX=v7pipeline_on_1849lib
REPORT_DIR=$PROJ/experiments/${DATE}/${DATE}_${SUFFIX}/reports

export PATH="${SKILLRL_CONDA_ROOT:-$HOME/anaconda3}/envs/slime/bin:$PATH"
export NO_PROXY=127.0.0.1,localhost,0.0.0.0
export HF_ENDPOINT="https://hf-mirror.com"
export SGLANG_DISABLE_CUDNN_CHECK=1

cd $PROJ

echo "[STEP 1/6] preflight: any active eval tmux?"
if tmux ls 2>/dev/null | grep -qE '^v[78]_(tb2|claw|sb|seta|swe|skillsbench)'; then
    echo "ABORT: eval tmux still alive — wait/kill first"
    tmux ls | grep -E '^v[78]_'
    exit 1
fi
echo "[STEP 1/6 DONE] no active eval tmux"

echo "[STEP 2/6] kill SGLang + wait GPU release"
tmux kill-session -t sglang 2>/dev/null || true
pkill -9 -f 'sglang.launch_server' 2>/dev/null || true
sleep 6
for i in 1 2 3 4 5 6 7 8 9 10; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1{print $1}')
    [ "$free" -gt 60000 ] && { echo "[STEP 2/6 DONE] GPU0 free=$free MB"; break; }
    echo "  waiting GPU iter $i, free=$free MB"
    sleep 4
done

echo "[STEP 3/6] rebuild embedding index (Qwen3-Emb-8B, 1849 skill)"
cd $PROJ/GeneralAgent/eval_scripts/skills_retrieval
python build_index.py \
    --model Qwen/Qwen3-Embedding-8B \
    --output skill_index_qwen3emb8b.pkl 2>&1 | tail -10
echo "[STEP 3/6 DONE]"

echo "[STEP 4/6] rebuild BM25 index"
python build_bm25_index.py 2>&1 | tail -5
echo "[STEP 4/6 DONE]"

echo "[STEP 5/6] run v7 pipeline (3-stage retrieve_v6_3stage.py) on 5 benches"
cd $PROJ
python GeneralAgent/eval_scripts/skills_retrieval/retrieve_v6_3stage.py \
    --bench skillsbench tb2 seta claw swe \
    --coarse-k 50 --rerank-k 10 \
    --out-dir $PROJ/experiments \
    --date $DATE \
    --suffix $SUFFIX \
    --skip-gpu-check 2>&1 | tail -25
echo "[STEP 5/6 DONE]"

echo "[STEP 6a] compare 1849 (NEW) vs 1651 (PREV ablation Apr 23)"
mkdir -p "$REPORT_DIR"
python $PROJ/skill_libraries/compare_skill_lib_ablation.py $DATE $SUFFIX 2>&1 | \
    tee "$REPORT_DIR/skill_lib_1849_vs_1651.md"
echo "[STEP 6a DONE]"

echo "[STEP 6b] also compare 1849 vs 1143 (the original v7_3stage on Apr 22)"
# Patch comparison: temporarily swap the OLD_DIR to v7_3stage (1143-skill original)
python3 <<PY 2>&1 | tee "$REPORT_DIR/skill_lib_1849_vs_1143.md"
import sys
sys.argv = ['compare', '$DATE', '$SUFFIX']
import importlib.util
spec = importlib.util.spec_from_file_location('cmp', '$PROJ/skill_libraries/compare_skill_lib_ablation.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
# monkey-patch main: just rerun
def run():
    OLD_DIR = m.retrieval_dir("20260422", "v7_3stage")
    NEW_DIR = m.retrieval_dir("$DATE", "$SUFFIX")
    print(f"OLD (1143 skill, original v7): {OLD_DIR}")
    print(f"NEW (1849 skill, latest):      {NEW_DIR}")
    print()
    totals = {"old_low_tail": 0, "new_low_tail": 0, "old_top1_sum": 0.0, "new_top1_sum": 0.0,
              "old_top10_mean_sum": 0.0, "new_top10_mean_sum": 0.0, "n": 0}
    for bench in m.BENCHES:
        old = m.load(OLD_DIR / f"{bench}.jsonl")
        new = m.load(NEW_DIR / f"{bench}.jsonl")
        common = sorted(set(old) & set(new))
        if not common: continue
        print(f"=== {bench} N={len(common)} ===")
        old_t1 = [m.top_scores(old[t])[0] for t in common]
        new_t1 = [m.top_scores(new[t])[0] for t in common]
        old_low = sum(1 for s in old_t1 if s < 0.2)
        new_low = sum(1 for s in new_t1 if s < 0.2)
        print(f"  avg top-1: OLD={sum(old_t1)/len(old_t1):.3f}  NEW={sum(new_t1)/len(new_t1):.3f}  Δ{(sum(new_t1)-sum(old_t1))/len(old_t1):+.3f}")
        print(f"  gap (top1<0.2): OLD={old_low}/{len(common)}  NEW={new_low}/{len(common)}  Δ{new_low-old_low:+d}")
        print(f"  Top-1 buckets:")
        print(f"    OLD: " + m.fmt_buckets(m.bucket_counts(old_t1), len(common)))
        print(f"    NEW: " + m.fmt_buckets(m.bucket_counts(new_t1), len(common)))
        # Biggest improvements
        deltas = sorted([(t, m.top_scores(new[t])[0] - m.top_scores(old[t])[0]) for t in common], key=lambda x: -x[1])
        print(f"  Top-3 improvements:")
        for t, d in deltas[:3]:
            if d <= 0: break
            n_name = new[t].get("reranked_top10",[{}])[0].get("skill_name","?")
            o_name = old[t].get("reranked_top10",[{}])[0].get("skill_name","?")
            print(f"    {t:<40} Δ={d:+.3f}  '{o_name}' → '{n_name}'")
        print()
        totals["n"] += len(common); totals["old_low_tail"] += old_low; totals["new_low_tail"] += new_low
        totals["old_top1_sum"] += sum(old_t1); totals["new_top1_sum"] += sum(new_t1)
    n = totals["n"] or 1
    print("="*60)
    print(f"GRAND TOTAL N={n}")
    print(f"  avg top-1: OLD={totals['old_top1_sum']/n:.3f}  NEW={totals['new_top1_sum']/n:.3f}  Δ{(totals['new_top1_sum']-totals['old_top1_sum'])/n:+.3f}")
    print(f"  gap (top1<0.2): OLD={totals['old_low_tail']}/{n}  NEW={totals['new_low_tail']}/{n}  Δ{totals['new_low_tail']-totals['old_low_tail']:+d}")
run()
PY
echo "[STEP 6b DONE]"

echo "[STEP 7] restart SGLang"
if [ -f /tmp/restart_sglang.sh ]; then
    tmux new-session -d -s sglang "bash /tmp/restart_sglang.sh 2>&1 | tee /tmp/sglang_restart.log"
    echo "[STEP 7 DONE] SGLang launching"
else
    echo "[STEP 7 WARN] no /tmp/restart_sglang.sh"
fi

echo "[ALL DONE $(date -Iseconds)]"
echo "Reports:"
echo "  $REPORT_DIR/skill_lib_1849_vs_1651.md"
echo "  $REPORT_DIR/skill_lib_1849_vs_1143.md"
