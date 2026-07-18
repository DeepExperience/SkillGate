# Fast Iteration Loop

This project should not wait for full collection / full SFT / full eval before
learning something. Use three nested loops:

1. **Live data-quality dashboard** while collection is running.
2. **Small SFT loop** on the latest collected successful trajectories.
3. **Quick holdout eval** on a fixed 20-30 task subset of the sacred test set.

## 1. Data Quality Dashboard

Run this any time; it is read-only:

```bash
python3 GeneralAgent/sft_data_collection/data_quality_dashboard.py \
  20260427_0847_sft_pipeline_full
```

Outputs:

- `experiments/<date>/<run_id>/reports/data_quality_dashboard.md`
- `experiments/<date>/<run_id>/reports/data_quality_dashboard.json`

Track these numbers first:

- `strict_used_skill_success`: successful trajectories where the agent really
  opened a skill file.
- `strict_used_skill_success_non_meta`: same, excluding meta-talk leakage.
- `teacher_only`: tasks that Phase 1 did not solve but teacher fallback solved.
- token p50/p90: detects trajectories too long for SFT.

## 2. Small SFT Loop

Prepare a small LoRA config from the currently available data:

```bash
MAX_EXAMPLES=256 MAX_STEPS=30 \
  bash GeneralAgent/sft_training/scripts/prepare_small_sft_loop.sh \
  20260427_0847_sft_pipeline_full
```

This creates:

- partial combined plan from phase1 + generated teacher plans,
- filtered `collected_small_256/sft_messages.jsonl`,
- LLaMA-Factory OpenAI-format data,
- a small `max_steps` LoRA config under the run root.

To actually train:

```bash
EXECUTE=1 MAX_EXAMPLES=256 MAX_STEPS=30 \
  bash GeneralAgent/sft_training/scripts/prepare_small_sft_loop.sh \
  20260427_0847_sft_pipeline_full
```

Use this loop to test whether the trained model changes behavior on quick
holdout, especially skill-read rate and skill-use success rate. Full SFT should
only happen after the small loop shows a signal.

## 3. Quick Holdout

Generate the fixed quick subset once:

```bash
python3 GeneralAgent/sft_data_collection/make_quick_holdout.py
```

Default counts total 30 retrieval-covered tasks:

- claw: 8
- tb2: 6
- sb_ns: 5
- seta_synth: 3
- swe_lite: 8

Output:

- `GeneralAgent/sft_data_collection/outputs/splits/default/quick_test/quick30/holdout_split.json`
- per-bench task lists in the same directory.

Run a retrieval eval on the currently served model. This fills the
`bs-retrieval` row in the quick table: retrieval top-10 is injected, but no
hidden use-skill/no-skill instruction is added.

```bash
RUN_ID=$(date -u +%Y%m%d_%H%M)_quick_eval_base \
MODEL=qwen3.5-9b \
ARM=retrieval \
WORKERS=4 \
bash ops/launch/run_quick_holdout_eval.sh
```

Run the matching baseline row with the same heldout tasks and no retrieval
skill injection:

```bash
RUN_ID=$(date -u +%Y%m%d_%H%M)_quick_eval_base_noretrieval \
MODEL=qwen3.5-9b \
ARM=baseline \
WORKERS=4 \
bash ops/launch/run_quick_holdout_eval.sh
```

For an SFT model, serve it on the same OpenAI-compatible endpoint and use its
served model id:

```bash
RUN_ID=$(date -u +%Y%m%d_%H%M)_quick_eval_sft256 \
MODEL=<served-sft-model-id> \
OPENAI_API_BASE=http://127.0.0.1:30000/v1 \
ARM=retrieval \
bash ops/launch/run_quick_holdout_eval.sh
```

The quick eval plan uses `mode=eval_retrieval` for retrieval rows and
`mode=eval_baseline` for no-retrieval rows. Neither mode adds hidden runtime
nudges.

## Daily Cadence

- Morning: dashboard on overnight collection/training.
- Midday: prepare small SFT if at least ~100 usable samples exist.
- Afternoon: quick holdout eval on base vs latest SFT.
- Evening: start the next large collection/training run with one explicit
  bottleneck hypothesis.
