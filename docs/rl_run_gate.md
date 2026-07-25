# RL Run Gates

Purpose: make RL changes cheaper to validate before launching a full 8-GPU or
16-GPU run. A full train step is too slow to be the first correctness check:
Ray/Serve/SGLang startup, verifier Docker work, pair-gating refills, logprob,
backward, and checkpointing can hide simple bugs for 20-60 minutes.

This document records the intended gate design only. It does not add new
entrypoints yet.

## Principles

- Use the cheapest gate that exercises the changed surface.
- Keep gates restartable and based on existing run artifacts when possible.
- Separate algorithm/data correctness from infrastructure and long-run
  stability. A fast gate can reject bad changes; it cannot prove a full run is
  stable.
- Do not use full 8-GPU or 16-GPU training as the first smoke test for a code
  change.

## Gate 0: Static And Data Sanity

Target runtime: under 1 minute.

Use for launch, schema, import, and data-prep mistakes.

Checks:
- `bash -n` for touched launch scripts.
- `python3 -m py_compile` for touched Python files when imports are safe.
- Parquet path existence, row counts, and required fields.
- Train/eval split counts and `update_kind` distribution.
- Action-span parser smoke on representative responses.
- Loss-function tensor-shape smoke with synthetic or tiny samples.
- Launch env expansion, including retrieval mode, Docker mode, eval flags,
  action-mask flags, pair-gating flags, and length/batch settings.

Expected output:
- A small text/JSON summary of paths, counts, env flags, and pass/fail.

## Gate 1: Train-Only Replay

Target runtime: 2-5 minutes after the model service/runtime is warm.

Use for loss, action-mask, BC/GRPO weighting, logprob, TP/CP, and actor memory
changes. This gate should replay one or a few saved `rollout_result/train/*.jsonl`
batches without running SGLang generation or verifier Docker.

Inputs:
- Existing train JSONL batches from a known run.
- Optional curated stress batches: long no-skill group, oracle-BC group with
  nonzero action spans, near-context-limit group, and mixed benchmark group.

Checks:
- `convert_samples_to_train_data` still produces required fields.
- `shadow_action_loss_masks` exists when `RELAX_SHADOW_BC_ACTION_MASK=1`.
- Custom loss runs and logs expected metrics.
- Actor/reference logprob and backward fit the selected TP/CP/batch settings.
- No CUDA OOM, Triton OOM, shape mismatch, missing-key, or zero-token
  deadlock.

Limitations:
- Does not test prompt construction, live generation, reward functions,
  verifier Docker, pair-gating refill, eval scheduling, or teardown.

## Gate 2: Rollout-Only Micro

Target runtime: 3-8 minutes with warm Docker/images.

Use for prompt, reward, verifier, pair-gating, audit-writing, eval/abort, and
rollout-state changes. This gate should generate on a tiny fixed fast subset
and stop before actor training.

Profile:
- `n_samples_per_prompt` small.
- Response/context caps much smaller than full train, but still large enough to
  include at least one tool-call/action-span path.
- 1-2 fast tasks with prebuilt images and deterministic expected verifier
  behavior.

Checks:
- Rollout rows have expected `status`, `reward`, `update_kind`, pair fields,
  action-span audit fields, and prompt-clean audit fields.
- Pair-gating accepts/drops/defers according to the intended rule.
- Abort cleanup and eval scheduling do not race.
- Docker containers start and tear down without leaving obvious active-step
  leaks.

Limitations:
- Does not validate actor backward or large-batch memory.

## Gate 3: End-To-End Mini

Target runtime: 8-15 minutes when services are warm; cold start can be longer.

Use for full-path integration before a real run.

Profile:
- `NUM_ROLLOUT=1`.
- Small but real batch, for example `ROLLOUT_BATCH_SIZE=1-2` and
  `GLOBAL_BATCH_SIZE=8-16`.
- Fixed fast subset with at least one no-skill mixed group and one oracle-BC
  replacement candidate if the recipe uses pair-gating.
- Same major serving topology as the intended run when testing TP/CP/resource
  changes.

Checks:
- Ray Serve/SGLang/rollout/reference/actor_fwd/actor services deploy.
- One rollout finishes and writes train JSONL.
- Logprob, custom loss, backward, metrics, and checkpoint/save path all work.
- No data-system catch-up stall after the update.

Limitations:
- It is still a smoke test. It does not prove long-run stability.

## Gate 4: Stability Probe

Target runtime: 30-60 minutes depending on task mix.

Use only after Gates 0-3 pass when changing memory-sensitive settings or core
training behavior.

Profile:
- 3-5 completed train updates.
- Same length caps, TP/CP, resource layout, and batch pressure as the proposed
  production run when possible.
- Eval can stay disabled unless the change touches eval scheduling or eval
  reward handling.

Checks:
- Multiple completed actor train updates.
- Multiple completed rollout batches.
- At least one checkpoint save.
- No OOM, no Triton OOM, no `training failed at step`, and no extended
  data-system catch-up.
- Basic metric sanity: reward/passrate proxy, BC/action-token metrics if
  applicable, KL, grad norm, response length, truncation rate.

## Which Gate To Run

- Loss/action mask/BC coefficient changes: Gate 0, then Gate 1.
- Pair sampler/reward/eval/abort changes: Gate 0, then Gate 2.
- TP/CP/max length/batch/resource changes: Gate 0, Gate 1 stress replay, then
  Gate 3 or Gate 4.
- Launch/env/Ray placement changes: Gate 0, then Gate 3.
- Docker/verifier changes: Gate 0, then Gate 2.
- Full recipe changes before a costly launch: Gates 0-3, then Gate 4 if memory
  or long-run stability is uncertain.

## Current Gaps

- No canonical train-only replay entrypoint exists yet.
- No curated fixed fast rollout subset exists yet.
- No shared summary format exists for comparing gate outputs across changes.
- Current full-run scripts can be parameterized into small runs, but that is
  slower and noisier than purpose-built gates.
