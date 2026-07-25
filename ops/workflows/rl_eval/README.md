# RL Evaluation Workflows

The maintained post-training path is `run_eval70_checkpoint_set.sh`. It accepts
any number of exported HF models or Relax checkpoint roots and evaluates every
row under the same condition.

```bash
bash ops/workflows/rl_eval/run_eval70_checkpoint_set.sh \
  --group my_eval70_comparison \
  --skill-mode mixed \
  --snapshot skill_libraries/snapshots/rl/eval70_v8prod_deepfrozen_20260712/snapshot_eval70 \
  --manifest skill_libraries/snapshots/rl/eval70_v8prod_deepfrozen_20260712/slate_manifest_eval70.jsonl \
  --model <baseline-experiment> baseline experiments/rl/runs/<baseline-experiment>/model/final_hf \
  --checkpoint <trained-experiment> best experiments/rl/runs/<trained-experiment>/segments/<segment> 79 best \
  --checkpoint <trained-experiment> final experiments/rl/runs/<trained-experiment>/segments/<segment> 99 final \
  --dry-run
```

Remove `--dry-run` to queue the run. The workflow waits for all requested
checkpoints and for both Ray GPU nodes to become idle, exports CP2 checkpoints,
starts four TP4 SGLang engines, resumes incomplete trials, and writes a combined
report to `z_cc_terminal_imgs/<group>_results.md`.

For an exactly two-row comparison that should run concurrently, add
`--parallel-two-rows`. Row 1 is served by two local TP4 engines and row 2 by two
TP4 engines on the remote GPU node. Both eval rows start together with `WORKERS`
workers each; `PARALLEL_DOCKER_START_CAP` defaults to 64 per row so the shared
Docker daemon remains capped at 128 concurrent starts. The mode is resume-safe
and requires exactly two rows.

On a standalone 8-GPU worker, add `--local-only`. Sequential rows use two local
TP4 replicas. With `--parallel-two-rows`, the two models run concurrently on
disjoint four-GPU halves with one TP4 engine each; lower the Claw bench cap to
keep per-engine load comparable with the two-node protocol. The default remains
the two-node topology. For in-place skill-body revisions, also pass
`--fingerprint-referenced-skills` so stale completed rows cannot be reused when
the retrieval JSONL and manifest paths are unchanged.

Each model row itself lives under its owner experiment at
`experiments/rl/runs/<experiment>/eval/<eval-id>/rows/<row-id>/`; no central
`experiments/rl_eval` result tree is created.

The report contains task-level/trial outcome tables plus, when `--manifest` is
provided, oracle, misleading, other, and no-read attribution and their success
rates. The canonical default is eval70 with four repeats per task.

An explicit task TSV may add a third `repeats` column. This is used by the
paper-wide 385-trial protocol: Claw161 uses one trial per task while the other
56 tasks use four. Heterogeneous schedules require `--report-style main-only`
because one shared task-level pass@R definition would be ambiguous; row
validation still checks the exact repeat count for every task.

For a still-running training job whose complete checkpoints should each be
evaluated and published into the paper master, use the restartable watcher:

```bash
python3 ops/workflows/rl_eval/watch_eval70_checkpoints.py \
  --owner <training-experiment> \
  --checkpoint-root experiments/rl/runs/<training-experiment>/segments/<segment> \
  --start-iteration <current-complete-iteration> \
  --snapshot skill_libraries/snapshots/rl/eval70_final_v8prod_fixed4/snapshot_eval70 \
  --manifest skill_libraries/snapshots/rl/eval70_final_v8prod_fixed4/slate_manifest_eval70.jsonl
```

The watcher discovers only checkpoints with a valid completion marker, invokes
`run_eval70_checkpoint_set.sh` sequentially, and publishes a checkpoint only
after validating 280 records as 70 tasks x4. Its state and per-checkpoint reports
live under `experiments/skillgate_paper/masked_task_only_checkpoint_watch/` by
default. Restarting the same command resumes both incomplete eval rows and the
checkpoint queue; a file lock prevents duplicate watchers. Paper publication is
restricted to the three `masked-task-only` rows plus one status block.

Lower-level reusable components:

- `run_eval70_model.py`: plan, execute, resume, and render one model row.
- `convert_cp2_qwen35_hf.py`: Relax TP4/CP2 checkpoint to HF export.
- `analyze_eval70_3tables.py` and `format_eval70_zcc.py`: outcome tables.
- `analyze_slate_reads.py`: mixed-skill behavior attribution.
- `analyze_claw147_paper_eval.py`: Claw147 task, chronological first-read,
  conditional-success, and paired paper tables.
- `ray_remote_sglang.py`: remote-node SGLang lifecycle.

Paper selector interventions use one restartable route builder rather than
method-specific conversion scripts:

```bash
python3 ops/workflows/rl_eval/build_skillgate_eval_routes.py \
  --mode oracle \
  --output-root experiments/skillgate_paper/routes/oracle_eval70

python3 ops/workflows/rl_eval/build_skillgate_eval_routes.py \
  --mode router \
  --api-base http://127.0.0.1:30000/v1 \
  --router-model <served-model-name> \
  --output-root experiments/skillgate_paper/routes/<router-name>
```

The four modes are `oracle`, `misleading`, `router`, and `reranker`. Router
choices are constrained to the current 16 candidate names with JSON Schema;
the output root is accepted by `run_eval70_checkpoint_set.sh --skill-mode
retrieve --snapshot <output-root>`. `summary.json` records category counts and
the input fingerprint.

The paper's offline sections and figures are regenerated with:

```bash
/usr/bin/python3 ops/workflows/rl_eval/build_skillgate_paper_analysis.py
```

It reads immutable train/eval artifacts, caches expensive scans under
`experiments/skillgate_paper/analysis/`, and writes a structured JSON plus the
Markdown fragment for sections 4.1--4.12. The system Python is used because it
contains the plotting stack; model training and evaluation still use their
existing environments.

Historical model-specific export/launch/eval wrappers are preserved under
`archive/ops_workflow_cleanup_20260712/rl_eval/`.

## Claw 147 Mixed-Slate Dataset

The standalone 147-task Claw paper split (the frozen 161-task T-series list
minus the 14 Claw tasks in FINAL eval70) is built and resumed in place with:

```bash
bash ops/workflows/rl_eval/run_eval_claw_147_slate.sh
```

Canonical outputs are under `skill_libraries/snapshots/rl/eval_claw_147/`:
`task_ids.txt` is accepted by `run_unified_claw.py --tasks-file`,
`snapshot_eval_claw_147/claw.jsonl` is the mixed retrieval input and must be
used with `--retrieval-top-n 16`, and `slate_manifest_eval_claw_147.jsonl`
provides oracle/misleading/relevant/irrelevant attribution. `COMPLETE` plus
`audit_report.json` are required before evaluation. The builder keeps only one
fixed output set and regenerates failed oracle or misleading entries in place.

`FLEET_MODE=auto` uses two GPU nodes when exactly one live Ray peer exists and
otherwise uses the current 8-GPU node with two local TP4 endpoints. Set
`FLEET_MODE=single` or `FLEET_MODE=dual` to require a topology explicitly;
`DRY_RUN=1` validates the topology and current snapshot without starting models.

The authoritative post-generation oracle-body audit is restartable and writes
only into that same snapshot:

```bash
python3 ops/workflows/rl_eval/audit_claw147_oracles_with_claude.py --workers 4
```

It invokes Claude Code Opus at medium effort once per task, preserves each YAML
frontmatter byte-for-byte, and records per-task verdicts and hashes under
`experiments/skill_slate_build/eval_claw_147/claude_oracle_audit/`.

Each row contains one oracle, five relevant, five irrelevant, and five
misleading skills. Oracle audits are grounded in the task definition, grader,
mock-service source, fixtures, and sandbox grader files. Every misleading skill
must pass both an independent semantic audit and the v5 frozen-instance outcome
audit with a separate falsifier; description similarity to the oracle is capped
at `0.92`. A targeted rejection remains attached to the same skill name until
the outcome audit passes, so retries cannot silently fall back to an earlier
invalid premise.
