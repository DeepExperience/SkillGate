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

Each model row itself lives under its owner experiment at
`experiments/rl/runs/<experiment>/eval/<eval-id>/rows/<row-id>/`; no central
`experiments/rl_eval` result tree is created.

The report contains task-level/trial outcome tables plus, when `--manifest` is
provided, oracle, misleading, other, and no-read attribution and their success
rates. The canonical default is eval70 with four repeats per task.

Lower-level reusable components:

- `run_eval70_model.py`: plan, execute, resume, and render one model row.
- `convert_cp2_qwen35_hf.py`: Relax TP4/CP2 checkpoint to HF export.
- `analyze_eval70_3tables.py` and `format_eval70_zcc.py`: outcome tables.
- `analyze_slate_reads.py`: mixed-skill behavior attribution.
- `ray_remote_sglang.py`: remote-node SGLang lifecycle.

Historical model-specific export/launch/eval wrappers are preserved under
`archive/ops_workflow_cleanup_20260712/rl_eval/`.
