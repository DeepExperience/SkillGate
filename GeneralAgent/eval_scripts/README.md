# Evaluation Scripts

This directory contains evaluation-domain code. It is not the preferred place
for maintained multi-stage launch scripts.

## Maintained Path

- `unified_runner/`: current agent/evaluation stack used by SFT collection,
  quick/full evaluation, and the Relax RL environment.
- `skills_retrieval/`: retrieval utilities used to build the skill-augmented
  prompt inputs.
- `prebake_images/`: Docker/image preparation utilities for benchmark runs.
- Bench-specific directories keep dataset maintenance helpers, patch scripts,
  verifiers, and small inspection utilities when they are still needed by the
  unified runner or data-preparation flows.

## Deprecated Path

Historical direct benchmark runners and watchdogs were archived under:

- `archive/generalagent_cleanup_20260526/originals/GeneralAgent/eval_scripts/`

New reproducible evaluations should be launched from:

- `ops/workflows/`
- `ops/launch/run_dynamic_bench.sh` when a lower-level dynamic benchmark entry
  is explicitly needed.

