# Canonical Operations

Maintained launch code is organized by responsibility:

- workflows/: reproducible collection, SFT, RL data, RL training, and eval.
- launch/: reusable infrastructure bring-up and image/runtime preparation.
- monitor/: reusable monitors, audits, and keepalive utilities.
- cleanup/: targeted cleanup; avoid broad Ray or Docker teardown.
- recipes/catalog.toml: recipes exposed through the top-level skillrl CLI.
- cache/: curated offline verifier/runtime payloads required by local Docker.

Run maintained workflows from the repository root. New experiment-specific
source must not be placed under experiments/. Extend an existing canonical
entrypoint before creating another wrapper.

The current RL entrypoint is workflows/rl_training/run_rl.sh. The current
owner-aware evaluation entrypoint is
workflows/rl_eval/run_eval70_checkpoint_set.sh.
