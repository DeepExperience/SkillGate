# Third-party components

This repository vendors several third-party source trees directly (without
nested `.git` metadata). The exact vendored commits are recorded in
`assets/source-snapshot.json`; this file maps each vendored path to its
upstream project and license.

Each vendored tree keeps its upstream `LICENSE`/`NOTICE` files in place —
those files, not this summary, are authoritative. Do not replace a vendored
tree with a fresh upstream clone: the vendored copies carry local patches that
the training and serving stack depends on, and any re-sync must reapply and
re-validate those changes.

## Vendored components

| Path | Upstream | Pinned commit | License | Local modifications |
|---|---|---|---|---|
| `Relax/` | An internal RL framework, vendored in this repo with permission — confirm licensing with the team before public release | `5215cf1605392d2724fe49b11d9037db8b0b3671` | Apache-2.0 (`Relax/LICENSE`) | Carries project patches for agent RL training. First-party code for this paper lives in `Relax/examples/agent_bench/` (selector action credit: `selector_action_credit.py`, `selector_action_grpo_loss.py`) |
| `sglang/` | https://github.com/sgl-project/sglang | `a598eae1d1f5c85d99d6dc36c3c449f231a953eb` | Apache-2.0 (`sglang/LICENSE`) | Carries local patches for model serving used by evaluation and RL rollout (including Qwen3.5 support) |
| `Relax/deps/sglang/` | https://github.com/sgl-project/sglang | `bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2` | Apache-2.0 (`Relax/deps/sglang/LICENSE`) | Pinned serving dependency of Relax; carries local Qwen3.5-related changes for RL rollout |
| `Megatron-LM/` | https://github.com/NVIDIA/Megatron-LM | `fcd446b52f47c9fe50b02546fb6d3efa6df6e224` | NVIDIA BSD-3-Clause-style license with per-file third-party notices, some files Apache-2.0 (`Megatron-LM/LICENSE`) | Carries local patches used by the training stack |
| `Relax/deps/Megatron-Bridge/` | https://github.com/NVIDIA-NeMo/Megatron-Bridge | `2faedbf6fe3c422835a44b2b360cadcb2a116a54` | Apache-2.0 (`Relax/deps/Megatron-Bridge/LICENSE`; bundled `3rdparty/Megatron-LM` keeps its own LICENSE) | Pinned training dependency of Relax |
| `Relax/deps/Megatron-LM/` | Partial tree combining code originating from https://github.com/NVIDIA/Megatron-LM (`megatron/core`, `megatron/legacy`, `megatron/training`, `megatron/inference`, `megatron/post_training`, `megatron/rl`) and https://github.com/NVIDIA-NeMo/Megatron-Bridge (`megatron/bridge`) | not recorded in `assets/source-snapshot.json` | Upstream LICENSE copies restored in-tree: `Relax/deps/Megatron-LM/LICENSE` (from the vendored `Megatron-LM/LICENSE`, NVIDIA BSD-3-Clause-style) and `Relax/deps/Megatron-LM/megatron/bridge/LICENSE` (from `Relax/deps/Megatron-Bridge/LICENSE`, Apache-2.0). Verify exact upstream commits before public release | Pinned training dependency of Relax |
| `slime/` | https://github.com/THUDM/slime (vendored from a maintainer fork) | `913b3c440594d2ab70a15bf87f36fcfe76be9721` | Apache-2.0 (`slime/LICENSE`) | Fork-local modifications for this project's environment/runtime stack |
| `GeneralAgent/third_party/LLaMA-Factory/` | https://github.com/hiyouga/LLaMA-Factory | `e0bc3c19713c263fb542daef8096dbbb4cf34d7b` | Apache-2.0 (`GeneralAgent/third_party/LLaMA-Factory/LICENSE`); vendored version string `0.9.5.dev0` | Used for SFT training and adapter merge; entrypoints under `ops/workflows/sft_training/` |

## Skill libraries

`skill_libraries/` in git contains only first-party merge/ablation scripts and
merge manifests. The merged skill library content itself (skills aggregated
from public community skill repositories) is delivered through the side-car
asset bundle (see `assets/`), not through git. The merge manifests
(`skill_libraries/merged_*_manifest.json`) record the source repository of
every merged skill (e.g. `superpowers`, `SciAgent-Skills`, `bioSkills`,
`dev-skills`, `ctf-skills`, `sf-skills`,
`Claude-Skills-Governance-Risk-and-Compliance`, and others).
**unknown-verify-before-publish**: check the license of each upstream skill
repository before redistributing the merged skill library content.

## Removed components

The `agent-world-model` project (https://github.com/Snowflake-Labs/agent-world-model)
was vendored in an earlier iteration of this handover and has been removed; it
is not part of this repository.

## Pre-release checklist

- [ ] Confirm authorization to publish `Relax/` (internal RL framework) with
      the owning team, including how it should be attributed.
- [ ] Restore upstream LICENSE files into `Relax/deps/Megatron-LM/` (see table).
- [ ] Verify licenses of the community skill repositories aggregated in the
      skill-library asset bundle before redistributing it.
