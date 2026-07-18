# SFT Training Setup

This directory stores the project-side training glue for LLaMA-Factory.

## Environment

Activate the lightweight LLaMA-Factory environment:

```bash
cd /path/to/skillRL
source GeneralAgent/sft_training/activate_llamafactory.sh
llamafactory-cli env
```

Installed layout:

- `GeneralAgent/third_party/LLaMA-Factory`: upstream LLaMA-Factory source.
- `GeneralAgent/.venvs/llamafactory`: lightweight venv using the existing CUDA Torch from `slime`.
- `GeneralAgent/.venvs/llamafactory/bin/torchrun`: wrapper forcing torchrun subprocesses to use the venv Python.

## Current Caveat

The environment intentionally avoids a full dependency copy because the project directory hit quota during full install. It reuses the existing `slime` packages for Torch, Transformers, Datasets, and Accelerate, and installs only LLaMA-Factory plus small missing packages in the venv.

Before a real training run, use the activation script and run a dataset conversion/tokenization dry run.

## Data Bridge

After `GeneralAgent/sft_data_collection/collect_successes.py` writes `sft_messages.jsonl`, first convert it to the OpenClaw-compatible message format:

```bash
python GeneralAgent/sft_training/convert_to_openclaw_compat.py \
  --input GeneralAgent/sft_training/datasets/<run>_hindsight/sft_messages.jsonl \
  --output GeneralAgent/sft_training/datasets/<run>_openclaw_compat/sft_messages.jsonl
```

This step is mandatory for train / inference / OpenClaw alignment:

- rewrites system prompt to `unified_runner.openclaw_compat.build_openclaw_system_prompt()`;
- moves benchmark-specific context from system into the first user message;
- rewrites skills to OpenClaw `<available_skills>` with exact `SKILL.md` locations;
- migrates legacy `edit(old_string,new_string)` and unified-only `ls/grep/find/apply_patch` calls to OpenClaw-deployable calls (`exec` for shell-native operations);
- renders the full OpenClaw-style sections expected by deployment; unavailable OpenClaw tools are still declared for prompt alignment and fail clearly in unified_runner if called;
- keeps workspace per benchmark (`claw=/workspace`, Harbor-style tasks `/root`, SWE `/testbed`) while task-specific HTTP/repo instructions stay in the first user message;
- drops records with non-OpenClaw tool calls or malformed tool XML.

Prompt/tool profiles are controlled centrally by `UNIFIED_PROMPT_PROFILE`:

- `openclaw_full` (default): OpenClaw-aligned prompt + 28 OpenClaw-style tools (probe 27 plus `web_search`; the system `## Tooling` text renders `web_search` before `web_fetch`, while the provider schema block follows OpenClaw's tool-manifest order).
- `legacy_11`: pre-OpenClaw unified_runner prompt + 11 legacy tools (`read/write/edit/apply_patch/grep/find/ls/exec/process/web_fetch/web_search`).

Use `legacy_11` only for ablation or backward-compatible training, not final OpenClaw deployment comparison.

Then convert the compat file to LLaMA-Factory OpenAI format under:

- `GeneralAgent/sft_training/llamafactory_data/agent_sft_pilot.json`
- `GeneralAgent/sft_training/llamafactory_data/dataset_info.json`

Recommended mapping:

- `system` message → top-level `system`.
- `user` message → `{"from": "human", "value": ...}`.
- `assistant.content` → `{"from": "gpt", "value": ...}`; keep the original `<tool_call>` text.
- `tool` message → `{"from": "observation", "value": ...}`.

Do not use LLaMA-Factory tool-call rewriting in the first pass. The eval runner expects the OpenClaw-style assistant text.

Current 1667 smoke output:

- input: `GeneralAgent/sft_training/datasets/20260506_sft_campaign_1667_replace_run05_hindsight_en/sft_messages.jsonl`
- output: `GeneralAgent/sft_training/datasets/20260506_sft_campaign_1667_openclaw_compat/sft_messages.jsonl`
- kept: `1598 / 1666`
- report: `GeneralAgent/sft_training/datasets/20260506_sft_campaign_1667_openclaw_compat/openclaw_compat_report.json`

## First Training Template

Start from `configs/qwen35_9b_lora_agent_template.yaml` after data conversion. The important defaults are:

- `template: qwen3_5_nothink`
- `train_on_prompt: false`
- `mask_history: false`
- `packing: false`
- `cutoff_len: 32768`
