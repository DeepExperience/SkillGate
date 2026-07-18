# SFT Data → Training 全流程

从 phase1 trial 采集出来到 LLaMA-Factory 起训之间所有步骤、脚本、产物路径。每一步都给出输入输出和关键参数。

更新日期：2026-05-05。

---

## 全景图

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ phase1+phase2│ →  │ collect_     │ →  │ augment_     │ →  │ export_      │ →  │ LLaMA-Factory│ →  │  merge +     │
│ trial trajec.│    │ successes.py │    │ hindsight.py │    │ llamafactory │    │ SFT          │    │  SGLang serve│
│ (per-task)   │    │ (per-run)    │    │ (per-run)    │    │ (per-run)    │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
   trajectory.json     sft_messages       sft_messages         <name>.json +        outputs/.../        merged_models/
   results/<bench>/    .jsonl             .jsonl (with         dataset_info.json    checkpoint-*/        <run>/
   .../trajectories/   datasets/<run>/    hindsight prefix)    llamafactory_data/   adapter_*            <weights>
                                          datasets/<run>_      <run>/
                                          hindsight/
```

每一步只读上一步产物，不依赖更早的中间状态。

---

## 1. 采集 phase1 + phase2 trial

**入口**：`ops/workflows/sft_data_collection/run_qwen27b_campaign_pipeline.sh`（包装 `ops/workflows/sft_data_collection/run_sft_pipeline.sh`）  
**phase2 独立 worker**：`ops/workflows/sft_data_collection/run_phase2_teacher_worker.sh`

执行流：

| 子步骤 | 脚本 | 作用 |
|---|---|---|
| 生成 plan | `make_trial_plan.py` | 把 split 展开成 task × mode × trial 计划，写 plan jsonl |
| campaign 增量过滤 | `filter_plan_for_campaign.py` | 把过去 run 已 strict_used_success 的 task 从 plan 剔除 |
| 启动 phase1 launcher | `launch_trials.py --plan ...phase1.jsonl --workers 16 --task-window 36 --bench-cap ...` | 真跑 trial subprocess（agent rollout in docker），并发 16 worker |
| watcher 推 phase2 | `watch_phase1_teacher_queue.py` | 监控 phase1 全失败 task → 用 `make_teacher_fallback_plan.py` 生成 teacher reflection chunks 推 queue/ |
| phase2 worker | `run_phase2_teacher_worker.sh` → 内部调 `launch_trials.py` | 从 queue/ 拉 chunks，跑 27B teacher reflection；存活独立于 wrapper |

**产物**：
- `experiments/sft_skill_use_campaign/runs/<RUN_ID>/results/<bench>/<task_arm>/trajectories/<task>.json` — 每条 trial 的完整 trajectory
- `experiments/sft_skill_use_campaign/runs/<RUN_ID>/results/<bench>/<task_arm>/incremental.jsonl` — verifier 输出
- `experiments/sft_skill_use_campaign/runs/<RUN_ID>/logs/sft_collection/status.jsonl` — 每 trial 的 launcher 状态（rc、elapsed、error_kind）

**一次 run 通常出 2000-6000 条 trial**（含 phase1 student × 8 trials + phase2 teacher × 8 trials）。

---

## 2. 综合：把 trajectory 转成 SFT 候选样本

**脚本**：`collect_successes.py`

输入 phase1+phase2 的所有 trajectory.json + status.jsonl，做四件事：

1. 按 `(bench, task_id, mode, used_skill)` 分组 dedup（每组保留最短 N 条）
2. 优先级筛选：`student_use_skill > student_no_skill > teacher_retrieval_reflection`
3. 排除 `meta_talk_detected=true` 的轨迹（回答里 leak 了 skill 文件名等不该出现的 token）
4. 结合 `used_skill_via_path` / `used_skill_via_name` 标记每条样本的 strict_used 状态

**产物**：
- `GeneralAgent/sft_training/datasets/<RUN_NAME>/sft_messages.jsonl` — 每行 `{"messages":[...], "metadata":{...}}`，messages 是 OpenAI 格式（system/user/assistant/tool 交替）

**关键**：metadata 里有 `bench`、`task_id`、`mode`、`used_skill`、`resolved`、`turns` 等字段，下游 augment 和 LF 训练都依赖这些。

---

## 3. Hindsight reasoning 增强（v3 新加）

**脚本**：`augment_hindsight.py`

为每条样本，把 27B 当 oracle，让它写一段 ≤500 tokens 的"事后理由化"reasoning，结论与 trajectory 实际行为（读 / 不读 skill）一致，**前置**到第一条 assistant message 的 content：

```
<skill_reasoning>
{27B 写的 reasoning，匹配 user task 语言}
</skill_reasoning>

{原始 first-assistant content (含 <tool_call> XML)}
```

**关键参数**：
- `--api-base http://127.0.0.1:30000/v1` — 复用 phase1 的 27B 端点（无需另起 SGLang）
- `--workers 8`
- `--min-turns 0` — 可选过滤过短轨迹（默认不过滤）
- `--limit 0` — 全跑

**产物**：
- `GeneralAgent/sft_training/datasets/<RUN_NAME>_hindsight/sft_messages.jsonl`
- 每条 metadata 多两个字段：`hindsight_reasoning`（原文）、`hindsight_model`

**用途**：注入"判断是否读 skill 的推理过程"训练信号，让 9B 模型学到 conditional decision，而不是表面模仿 read 动作。

---

## 4. OpenClaw 兼容化（SFT 前必须）

**脚本**：`../sft_training/convert_to_openclaw_compat.py`

这一步把历史采集数据统一改成 **train / inference / OpenClaw deployment** 三者一致的格式。它不重新采集 trajectory，只改 messages 和可安全改写的 tool-call 参数。

输入：

- `GeneralAgent/sft_training/datasets/<RUN_NAME>_hindsight*/sft_messages.jsonl`

输出：

- `GeneralAgent/sft_training/datasets/<RUN_NAME>_openclaw_compat/sft_messages.jsonl`
- `.../openclaw_compat_report.json`

做的事情：

1. **system prompt 重写**：统一改成 `unified_runner.openclaw_compat.build_openclaw_system_prompt()` 生成的 OpenClaw-style prompt。
2. **benchmark context 下沉到 user**：原来各 bench 放在 system 里的 HTTP endpoint docs、SWE repo listing/git log、Harbor runtime 说明，迁移到第一条 user message 的 `Benchmark Runtime Context` / `Repository Runtime Context`。
3. **skills 格式对齐 OpenClaw**：Markdown bullet skills 改成 `<available_skills>` XML；`<location>` 指向精确 `.../SKILL.md` 文件路径。
4. **tool schema 对齐 OpenClaw deploy subset**：用 `GeneralAgent/eval_scripts/unified_runner/tool_schemas.py` 的 OpenClaw-deployable schema 重新渲染 Qwen chat-template schema block；默认只暴露 `read/write/edit/exec/process/web_fetch/web_search`。
5. **tool-call 参数迁移**：
   - `edit(old_string,new_string)` → `edit(edits=[{oldText,newText}])`
   - `ls/grep/find/apply_patch` → `exec(command=...)`
6. **capability-dependent sections gating**：
   - 不暴露 memory/session/gateway/docs/model-alias 工具或上下文时，不在 system prompt 里渲染对应 section，避免训练模型调用部署时不可用的能力。
   - workspace 仍按 bench 设置：`claw=/workspace`，Harbor/TB2/SETA/SB=`/root`，SWE=`/testbed`；HTTP endpoint、repo listing、verifier 等 bench-specific 内容放到第一条 user message。
7. **process 参数迁移**：`process(read/signal,pid=...)` → `process(log/kill,sessionId=...)`
8. **严格过滤**：包含非 OpenClaw deploy subset 工具调用、 malformed tool XML、无法迁移旧参数的样本会整条丢弃，避免训练模型学部署时不存在的工具。

统一开关：`UNIFIED_PROMPT_PROFILE`

- `openclaw_gated`（默认）：OpenClaw 对齐 prompt + 7 deploy tools。
- `legacy_11`：未对齐 OpenClaw 之前的 unified_runner prompt + 11 legacy tools，用于 ablation / 回溯对照。

1667 数据当前测试产物：

```bash
python GeneralAgent/sft_training/convert_to_openclaw_compat.py
```

得到：

- 输入：`1666` 条
- 输出：`1595` 条
- bench 分布：`seta_synth=839, claw=356, swe_lite=191, tb2=133, sb_ns=76`
- 默认输出目录：`GeneralAgent/sft_training/datasets/20260506_sft_campaign_1667_openclaw_compat_exec_gated/`

**从这一版开始，LLaMA-Factory export 应该读 `_openclaw_compat*_gated/sft_messages.jsonl`，不要再直接读 hindsight 原始文件。**

---

## 5. 转 LLaMA-Factory 格式

**脚本**：`../sft_training/export_llamafactory.py`

输入 sft_messages.jsonl（来自上一步），输出 LF 用的 OpenAI-format `.json` + `dataset_info.json`，做这些事：

1. **stringify content** — list-of-blocks 拍平成 string
2. **assistant content + tool_calls 合并** — 修过的 bug：当 assistant message 同时有 content 和 native tool_calls 时，把 tool_calls 用 `<tool_call><function=...>...</function></tool_call>` XML format **append 到 content 末尾**（之前 codex 写的版本是 either-or，会丢 tool_calls 当 hindsight 把 content 填非空时）
3. **merge consecutive same-side** — 连续 user/tool 或 assistant/assistant 合并
4. **enforce alternation** — odd index = user/tool，even = assistant；不满足的轨迹整条丢弃
5. **拼出最终 messages** — system 单独一条 + 交替对话

**产物**：
- `GeneralAgent/sft_training/llamafactory_data/<RUN_NAME>/<dataset_name>.json`（list[messages]）
- `.../dataset_info.json`（LF 用来注册数据集名）

`<dataset_name>` 跟 yaml 的 `dataset:` 字段必须对得上。

---

## 6. SFT 训练

**入口**：`ops/workflows/sft_training/run_9b_clean_plus_claw_lora.sh`（历史 LLaMA-Factory 脚本已归档）

每个脚本对应一份 yaml `configs/qwen35_9b_lora_*.yaml`。LoRA 关键设置：

| 参数 | 当前默认 | 含义 |
|---|---|---|
| `lora_rank` / `lora_alpha` | 32 / 64 | 容量 |
| `cutoff_len` | 49152 | seq 长度 |
| `enable_liger_kernel` | true | fused linear cross-entropy 干掉 vocab=250k 的 logits OOM |
| `deepspeed` | `ds_z3_config.json` | model+grad+optim 全切，4 卡 H800 LoRA 友好 |
| `use_reentrant_gc` | true | 跟 ZeRO-3 + grad checkpointing 兼容（避开 metadata mismatch） |
| `template` | `qwen3_5_nothink` | LF 内置，匹配 Qwen3.5-9B base model |
| `num_train_epochs` | 5 | 1015-1700 量级数据下经验值 |
| `save_strategy` | `epoch` | 每 epoch 末 checkpoint |

**Env 要求**：
- `DISABLE_VERSION_CHECK=1`（因为我们用 transformers main 而不是稳定版）
- `LLAMAFACTORY_ALLOW_TORCH29_CONV3D=1`（base 是 VLM 但纯文本 SFT 跳过 Conv3D 校验）
- `CUDA_HOME` = slime conda env（提供 nvcc）
- venv = `GeneralAgent/.venvs/llamafactory`（独立于 slime，避免污染）

**产物**：
- `GeneralAgent/sft_training/outputs/<RUN_NAME>/checkpoint-{N,2N,...}/adapter_model.safetensors` — 每 epoch 末
- `GeneralAgent/sft_training/outputs/<RUN_NAME>/adapter_model.safetensors` — final
- `trainer_log.jsonl`、`train_results.json` — 损失曲线和指标

---

## 7. Merge LoRA + Serve + Eval

**Merge**：`llamafactory-cli export <export_yaml>`，写到 `merged_models/<RUN_NAME>/`。完整流程脚本是 `ops/workflows/sft_training/run_sft_v2_serve_and_eval_chain.sh`：等 merge → 起 SGLang → 等 ready → 跑 quick30 holdout eval。

**Eval**：`ops/launch/run_quick_holdout_eval.sh` 用 `make_quick_eval_plan.py` 生成 30-task plan + `launch_trials.py` 跑。

**Dashboard**：`data_quality_dashboard.py` 解析 trajectory，给出 resolved / used_skill / strict_used_success / meta_talk 等指标，输出 `<run>/reports/data_quality_dashboard.{md,json}`。

---

## 8. 训练-推理 prompt 对齐（关键 ⚠️）

OpenAI `tools=[...]` 是请求级参数，**不在 messages 里**。之前的链路：

| 阶段 | 模型实际看到的 prompt | trajectory 保存的 |
|---|---|---|
| 采集 phase1 | system + user + (SGLang chat-template 注入的 tool-schema) | 只保存 messages — **不含** tool-schema 段 |
| 训练 LF | LF 渲染 messages — **不含** tool-schema | — |
| 推理 unified_runner | system + user + (SGLang 注入的 tool-schema) | — |

→ **训练时 ≠ 推理时**。模型在训练时学的 `<skill_reasoning>` prefix 在推理时被新插入的 schema 段干扰，触发率从 ~67% → 0%（v2 数据已实证）。

**修复**：`agent_loop.py` 用三态 env `UNIFIED_TOOLS_SCHEMA_MODE` 显式选模式（替代旧的二态 `UNIFIED_DISABLE_TOOLS_SCHEMA=1`）。同时，`convert_to_openclaw_compat.py` 和 unified runner eval 都使用同一份 OpenClaw-compatible `tool_schemas.py`，避免训练数据和推理 schema 内容不同。

| mode | 行为 | 适用场景 |
|---|---|---|
| `openai_tools`（默认） | 传 `tools=...`，SGLang chat template 自动注入 schema 块 | baseline / retrieval / 老的无 schema SFT 数据 |
| `none` | 不传 tools=、也不手动注入 | 老的无 schema SFT 数据训练后的推理（保留兼容） |
| `manual_schema` | 不传 tools=，但启动时按同 tokenizer 渲染同 schema 块前置到 system | **schema-injected SFT 数据**（augment_hindsight.py `--inject-tools-schema` 产物）|

向后兼容：旧的 `UNIFIED_DISABLE_TOOLS_SCHEMA=1` 自动映射成 `none`（字面意思一致）。**不要**用 `disable=1` 期望"自动注入 schema"——那是 `manual_schema` 才做的事。

**对齐保证**（用同一 Qwen3.5 tokenizer / chat template 渲染验证）：
- TRAIN（schema 注入数据 + LF 渲染）= INFER_OPENAI_TOOLS（baseline + SGLang 自注入）= INFER_MANUAL_SCHEMA（SFT v3 + agent_loop 手动注入）→ 三者 byte-identical (7519 chars)
- INFER_NONE 比上述短 6776 chars → 仅适合训练数据**也没** schema 的实验

**采集时**：runner system prompt 已通过 `openclaw_compat.py` 统一；历史数据则通过 `convert_to_openclaw_compat.py` 迁移。
**eval 时**：OpenClaw-compatible SFT 必须设 `UNIFIED_TOOLS_SCHEMA_MODE=manual_schema`；baseline / retrieval 可用默认 `openai_tools`，但如果要 byte-level 对齐 SFT prompt，也应使用同一 schema 和同一 system builder。

---

## 9. 一个完整的 OpenClaw-compatible 跑法（参考 1667 数据集）

```bash
# 假定已有 datasets/20260505_sft_campaign_1667_replace_run05/sft_messages.jsonl

# (1) hindsight augment（用 phase1 的 27B 共享端口 30000）
#     --inject-tools-schema 把 SGLang 在推理时自动注入的 schema 块前置写进 system，
#     这样训练数据 system 已带 schema，推理可走 manual_schema 模式对齐。
/path/to/conda/envs/slime/bin/python3 \
  GeneralAgent/sft_data_collection/augment_hindsight.py \
  --input GeneralAgent/sft_training/datasets/20260505_sft_campaign_1667_replace_run05/sft_messages.jsonl \
  --output GeneralAgent/sft_training/datasets/20260505_sft_campaign_1667_replace_run05_hindsight/sft_messages.jsonl \
  --api-base http://127.0.0.1:30000/v1 \
  --model qwen3.5-27b \
  --workers 8 \
  --inject-tools-schema

# (2) OpenClaw-compatible data migration
/path/to/conda/envs/slime/bin/python3 \
  GeneralAgent/sft_training/convert_to_openclaw_compat.py \
  --input GeneralAgent/sft_training/datasets/20260505_sft_campaign_1667_replace_run05_hindsight/sft_messages.jsonl \
  --output GeneralAgent/sft_training/datasets/20260505_sft_campaign_1667_openclaw_compat/sft_messages.jsonl

# (3) LF format export
/path/to/conda/envs/slime/bin/python3 \
  GeneralAgent/sft_training/export_llamafactory.py \
  --input GeneralAgent/sft_training/datasets/20260505_sft_campaign_1667_openclaw_compat/sft_messages.jsonl \
  --out-dir GeneralAgent/sft_training/llamafactory_data/20260505_sft_campaign_1667_openclaw_compat \
  --dataset-name agent_sft_campaign_20260505_1667_openclaw_compat

# (4) 启动 SFT（占 GPU 4-7，rank 32 / 5 epoch）
tmux new-session -d -s sft-v3 \
  "bash ops/workflows/sft_training/run_9b_clean_plus_claw_lora.sh"

# (4) 训完 merge + serve + eval（chain 自动化）
MERGED=GeneralAgent/sft_training/merged_models/qwen35_9b_sft_campaign_20260505_1667_hindsight_4gpu_49k_5epoch_r32_liger \
EVAL_RUN_ID=$(date -u +%Y%m%d)_quick_holdout_eval_v3_retrieval \
SERVED_NAME=qwen3.5-9b-sft-v3 \
bash ops/workflows/sft_training/run_sft_v2_serve_and_eval_chain.sh

# (5) eval SFT v3 时 UNIFIED_TOOLS_SCHEMA_MODE=manual_schema 是必须的，由 chain 自动设
#     （如果手动跑 eval，记得加）。baseline/retrieval 跑时不要设，让默认 openai_tools 生效。
```

---

## 关键文件目录速查

| 用途 | 路径 |
|---|---|
| 数据收集脚本 | `GeneralAgent/sft_data_collection/` |
| 训练 yaml/脚本 | `GeneralAgent/sft_training/configs/`, `GeneralAgent/sft_training/scripts/` |
| 训练数据 (sft_messages.jsonl) | `GeneralAgent/sft_training/datasets/<RUN_NAME>/` |
| LF format 数据 | `GeneralAgent/sft_training/llamafactory_data/<RUN_NAME>/` |
| 训练输出 (adapter) | `GeneralAgent/sft_training/outputs/<RUN_NAME>/` |
| Merged model | `GeneralAgent/sft_training/merged_models/<RUN_NAME>/` |
| SFT 流程入口 | `ops/workflows/sft_data_collection/*.sh` and `ops/workflows/sft_training/*.sh` |
| Phase2 worker | `ops/workflows/sft_data_collection/run_phase2_teacher_worker.sh` |
| Eval 入口 | `ops/launch/run_quick_holdout_eval.sh` |
| Merge+serve+eval chain | `ops/workflows/sft_training/run_sft_v2_serve_and_eval_chain.sh` |
| 数据质量 dashboard | `GeneralAgent/sft_data_collection/data_quality_dashboard.py` |
