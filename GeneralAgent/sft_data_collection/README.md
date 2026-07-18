# SFT Data Collection Plan

这个目录是后续 SFT 数据采集的唯一迭代入口。目标是把 v9 skills 库和冻结 retrieval pipeline 作为固定环境，采集能训练 9B agent 判断并使用 skills 的成功轨迹。

快速迭代入口见 `FAST_ITERATION.md`：实时数据质量 dashboard、小 SFT loop、quick holdout eval。

## 决策

- 训练目标：让 `qwen3.5-9b` 在看到 retrieved top10 skills 后，学会是否读 skill、读哪个 skill、以及如何把 skill 转化成 agent action。
- 分类方式：采集和汇总阶段保留 task bucket；SFT 样本本身不显式训练分类标签。
- 数据优先级：`9B use-skill branch 成功轨迹` > `9B no-skill branch 成功轨迹` > `glm-5.1 teacher fallback 成功轨迹`。
- 去重策略：普通 `(task, mode, used_skill)` 成功组默认保留 2 条最短轨迹；`student_use_skill*` 分支或严格 `used_skill=true` 组默认保留 4 条，避免 skill 相关样本过少。
- baseline 作用：`student_baseline` 用来判断 task 是否 9B 不依赖 skill 也能解决；默认不直接进入 SFT，因为它没有同样的 retrieved-skill prompt 分布。
- teacher 作用：`teacher_retrieval_reflection` 只补 9B 全失败的覆盖缺口；它会看到 phase1 失败轨迹摘要和 verifier 反馈，导出时带 `model_role=teacher` 和 `task_bucket=teacher_only`，后续可以降权或单独 ablation。

## 目录结构

- `configs/default_collection_config.json`：冻结模型、skill 库、retrieval 文件、预算、split 来源。
- `build_splits.py`：生成固定 train/test split 和 freeze manifest。
- `make_trial_plan.py`：把 split 展开成 phase1 的每个 task × mode × trial 运行计划。
- `split_plan_chunks.py`：把 phase1 plan 按 task 分块，供 8 卡流水线 wrapper 边跑 phase1 边跑 teacher fallback。
- `make_teacher_fallback_plan.py`：对 phase1 全失败 task 生成 teacher fallback phase2 计划。
- `launch_trials.py`：安全执行 trial plan；默认 dry-run，必须显式传 `--execute` 才会跑。
- `collect_successes.py`：读取 trial 输出，判断是否真实使用 skill，生成 SFT JSONL 和 task buckets。
- `outputs/splits/`：固定 train/test split。Plan、logs、results、collected SFT 数据和 LLaMA-Factory export 统一写到 `experiments/<date>/<RUN_ID>/`。

历史 one-off 辅助脚本和已废弃配置已保守归档到
`archive/generalagent_cleanup_20260526/originals/GeneralAgent/sft_data_collection/`。
新采集链路应优先从 `ops/workflows/sft_data_collection/` 启动，核心逻辑仍在本目录维护。

## Split 策略

默认 split 由 `build_splits.py` 固化：

- `claw`：使用 `claw_161_t_series.txt`，均匀抽 16 个 heldout。
- `tb2`：89 个可运行 task，均匀抽 9 个 heldout。
- `sb_ns`：排除结构性不可运行的 `scheduling-email-assistant` 后，均匀抽 6 个 heldout。
- `seta_synth`：train 来自 `seta_300.txt`；test 是现有 `seta_baseline_30`，重叠 task 会从 train 移除。
- `swe_lite`：train 来自 `swe_lite_100.txt`；test 是 `run_unified_swe.py` 的 legacy `ALL_IMAGES`，重叠 instance 会从 train 移除。

生成 split：

```bash
cd /path/to/skillRL
python3 GeneralAgent/sft_data_collection/build_splits.py
```

输出：

- `GeneralAgent/sft_data_collection/outputs/splits/default/holdout_split.json`
- `GeneralAgent/sft_data_collection/outputs/splits/default/train/*.txt`
- `GeneralAgent/sft_data_collection/outputs/splits/default/test/*.txt`

## Trial Modes

Phase 1 每个 task 默认展开为：

- `student_use_skill`：9B，注入 frozen retrieval top10，并加隐藏 use-skill nudge；成功轨迹默认可进 SFT。
- `student_no_skill`：9B，注入同样的 frozen retrieval top10，并加隐藏 no-skill nudge；成功轨迹默认可进 SFT。
- `student_baseline`：9B，不注入 retrieval skills；默认关闭，只做 control，不直接进入 SFT。

默认两个 student branch 各 4 trials；`tb2` 因为单条较慢，当前覆盖为各 2 trials。Phase 2 由 `make_teacher_fallback_plan.py` 在 phase1 后生成：只对两个 branch 都未成功的 task，创建 4 条 `teacher_retrieval_reflection` trial；默认 teacher 是 MaaS `glm-5.1`。预算在 `configs/default_collection_config.json`。

## 生成 Pilot Plan

先跑 pilot，不要直接全量：

```bash
python3 GeneralAgent/sft_data_collection/make_trial_plan.py \
  --run-id sft_pilot_20260426 \
  --date 20260426 \
  --pilot-per-bench 2 \
  --benches claw tb2 sb_ns seta_synth swe_lite
```

输出：

- `experiments/20260426/sft_pilot_20260426/plans/sft_pilot_20260426.jsonl`
- `experiments/20260426/sft_pilot_20260426/plans/sft_pilot_20260426.warnings.txt`

注意：`seta_synth` retrieval 已更新为覆盖 300 个 train task；如果之后更换 train set，必须先重跑 retrieval 并更新 `configs/default_collection_config.json` 中的 `frozen_retrieval.files.seta_synth`。

## 执行 Trials

`launch_trials.py` 默认不会执行，只打印 dry-run。正式跑时必须加 `--execute`。

9B phase1 pilot：

```bash
tmux new-session -d -s sft-pilot9b '
cd /path/to/skillRL &&
python3 GeneralAgent/sft_data_collection/launch_trials.py \
  --plan experiments/20260426/sft_pilot_20260426/plans/sft_pilot_20260426.jsonl \
  --model qwen3.5-9b \
  --mode student_use_skill \
  --mode student_no_skill \
  --workers 8 \
  --execute
'
```

Phase 2 teacher fallback 需要在 phase1 完成后生成。默认从 `secrets/.env.secrets`
读取 `MAAS_API_BASE` / `MAAS_API_KEY`，teacher plan 指向 MaaS `glm-5.1`，但 plan
文件本身不保存 key：

```bash
python3 GeneralAgent/sft_data_collection/make_teacher_fallback_plan.py \
  --plan experiments/20260426/sft_pilot_20260426/plans/sft_pilot_20260426.jsonl \
  --out experiments/20260426/sft_pilot_20260426/plans/sft_pilot_20260426.teacher.jsonl

python3 GeneralAgent/sft_data_collection/launch_trials.py \
  --plan experiments/20260426/sft_pilot_20260426/plans/sft_pilot_20260426.teacher.jsonl \
  --model glm-5.1 \
  --workers 8 \
  --execute
```

覆盖 teacher endpoint 示例：

```bash
TEACHER_OPENAI_API_BASE="${MAAS_API_BASE}" \
python3 GeneralAgent/sft_data_collection/make_teacher_fallback_plan.py \
  --plan experiments/20260426/sft_pilot_20260426/plans/sft_pilot_20260426.jsonl \
  --out experiments/20260426/sft_pilot_20260426/plans/sft_pilot_20260426.teacher.jsonl
```

执行保护：

- 如果 plan 选择了多个模型，launcher 会拒绝启动。
- 如果 plan endpoint 当前 served model 和 plan model 不一致，launcher 会拒绝启动。
- retrieval mode 默认跳过缺失 retrieval entry 的 trial；不要用 `--allow-missing-retrieval` 跑正式数据。
- Claw trial 会被 launcher 串行化，避免多个独立 Claw 进程抢 shared mock infra。
- `status.jsonl` 会记录 `elapsed_sec`、`lock_wait_sec`、`subprocess_elapsed_sec`：
  分别表示 launcher 总耗时、等待 Claw/task lock 的耗时、真正子进程执行耗时。

## 汇总 SFT 数据

```bash
python3 GeneralAgent/sft_data_collection/collect_successes.py \
  --plan experiments/20260426/sft_pilot_20260426/plans/sft_pilot_20260426.jsonl
```

默认去重会提升 skill 相关样本权重：`--max-successes-per-task 2`，
`--max-successes-per-use-skill-task 4`。如果想更保守，可把后者设为 `3`。

如果已跑 phase2，先把 phase1 plan 和 teacher plan 拼成 combined plan，或者直接用 SFT wrapper 自动生成的
`experiments/<date>/<RUN_ID>/plans/<RUN_ID>.combined.jsonl`。

输出：

- `experiments/<date>/<RUN_ID>/collected/sft_messages.jsonl`：SFT 训练候选样本。
- `experiments/<date>/<RUN_ID>/collected/successful_trials.jsonl`：所有成功 trial 的 metadata。
- `experiments/<date>/<RUN_ID>/collected/task_buckets.json`：按 task 聚合的 bucket。
- `experiments/<date>/<RUN_ID>/collected/summary.md`：统计摘要。

## Task Buckets

`collect_successes.py` 会按 task 聚合为：

- `no_skill_solvable`：只有 9B no-skill branch 成功。
- `skill_helpful`：只有 9B use-skill branch 成功。
- `both_solvable`：9B use-skill 和 no-skill branch 都有成功。
- `teacher_only`：9B 全失败，teacher retrieval 成功。
- `unresolved`：没有成功轨迹。

这些 bucket 不写入 prompt，但写入 metadata，用于 sampling、ablation 和最终分析。

## Skill 使用检测

当前 `used_skill=true` 的主口径是严格的：必须在 assistant tool call 或 tool result 中看到 agent 实际访问 skill 文件路径，例如：

- `/root/.claude/skills`
- `/root/.codex/skills`
- `/root/.agents/skills`
- `/root/.gemini/skills`
- `/root/.factory/skills`
- `/root/.goose/skills`
- `/root/.opencode/skill`
- `SKILL.md`

这不会把 system prompt 里的 skill 列表本身算作使用 skill。`collect_successes.py` 也会额外记录 `used_skill_via_name`，表示 agent 在后续输出或 tool call 中提到了 injected skill 名字；这是弱诊断信号，不计入 `used_skill`。后续如果 runner 增加显式 skill-open event，这里只需要替换 `collect_successes.py` 的检测函数。

## SFT 格式

`sft_messages.jsonl` 每行：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": []},
    {"role": "tool", "tool_call_id": "...", "content": "..."}
  ],
  "metadata": {
    "bench": "tb2",
    "task_id": "...",
    "model_role": "student",
    "mode": "student_retrieval",
    "used_skill": true,
    "task_bucket": "skill_helpful",
    "loss_policy": {
      "train_on_roles": ["assistant"],
      "mask_system": true,
      "mask_user": true,
      "mask_tool_results": true
    }
  }
}
```

训练时默认只对 assistant turn 计算 loss。system/user/tool result 都 mask。

## 第一阶段建议

1. 固化 split。
2. 给 `seta_300` 重新跑 frozen retrieval；SWE 当前已覆盖 `swe_lite_100`。
3. 先做 pilot：每个 bench 2-5 个 train task，跑 9B 的 `student_use_skill` 和 `student_no_skill`。
4. 汇总看三件事：成功率、严格 `used_skill=true` 比例、轨迹长度和可训练性。
5. 8 卡环境用 `ops/workflows/sft_data_collection/run_sft_pipeline.sh`，让已完成 chunk 的 MaaS teacher fallback 和后续 9B phase1 并行。
