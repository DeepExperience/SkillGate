# SkillGate 论文收尾实验清单（2026-07-21 v3.1，执行版）

前提（细节见 `docs/rl_log.md` 的 FINAL data protocol 条目）：

- **我们的方法（论文主行）**：direct-SFT9B + clean-oracle token-local action credit，final99 artifact owner 为 `selector-clean-oracle-action-credit-sft9b-hybridv8b0704d-20260716_121116`，主表 140/280。owner 名中的历史数据版本标签只作 artifact 定位；论文统一称最终 mixed-slate 数据，不做开发期 description 版本比较。
- **对照 RL（mixed baseRL）**：同初始化、同训练预算的 task-only run `mixed-skills-task-reward-v8prod-20260713_185407` final99，主表 122/280。论文不把两个 owner 名中的数据版本标签当成方法变量。
- 单次训练的结果就是论文结果，不加 seed、不为严谨性重跑。
- 评测协议统一（FINAL）：70 任务 ×4=280，快照 `eval70_final_v8prod_fixed4`，eval id `eval70-mixed-r4-4023950044`，入口 `ops/workflows/rl_eval/run_eval70_checkpoint_set.sh`，单行约 1 小时。
- **所有实验结果只更新 `z_cc_terminal_imgs/skillgate_paper_master.md`（总表文档），不再新建结果文档。**
- 资源：本机 = 两节点 16×H800 的 Ray 集群（<gpu-node-ip> + <gpu-node-ip>）；另一台 16 卡机在做 claw 数据补充（第四节），不要动。
- **代码安全红线（适用于全部实验）**：不得修改任何现有文件的现有功能——clean-oracle 主结果必须保持可复现（用户已 git 提交基线）。允许：新建脚本/新建目录/新建 owner experiment 目录；对训练 launcher 只用**环境变量覆盖**；如需在 slime env 装新 python 包（如 trl），先 `pip list` 核验缺失再装，并在总表文档变更记录里写明装了什么版本。

---

## 一、训练型 baseline（最先做；共 3 个，已定稿不再增减）

> 这三个回答审稿人最可能的三连问："训练数据里明明知道 gold，为什么不用监督学习？（→BC）为什么不用已发表的偏好学习方案？（→SelSkill 式 DPO）你的收益是不是只来自把 read call 从 task loss 里挖掉？（→masked-task-only）"。
> 执行顺序：BC 与 DPO 的数据构造 + 训练（8 卡，各小时级）→ 两行评测（各 ~1h）→ 挂 masked-task-only（16 卡 ≈2 天）→ RL 结束后清第二节的评测队列。
> 结果登记：三行都进总表文档的 T1/T2/行为表（占位行已建好）；执行日期只记在变更记录，不写进逐行实验条件。

### 1.1 Gold Selector BC（监督选择对照，自建）

**目的**：训练数据 gold 已知——直接用监督学习 teacher-force "第一步就 read(gold)"，看能不能达到 SkillGate 的选择行为和任务分。预期弱点（论文论点）：BC 只学"这题→选这个名字"的映射，没有对比推理，在 v8 难描述上泛化差、且没有"读完怎么用"的信号。这不是某篇论文的现成方法，论文中称 selection-turn BC。

**关键事实（已核查，照此实现）**：

- `train.parquet` 的 `prompt` 列**不是**成品模板文本，是 message dict 数组：`[{role: system, content: ~70k 字符（<tools> JSON 块 + OpenClaw system 文本 + <available_skills> 16 候选 XML）}, {role: user, content: 任务}]`，内容里没有任何 `<|im_start|>` 标记。渲染交给 LLaMA-Factory 的 `qwen3_5_nothink` 模板（注册于 `GeneralAgent/third_party/LLaMA-Factory/src/llamafactory/data/template.py:2105`），它与 RL rollout 的渲染逐字节一致——前提是 assistant content 以字面 `<think>\n\n</think>\n\n` 开头（thinkwrap 约定在数据里，不在模板里）。
- 训练环境**必须用** LF 专用 venv：`GeneralAgent/.venvs/llamafactory`（transformers 5.8.0.dev0 能加载 qwen3_5；含 trl 0.24.0 / peft 0.18.1 / deepspeed 0.18.9 / liger）；激活脚本 `GeneralAgent/sft_training/activate_llamafactory.sh`。**slime env 的 transformers 4.57.1 加载不了 qwen3_5，不要用它训练**（数据构造用 slime 的 pandas 没问题）。
- init 模型 = RL 同款 SFT9B：`GeneralAgent/sft_training/merged_models/qwen35_9b_sft_campaign_20260512_clean_plus_claw_thinkwrap_4gpu_49k_5epoch_r32_liger`（VLM 壳 `Qwen3_5ForConditionalGeneration`，嵌套 text_config）。
- target 回合的真实格式（从 rollout 已核实）：

```
<skill_reasoning>
…选择理由…
</skill_reasoning>

一句过渡 prose。

<tool_call>
<function=read>
<parameter=path>
/root/.claude/skills/<skill名>/SKILL.md
</parameter>
</function>
</tool_call>
```

**数据构造**（新建脚本 `GeneralAgent/rl_data_prep/build_selector_bc_dpo_data.py`，slime python + pandas）：

1. 读 `datasets/rl/parquet_4bench_final_hybridtrain_v8prodfixed4eval_20260720/train.parquet`（491 行）。
2. **优先收割真实的干净首回合**：把 `experiments/rl/runs/selector-clean-oracle-action-credit-sft9b-hybridv8b0704d-20260716_121116/segments/20260716_121116-initial/rollout_result/train/*.jsonl`（75 个文件 ×128 条；另有 `segments/20260718_1243-resume74-evalcontract/rollout_result/train/` 可补覆盖）按 `(reward.bench, reward.task_id)` 建索引，筛 `reward.selector_clean_oracle == 1` 且 read 的 skill 名 == 该任务 `extra_info.slate_gold_name` 的记录，取 `response` 开头到第一个 `</tool_call>` 为该任务的 target 回合（已核实：initial 段 5418 条干净记录覆盖 265/290 个任务）。覆盖不到的任务用固定句式合成（`<skill_reasoning>` 里引用 gold 的 description 一句话 + 上面的 tool_call 骨架）。
3. 输出 LF openai 格式：`{"messages": [system原样, user原样, {"role":"assistant", "content": "<think>\n\n</think>\n\n" + target回合}]}` 共 491 条，写到 `GeneralAgent/sft_training/llamafactory_data/20260721_selection_bc/` 并配 `dataset_info.json`（`"formatting": "openai"`，columns.messages=messages；照抄 `llamafactory_data/20260512_sft_campaign_clean_plus_claw_thinkwrap/dataset_info.json` 的样式）。
4. 安全过滤：导出前 grep 含字面 `<image>` 的样本直接丢弃并记录条数（LF 会 ValueError abort，历史踩过）。

**训练**（新建 yaml，照抄 `GeneralAgent/sft_training/configs/qwen35_9b_lora_campaign_20260512_clean_plus_claw_thinkwrap_4gpu_49k_5epoch_r32_liger.yaml` 作模板改）：

- base = 上面的 merged SFT9B 目录（在 SFT9B 之上继续训，**不是**从 Qwen3.5-9B base）；`template: qwen3_5_nothink`；LoRA r32/α64（与原 campaign 同法，训完 LF export merge）；lr 1e-4、2 epochs、cutoff_len 24576（prompt 实测 ~18k tokens + target ~1k）；deepspeed zero3 + liger；8×H800，预计 <1 小时。
- merge 后的 HF 目录 → `experiments/rl/runs/goldbc-selection-sft9b-20260721/model/exports/bc-lora-merged/`。owner 目录先建 `experiments/rl/runs/goldbc-selection-sft9b-20260721/experiment.json`（照抄 `experiments/rl/runs/reference-qwen3-5-9b-base/experiment.json` 的 schema 改 id/objective）——评测 wrapper 会校验它存在。

**评测**（现有代码）：

```bash
bash ops/workflows/rl_eval/run_eval70_checkpoint_set.sh \
  --group 20260721-goldbc-v8-fixed4-r4 --skill-mode mixed \
  --snapshot skill_libraries/snapshots/rl/eval70_v8prod_fixed4_deepfrozen_20260719/snapshot_eval70 \
  --manifest skill_libraries/snapshots/rl/eval70_v8prod_fixed4_deepfrozen_20260719/slate_manifest_eval70.jsonl \
  --workers 128 \
  --model goldbc-selection-sft9b-20260721 goldbc-sft9b-v8-fixed4 <BC导出的绝对路径>
```

（该 wrapper 生成的 `--group` 报告文件只作中间产物；**数字最终手工登记进总表文档**，登记后可留可删。）

### 1.2 SelSkill 式偏好学习（DPO，已发表方法的适配）

**目的**：SelSkill（arXiv 2606.00510，invoke-vs-skip 的偏好学习）是审稿人最可能点名的已发表可训练方案。忠实适配到我们的 16 选 1 环境：同一 prompt 下偏好 `read(gold)` 胜过 `read(misleading)`。论文中称 "SelSkill-style preference adaptation"（如实说明原文是 invoke/skip 二元，我们扩展为多候选）。

**数据**：与 1.1 同脚本产出。每任务用 5 个 `extra_info.slate_misleading_names` 各构 1 对：chosen = 1.1 的 target 回合（read gold），rejected = 同格式但 skill 名/路径/理由换成该 misleading → 491×5 = 2455 对。LF 偏好数据格式（converter 在 `third_party/LLaMA-Factory/src/llamafactory/data/converter.py:302-335`）：`{"messages": [system, user], "chosen": {"role":"assistant","content":"<think>\n\n</think>\n\n"+gold回合}, "rejected": {…misleading回合}}`，`dataset_info.json` 加 `"ranking": true`，写到 `GeneralAgent/sft_training/llamafactory_data/20260721_selection_dpo/`。

**训练**：LF 的 DPO stage（venv 已含 trl 0.24.0，**无需装任何包**）。base 与 ref 都是 1.1 同款 SFT9B；`template: qwen3_5_nothink`；LoRA r32/α64；β=0.1，lr 5e-6（LoRA 量级），1 epoch，cutoff_len 24576，zero3，8×H800，预计 2-3 小时。merge 后 → `experiments/rl/runs/selskill-dpo-selection-sft9b-20260721/model/exports/dpo-lora-merged/`，owner experiment.json 同 1.1 方式建。

**评测**：同 1.1 命令，换 group/owner/label（`20260721-selskill-dpo-v8-fixed4-r4` / `selskill-dpo-sft9b-v8-fixed4`）。

### 1.3 masked-task-only（机制消融，一次 100 步 RL，零代码改动）

**目的**：SkillGate = ①把 skill-read call 从 task loss 挖掉 + ②在 skill 名字 token 上加 clean selector credit。本实验只保留 ①，回答"收益有多少只来自去掉错误的 task credit"。

**做法（核查后修订：纯环境变量方案不通，需新建一个 profile 文件——新增文件不违反红线）**：

核查确认 Relax 损失栈本身完全支持 coef=0（`selector_action_grpo_loss.py:108-109` `loss = task_pg + coef*selector_pg`，coef=0 时 read-call 的 task mask 剔除仍生效于 `selector_action_credit.py:514-517`，无除零/断言，动态过滤不受影响）。但 ops 启动链有三个阻塞：① `run_rl.sh:52`→`lib/runtime.sh:12` 在 source profile 前会 **unset** `RELAX_SELECTOR_ACTION_LOSS_COEF`（shell export 到不了 profile）；② profile:116-120 的校验器硬性要求 coef==0.20，覆盖即 FATAL；③ prepare 步骤见 FINAL 目录缺 `build_report.json` 会调 builder，builder 拒写非空目录 → launch 挂掉（byte-copy 本身安全，不会被覆盖）。

因此正确做法是**新建一个兄弟 profile**（拷贝改三处，不动原文件）：

1. 新文件 `ops/workflows/rl_training/profiles/selector_clean_oracle_masked_task_only.sh` = 现有 `selector_clean_oracle_action_credit.sh` 的拷贝，改：coef 默认值与校验值 `0.20`→`0`；`DATA_DIR` 默认指向 `datasets/rl/parquet_4bench_final_hybridtrain_v8prodfixed4eval_20260720`；prepare 步骤替换为 sha256 校验（train=`6dd23508…`、eval=`4d6ebede…`，与目录 README 一致）+ 保留原有 smoke 检查。其余超参逐字不动（lr 1e-6、KL、n=8、batch 128、100 步）。
2. 启动（`EXPERIMENT_ID`/`RUN_NAME`/W&B 名会自动生成）：

```bash
export EXPERIMENT_BASENAME=selector-clean-oracle-maskedtaskonly-sft9b-finalhybrid-lr1e6
export RELAX_PIN_NODE_ACTOR=<gpu-node-ip>
export RELAX_PIN_NODE_ROLLOUT=<gpu-node-ip>
bash ops/workflows/rl_training/run_rl.sh selector_clean_oracle_masked_task_only
```

3. **禁止**绕过 `run_rl.sh` 直接 exec `Relax/examples/agent_bench/run_agent_grpo_9B.sh`——会丢 preflight、守护五件套、manifest 记录和 resolved-config 存档。

**执行前核验**：上一个占用本集群的评测/服务已全部退出（GPU 空闲、30000/30001/30100 端口无监听）；新 profile 与原 profile `diff` 仅上述三处。

**时长**：按 hybrid run 实测 38-43 min/iter，100 步 ≈65h（2.7 天），含一次可能的重启预留 ~3.2 天。训完用评测 wrapper 的 checkpoint 模式自动导出+评测：

```bash
bash ops/workflows/rl_eval/run_eval70_checkpoint_set.sh \
  --group 20260723-maskedtaskonly-v8-fixed4-r4 --skill-mode mixed \
  --snapshot ... --manifest ...（同 1.1） --workers 128 \
  --checkpoint masked-task-only-sft9b-hybridtrain-20260721 maskedtaskonly-final99-v8-fixed4 \
    experiments/rl/runs/masked-task-only-sft9b-hybridtrain-20260721/segments/<segment>/ 99
```

**登记**：训练期间顺手把 wandb 曲线名（experiment id 即 run 名）记进总表变更记录，供机制图使用。

### 明确不跑的方案（写进论文 related work 即可；均已网查核实）

- SRA-Bench（arXiv 2604.24594）：诊断 benchmark（5400 实例/636 gold skills），无训练方法；引用作动机（其结论"何时用哪个 skill 是瓶颈"支持本文命题）。
- SkillRL（arXiv 2602.08234）及 Skill-R1/ReSkill/SkillOS 等：做 skill 生成/演化/管护，不做选择质量；SkillRL 的训练配方 = 已有 mixed baseRL 行。
- Agent Lightning（arXiv 2508.03680）：其 LightningRL 在单一可训策略下把整条轨迹回报均匀分给各 turn——数学上就等于我们的 task-reward-only GRPO 行，论文一句话引用说明等价即可，审稿人无法要求单独跑。
- TRACE（arXiv 2607.13988）：并发工作（2026-07-15 挂出，ICLR 政策豁免），且其价值信号需要 gold answer 字符串的参考模型 log-prob，对执行/测试判分型任务不可迁移——引用并说明不可迁移性。
- SkillRouter（arXiv 2603.22455）：可训练的外部 reranker（1.2B，8 万 skill 库）。nice-to-have：若日程富余可在我们 slate 上微调一个小 reranker 补一行；不跑则引用并说明第二节的 training-free rerank 行已覆盖"外部选择"这条轴。
- SkillResolve-Bench（arXiv 2606.10388）：risky-sibling 检索 benchmark，引用其 sibling-confusion 框架。

---

## 二、统一评测行的执行记录（已完成；不再代表主表分组）

以下条目记录各 artifact 的生成方式。最终论文按证据角色重排：端到端训练方法进主表，credit 变体进 Ablation，prompt/router/oracle-only 进 Selector 专项。

1. **sft9b（mixed 条件）**
   - 目的：主表 RL 起点锚点行；最终统一评测值已登记为 104/280。
   - 怎么跑：`run_eval70_checkpoint_set.sh --model reference-qwen3-5-9b-sft sft9b-v8-fixed4 GeneralAgent/sft_training/merged_models/qwen35_9b_sft_campaign_20260512_clean_plus_claw_thinkwrap_4gpu_49k_5epoch_r32_liger`。
   - 需要做什么：纯现有代码。

2. **action credit 非 clean 版（v1）**
   - 目的：消融阶梯中间级——action credit 但不要求单次读对，检验多读是否会 game selector credit；最终结果已移入 Ablation。
   - 怎么跑：用 `selector-action-credit-v1-sft9b-v8prod-20260714_164200` 的 final_iter99 已有导出，同上跑一行。owner 名只作 artifact provenance；论文统一按最终 mixed-slate 口径报告，不把开发期 description 版本当作实验变量。
   - 需要做什么：纯现有代码。

3. **sft9b + 选择指令 prompt**
   - 目的：回答"一句 prompt 行不行"——system prompt 加"候选 skill 只挑一个最相关的读，读完就开始做任务，不要多读"。
   - 怎么跑：mixed 模式跑 sft9b + 该指令。
   - 需要做什么：小改——在 prompt 组装处（`retrieval_skill_inject.py` 的 hint 或 prompt_profile）加**可选**环境变量开关注入指令；默认路径行为必须不变（开关不设时与现在逐字节相同）。

4. **frozen router 选一注入（SFT9B router 与 27B router 两行）**
   - 目的：回答"为什么不用独立选择器模型"。router 只看任务 + 16 候选 name/description 选 1 个，只注入该 skill 给 sft9b 执行。SFT9B-router 是同预算对照，27B-router 是更强选择器参考。
   - 怎么跑：新建 router 脚本（~100 行）：读快照 `slate_manifest_eval70.jsonl` 的 16 候选 → 调本地 SGLang（先起 SFT9B 或 27B）让其输出选择 → 产出每 bench 的 retrieval jsonl（格式抄现有 `_retrieval_*.jsonl`，`reranked_top10` 里只放选中的 1 个元素，注入通路已验证单元素=top-1）→ `--skill-mode retrieve --retrieval-root <jsonl目录>` 跑评测。
   - 需要做什么：新建 router 脚本；评测现有代码。

5. **retrieval top-1 注入**
   - 目的：传统检索排序方案对照。
   - 怎么跑：用 `GeneralAgent/eval_scripts/skills_retrieval/` v6 管线的 `Qwen3Reranker.score_pairs` 对每任务 16 候选打分取 top-1，产同格式 jsonl 后跑评测。reranker 占 ~16GB 显存，先打分产 jsonl，再起服务评测。
   - 需要做什么：新建打分脚本；评测现有代码。

6. **oracle-only 注入上界（sft9b 与 SkillGate final99 两行）**
   - 目的：天花板参照——每题只注入正确的那一个 oracle skill（无选择问题），sft9b 行给上界，SkillGate 行验证训练没损伤 skill 利用能力。
   - 怎么跑：从快照 `slate_manifest_eval70.jsonl` 的 `oracle:[{name,path}]` 生成 gold-only jsonl，`--skill-mode retrieve` 跑两行。
   - 需要做什么：~30 行转换脚本。⚠️ **不要**用 `--skill-mode oracle` 旧默认——`eval70_oracle_selfread_20260612` 的 692 个路径全部失效且被静默丢弃（rl_log 2026-07-20 23:30 条目）；显式传新 jsonl，跑完顺手修 `run_eval70_model.py:64` 与 `specs/eval70_v1/spec.json:9` 的失效默认值（这是修 bug，不属于红线禁止的功能改动，但要单独 git commit）。

7. **corrected Claw147 mixed-slate 扩展评测**
   - 目的：在与 eval70 的 14 道 Claw 题零重叠的 147 题上检查 selector 行为与 pass@1。
   - 怎么跑：147 题 ×1 遍 ×每个模型，使用唯一一套审计修正后的 16-candidate mixed slate；结果单列在总表第七节，不与 14 题 ×4 合并。
   - 状态：当前 11 个可用模型已完成；masked-task-only 待 checkpoint 产出后补。

## 三、论文分析实验（离线或轻量评测；结果进总表文档第 4–6 节）

1. **credit-design ablation**：把 Slate Regret v2、轨迹级 behavior bonus、非 clean action credit 从主表移到独立 Ablation；统一展示任务结果、clean single-oracle、oracle/misleading exposure 与读取成本。
2. **selector 专项**：把 prompt、SFT9B/27B router、reranker top-1 和 oracle-only 从主表移到独立模块；先报 task outcome，再分别报 route truth 与 executor 实际读取行为。
3. **主结果 task-cluster bootstrap**：只比较 SkillGate 与同预算 mixed baseRL，70 个 task 为 cluster、4 repeats 整体重采样。
4. **同路径 executor 审计**：两种方法各 280 条轨迹，按 `(bench, task, 首读 skill 名)` 和完整有序读取序列建立共享 strata；先算 stratum 成功率，再在 task 内等权平均 route、task 间等权平均，明确报告 strata、task、raw trajectory N 和 task-cluster CI。两次独立 rollout 没有稳定 repeat id，不伪造逐 trial 配对。
5. **单-skill exposure 干预**：固定同一个 frozen direct-SFT9B，在 30 题 ×4 上对比 oracle-only / misleading-only / no-skill；它回答正确 skill 内容是否具有因果任务价值，不用于声称 SkillGate 训练改善了 executor。
6. **trajectory credit 稀释**：在 mixed baseRL 的 12,800 条训练轨迹中重建 E[advantage | oracle read] 与 E[advantage | misleading-only read]，展示 task reward 对 selector token 的错归责。
7. **身份泛化**：统计 train/eval oracle 精确名称重叠；主评测 70/70 为未见 oracle identity。
8. **行为成本**：只比较 SkillGate 与 mixed baseRL 的 reads、turns 和 tokens；serving 拓扑不同，wall time 不作方法结论。
9. **oracle 修复未触及子集**：在 53 个未修改 oracle 的任务上重算 SkillGate 与 mixed baseRL，检查主方向是否由正文修复驱动。
10. **context 压力与真实案例**：16 个完整 SKILL.md 的 token CDF，以及“读对但执行失败 / 读错但任务成功”的真实训练轨迹与 token-local mask 图。

明确删除的论文分析：开发期 description 版本训练曲线、support-collapse 版本对比、Oaxaca 式 routing/executor 标准化、base9 协议翻转和另一 description 版本伴随表。这些最多作为开发记录留在 `docs/rl_log.md`，不进入论文主文。

## 四、数据补充（另一台 16 卡机，用户安排中）

**Claw 其余 147 题的 oracle + misleading（最终论文 slate 标准）**：oracle 从 `oracle_skills_full692_20260612`（claw 145 个）起步，必须过 07-19 claw oracle 审计的 7 类缺陷清单（`z_cc_terminal_imgs/20260719_claw_oracle_skill_audit.md`）并对照各题 grader/mock server 核验；misleading 的实现 provenance 仍可追溯到 `archive/ops_workflow_cleanup_20260712/rl_eval/slate_hard_negative_misleading.py` + `make_hybrid_v8body_0704desc_slate.py`，论文中统一称最终 mixed-slate 标准。

## 五、可选（时间富余才考虑）

1. **sft9b no-skill 行**：无 skill 下界参照。1 小时。
2. **gold-absent 压力测试**：把 oracle 拿掉看是否乱读，结果进 limitation。1 行。
3. **候选顺序打乱重测**：换排列种子 1 行，排除位置记忆质疑。
4. **梯度方向审计**：固定 minibatch 分别 backward task/selector loss 看名字 token 梯度方向——三.6 已便宜覆盖主要信息，只在想给图加"梯度证据"时做。

---

当前执行状态：除 masked-task-only final row 外，主表、Ablation、Selector 专项和第三节列出的离线分析均已登记到 `z_cc_terminal_imgs/skillgate_paper_master.md`。后续重建统一运行 `python3 ops/workflows/rl_eval/build_skillgate_paper_analysis.py`，不得重新引入已删除的开发期版本比较。
