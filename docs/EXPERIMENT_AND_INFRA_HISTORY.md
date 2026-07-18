# 实验历程与结论（Skills-Augmented Agent Training 项目全史）

> 整理日期：2026-07-13。本文由原 Projects 工作区三个月的 RL/SFT 日志、组会材料、审计报告和实验产物汇编，并补入 07-12/13 的终测、实验目录迁移和新训练设计。交接库不保留那些分散的原始文档；本文是其浓缩后的长期实验簿。历史数字仍须按第六章《信息甄别表》判断口径，关键结论优先以保留的 owner-local eval metadata 和机器可读结果为准。

**读者对象**：leader 与后续可能接手本方向的同事。
**本文档定位**：两份交接文档中的实验与踩坑记录本——完整、诚实地呈现项目迭代、负结果、事故与修复，并显式甄别历史材料中已被推翻或过时的信息。

---

## 目录

1. 执行摘要
2. 按阶段的实验历程（六阶段）
3. 基建战役史（独立成章）
4. 方法论 insights 汇总
5. 踩坑大全
6. 信息甄别表（历史文档中不可信/已推翻的说法）
7. 未竟之事与建议路线图
8. 附录：关键 run 速查表 + 重要文档索引

---

# 1. 执行摘要

## 1.1 问题定义

给定一个人写 skill 库（结构化 procedural instruction）+ 交互式环境，训练小型 LLM policy（Qwen3.5-9B/27B），使其在 5 个 coding/system benchmark（SkillsBench / Terminal-Bench 2.0 / SETA / SWE-Gym / Claw-Eval，统一 OpenClaw 风格 9-tool 接口）上：(a) 判断当前任务是否需要 skill；(b) 选对 skill；(c) 正确执行；最终目标是把 skill 知识**内化**进权重，使部署时不带 skill 也能受益。技术路线：统一评测框架 → skill 检索注入 → SFT → RL（Relax/GRPO）。

## 1.2 最终结论（截至 2026-07-13）

1. **外部 skill 检索有小幅、分 bench 的真实收益，但远小于早期表面数字**。可信的净 uplift：27B TB2 +4.5pp（solo p8 同并发对照）、9B sb_ns +4.7pp（三臂齐）；Claw 早期 +3.7pp 被证明主要是并发数混淆（同并发下归零）；SETA/SWE 无信号。检索质量瓶颈在库覆盖与"何时读"的判断，不在检索算法本身。
2. **SFT 教会"怎么用"但教不会"何时用"**：SFT 后 P(成功|读 skill) 翻 2.2 倍，但读 skill 变成 token-level reflex（42.5% 的读后第一动作与 base 完全一样），不该读的也读、被误导。这直接驱动了 RL 阶段。
3. **skill 内化的 off-policy token BC 族在本设定下系统性失败，且原因已闭环**。M1 清洗 GRPO → hybrid BC → prompt-only pair BC → action-span BC → CompatTraj gap 过滤 → hard-span v1-v4 → BC60 退火，全部 ≤ no-skill RL 基线 39.6%（eval70 no-skill 口径）。07-02/03 审计给出闭环证据：六条实现链路无 bug；BC 监督 token 的 NLL 从 step0 就在地板（模型早"会"这些 token，缺的是行为分布迁移）；BC 有效梯度贡献中位仅 ~3.4%，代价是 10-19% 的 GRPO 稀释。对照文献总结出**五前置条件框架**（蒸馏目标在学生 support 内 / skill 是可压缩程序性知识 / action 短选择型 / 有步级 credit 护航 / 基线在"能力已具备"区间）——我们五条全不满足，失败是**可预判的设定问题，不是工程或运气问题**。OPSD（on-policy 自蒸馏）两版补齐了 reachability 三面：BC 学不会（不可达）、k1 放大可达的坏（skill 话术）、k3 压制不可达的好（action token）。
4. **两个新发现现象**（论文级素材）：① **teacher 自衰减**——共享参数 teacher 在 oracle prompt 下的 `<skill_reasoning>` 出现率 98%→31%、救活率 25%→17%，off-policy BC 信息量自动枯竭；② **毕业≠迁移**——BC 教过的 80 个任务关断监督后仅 52.6% 弱通过（无一全通过），且存活全部集中在程序性任务，精确内容型任务（sb_ns）全灭。
5. **唯一双超基线的正结果是 oracle-GRPO60（E1，hint-in-prompt）**：与 BC60 只差一个变量（被接受的 oracle 组进组内 GRPO 而非 BC），no-skill 41.1% vs 基线 39.6%、mixed 46.1% vs 44.3%。本质是把 oracle 当 **on-policy 探索/课程机制**而不是 token teacher。注意幅度 1.5-1.8pp 在 eval70 的 MDE≈7.4pp 下统计上不可判读，只有方向性价值。
6. **第二赛道（外部 skill 可靠性 / mixed-slate 选择判断）仍无正结果，但失败归因已由终测确认**。SlateRL v8prod 训成整体不读；hybrid v2 final99 也通过少读来规避误导，两个 snapshot 上 oracle/attributed-read 仅 37.1%/37.7%。separated-advantage 则把 strict read 推到 96.4%，但 oracle 和 misleading 同时上升，oracle/attributed-read 只到 44.8%，task pass@4 未超过同 snapshot 的 no-skill RL baseline。也就是说，“让模型多读”和“让模型会选”确实是两个问题；整轨迹行为分与组级 regret 都没有解决 selector credit。
7. **07-13 开始把下一轮问题拆成两个干净实验**：mixed-task-reward control 只用最终 verifier reward，测 mixed 环境本身是否会诱发选择变化；selector-action-credit 在普通 task GRPO 之外，只对 oracle-vs-distractor 的 skill identity token 给局部信用，避免再次把读行为奖励摊到整条成功/失败轨迹。二者截至本文整理时仅完成实现、静态验证和启动，不能写成结果。

## 1.3 项目研究价值定位

本项目最有价值的产出不是某个"涨点"，而是一条**审计级可信的完整负结果证据链**："人写 skill 内化的边界与修复"——五前置条件框架 + 四变体系统性负结果 + teacher 自衰减/毕业≠迁移两个新现象 + oracle-GRPO60 作为修复方向。这与 SIRI/Skill-SD/OPSD 等同期工作互补（它们对"人写外部内容能否内化"零证据）。第二资产是 128 并发本地 Docker RL 的完整基建体系与事故簿（第三章），第三资产是评测方法论（第四章）。

---

# 2. 按阶段的实验历程

评测口径约定（全文适用）：
- **eval70** = `datasets/rl/parquet_4bench_base_20260523/eval.parquet` 对应的冻结 70 个 heldout 任务（claw 14 / seta_synth 30 / swe_lite 10 / sb_ns 8 / tb2 8），2026-06-11 起标准为 **4 repeats = 280 trials**，pass = resolved，error 计入分母。
- eval70×4 的最小可检出效应 **MDE≈7.4pp**——任何 <7pp 的差距都应视为方向性而非结论性。
- 内部 eval（训练中 EVAL_INTERVAL=20 的 56 任务 no-skill smoke）是另一口径（n=1/task，抖动带 ±2-3 任务），不可与 eval70 混引。

## 2.1 阶段一（2026-04-13 ~ 04-20）：统一评测框架 v1→v6

**目标**：把 5 个异构 benchmark 统一到 OpenClaw 风格 9-tool 接口（`read/write/edit/apply_patch/grep/find/ls/exec/process`），验证统一接口不损失成绩。

**关键事件与结果**：
- 04-13 立项：三阶段 curriculum 蓝图（SFT warmup → 单 skill RL → multi-skill adversarial）。后见之明：只走到 Phase 1 + Phase 2 变体；PRM/step-level reward、主/子 agent 架构从未实现，reward 始终 outcome/verifier-based。
- 04-15 OpenClaw 源码考古确立核心架构原则："**不管训练源头是什么，rollout 的工具暴露层必须与 OpenClaw tool schema 一致**"。AWM 被判死（双层 meta-tool 嵌套与 OpenClaw 不兼容，改造=与设计对抗），04-20 正式放弃。
- **最大单点修复：PersistentShell**。最初每次 `exec` 用独立 `subprocess.run`，多步 shell 任务全崩；改为 ToolLayer 持有一个持久 bash 进程（cd/export/source 跨调用保持）后，**unified SETA 0.077 → 0.576（7.5×）**，pre-fix 数据全部作废归档。
- 04-20 首个 unified vs native 全量对照表：SETA 0.633 vs 0.600、TB2 0.326 vs 0.315、Claw 0.422 vs 0.412、SB w/s 0.157 vs 0.114——**统一接口不损失成绩**，这是后续一切实验的合法性基础。当时 SWE 双边全 0，后查明是 max_turns=20 天花板（v6 起默认 50）。
- Claw runner 经历三次大改：v1 全 0（grader 未接线）→ v2 重写（mock service + `scoring_components` 声明式打分，13.8%）→ host-mode 泄漏事故（~30 个临时文件写到 Projects/ 根目录）→ **per-task docker sandbox 成为 MANDATORY**。v6 claw docker 首跑 7.5% 的灾难是 3 个 infra bug（docker timeout / loopback audit / env_snapshot 漏拷），修复后 45.9-47.8%。
- v6 推理参数定型：**no-think + presence_penalty=1.5 + early-stop=3**；SGLang 固定 seed 1063810697；3-arm 对比绝不跨 SGLang 进程（跨进程 ±10-15pp）。

**当时结论**：框架可信，可以开始跑对照实验。**后见之明修正**：4.13"跨机 docker 架构没问题（~2% 开销）"的判断只在低并发成立，RL 阶段被彻底推翻（见第三章）。

## 2.2 阶段二（2026-04-17 ~ 04-26）：skill 检索 pipeline 与三臂对照

**目标**：验证"检索注入公开 skill 库"是否提升 pass rate。三臂设计：baseline（无 skill）/ retrieval（重排 top-10 注入）/ irrelevant（哈希随机负对照）。

**pipeline 演进**：
- v1（纯 embedding top-3）：**负收益**（SETA ret−irr −1.9pp、Claw −1.3pp）。诊断：对称 embedding（bge）不适合 task→skill 非对称检索；字面误匹配占 40%；模型根本不读（真读率 16-18%，"skill 提示是 prompt 末尾脚注"）。这个阴性结果是全项目的第一个转折点。
- v6（3-stage 定型）：Qwen3-Embedding-8B coarse top50 → Qwen3-Reranker-8B yes/no → top10 注入。入口 `GeneralAgent/eval_scripts/skills_retrieval/retrieve_v6_3stage.py`。
- v8（4-stage，+DeepSeek-V3.2 LLM judge 重排）：**尝试后弃用**——库扩到 1651 后 rerank 低分占比反升，38.4% 注入条目被 LLM 自评 0-1 分，无 uplift。代码残留 `retrieve_v8_4stage.py`。
- 库扩容：573 → 775 → 1143 → 1651 → 1849 → **2046**（04-24 修复丢失的 197 个 SkillsBench 原生 skill 后冻结）。SB 原生 skill 合入后 top-1 命中原生 9.1%→64.8%。**最终冻结产物 = 2046 库 + v7 pipeline 输出 `experiments/archive_sft_runs/20260424/20260424_v7pipeline_on_2046lib/retrieval_results/`**（原 `experiments/20260424/` 日期目录已整体归档至此），此后所有 retrieval 臂都用这个快照。注意早期文档仍写"573-skill library"，已过时（见 §6 #1）。
- 04-24 **sb_ns 真剥离**：此前 no-skills 变体环境里其实还留着 222 个 SKILL.md，全删后 baseline 才是真 no-skill。

**三臂最终证据表**（详细甄别见 §6）：

| 证据 | bench | 净 uplift | 可信度 |
|---|---|---|---|
| v7/v8 27B solo p8 | TB2 | **+4.5pp**（34.8→39.3，历史最高） | 较可信（同并发对照） |
| v9_9b 9B 三臂 | sb_ns | **+4.7pp**（2.3→7.0，irr 3.5） | 小样本但三臂齐、方向可信 |
| v9c 27B（2046 库） | sb_ns | retrieval 19.8%（真剥离后 baseline 近 0） | 方向性可信 |
| v6 27B（parallel=4） | Claw | +3.7pp | **被推翻**：v8 期同 parallel=8 对照 uplift=0，混淆变量是 worker 数 |
| v9_9b 9B | claw | +1.3pp | 噪声级 |
| 各版 | SETA / SWE | 无信号 | SETA 30 题 ±6.7pp 噪声；SWE 模型能力不够测不出 |

**当时结论**（04-26，官方）：retrieval 对 Claw/SkillsBench 有 modest 帮助，但幅度温和且模型不会判断何时读 → **转向 SFT/RL 训 skill-use 行为**。

## 2.3 阶段三（2026-04-27 ~ 05-18）：SFT——教会"怎么用"，教不会"何时用"

**数据采集**（`GeneralAgent/sft_data_collection/`）：Phase 1 学生模型 use-skill/no-use-skill 双 branch 各 4 rollout 收成功轨迹；Phase 2 对失败任务用更强 teacher（后期默认 glm-5.1，经 OpenAI-compatible API）+ reflection 重试。首批漏斗 320 trials → verifier 成功 40（40/317=12.6% 成功率）→ 去重/过滤后 **26 条 records**，驱动了后续大规模并发采集基建。

**6 步数据产线**：raw → collect_successes.py → augment_hindsight.py（27B/teacher API 反向生成 `<skill_reasoning>`）→ apply_think_wrap.py（空 `<think>` 前缀）→ filter_clean_dataset.py（去 stale URL/loop/`<image>` 字面量）→ export_llamafactory.py。配方迭代 1667→2042→2093→clean 1535→+claw 重采 173 = **最终 1708 records / 384 unique tasks**（`20260512_sft_campaign_clean_plus_claw_thinkwrap`）。

**训-推-部署三对齐重构**（5.13，重要）：system prompt 完整对齐 OpenClaw；tools schema 改 OpenClaw-full 28 tools 声明（实际可用 7 个）；skills 声明改 `<available_skills>` XML；grep/find/ls/apply_patch 机械替换为 exec 等价（OpenClaw 核心只有 read/bash/edit/write 4 工具）。**这次重构显著改变了 base 模型绝对分数**（27B SB baseline 15.1%→1.2%），跨重构前后的数字不可直接比。

**训练**：LLaMA-Factory LoRA r32/alpha64、49K ctx、5 epoch、9B 4 卡 / 27B 8 卡，merged 产物在 `GeneralAgent/sft_training/merged_models/`。9B merged 是后续所有 RL 的起点。

**结果**（5.13 组会，held-out 30 题 / full set）：
- 27B SFT_add_claw held-out 46.7%（baseline 30%）、9B 36.7%（baseline 20%）；SFT 后 strict-used 81-82%（base 仅 0.5-2.6%）、P(成功|读) 翻 2.2 倍。
- **但 full set 上 SFT 对 SWE/TB2 反而降**；only_sft 子集 SFT 28.3% vs base 36.8%（−8.5pp）；42.5% 读后第一动作与 base 完全一样（读=reflex）；同 task 同教师，27B teacher 100% 解决 → SFT 学生只复现 38%（62% 蒸馏 gap）。
- **核心诊断**："模型没学会该不该读——不该读的也读、被误导。RL with verifier reward 可解，SFT 范式解决不了。"

**SFT-9B eval70 基线 = 22/70 = 31.4%**（05-25，pass@1 单次；后 4-repeats 口径 28.9-29.6%）——此后所有 RL 的对比起点。

## 2.4 阶段四（2026-05-18 ~ 06-04）：RL 基建攻坚与训练学启蒙

**目标**：在 Relax（一个内部 RL 训练框架，已 vendored 于本仓 `Relax/`）上跑通 5-bench GRPO 并拿到第一条可信曲线。实际上这个阶段 80% 的时间花在 infra（详见第三章），但训练学上有四个里程碑式认知：

**(1) 5-bench 混训失败与 reward plateau 诊断**（05-19~05-25）。首个多机 run（v6→v15 链，ctx 40k→52k）跑到 iter159，reward 曲线平。逐层排查后的最终归因：
- GRPO 同质 group ~50%（8 样本全 0/全 1 → advantage=0），grad_norm=0 步占 27%；
- lr=1e-6 过小 + rollout_batch=4（每步只 4 个 prompt 组，比文献小一个数量级）；
- 5 bench reward 分布异构互相稀释；文献裁决：**没有已发表工作把 4-5 个异构多轮 agentic bench 混进一个 GRPO batch 跑出漂亮曲线**；
- **KL 分析乌龙**（本项目最大翻案之一）：曾诊断"KL 反向拉梯度"并建议 KL→0，后发现 0-159 步全程 `use_kl_loss=False`——**KL 从未生效，该分析全错**。教训：`kl_loss_coef` 数值 ≠ KL 生效，必须 `--use-kl-loss`。
- 首次 eval70 对比：SFT 31.4% / RL iter119 35.7% / iter159 32.9%。+4.3pp 当时带保留意见（tis≈1.0"模型几乎没动"），后被 4-repeats 部分确认（RL +5pp，主要来自 SWE，Fisher p≈0.451 不显著）。同时发现 strict skill-read 65.7%→20%——"嘴上说要读（`<skill_reasoning>` 100%），手上不读"，精确应验了 5.18 的 collapse 预判。

**(2) claw-only：skill-read 崩塌第一现场 + LLM-judge reward 被判不可训**（05-25~05-28，dynamic16 配置：RBS16/GBS128 + DAPO 式 dynamic sampling）。40 步内：raw_score 0.366→0.323、truncated 29.8%→94.5%、strict skill-read 36.6%→**0%**、repeated-line 0.849。铁证样本：step44 得 0.9 分但截断+216 个 tool marker+复读。**定性：claw 的 LLM-judge partial credit 系统性奖励"长篇总结/部分完成/反复确认"的坏轨迹**——claw 从此退出主训练组合（后续为"4bench"）。

**(3) SETA-only：KL 必需性 + true-KL 打通**（05-28~06-02）。无 KL 版 step7 死锁，前置退化触目惊心：格式错误样本 2%→43.8%、调用不存在的 bash 工具 0→7——**RL 诱发工具格式退化**。归因：SETA 二值 reward 不含格式信号，GRPO 把 advantage 摊给全轨迹 token，通过样本里 42% 也含格式错误被一起强化；文献能丢 KL 是因为大规模同质数据，我们是脆弱 XML 格式+稀疏 reward+小异构数据，**KL 从"可选"变"必需"**。开真 KL 过两关：GBS=128 全量 backward OOM（杠杆是 `ACTOR_MAX_TOKENS_PER_GPU` 6144→4096，不是 num_iters）+ NCCL bind 冲突（开 KL 激活 reference 独立 Megatron 组，Ray PACK 与 actor 同节点相撞 → `RELAX_PIN_NODE_<ROLE>` 节点 pin）。true-KL 后训练健康，但 response length 无限膨胀（step45 median 33k）→ 引入 **DAPO soft overlong penalty**，长度被压制。这个 penalty 后来被认定是 skill-read 崩塌的真凶（见阶段五）。

**(4) skill-group reward 与 4bench 首跑**（06-02~06-04）。`Relax/examples/agent_bench/skill_group_reward.py`：strict skill-read 检测（真读 SKILL.md 文件才算）+ 组内 read/no-read 子组方向性 bonus + 子组 advantage。4bench（491 任务，有 overlong penalty）首跑 skill-read 8 步崩塌（68%→0%）——第二次崩塌实证。**06-04 正式 run 四改动**：ctx 70k（CP=2）、**去掉 length penalty**、lr 1e-6、skill-read gate 30%（`RELAX_DYNAMIC_FILTER_MIN_SKILL_READ_FRAC=0.30`）。效果：strict skill-read 稳定 43-63% 不再崩。——这就是"overlong penalty 故意关 + skill gate 30%+bonus 是保护探索"这条项目约定的完整由来。
**后见之明（重要）**：06-11/06-14 消融证明 **bonus/gate 是冗余的**，保住读行为的真因是去掉 length penalty；6.8 组会把四项联合改动的功劳记给 skill reward 是错误归因（详见 §6）。

## 2.5 阶段五（2026-06-05 ~ 07-04）：skill 内化 RL 谱系——系统性负结果与闭环

### 2.5.1 铺垫：三条裁决性证据把问题从"使用行为"移到"skill 质量"，再移到"内化"

1. **组内条件 gap≈0**（06-10 修正指标 bug 后）：同任务对照下读 skill 收益 early −0.014 / late −0.007；06-11 梯度加权复算修订为"小正 +4.1pp（主要 seta）"。且"读了才做对"的任务只有 ~21 个（任务池的 3.4%）——**绝大多数任务根本不需要这 2046 个 skill**。
2. **27B oracle skill campaign**（06-09~06-15）：每 task 用 27B 依据 baseline 轨迹生成专属 SKILL.md。campaign-1（692 题）oracle 自选读 41.3% vs baseline 42.7% **持平**——因为 27B 自选读率仅 0.4-3.7%。但 **oracle preload（全文注入）51.9%（+10.4pp 全 bench 正增长）**；9B preload +14.6pp，**9B+oracle preload ≈ 27B 裸跑**。结论：好 skill 喂进上下文价值巨大，但"自主读"通路死于读取意愿；skill 质量被探索成功率卡死（baseline 有成功轨迹的题 oracle 生成成功率 77.2% vs 7.1%）。
3. **bonus/gate 消融**（06-12~06-17）：retrieval 通道去掉 bonus+gate 的 baseline run 跑满 100 步，strict 真读 0.39→0.60 温和上升**无崩塌**、eval70 无显著差异；oracle 通道 A/B 同样无差（50.4% vs 50.0%）。**裁决：skill 保护机制既非维持阅读所必需，也无 holdout 收益；历史崩塌是 claw reward 特性 + length penalty**。

同期建立两个关键锚点：
- **no-skill RL 基线**（删 prompt skill 段的纯 GRPO，iter99）：eval70 no-skill **39.6-40.7%**（两次重放，跨协议 ±3pp）；这是内化线所有对比的 bar。
- **6.22 大组会 3×3 矩阵核心矛盾**：no-skill-RL 在 no-skill 和 retrieve 两种 eval 下都最好（40.7%/42.1%）；oracle 训练+测试最好（50%+）但**部署不带 skill 时留不下来**（37.5-38.9%）→ 立项 **skill 内化**。

### 2.5.2 内化变体对照总表

| # | 方法 | launcher / 开关 | 核心机制 | 关键结果（eval70 no-skill 除注明） | 失败模式 | 淘汰原因 |
|---|---|---|---|---|---|---|
| 1 | M1 clean GRPO（shadow 尝试一） | `run_4bench_m1clean_oracle_from_sft.command.sh` / `RELAX_M1_CLEAN=1` | oracle rollout → 清洗成 no-skill transcript → 直接当 on-policy GRPO 样本 | 40+ steps 行为 collapse | 清洗后重算 logprob 只是反事实分数，state 错位破坏 on-policy 假设；清洗残留被正 advantage 放大 | shadow 数据是 privileged teacher data，不能硬套 PPO ratio |
| 2 | hybrid shadow BC/AWR（档位 B） | `run_4bench_m1_hybrid_shadow_grpo_from_sft.command.sh`，λ=0.4 | 491 no_skill GRPO + 491 清洗 shadow BC 混批 | 未到终测即改版 | BC dense 信号压过 GRPO；进 BC 的 43.1% 有清洗瑕疵；长度膨胀反馈环（cleaned 中位 9.2k→18.7k token） | response 手术清洗不可救 |
| 3 | prompt-only pair BC（atomic pair v1-v5） | `run_4bench_oracle_promptbc_pair_from_sft.command.sh` / `RELAX_PROMPT_ONLY_SHADOW_CLEAN=1` | oracle SKILL.md 全文 preload 进 system prompt，训练时只删 prompt 块不动 response；pair gate：no-skill 全错 ∧ oracle 有成功才 BC（λ=0.2）；spec8 投机超发 | 行为污染：no-skill 轨迹提 skill 比例 14%→74% 单调上升；no-skill GRPO 部分 reward 不涨 | 整条 response BC 把 oracle 条件下的话术风格克隆进模型 | 印证 SIRI"只 BC action span"的道理 |
| 4 | action-span BC（SIRI 式） | `..._actionspan_pair_spec8_coef02_...20260625_134936` / `RELAX_SHADOW_BC_ACTION_MASK=1` | 只对可执行 `<tool_call>` span 做 BC | **38.2%**（16 卡）/ 36.8%（single8）；oracle 口径 45.4% | NLL probe 证明确实内化了 teacher action（−0.066），但失败在控制流（重复/提前停/缺验证收尾），不是 action 语法 | 学会 token 不等于学会行为序列 |
| 5 | CompatTraj-BC（logprob gap 过滤） | `RELAX_COMPAT_*`，GAP 阈值 0.005/0.01 | 全轨迹 BC，按 teacher−student token gap 分桶加权 | single8 iter99 oracle 口径 47.1%（该批最高）；no-skill 未超基线 | 50 样本 gap audit：action 与 skill 话术在低 gap 区大量重叠，阈值 0.25 下 disclosure 误保留 66.3%——**gap 无法区分"该学"与"不该学"**（reachability mismatch） | 阈值救不了，必须显式 mask → hard-span |
| 6 | hard-span v1 | `..._hardspan_pair_...20260629_010750` | 硬规则：保 action span + 命中关键词短 reasoning + final，整块删 skill prose | 只留 iter29/34，未终测 | 太保守 ≈ action-span-only（已知无效） | 弃 |
| — | hard-span v2 | （未成 run） | 从 `<skill_reasoning>` 内 scrub 有用推理 | — | 该 block 本质是特权推理，删词也泄漏 oracle 思路 | 方向放弃 |
| 7 | hard-span v3(+sbfix) | `..._hardspanv3sbfix_...20260629_233428` + resume84 | 更宽的 action-grounded reasoning 逐行保留 | **iter64 38.2% / final99 36.4%**（训到后段在退化）；oracle 46.4-48.9% | 100 样本审计发现 10 例 privileged-source 泄漏（"oracle's working approach"、`/root/solutions` 等无"skill"字样的隐性捷径） | 泄漏防不胜防 → v4 |
| 8 | hard-span v4 + BC60 退火 | `..._hardspanv4_pair_bc60_then_noskill_...20260630_221518` + resume34 / `RELAX_PAIR_ORACLE_BC_UNTIL_STEP=60` | v4 加特权路径/prose/direct-skill-tail 三重 guard + action-first；步 0-59 pair BC、60 起纯 no-skill GRPO | **final99 no-skill 38.6%**（07-04 协议；07-06 新协议重测 41.1%）、oracle 47.5%、mixed **35.4%（全场最差）**；内部 eval99 25/56 | 行为面全成功（skill 话术排放→0、关断后无回潮、无"BC 拐杖依赖"），能力不涨；mask 四轮审计做到 strong-leak=0 也没用 | **38.6%≤40% 触发预注册关闭判据 → off-policy token BC 族 07-04 正式关闭** |
| 9 | OPSD v1（k1-in-advantage） | `..._opsd_pair_selfteacher_kl02_...20260702_155210` / `RELAX_OPSD_MODE=1` | 学生纯 no-skill GRPO；同权重自教师+oracle prompt 对已接受样本重打分，reverse-KL 折入 advantage（coef 0.2） | step32 用户干净停止 | **放大 SFT 教出的 skill register 话术**（38%→86%→100%，对不存在的 skill 发 read 调用） | k1 放大"可达的坏" |
| 10 | OPSD v2（k3 loss + skill mask，from base9B） | `..._opsd_k3_skillmask_...frombase9b_20260703_141808` | Skill-SD Eq.10-12 可微 loss + trust-region + skill-register token 蒸馏置零；从 base 9B 起训 | step20 停止：eval19 8/56 < eval0 10/56；zero-tool-call"礼貌规划不行动"退化加速 | teacher 在 oracle prompt 下不认可学生 no-skill 的 action token → 系统性教"你的动作是错的、你的散文没问题" | k3 压制"不可达的好"；与 k1 合成 **reachability 三面** |
| 11 | **oracle-GRPO60（E1）** ✅ | `..._oraclegrpo60_hardspanv4params_...20260704_010745` / `RELAX_PAIR_ORACLE_GRPO=1` | 与 #8 唯一差异：被接受的 oracle 组做**组内 GRPO**（on-policy，oracle prompt 保留）而非 BC | **final99 no-skill 41.1% / mixed 46.1%——全项目唯一双超基线**；tb2 28.1% 全场最高；eval19 25/56 十九步达到 BC 臂 99 步水平 | eval39/59 曾回落（质量波动非 infra），eval79 回升 | 存活；"oracle 当探索/课程，不当 token teacher" |
| 12 | cross-arm oracle-GRPO60（keep-all-pass） | `..._crossarm_keepallpass_...resume19_...20260708_020347` | oracle 组 advantage 改跨臂 `r_i − mean(no-skill组)`，且保留 oracle 全对组 | final99 no-skill **35.0%**（比 #11 差 6pp）、mixed 44.3%（vs 祖先 eval19 40.4%，McNemar p=0.135 不决定性） | 更强 counterfactual 信号反而更差；mixed 增益来自鲁棒性非选择 | 保守组内版保持最优 |

### 2.5.3 07-02/03 审计与关闭决策（证据链）

三份历史审计/决策文档构成闭环（原工作区文档，未随交接库保留；其关键证据与结论已并入本节）：
1. BC-vs-noskill 深审（audit_bc_vs_noskill_20260702）：8 路深审+8 路对抗复核，扫 ~6.5GB train dump。**六链路无功能性 bug**；BC 监督 token NLL 从 step0 在地板（~0.1-0.2 nats）且不降反升；BC 有效梯度贡献中位 ~3.4% vs GRPO 稀释 10-19%；eval70 MDE≈7.4pp，观察差距 1.4-3.2pp 统计上本就不可判读。顺带发现 3 个**观测层 bug**（均值型 W&B 标量被除以全 batch token 数失真 ×~3640、梯度 cosine 构造性恒 0、NLL baseline env 从未设置）——只影响读数不碰梯度，**至项目结束未修**。
2. BC60 中期报告（v4bc60_midrun_report_20260703）：BC60 关断三方互证干净生效；两个新现象（teacher 自衰减、毕业≠迁移）在此坐实。
3. 关闭决策（next_step_decision_20260703）：**最终判词**——"这种方式在我们环境不 work 成立，不是工程问题、不是运气问题，是可预判的设定问题。不要再在 mask/系数/退火日程上迭代。" 行动序列 E1（oracle-GRPO，实施 ✅）/E2（OPSD，实施两次均败）/E3（可内化性二分审计，未独立实施）/eval 扩容（未做）；明确不做干净两阶段。

## 2.6 阶段六（2026-07-03 ~ 07-11+）：第二赛道——外部 skill 可靠性（SlateRL）

**立项**（历史提案 skill_reliability_proposal_20260703，原文未随交接库保留）：训练 agent 在混杂 slate（每任务 16 条：oracle×1 + relevant×5 + irrelevant×5 + misleading×5）下做到 **no-regret 使用**。查新：SRA-Bench 等只有 benchmark 没有训练方法，三重空白干净。预注册最大风险："训成从不读"——**后来精确命中**。

**V0 mixed8 评测**（07-04，slate_skills_20260704 软负例）：8 模型 mixed eval70×4：

| 模型 | mixed ALL | strict read | oracle 读 | misleading 读 |
|---|---:|---:|---:|---:|
| **noskillRL9b** | **124/280=44.3%** | 90.7% | 33.2% | 52.5% |
| base27b | 41.4% | 17.1% | 5.0% | 9.3% |
| oracle1baseline | 41.1% | 100% | 41.4% | 68.2% |
| oracle1skillaware | 40.4% | 98.2% | 35.0% | 62.9% |
| actionspan | 40.0% | 82.9% | 33.6% | 63.9% |
| sft9b | 37.5% | 95.7% | 35.7% | 58.6% |
| hardspanv4bc60 | 35.4% | 92.9% | 33.6% | 55.0% |
| base9b | 19.4% | 18.7% | 6.1% | 15.8% |

三个关键读数：no-skill RL 最高但"乱读且读了白读"（P(pass|读)≈P(pass|不读)）；base27b 几乎不读反而 41.4%；**所有 9B 检查点 misleading 读得比 oracle 多** → 判断力是真空白；oracle 训练的模型逢 skill 必读但不会判断。同时初版（0704）misleading 被判太弱：错误在细节不在行动框架，读了顺着走还能做对。

**SlateRL v1**（`..._slate_regret_pair_...20260704_232301`）：组级 regret GRPO（Δ=mean_slate−mean_noskill shift，coef 0.5）。final99 mixed 42.5% < bar 44.3% → 否决，07-07 停掉配套 control。

**hard-negative misleading 迭代 v3→v25**（07-06~07-08，27B 生成 + no-skill RL iter99 行为 screen，280 trials/版；当时的生成脚本未随交接库保留，配套审计/筛选脚本见 `ops/monitor/audit_hard_negative_slate.py` 与 `ops/monitor/select_hard_negative_hybrid.py`）：目标三重约束——够诱人被读、照做必挂、认真 gating 可分（description 保持逻辑可分性）。screen 结果（gap = P(res|oracle) − P(res|misleading)）：

| 版本 | P(res\|oracle) | P(res\|misleading) | gap | 判决 |
|---|---:|---:|---:|---|
| v3 | 41.8% | 51.9% | −10.1 | 拒（方向反了） |
| v4 | 41.9% | 34.8% | +7.1 | 不 promote，作中间基 |
| v5 | 39.6% | 42.3% | −2.7 | 拒 |
| v6 | 42.9% | 39.1% | +3.8 | 拒 |
| v7hy | 44.8% | 42.1% | +2.7 | 拒（投影过估） |
| **v8** | **51.4%** | **35.8%** | **+15.6** | **不 promote，定为后续基底（v25 失败后生产回退 v8 策略）** |
| v9 | 40.4% | 37.2% | +3.2 | 拒 |
| v10hy | 42.3% | 44.3% | −2.0 | 拒（反转） |
| v25（全量新 profile） | 39.4% | 39.2% | +0.2 | 拒 → 生产回退 v8 策略 |

v11-v22 为 v8 retry 任务集上的 profile 微调循环，接受率始终个位数——核心难点：no-skill 模型读了 misleading 仍能靠现场测试/调试恢复；hybrid/投影式挑选（v7hy/v10hy）对 4-repeat screen 过拟合。生产落地 `slate_skills_20260708_hard_negative_v8_production`（eval70 350/350、train 2455/2455 全 hard-negative）。v8 局限：weak gate 仅 41/70 任务通过、70 题中只有 6 题有"oracle-only 成 ∧ misleading-only 败"的干净对。

**SlateRL v8prod run**（`..._slate_regret_v8prod_...20260708_212906`，跑满 100 步，43h 健康收官）——**本阶段最重要的负结果**：
- final99 mixed eval70 = 103/280 = **36.8%**；oracle 读 32.5% < misleading 读 49.3%；**P(pass|read)=36.8% ≈ P(pass|noread)=36.7%——行为 gate 失败**。
- 训练中**读取坍塌**：完全不读率 43%→67%，第一动作=读 skill **11.5%→0.2%**。
- **07-10 三因归因**（历史归因文档 slate_regret_v8prod_failure_attribution_20260710，原文未随交接库保留；方法=数据盲测+代码精读+631 组全量数值回放）：
  1. **决策区分度≈0 且反向**（部分成立）：读后的效果区分度没问题（读 misleading −11~−12pp），但读前只看 name+description 无法区分——oracle 与 misleading 描述 Jaccard 中位 0.76，oracle 系统性更短更泛化（167 vs 216 字符），"挑最具体"启发式 238/238 全选中 misleading。
  2. **credit 不指向选择**（主因级）：组级 regret shift 是组内常数，对"组内谁选对"的信用严格为零；剩下的组内 outcome-GRPO 教的是"别读"（"决定去读"期望 advantage −0.017 vs 不读 +0.015；读中 misleading 概率是 oracle 的 3 倍）；选择 token 仅占轨迹 2-3%，被均摊稀释 30-50 倍。
  3. **分布结构**：5:1 诱饵 + 24% gold 缺席 + slate 在 parquet 层冻结（同任务 100 步逐字节不变→拒读可记忆化）。环境必要条件不满足：P(pass|不读)=0.67 ≥ P(pass|读 oracle)=0.656——**oracle 根本不值得读，credit 修得再好也顶不住**。
  4. 重要教训：立项文档 §4.2.3 精确预见了这个失败并给了解药（stratified advantage + 课程），**落地时被砍掉**，事后被归因为主因级。
- **修复方案 5 条**：① 成员级分层 advantage；② 成员级动作条件 regret；③ 选择 token 局部信用；④ 决策区分度环境侧修复；⑤ 任务分布偏向"oracle 必需"。优先级：⑤ 和 ② 是必要条件级。

**修复尝试（截至 07-11 20:05 日志截止）**：
- **bonuscompare**（`..._mixedskills_bonuscompare_...20260710_191150`，规定式行为分）：step9 主动停止，暴露三个系统 bug（shaped reward 污染 dynamic filter——52.1% 已收组本应 refill；W&B passrate 回退到 shaped reward 不可比；**no-read shortcut**：成功不读 1.35 > 成功读 oracle 1.30——奖励显式偏好不读，行为 step0→8 no-read 52→84）。修复已落地（`keep_raw_task_reward_nonzero_std`、passrate 恒用 raw_score）。机制可算：判断精度 ~26% 时读书是期望亏损，"学会判断"要先穿过被惩罚的谷底，on-policy GRPO 不会主动穿谷。
- **hybrid v2 gold-stratified**（`..._hybridv8b0704d_gold_stratified_...20260710_232029` + resume9）：misleading 用 v8 body（真毒）+ 0704 description（可学的选择特征）解耦毒性与可学性；成员级分层 advantage 补回；全 gold；训推一致 slate。**日志截止时在跑**（step~39，eval19 22/56、eval39 20/56 抖动带内；steps17-24 观察到"毒性鲁棒先于选择学习"的涌现——条件 gap 塌到 2pp 后 30-35 步重新拉开到 31pp；36-60 步是选择是否响应的决定窗口）。
- **separated-advantage**（`..._mixedskills_separatedcontinuousadv_...20260710_231945`，W&B fk463h2v）：行为效用 oracle+1/misleading−1/no-read−0.25 在成功/失败层内分别去中心，自适应缩放保证任意成功总优势>任意失败。**日志截止时在跑且趋势为负**：steps30-40 oracle 占全部样本 15.3%→9.2%（占读取样本 28.2%→19.6%），策略漂向 no-read——尽管 P(pass|oracle)=68.5% 一直最高。预注册门槛：oracle/read 份额不逆转下滑，不得声称 bonus 方案成功。

**07-13 终测裁决**：判断力可训性在这两种修复下仍未得到正证据。双 snapshot 的十行 eval70×4 全部 280/280 完成。separated final99 把 strict read 提到 96.4%，但 oracle/attributed-read 在两套 snapshot 都只有 44.8%，misleading read 也同步增加，task pass@4 只追平或低于 no-skill RL baseline。hybrid final99 则 strict read 降到 66.8%/72.9%，oracle/attributed-read 降到 37.1%/37.7%，靠少读规避损害。共同的七个 Claw task 遇到 grader HTTP 403，绝对分不完整，但每行暴露对称，within-panel 行为结论仍可信。

## 2.7 阶段七（2026-07-12 ~ 07-13）：结果收口、实验存储重构与局部 selector credit

### 2.7.1 双 snapshot 终测

终测把 no-skill RL baseline、mixed separated 的 eval59/final99、hybrid 的 surviving eval79/final99 同时放在 v8prod 和 hybrid-v8 两个冻结 snapshot 中。每个 panel 内 task、4 repeats、prompt、manual schema、OpenClaw adapter、model serving、seed 与并发固定；两个 panel snapshot 不同，禁止跨 panel 排名。

| snapshot | 模型 | task pass@4 | strict read | oracle read | misleading read | oracle / attributed read |
|---|---|---:|---:|---:|---:|---:|
| v8prod | no-skill RL baseline | 34/70 | 81.4% | 36.1% | 55.4% | 44.3% |
| v8prod | separated final99 | 34/70 | 96.4% | 43.2% | 61.8% | 44.8% |
| v8prod | hybrid final99 | 35/70 | 66.8% | 24.6% | 43.6% | 37.1% |
| hybrid-v8 | no-skill RL baseline | 36/70 | 82.1% | 36.4% | 53.6% | 44.3% |
| hybrid-v8 | separated final99 | 35/70 | 96.4% | 43.2% | 56.8% | 44.8% |
| hybrid-v8 | hybrid final99 | 35/70 | 72.9% | 27.5% | 48.9% | 37.7% |

结论不是“reward 没起作用”：separated 的行为信号非常强，成功把模型推向几乎必读；问题是它主要学到 reading propensity，没有学到 oracle-over-misleading identity selection。hybrid 的 regret 也有作用，但最稳定的 no-regret 策略是少读。这正是后来将 selector signal 限定到 skill identity token 的直接原因。

### 2.7.2 Canonical 实验存储

07-13 对原 Projects 实验树做了先归档、后迁移。原 rl 与 rl_eval 被完整冻结到 Projects 的 archive 后，live store 改成一个科学实验一个 owner：

    experiments/rl/runs/<experiment_id>/
      experiment.json
      segments/<run_name>/
      model/
      eval/<eval_id>/rows/<row_id>/

迁移盘点为 79 个历史训练实验、149 个 segment、3 个 reference owner、24 份数据、36 个 skill snapshot、12 个 HF export 和 121 个 eval row。105 个历史完整 eval row 保留 task-level metrics；16 个不完整 row 保持不完整，不伪造完成状态。resume/retry 只成为同一 owner 下的新 segment，eval 归模型 owner，跨模型对比表只作为派生产物生成（原工作区 z_cc_terminal_imgs 目录，未随交接库保留）。交接库没有携带 Projects 的大 archive，而是保留关键 owner 的轻量 provenance、compact eval row、selected.json 和样本轨迹。

### 2.7.3 两个新对照

mixed-task-reward control 使用 v8prod all-gold slate16，491 train/56 eval、每 task 8 rollouts，只做 raw verifier task-reward GRPO。所有 bonus、separated advantage、slate regret、pair、BC、OPSD 和 continuous reward 关闭；dynamic sampling 只看 raw reward 非零方差，W&B passrate 也只看 raw outcome。它回答“在 mixed skill prompt 下，即使不给行为信号，普通 RL 自己会把策略推向哪里”。

selector-action-credit 从 oracle-GRPO60 final99 初始化，使用 hybrid-v8 body 加 0704 description 的 all-gold slate16。task GRPO 排除 dispatched skill-read call token；辅助 loss 只更新 skill identity token。每组首次 oracle read utility=1，misleading/other/repeated/no-read utility=0，组内中心化，selector coef 0.20，LR 5e-7。task 与 selector token support 有显式 disjoint 断言。这个设计仍是待验证假设，不是已有正结果。

### 2.7.4 本阶段新增的工程教训

- pre-launch 失败也必须立即写 driver.log 和 segment manifest，否则“没进训练”会变成无记录黑洞；
- keep-best 与 pre-train eval0 的关联必须明确，未更新模型的 rollout0 应作为行为 baseline，而不是伪装成可保留的训练 checkpoint；
- checkpoint export 要临时目录加完整性检查再原子 rename；
- owner-local eval dry-run 必须先验证 owner experiment 存在，防止结果重新漂到中央垃圾场；
- raw pass、shaped utility、task advantage、selector advantage 必须分开统计，任何一个都不能借用另一个的字段名。

---

# 3. 基建战役史（团队最可复用的资产之一）

## 3.1 Docker 并发体系演进

| 代际 | 形态 | 并发上限 | 淘汰原因 |
|---|---|---|---|
| G1 | 远程 Docker 节点 dockerd via `ssh://<remote-docker-host>`（每条 docker 命令一个 ssh） | ~8-14 | SSH MaxSessions 撞墙；ssh 退出成 zombie 被 PID1（`ray start --block`）不回收 → **577k（57.7 万）ssh/nc 僵尸**打爆 pids |
| G2 | SSH 隧道 `tcp://127.0.0.1:2375/2376`（`ops/launch/resolve_rl_docker_host.sh` 自动起隧道） | ~32 | 远程 dockerd bridge 锁 + overlay2 全局锁 + 243ms RTT，O(N) 退化；187 个 stale 容器把一切拖慢；start p50 70s |
| G3 | **本地 overlay2 dockerd（特权 pod，data-root 必须本地 ext4 nvme）+ subreaper 包裹 + `--network host`** | **128** | 现役。start p50 5-8s；原 Projects 保存 551 个镜像 tar/约 730G，重启后约 25min 恢复。交接库只保留迁移 provenance 和关键外部 cache，不携带整套镜像 tar |

这些规则现已合并进本交接库唯一运行手册 **`docs/OPERATIONS_GUIDE.md`**：包括 cpuset 24-179、pids-limit 1024、timeout child cleanup、host network、代理、fsize、subreaper、disk/stale cleaner、preflight、teardown 和 resume。旧 rl_ops_guide 不再单独保留，避免两份规则漂移。

## 3.2 六大事故簿（现象 / 根因 / 最终防线）

| # | 事故 | 现象 | 根因 | 最终防线 |
|---|---|---|---|---|
| 1 | **CPU/pip 编译风暴**（06-06，step29） | 数千编译进程，trainer NCCL 线程被饿，30min watchdog DistBackendError | 数十容器同时重型 pip/cmake build，pip 并行=nproc(180) | `UNIFIED_DOCKER_CPUSET=24-179`（0-23 核独占给 trainer/SGLang/Ray），kernel 级隔离零训练开销；验证：steps30-38 干净跑 4.5h |
| 2 | **containerd-shim 熔毁**（06-08 首发；07-03 复发 198 shims） | 7513 shim vs 104 容器；netlink/cgroup 锁死锁，load 21012 硬 wedge | PID1=`ray start --block` 不 reap 孤儿 shim | `ops/launch/subreaper_exec.py` 包 dockerd（PR_SET_CHILD_SUBREAPER）+ `ops/cleanup/reap_orphan_shims.py` 兜底。⚠️ 07-10 发现通用 shim reaper 在 eval 128 并发下会误杀刚起步的合法 shim → **已改 opt-in 默认关** |
| 3 | **netns/rtnl_lock 内核卡死**（06-08/09 确诊） | load 5 万+、Dstate 堆积、`ip link` 卡死、`docker rm -f` 超时、节点级不可恢复 | 内核 5.15.0-124 unregister_netdevice/netns 引用泄漏：默认 bridge 每容器 netns/veth + IPv6 开启 + 80-98 容器/min churn × rm -f 强删 | 主容器 **`--network host`**（`UNIFIED_DOCKER_NETWORK_HOST=1` 默认开，根除 per-container netns）+ teardown 先 `docker stop -t 3` 再 rm + dmesg watcher 抓现行。修复后连过四个历史风暴存点验证 |
| 4 | **磁盘炸弹三代变体**（06-10~06-14） | 8 分钟写满 474GB；两次 pod 被 kubelet ephemeral-storage 驱逐 | ① 无 `count=` 的 `dd if=/dev/urandom`；② python open('/dev/urandom') 死循环（dd 匹配漏过）；③ 慢写 227GB 不触水位。另有 reaper 自身 bug：正则不识别 "TB" 单位（3.08TB 被解析成 0.0GB） | 三层闭合：容器 `--ulimit fsize=32G`（SIGXFSZ 只杀单命令）+ dockerd `--log-opt max-size=64m` + `ops/cleanup/reap_disk_bombs.py`（800GB 水位 + `/proc/<pid>/io write_bytes>20GB` 语言无关杀 + dirty≥50GB 哨兵 + 60s 无条件扫 ≥40GB 可写层）。政策：炸弹任务不入排除表，无界写按 fail 给负信号 |
| 5 | **Ray head GCS 压崩 / Serve ghost**（06-09、06-20） | pod 重启后新 run 永远 pending；连续 5 次启动失败 | head 仅 10.7GB 内存；`pkill` 杀 Serve actor 留 ghost controller 在已死节点；churn 压崩 GCS | teardown 用 `serve.delete(app)` 不 pkill；恢复配方（现并入 `docs/OPERATIONS_GUIDE.md` §8.3）：静默 10min → worker 远程 task 强健康门（≥4 次连过，driver-level ray.init 不够）→ 清 Serve 残留/ghost node。清理铁律：只做 run-id-scoped 清理，绝不广杀 Ray/Docker |
| 6 | **docker exec 回传静默丢**（06-11，hard354 期间） | 新栈段 **1713 个 trial 判分全 0** 报废（verifier/git diff 输出全空） | 自写 asyncio TCP↔unix 代理在单向 EOF 时关死整条连接（未处理 TCP 半关），一次性 `docker exec` 输出被吞 | `write_eof()` 半关修复；教训固化：**上新 docker 通路必须先烟测一次性 exec 三路回传**；后彻底单机化消除跨节点代理 |

另有 fork bomb（SETA 任务 1247 模型自写递归清理脚本 → ~95 万进程）由 `--pids-limit 1024` 根治——注意 pids-limit 只防容器内 fork bomb，对以上六类均无效。

## 3.3 排障方法论教训（比单个修复更重要）

- **06-06 回滚 repro 的转折**：连续 3 天 step21 必崩、加了十几个 gate 全没用之后，从 step0 无任何新 gate 裸跑 3 步全过——**"active128 不可行"的一系列结论全被推翻**，中间加的 gate 一律标 suspect。教训：排障陷入"越修越多"的螺旋时，先做干净回滚 repro。
- 类似的三连翻还有：netns 风暴"降并发根治"（被推翻）、"teardown 泄漏 shim 是主因"（被用户实证推翻：7552 僵尸 shim 冻结 15h 期间训练正常）。
- **verifier timeout 大排查**（05-24）同型：按 task 过滤越滤越多（618→341 行），每次都在下一步出新 timeout——治标；真根因是 RL 漏了 verifier_timeout_multiplier、SETA verifier 在线装包、187 个 stale 容器。由此确立**新任务准入协议**（build 审计→prebuild→patch 依赖→j32 preflight→结构性失败才进 `GeneralAgent/task_exclusions.py`）。
- 健康判据校准：load 高 ≠ 风暴（介入判据 = load 单调>1000 ∧ D-state 堆积 ∧ 容器冻结 ∧ unregister_netdevice firing）；`compute_ref_log_prob` 长数据单遍 30min 是真慢不是死锁（"多久没 completed 就判死"是错误判据）；监督脚本不能裸 grep `Traceback`（轨迹文本假阳）。

## 3.4 eval 侧基建

- **并发 64 标准的由来**：hard354 期间 200 并发 → llm_timeout 0%→24%、wallclock 超时 4%→60%——瓶颈是 **SGLang 吞吐 ÷ 850s wallclock 协议常数**，不是 docker；对 eval 是单向系统性压低。06-22 A/B：`--workers 64 --concurrent-trials` 41min（2× 提速无伪影）优于 128（每任务拖 2×、超时翻倍）。**eval70 标准 = `--workers 64 --concurrent-trials`**（8 卡机；后期部分队列用 128 + `--bench-cap claw=6`）。并发换算公式：`N ≈ N_baseline × (吞吐_now/吞吐_baseline)`，上量前先跑 150 trial 对照 error-rate。
- **hard354 1713 trial 报废事故**：见 §3.2 #6。
- **环境可比性铁律**（06-21 审计）：同一模型新旧环境差 4-5pp，真因是旧 eval 与训练同机抢资源 + claw host/docker 模式差异——**跨快照/跨环境的数字不可横比，要比必须同机同环境同进程重测**。
- eval 自动化：`launch_trials.py` 防 missing 自动重试（`--retry-rounds 2`）+ run.json manifest；`analyze_eval70_3tables.py` 防 retry 膨胀分母（每 planned leaf 只取 incremental 末行）；MISSING_TRAJECTORY 全量误判修复（patch_plan_env 硬覆盖 UNIFIED_EXP_VERSION 导致目录错位，改从 trajectory_path 反推）。

---

# 4. 方法论 insights 汇总（跨实验普适教训）

## 4.1 评测学

1. **先算 MDE 再下结论**（来源：07-02 审计）：eval70×4 的 MDE≈7.4pp（ICC≈0.6-0.7）；检出 +2pp 需 ~955 任务×4 reps。项目大量"A 比 B 好 2-3pp"的中间结论都在噪声内。这是内化线判不出胜负的根因之一。
2. **pass@1 单次 eval 不可用**（来源：06-11）：单跑 70 题噪声 ±4-5pp；ckpt34 的 37.1% 被证明是幸运单抽（真实均值 ~32.5%）。标准 = ≥4 repeats + 任务级配对 permutation/McNemar。
3. **同进程、同并发、同环境**：3-arm 绝不跨 SGLang 进程（±10-15pp）；v6 claw "+3.7pp" 教训 = 不同 parallel 数不可比；06-21 教训 = 不同机器/是否与训练抢资源不可比。
4. **error 计入分母**；主表必须给 N_total/N_pass/N_error（项目固定 metric 口径）。
5. **训练内指标 ≠ 能力指标**（来源：06-04/06-11）：dynamic filter 后的 raw_score 被结构性钉住（进步表现为任务"离开均值"）；filter 后 pass@8≈1 是构造性结果；动态采样训练 passrate ≠ 固定集泛化。判能力只看固定 held-out。
6. **内部 eval（56 任务 n=1）抖动带 ±2-3 任务**；单点 eval19 结论会被 eval39/59 推翻（oracle-GRPO60 就发生了）；eval39 凹陷是常态。
7. 跨协议不可混引：v4 final99 no-skill 有 38.6%（07-04 协议）与 41.1%（07-06 新协议）两个数，±3pp 漂移；关闭决策基于 38.6%。
8. 分 bench 子集只有 5-31 题，±1-2 题的挪动全是噪声——组会表格里多次被过度解读又收回。

## 4.2 RL 训练学

1. **KL 开关语义**（05-24 乌龙 + 05-28 打通）：`kl_loss_coef` 数值≠生效，必须 `--use-kl-loss`；开 KL ⇒ reference 独立 Megatron 组 ⇒ 必须与 actor 分节点 pin。KL 对保护脆弱格式（XML tool-call）在小异构数据上是必需的（SETA 无 KL 版格式错误 2%→43.8%）。
2. **length penalty 与 skill-read 崩塌**（06-08→06-11 归因反转）：soft overlong penalty 会奖励"更短的不读 skill 轨迹"，是两次崩塌的真凶；skill bonus/gate 被消融证明冗余。后续设计长度控制时必须兼容此约束（由来见 §2.4(4)）。
3. **dynamic sampling 三坑**：① reward=None 样本会打崩 filter（需 keep=False 补采）；② 必须加 max-reject 兜底否则全 0 组多时死锁；③ survival bias——模型变强后全对组被丢，过采样倍数 1.4→3.6×拖慢训练，且训练分布漂移。
4. **pair-gating 设计**（阶段五底座）：no-skill mixed→GRPO / all-pass→drop / all-fail→触发配对 oracle 臂；严格填组太慢，`RELAX_PAIR_SPECULATIVE_EXTRA_GROUPS=8` 有界投机超发是实用解。keep-all-pass 变体（保留最强内化信号）实测更差。
5. **reward plateau 多维诊断清单**（05-24/06-04）：同质 group 率、grad_norm=0 占比、unique prompts/步（GBS 是单位错觉：128 样本=16 prompts）、lr、任务混合是否中途变过（曲线可比性第一主因）、reward 连续 vs binary 分布。
6. **reachability 三面**（07-03 定理级）：oracle 条件轨迹的 token 在 no-skill 下不可达——BC 学不会（不可达）、k1adv 放大可达的坏、k3 压制不可达的好。任何 token 级模仿（BC/mask/gap/reverse-KL）都绕不开。**outcome reward > teacher preference；on-policy 采样 > off-policy replay。**
7. **组内常数 shift 对成员级选择恒零信用**（07-10）：组级 regret/shift 在任何成员间比较中严格抵消；选择类 credit 必须做到成员级/token 级。
8. **行为 bonus 无法替代判断力**（6/14 与 bonuscompare 两次独立证伪）；且 bonus 相对大小会造 shortcut（no-read 1.35 > oracle 1.30）。行为奖励与任务奖励必须账目分离——**passrate/filter 恒用 verifier raw_score**（bonuscompare 三 bug 教训）。
9. **环境必要条件先于 credit 设计**：oracle 必须"值得读"（P(pass|读 oracle) > P(pass|不读)），否则 credit 修得再好，"不读"仍是 reward 最优解，模型只是诚实地学到它。
10. claw 型 LLM-judge partial credit 不可直接当 RL reward（奖励冗长/复读）；verifier 客观性（test-case 判定）决定可训性。
11. Ray runtime_env env 白名单：任何要进 rollout/actor 的 env 必须加 `Relax/relax/utils/utils.py` 白名单 + `ray-job.sh`，只在 wrapper export **静默失效**——同型坑踩了 ≥6 次（wallclock cap、JINA、soft-overlong、cpuset、host-net/proxy、EXTRA_SKILL_ROOTS）。验证法：docker inspect 活容器 / 读 actor `/proc/<pid>/environ`。
12. 模式互斥必须显式 pin+raise（OPSD=1 泄漏会把 oracle-GRPO run 静默变 OPSD run 且无 crash）；委托链禁止无条件 `export VAR=...`（会静默 clobber 上游覆盖），一律 `${VAR:-default}`。
13. `update_kind` 是 provenance 标签不是 loss 开关，审计看 grpo_weight/shadow_weight/bc_enabled。
14. fully-async 下 `perf/actor_train_time` 含数据等待，不是纯训练时间；判卡死看 GPU util + log 增长 + py-spy。

## 4.3 数据学

1. **严格筛选 > 数量**：SFT 只收 verifier 成功轨迹 + 去重 + 长度过滤（320 trials→26 records）；后期 clean 配方（1535→1708）优于原始 2093。
2. **thinkwrap 契约**：SFT 数据首条 assistant 以空 `<think>\n\n</think>\n\n` 开头 ⇒ 下游 rollout 必须 `enable_thinking:false`，否则模型面对没学过的分布。
3. **schema 三态对齐**：schema-injected SFT 的 eval/runner 必须 `UNIFIED_TOOLS_SCHEMA_MODE=manual_schema`；base 模型默认 openai_tools——跨模型对比要显式统一。
4. `<image>` 字面量会让 LLaMA-Factory export ValueError abort；export 必须把 assistant content 与 native tool_calls 合并成 `<tool_call>` XML 否则丢调用。
5. hindsight reasoning（教"为什么"）缓解 first-action reflex，但也埋下了 skill register 话术（OPSD k1 放大的正是它）。
6. bench 官方设计（verifier 在线装包、宽 timeout）适合低并发评测，不适合高并发 RL——必须预烘焙（image commit 补依赖 / uv cache bind-mount / runtime install guard）。

## 4.4 skill 注入学

1. **oracle preload ≫ oracle 自选读 ≈ base**：内容再好，读取意愿为 0 就白搭（27B 自选读率 0.4-3.7%）。"怎么给"不是瓶颈（全文注入 vs 自主读 ±3pp），"给什么"才是。
2. **模型越小外置知识边际价值越大**：9B preload +14.6pp vs 27B +10.4pp；9B+oracle ≈ 27B 裸跑。
3. **description 与正文的信息差是决定性的**（SB 自 judge 6.5→preload 26.0）；description 注入 system prompt（fixeddesc 修复）值 +3pp。
4. **描述可分性是选择学习的前提**：v8 slate oracle/misleading 描述 Jaccard 0.76 且 oracle 更短更泛化——读前不可分，选择就不可学。"骗得过 judge"与"描述可分"存在结构性张力。
5. 库覆盖是检索的硬上限：2046 库对多数任务无相关 skill；"读了才做对"的任务仅 3.4%。
6. skill 注入必须校验真的进了容器（8-worker docker cp 曾静默失败 `retrieval_skills_injected=0`）；注入失败要 fail-hard（ABORT）不能 0 分入 reward。
7. oracle skill 目录名=task_id 是泄漏，slate 使用必须改名；deep-freeze 改目录名=改注入名，需 `--preserve-skill-names`。

---

# 5. 踩坑大全（症状级速查，供接手者避雷）

## 5.1 训练

- resume 三件套：`--load` 给 ckpt **根目录**（含 latest_checkpointed_iteration.txt，给 iter_ 叶子会静默把 start_rollout_id 归 0）+ `--start-rollout-id` + `EXPECTED_LATEST_CKPT` 校验；dataset cursor 从 `LOAD_DIR/rollout/global_dataset_state_dict_<k>.pt` 恢复。
- `--num-rollout` 是**终点步数不是增量**：resume 时给到目标终点，否则到点停、GPU 空闲被误判为崩溃。
- 完成的 run 同名重启会覆写 rollout JSONL 并回拨 marker——恢复后立刻核对 `latest_checkpointed_iteration.txt` 并留备份。
- `--max-actor-ckpt-to-keep 1` 与"留多 ckpt 挑点"冲突 → `preserve_ckpts_guard.sh` 用 `cp -al` 硬链保全（JuiceFS 同 FS 零额外占用；轮转后 preserved 副本成为 inode 唯一持有者）。
- 不完整 ckpt 特征：shard 尺寸异常（4GB/8.5-9.6GB 截断）、缺 .metadata——resume 前逐项核对（07-10 后 Relax 已内置完整性 manifest/O_EXCL/owner lease）。
- JuiceFS fsync stall：ckpt 保存卡 20min 是自愈型，**等待是唯一安全恢复**，绝不手改 latest 标记/杀 save rank；共享 FS EIO 会同时打 driver tee log 和 verifier——内层 tee log 移本地盘（`RUN_AGENT_GRPO_LOG_DIR`）。
- CP2 配方（16 卡 9B/70k）：actor[1,8] TP4×CP2×DP1 独占一台 + rollout[1,4]/ref[1,2]/fwd[1,2] 另一台；MTP 必须 override 关（否则 CP 走不支持的 P2P 路径 + resume optimizer shape mismatch）；torch compile 关；所有显式 master port 移出 ephemeral 段且分角色不相交；`NCCL_CUMEM_ENABLE=0`/RAS off/AF_INET；actor_fwd（TP2）不能直接载 TP4 分布式优化器 ckpt，需 ref_load 重定向。
- CP2 ckpt→HF 导出：必须 TEXT bridge（Qwen3_5ForConditionalGeneration，rope_theta=1e7），VL bridge 导出 serve 乱码；TE `_extra_state` 需 skip patch。工具 `ops/workflows/rl_eval/convert_cp2_qwen35_hf.py`。
- single8（8 卡）稳定配置：ctx40k/resp24k、TP2 五服务、RBS4/GBS32/iters1、mtpg1024；普通 GRPO 比 TIS-off 的 BC 臂显存更紧，`GRAD_REDUCE_IN_BF16=1` 是解。
- httpx 不认 CIDR 形式 NO_PROXY（10.0.0.0/8 无效）——内部 HTTP 走代理挂死，NO_PROXY 要加确切 IP；`hostname -I` 首位可能是 docker bridge IP。
- CC shell 里 `pkill -f` 会匹配自身 wrapper 自杀（exit 144），用 bracket trick。
- launcher `${VAR:-{json}}` 的 `}}` 会被 bash 截断 JSON——用 if-based 默认。
- eval/abort race：train abort 清理未完时触发 scheduled eval → 全 ABORTED 空行；已跑的 Ray actor 不热载修复，live run 的坏 eval 点须剔除。

## 5.2 评测

- `bash -lc` 登录 shell 会重置 DOCKER_HOST → preflight 假"0 镜像"（两次踩坑后在 `docker_cmd()` 内显式 re-export 修死）。
- ray job submit 的 driver 会落 10GB head 被 OOM 杀——改本地 driver + ray actor 起远端 SGLang。
- Prometheus exporter 端口冲突使 router 崩而引擎健康——router 可单独重启；`--prometheus-port` 已参数化。
- claw host-mode mock 服务 port_offset=0 时全绑 30000 撞 SGLang；claw 并发要 `--bench-cap claw=6`（本地桥接网段地址池会被孤儿 bridge 耗尽——只删孤儿桥重建 w1/w2/w4）。
- 外网依赖任务（claw T066 → r.jina.ai）是 eval 卡死惯犯，可手动关并记 missing_trials。
- error-only incremental 会阻塞自动 resume：移 `repair/` 重跑并强制 strict audit。
- Relax pass@k 按 raw_reward==1 判，严于 claw resolved≥0.75——W&B pass@k 低估 claw。
- W&B `skill_path_frac` 是宽松口径 ≠ strict 真读。

## 5.3 数据

- SETA 坏任务（25/244/436/729/1132 等）与 SB 坏任务（scheduling-email-assistant、multilingual-video-dubbing 等）在 `GeneralAgent/task_exclusions.py`——"不要把 mere timeouts 放进这里"。
- `str.splitlines()` 会按 U+2028 切分 → JSONL 审计假截断，只按 `\n` 切。
- 技能文本含 `\x` 时 `re.sub` 把替换串当转义 → 用 lambda 替换。
- 27B 生成的 `<skill>` 块可能与 XML wrapper 分离——parser 要放宽否则大批假失败。
- SFT 采集与 RL 的隐性差异清单（为什么 SFT 没暴露 RL 就炸）：verifier_timeout_multiplier 1.2×、低并发、失败样本被静默过滤——**RL 对 infra 稳定性要求比 SFT 高一个数量级**。

## 5.4 环境

- SGLang：启动加 `--random-seed 1063810697`；`SGLANG_DISABLE_CUDNN_CHECK=1`；warmup 需 NO_PROXY 含 0.0.0.0；引擎 18h 后可能僵尸且 Relax 无自动重启（health-check timeout 调 120s + max-fails 4 防过敏误杀）。
- W&B key 只在登录 shell 的个人 init 脚本里注入，tmux/Ray 非登录启动 → 整段 wandb 一条不记且不报错——launcher 必须 eval 注入 + 空值 fail-fast。
- 本机 `python3` 可能解析到无 ray 的 conda——launcher 需 `PATH=/usr/bin...` 前缀。
- 静态节点 pin 过期 → Ray 资源不可满足即退，改动态发现 alive worker。
- proxy env 会污染网络探针（squid 503 假响应）——诊断端口前先清 proxy。
- BuildKit build 阶段独立 netns：build.args 里 `host.docker.internal` 解析不到，用 docker0 IP。
- docker data-root 不能放 JuiceFS/overlay 之上（嵌套 overlay2 必败），必须本地 ext4。

---

# 6. 信息甄别表（历史文档中已被推翻/口径有陷阱的说法）

> 格式：文档/说法 → 为什么不可信 → 现在的正确认识。按危害程度排序。
> 表中提到的历史文档（原工作区的 CLAUDE.md、rl_log、组会材料、rl_run_gate.md 等）未随交接库保留，仅作历史指认，其可用结论已并入本文。

| # | 文档/说法 | 为什么不可信 | 现在的正确认识 |
|---|---|---|---|
| 1 | CLAUDE.md："573-skill library"、"merged/ 573 dedup'd skills" | 04-21~24 已扩到 ~2046 并冻结 | merged 库现存实测 **2045 个技能目录（其中 2043 个含 SKILL.md，与 embedding 索引 2043 条一致）**；历史日志/快照名口径为 2046（`20260424_v7pipeline_on_2046lib`），引用时以实测为准 |
| 2 | CLAUDE.md："verl 被 training/serving 使用" | verl 是用户 2025 年个人旧项目，未安装无引用 | verl 不属于本项目，不进团队 git |
| 3 | CLAUDE.md："Qwen3 8B/14B/27B"、`docs/current_task.md`/`docs/runbooks/` 等路径 | 模型线 4.27 起是 Qwen3.5-9B/27B；所列路径 05-26 清理后已不存在 | CLAUDE.md 的 Project purpose/Headless batch 段整体过时 |
| 4 | 4.21 组会："Claw retrieval +3.7pp 是最强信号" | v8 期同 parallel=8 对照 uplift=0，+3.7pp 主要是 worker 数 4→8 的混淆 | Claw retrieval 无可信 uplift；引用 v6 结论必须带此注记 |
| 5 | 4.13："SkillsBench skill 无增益（6.7% vs 6%）" | 被 timeout 污染 | 4.15 放宽 timeout 重跑 = 0.1345 vs 0.0721，有增益。教训：先修 harness 再下结论 |
| 6 | 5.24/5.25 组会："KL 反向拉梯度，P0 建议 KL 0.001→0" | 当时 use_kl_loss=False，KL 从未生效，分析全错（文档自认） | 历史所有"用过 KL"的 run 全是假 KL；真 KL 5-30 才第一次打通 |
| 7 | 6.8 大组会头条："skill reward+配平堵住崩塌，本周最直接成果" | 四项同时改动的联合效果被错误归给 skill reward | 06-11/14 消融：去掉 bonus/gate 读得更多（80.4%>74.3%）效果不差——真因是**去掉 length penalty**；bonus/gate 冗余 |
| 8 | 6.10："读 skill 组内收益≈0" 与其原始 gap 曲线图 | 原始曲线是空子组写 0 的成分假象；等权 gap 又被一边倒组孤样本拖负 | 6.11 梯度加权复算：平衡组小正 +4.1pp（主要 seta）。引用以 6.11 版为准 |
| 9 | "ckpt34 pass@1 37.1% 领先" | 单次 eval70 噪声 ±4-5pp；4 repeats 后 32.5% 反低于 rl119 | pass@1 单次口径的所有对比表都不可靠；标准=4 repeats+配对检验 |
| 10 | num_iters_per_train_update 的两版解释（"minibatch 数"/"决定 backward 显存"） | 5.28 codex 实测推翻 | 它只切 fully-async 数据搬运分块，**不切 actor 训练 batch**；OOM 杠杆是 max_tokens_per_gpu |
| 11 | 训练曲线类："5-bench 长链 reward 在上升"、"raw_score 诚实"、"pass@8≈1" | 中途改任务混合不可比；dynamic filter 后条件统计被钉住；filter 构造性结果 | 唯一干净显著上升曲线是 4bench 70k run（slope +0.094pp/step, p=0.025），且涨幅几乎全由 seta 驱动 |
| 12 | "retry 占 wall clock 88.5%" | 并行累计值（文档自注） | 不能当 wall clock 浪费直接引用 |
| 13 | v4 BC60 final99 no-skill 的 38.6% 与 41.1% | 两个协议（07-04 workers64 vs 07-06 workers128），±3pp 漂移 | 不可混用；BC 族关闭决策基于 38.6%。另：06-29 曾有一次 38.6% "重测"用错了 HF 权重（是 06-14 baseline 模型），与 40.7% 不是同权重重放 |
| 14 | 2026-06 期间 W&B 上所有 `hybrid_*`/`hard_span_*`/`compat/*` 均值型自定义标量 | model.py:597-613 把均值除以全 batch token 数，失真 ×~3640，**至项目结束未修** | 复盘这些曲线要乘还原系数；梯度不受影响 |
| 15 | "梯度 cosine=0 说明 BC 与 GRPO 无冲突" | 两支 loss 的 token support 构造上不相交，cosine 恒 0 | 该诊断全部无效 |
| 16 | oracle skill "每题已验证可解"标签 | hard354 上 baseline 8 重跑 85 题至少过一次 vs v1 89 题；v1 全部 2832 trial 中 success_strict_used_skill=0 | "验证通过"相当比例是 8 次重跑的运气；oracle **preload** 的 +10pp uplift 本身仍成立 |
| 17 | "TB2 26% vs 官方 41.6% 说明 harness 有问题" | 协议口径不同（30 turns/850s/no-think/64k vs 官方 256k+thinking） | 测的是"RL 协议口径下表现"，跨口径不可比 |
| 18 | rl_log 中 v3 中期 "step59 42.9%"、oracle-GRPO60 "eval19 44.6%" 之类单点内部 eval | 内部 eval n=1、抖动 ±2-3 任务，v3 终测回落到 36.4-38.2%，oracle-GRPO eval39/59 也回落 | 内部 eval 单点永远不下结论 |
| 19 | "swe_lite = SWE-bench Lite" | 实为 **SWE-Gym-Lite**（HF parquet 230 rows） | 命名陷阱，写论文/对外口径注意 |
| 20 | `docs/rl_run_gate.md` 的 Gate 0-4 | 文档自述"只是设计、无 entrypoint"，但 rl_log 把它当已有流程引用 | 实际执行靠临时脚本；接手时不要假设有现成 gate 工具 |
| 21 | v8prod mixed 36.8% vs bar 44.3% 的直接对比 | 两者评在不同 slate（v8 硬负例 vs 0704 软负例），模型×负例难度同变 | 不可直接比；但"没有任何 run 带判断力超过 44.3%"结论仍成立 |
| 22 | 早期文档："claw 不需要 docker 是优点"、"RL 用非 Docker+firejail" | host mode 曾泄漏文件到仓库根目录；安全与环境一致性优先 | v6 起 claw docker mode MANDATORY |
| 23 | rl_log/组会中 SFT eval70 分 bench 计数两套（claw4/14 seta9/31... vs seta16/30...）；6.4 文档 SETA 27B base 两套数（57.9% vs 61.3%） | 不同 split/复算口径并存，未标注 | 总分 31.4% 一致；引用分 bench 数必须注明口径来源 |
| 24 | 6.15 表 oracle RL ckpt79 "44.4%" 与 6.22 矩阵 50.7% | 不同 ckpt（79 vs 99）不同批次 | 不能混引 |
| 25 | "shim 泄漏 = netns 风暴" 的混同（部分记忆文档） | 是两个不同事故（shim 堆积≈06-08 前；netns/rtnl 06-08/09 确诊） | 防线也不同：subreaper+reaper vs host-net |

---

# 7. 未竟之事与建议路线图

## 7.1 交接时点的开放状态

1. hybrid v2 与 separated-advantage 已完成终测，均未解决 oracle-over-misleading selector；不要继续把它们当“待收割的在跑实验”。
2. mixed-task-reward control 与 selector-action-credit 已启动，但本文没有其训练终局；前者是干净普通 GRPO 对照，后者是新的局部信用假设。接手者必须从 owner segment、W&B 与固定轨迹重新核对当前状态，不能从本历史簿推断已成功。
3. E1（oracle-GRPO60）仍只有一次 41.1% 点估计，无独立重复、无扩容复核；back-generalization、privilege dropout 与 oracle annealing 未验证。
4. 07-02 的历史观测层问题中，旧 W&B 均值摊薄、构造性梯度 cosine 与 baseline env 对旧 run 的解释仍有效；新 selector 路径增加了独立指标，但不能反向修复旧记录。
5. sb_ns 0/32 死区贯穿主要 no-skill 结果，专项诊断未系统完成；精确内容型任务是否适合作为“内化”目标仍是开放问题。
6. test-oracle shortcut（读取 tests 期望输出后直接拟合答案）仍没有被可靠 guard 覆盖。
7. forced-choice separability 应在新 selector 实验前后补做：若只看 16 个 name+description 仍接近随机，局部 token credit 也缺少可学输入，必须回到环境设计。
8. 交接库不含大 checkpoint 与模型 shard；任何 resume 或 eval 要先按 HANDOVER_MANIFEST 恢复外部资产并核验 lineage。

## 7.2 建议路线图（按优先级）

1. **先修尺子再做实验**：eval 从 70 tasks 扩到约 300+ tasks，并保持 2-4 repeats；路线可以从 692 universe 重切或新增至少 230 个任务。eval70 的 MDE≈7.4pp，任何 <5pp 的方法改进都无法靠现有尺子可靠裁决。
2. **slate 线按 07-10 归因的必要条件级修复走**：⑤ 任务分布偏向"no-skill 难 + oracle 可救"（先离线筛 p̂_ns≤0.5 且 oracle 确实救活的任务子集）+ ② 成员级 regret。同时做 forced-choice probe 裁决 description 可分性；若不可分，走"读内容后对比"的环境设计（当前碰罚机制恰好禁止对比阅读）。更远一档：episode GRPO + 首次 read/高熵决策点的局部 counterfactual 分叉（先做 100-200 个 read 点的离线分叉审计验证相关性）。最终指标改报：正因果调用率 / 无必要调用率 / 有害调用率 / oracle 选择率 / distractor 拒用率——而非读取率。
3. **E3 可内化性二分审计**（07-03 决策排第 1 的离线工作，未做）：把任务按"程序性 vs 精确内容型"分型，sb_ns 类精确内容任务应从内化目标中显式踢出——"毕业≠迁移"现象已给出初步证据（存活全在程序性任务）。
4. **内化线唯一值得续的是 oracle-GRPO 族**：重复 E1、扩容复核、试 privilege annealing；**不要**再碰 mask/系数/退火日程（07-03 决策明令）。
5. **外部可靠性赛道对标 SRA-Bench**：本项目的差异化可守点——真实 coding/terminal 长程任务、15/16 混杂主动误导 slate、skill identity 信任判断、相对 no-skill 的 paired no-regret 目标、局部因果评测。
6. **论文定位建议**：不做“又一个内化框架”，定位 **“人写 skill 内化的边界与修复”**——五前置条件框架、审计级负结果、teacher 自衰减、毕业不等于迁移、oracle-GRPO 修复方向，以及“读倾向不等于选择能力”的双 snapshot 终测。选择性调用方向不能声称首创训练。

---

# 8. 附录

## 8.1 关键 run 速查表

> 本表中的长 run 名是迁移前历史身份。当前 canonical 位置统一为 **`experiments/rl/runs/<experiment_id>/`**：checkpoint 与 rollout 在 `segments/<run_name>/`，HF export 在 `model/`，eval 在 `eval/<eval_id>/rows/<row_id>/`。交接库只带关键 owner 的轻量记录，不带历史 checkpoint shard；旧 Projects 的 archive 才是完整追溯源。SFT 起点模型仍应恢复到 `GeneralAgent/sft_training/merged_models/qwen35_9b_sft_campaign_20260512_clean_plus_claw_thinkwrap_4gpu_49k_5epoch_r32_liger`（外部资产，按 HANDOVER_MANIFEST 恢复）。本表中 `z_cc_terminal_imgs/` 前缀的报告为原工作区派生表目录，未随交接库带出。

| run（简名 → 完整 run 名） | 日期 | 方法 | 关键结果（eval70×4 除注明） |
|---|---|---|---|
| 5-bench v15→159 → `full_train_5bench_workerlaunch_ctx52k_all4_v15_...20260521_170241` + resume 链 | 05-21~25 | 5-bench GRPO（假 KL） | iter119 35.7%（pass@1 单次）；skill-read 65.7→20% |
| claw-only dynamic16 → `run_claw_only_dynamic16_gbs128_from_sft`（W&B 7feefvsd） | 05-25~28 | claw 单 bench + DAPO dynamic | 40 步崩：skill-read 36.6→0%，truncated→94.5% |
| SETA true-KL → `seta_only_..._kl3e5_lr3e6_...20260530_015320`（W&B 0wipbu2h） | 05-30~06-02 | SETA 单 bench 真 KL + overlong penalty | KL 锚住格式；length 崩塌→penalty 压制 |
| 4bench skill-reward 首跑 → `4bench_factual_skill_gigpo_..._20260602_193946` | 06-02~04 | 4bench + bonus + penalty | skill-read 8 步崩（68→0%）；淘汰 |
| **4bench 70k 主线** → `4bench_cp2_70k_lr1e6_active128_nolen_skillgate30_from_sft_20260604_233222` + resume 链至 iter99 | 06-04~10 | ctx71680/CP2、去 penalty、gate30+bonus | ckpt34/84 pass@1 37.1；4-repeats ckpt99 34.3%；唯一显著上升曲线（seta 驱动） |
| baseRL 消融 → `baseline_noskillrw_nogate`（f9gq6 集群，06-12 点火） | 06-12~14 | 去 bonus/gate，其余同 | 35.4%（retrieve 口径 pass@4 最高）；strict 80.4%——bonus/gate 冗余定论 |
| oracle1 RL / oracle1-baseline 双跑 | 06-13~17 | 每 task 唯一 oracle skill ±bonus/gate | ckpt99 oracle 口径 46.8-50.7%；A/B 无差；收益=sharpening |
| **no-skill RL 基线** → no-skill parquet 0→100 | 06-17~19 | 删 prompt skill 段纯 GRPO | **no-skill 39.6-40.7% / mixed 44.3% / oracle 47.9%——全项目 bar** |
| M1/hybrid → `run_4bench_m1clean_oracle_...` / `run_4bench_m1_hybrid_shadow_grpo_...` | 06-20~22 | 清洗 GRPO→BC/AWR hybrid | collapse / 未终测改版 |
| action-span BC → `..._actionspan_pair_spec8_coef02_eval0_full_from_sft_20260625_134936`（16 卡）+ single8 版 | 06-25~28 | 只 BC tool_call span | 38.2% / 36.8%；oracle 45.4/43.2% |
| single8 no-skill 同参对照 | 06-26~28 | 8 卡同参纯 GRPO | 40.0%（no-skill）/45.4%（oracle） |
| CompatTraj single8 | 06-27~29 | logprob-gap 加权 BC | oracle 口径 47.1%（该批最高）；gap 过滤路线判死 |
| hard-span v3 → `..._hardspanv3sbfix_pair_...20260629_233428` + `resume84_start85_20260702_0018` | 06-29~07-02 | v3 mask BC | iter64 38.2% / final99 36.4%（oracle 48.9/46.4%） |
| **hard-span v4 BC60** → `..._hardspanv4_pair_bc60_then_noskill_...20260630_221518` + `resume34_start35_20260702_111727`（W&B b2za50qp/r1dvocww） | 06-30~07-04 | v4 mask + BC 60 步退火 | **final99 no-skill 38.6% → 触发 BC 族关闭**；oracle 47.5%；mixed 35.4% |
| OPSD v1 → `..._opsd_pair_selfteacher_kl02_...20260702_155210` | 07-02~03 | k1-in-advantage 自蒸馏 | step32 停：放大 skill register |
| OPSD v2 → `..._opsd_k3_skillmask_...frombase9b_20260703_141808` | 07-03~04 | k3 loss + skill mask, base9B | step20 停：eval19 14.3%，zero-tool-call 退化 |
| **oracle-GRPO60（E1）** → `..._oraclegrpo60_hardspanv4params_pair_spec8_eval0_from_sft_20260704_010745` | 07-04~06 | oracle 组进组内 GRPO | **no-skill 41.1% / mixed 46.1%（唯一双超基线）**；报告 `z_cc_terminal_imgs/20260706_eval70_oraclegrpo60_...` |
| SlateRL v1 → `..._slate_regret_pair_...20260704_232301` | 07-04~06 | 组级 regret GRPO（0704 slate） | mixed 42.5% < 44.3%；否决 |
| cross-arm keepallpass → `..._crossarm_keepallpass_...resume19_start20_rerun_from_sft_20260708_020347` + resume69 | 07-07~10 | 跨臂 advantage + 保留全对组 | no-skill 35.0% / mixed 44.3%；不如 E1 |
| **SlateRL v8prod** → `..._slate_regret_v8prod_pair_spec8_eval0_from_sft_20260708_212906` | 07-08~10 | 组级 regret + v8 硬负例 | mixed 36.8%；**selector gate 失败**；原始表 `z_cc_terminal_imgs/20260710_eval70_slate_regret_v8prod_final99_mixed_4repeats_{results,three_tables,slate_reads}.md`；归因见 §2.6 |
| bonuscompare → `..._mixedskills_bonuscompare_v8prod_allgold_from_sft_20260710_191150` | 07-10 | 规定式行为分 | step9 停；三系统 bug + no-read shortcut |
| hybrid v2 → `..._hybridv8b0704d_gold_stratified_...20260710_232029` + resume9 | 07-10~13 | 分层 advantage + 混合 slate | final99 通过少读规避误导；两 snapshot oracle/attributed-read 37.1%/37.7%，未学会选择 |
| separated-advantage → `..._mixedskills_separatedcontinuousadv_v8prod_allgold_from_sft_20260710_231945`（W&B fk463h2v） | 07-10~13 | 成功/失败层内行为去中心 | final99 strict read 96.4%，但 oracle 与 misleading 同涨；oracle/attributed-read 44.8%，task pass@4 未超 baseline |
| mixed task-reward control → owner `mixed-skills-task-reward-v8prod-20260713_185407` | 07-13~ | all-gold mixed prompt + 纯 task GRPO | 已实现并启动；本文时点无终局，不能写结果 |
| selector-action-credit → owner `selector-action-credit-v1-oraclegrpo60-20260713_193000` | 07-13~ | task GRPO + skill identity token 局部 selector loss | 已实现并启动；本文时点无终局，不能写结果 |

eval70 三臂总对照（数字出处：07-09 周报大表与 rl_log part3 §17，历史文档原文未随交接库保留；均 280 trials）：

| 检查点 | no-skill | oracle | mixed(0704) | mixed(v8prod) |
|---|---:|---:|---:|---:|
| base27b | 37.1% | 41.4% | 41.4% | — |
| base9b | 23.9% | 28.2% | 19.4% | — |
| sft9b | 29.6% | 39.6% | 37.5% | — |
| **no-skill RL（bar）** | **39.6%** | 47.9% | **44.3%** | — |
| action-span BC | 38.2% | 45.4% | 40.0% | — |
| oracle1-skillaware ckpt99 | 38.9% | **50.7%** | 40.4% | — |
| oracle1-baseRL ckpt99 | 37.5% | 50.4% | 41.1% | — |
| hard-span v4 BC60 | 38.6% | 47.5% | 35.4% | — |
| **oracle-GRPO60** | **41.1%** | — | **46.1%** | — |
| slate-regret v1 | 38.6% | — | 42.5% | — |
| crossarm keepallpass | 35.0% | — | 44.3% | — |
| slate-regret v8prod | — | — | — | 36.8% |

（注：sb_ns 在所有模型 no-skill 下恒 0/32；mixed 两列 slate 不同不可横比。）

## 8.2 交接库保留证据索引

原 Projects 的逐日日志、组会 PDF、审计草稿和 archive 没有复制进精简交接库；它们的结论已经被合并进本文。接手时使用以下仍存在的机器可读证据：

| 想了解 | 交接库中的位置 |
|---|---|
| 如何启动和恢复全部流程 | `docs/OPERATIONS_GUIDE.md` |
| 被保留的关键 experiment owner | `experiments/rl/catalog.json` |
| 哪些权重/rollout 被省略 | `experiments/rl/HANDOVER_MANIFEST.json` |
| 单个实验的起点、segment、配置、export/eval lineage | `experiments/rl/runs/<experiment_id>/` |
| 决定性的固定轨迹审计样本 | `experiments/rl/sample_trajectories/` |
| RL task split 与输入 parquet | `datasets/rl/` |
| oracle、v8prod、hybrid 冻结技能 | `skill_libraries/snapshots/rl/` |
| eval70 固定 protocol | `ops/workflows/rl_eval/specs/eval70_v1/` |
| 早期冻结 retrieval 结果 | `experiments/archive_sft_runs/20260424/20260424_v7pipeline_on_2046lib/` |
| local Docker 迁移 provenance | `experiments/infra/rl/local_docker_migration/` |
| 新实验 canonical launcher | `ops/workflows/rl_training/run_rl.sh` |
| 新 owner-local eval launcher | `ops/workflows/rl_eval/run_eval70_checkpoint_set.sh` |

（完）
