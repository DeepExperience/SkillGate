# SkillGate 代码库与全流程运行手册

更新时间：2026-07-13

这份文档是交接后的唯一运行手册，覆盖数据采集、技能检索、SFT、RL、
评测，以及两台 8 卡节点上的 Docker、Ray、SGLang 和保全措施。另一份
EXPERIMENT_AND_INFRA_HISTORY.md 负责解释过去三个月做过什么、为什么失败、
哪些结论已经被证伪。

本仓库是精简交接版：代码、关键数据、冻结 skill snapshot、少量决定性实验
记录在；大模型权重、完整 RL checkpoint、全量 rollout 和私密凭据不在。
datasets/、skill_libraries/snapshots/、experiments/ 等大体积输入与证据不进
git，由旁挂资产包恢复；恢复机制与清单见 assets/README.md。

# 1. 先建立正确的系统模型

整个项目不是一个单包程序，而是五条相互衔接的流水线：

1. benchmark 与 skill 数据准备；
2. teacher/student 轨迹采集与清洗；
3. LLaMA-Factory SFT；
4. Relax/GRPO RL；
5. owner-local eval 与跨模型派生汇总。

各目录职责固定：

| 目录 | 职责 | 是否允许放长期维护脚本 |
|---|---|---|
| GeneralAgent/ | runner、SFT 采集、数据转换、评测适配器 | 是，限模块内部代码 |
| Relax/ | RL 框架和本项目维护 patch | 是，但属于高风险修改 |
| ops/workflows/ | 标准数据、SFT、RL、eval 入口 | 是，优先放这里 |
| ops/launch/ | 通用 Docker、Ray、SGLang 启动工具 | 是 |
| ops/monitor/ | 通用监控与审计 | 是 |
| ops/cleanup/ | 定向清理工具 | 是 |
| experiments/ | 运行输出与机器可读证据 | 否 |
| datasets/rl/ | 跨实验复用的冻结 RL 输入 | 否 |
| skill_libraries/snapshots/rl/ | 冻结 skill 输入 | 否 |
| docs/ | 本手册和历史实验簿 | 否 |

最重要的边界是：实验输出属于 experiments，复用脚本属于 ops，跨实验共享
输入属于 datasets 或 skill_libraries。不要把一个实验目录变成第二份代码库。

# 2. 第一次接手与仓库自检

进入仓库后先执行：

    ./skillrl doctor
    ./skillrl recipes
    ./skillrl verify

含义：

- doctor 检查关键文件、坏软链接、残留的旧工作区绝对路径和基本资产；
- recipes 显示标准配方，以及因为缺权重而暂时不可执行的配方；
- verify 做更完整的静态结构检查；
- show 可展开某个 recipe 的实际入口和缺失依赖。

    ./skillrl show rl.mixed-task-reward
    ./skillrl show rl.selector-action-credit
    ./skillrl show eval.eval70-checkpoint-set

仓库换了绝对路径后：

    ./skillrl relocate
    ./skillrl doctor

canonical RL、eval、SFT 和 retrieval 入口会从脚本所在位置推导根目录；relocate
用于修正历史 metadata、parquet metadata 和较老 helper 中嵌入的绝对路径。

# 3. 环境、权重与 secrets

## 3.1 三套 Python 环境不要混用

slime 环境负责：

- SGLang serving；
- unified runner 与 eval70；
- SFT 轨迹采集；
- parquet、Ray 操作工具；
- skill embedding/reranker。

relax 环境负责：

- Relax 与 Megatron actor；
- GRPO/custom loss；
- Transformer Engine、FlashAttention；
- RL 侧训练 smoke。

GeneralAgent/.venvs/llamafactory 负责：

- LLaMA-Factory LoRA；
- merge；
- SFT YAML 对应版本。

环境重建时先以现有 env 的 freeze 和实际 import smoke 为准，不要仅根据最新
PyPI 版本猜测。RL 栈对 torch、transformers、Transformer Engine、
flash-attn、Megatron commit 很敏感。

最低验证：

    conda activate slime
    python -c "import torch, ray, pyarrow, transformers; print(torch.__version__)"

    conda activate relax
    unset LD_LIBRARY_PATH
    python -c "import torch, ray, transformer_engine; print(torch.__version__)"

LLaMA-Factory：

    source GeneralAgent/sft_training/activate_llamafactory.sh
    llamafactory-cli version

不要通过修改个人 shell init 脚本来修项目环境。项目变量放在启动命令、secrets
文件或标准 workflow 中。

## 3.2 外部权重恢复

交接库的 models/ 默认只有说明文件。至少按需要恢复：

    models/Qwen3.5-9B/
    models/Qwen3.5-27B/
    models/Qwen3-Embedding-8B/
    models/Qwen3-Reranker-8B/

最终 9B SFT 模型放在：

    GeneralAgent/sft_training/merged_models/
      qwen35_9b_sft_campaign_20260512_clean_plus_claw_thinkwrap_4gpu_49k_5epoch_r32_liger/

该 merged_models/ 目录不随仓库分发；按 §5.3 重新训练并 merge 后会生成，或从
发布的模型渠道恢复到同一路径。

RL 模型作为另一个实验的起点时，完整 HF export 属于原实验：

    experiments/rl/runs/<owner>/model/exports/<export_id>/

不要把只有 config.json 的半成品当作完整模型。至少检查 tokenizer、config、
generation config、所有 shard 和 index，且 export 目录必须由临时目录原子
rename 得到。

## 3.3 Secrets

复制 .env.example 为 secrets/.env.secrets，至少可能需要：

- WANDB_API_KEY；
- teacher/model provider key；
- Hugging Face token；
- 代理或内部 endpoint 凭据。

文件权限设为 600。不要把 key 写进命令行、run.json、resolved_config.env 或
driver.log。run_rl.sh 会优先从 secrets/.env.secrets 读取 W&B key，并拒绝启动
没有可追踪身份的 RL。

# 4. 数据与冻结输入

## 4.1 原始 benchmark

主要 benchmark payload 位于 datasets/：

- claw-eval；
- terminal-bench-v2；
- skillsbench；
- seta；
- swe-gym。

原始 task 目录按只读资产对待。若 task 因镜像、Dockerfile 或 verifier 确定性
损坏被排除，必须记录 bench、task id、理由、影响的 split 和这是临时还是结构性
排除。

基础 RL split：

    datasets/rl/rl_split_v2.json

基础可比 parquet：

    datasets/rl/parquet_4bench_base_20260523/train.parquet
    datasets/rl/parquet_4bench_base_20260523/eval.parquet

任何 count 问题都直接读取 parquet、split 或 build_report.json，不凭记忆回答。

## 4.2 当前 RL 数据

| 数据目录 | 用途 |
|---|---|
| parquet_4bench_factual_noskills_20260617 | no-skill task-reward control |
| parquet_4bench_mixed_skill_bonus_compare_v8prod_allgold_20260710 | 16 skill、每 task 有 gold、纯 task reward |
| parquet_4bench_mixed_skill_separated_continuous_advantage_v8prod_allgold_20260710 | 分离行为 advantage |
| parquet_4bench_slate_regret_hybridv8b0704d_gold_stratified_20260710 | hybrid paired regret |
| parquet_4bench_selector_action_credit_hybridv8b0704d_allgold_20260713 | token-local selector credit |

当前 all-gold mixed 数据的关键验收是：

- train 491 个 task、491 行；
- eval 56 个 task、56 行；
- 每行 slate size 16；
- gold_present 等于行数；
- gold_absent 为 0；
- prompt、manifest、skill roots 对得上。

每个 profile 在真正 launch 前都会重新跑 validate-only，不只是信任目录名。

## 4.3 Skill snapshot

冻结输入在：

    skill_libraries/snapshots/rl/

主要快照：

- oracle_skills_full692_20260612；
- eval70_oracle_selfread_20260612；
- slate_skills_20260704；
- slate_skills_20260708_hard_negative_v8_production；
- slate_skills_20260710_hybrid_v8body_0704desc。

已经被实验使用的 snapshot 永远不原地修改。新 description、misleading body、
排序或 manifest 都应生成新 snapshot，并记录 fingerprint。

v8 production 的设计动机不是“misleading 看起来更难”这么简单，而是要同时满足：

- 模型确实会读到 misleading，而不是一眼从 description 排除；
- oracle 与 misleading 的读取概率处在可比较量级；
- 读 oracle 后任务成功率明显高于读 misleading；
- description 仍有可学差异，否则 selector 无法从读前信息判断。

后续变体必须先离线复核这四点，再花 16 卡训练。

## 4.4 eval70 spec

eval70 的共享 protocol 位于：

    ops/workflows/rl_eval/specs/eval70_v1/

其中：

- spec.json 记录 protocol；
- tasks.tsv 固定 task 列表；
- source_plan_retrieval.jsonl 是重放来源（属大文件资产，随资产包恢复）。

它们是多个实验共同使用的输入，不属于任何一个模型的 eval 输出。

# 5. SFT 数据采集与训练

## 5.1 采集原则

标准采集过程：

1. 生成冻结 split 和 plan；
2. 为每条 trial 分配稳定 id；
3. student 执行，必要时 teacher 提供成功轨迹；
4. verifier 判定；
5. status 文件使失败和中断可恢复；
6. 只在明确规则下收集成功样本；
7. 记录原始 source、任务数、trial 数、成功数和排除数。

维护入口在：

    ops/workflows/sft_data_collection/

模块逻辑在：

    GeneralAgent/sft_data_collection/

优先扩展已有 run_sft_pipeline.sh、collect_and_export.sh 或 campaign workflow，
不要在 experiment 目录再写 run_final_v3.sh。

## 5.2 后处理顺序

最终数据链的语义顺序不能随意交换：

1. verifier-confirmed success filtering；
2. hindsight 增强；
3. tool schema / tokenizer compatibility 注入；
4. OpenClaw 兼容转换；
5. think-wrap；
6. 清洗与重复检查；
7. LLaMA-Factory 导出。

每一步都要保存输入列表、输出行数和拒绝理由。尤其检查：

- bench/task metadata 没丢；
- tool call 与 observation 配对；
- 无残余 image 占位符；
- system prompt 没重复注入；
- dataset name 与训练 YAML 一致；
- 同一 task 的重复样本策略明确。

最终保留的训练 JSON 位于：

    GeneralAgent/sft_training/llamafactory_data/
      20260512_sft_campaign_clean_plus_claw_thinkwrap/

## 5.3 SFT 启动

先只检查 recipe：

    ./skillrl show sft.final-9b

确认 base model、最终 1708-record 数据、LLaMA-Factory checkout、GPU4-7 和输出
目录后：

    ./skillrl run sft.final-9b --execute

canonical 脚本：

    ops/workflows/sft_training/run_9b_clean_plus_claw_lora.sh

训练结束后先验证 adapter，再 merge 到新目录；不要覆盖 base model。merge 完整后
用 SGLang 起服务，并跑固定 holdout。SFT 与 RL 的 tool schema、chat template、
disable-thinking 和 OpenClaw adapter 必须一致。

# 6. Skill 检索

主技能库：

    skill_libraries/merged/

主检索代码：

    GeneralAgent/eval_scripts/skills_retrieval/

重建与检索 recipe：

    ./skillrl show retrieval.rebuild-v6
    ./skillrl run retrieval.rebuild-v6 --execute

该 workflow 会占 GPU 并管理 SGLang，执行前先确认没有活跃 eval/serve session。
embedding index 必须写到消费者实际读取的位置，不能因为 cwd 不同悄悄生成一份
无人使用的新 pkl。

检索评测要区分：

- no-skill：不提供 skill；
- retrieve：模型看到检索到的 skill；
- oracle：按 task 提供正确 skill；
- mixed slate：提供 gold 与 misleading/distractor 集合，由模型选择是否读。

oracle 不是正常可部署检索，它是上界和因果诊断臂。

# 7. RL 训练

## 7.1 Canonical 入口与 profile

唯一标准入口：

    bash ops/workflows/rl_training/run_rl.sh PROFILE --dry-run

或 operator wrapper：

    ./skillrl show rl.mixed-task-reward
    ./skillrl run rl.mixed-task-reward

当前 profiles：

| profile | 数据/行为 | 训练信号 |
|---|---|---|
| no_skill | 没有 skill prompt | 普通 task-reward GRPO |
| mixed_task_reward | all-gold slate16，每 task 8 rollouts | 只有最终 verifier task reward |
| mixed_separated | all-gold slate16 | task advantage + outcome 分层的行为 advantage，coef 0.30，clip 0.40 |
| hybrid_slate | paired no-skill/mixed | regret coef 0.5 + stratified coef 1.0，clip 0.5 |
| selector_action_credit | all-gold hybrid slate | task GRPO + oracle-vs-distractor identity token local loss，coef 0.20 |

mixed_task_reward 是最干净的对照：代码仍统计 read/no-read/oracle/misleading 行为，
但所有行为 reward、bonus、regret、pair 和 custom loss 开关都必须为 0。

selector_action_credit 不是“给整条轨迹再加一个 reward”。任务成败仍用原始 GRPO；
selector 信号只作用在选择具体 skill identity 的 token，避免把失败的 oracle-read
整条轨迹抬到成功轨迹之上。其起点是 oracle-GRPO60 的 final99 export，而不是基础
SFT；所以交接库未恢复该 export 时 recipe 会明确显示 missing。

## 7.2 共同超参与拓扑

当前 16 卡标准拓扑为两台 8 卡 GPU 节点：

- actor：8 卡；
- rollout：4 卡；
- reference：2 卡；
- actor_fwd：2 卡；
- advantage：CPU。

共同训练口径：

- rollout batch 16 tasks；
- 每 task 8 samples；
- global batch 128；
- 4 次 train update；
- active env concurrency 128；
- Docker start concurrency 128；
- TP4、CP2；
- context 71680；
- actor max tokens/GPU 4096；
- log-prob chunk 4096；
- 默认 LR 1e-6；
- selector profile LR 5e-7；
- true KL coef 3e-5；
- 100 rollout steps；
- save interval 5；
- 普通 profile eval interval 20，selector 为 10。

不要把 128 并发理解为“总共只提交 128 个容器”。dynamic sampling 可以为了收齐
有效 group 继续请求候选，但必须有上限和明确的 accepted/rejected 统计。pair
atomic 类 profile 默认 speculative extra groups 8，用于减少等待，不能把这些候选
静默算成额外训练 task。

## 7.3 Dynamic sampling 的口径

普通 task-reward GRPO 只保留同 task 的 8 条中 raw task reward 有方差的 group，
即排除全 0 和全 1；这是因为全同 reward 没有组内学习信号。

关键规则：

- dynamic filter 看 raw task outcome，不看人为行为 bonus；
- W&B 的 passrate 仍用原始 verifier 成败，不混入 bonus/advantage；
- accepted group、rejected group、force-accept、aborted 必须分开统计；
- mixed separated 与 selector profile 禁止悄悄 force-accept 无信号 group；
- paired profile 以完整 pair 为原子，不可只收一边；
- 动态采样改变了实际训练 task 分布，比较实验时要报告 accepted task 分布。

## 7.4 启动前五道门

1. 身份门：EXPERIMENT_ID、RUN_NAME、W&B run id 唯一且可追溯。
2. 数据门：profile 的 validate-only 和 smoke 全过。
3. 拓扑门：Ray 恰好看到两台 live GPU node、总 16 卡，actor 与 rollout 分离。
4. Docker 门：两节点 local overlay2 可用、镜像数量和代表性 exec smoke 通过。
5. 算法门：互斥 feature flags、loss、reward post-process、dynamic filter 与 profile
   设计完全一致。

第一次启动：

    ./skillrl run rl.mixed-task-reward

确认 dry-run 输出后：

    ./skillrl run rl.mixed-task-reward --execute

run_rl.sh 会依次完成身份配置、数据验证、节点解析、local Docker preflight、
manifest 登记、guard 启动和 Relax delegate。driver 全过程写入本 segment 的
driver.log。

## 7.5 新实验、retry 与 resume

存储模型：

    experiments/rl/runs/<experiment_id>/
      experiment.json
      segments/<run_name>/
      model/
      eval/

首次启动可自动生成 EXPERIMENT_ID 和 initial RUN_NAME。恢复时必须：

- 明确复用已有 EXPERIMENT_ID；
- 使用新的 RUN_NAME；
- LOAD_DIR 指向上一个 segment 根；
- EXPECTED_LATEST_CKPT 与 latest_checkpointed_iteration.txt 相等；
- START_ROLLOUT_ID 等于 checkpoint + 1；
- 默认不允许跨实验 LOAD_DIR；
- 不允许新 segment 覆盖旧 segment。

示例：

    EXPERIMENT_ID=my-experiment \
    RUN_NAME=20260714-resume-39 \
    LOAD_DIR=experiments/rl/runs/my-experiment/segments/20260713-initial \
    EXPECTED_LATEST_CKPT=39 \
    START_ROLLOUT_ID=40 \
    ./skillrl run rl.mixed-task-reward --execute

基础设施重启不等于新科研实验；它只是同一个 experiment 的新 segment。算法、数据、
起点模型或核心假设变化时才新建 experiment。

## 7.6 运行中监督

至少同时看四组指标：

能力指标：

- raw task passrate；
- internal eval；
- bench/task 分层；
- aborted 与 infra failure。

行为指标：

- no read / oracle read / misleading read；
- 首次 read 的 turn；
- 每类行为的成功率；
- oracle 选择率、distractor 拒绝率；
- zero-tool-call、truncation。

训练信号：

- accepted/rejected groups；
- raw reward std；
- task advantage；
- behavior/selector advantage；
- selector token 数量与 loss；
- KL、entropy、grad norm；
- actor、rollout、verifier 和 Docker 等待时间。

固定轨迹审计：

- 固定 task/seed 在 step 0、关键 checkpoint、final 对照；
- 查看模型为什么不读、为什么读错；
- BC 类样本检查 mask 是否只覆盖目标 action span；
- 确认 read call、observation 和后续任务行为存在真实因果链。

不要只看 W&B 上一条总 passrate 曲线就判断算法成功。

# 8. Docker、Ray 与长期运行保全

## 8.1 Local Docker

标准 RL 使用每个 GPU 节点自己的 local overlay2：

    unix:///tmp/local-docker-overlay2.sock

数据目录默认：

    /data/cache/local-docker-overlay2-root
    /data/cache/local-docker-overlay2-exec

准备入口：

    bash ops/launch/prepare_local_rl_docker_runtime.sh
    bash ops/launch/start_local_overlay2_docker.sh

容器内禁止运行时临时下载重依赖；训练前恢复镜像、预热 TB2 uv cache、预构建缺失
镜像。ops/cache/pkg 中的打包依赖是为这一点保留的。

## 8.2 Claw stale container

Claw 任务频繁创建 sandbox，Docker retry storm、rollout 越来越慢或残留容器增长时，
先检查定向 cleaner：

    ops/cleanup/cleanup_claw_rl_stale_containers.py
    ops/cleanup/watch_rl_stale_containers.py

只清理能由 run ownership、标签和 age 证明为 stale 的容器。不要 docker rm -f
所有容器，也不要删除整库镜像缓存。

## 8.3 Ray 安全

正常实验恢复禁止：

    pkill -f ray
    ray stop

这类命令可能连 head、GCS、runtime env agent 和别的实验一起杀掉。先：

- ray list nodes；
- 识别 head 与两台 GPU worker；
- 确定是单个 actor、serve deployment、SGLang engine 还是整个 control plane；
- 用 exact PID、Ray actor id、tmux session 或维护脚本处理。

Pod 或 Ray 容器重建后必须重新发现 live worker IP，并重建 local RL/eval Docker；
旧 IP、旧 overlay2 状态和旧 SSH tunnel 都不可信。

## 8.4 网络与代理

Ray、NCCL、SGLang 与 Docker 内部地址必须进入 NO_PROXY。代理适合外部下载，不应
劫持 10/8、172.16/12、localhost 或 Ray node IP。

仓库脚本不内置任何真实代理地址，只有占位 default。若你的环境需要经代理访问
外网，请自行 export HTTP_PROXY/HTTPS_PROXY/ALL_PROXY 与
UNIFIED_CONTAINER_PROXY，并把集群内部地址加入 NO_PROXY。

Docker tunnel 要区分：

- 客户端 Shell 的 DOCKER_HOST；
- SSH tunnel 的本地 listener；
- 远端 /var/run/docker.sock；
- Ray worker 进程实际继承的环境。

只在交互 Shell docker ps 成功，不代表 Ray task 里也成功。

## 8.5 机器空闲与无人监督

长流程拆成可恢复阶段：等待训练、导出、起服务、评测、汇总。推荐用能读状态并修复
错误的 supervisor，而不是一个无法判断语义错误的长 bash。

对长期 GPU 占用：

- 明确当前 tmux/session/PID；
- 监控 GPU、driver.log、Ray actors、Docker 容器和磁盘；
- 非致命 trial failure 允许重试，不因单条失败停训练；
- correctness、数据串线、权重覆盖或 checkpoint 损坏属于立即停止条件；
- 训练接近 checkpoint 时不要为普通 infra 噪声轻易打断。

# 9. Evaluation

## 9.1 Owner-local 结果

评测一个模型时，结果写回拥有该模型的 experiment：

    experiments/rl/runs/<owner>/eval/<eval_id>/rows/<row_id>/

eval.json 描述 protocol 和行状态。跨模型对照仅从各 owner 的 row 提取，生成
一个派生视图目录（如 experiments/derived/）下的小型 Markdown 表；不再创建
experiments/rl_eval。

## 9.2 Canonical 命令

完整 HF 模型：

    bash ops/workflows/rl_eval/run_eval70_checkpoint_set.sh \
      --group my-comparison \
      --skill-mode mixed \
      --snapshot skill_libraries/snapshots/rl/slate_skills_20260708_hard_negative_v8_production \
      --manifest skill_libraries/snapshots/rl/slate_skills_20260708_hard_negative_v8_production/manifest/slate_manifest_eval70.jsonl \
      --model owner-experiment final path/to/complete_hf \
      --dry-run

Relax checkpoint：

    --checkpoint owner-experiment final path/to/checkpoint-root 99 final

先 dry-run。正式执行前核对：

- owner experiment.json 存在；
- model/export 完整；
- task list 是 eval70_v1；
- skill mode 与 snapshot 相符；
- manifest 与 snapshot 同源；
- tools schema 为预期 manual_schema；
- prompt profile 为预期 openclaw_full；
- context、repeats、seed 一致；
- no-skill 行不得残留 skill prompt；
- endpoint 指向正确模型，而非旧 SGLang session。

默认 4 repeats、64 workers、两台节点各两个 TP4 engine。并发是否真实达到 64 要从
active trials、endpoint utilization 和 Docker start 统计确认，不能只看 CLI 参数。

## 9.3 报告口径

必须报告 task-level pass@k。trial-level passrate 可附带，但不能代替 task-level。

比较前必须写明是否完全相同：

- task split；
- repeats/seed；
- prompt profile；
- tool schema；
- retrieval/snapshot；
- model endpoint；
- Docker mode；
- verifier version；
- rerun/completion policy。

mixed(0704)、mixed(v8prod)、hybrid snapshot 不是同一环境，表中必须标 snapshot，
不能把数字直接当成同一列的训练增益。

# 10. 故障定位顺序

“任务失败”先拆成五层：

1. 模型行为失败；
2. prompt/tool schema/adapter 不兼容；
3. verifier 确定性失败；
4. Docker 镜像或依赖失败；
5. Ray、网络、磁盘、GPU 基建失败。

常见现象与第一检查：

| 现象 | 第一检查 |
|---|---|
| RL 速度突然很快且 passrate 极低 | 是否把全 0 group、aborted 或 infra failure 混进训练 |
| 并发参数 64 但只有十几条 active | endpoint 数、Docker start cap、bench cap、串行队列 |
| rollout 越来越慢 | stale Claw container、Docker retry、磁盘与 verifier timeout |
| 模型完全不读 skill | prompt 可见性、read 成本、raw task 分布、advantage 是否真正落到 selector token |
| 给 oracle bonus 仍不读 | bonus 是否被组内归一化抵消、no-read 是否也得正分、行为信号是否覆盖整条失败轨迹 |
| resume 从错误 step 开始 | latest marker、EXPECTED_LATEST_CKPT、START_ROLLOUT_ID |
| 只有 config.json 的 export | 原子 export 未完成，禁止评测 |
| eval 表看似完成但数字异常 | endpoint 模型串线、旧 queue、snapshot 或 prompt 不同 |
| Docker exec 无 stderr/返回码异常 | local daemon/exec wrapper smoke |
| Ray 只见一台 GPU node | 先修 cluster，不要用宽泛 kill |

# 11. 修改与交接验收

Relax、runner、tool schema、prompt compatibility 或 data conversion 修改后至少：

    bash -n changed_script.sh
    python3 -m py_compile changed_module.py
    ./skillrl doctor
    ./skillrl verify

再做与风险成比例的 smoke：

- 数据 builder：validate-only、行数、gold/slate 断言；
- reward/advantage：CPU synthetic group；
- selector mask：token span 与 task token disjoint；
- launcher：dry-run resolved config；
- Docker：代表性 run/exec/timeout；
- eval：只生成 plan 或 1 个小 task；
- export：完整性检查后原子 rename。

一个有意义的训练或评测变更交接至少写清：

    Objective:
    Canonical entrypoint:
    Input data / split:
    Output root:
    Resume behavior:
    Changed files:
    Validation performed:
    Known risks / next checks:

本交接库的历史 owner 目录是轻量证据，不是可直接 resume 的完整 checkpoint。先读
（这两个文件随旁挂资产包提供，见 assets/README.md）：

    experiments/rl/HANDOVER_MANIFEST.json
    experiments/rl/catalog.json

恢复外部权重后再执行 recipe。任何新 run 从一开始就应使用 owner-local 结构，避免
重新出现 checkpoint、eval、日志分散在几十个平级长目录而无法对准的问题。
