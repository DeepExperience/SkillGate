# Mixed-Skill Selector Action Credit GRPO 方案（2026-07-12）

> 状态：设计已冻结，尚未实现、尚未启动。本文描述最后一个正式 selector 训练实验；工程 smoke、mask 回放审计和单 batch 数值校验不计为独立实验。

## 0. 结论

最后一次实验采用：

**全正常 on-policy mixed-slate rollout + skill-read action 局部 selector advantage + 任务 GRPO。**

不使用 teacher-forced 特殊轨迹，不新增 `select_skill` 工具，不做两阶段训练，也不继续使用整轨迹 skill bonus、组级 regret shift、OPSD、executor BC 或 oracle-prompt privileged trajectory。

核心目标是把两个信用通道严格分离：

1. **selector credit** 只更新一次 skill-read action 中决定 skill 身份的 path tokens；
2. **task credit** 只更新其余任务执行 tokens，并排除整个 skill-read tool call；
3. skill 文件内容属于 tool response，继续保持 `loss_mask=0`。

这保证：读对 oracle 但任务失败时不惩罚 selector；读错 misleading 但任务侥幸成功时不奖励错误 selector。

## 1. 为什么需要 token-local credit

当前 SlateRL 家族已经证明，仅在 sample 或 group 级改变 reward/advantage 不足以训练 selector：

- 组级常数 shift 在组内居中后对“同组谁选对”没有区分度；
- task outcome 会把 selector 和 executor 混在一起：
  - oracle-read 后任务失败，会错误降低 oracle read 的概率；
  - misleading-read 后靠模型自身能力做成，会错误提高 misleading read 的概率；
- skill-read action 只占长轨迹很小一部分，整轨迹广播会把选择信号稀释到大量执行 tokens；
- 当前 hybrid v2 在能力上出现回升，但 selector 没有稳定改善：固定 mixed eval `22/56 -> 20/56 -> 25/56`，而训练后期仍有约 `16% oracle / 41% misleading / 39% no-read`。

因此最后一次实验不再尝试“怎样调 scalar bonus”，而是直接改变 **哪个 token 接收哪一种 advantage**。

## 2. 为什么全部使用正常 rollout

特殊 oracle 轨迹原本只有两个用途：

1. 保证每个 prompt group 都出现一个 oracle selector 正样本；
2. 保证 executor 至少看到一次 oracle 内容后的任务 continuation。

但推荐初始化 `oracle-GRPO60 final99` 已经具备：

- mixed task pass `129/280 = 46.1%`；
- strict any-skill read `264/280 = 94.3%`；
- oracle read `119/280 = 42.5%`；
- misleading read `159/280 = 56.8%`。

按 any-oracle read `42.5%` 粗估，`n=8` 时一组完全没有 oracle action 的概率为：

```text
(1 - 0.425)^8 ~= 1.2%
```

这是按“轨迹任意时刻读到 oracle”计算的乐观估计。启动前必须用冻结 slate 离线审计 first-read/clean-oracle 的真实比例；即使有效 oracle action 率只有 `30%`，一组无正例概率也只有约 `5.8%`。

因此正常 rollout 已有足够 action support。引入 forced action 反而会带来 off-policy token、assistant turn 模板、工具执行状态和额外 BC loss，增加最后一次实验的实现风险。

## 3. 实验设置

### 3.1 初始化

推荐从已经导出的 `oracle-GRPO60 final99` HF 模型重新初始化 actor 和 reference，而不是继续旧 optimizer state：

```text
experiments/rl_eval/
  20260706_eval70_oraclegrpo60_hardspanv4_final99_vs_eval79_noskill_mixed_4repeats/
  models/qwen35_9b_oraclegrpo60_hardspanv4_final99_hf
```

理由：它已经会稳定调用 skill，并具有当前最可靠的 mixed/task 能力。最后一次实验只需要把 read 分布从 misleading 移向 oracle，而不是同时从低 read 起点重新学习工具调用习惯。

### 3.2 Slate 与 rollout

- 使用当前 hybrid slate：`1 oracle + 5 misleading + 5 relevant + 5 irrelevant`；
- misleading 使用 v8 body，description 使用可分性更好的 0704 版本；
- train/eval 均为 all-gold，禁止 gold 缺席；
- oracle 位置必须平衡，slate 顺序必须随 task 的稳定种子随机化；
- `n_samples_per_prompt=8`，8 条全部由当前 policy 正常 rollout；
- 不强制第一步读取，不注入 oracle body，不修改 OpenClaw tool schema；
- task raw verifier score 保持唯一能力 reward。

### 3.3 推荐冻结的训练参数

```text
init/reference       = oracle-GRPO60 final99 HF
learning_rate        = 5e-7
n_samples_per_prompt = 8
rollout_batch_size   = 16 tasks
global_batch_size    = 128 trajectories
num_rollout          = 40
eval_interval        = 10
save_interval        = 5
selector_loss_coef   = 0.2
kl_loss_coef         = 3e-5
```

`selector_loss_coef=0.2` 的前提是 selector loss 按“每个 action 内 token 均值，再对 action 均值”归一化，且不按整条 response 长度重新放大。单 batch smoke 只验证实际 selector/task coefficient 与 gradient norm 没有数量级失衡；不据此开多个实验臂调参。

## 4. Selector action 的定义

### 4.1 哪些调用算 selector action

一个被审计的 skill-read action 是 assistant 主动发起、且显式打开 skill 文件的工具调用：

```text
/root/.claude/skills/<skill-name>/SKILL.md
/root/.claude/skills/<skill-name>/README.md
```

支持现有 strict 口径：

- `read(path=...)`；
- `exec` 中明确使用 `cat/sed/awk/grep/head/tail/...` 打开上述路径。

只解析 assistant inference text，不解析 tool response，避免回显误归因。与该 sample 的 `retrieval_skills_top_n` 相交的调用是有效 selector action；未 advertised 路径仍从 task mask 排除，并作为 utility 为 0 的 invalid/unadvertised action 参与诊断，不能靠 task outcome 获得正 credit。

### 4.2 每个 read action 的 utility

对 prompt group `g` 中的每个有效 read action `a`：

```text
u(a) = 1    该轨迹第一次读取该组的 oracle skill
u(a) = 0    misleading/relevant/irrelevant/unadvertised/repeated read
```

同一轨迹重复读取 oracle 时，只有第一次 oracle read 的 `u=1`；后续重复读取为 `0`，防止通过反复读取 oracle 累积正 credit。

在 group 内、以 action 为单位去中心：

```text
b_g      = mean_{a in reads(g)} u(a)
A_sel(a) = u(a) - b_g
```

- group 没有 read action：selector loss 为 0；
- group 有 read、但没有 oracle action：所有 `u=0`，selector loss 为 0；
- group 同时出现 oracle/non-oracle：oracle action 为正 advantage，错误/重复 action 为负 advantage；
- 不做 std normalization，避免小样本 action count 放大噪声。

该设计保持 group 内 selector advantage 的 action-count 加权和为 0，不会给通用 `read` 语法施加持续的全局正/负压力；真正有区分度的是 path 中的 skill identity。

## 5. 两套 token mask

### 5.1 Selector mask

`selector_action_loss_mask` 只覆盖 selector action 内的 path/command 参数中决定 skill 身份的 token span，优先覆盖：

```text
/root/.claude/skills/<skill-name>/SKILL.md
```

不覆盖：

- read 前后的 reasoning/prose；
- 通用 `<tool_call>` wrapper；
- 其他任务工具调用；
- tool response 和 skill body。

每个 selector action 还要有对应的 token-level `selector_action_advantage`，在其 mask 内填同一个 `A_sel(a)`，mask 外填 0。

### 5.2 Task mask

基础 `loss_mask` 已将 assistant token 标为 1、observation/tool-response token 标为 0。新增：

```text
task_loss_mask = base_loss_mask - union(all skill-read tool-call spans)
```

即 task GRPO 排除整个 skill-read call，而不只是 path。原因是 read wrapper/function/path 都属于 selector 决策，不应再接收 task outcome advantage。

要求以下不变量逐 sample 成立：

```text
selector_action_loss_mask & task_loss_mask == 0
selector_action_loss_mask <= base_loss_mask
task_loss_mask <= base_loss_mask
tool_response tokens 在两种 mask 中都为 0
```

### 5.3 为什么不在最终 response 上直接 regex 后处理

`sample.response` 是 assistant turn 与 observation/tool response 的拼接。只对最终文本做 regex 存在三个风险：

1. tool response 可能回显路径或 tool-call XML；
2. 多轮 chat-template token 插入使字符位置到全局 response token 的映射易偏移；
3. SGLang 返回的 `new_tokens` 与重新 encode 整段文本可能在边界 token 上不同。

推荐在 rollout 当场记录 token span：

1. 每次 `_run_inference_step` 前记录 `response_token_start=len(response_tokens)`；
2. 只在该轮 `response_text/new_tokens` 内解析 skill-read；
3. 用相同 tokenizer 将局部字符 span 对齐到该轮 `new_tokens`；
4. 加上 `response_token_start` 得到全局 response span；
5. 将 action category、skill name、tool-call span、identity span写入 sample metadata；
6. encode/new-token 对齐失败时 fail closed：该 sample selector mask 置零并记 mismatch，正式训练前要求 mismatch 为 0。

## 6. Loss

对 trajectory `i` 的 task outcome 先按现有 GRPO 得到 `A_task(i)`。定义：

```text
L_task = PPO clipped loss(A_task, task_loss_mask)

L_sel  = mean over selector actions a:
           mean over t in selector_action_loss_mask(a):
             PPO clipped loss(A_sel(a), token t)

L = L_task + lambda_sel * L_sel + beta_kl * L_KL
```

要求：

- task 和 selector 都使用正常 rollout 的真实 `rollout_log_probs`；
- selector term 是 on-policy PPO，不是 BC/CE/AWR；
- `L_sel` 先在每个 action 内取 token 均值，再对 action 取均值，避免长 skill name 或多次读取的轨迹支配 loss；
- KL 计算保持现有 reference 路径；
- task pass@k 始终由 `raw_score` 计算，不被 selector utility 污染。

四类关键轨迹的梯度语义：

| 轨迹 | selector 更新 | executor 更新 |
|---|---|---|
| oracle-read + success | 提高 oracle path | 奖励成功执行 |
| oracle-read + failure | 提高 oracle path | 惩罚失败执行 |
| misleading-read + success | 降低 misleading path | 奖励成功执行 |
| misleading-read + failure | 降低 misleading path | 惩罚失败执行 |

## 7. No-read 与多次读取

### 7.1 No-read 的固有限制

no-read 没有 selector action token，因此纯 action-local PPO 无法直接给“没有发生的动作”负 credit。最后一次实验接受这一限制，因为初始化模型的 strict no-read 只有约 `5.7%`。

不采用以下补丁：

- 把 no-read penalty 广播到整条轨迹；
- 随意 mask 第一段 reasoning 作为伪 selector；
- 新增 `NONE` token/tool；
- 为 no-read 另造 teacher-forced 样本。

这些做法都会改变研究问题或重新引入信用污染。

硬监控：fixed mixed eval 的 no-read rate 若升到 `>15%`，或连续两个 eval 上升且 oracle-read 不升，则判定发生 read-collapse，本 run 失败，不再追加补丁实验。

### 7.2 多次读取

- 第一次 oracle read 可得 `u=1`；
- non-oracle 和重复读取均为 `u=0`，在含 oracle action 的 group 中获得负 advantage；
- 所有 skill-read call 都从 task mask 排除，不能靠任务成功为 read-all 行为洗正；
- eval 必须同时报告 first-read oracle、oracle-only、misleading exposure 和平均 read count，不能只报“曾经读到 oracle”。

## 8. 为什么不使用 teacher-force

### 8.1 如果实现，具体流程是什么

forced oracle read 的插入点应在 `env.reset()` 成功之后、第一次正常 `_run_inference_step()` 之前：

1. 从 `extra_info.slate_gold_name` 构造规范 OpenClaw `read` tool-call；
2. 用同一 chat template 编码完整 assistant turn suffix，包括正确 turn terminator；
3. 将 forced tokens 追加到 `sample.tokens/response_tokens`；
4. forced tokens 没有真实 rollout logprob，必须从 task PPO 中排除并使用独立 CE/BC mask；
5. 调用 `_process_env_step(env, forced_text, ...)`，使 `env.step` 真正通过 `ToolLayer.dispatch` 读取 oracle 文件；
6. 将 tool observation 以 `loss_mask=0` 追加；
7. 再进入正常 SGLang loop，让模型从 oracle observation 后继续 on-policy 完成任务。

这不是 sampling parameter 中的一个开关，而是主动修改多轮 token state 和 environment state。

### 8.2 最终否决理由

- forced tokens 是 off-policy，不能使用正常 selector PPO；
- raw `tokenizer.encode(forced_text)` 不足以保证 assistant turn boundary 与 `<|im_end|>` 正确；
- 必须同时维护 synthetic logprob、BC mask、task mask、budget、turn index、trace 和 env dispatch；
- 它引入“7 个正常样本 + 1 个特殊样本”的额外分布差异；
- 当前强初始化已经提供足够 oracle action support，收益不足以抵消工程风险。

只有在离线 first-read 审计证明大部分 group 没有 oracle action 时，才重新考虑显式 selector token 或 forced action；这不属于本次冻结实验。

## 9. 代码落点

### 9.1 建议新增

- `Relax/examples/agent_bench/selector_action_credit.py`
  - strict action attribution；
  - group action utility/baseline；
  - metadata 与诊断汇总。
- `Relax/examples/agent_bench/selector_action_grpo_loss.py`
  - task/selector 两套 mask；
  - 两个 PPO term；
  - selector loss/clipfrac/KL/overlap 指标。
- `ops/workflows/rl_training/run_4bench_selector_action_credit_from_oraclegrpo60.command.sh`
  - 唯一正式入口；
  - 固定初始化、数据、互斥开关、eval 与 checkpoint 策略。
- `ops/workflows/rl_training/tools/smoke_selector_action_credit.py`
  - CPU attribution/mask/advantage 测试。

### 9.2 需要最小修改

- `Relax/examples/agent_bench/rollout.py`
  - 记录每轮 assistant token offset 与 skill-read action span。
- `Relax/relax/utils/utils.py`
  - 生成 `task_loss_masks`、`selector_action_loss_masks`、`selector_action_advantages`。
- `Relax/relax/backends/megatron/actor.py`
- `Relax/relax/backends/megatron/data.py`
  - 传输新的 jagged token fields。
- `Relax/relax/utils/training/train_dump_utils.py`
  - 持久化 mask/action 统计，支持轨迹审计。

现有 `shadow_action_loss_masks` 只能用于 oracle BC 且会选择所有 tool call，不应通过改名复用其语义。可以复用 CP slicing、jagged mask transport 和 mask-length 校验辅助函数。

## 10. 必须记录的指标

### 10.1 能力指标

- fixed hybrid mixed task pass@1；
- per-bench pass；
- final/best checkpoint 的 eval70 x4 task-level pass@4；
- `P(success | oracle-only)`、`P(success | misleading-only)`、`P(success | no-read)`。

### 10.2 行为指标

- first-read oracle rate；
- oracle-only rate；
- misleading exposure / misleading-only rate；
- no-read rate；
- multi-read rate 与平均 read action 数；
- `P(read oracle | read any advertised skill)`；
- **joint success-and-oracle-only rate**（本实验的主行为-能力联合指标）。

### 10.3 训练信号指标

- selector-active group fraction；
- no-read group / no-oracle-action group fraction；
- oracle/non-oracle action count；
- selector advantage mean/min/max 与 weighted-zero-mean error；
- selector mask token count、zero fraction、alignment mismatch；
- task/selector mask overlap（必须恒为 0）；
- `selector_pg_loss`、`selector_clipfrac`、task PG loss、KL、grad norm；
- selector/task coefficient norm proxy，防 selector loss 压过能力训练。

### 10.4 固定轨迹审计

每个 eval checkpoint 固定保留同一批任务的完整 JSON/JSONL，至少覆盖：

- oracle -> oracle；
- misleading -> oracle；
- no-read -> oracle；
- oracle -> misleading/no-read；
- task fail -> pass 与 pass -> fail。

样本放在 `experiments/rl/sample_trajectories/selector_action_credit/`，不在旁边创建 Markdown；结论追加到 `docs/rl_log.md`。

## 11. 验证与停止条件

### 11.1 启动前工程 gate

必须全部通过：

1. 对历史 mixed 轨迹回放，strict action category 与现有 analyzer 一致；
2. assistant/action token alignment mismatch `=0`；
3. task/selector mask overlap `=0`；
4. observation/tool-response 两种 mask 均为 0；
5. selector advantage group weighted mean误差 `<1e-6`；
6. disabled path 与原 task GRPO loss 数值等价；
7. CP2 slicing 前后 mask token count 一致；
8. 单 batch update 后 oracle action logprob 上升、non-oracle action logprob 下降；
9. internal eval 确认为同一 hybrid mixed split，不是 no-skill selector-blind eval；
10. checkpoint 策略确认为 last + joint-best。

### 11.2 Eval gate

推荐 eval at `0/9/19/39`：

- eval0：冻结同协议初始化基线；
- eval9：first/oracle-only 应较 eval0 有可见上升，no-read 必须 `<15%`；
- eval19：oracle-only 目标至少较 eval0 `+15pp`，task pass 不得低于 eval0 超过 `2/56`；
- eval39：作为 final，选择 last 与 joint-best 做外部 x4。

以下任一条件触发提前失败，不再衍生新实验：

- no-read `>15%`；
- selector-active group 长期过低，说明初始化 support 假设不成立；
- oracle action share 不升而 misleading/no-read 上升；
- task pass 连续两个 eval 低于 eval0 超过 `2/56`；
- mask mismatch、overlap、tool-response contamination 非零；
- selector loss 主导总 grad 并伴随 task 能力下降。

### 11.3 Checkpoint 选择

保存：

1. 最后一个 checkpoint；
2. joint-best checkpoint。

joint-best 使用约束式选择，而不是任意加权和：

1. 先要求 fixed mixed task pass 不低于 eval0 超过 `2/56`；
2. 在满足能力约束的 checkpoint 中最大化 `success AND oracle-only`；
3. 若并列，依次比较 task pass、oracle-only、较低 misleading exposure。

## 12. 成功与失败如何解释

### 成功

若 oracle-only/first-oracle 明显提高，同时 task pass 保持或提高，说明过去失败的关键确实是 token-level credit assignment，而不是 description 完全不可分或 9B selector 能力不足。

### Selector 提高但 task 下降

说明 selector 可学，但 oracle skill 内容、任务分布或 executor 对 skill 的利用能力与任务目标不一致。不能把它写成完整成功，只能写成 routing 成功、utility 失败。

### Task 提高但 selector 不变

仍属于当前 hybrid v2 同类结果：能力/鲁棒性提高，不代表选择性调用学会。

### 两者都不提高

在 action support、mask 和梯度信号均经审计正确的前提下，应接受强负结论：当前 description 特征或 9B 模型不足以在真实长程 mixed-skill 环境中学会该 selector；不再继续调 bonus、mask 系数或训练日程。

## 13. 关联材料

- `docs/idea/skill_reliability_proposal_20260703.md`
- `docs/idea/slate_mixed_eval_and_regret_grpo_impl_20260704.md`
- `docs/final_dance/03_实验历程与结论.md`
- `z_cc_terminal_imgs/20260706_eval70_oraclegrpo60_hardspanv4_final99_vs_eval79_noskill_mixed_4repeats_results.md`
- `Relax/examples/agent_bench/skill_group_reward.py`
- `Relax/examples/agent_bench/slate_regret_stratified_gating.py`
- `Relax/examples/agent_bench/action_span_mask.py`
- `Relax/examples/agent_bench/hybrid_shadow_grpo_loss.py`
- `Relax/examples/agent_bench/rollout.py`
- `Relax/examples/agent_bench/env_agent_bench.py`

## 14. 最小交接

```text
Objective: 在正常 mixed-slate on-policy rollout 中，仅对 skill identity action tokens 施加 oracle-vs-distractor selector advantage，并将 task outcome credit 限定到非 skill-read 执行 tokens。
Canonical entrypoint: ops/workflows/rl_training/run_4bench_selector_action_credit_from_oraclegrpo60.command.sh（待实现）
Input data / split: hybrid v8-body + 0704-description all-gold slate；fixed hybrid mixed eval
Output root: experiments/rl/v2/checkpoints_from_sft_dynamic16_gbs128/<run_name>
Resume behavior: 标准 Relax checkpoint resume；保留 last + joint-best
Changed files: 尚未实现
Validation performed: 本文仅设计；实现后必须通过第 11.1 节全部 gate
Known risks / next checks: no-read 无显式 action credit；启动前先审计 first-read support 和在线 token alignment
```
