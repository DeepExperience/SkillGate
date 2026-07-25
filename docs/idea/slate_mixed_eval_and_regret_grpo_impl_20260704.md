# SlateRL 落地实现说明（mixed skills 评测 + 配对 regret GRPO）（2026-07-04）

对应立项方案 `skill_reliability_proposal_20260703.md` 的工程落地。两条线：①给 train(491)+eval70(70) 每个任务配 4 类 skill 的混杂 slate 并把 8 个模型跑 mixed 评测（带按类别读取归因）；②训练侧实现"外部 skill（非内化）版"的配对 regret GRPO（只做 outcome-only reward + Δ shaping 两点），**默认全关、不影响任何在跑实验，先不开训**。

## 1. Slate 资产：每任务 16 条（oracle-1 / misleading-5 / relevant-5 / irrelevant-5）

产物统一在 `experiments/rl/v2/slate_skills_20260704/`（manifest/ + skills/ + snapshot_*/ + logs/）。

| 类别 | 来源 | 实现 |
|---|---|---|
| oracle ×1 | `20260612_qwen27b_full692_oracle_selfread/oracle_skills_snapshot`（692 任务，含 claw；训练侧 flat 目录不含 claw 所以统一用 snapshot） | **只复制不动原目录**：拷到 `skills/<新名>/` 并改写 frontmatter `name:`；新旧名双向映射在 `manifest/oracle_rename_map.json`。新名由 27B 起（内容描述式 kebab 名，禁止 verbatim 复用 task_id——旧目录名就是 task_id，这正是要修的泄漏） |
| relevant ×5 | 20260424/v7 retrieval（`retrieve_v7_aligned/*.jsonl`，覆盖 491+70 全部任务）的 `reranked_top10[:5]` | 用 `basename(skill_path)` 做 join key（`skill_name` 字段与目录名有 ~15% 不一致，不能用）；去重不足时从 coarse 榜补 |
| irrelevant ×5 | merged 库 2043 个有 SKILL.md 的 skill 中**排除该任务 coarse_top50 ∪ reranked_top10 后**哈希种子随机 5 个 | ⚠️ 用户要求"top100 之外"，但当年 retrieve 只算到 coarse_top50、无 top100 存档，且 Qwen3-Embedding-8B 权重不在本机（补算需走计费代理下 16GB）。故用现有资产支持的最深排除（top50∪rerank10），manifest 里每行标 `irrelevant_exclusion: top50_fallback`。已写好 `slate_compute_top100.py`（复用 index pkl + 存档 task_description 重嵌入查询），以后要严格化随时可跑 |
| misleading ×5 | 27B 对 oracle skill **定点篡改**（5 个策略各一条：错命令/flag、错路径、乱步骤+错验收、错参数值、目标偏移到隔壁问题） | `slate_gen_misleading.py`：naming 阶段一次调用出 oracle 新名+5 个同族异名（全局查重防 flat 目录互覆盖）；corrupt 阶段每变体一次调用，要求 3-6 处针对性篡改、表面自信专业、description 重写但**保持与 oracle 同等任务指向性**（对称性：oracle 的 description 本来就提任务场景，misleading 不提反而成了反向 tell）；QC：污染词检测、长度比 0.3-2.5、frontmatter 规范化、断点续跑 |

流水线（`slate_skill_pipeline.py`）：`build-base`（无 GPU，已跑完 70+491 行）→ 27B `naming`+`corrupt`（`run_slate_gen_20260704.sh` 一键：2 本地 TP4 + 2 远端 .233 TP4 + router，keepalive eval 结束后自动接卡）→ `finalize`（复制改名 oracle、校验 misleading、写终版 manifest + rename map）→ `make-snapshot`（每任务 16 条按任务种子洗序，写成 retrieval_snapshot 同款 per-bench jsonl）→ `check`。

## 2. eval70 mixed 模式（几乎零改动，复用注入链）

注入链本来就是 N-skill 通用的（`--inject-retrieval-skills <jsonl>` + `--retrieval-top-n`，`retrieval_skill_inject.py` 切 `skills[:top_n]`、docker cp 进 7 个 agent skill 目录、prompt hint 列 name+description）。所以 mixed 模式 = 造好 snapshot + `run_eval70_model.py` 5 处小改：`--skill-mode` 加 `mixed`（top_n=16、要求 `--retrieval-root`、arm 归 retrieval、报表标签 "mixed skills"）。

**按类别读取归因**：`collect_successes.py` 新增 `SKILL_NAME_CAPTURE`（捕获挂载路径下的 skill 目录名），`detect_skill_use()` 返回值加 `read_skill_names`（口径与 strict 布尔一致，含 tool 结果回显）和 `read_skill_names_agent`（只算 assistant 主动引用/打开）两个新字段——纯增量，旧字段不动（用 v4 final99_oracle 跑回归：47.5% / 96.8% 271/280 与存档完全一致）。`analyze_eval70_3tables.py` 把名字带进 trial 记录并支持 `EVAL70_DUMP_TRIALS_DIR` 落盘；新 `analyze_slate_reads.py` join manifest 出报表：各类别读取率（any/agent 两口径）、P(读到 oracle|读了)、"读了 misleading 没读 oracle"的 trial 数及其 pass 率、没读任何 skill 的 pass 率。

**8 模型队列** `run_eval70_mixed8_20260704.sh`：沿用现行标准（4×TP4 = 本机 2 + <gpu-node-ip> 远端 2（`ray_remote_sglang.py` 起）、router round_robin、`--workers 64 --concurrent-trials` + `DOCKER_START_CAP=128` + retry 2 轮补跑、70×4=280、ctx 65536、seed 1063810697、守护 4 件套）。⚠️ 用户说的"128 并发"按 docker-start-cap=128 + workers=64 落地——workers=128 是 MEMORY 里记过的掉分伪影（争用拖慢 2×），若确要 128 workers 改 `WORKERS=128` 重跑即可。行顺序按分析价值排：noskillRL9b → oracle1skillaware → hardspanv4bc60 → oracle1baseline → actionspan → base27b → sft9b → base9b（RL ckpt 读取率 74-96%，是判断力信号的主要来源；base9b/sft9b 读取率仅 ~10% 放最后）。每行结束自动出 zcc 表 + slate_reads 归因报表，汇总写 `z_cc_terminal_imgs/20260704_eval70_mixed8_4repeats_results.md`。

## 3. 训练侧：配对 regret GRPO（env-gated，默认关，未开训）

只实现方案 4.2 的两点：outcome-only reward（继承 `RELAX_SKILL_GROUP_REWARD=0`，不奖励读取行为）+ 配对 Δ shaping。

**双臂结构**（复用 pair-atomic 机制，slate 臂占原 oracle 的 deferred 槽位）：
- 数据：`make_4bench_slate_parquet.py` 把 pair parquet（no_skill_grpo + oracle_prompt_bc 各 491）的 oracle 行改成 `slate_grpo` 行——prompt 里 `<preloaded_oracle_skill>` 整块换成 `<available_skills>` 16 条列表（**self-read 形态**：只列 name+description+路径，skill 文件由 `retrieval_skills_top_n` + `AGENT_BENCH_EXTRA_SKILL_ROOTS`（slate skills 根）在容器建立时注入，模型要自己读——16 条全文内联会爆 prompt 预算，且"读不读、读哪个"本身就是要训的行为）。每任务按种子掷 `p_gold=0.7` 决定 gold 在不在场（缺席=15 条，练 negative rejection）；`slate_contains_gold/slate_gold_name/slate_size` 进 extra_info。no_skill 行逐字节不动，行序不动（ROLLOUT_SHUFFLE=0 配对依赖）。
- rollout（`sglang_rollout.py`，全部挂在 `RELAX_SLATE_REGRET_GRPO=1` 后面）：`slate_grpo` 进 deferred 槽、标 `relax_pair_role="slate"`；**no-skill 臂完成后无论 mixed/all-pass/all-fail 都提交 slate 臂**（不再是 all-fail 才 rescue——Δ 需要两臂都有），提交时把 no-skill 组均值 stamp 到 slate 组（`relax_pair_no_skill_mean_reward`）；slate 组完成后：mixed → GRPO 接收；uniform（全对/全错）→ |Δ|≥`RELAX_SLATE_UNIFORM_MIN_DELTA`(0.25) 才接收（全错 slate + no-skill 有过 = "被误导"，是最重要的罚样本，不能像普通零方差组一样丢）；否则丢弃。互斥守卫：与 `RELAX_PAIR_ORACLE_GRPO`/`RELAX_OPSD_MODE`/`RELAX_PAIR_ORACLE_BC_UNTIL_STEP` 同开即 raise。
- shaping（新模块 `slate_regret_gating.py`，wrap `hybrid_pair_gating.post_process_rewards`）：对 slate 组在**组内居中之后**整组加 `coef × clip(mean_slate − mean_noskill, −1, 1)`（居中前加会被均值减掉；参照 `subgroup_adv_coef` 先例）。gold 在场用 `RELAX_SLATE_REGRET_COEF`(0.5)，缺席用 `RELAX_SLATE_REGRET_COEF_NOGOLD`(默认同前者)——线性形式天然覆盖用户给的四种语义：在场 Δ<0 罚/Δ>0 奖；缺席 Δ<0 罚（被误导）/Δ≈0 无操作（正确忽略）。shift 与 Δ 写进 `train_metadata`+reward dict，train dump 可审计。
- 入口：独立 launcher `run_4bench_slate_regret_pair_from_sft.command.sh`（**pair brain 的独立拷贝**，一个字符都不改共享链路——v4/oracle-GRPO60 的 resume 都走那条链）；env 白名单 `Relax/relax/utils/utils.py` + `ray-job.sh` parity 各加 4 个 `RELAX_SLATE_*`。

**不影响在跑实验的保证**：新 env 全关时——`slate_grpo` 是 unknown kind（fail-soft 原语义）、role 映射/配对/completion 分支逐字节同前、W&B stats 预置键未动（新键全部 `.get()` 惰性）、post-process 原样 delegate、旧 launcher 仍指向 `hybrid_pair_gating`。已做合成单测：关=输出与原函数完全一致；开=shift 数学正确（+0.25 / −0.225 案例）、配对角色正确、oracle kind 不被劫持。另跑了 6 维对抗性 review（rollout 字节等价、shaping 正确性、eval 回归、资产流水线、队列/parquet、在跑实验安全）。

## 4. 待跑清单（按序）
1. keepalive eval 结束 → `bash ops/workflows/rl_eval/run_slate_gen_20260704.sh`（27B 生成 + finalize + snapshot，估 1-2h）
2. `bash ops/workflows/rl_eval/run_eval70_mixed8_20260704.sh`（8 行 × ~2-3h，可断点续跑）
3. 训练不开；何时开 slate regret probe 由用户决策（V0 mixed 评测结果先看 headroom）
