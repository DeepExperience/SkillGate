# SETA local install (data-only)

Installed: 2026-04-14
Goal: get the **1376-task pack** in place for downstream采集/转换/RL，**不**安装 SETA 自带的训练/评估栈
（AReaL / camel / rllm / terminal-bench 这 4 个 git submodule 留空）；不跑 `setup.sh`（它要 sudo + 改
`/etc/docker/daemon.json`）。

## 当前布局

```
datasets/
├── seta/                        # 主代码仓 (shallow clone, 28 MB)
│   ├── dataset/
│   │   ├── synth_data        -> ../../seta-env/Dataset          # TB 1.0 格式 (1376 tasks)
│   │   ├── synth_data_harbor -> ../../seta-env/Harbor-Dataset   # TB 2.0 / Harbor 格式 (1376 tasks)
│   │   ├── tbench-tasks      -> ../external/terminal-bench/tasks  # 留空 (子模块未拉)
│   │   └── tbench-tasks_convert/  # 仓库自带的 train/val.parquet 示例 (~9 MB)
│   ├── training/data_utils/   # convert_tasks_to_dataset.py 等转换脚本
│   ├── evaluation/            # tb run / eval 脚本（依赖 camel + terminal-bench，未装）
│   ├── external/{camel,rllm,terminal-bench,AReaL}/   # 4 个空 submodule
│   └── INSTALL_NOTES.md       # ← 本文件
└── seta-env/                    # 任务数据仓 (full shallow clone, 154 MB)
    ├── Dataset/0..1375/         # TB 1.0:  task.yaml + Dockerfile + docker-compose.yaml +
    │                            #          run-tests.sh + tests/ + solution.sh + weights.json
    └── Harbor-Dataset/0..1375/  # TB 2.0:  task.toml + environment/ + tests/ +
                                 #          instruction.md + solution/
```

## Sanity check 结果（已跑）

| 数据集 | task 数 | 完整文件数 | 备注 |
|---|---|---|---|
| `synth_data` (TB1) | 1376 | 1375 | `373/` 缺 `weights.json`，其它字段齐全 |
| `synth_data_harbor` (TB2) | 1376 | 1376 | 全部 task 五件套齐全 |

类别分布（Harbor）：software-engineering 45%、system-administration 40%、其余 devops/linux/debugging/security 等共 ~15%。
难度分布：hard 52.5%、medium 45.7%、easy 1.8%（baseline 时注意采样要分层）。

## 已走通的运行路径：Harbor + 远程 Docker 主机

远程 Docker 主机跑 dockerd（TCP :2375），
本机 SSH 隧道 `2375:127.0.0.1:2375` + `DOCKER_HOST=tcp://127.0.0.1:2375`，本地 `harbor` CLI 透明地用远程 daemon。

### 已知坑：SETA bulk 版 `tests/test.sh` 用 uv 拉 Python 3.13，会过远程主机代理超时

1114/1376 任务（共享同一份 `test.sh`）走 `curl astral.sh/uv → uv init → uv add pytest`，
中间 `uv` 要拉 ~30MB 的 cpython-3.13 tarball，clash 代理下大概率 `request or response body error`。
已写 `scripts/patch_test_sh.py` 把 bulk 版替换为 SkillsBench 风格的 `pip3 install --break-system-packages pytest`
（Ubuntu 24.04 自带 python3.12，免下载）。

```bash
# 已对 1114 个 bulk task 应用过；其余 262 个 custom test.sh 未动，需要时再单独 patch
python scripts/patch_test_sh.py            # 真改
python scripts/patch_test_sh.py --dry-run  # 只看数
```

### 跑通的 sanity 用例

```bash
export PATH="$HOME/.local/bin:$PATH"
export DOCKER_HOST=tcp://127.0.0.1:2375
RESULTS_ROOT=/path/to/skillRL/GeneralAgent/eval_scripts/seta_eval/results

# 单任务
harbor run -p dataset/synth_data_harbor/0 -a oracle \
  -o $RESULTS_ROOT --job-name seta_t0 --quiet
# → reward=1, ~90s

# Mini-batch (8 个分层任务并行)
TASKS=(488 122 798 324 297 190 1315 487)   # 3 sysadmin + 3 sw-eng + 1 linux-admin + 1 devops
for t in "${TASKS[@]}"; do
  harbor run -p dataset/synth_data_harbor/$t -a oracle \
    -o $RESULTS_ROOT/mini --job-name t$t --quiet 2>&1 | tail -3 &
done; wait
# → 8/8 reward=1，并行总耗时 ~3min
```

### 任务结果布局（每次 trial）

```
$RESULTS_ROOT/<job-name>/<trial-id>/
├── agent/oracle.txt          # oracle agent 不输出，其它 agent 这里有 stdout
├── verifier/reward.txt       # 1 / 0
├── verifier/test-stdout.txt  # pytest 输出
├── artifacts/                # --artifact 指定的容器路径下载
└── ../result.json            # 整 job 的 reward 分布与统计
```

## 下一步要做的事

1. **(已 DONE)** 通了 Harbor + oracle 8 任务 reward=1，pipeline 可用。
2. **跑真模型 baseline**（替换 `-a oracle` 为 `-a terminus-2 -m <model>` 或 `--agent-import-path` 自定义 agent）。
3. **验证 reward 一致性 / 排除率**（memory 标记的"已知风险 #1"）：
   - 抽样 ~50 task 跑 oracle（验证 reward 上限）+ 跑 nop（验证 reward 下限），比对 `pass@oracle` vs `pass@nop`。
   - oracle 不通 = task 自身坏（分子 / 分母失衡）；nop 通 = test 太弱。两类都该排除。
4. **patch 剩余 262 个 custom test.sh**：很多是装额外 pip 包的；按需逐个改写或加 sed 兼容层。
5. **训练栈按需补装**：
   ```bash
   cd datasets/seta
   git submodule update --init --depth 1 external/terminal-bench   # tasks-runner
   # camel / rllm / AReaL 是 RL 训练栈，本仓用 verl/slime，不必装
   ```

## 注意事项 / 已知坑

- `seta/dataset/tbench-tasks` 这个 symlink 指向空 submodule，**不要**当成数据用。
- `seta-env` 是 shallow clone（`--depth 1`），如要溯源旧版本任务需 `git fetch --unshallow`。
- 转 parquet 需要 `pip install pandas pyarrow pydantic pyyaml tqdm`，本机 base/coderl2 都没装；
  建议用 `slime` env (Python 3.12) 或新建轻量 env。
- SETA 训练框架硬编码了 4-tool prompt + Qwen2.5 parser（见 v2 经验教训），**只取任务文件**这条决策不要回退。
