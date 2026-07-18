# SWE-Gym + SWE-bench Verified local install

Installed: 2026-04-15 (HF 数据 + 小子集 docker image 预拉)
本仓位放：HF 数据 + SFT 轨迹；配套脚本在 `GeneralAgent/eval_scripts/swe_gym_eval/scripts/`。不装 OpenHands fork。

## 布局

```
datasets/
├── swe-gym/
│   ├── lite/                                    # SWE-Gym-Lite：230 训练 instances（SWE-bench 同构 schema）
│   │   ├── data/train-00000-of-00001.parquet    # 910 KB
│   │   └── README.md
│   ├── sft-trajectories-openhands/              # 491 条成功 rollout（messages 格式）
│   │   ├── data/train.success.oss-00000-of-00001.parquet  # 10 MB
│   │   └── README.md
│   └── INSTALL_NOTES.md                         # 本文件
└── swe-bench-verified/                          # eval only（绝不用于训练）
    └── data/
        ├── data/test-00000-of-00001.parquet     # 2.0 MB, 500 instances
        └── README.md
```

## 数据 schema

### SWE-Gym-Lite（`230 × 11 cols`）

`instance_id · hints_text · patch · test_patch · created_at · problem_statement · repo · base_commit · version · PASS_TO_PASS · FAIL_TO_PASS`

Repo 分布（11 个）：
```
getmoto/moto                    59
python/mypy                     40
iterative/dvc                   36
Project-MONAI/MONAI             27
pydantic/pydantic               20
dask/dask                       14
conan-io/conan                  12
facebookresearch/hydra          11
pandas-dev/pandas                5
modin-project/modin              5
bokeh/bokeh                      1
```

### SWE-bench Verified（`500 × 13 cols`，**eval only**）

比 SWE-Gym 多了 `environment_setup_commit · difficulty` 两列。
Repo 分布（12 个，**与 SWE-Gym 完全不重叠** — 天然无 train/test leakage）：
```
django/django                  231
sympy/sympy                     75
sphinx-doc/sphinx               44
matplotlib/matplotlib           34
scikit-learn/scikit-learn       32
astropy/astropy                 22
pydata/xarray                   22
pytest-dev/pytest               19
pylint-dev/pylint               10
psf/requests                     8
mwaskom/seaborn                  2
pallets/flask                    1
```

### SFT Trajectories（`491 × 1 col`）

单列 `messages = [{role, content}, ...]`。直接喂 slime / torchtune / axolotl 都行。

## 关键：这不是 Harbor 格式

SWE-Gym 和 Harbor/SETA/TB 2.0 **执行模型根本不同**：

| 维度 | Harbor 格式（SETA/TB2/Skills） | SWE-Gym |
|---|---|---|
| 任务定义 | `task.toml + tests/test.sh + solution/` | HF parquet 行（无 task dir） |
| Reward | agent 在容器里做事，`test.sh` 退出码 = reward | agent 编辑 repo，harness 从 `git diff` 抽 `model_patch`，应用到 base_commit，跑 `FAIL_TO_PASS`/`PASS_TO_PASS` pytest |
| Runner | `harbor run` | **OpenHands fork** (`SWE-Gym/OpenHands`) 或 **Moatless fork** |
| Docker image | 任务内 `environment/Dockerfile` 或 prebuilt `alexgshaw/*` | **Princeton SWE-bench 预构建** `xingyaoww/sweb.eval.x86_64.<instance>:latest` |
| 镜像数量 | 89 (TB2) | **~2400 per instance** |
| 每镜像大小 | ~500 MB | **2-5 GB**（repo + conda env + 依赖都烤进去） |

## 镜像命名约定（已验证）

来源：Docker Hub `xingyaoww` namespace 共 4798 个镜像。
- Instance image: `xingyaoww/sweb.eval.x86_64.<repo_sanitized>_s_<name-version>:latest`
  - `/` → `_`
  - `__` → `_s_`（**关键！**parquet 里 `getmoto__moto-5752` → image `getmoto_s_moto-5752`）
- Env-level image: `xingyaoww/sweb.env.x86_64.<hash>`（更小，按 Python-repo-version 共用）

实例：
- `getmoto__moto-5752` → `xingyaoww/sweb.eval.x86_64.getmoto_s_moto-5752:latest`
- `astropy__astropy-12907` → `xingyaoww/sweb.eval.x86_64.astropy_s_astropy-12907:latest`

## ⚠️ Docker Hub rate limit（踩过的坑）

**无账号 pull 限制：100 次 / 6 小时 / 公网 IP**。  
TB 2.0 prepull（89 个）叠加 SWE 的几次探测触发了：
```
Error response from daemon: toomanyrequests: You have reached your unauthenticated pull rate limit
```

**当前登录状态（2026-04-15）**：远程 Docker 主机上 `~/.docker/config.json` 已有 Docker Hub OAuth token（GitHub 登录，走远程主机代理）。登录把 rate limit 从匿名 100/6h 提到 200/6h。

**以后大批量预拉前仍然需要**：
1. Lite 230 + Verified 500 = 730 个镜像，即使登录后也至少要分 4 个 6h 窗口
2. 或配 registry mirror（`dockerproxy.net` / `daocloud.io/mirror`）绕开 Docker Hub 配额
3. **直接只拉会真正跑到的子集**（当前策略）

## 已预拉的小子集（2026-04-15）

位置：远程 Docker 主机 `/var/lib/docker`（占用约 50-60 GB）。拉取脚本：
`GeneralAgent/eval_scripts/swe_gym_eval/scripts/{pick_subset.py,prepull_subset.sh}`

**SWE-Gym-Lite 10 个**（按 repo 多样性抽）：
- getmoto/moto × 2, python/mypy × 2, iterative/dvc × 2
- Project-MONAI/MONAI × 1, pydantic/pydantic × 1, dask/dask × 1, facebookresearch/hydra × 1

**SWE-bench Verified 10 个**（按小 repo 抽，省磁盘）：
- pallets/flask × 1, mwaskom/seaborn × 2, psf/requests × 2
- pylint-dev/pylint × 2, pytest-dev/pytest × 2, pydata/xarray × 1

具体 instance_id 清单在 `/tmp/pull_lite.txt` 和 `/tmp/pull_verified.txt`，都能从 parquet 里
重新生成（`pick_subset.py`）。

## 下一步（什么时候做什么）

### Phase 1（现在就能做，无需 docker）
- **SFT 起步**：用 491 条 `OpenHands-SFT-Trajectories` 在 slime 里做 SFT 热身（messages 格式直接喂）。
- **metadata inspection**：`pandas.read_parquet` 看 problem_statement / patch 长度分布，挑子集。

### Phase 2（要 RL 才做）
- 远程 Docker 主机 `docker login` 解除 rate limit
- 选 Lite 中的 20-30 个 instance 预拉镜像（~60-150 GB）做 rollout sanity
- Clone `SWE-Gym/OpenHands` fork 到 `datasets/swe-gym/OpenHands/`
- 跑 `scripts/rollout-swe-train-lite.sh` 验证 pipeline

### Phase 3（RL 闭环）
- **不要**在 slime 里从零接 SWE-Gym。参考 **SkyRL-v0** 的 OpenHands rollout backend（已接 SWE-Bench online RL，SWE-Gym 同 schema）
- 预拉到 Lite 全量（230 × 3 GB ≈ 700 GB）
- rollout 预算：单 task 5-60 min，N-trial × 230 ≥ 数千 GPU·h

### Phase 4（Verified eval）
- **只用 `SWE-bench_Verified` 做最终评测**。predict `model_patch` → 官方 `swebench` CLI 跑验证 → resolved 率。500 条里 django 就占 46%，主导指标。

## 路障提醒（记忆里也有）

1. **Trajectory 格式不通**：OpenHands event-stream ≠ Harbor `trajectory.json` 回合制，统一数据 pipeline 需要适配器，且可能丢 tool-use 语义。
2. **slime 无原生 adapter**：RL 大概率 fork SkyRL 而不是 slime。
3. **Docker Hub rate limit**：见上。
4. **磁盘**：Lite ~700 GB，Verified ~1.5 TB（每镜像更大，因为都是 Django/sympy 这种大 repo），Full 5-10 TB。远程 Docker 主机空间要算。
5. **Context length**：repo-level 任务常 >128K token，40K ctx 的 MaaS 模型几乎无法完成。
6. **固定 agent scaffold**：SWE-Gym 数值再现依赖 OpenHands fork（非 All-Hands main）。
