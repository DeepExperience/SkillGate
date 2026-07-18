# Terminal-Bench 2.0 local install (data-only)

Installed: 2026-04-15
Source: `harbor dataset download terminal-bench@2.0` (registry.harborframework.com)

## 布局

```
datasets/
├── terminal-bench/              # upstream repo shallow clone 430 MB (TB 1.0 original-tasks + code, 备查)
└── terminal-bench-v2/           # THIS dir — 89 TB 2.0 tasks (Harbor format)
    ├── <task>/                  # 每个任务：task.toml + environment/ + instruction.md + tests/ + solution/
    ├── .patch_test_sh.py        # 修 verifier 容器的 uv→Python 3.13 下载超时
    ├── dataset.toml             # 89 task 的 dataset 清单（harbor run -p . 可用）
    ├── scripts/
    │   ├── prepull_images.sh    # 批量并行预拉（易失败，不推荐）
    │   └── prepull_retry.sh     # 串行 + 指数退避，推荐
    └── INSTALL_NOTES.md
```

## 推荐跑法（已验证）

前置：远程 Docker 主机侧 dockerd TCP :2375，SSH 隧道 `ssh -L 2375:127.0.0.1:2375 your-docker-host -N -f`，
本机 `DOCKER_HOST=tcp://127.0.0.1:2375`。

### Step 1: 预拉 89 个 prebuilt 镜像（一次性）

```bash
tmux new -s tb2-prepull 'bash scripts/prepull_retry.sh'
# 串行 + 5 次重试 + 指数退避（2/4/8/16s），大约 10-15 分钟拉完 ~10 GB
```

### Step 2: patch verifier 的 bulk test.sh（一次性）

```bash
python .patch_test_sh.py           # 30 个 bulk 任务转系统 pip，其它 59 个留 custom
```

### Step 3: 跑 harbor（无需任何 hack flag）

```bash
# 单任务
harbor run -p ./chess-best-move -a oracle -o /tmp/tb2_jobs --job-name foo --quiet

# 整 dataset (或 -i glob)
harbor run -p . -i 'chess-*' -a oracle -o /tmp/tb2_jobs --job-name bar --quiet
```

## Root cause 记录（我之前诊断错了，此处更正）

### 错：「Docker Hub 被代理拦了，必须 `--force-build`」
**实测推翻**：
- 远程 Docker 主机的代理 (:7890) 正常，dockerd systemd proxy 配置正常。
- **直接 `docker pull alexgshaw/<任意 task>:20251031`** 通过 SSH 隧道走 classic `docker pull`
  → 7-14s 成功。
- **`docker pull` 并发（xargs -P 6）** → auth.docker.io 约 70% EOF 失败。
  猜测是 clash 上游对 auth.docker.io 的 rate limit / 连接数限制。
- **Harbor 走 `docker compose up --wait`** → compose/buildkit 的 pull 路径看起来对 dockerd
  的 systemd HTTP_PROXY 继承不完全，首次拉 alexgshaw/* 也会 EOF。
  （root cause 没深挖到 compose 源码级别，但现象稳定可复现）

### 正：预拉一次（串行 + 重试）→ 所有后续运行都是 cache hit

镜像一旦在远程 dockerd 缓存里，Harbor 的 compose pull 发现本地有就不会再 EOF，
整个问题消失。不需要 `--force-build`，也不用改任何 Harbor 源码。

### 也别用 `--force-build`：又慢又脆
- Prebuilt image 直接用：容器启动 ~5-10s；
- `--force-build` 每次 trial 都要 apt/curl/pip 从 Dockerfile 重建，~60-120s；
- TB 2.0 Dockerfile 里有些 task `RUN wget https://go.dev/dl/...` 等直连外网步骤，
  `--force-build` 时会失败（~10% RuntimeError）。Prebuilt 镜像因为已经 baked 过，
  不受网络波动影响。

## Verifier test.sh 的 patch 仍然必要（独立于镜像路径）

82/89 个 task 的 `tests/test.sh` 是 TB 2.0 官方 boilerplate，用 `uvx -p 3.13 pytest ...`，
uv 运行时会拉 30 MB 的 cpython-3.13 tarball。verifier 是 Harbor 用另一个 sidecar 容器
挂载 `tests/` 目录起的，**跟任务用不用 prebuilt image 无关**，仍会过 clash 代理。

`.patch_test_sh.py` 把 sha256 `4770437ea96c` 这 30 个 bulk 变体换成 SkillsBench 风格的
系统 pip（`pip install --break-system-packages pytest==8.4.1 pytest-json-ctrf==0.3.5`）。
另 7 个本来就没用 uv；剩 52 个 custom 变体含任务专属 pip 依赖，没一键 patch（需逐个改）。

**工作稳定子集**：patched bulk (30) + no-uv native (7) = **37/89 task 不依赖 verifier
外网**；另外 52 custom 预期部分会因 uv 下载 Python 失败，baseline 时再评估。

## Sanity 已验证（2026-04-15）

| task | 设定 | 结果 |
|---|---|---|
| chess-best-move | `--force-build`（老路径） | reward=1, ~2min |
| build-cython-ext | `--force-build` | reward=1, ~2min |
| constraints-scheduling | 预拉后默认跑（**推荐路径**） | reward=1, ~42s |

## 任务分布

- **Category**: software-eng 26, sysadmin 9, sci-computing 8, security 8, data-science 8,
  debugging 5, file-ops 5, model-training 4, math 4, data-processing 4, ml 3, 其余 <3
- **Difficulty**: medium 55, hard 30, easy 4 (偏难，比 SETA synth_data_harbor 难一档)

## 下一步

1. 预拉完成后，跑一遍 full 89-task oracle baseline，找出 Dockerfile/test.sh 真失败的 task。
2. 对照 memory `harbor_bench_setup_checklist` 4 道防线检查，再上真模型 baseline。
3. terminus-2 + qwen3.5-27b 做 baseline 对照（TB 2.0 比 SETA 更难，pass 率应显著低于 SETA 的 0.633）。
