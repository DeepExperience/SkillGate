# RL Operations Guide

This file condenses the former RL Docker/prebuild/verifier/reward-analysis docs into one active reference. Full originals are archived under `archive/docs_cleanup_20260526/docs_originals/`.

## Scope

This guide covers operational facts for Relax-based agentic RL on the 5-bench train/eval pool:

- Docker image prebuild and missing-image handling.
- Verifier dependency and timeout triage.
- Reward plateau interpretation.
- Sample trajectory storage policy.
- Local 8-GPU single-node RL launch rules.
- **128 并发本地 Docker 守护 canonical 清单**（2026-06-11 定稿，见下半部分）。

Detailed chronological events stay in `docs/rl_log.md`.

## Canonical 16-GPU RL Launch and Ownership

Use only `ops/workflows/rl_training/run_rl.sh` for maintained 16-GPU training.
Its profiles are `no_skill`, `mixed_task_reward`, `mixed_separated`, and
`hybrid_slate`. Historical
single8 and experiment-specific wrappers are archived and are not maintained
entrypoints.

Every scientific experiment owns its retry/resume segments, exports, and evals:

```text
experiments/rl/runs/<experiment_id>/
  experiment.json
  segments/<segment_id>/{run.json,resolved_config.env,driver.log,iter_*,rollout_result/}
  model/{exports/,best_hf,final_hf,selected.json}
  eval/<eval_id>/{eval.json,rows/<row_id>/}
```

First launch and dry-run:

```bash
cd /path/to/skillRL
bash ops/workflows/rl_training/run_rl.sh mixed_separated --dry-run
bash ops/workflows/rl_training/run_rl.sh mixed_separated
```

The launcher generates a new experiment id and initial segment id. For a resume,
reuse the original `EXPERIMENT_ID`, choose a new `RUN_NAME`, and point `LOAD_DIR`
at an existing segment in that experiment:

```bash
EXPERIMENT_ID=<existing-experiment> \
RUN_NAME=$(date +%Y%m%d_%H%M%S)-resume-59 \
LOAD_DIR=experiments/rl/runs/<existing-experiment>/segments/<source-segment> \
EXPECTED_LATEST_CKPT=59 START_ROLLOUT_ID=60 \
bash ops/workflows/rl_training/run_rl.sh mixed_separated
```

The launcher refuses an implicit cross-experiment resume and refuses to overwrite
a non-empty segment. `experiments/rl/current/latest.txt` points to the most recent
segment. Data lives under `datasets/rl/`, frozen skill assets under
`skill_libraries/snapshots/rl/`, and operational Docker/preflight artifacts under
`experiments/infra/rl/`.

For startup acceptance, inspect the owner segment's `driver.log`,
`rollout_result/train/`, and checkpoint completion marker. Do not use broad
`pkill -f ray` or `ray stop` while another run is active. Record meaningful launch
or incident details in `docs/rl_log.md`.

## Docker Image Prebuild

Use `ops/launch/rl_prebuild_missing_images.py` to audit current RL train/eval parquet files and prebuild or pull missing images before training.

Typical command shape:

```bash
python3 ops/launch/rl_prebuild_missing_images.py \
  --parquet datasets/rl/parquet_4bench_base_20260523/train.parquet \
  --parquet datasets/rl/parquet_4bench_base_20260523/eval.parquet \
  --out-dir experiments/infra/rl/prebuild/current_missing_images \
  --swe-workers 1 \
  --seta-workers 4 \
  --sb-workers 2 \
  --build-timeout-sec 1200
```

Operational rules:

- SWE uses official images when available.
- SETA and SkillsBench local-build images must be prebuilt on the remote Docker daemon host.
- Deterministic Dockerfile/build-context failures should be removed from RL train/eval splits rather than retried during training.
- Timeout-only failures should be diagnosed with verifier/image preflight before exclusion.

## Known Docker/Verifier Failure Classes

- `missing_or_unbuilt_image`: image tag is absent and online build is disabled or times out.
- `deterministic_dockerfile_failure`: Dockerfile or build context is broken; exclude from train/eval until fixed.
- `verifier_dependency_install_timeout`: verifier repeatedly installs packages such as `pytest`, `python3-pip`, `uvx`, torch/CUDA wheels, Playwright browsers, or conda envs.
- `docker_daemon_contention`: verifier passes in isolation but times out under stale containers or high concurrent Docker starts.

The old one-off exclusion application JSON is archived at:

```text
archive/docs_cleanup_20260526/docs_originals/task_exclusion_apply_20260523_191455.json
```

## SETA Verifier Notes

SETA had many plain-assert verifier cases where the original test path installed `python3-pip` and `pytest` at runtime. The effective fix was a mini-runner path for plain asserts plus image-side `python3` availability where needed.

Historical classification:

- `85` SETA train/eval ids were fast after the mini-runner patch without extra package installs.
- `212` SETA ids likely needed `python3` in-image to avoid repeated verifier overhead.
- Known hard/bad SETA exclusions included `25`, `244`, `436`, and `729`; later build-context checks added more task-specific exclusions as recorded in `docs/rl_log.md`.

## Heavy Dependency Candidates

Tasks likely to need verifier-ready images or cache mounts:

- SkillsBench heavy tasks: `mhc-layer-impl`, `speaker-diarization-subtitles`, `latex-formula-extraction`, `manufacturing-fjsp-optimization`, `multilingual-video-dubbing`, `scheduling-email-assistant`, `simpo-code-reproduction`, `video-tutorial-indexer`, and Playwright-heavy web tasks.
- TB2 heavy tasks: `extract-moves-from-video`, `pytorch-model-recovery`, `pytorch-model-cli`, `sam-cell-seg`, `torch-pipeline-parallelism`, `torch-tensor-parallelism`, `hf-model-inference`, and Playwright-heavy JS extraction tasks.
- Seta heavy tasks: `1022`, `1132`, `586`, `874`.

The main speed issue for these is usually runtime dependency resolution/download, not model inference.

## Verifier Preflight

Verifier-only preflight should be used before excluding a task. It separates real task failure from runtime/package/Docker pressure.

Representative command shape (use the local daemon socket or the your-docker-host TCP tunnel; the old `ssh://your-docker-host` form is deprecated — per-call SSH sessions exhaust `MaxSessions` under concurrency):

```bash
PYTHONPATH=Relax:GeneralAgent/eval_scripts \
DOCKER_HOST=${DOCKER_HOST:-unix:///tmp/local-docker-overlay2.sock} \
python ops/monitor/rl_verifier_preflight.py \
  --timeout-cap 120 \
  --jobs 4 \
  --out experiments/infra/rl/preflight/verifier_preflight.jsonl \
  --task seta_synth/817:seta_probe \
  --task tb2/extract-elf:tb2_probe \
  --task sb_ns/exoplanet-detection-period:sb_probe
```

Historical verifier findings:

- `seta_synth/817` improved from timeout under concurrency to about `2-3s` after the mini-runner patch.
- `swe_lite/Project-MONAI__MONAI-1121` had verifier time around `7-24s` and was not the core timeout source.
- Some SkillsBench tasks that timed out during RL passed in clean preflight; stale container cleanup and daemon pressure mattered.

## Reward Plateau Interpretation

The 2026-05-24 reward plateau analysis found no clean monotonic reward improvement over the first 90+ rollout steps. The key interpretation was:

- Reward buckets were confounded by changing task mix after resume/filtering.
- GRPO signal was sparse: many prompt groups were all-zero or all-one.
- Update magnitude was small: KL/logprob-diff/clipfrac indicated conservative policy movement.
- Behavior changed before reward: trajectories became shorter and less loopish, but verifier success did not rise proportionally.
- Compared with Endless Terminals, this setup has noisier heterogeneous benchmark tasks, a larger OpenClaw-style prompt/tool scaffold, remote Docker, and mixed verifier systems.

Practical implication: do not diagnose plateau from scalar reward alone; inspect fixed-slice eval, mixed-group fraction, all-zero-group fraction, loopish fraction, length distribution, and per-bench pass@k.

## Sample Trajectory Policy

Active sample trajectories are stored under:

```text
experiments/rl/sample_trajectories/
```

Rules:

- Keep JSON or JSONL data only.
- Do not keep Markdown summaries beside samples; summarize notable behavior in `docs/rl_log.md` or a run report.
- Organize samples by experiment/run family rather than by global docs state.

---

# 128 并发本地 Docker 守护（canonical checklist）

> 适用对象：任何在 **local dockerd** 上跑 **≥96 并发容器 churn** 的工作负载（RL rollout 或大规模 eval）。
> 目标：照此执行——**不多做、不少做**——即可复现 resume89 段"128 并发满载 23h 零风暴"的稳定性。
> 参考实现：`ops/workflows/rl_training/lib/runtime.sh`（训练）和
> `ops/workflows/rl_eval/run_eval70_model.py`（评测）；新 run 通过 canonical
> entrypoint 继承这些守护，不复制历史 wrapper。
> 定稿依据：`docs/rl_log.md` 2026-06-10 20:40 canonical checklist + 21:00 磁盘炸弹三件套。本节取代旧版 "Local Docker Runtime for RL/Eval" 与 "RL Docker Lifecycle Guard" 两节。

## 0. 事故索引（每项措施对应一次真实事故，删项前先读这里）

| 事故 | rl_log 日期 | 机制 | 对应措施 |
|---|---|---|---|
| CPU/pip 风暴 | step-29 段 | 128 容器内 pip/编译抢满全部核 → trainer NCCL / SGLang 饿死 | A1 cpuset |
| containerd-shim 熔毁硬 wedge | 2026-06-08 | 高并发 teardown 超时 → shim 进程泄漏 → pod PID1（`ray start --block`）不回收孤儿 → 内核 cgroup/netlink/rtnl 锁被 D-state 占满 → load 21000、节点只能平台重启 | C2 subreaper；B2 只作静默期修复 |
| shim reaper 启动竞态 | 2026-07-10 | 容器启动时 shim 先于 `docker ps` 可见；128 并发下通用 reaper 把合法启动中的 shim 误判为 orphan 并杀掉，产生 `missing_trajectory` | B2 高并发运行期默认关闭 |
| netns/rtnl_lock 泄漏 | 2026-06-08~09 | 内核 5.15 bug：bridge 模式高频建删 netns → `unregister_netdevice: waiting for ... to become free` 卡死 rtnl_lock | A4 host-net（结构性消除）+ B1 dmesg watcher（烟枪监测） |
| 磁盘炸弹 | 2026-06-10 | bench 任务 `dd` 无界写 → 444GB 脏页 → `balance_dirty_pages` 节流 → D-state 堆积、险些写满 docker 盘（写满 = dockerd 死 + ckpt 写不出 = 致命） | A6 fsize ulimit + C3 json-log 封顶 + B3 disk reaper |
| stdout 洪水隐形写穿 | 2026-06-10（14:49 案嫌疑） | 容器 stdout → dockerd json-log，`docker ps -s` 看不见 | C3 `--log-opt` 封顶 |
| step21 wedge | 2026-05 | 设置 docker exec/teardown 并发等"优化" env 反而触发死锁 | E 显式 unset 有害门 |

## A. 容器启动 env（launcher 统一 export，6 项）

由 unified runner（`run_unified_harbor.py` / `run_unified_swe.py` / claw runner）读取并转成 `docker run` 参数。**Ray 路径注意**：新 env 必须同时进 `Relax/relax/utils/utils.py` 的 runtime_env forward 白名单 + `Relax/scripts/entrypoint/ray-job.sh` 的 env_vars JSON，否则 worker 上不生效（cpuset、netnsfix 两次都踩过这个坑）。

| # | export | 防什么 | 备注 |
|---|---|---|---|
| A1 | `UNIFIED_DOCKER_CPUSET=24-179` | CPU 风暴饿死 0-23 核上的 trainer NCCL + SGLang | 数值按本机调：原则 = 给宿主关键进程留前 24 核，容器只用其余。180 核机即 `24-179` |
| A2 | `UNIFIED_DOCKER_PIDS_LIMIT=1024` | 容器内 fork 炸弹 | |
| A3 | `UNIFIED_TOOL_TIMEOUT_CHILD_CLEANUP=0` | tool timeout 触发的 `pkill` 在高压容器内自激成风暴 | `=1` 已被实证有害，保持 0 |
| A4 | `UNIFIED_DOCKER_NETWORK_HOST=1` | 内核 5.15 netns/rtnl_lock 泄漏；顺带消除 bridge iptables 全局锁（高并发创建的串行点） | runner 内已配套 graceful `docker stop` → `rm`（teardown 风暴的另一半解法）。host-net 下容器与 pod 共享端口空间，见 H/I |
| A5 | `UNIFIED_CONTAINER_PROXY=<本节点可达代理>` | 容器出网；死代理 = pip/git 全挂 | RL 节点 = node squid `http://<proxy-host>:3128`；本 4 卡 eval 机 = clash `http://127.0.0.1:7890`（host-net 可直达）。启动前必须 `curl -x <proxy> https://github.com` 验活 |
| A6 | `UNIFIED_DOCKER_ULIMIT_FSIZE_GB=32` | 单文件磁盘炸弹 | 内核 SIGXFSZ 只杀写该文件的那条命令（exit 153，agent loop 不死）；32G > 任务申报上限 20G > 实测可写层 ≤6G，误伤是良性可观测失败，可在轨迹里 grep `exit 153` / `File size limit exceeded` |

## B. 守护 tmux（默认 3 个；shim reaper 仅静默期按需启用）

| # | session 前缀 | 命令要点 | 周期 | 防什么 / 健康判据 |
|---|---|---|---|---|
| B1 | `rl-dmesg-*` | `dmesg -w -T \| grep -iE 'unregister_netdevice\|waiting for .* to become free\|kobject_uevent' >> <run>/cleanup/dmesg_unregister_netdevice.log` | 流式 | netns 泄漏烟枪。**健康 = log 恒空**（`wc -l` 应为 0）；出现行数即说明 host-net 防线被绕过，立即排查 |
| B2 | `rl-shimreaper-*`（可选） | `REAP_INTERVAL_SEC=120 python ops/cleanup/reap_orphan_shims.py` | 120s | **≥96 并发启动/运行期禁止启用。** shim 会先于容器出现在 `docker ps`，通用 reaper 存在误杀合法启动的竞态。`run_eval70_model.py` 默认关闭；只在没有新容器启动的静默期、确认确有 orphan shim 时短时启用。常态防线是 C2 subreaper |
| B3 | `rl-diskreap-*` | `DISK_REAP_PATH=<docker盘> DISK_REAP_INTERVAL_SEC=5 DISK_REAP_WATERMARK_GB=800 python ops/cleanup/reap_disk_bombs.py` | 5s | 磁盘炸弹两段式：① 每周期杀无界 `dd`（urandom/zero，豁免带 `count=` 的有界写）；② free < 水位线时猎杀可写层 ≥`DISK_REAP_KILL_GB`(40) 的容器（`docker kill` 先行、`rm -f` 降级）。**启动后必须确认 log 首行 `watermark=<期望值>GB`**。水位线按盘容量调（3.5T 专用盘=800；本 eval 机 1T 共享盘=300，见 I） |
| B4 | `rl-stale-*` | `RELAX_RL_RUN_ID=<run> python ops/cleanup/watch_rl_stale_containers.py --run-id <run> --driver-log <run>/driver.log --loop --interval-sec 120 --max-remove 32 --max-running-remove 4 --remove-running-after-sec 3600 --keep-recent-steps 2` | 120s | run-scoped 残留容器慢清。只动 `relax.managed=true` 标签的容器；先清非 running、Docker 不健康时跳过周期；running 只清属于旧 rollout step 且超龄 1h 的。**活跃 run 期间禁止任何人工广撒网 `docker rm`** |

配套 env：`AGENT_BENCH_DOCKER_LIFECYCLE_DIR=<run>/docker_lifecycle`（teardown/start 失败审计 JSONL）、`RELAX_RL_RUN_ID=<run_name>`（容器归属标签）。

## C. dockerd 本体（4 项硬要求）

| # | 要求 | 怎么做 | 为什么 |
|---|---|---|---|
| C1 | data-root 在**本地 ext4** | `bash ops/launch/start_local_overlay2_docker.sh`（幂等；socket 健康则复用）。data-root 不能在 `/`、普通 `/tmp`（overlay 嵌套 mount 失败 `invalid argument`）、tidalfs/JuiceFS（太慢） | overlay2 比 vfs 快一个量级；vfs 仅作无 ext4 时的退路 |
| C2 | dockerd 必须跑在 **subreaper** 下 | `LOCAL_DOCKER_USE_SUBREAPER=1`（脚本内置，经 `ops/launch/subreaper_exec.py` 包裹）。验证：`pgrep -f subreaper_exec` | pod PID1 是 `ray start --block`，不回收孤儿；dockerd 裸跑 → shim 泄漏无人收尸 → 06-08 熔毁复发 |
| C3 | json-log 封顶 | `--log-opt max-size=64m --log-opt max-file=2`（脚本已内置，env 可调） | 容器 stdout 洪水写进 dockerd json-log，`docker ps -s` 不可见，是写穿盘的隐形路径 |
| C4 | 镜像**离线灌**，不在线 pull | 首次：`python ops/launch/migrate_apex_images_to_local.py`（从 your-docker-host 导 tar 到 tidalfs `experiments/infra/rl/local_docker_migration/image_tars/`，551 个/730G）。重建恢复：`RESTORE_LOCAL_IMAGES=1 RESTORE_LOCAL_IMAGE_WORKERS=4 bash ops/launch/prepare_local_rl_docker_runtime.sh`（offline-only，~25min，预期 ok≥550） | 在线 pull/build 走代理大概率超时 + Docker Hub 匿名限流；高并发跑到一半缺镜像 = 任务批量 setup 失败 |

eval 路径配套（prebuilt-only 模式，防运行期联网安装）：

```bash
export DOCKER_HOST=unix://$(cat /tmp/local-docker-active.sock)
export DOCKER_HOST_VALUE=${DOCKER_HOST}
export TB2_UV_CACHE_BIND_MOUNT=1
export TB2_UV_CACHE_REMOTE_DIR=/data/cache/tb2_uv_cache/tb2-uv   # tar: external_caches/tb2_uv_cache_tb2-uv.tar
export UNIFIED_HARBOR_REQUIRE_PREBUILT_LOCAL=1
export UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS=1
```

注：镜像 restore 清单自动含 compose sidecar `skillsbench-visual-stability-api:latest`（`sb_ns/fix-visual-stability` 依赖）。

## D. preflight fail-fast（启动脚本内断言，缺一不启）

| # | 断言 | 防什么 | 适用 |
|---|---|---|---|
| D1 | `eval "$(grep -E '^export WANDB_(API_KEY\|SILENT)=' ~/my_init.sh)"` 后 `WANDB_API_KEY` 非空，否则拒启 | key 只存在于 my_init.sh 登录环境；非登录 shell 启动 = 整段训练不记曲线（06-10 实战坑） | 训练 |
| D2 | `latest_checkpointed_iteration.txt == EXPECTED_LATEST_CKPT` | resume 错点 | 训练 |
| D3 | `pgrep -f subreaper_exec` 命中，否则拒启 | dockerd 裸跑（见 C2） | 训练+eval |
| D4 | `docker images -q \| wc -l` ≥ 500 | 镜像不全 + prebuilt-only = 批量 setup 失败 | 训练+eval（按所跑 bench 的清单数调阈值） |
| D5 | **一次性 exec 回传烟测**：起测试容器后 `docker exec <c> echo ok` 必须真打出 `ok`；`bash -lc 'echo a; echo b >&2; exit 3'` 三路（stdout/stderr/rc=3）全对 | 自建 docker 通路（TCP↔unix 代理/隧道）若不处理 TCP 半关语义，只吞一次性 exec 回传而持久 shell 正常 → agent 轨迹真实但 verifier/swe-patch 判分静默全空（2026-06-11 hard354 v1 新栈段 1713 trial 报废） | 训练+eval，**换 docker 通路后必做** |

## E. 显式 unset 有害门（step21 wedge 元凶，必须保持未设）

```bash
unset AGENT_BENCH_DOCKER_EXEC_CONCURRENCY AGENT_BENCH_DOCKER_TEARDOWN_CONCURRENCY \
      AGENT_BENCH_DOCKER_HOSTS UNIFIED_DOCKER_NPROC_LIMIT \
      UNIFIED_DOCKER_MEMORY_LIMIT UNIFIED_DOCKER_RM_TIMEOUT_SEC
```

另：**不要设** `UNIFIED_DOCKER_BUILD_JOBS`（resume24 教训）。这些都是看起来像优化、实测引发死锁/风暴的开关；launcher 里显式 unset 是为了防止从旧 shell 环境继承。

## F. 框架内置（代码层自动，无需配置，列出仅为知情）

- ABORTED 组丢弃重填（≤64 组）；
- dynamic filter 拒采上限（64 组 / 300 样本）；
- skill gate 250 样本强制接受；
- 任务 setup 3 次重试 / 600s；
- 样本 wallclock 850s / verifier 300s 超时。

## G. 节点/pod 重建 runbook（4 步，工具 `ops/launch/rollout_node_bringup.py`；全程可远程）

1. **B1 worker init**：tmux 跑 `ray_worker_init.sh`，等 `DONE_rc=0`（含 apt 装 dockerd——pod 刚建时 dockerd 二进制可能未装完，subreaper 报 `exec failed: FileNotFoundError` 等 1-2min 重试即可）。
2. **B2 dockerd**：`export PATH=/usr/sbin:$PATH; LOCAL_DOCKER_USE_SUBREAPER=1 LOCAL_DOCKER_DATA_ROOT=<ext4路径>/local-docker-overlay2-root LOCAL_DOCKER_EXEC_ROOT=<ext4路径>/local-docker-overlay2-exec bash ops/launch/start_local_overlay2_docker.sh`。验证 `docker info` Driver=overlay2 + `pgrep -f subreaper_exec`。
3. **B3 镜像重灌**：C4 的 restore 命令，551 tar ~25min（JuiceFS 读快），确认 ok≥550。
4. **B4 飞行记录仪**：tmux `flight-recorder`，30s 周期把 `load / memavailGB / dirtyGB / D-state数 / 容器数 / diskfreeGB` 追加写到 **tidalfs** 上的 log（写 tidalfs 是为了 pod 死后还能取证）。形如：

```bash
tmux new-session -d -s flight-recorder "while true; do
  printf '%s load=%s memavail=%sG dirty=%sG D=%s ctr=%s diskfree=%sG\n' \
    \"\$(date -Is)\" \"\$(cut -d' ' -f1 /proc/loadavg)\" \
    \"\$(awk '/MemAvailable/{printf \"%d\", \$2/1048576}' /proc/meminfo)\" \
    \"\$(awk '/^Dirty/{printf \"%d\", \$2/1048576}' /proc/meminfo)\" \
    \"\$(ps -eo stat= | grep -c '^D' || true)\" \
    \"\$(DOCKER_HOST=unix:///tmp/local-docker-overlay2.sock docker ps -q 2>/dev/null | wc -l)\" \
    \"\$(df -BG --output=avail <docker盘> | tail -1 | tr -dc 0-9)\" \
    >> /path/to/skillRL/experiments/infra/rl/diagnostics/flight_recorder_<node>.log; sleep 30; done"
```

pod 重建后 `/data/cache`、`/tmp/ray` 这类本地盘内容全丢（empty-dir/宿主路径），**唯一持久层是 tidalfs 的 tar 缓存**——它把重建成本钉在 ~35min。

## H. 已知残余风险（诚实清单，非 100% 覆盖）

- **多文件累积型写手**（每文件 <32G 的批量写）：A6/C3 管不到，靠 B3 水位线 reaper 兜底。
- **kubelet 驱逐红线**：RL 节点 `/tmp/ray` free ≈246GiB 触发驱逐；飞行记录仪 diskfree < 水位线+200G 即应人工关注。
- **平台维护/抢占**：不可控（resume89 两杀均判定平台侧）；靠 tar 缓存把重建成本控制在 ~35min。
- **监控时 load 高的鉴别法**：先分 running vs D-state。running 高 = CPU 风暴（查 cpuset）；D 高 running 低 = 控制面/IO 风暴（查 shim 泄漏、脏页、netns log）。

## I. 适配：4×H800 eval 机（qs-103584）96 并发差异表

> **前置硬条件（2026-06-11 实测）**：本地 dockerd 要求 pod 具备 CAP_SYS_ADMIN/CAP_NET_ADMIN（特权 pod，同 RL rollout 节点）。当前 eval pod **没有**（CapBnd=a80425fb；`mount` 被拒、`unshare -m` EPERM、无 `/dev/fuse`）——overlay2/vfs/rootless 全部不可行，**必须先走平台把 pod 重建为特权配置**。重建 = 整机重启（SGLang 进程/所有 tmux/本地盘内容全失），所以只能安排在 campaign 间隙；且重启后 SGLang 是新进程，跨进程对比禁忌照旧。rl_log 2026-06-03 那条"local docker smoke 成功"是 RL 节点上的记录，不适用本机。

pod 特权就绪后，本机跑大规模 eval（launch_trials / oracle pipeline）按 A–E 执行，差异仅以下几项：

| 项 | RL rollout 节点 | 本 eval 机 |
|---|---|---|
| 并发 | 128 | **96**（27B TP4 单 SGLang 实例吞吐先封顶，实测 128 容器也只有 ~65 有效并发打到模型；96 足够喂满且 churn 压力更小） |
| docker 盘 | `/tmp/ray` 专用 3.5T nvme | `/data/cache`（nvme2n1 ext4，与 `/data/temp` 同盘）——**仅 ~1T 可用且与同宿主其他 pod 共享**，无更优盘（nvme1n1/nvme3n1 只以单文件 bind 进 pod） |
| B3 水位线 | `DISK_REAP_WATERMARK_GB=800` | **`DISK_REAP_WATERMARK_GB=300`** + `DISK_REAP_PATH=/data/cache`；另设 free<150G 时停止接受新 trial（launch_trials 侧守护） |
| A5 代理 | squid `http://<proxy-host>:3128` | clash `http://127.0.0.1:7890`（host-net 直达本机） |
| A1 cpuset | 24-179（0-23 给 trainer NCCL + SGLang） | 同 `24-179`（0-23 给 SGLang tokenizer/server 线程 + 系统） |
| D1/D2 preflight | WANDB key + ckpt 断言 | 不适用；换成 **SGLang `/health` 200** + D3 subreaper + D4 镜像数 |
| host-net 端口 | Ray/NCCL/SGLang 与容器共享 netns | SGLang `:30000`、clash `:7890`、ssh 隧道 `:2376` 与容器共享 netns；claw 槽位端口体系已隔离，新服务避开这些端口 |
| 镜像持久性 | /tmp/ray 是 empty-dir，pod 重建即丢 | /data/cache 同样不保证存活；tar 缓存在 tidalfs，重灌 ~25min |

切换前置（一次性）：先 8 任务冒烟（claw+harbor 各若干）对齐 your-docker-host 路径的判分结果，再上 96 并发；**同一对比实验的所有臂必须用同一 docker 后端**（容器启动时延不同会影响 wallclock 850s 预算下的口径）。

## J. RL teardown / 重启标准流程（2026-06-20 事故教训）

> 背景：一次 M1 续跑，我用 `pkill` 杀 Ray Serve actor 做 teardown，留下 ghost Serve replica 并把 **10.7GB 小 head** 的 GCS 压崩（GCS 4.24GB + dashboard 平时已 9.82/10.7GB），导致 **5 次启动失败**（`Failed to connect to GCS` / `HTTP proxies not available after 60s` / worker 注册不上 raylet）。head **GPU=0、不参与计算**，问题纯在其 GCS 被 churn 压垮。详见 rl_log 2026-06-20 ~11:50。

**J1. 正确 teardown（杀一个 RL run）**
- ✅ Serve：全清用 `serve.shutdown()`；**只清单个 app 用 `serve.delete('<app>')`**（保 ServeController/ProxyActor 暖，避免下次冷启动撞 60s proxy 超时）。
- ✅ driver：杀 `relax.entrypoints.train` 指定 PID 或 `tmux kill-session`（launcher EXIT trap 自动收守护 tmux）。
- ✅ 再清：孤儿 PG（driver 死后 Ray 通常自动 REMOVE，复核 `placement_group_table()`）、残留容器（`docker rm -f`）、僵尸。
- ❌ **绝不** `pkill -f 'ray::ServeReplica/RolloutManager/TransferQueueController'` —— 绕过 Serve controller 留 ghost replica + churn GCS（本次事故元凶）。
- ❌ 绝不碰 raylet / gcs_server / runtime_env_agent / dashboard / PID1（老规则）。

**J2. head GCS 先天脆弱** —— head 仅 10.7GB（GCS+dashboard 已占 ~9.82G）。任何 churn（反复 kill/launch、密集 ray.init/serve probe、大量 actor 同时生灭）都可能把它推入颠簸。**原则：一次性干净操作，不反复试**。

**J3. 重启前强健康门（防 churn 后假启动）**
- driver-level `ray.init` 连通**不够**（GCS 短暂 lull 会假阳性，本次就因此误判 "GCS_STABLE" 后启动仍失败）。
- 必须：在每个 GPU worker（`.219`/`.233`）`@ray.remote(num_cpus=0.1, resources={'node:<ip>':0.01})` spawn 远程 task + 验 `cluster_resources()['GPU']==16`，**连续 ≥4 次全过**才启动。
- 若 churn 后 GCS 颠簸：**完全停手静默 ~10min** 让 GCS GC 死 run 状态、释放内存，再过强健康门。
- 冷 Serve 运行时在健康 head 上 `serve.start` ~15s 完成；若卡 `HTTP_PROXY_TIMEOUT=60s`（硬常量，不可 env 调）→ 是 head 没缓过来的信号，**别硬重试**，回静默。

**J4. 兜底** —— 若静默 + 强健康门后仍反复 `Failed to connect to GCS` / proxy 超时 → head GCS 真病态，从平台 UI **重启整个 RayCluster**（按老规则不手动拉 raylet/gcs），重启后两 worker 重跑 `ray_worker_init.sh` + 确认 551 镜像 / subreaper dockerd，再从 ckpt 续跑。

## K. Megatron RL resume source root rules（2026-06-27 no-skill single8 教训）

> 适用对象：从一个已保存 iteration 手动构造 resume source，再用 `START_ROLLOUT_ID=k+1` 续跑的本地或多机 Megatron RL run。

**K1. `LOAD_DIR` 必须指向 checkpoint root，不是 `iter_000000k/`**
- 正确 root 至少包含：
  - `latest_checkpointed_iteration.txt`，内容为 `k`
  - `iter_000000k/`
  - `transformer_config.pkl`
  - `rollout/global_dataset_state_dict_k.pt`
- 如果 `LOAD_DIR=.../iter_000000k`，Relax/Megatron 可能无法识别最新 iteration，并把 `start_rollout_id` 重置为 0。这会造成看似启动成功、实际从错误 rollout step 重跑。

**K2. rollout dataset state 必须放在 `${LOAD_DIR}/rollout/`**
- 数据流 resume 读取的是 `${LOAD_DIR}/rollout/global_dataset_state_dict_k.pt`。
- 只把 `global_dataset_state_dict_k.pt` 放进新的 save root 不够；那只能帮助后续 checkpoint 保存，不能帮助本次 load 恢复采样位置。

**K3. resume 前的最小检查**
```bash
LOAD_DIR=/path/to/resume_root
k=$(cat "$LOAD_DIR/latest_checkpointed_iteration.txt")
test -d "$LOAD_DIR/iter_$(printf '%07d' "$k")"
test -f "$LOAD_DIR/transformer_config.pkl"
test -f "$LOAD_DIR/rollout/global_dataset_state_dict_${k}.pt"
```

启动后必须在 driver log 里看到两类证据：
- actor 侧：`Actor initialized with starting step <k+1>` 或等价 step log。
- rollout/data 侧：从 `${LOAD_DIR}/rollout/global_dataset_state_dict_k.pt` load metadata，并打印 `Loaded streaming dataset state`。

**K4. 本地 single8 no-skill wrapper 的附带规则**
- `ops/workflows/rl_training/run_4bench_noskills_single8_from_sft.sh` 默认 `LOCAL_DOCKER_USE_SUBREAPER=1`。
- 启动 local dockerd 后会检查 `subreaper_exec.py`；若 dockerd 裸跑，直接 fail-fast。no-skill dynamic sampling 的 verifier/container churn 不比 action-span 更简单，不能用裸 dockerd 长跑。

## L. Eval-best checkpoint retention（2026-07-10）

- `--keep-best-actor-ckpt` 是 opt-in；目前 canonical SlateRL launcher 显式启用，其他 run 默认不受影响。
- 配置必须满足：eval 开启、`eval_interval % save_interval == 0`、`--skip-eval-before-train`、top-level keep cap 至少为 1，并提供不可空的 eval-contract fingerprint。
- 保存布局固定为：top-level `iter_N` 只保留最新完整恢复点；最佳完整 eval 对齐点保存在 `<save>/best_eval/iter_N`，其标准 tracker 是 `<save>/best_eval/latest_checkpointed_iteration.txt`。从 best 恢复时 `LOAD_DIR` 指向 `best_eval` root，不指向内部 `iter_N`。
- `best_eval` 同时保存 `transformer_config.pkl` 和该 step 的 `rollout/global_dataset_state_dict_N.pt`。从旧 `LOAD_DIR` 续训到新 save root 时，同 fingerprint/样本数 contract 的旧 best 会在 actor 启动时原子硬链导入；新旧 best 取分数更高者，平分取较晚 step。跨文件系统无法硬链时会 fail closed，不能静默丢弃旧 best。
- best 只接受满足 configured rows x eval samples、JSONL/marker 行数、SHA256、rollout id、dataset name 和 fingerprint 全部一致的结果。空、少样本、写盘失败或 pending eval 必须 fail closed；不要手工伪造 complete marker。
- checkpoint 必须有 `.relax_complete.json`，且 `.metadata` 中列出的 distcp shard 集合和文件大小全部吻合。异步保存只在 finalize 后写 completion。
- `<save>/.relax_save_owner.json` 在 actor 初始化时取得，禁止两个 driver 共写一个 save root。确认旧 driver 已死且同 run、同 host 时，才可显式设置 `RELAX_RECLAIM_STALE_SAVE_OWNER=1`；跨 host 恢复不能自动证明旧 PID 已死，需人工审计后换新 save root或处理 lease。
- best 的 eval JSONL 和 complete marker 会固化在 `best_eval/eval/`；后续 same-step eval 重跑不能覆盖已选 best 的证据。
