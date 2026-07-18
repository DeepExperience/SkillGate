# Prebake image cache · SWE-lite 100 + SETA synth 300

2026-04-19 · 扩数据准备：为 RL/SFT 训练提前 build / pull 好 docker image，下批跑时 100% 命中本地 cache，零 clash 流量。

## 目标

- **SWE-Gym-lite 100 instance** — stratified-by-repo 从 230 里选 100（保持 11 repo 多样性）
- **SETA synth_data 300 task** — 按"个人助手 / 文件操作"语义打分，从 1376 里挑 top 300

## 流量预估（单次，之后全 cache）

| 任务 | 方式 | image 数 | 预估 clash 流量 | 预估耗时 |
|---|---|---|---|---|
| SWE-Gym-lite 100 | `docker pull xingyaoww/sweb.eval.*` | 98 new（本地 2 cached） | **~245 GB** via registry-mirror → clash | 40-60 min (concurrency=4) |
| SETA synth 300 | `docker build` 每 task | 300 new | **~20-30 GB** via clash（多数走清华 pip / apt）+ 60-80 GB via tsinghua | 3-6 h (concurrency=2) |
| **合计** | | 398 images | **~280 GB clash** | ~5-8 h |

clash quota 若不够，可分步：先 `--swe-only`，等 quota reset 后 `--seta-only`。

## 使用顺序

```bash
cd /path/to/skillRL/GeneralAgent/eval_scripts/prebake_images/

# Step 1: 生成 task id list（可复现，同输入→同输出）
python3 select_seta_300.py            # → seta_300.txt (300 task ids)
python3 select_swe_lite_100.py        # → swe_lite_100.txt + swe_lite_100_images.txt

# 可选：preview 看挑选质量
python3 select_seta_300.py --preview  # top 10 + boundary (排名 298-302)
python3 select_swe_lite_100.py --preview  # per-repo split

# Step 2: dry-run 看总计划
bash prebake_all.sh --dry-run

# Step 3: 真跑（先 SWE 再 SETA，一条挂了不触发下一条）
bash prebake_all.sh

# 或分开跑
bash prebake_all.sh --swe-only
bash prebake_all.sh --seta-only
```

## 各组件说明

### `select_seta_300.py` — SETA 挑选

**算法（LLM-free 关键词匹配）**：
- **正向**（每次匹配 +1.5~2）：`file/directory/folder/archive/backup/compress/extract/rename/organize`、`search/sort/filter/count/inventory/convert`、`csv/json/yaml/markdown/html/pdf`、`document/calendar/email/notes/metadata`
- **负向**（每次匹配 -3）：`kernel/systemd/grub/DNS/DHCP/iptables/firewall/apache/nginx/mysql/postgres/redis/kubernet/LDAP/Docker container/SELinux/IRQ/PCI`

得分 = 2×file + 2×assist + 1.5×data + 2×doc - 3×neg

Top 300 成绩：
- score range `[9.5, 30.5]`，median `12.0`
- difficulty: hard 159 / medium 141（平衡）
- category: 200 software-engineering + 62 system-administration

Top 10 样例：
- 805: "Build a File Discovery and Audit Tool for a Large Project Directory"
- 560: "Build a PDF Metadata Normalization Pipeline"
- 1316: "Build a batch document conversion pipeline (ODT → PDF)"
- 836: "Organize a chaotic media archive directory"

### `select_swe_lite_100.py` — SWE-Gym-lite 挑选

**算法**：repo 分层比例（100 × repo_count / 230），每 repo 按 `instance_id` 字典序取前 k 个。**确定性、无随机种子**。

Per-repo 分布：
```
 20 / 59  getmoto/moto       |  7 / 14  dask/dask
 18 / 40  python/mypy        |  6 / 12  conan-io/conan
 16 / 36  iterative/dvc      |  5 / 11  facebookresearch/hydra
 12 / 27  Project-MONAI      |  3 /  5  pandas-dev/pandas
  9 / 20  pydantic/pydantic  |  3 /  5  modin-project/modin
                             |  1 /  1  bokeh/bokeh
```

### `prebake_swe_lite_100.sh` — docker pull 98 个

- `CONCURRENCY=4`（默认）：4 个并行 pull；超过 4 容易触发 Docker Hub rate limit
- 每 pull 重试 2 次，失败 task id 写入 `/tmp/prebake_swe_failures.txt`
- 重跑 idempotent：已 cached 的跳过
- 预估 245 GB（2.5 GB/image 均值）

### `prebake_seta_300.sh` — docker build 300 个

- `CONCURRENCY=2`（默认）：docker build 有 layer lock，>2 并行收益递减
- 注入和 `run_unified_harbor.build_image()` 相同的 build-args：`HTTPS_PROXY=clash, NO_PROXY=tsinghua`
- 每 build 重试 2 次，失败 task id 写入 `/tmp/prebake_seta_failures.txt`
- image tag: `unified-seta-synth-<id>:latest`（避免跟 `unified-seta-<id>:latest` 的 baseline_30 撞名）
- 预估 20-30 GB clash + 60-80 GB 清华源流量

### `prebake_all.sh` — 一键入口

先 SWE 后 SETA 顺序执行。若 SWE 挂停止，避免 SETA 也消耗流量。

## 失败恢复

失败 task id 自动记录：

```bash
# SWE 失败重试
while read img; do docker pull "$img"; done < /tmp/prebake_swe_failures.txt

# SETA 失败重试
bash prebake_seta_300.sh --from-list /tmp/prebake_seta_failures.txt
```

## 跑完后校验

```bash
# SWE 100 全在本地？
diff <(grep -v '^#' swe_lite_100_images.txt | grep -v '^$' | sort) \
     <(docker images --format "{{.Repository}}:{{.Tag}}" | grep xingyaoww | sort) | head

# SETA 300 全在本地？
for tid in $(grep -v '^#' seta_300.txt | grep -v '^$'); do
  docker images -q "unified-seta-synth-${tid}:latest" >/dev/null || echo "MISSING: $tid"
done
```

## 跑 eval 时 runner 能 reuse 吗？

- **SWE**：`run_unified_swe.py` 里 `instance_id_to_image()` 生成的 tag 就是 `xingyaoww/sweb.eval.x86_64.<id>:latest`，与 prebake 完全一致。直接跑即可，会全 cache 命中。
- **SETA synth**：当前 `run_unified_harbor.py` 的 `DATASET_PATHS["seta"]` 只指向 `seta_baseline_30`。要跑 synth_data 需扩一个 dataset option — 见下节。

## SETA synth 跑 eval 的 runner 扩展（TODO，跑前做）

需要在 `run_unified_harbor.py` 加：

```python
DATASET_PATHS = {
    ...
    "seta-synth": BASE_DIR / "datasets/seta/dataset/synth_data_harbor",  # 新加
}
```

以及 `main()` 里 `--dataset` 选项加 `"seta-synth"`。`build_image()` 已经会产出 `unified-seta-synth-<id>:latest` 对应 `dataset_tag="seta-synth"`，prebake 的 image tag 对齐，命中 cache。

这个改动 5 行，跑 eval 前做。
