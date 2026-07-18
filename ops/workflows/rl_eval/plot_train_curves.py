#!/usr/bin/env python3
"""Pairwise RL training-curve comparisons (2 runs each), resumed segments stitched.

Outputs 3 comparison folders under docs/plot_analysis/, each with 3 figures
(train dynamics / skill behavior / per-bench):
  1_skillaware_oracle_vs_retrieve   : skill-aware RL, oracle vs retrieve skill
  2_oracle_baseRL_vs_skillaware     : on oracle skill, baseRL vs skill-aware
  3_retrieve_baseRL_vs_skillaware   : on retrieve skill, baseRL vs skill-aware

jsonl (rollout_result/train/<step>.jsonl) -> reward, length, truncation, read-frac
(uniform response-text detection so baseRL & skill-aware compare apples-to-apples),
per-bench reward/read, and skill-subgroup metrics (skill-aware only: bonus, winner, gap).
W&B history -> entropy, ppo_kl, kl_loss, grad_norm, pg_clipfrac, pg_loss.
"""
import os, glob, json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_ROOT = Path("experiments/rl/runs")
OUT = "docs/plot_analysis"
SKILL_READ = re.compile(r"SKILL\.md|\.claude/skills/", re.I)

SEGS = {
    "sa_retrieve": ["4bench_cp2_70k_lr1e6_active128_nolen_skillgate30_from_sft_20260604_233222",
        "4bench_cp2_70k_lr1e6_active128_save5_diskray_resume20_to100_20260607_101805",
        "4bench_cp2_70k_lr1e6_active128_save5_diskray_resume24_to100_20260607_193847",
        "4bench_cp2_70k_lr1e6_active128_save5_diskray_resume29_cpusetonly_20260607_232027",
        "4bench_cp2_70k_lr1e6_active128_save5_diskray_resume34_subreaper_20260608_124544",
        "4bench_cp2_70k_lr1e6_active128_save5_resume59_conc128_netnsfix_20260609_192502",
        "4bench_cp2_70k_lr1e6_active128_save5_resume89_conc128_podrestart_20260610_162522",
        "4bench_cp2_70k_lr1e6_active128_save5_resume89_conc128_podrestart_20260610_200431"],
    "base_retrieve": ["4bench_cp2_70k_lr1e6_active128_baseline_noskillrw_nogate_from_sft_20260612_110138"],
    "sa_oracle": ["4bench_cp2_70k_lr1e6_active128_nolen_skillgate30_oracle1_from_sft_20260613_112048",
        "4bench_cp2_70k_lr1e6_active128_nolen_skillgate30_oracle1_from_sft_20260614_060952",
        "4bench_cp2_70k_lr1e6_active128_nolen_skillgate30_oracle1_from_sft_20260615_210247"],
    "base_oracle": ["4bench_cp2_70k_lr1e6_active128_oracle1_baseline_noskillrw_nogate_from_sft_20260614_231248",
        "4bench_cp2_70k_lr1e6_active128_oracle1_baseline_noskillrw_nogate_from_sft_20260615_161851"],
}
RED, BLUE, GRAY = "#d62728", "#1f77b4", "#7f7f7f"
COMPARISONS = [
    ("1_skillaware_oracle_vs_retrieve", "skill-aware RL: oracle vs retrieve skill",
        [("skill-aware · oracle", "sa_oracle", RED), ("skill-aware · retrieve", "sa_retrieve", BLUE)]),
    ("2_oracle_baseRL_vs_skillaware", "on ORACLE skill: baseRL vs skill-aware RL",
        [("skill-aware · oracle", "sa_oracle", RED), ("baseRL · oracle", "base_oracle", GRAY)]),
    ("3_retrieve_baseRL_vs_skillaware", "on RETRIEVE skill: baseRL vs skill-aware RL",
        [("skill-aware · retrieve", "sa_retrieve", BLUE), ("baseRL · retrieve", "base_retrieve", GRAY)]),
]
WKEYS = ["rollout/entropy_mc/mean", "train/ppo_kl", "train/kl_loss", "train/grad_norm",
         "train/pg_clipfrac", "train/pg_loss"]
BENCHES = ["seta_synth", "tb2", "sb_ns", "swe_lite"]


def segment_index():
    """Resolve historical W&B/run names to their canonical owner segments."""
    result = {}
    for manifest_path in RUNS_ROOT.glob("*/segments/*/run.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        segment_dir = manifest_path.parent
        for key in (segment_dir.name, manifest.get("segment_id"), manifest.get("legacy_run_id")):
            if key:
                result[str(key)] = segment_dir
    return result


SEGMENT_INDEX = segment_index()


def _f(x):
    try: return float(x)
    except: return np.nan


def jsonl_seg(seg):
    out = {}
    segment_dir = SEGMENT_INDEX.get(seg)
    if segment_dir is None:
        return out
    for f in glob.glob(str(segment_dir / "rollout_result" / "train" / "*.jsonl")):
        step = int(re.findall(r"(\d+)\.jsonl", f)[0])
        recs = []
        for line in open(f):
            line = line.strip()
            if line:
                try: recs.append(json.loads(line))
                except: pass
        if not recs:
            continue
        def rw(r):
            x = r.get("reward"); return x if isinstance(x, dict) else {}
        raw = [rw(r).get("raw_score") for r in recs]; raw = [x for x in raw if isinstance(x, (int, float))]
        rlen = [r.get("response_length") for r in recs if isinstance(r.get("response_length"), (int, float))]
        trunc = [1.0 if r.get("status") == "truncated" else 0.0 for r in recs]
        read = [1.0 if SKILL_READ.search(r.get("response", "") or "") else 0.0 for r in recs]
        m = dict(reward=np.mean(raw) if raw else np.nan,
                 resp_len=np.mean(rlen) if rlen else np.nan,
                 trunc=np.mean(trunc) if trunc else np.nan,
                 read_frac=np.mean(read) if read else np.nan)
        used = [rw(r).get("skill_group_used_strict") for r in recs]; used = [x for x in used if isinstance(x, (int, float))]
        if used: m["read_strict"] = np.mean(used)
        bonus = [rw(r).get("skill_group_bonus") for r in recs]; bonus = [abs(x) for x in bonus if isinstance(x, (int, float))]
        if bonus: m["bonus_abs"] = np.mean(bonus)
        for fld, key in [("skill_group_preferred_read", "pref_read"), ("skill_group_preferred_no_read", "pref_noread"),
                          ("skill_group_gap", "gap")]:
            v = [rw(r).get(fld) for r in recs]; v = [x for x in v if isinstance(x, (int, float))]
            if v: m[key] = np.mean(v)
        # per-bench reward + read
        for b in BENCHES:
            br = [rw(r).get("raw_score") for r in recs if rw(r).get("bench") == b]
            br = [x for x in br if isinstance(x, (int, float))]
            bk = [1.0 if SKILL_READ.search(r.get("response", "") or "") else 0.0 for r in recs if rw(r).get("bench") == b]
            if br: m[f"reward__{b}"] = np.mean(br)
            if bk: m[f"read__{b}"] = np.mean(bk)
        out[step] = m
    return out


def stitch_jsonl(segs):
    merged = {}
    per_seg = {}
    for seg in segs:
        d = jsonl_seg(seg); per_seg[seg] = sorted(d.keys())
        for step, m in d.items():
            merged[step] = m
    steps = sorted(merged)
    keys = set()
    for m in merged.values(): keys |= set(m.keys())
    return np.array(steps), {k: np.array([merged[s].get(k, np.nan) for s in steps]) for k in keys}, per_seg


def stitch_wandb(segs, runs_by_name, per_seg):
    merged = {}
    for seg in segs:
        if seg not in runs_by_name: continue
        try: df = runs_by_name[seg].history(samples=5000)
        except Exception: continue
        if df is None or len(df) == 0 or "_step" not in df.columns: continue
        wmin = float(np.nanmin(df["_step"].values))
        jmin = min(per_seg.get(seg, [wmin]))
        offset = int(round(jmin - wmin))
        present = [k for k in WKEYS if k in df.columns]
        for _, row in df.iterrows():
            s = row.get("_step")
            if s is None or (isinstance(s, float) and np.isnan(s)): continue
            d = merged.setdefault(int(round(s)) + offset, {})
            for k in present:
                v = row.get(k)
                if v is not None and not (isinstance(v, float) and np.isnan(v)): d[k] = v
    steps = sorted(merged)
    return np.array(steps), {k: np.array([_f(merged[s].get(k)) for s in steps]) for k in WKEYS}


def smooth(y, k=3):
    y = np.asarray(y, float)
    if len(y) < 2: return y
    out = np.copy(y)
    for i in range(len(y)):
        seg = y[max(0, i - k // 2):min(len(y), i + k // 2 + 1)]; seg = seg[~np.isnan(seg)]
        if len(seg): out[i] = seg.mean()
    return out


def panel(ax, runs, getx, gety, title, ylog=False):
    any_data = False
    for r in runs:
        xs, ys = getx(r), gety(r)
        if ys is None or len(xs) == 0 or np.all(np.isnan(ys)): continue
        ax.plot(xs, smooth(ys), color=r["color"], label=r["label"], lw=2.0, alpha=0.9)
        any_data = True
    ax.set_title(title, fontsize=11); ax.set_xlabel("rollout step"); ax.grid(alpha=0.25)
    if ylog: ax.set_yscale("log")
    return any_data


def main():
    import wandb
    api = wandb.Api(timeout=40)
    runs_by_name = {}
    for r in api.runs("relax-rl-agent", per_page=300):
        if r.name not in runs_by_name or (r.summary.get("_step", 0) or 0) > (runs_by_name[r.name].summary.get("_step", 0) or 0):
            runs_by_name[r.name] = r

    cache = {}
    def get_run(key, label, color):
        if key not in cache:
            segs = SEGS[key]
            js, jm, per_seg = stitch_jsonl(segs)
            ws, wm = stitch_wandb(segs, runs_by_name, per_seg)
            cache[key] = dict(js=js, jm=jm, ws=ws, wm=wm)
        c = cache[key]
        return dict(label=label, color=color, js=c["js"], jm=c["jm"], ws=c["ws"], wm=c["wm"])

    for folder, title, members in COMPARISONS:
        runs = [get_run(key, label, color) for label, key, color in members]
        outdir = os.path.join(OUT, folder); os.makedirs(outdir, exist_ok=True)
        jx = lambda r: r["js"]; wx = lambda r: r["ws"]
        jg = lambda k: (lambda r: r["jm"].get(k)); wg = lambda k: (lambda r: r["wm"].get(k))

        # --- Fig 1: training dynamics ---
        fig, ax = plt.subplots(2, 4, figsize=(20, 9))
        specs = [("reward (raw_score)", jx, jg("reward"), False), ("response length (mean)", jx, jg("resp_len"), False),
                 ("truncation frac", jx, jg("trunc"), False), ("entropy (entropy_mc/mean)", wx, wg("rollout/entropy_mc/mean"), False),
                 ("ppo_kl", wx, wg("train/ppo_kl"), False), ("kl_loss", wx, wg("train/kl_loss"), False),
                 ("grad_norm", wx, wg("train/grad_norm"), True), ("pg_clipfrac", wx, wg("train/pg_clipfrac"), False)]
        for a, (t, gx, gy, lg) in zip(ax.flat, specs):
            panel(a, runs, gx, gy, t, ylog=lg)
        ax.flat[0].legend(fontsize=10, loc="best")
        fig.suptitle(f"{title} — training dynamics (resumed segments stitched)", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(outdir, "01_train_dynamics.png"), dpi=120); plt.close(fig)

        # --- Fig 2: skill behavior ---
        fig, ax = plt.subplots(2, 3, figsize=(17, 9))
        specs = [("skill-read frac (response-detected, 可比)", jx, jg("read_frac")),
                 ("strict skill-read (logged, skill-aware only)", jx, jg("read_strict")),
                 ("bonus |value| (skill-aware only)", jx, jg("bonus_abs")),
                 ("winning subgroup: P(read 组胜)", jx, jg("pref_read")),
                 ("winning subgroup: P(no-read 组胜)", jx, jg("pref_noread")),
                 ("read − noread gap (skill-aware only)", jx, jg("gap"))]
        for a, (t, gx, gy) in zip(ax.flat, specs):
            panel(a, runs, gx, gy, t)
        ax.flat[0].legend(fontsize=10, loc="best")
        fig.suptitle(f"{title} — skill-read & subgroup mechanism", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(outdir, "02_skill_behavior.png"), dpi=120); plt.close(fig)

        # --- Fig 3: per-bench reward (top) + per-bench read (bottom) ---
        fig, ax = plt.subplots(2, len(BENCHES), figsize=(5 * len(BENCHES), 9))
        for j, b in enumerate(BENCHES):
            panel(ax[0, j], runs, jx, jg(f"reward__{b}"), f"reward · {b}")
            panel(ax[1, j], runs, jx, jg(f"read__{b}"), f"skill-read frac · {b}")
        ax[0, 0].legend(fontsize=9, loc="best")
        fig.suptitle(f"{title} — per-bench reward (top) & skill-read (bottom)", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(outdir, "03_per_bench.png"), dpi=120); plt.close(fig)
        print(f"[{folder}] wrote 3 figs; runs: " + ", ".join(f"{r['label']}(js{len(r['js'])}/ws{len(r['ws'])})" for r in runs))


if __name__ == "__main__":
    main()
