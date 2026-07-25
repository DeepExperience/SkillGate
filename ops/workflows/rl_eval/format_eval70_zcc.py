#!/usr/bin/env python3
"""Render checkpoint-set results into the z_cc_terminal_imgs comparison layout.

Reuses collect/analyze from analyze_eval70_3tables.py.  The first table follows
the historical comparison image: one method per row with trial-level success
counts over ALL=280 and the five benchmark subsets.

Tables:
 T1 trial success: model | ALL trials | claw trials | ...
 T2 task pass@R: model | ALL tasks | claw tasks | ...
 T3 task pass@R thresholds: model | >=1/R | ... | R/R
 T4 behavior: model | trial success | strict read | P(success|read/noread) | reasoning
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from analyze_eval70_3tables import collect, analyze  # noqa: E402

BENCH_COLS = [("ALL", "ALL"), ("claw", "claw"), ("sb_ns", "sb_ns"),
              ("seta_synth", "seta"), ("swe_lite", "swe"), ("tb2", "tb2")]
TABLE_CONTEXT = os.environ.get("EVAL70_TABLE_CONTEXT", "oracle skill")
TABLE_STYLE = os.environ.get("EVAL70_TABLE_STYLE", "full")
REPEATS = int(os.environ.get("EVAL70_REPEATS", "4"))


def cell_n_pct(n, d):
    return f"{n} ({100*n/d:.1f}%)" if d else f"{n} (—)"


def main():
    if TABLE_STYLE not in {"full", "main-only"}:
        raise SystemExit(f"invalid EVAL70_TABLE_STYLE={TABLE_STYLE!r}; expected full or main-only")
    if REPEATS not in {1, 2, 3, 4}:
        raise SystemExit(f"invalid EVAL70_REPEATS={REPEATS}; expected 1..4")

    if len(sys.argv) <= 1:
        raise SystemExit("pass one or more label=owner_local_eval_row arguments")
    runs = []
    for arg in sys.argv[1:]:
        if "=" not in arg:
            raise SystemExit(f"expected label=results_root argument, got: {arg}")
        label, root = arg.split("=", 1)
        runs.append((label, root))

    data = {}
    for label, root in runs:
        tr = collect(root)
        data[label] = (tr, analyze(tr))

    first_label = runs[0][0]

    print(f"## T1 — trial-level success ({TABLE_CONTEXT}; one row per method)\n")
    trial_headers = [
        f"{col} ({data[first_label][1]['t1'][key][1]})"
        for col, key in BENCH_COLS
    ]
    print("| model | " + " | ".join(trial_headers) + " |")
    print("|" + "---|" * (len(BENCH_COLS) + 1))
    for label, _ in runs:
        t1 = data[label][1]["t1"]
        cells = []
        for _col, key in BENCH_COLS:
            npass, n, _nerr = t1[key]
            cells.append(cell_n_pct(npass, n))
        print(f"| {label} | " + " | ".join(cells) + " |")

    if TABLE_STYLE == "main-only":
        return

    print(
        f"\n> T1 uses all independent trials. T2/T3 aggregate the {REPEATS} "
        f"repeat{'s' if REPEATS != 1 else ''} of each task and report task-level pass@{REPEATS}.\n"
    )

    print(
        f"## T2 — task pass@{REPEATS} (>=1 success in {REPEATS} "
        f"repeat{'s' if REPEATS != 1 else ''}; {TABLE_CONTEXT})\n"
    )
    task_headers = [
        f"{col} ({data[first_label][1]['t1_task'][key][1]})"
        for col, key in BENCH_COLS
    ]
    print("| model | " + " | ".join(task_headers) + " |")
    print("|" + "---|" * (len(BENCH_COLS) + 1))
    for label, _ in runs:
        t1 = data[label][1]["t1_task"]
        cells = []
        for _col, key in BENCH_COLS:
            npass, n = t1[key]
            cells.append(cell_n_pct(npass, n))
        print(f"| {label} | " + " | ".join(cells) + " |")

    ntasks = data[first_label][1]["t2"]["ntasks"]
    print(f"\n## T3 — task pass@{REPEATS} thresholds ({TABLE_CONTEXT}, % of {ntasks} tasks)\n")
    threshold_headers = [
        f"pass@{REPEATS} (≥1/{REPEATS})" if threshold == 1
        else f"{threshold}/{REPEATS} 全过" if threshold == REPEATS
        else f"≥{threshold}/{REPEATS}"
        for threshold in range(1, REPEATS + 1)
    ]
    print("| model | " + " | ".join(threshold_headers) + " |")
    print("|" + "---|" * (REPEATS + 1))
    for label, _ in runs:
        c = data[label][1]["t2"]["counts"]
        nt = data[label][1]["t2"]["ntasks"]
        cells = [cell_n_pct(c[threshold], nt) for threshold in range(1, REPEATS + 1)]
        print(f"| {label} | " + " | ".join(cells) + " |")

    print(f"\n## T4 — 读取行为总览 ({TABLE_CONTEXT})\n")
    print("| model | trial success | 显式读 skill strict_used | 读 skill 后成功率 | 不读 skill 后成功率 | skill_reasoning 出现率 |")
    print("|---|---|---|---|---|---|")
    for label, _ in runs:
        t3 = data[label][1]["t3"]
        noread = f"{100*t3['p_resolved_noread']:.1f}%" if t3['n_noread'] else f"— (N=0)"
        print(f"| {label} | {100*t3['pass1']:.1f}% | {100*t3['strict_read']:.1f}% | "
              f"{100*t3['p_resolved_read']:.1f}% | {noread} | {100*t3['skill_reasoning_rate']:.1f}% |")


if __name__ == "__main__":
    main()
