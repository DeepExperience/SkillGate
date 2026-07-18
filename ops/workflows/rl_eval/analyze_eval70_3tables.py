#!/usr/bin/env python3
"""3-table analysis for eval70 (oracle self-read arm), apples-to-apples across
checkpoints. Tables: (T1) task-level pass@4 per bench, (T2) per-task
threshold counts (>=1/4,>=2/4,>=3/4,4/4 of 70), (T3) behavior (task pass@1,
strict skill-read %, P(resolved|read), P(resolved|noread), <skill_reasoning> rate).

Usage: analyze_eval70_3tables.py <label>=<results_root> [<label>=<root> ...]
"""
import json, sys, os, glob
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "GeneralAgent", "sft_data_collection"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "GeneralAgent/sft_data_collection"))
from collect_successes import detect_skill_use  # noqa: E402

BENCH_DIRS = {"tb2": "tb2", "seta-synth": "seta", "skillsbench-no-skills": "sb_ns", "swe": "swe", "claw": "claw"}


def first_assistant_skill_reasoning(messages):
    for m in messages:
        if m.get("role") == "assistant":
            return "<skill_reasoning>" in (m.get("content") or "")
    return False


def collect(root):
    trials = []  # dict(bench, task, resolved, read, sr, has_traj)
    for bdir, bench in BENCH_DIRS.items():
        for leaf in sorted(glob.glob(os.path.join(root, "results", bdir, "*"))):
            inc = os.path.join(leaf, "incremental.jsonl")
            if not os.path.isfile(inc):
                continue
            recs = [json.loads(l) for l in open(inc) if l.strip()]
            if not recs:
                continue
            tjs = glob.glob(os.path.join(leaf, "trajectories", "*.json"))
            traj_info = {}
            for tj_path in tjs:
                task_stem = os.path.splitext(os.path.basename(tj_path))[0]
                try:
                    tj = json.load(open(tj_path))
                    msgs = tj.get("messages", []) or []
                    # used_skill_via_path is path-pattern based; injected names only
                    # feed the (unused) auxiliary name signal, and the field is an int
                    # count here, so pass [] explicitly.
                    info = detect_skill_use(msgs, [])
                    read = bool(info.get("used_skill_via_path", False))
                    sr = first_assistant_skill_reasoning(msgs)
                    traj_info[task_stem] = (read, sr, True,
                                            info.get("read_skill_names", []),
                                            info.get("read_skill_names_agent", []))
                except Exception as e:
                    print(f"  [warn] traj parse failed for {tj_path}: {e!r}", file=sys.stderr)
            fallback_task = os.path.splitext(os.path.basename(tjs[0]))[0] if len(tjs) == 1 else None
            fallback_info = next(iter(traj_info.values()), (False, False, False, [], [])) if len(traj_info) == 1 else (False, False, False, [], [])
            # Each leaf is one planned trial. Retry runs append a later result to
            # the same incremental.jsonl, so count only the final attempt.
            rec = recs[-1]
            resolved = bool(rec.get("resolved"))
            err = bool(rec.get("error"))
            task = str(rec.get("task_id") or rec.get("instance_id") or fallback_task or os.path.basename(leaf))
            read, sr, has_traj, read_names, read_names_agent = traj_info.get(task, fallback_info)
            trials.append(dict(bench=bench, task=task, resolved=resolved, error=err, read=read, sr=sr,
                               has_traj=has_traj, read_names=read_names, read_names_agent=read_names_agent,
                               leaf=leaf))
    return trials


def analyze(trials):
    benches = ["tb2", "seta", "sb_ns", "swe", "claw"]
    # T1: per-bench trial-level resolved/total
    t1 = {}
    for b in benches:
        bt = [t for t in trials if t["bench"] == b]
        n = len(bt)
        npass = sum(t["resolved"] for t in bt)
        nerr = sum(t["error"] for t in bt)
        t1[b] = (npass, n, nerr)
    npass = sum(t["resolved"] for t in trials)
    n = len(trials)
    nerr = sum(t["error"] for t in trials)
    t1["ALL"] = (npass, n, nerr)
    # T1 task pass@4 and T2 resolved-of-4 thresholds.
    bytask = defaultdict(list)
    for t in trials:
        bytask[(t["bench"], t["task"])].append(t["resolved"])
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    ntasks = len(bytask)
    for k, rs in bytask.items():
        c = sum(rs)
        for thr in (1, 2, 3, 4):
            if c >= thr:
                counts[thr] += 1
    t1_task = {}
    for bench in benches:
        task_rows = [rs for (task_bench, _task), rs in bytask.items() if task_bench == bench]
        t1_task[bench] = (sum(any(rs) for rs in task_rows), len(task_rows))
    t1_task["ALL"] = (counts[1], ntasks)
    # T3: behavior
    read_trials = [t for t in trials if t["read"]]
    noread_trials = [t for t in trials if not t["read"]]
    p_read = sum(t["resolved"] for t in read_trials) / max(len(read_trials), 1)
    p_noread = sum(t["resolved"] for t in noread_trials) / max(len(noread_trials), 1)
    t3 = dict(
        pass1=npass / max(n, 1),
        strict_read=len(read_trials) / max(n, 1),
        p_resolved_read=p_read,
        p_resolved_noread=p_noread,
        n_read=len(read_trials),
        n_noread=len(noread_trials),
        skill_reasoning_rate=sum(t["sr"] for t in trials) / max(n, 1),
        n_traj=sum(t["has_traj"] for t in trials),
    )
    return dict(t1=t1, t1_task=t1_task, t2=dict(counts=counts, ntasks=ntasks), t3=t3)


def main():
    runs = {}
    dump_dir = os.environ.get("EVAL70_DUMP_TRIALS_DIR", "")
    for arg in sys.argv[1:]:
        label, root = arg.split("=", 1)
        trials = collect(root)
        runs[label] = (trials, analyze(trials))
        print(f"[{label}] collected {len(trials)} trials from {root}")
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            dump_path = os.path.join(dump_dir, f"trials_{label}.jsonl")
            with open(dump_path, "w") as f:
                for t in trials:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            print(f"[{label}] per-trial dump -> {dump_path}")
    labels = list(runs.keys())
    benches = ["tb2", "seta", "sb_ns", "swe", "claw", "ALL"]

    print("\n================ T1: task-level pass@4 (>=1 success in 4 repeats) ================")
    hdr = "bench      " + "".join(f"{l:>22}" for l in labels)
    print(hdr)
    for b in benches:
        row = f"{b:<11}"
        for l in labels:
            npass, n = runs[l][1]["t1_task"][b]
            row += f"{f'{npass}/{n}={100*npass/max(n,1):.1f}%':>22}"
        print(row)

    print("\n================ T2: per-task threshold (of 70 tasks) ================")
    print("threshold " + "".join(f"{l:>16}" for l in labels))
    for thr in (1, 2, 3, 4):
        row = f">={thr}/4     "
        for l in labels:
            row += f"{runs[l][1]['t2']['counts'][thr]:>16}"
        print(row)
    row = "n_tasks    "
    for l in labels:
        row += f"{runs[l][1]['t2']['ntasks']:>16}"
    print(row)

    print("\n================ T3: behavior ================")
    metrics = [
        ("task pass@1", lambda t: f"{100*t['pass1']:.1f}%"),
        ("strict skill-read", lambda t: f"{100*t['strict_read']:.1f}% ({t['n_read']}/{t['n_read']+t['n_noread']})"),
        ("P(resolved|read)", lambda t: f"{100*t['p_resolved_read']:.1f}%"),
        ("P(resolved|noread)", lambda t: f"{100*t['p_resolved_noread']:.1f}% (N={t['n_noread']})"),
        ("<skill_reasoning> rate", lambda t: f"{100*t['skill_reasoning_rate']:.1f}%"),
        ("n_traj parsed", lambda t: str(t['n_traj'])),
    ]
    print(f"{'metric':<24}" + "".join(f"{l:>26}" for l in labels))
    for name, fn in metrics:
        row = f"{name:<24}"
        for l in labels:
            row += f"{fn(runs[l][1]['t3']):>26}"
        print(row)


if __name__ == "__main__":
    main()
