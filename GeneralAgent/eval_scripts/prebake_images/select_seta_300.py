#!/usr/bin/env python3
"""Select 300 SETA synth_data tasks best matching "personal assistant / file operation" theme.

Algorithm (LLM-free, deterministic):
    score = 2 × (file_op_matches) + 2 × (personal_assistant_matches)
          + 1.5 × (data_format_matches) + 2 × (document_workflow_matches)
          - 3 × (server_admin_negative_matches)

    Signals pulled from: task.yaml's `instruction`, `category`, `tags` fields.

Result: writes task-id list to seta_300.txt (reproducible: same input → same output).

Usage:
    python3 select_seta_300.py                 # default: writes seta_300.txt
    python3 select_seta_300.py --n 500         # select top 500 instead
    python3 select_seta_300.py --preview       # print top/bottom with scores, don't write
    python3 select_seta_300.py --out PATH      # custom output file
"""

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

SETA_SYN = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL")) / "datasets/seta/dataset/synth_data"

# ---- Scoring regexes -----------------------------------------------------------

FILE_OPS = [
    r"\bfile(s)?\b", r"\bdirector(y|ies)\b", r"\bfolder(s)?\b",
    r"\bfilename", r"\bextension", r"\bpath(s)?\b",
    r"\barchive", r"\bbackup", r"\bcompress", r"\bextract",
    r"\brename", r"\bcopy\b", r"\bmove\b", r"\borganize",
    r"\bdeduplicate", r"\bmerge\b",
]

PERSONAL_ASSIST = [
    r"\breport", r"\bschedule", r"\breminder",
    r"\bgenerate.*summary", r"\binventory\b",
    r"\blist all", r"\bfind all", r"\bcount",
    r"\bsearch", r"\bfilter", r"\bsort", r"\bconvert",
]

DATA_FORMAT = [
    r"\bcsv\b", r"\bjson\b", r"\byaml\b", r"\bxml\b",
    r"\bmarkdown", r"\bhtml\b", r"\blog file", r"\btext file",
    r"\bexcel\b", r"\bpdf\b", r"\bparse(r|s|d)?\b",
]

DOC_WORK = [
    r"\bdocument", r"\brename.*photos?", r"\bcalendar",
    r"\bemail", r"\bnote(s)?\b", r"\btimestamp", r"\bmetadata\b",
]

# Strong negative: server-admin / kernel / firewall — not personal-assistant scope.
NEGATIVE = [
    r"\bkernel", r"\bdriver\b", r"\bsystemd", r"\bboot loader", r"\bgrub",
    r"\bDNS\b", r"\bDHCP\b", r"\biptables", r"\bfirewall\b", r"\bnftables",
    r"\bapache\b", r"\bnginx", r"\bmysql", r"\bpostgres", r"\bmariadb",
    r"\bredis", r"\belasticsearch", r"\bkafka", r"\bselinux",
    r"\bssh.*(server|daemon|sshd)\b", r"\blxc\b", r"\bkubernet",
    r"\bDocker.*container", r"\bLDAP", r"\bldap",
    r"\bInfiniBand", r"\bBGP\b", r"\bOSPF\b", r"\bVLAN\b",
    r"\bcpuinfo", r"\bcpufreq", r"\bperformance governor",
    r"\binterrupt", r"\bcore\s*dump", r"\bkvm\b", r"\b/proc/sys",
    r"\bfirewalld", r"\bapparmor", r"\bauditd",
    r"\bopenssl.*certificate", r"\bkerberos", r"\bcipher",
    r"\bIOMMU", r"\bPCI\b", r"\bIRQ\b",
]

def count_matches(text: str, patterns: list[str]) -> int:
    if not text:
        return 0
    return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))


def score_task(text: str) -> tuple[float, int, int]:
    """Return (final_score, positive_matches, negative_matches)."""
    pos = (
        count_matches(text, FILE_OPS) * 2
        + count_matches(text, PERSONAL_ASSIST) * 2
        + count_matches(text, DATA_FORMAT) * 1.5
        + count_matches(text, DOC_WORK) * 2
    )
    neg = count_matches(text, NEGATIVE) * 3
    return pos - neg, pos, neg


def load_and_score_all() -> list[dict]:
    """Return list of dict per task, sorted by score desc."""
    entries = []
    for task_id in sorted(os.listdir(SETA_SYN), key=lambda x: int(x) if x.isdigit() else 10**9):
        p = SETA_SYN / task_id / "task.yaml"
        if not p.exists():
            continue
        try:
            d = yaml.safe_load(p.read_text())
        except Exception:
            continue
        instr = d.get("instruction", "") or ""
        cat = d.get("category", "") or ""
        tags = d.get("tags") or []
        tag_str = " ".join(str(t) for t in tags)
        combined = f"{instr} {cat} {tag_str}"
        final, pos, neg = score_task(combined)
        entries.append({
            "task_id": task_id,
            "score": final,
            "pos": pos,
            "neg": neg,
            "difficulty": d.get("difficulty"),
            "category": cat,
            "instruction_preview": instr[:100].replace("\n", " "),
        })
    entries.sort(key=lambda x: (-x["score"], int(x["task_id"]) if x["task_id"].isdigit() else 10**9))
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="Number of tasks to select (default 300)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "seta_300.txt"),
                    help="Output file path (default: seta_300.txt next to this script)")
    ap.add_argument("--preview", action="store_true", help="Print preview table, don't write")
    args = ap.parse_args()

    entries = load_and_score_all()
    top = entries[: args.n]

    diff_c = Counter(e["difficulty"] for e in top)
    cat_c = Counter(e["category"] for e in top)

    print(f"Scored {len(entries)} SETA synth_data tasks")
    print(f"Selecting top {args.n}:")
    print(f"  score range: [{top[-1]['score']:.1f}, {top[0]['score']:.1f}]")
    print(f"  median score in selection: {top[len(top)//2]['score']:.1f}")
    print(f"  difficulty: {dict(diff_c)}")
    print(f"  top-5 categories: {dict(cat_c.most_common(5))}")

    if args.preview:
        print(f"\n=== top 10 ===")
        for e in top[:10]:
            print(f"  [{e['task_id']:>4s}] score={e['score']:5.1f} diff={e['difficulty']:6s} "
                  f"cat={e['category'][:25]:25s} {e['instruction_preview']}")
        print(f"\n=== rank {args.n-2} to {args.n+2} (boundary) ===")
        for e in entries[max(0, args.n-3): args.n+3]:
            marker = "←" if e in top else " "
            print(f" {marker}[{e['task_id']:>4s}] score={e['score']:5.1f} diff={e['difficulty']:6s} "
                  f"cat={e['category'][:25]:25s} {e['instruction_preview']}")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# SETA 300 tasks selected by personal-assistant/file-ops scoring\n")
        f.write(f"# Generated by {Path(__file__).name}; criteria: see docstring\n")
        f.write(f"# score range: {top[-1]['score']:.1f} to {top[0]['score']:.1f}\n")
        f.write(f"# difficulty dist: {dict(diff_c)}\n")
        f.write("# format: one task_id per line (matches synth_data/<id>)\n\n")
        for e in top:
            f.write(f"{e['task_id']}\n")

    print(f"\nWrote {len(top)} task ids → {out_path}")


if __name__ == "__main__":
    main()
