#!/usr/bin/env python3
"""Mark a run's validity in its run.json + append a correction line to INDEX.jsonl.

Usage:
  python3 ops/workflows/rl_eval/run_validity.py <run_id> quarantined --reason "proxy bug ate verifier output"
  python3 ops/workflows/rl_eval/run_validity.py <run_id> superseded --by <other_run_id>
  python3 ops/workflows/rl_eval/run_validity.py <run_id> valid

Never delete artifacts to invalidate a run — set the flag so aggregation
scripts (and future operators) can filter without reading the journal.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "GeneralAgent" / "sft_data_collection"))
from run_manifest import manifest_path, index_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("validity", choices=["valid", "quarantined", "superseded"])
    ap.add_argument("--reason", default="")
    ap.add_argument("--by", default="", help="run_id that supersedes this one")
    args = ap.parse_args()

    mp = manifest_path(args.run_id)
    if not mp.exists():
        raise SystemExit(f"no run.json at {mp} — is the run_id right?")
    manifest = json.loads(mp.read_text())
    manifest["validity"] = args.validity
    manifest["validity_reason"] = args.reason
    manifest["superseded_by"] = args.by
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    with index_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ts": manifest["updated_at"],
            "run_id": args.run_id,
            "event": "validity_change",
            "validity": args.validity,
            "reason": args.reason,
            "superseded_by": args.by,
        }, ensure_ascii=False) + "\n")
    print(f"{args.run_id}: validity={args.validity}"
          + (f" reason={args.reason}" if args.reason else "")
          + (f" superseded_by={args.by}" if args.by else ""))


if __name__ == "__main__":
    main()
