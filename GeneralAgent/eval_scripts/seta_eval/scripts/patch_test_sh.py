"""
Patch SETA Harbor tasks' tests/test.sh to remove the uv→Python-3.13 download
that times out behind the remote host proxy. Replace with the SkillsBench-style
`pip3 install --break-system-packages pytest` boilerplate.

Strategy: only patch the bulk-shared variant (sha256 prefix fc9835e405cb,
1114/1376 tasks). Custom test.sh variants are left untouched — they may install
task-specific deps and require manual review.

Idempotent: skips files whose contents already equal the patched body.
"""
from __future__ import annotations
import hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "dataset" / "synth_data_harbor"
BULK_HASH_PREFIX = "fc9835e405cb"          # the SETA boilerplate, 1114 tasks
PATCHED_MARKER = "# PATCHED: system-pip variant"

PATCHED_BODY = """#!/bin/bash
# PATCHED: system-pip variant (was: SETA boilerplate that uv-downloaded Python 3.13).
# The original boilerplate timed out behind the remote host's proxy because
# `uv init` pulls a 30MB cpython tarball from python-build-standalone.
# Ubuntu 24.04 ships python3.12 + python3-pip, which suffices for pytest.
set -u

mkdir -p /logs/verifier

# Make sure pip is available (idempotent: ubuntu:24.04 base has python3, may not have pip).
if ! command -v pip3 >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq python3-pip
fi

pip3 install --break-system-packages -q pytest==8.4.1 || true

pytest /tests/test_outputs.py -rA
rc=$?

if [ $rc -eq 0 ]; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
exit 0
"""

def main(dry_run: bool = False) -> None:
    if not ROOT.exists():
        sys.exit(f"not found: {ROOT}")

    patched, skipped_custom, already_patched, missing = 0, 0, 0, 0
    for d in sorted(ROOT.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 1<<31):
        if not d.is_dir():
            continue
        f = d / "tests" / "test.sh"
        if not f.exists():
            missing += 1
            continue
        body = f.read_bytes()
        if PATCHED_MARKER.encode() in body:
            already_patched += 1
            continue
        h = hashlib.sha256(body).hexdigest()[:12]
        if h != BULK_HASH_PREFIX:
            skipped_custom += 1
            continue
        if not dry_run:
            f.write_text(PATCHED_BODY)
        patched += 1

    print(f"patched           : {patched}")
    print(f"already patched   : {already_patched}")
    print(f"custom test.sh    : {skipped_custom}  (left unchanged — review manually if needed)")
    print(f"missing test.sh   : {missing}")
    print(f"total task dirs   : {patched + already_patched + skipped_custom + missing}")

if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
