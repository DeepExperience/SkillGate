"""
Patch TB 2.0 tests/test.sh to avoid the `uv` → Python 3.13 download that times out
behind the remote host's proxy (same root cause as SETA's patch, variant below).

TB 2.0 bulk boilerplate (30 tasks, sha256 4770437ea96c):
    curl astral.sh/uv/0.9.5/install.sh | sh
    uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 pytest --ctrf /logs/verifier/ctrf.json ...

We replace it with SkillsBench-style system pip (no Python download):
    pip install --break-system-packages pytest==8.4.1 pytest-json-ctrf==0.3.5
    pytest --ctrf /logs/verifier/ctrf.json ...

Strategy: only patch the bulk-shared variant (sha256 prefix 4770437ea96c, 30 tasks).
The other 58 variants often include task-specific pip installs (e.g. numpy, torch,
scipy) — leave those alone; they need manual review.

Idempotent: skips files already marked PATCHED.
"""
from __future__ import annotations
import hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BULK_HASH_PREFIX = "4770437ea96c"
PATCHED_MARKER = "# PATCHED: tb2-system-pip variant"

PATCHED_BODY = """#!/bin/bash
# PATCHED: tb2-system-pip variant (was: TB 2.0 boilerplate that uv-downloaded cpython-3.13).
# uv's Python download from python-build-standalone times out behind the remote host proxy.
# Ubuntu base images ship python3 + pip; we use them directly like SkillsBench does.
set -u

mkdir -p /logs/verifier

if ! command -v pip >/dev/null 2>&1 && ! command -v pip3 >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq python3-pip
fi

PIP=$(command -v pip3 || command -v pip)
$PIP install --break-system-packages -q pytest==8.4.1 pytest-json-ctrf==0.3.5 || true

pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
rc=$?

if [ $rc -eq 0 ]; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
exit 0
"""

def main(dry_run: bool = False) -> None:
    patched = already = skipped_custom = missing = 0
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir() or d.name.startswith('.'):
            continue
        f = d / "tests" / "test.sh"
        if not f.exists():
            missing += 1
            continue
        body = f.read_bytes()
        if PATCHED_MARKER.encode() in body:
            already += 1
            continue
        h = hashlib.sha256(body).hexdigest()[:12]
        if h != BULK_HASH_PREFIX:
            skipped_custom += 1
            continue
        if not dry_run:
            f.write_text(PATCHED_BODY)
        patched += 1

    print(f"patched bulk     : {patched}")
    print(f"already patched  : {already}")
    print(f"custom test.sh   : {skipped_custom}  (left unchanged — review per-task)")
    print(f"no test.sh       : {missing}")

if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
