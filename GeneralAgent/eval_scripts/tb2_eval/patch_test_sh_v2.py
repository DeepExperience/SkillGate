#!/usr/bin/env python3
"""Patch test.sh boilerplate variants missed by v1 patch_test_sh.py.

v1 only matched strict regex (apt-get install -y curl + blank + uv install + blank + source).
v2 relaxes to cover variants like:
  - `apt-get install -y curl imagemagick` (extra pkgs on install line)
  - `apt-get install -y curl gcc`
  - Missing blank line between `sh` and `source ...`

Idempotent — detects both v1 and v2 markers and skips.
"""
import os
import argparse
import re
import sys
from pathlib import Path

TB2_DIR = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL")) / "datasets/terminal-bench-v2"
MARKER_V1 = "[PATCHED 2026-04-19 tb2-uv-cache]"
MARKER_V2 = "[PATCHED 2026-04-20 tb2-uv-cache-v2]"

# Strategy: find just the uv install block (`curl ... astral.sh/uv/.../install.sh`
# optionally preceded by a `# Install uv...` comment; followed by a blank line and
# `source $HOME/.local/bin/env`). Leave apt-get lines alone — even if they duplicate
# curl install, the prebake path skips them (curl/apt cached quick in CN).
# This covers all 20 remaining variants.
RELAXED_RE = re.compile(
    r"(?:^# Install uv[^\n]*\n)?"
    r"curl -LsSf https://astral\.sh/uv/([\d.]+)/install\.sh \| sh\s*\n"
    r"\s*\n?"
    r"source \$HOME/\.local/bin/env\s*\n",
    re.MULTILINE,
)

REPLACEMENT_TEMPLATE = """# {marker} uv/python prebaked; skips ~50-400MB download
# Original (uv {uv_version}):
#   curl -LsSf https://astral.sh/uv/{uv_version}/install.sh | sh
#   source $HOME/.local/bin/env
if [ -f /opt/tb2-uv/uv ]; then
    cp /opt/tb2-uv/uv /usr/local/bin/uv
    cp /opt/tb2-uv/uvx /usr/local/bin/uvx
    export UV_PYTHON_INSTALL_DIR=/opt/tb2-uv/data/python
    export UV_CACHE_DIR=/opt/tb2-uv/cache
else
    # Fallback: cache not injected — fall through to official install path
    curl -LsSf https://astral.sh/uv/{uv_version}/install.sh | sh
    source $HOME/.local/bin/env
fi
"""


def find_targets():
    out = []
    for d in sorted(TB2_DIR.iterdir()):
        if not d.is_dir():
            continue
        ts = d / "tests" / "test.sh"
        if not ts.exists():
            continue
        c = ts.read_text(encoding="utf-8", errors="replace")
        if MARKER_V1 in c or MARKER_V2 in c:
            continue  # any prior patch
        m = RELAXED_RE.search(c)
        if not m:
            continue
        uv_version = m.group(1)
        out.append((d.name, ts, c, uv_version))
    return out


def apply_patch(c: str, uv_version: str) -> str:
    replacement = REPLACEMENT_TEMPLATE.format(
        marker=MARKER_V2,
        uv_version=uv_version,
    )
    return RELAXED_RE.sub(replacement, c)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-show", type=int, default=3)
    args = p.parse_args()
    targets = find_targets()
    print(f"=== v2 PATCH mode: {len(targets)} target test.sh found ===")
    if not targets:
        print("Nothing to patch (all already patched or no matching boilerplate).")
        return
    for i, (name, path, content, uv_ver) in enumerate(targets):
        new = apply_patch(content, uv_ver)
        if i < args.max_show:
            import difflib
            diff = difflib.unified_diff(
                content.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{name}/test.sh",
                tofile=f"b/{name}/test.sh",
                n=2,
            )
            print(f"\n--- {name} (uv {uv_ver}) ---")
            sys.stdout.writelines(diff)
        if not args.dry_run:
            path.write_text(new, encoding="utf-8")
    print(f"\n=== {'DRY-RUN' if args.dry_run else 'APPLIED'}: {len(targets)} files ===")


if __name__ == "__main__":
    main()
