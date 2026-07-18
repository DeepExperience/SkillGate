#!/usr/bin/env python3
"""Patch 52 tb2 test.sh files to use the prebaked uv cache.

Background:
  Default tb2 test.sh boilerplate:
      apt-get install -y curl
      curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
      source $HOME/.local/bin/env
      uvx -p 3.13 -w <deps> pytest ...
  → 50-400MB downloaded per task (curl unbelievably slow via clash), causing
  verifier_timeout(1200-3600s).

Fix:
  Runner injects /opt/tb2-uv/ from ops/cache/pkg/tb2_uv_cache.tar.gz.
  Patched test.sh skips the download step and uses the prebaked uv directly.

Safety:
  - Idempotent: detects `[PATCHED 2026-04-19]` marker and skips.
  - --dry-run prints the exact diff per file, no writes.
  - --revert undoes the patch (restores original lines from marker-embedded
    "Original:" comment).

Usage:
  python3 patch_test_sh.py --dry-run          # show plan
  python3 patch_test_sh.py                     # apply patches
  python3 patch_test_sh.py --revert --dry-run  # preview revert
  python3 patch_test_sh.py --revert            # undo patches
"""

import os
import argparse
import re
import sys
from pathlib import Path

TB2_DIR = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL")) / "datasets/terminal-bench-v2"
MARKER = "[PATCHED 2026-04-19 tb2-uv-cache]"

# The uv install block we're replacing. Matches the official TB2 boilerplate
# (lines 3-10 of adaptive-rejection-sampler/tests/test.sh etc.).
UV_INSTALL_BLOCK_RE = re.compile(
    r"^# Install curl\n"
    r"apt-get update\n"
    r"apt-get install -y curl\n"
    r"\n"
    r"# Install uv\n"
    r"curl -LsSf https://astral\.sh/uv/([\d.]+)/install\.sh \| sh\n"
    r"\n"
    r"source \$HOME/\.local/bin/env\n",
    re.MULTILINE,
)

# The replacement block. Uses /opt/tb2-uv/ which the runner injects at container
# start (tarball from ops/cache/pkg/tb2_uv_cache.tar.gz).
REPLACEMENT_TEMPLATE = """# {marker} skipped online uv/python download (~50-400MB/task); uses prebaked cache
# Original (version {uv_version}): curl -LsSf https://astral.sh/uv/{uv_version}/install.sh | sh
if [ -f /opt/tb2-uv/uv ]; then
    cp /opt/tb2-uv/uv /usr/local/bin/uv
    cp /opt/tb2-uv/uvx /usr/local/bin/uvx
    # data = ~/.local/share/uv (python installations); cache = ~/.cache/uv (wheel cache)
    export UV_PYTHON_INSTALL_DIR=/opt/tb2-uv/data/python
    export UV_CACHE_DIR=/opt/tb2-uv/cache
else
    # Fallback: cache not injected — fall through to official install path
    apt-get update
    apt-get install -y curl
    curl -LsSf https://astral.sh/uv/{uv_version}/install.sh | sh
    source $HOME/.local/bin/env
fi
"""

# Revert pattern: find the patched block and restore original.
REVERT_RE = re.compile(
    re.escape(f"# {MARKER} skipped online uv/python download (~50-400MB/task); uses prebaked cache")
    + r".*?"
    + re.escape("fi\n"),
    re.DOTALL,
)
REVERT_VERSION_RE = re.compile(
    r"# Original \(version ([\d.]+)\):")


def find_targets():
    """Return list of (task_dir, test_sh_path, content, uv_version) for patchable files."""
    targets = []
    for d in sorted(TB2_DIR.iterdir()):
        if not d.is_dir():
            continue
        ts = d / "tests" / "test.sh"
        if not ts.exists():
            continue
        content = ts.read_text(encoding="utf-8", errors="replace")
        m = UV_INSTALL_BLOCK_RE.search(content)
        if not m:
            continue  # custom or already patched
        if MARKER in content:
            continue  # already patched
        targets.append((d.name, ts, content, m.group(1)))
    return targets


def find_patched():
    """Return list of (task_dir, test_sh_path, content, uv_version) currently patched."""
    patched = []
    for d in sorted(TB2_DIR.iterdir()):
        if not d.is_dir():
            continue
        ts = d / "tests" / "test.sh"
        if not ts.exists():
            continue
        content = ts.read_text(encoding="utf-8", errors="replace")
        if MARKER not in content:
            continue
        m = REVERT_VERSION_RE.search(content)
        uv_version = m.group(1) if m else "0.9.5"
        patched.append((d.name, ts, content, uv_version))
    return patched


def apply_patch(content: str, uv_version: str) -> str:
    replacement = REPLACEMENT_TEMPLATE.format(marker=MARKER, uv_version=uv_version)
    return UV_INSTALL_BLOCK_RE.sub(replacement, content)


def apply_revert(content: str, uv_version: str) -> str:
    original = (
        "# Install curl\n"
        "apt-get update\n"
        "apt-get install -y curl\n"
        "\n"
        "# Install uv\n"
        f"curl -LsSf https://astral.sh/uv/{uv_version}/install.sh | sh\n"
        "\n"
        "source $HOME/.local/bin/env\n"
    )
    return REVERT_RE.sub(original, content)


def show_diff(path: Path, before: str, after: str, ctx_lines: int = 3) -> None:
    """Print a minimal unified-style diff to stdout."""
    import difflib
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path.name}",
        tofile=f"b/{path.name}",
        n=ctx_lines,
    )
    sys.stdout.writelines(diff)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show changes, don't write")
    parser.add_argument("--revert", action="store_true", help="Undo patches")
    parser.add_argument("--max-show", type=int, default=3, help="Max files to show full diff for")
    args = parser.parse_args()

    if args.revert:
        targets = find_patched()
        print(f"=== REVERT mode: {len(targets)} patched test.sh found ===")
        if not targets:
            print("Nothing to revert.")
            return
        print(f"Sample task list: {[t[0] for t in targets[:5]]}...")
        for i, (name, path, content, uv_ver) in enumerate(targets):
            new_content = apply_revert(content, uv_ver)
            if i < args.max_show:
                print(f"\n--- {name} (revert) ---")
                show_diff(path, content, new_content)
            if not args.dry_run:
                path.write_text(new_content, encoding="utf-8")
        action = "DRY-RUN" if args.dry_run else "APPLIED"
        print(f"\n=== {action}: reverted {len(targets)} files ===")
        return

    targets = find_targets()
    print(f"=== PATCH mode: {len(targets)} target test.sh files ===")
    if not targets:
        print("Nothing to patch (all already patched or no matching boilerplate).")
        return
    print(f"Sample task list: {[t[0] for t in targets[:5]]}...")
    for i, (name, path, content, uv_ver) in enumerate(targets):
        new_content = apply_patch(content, uv_ver)
        if i < args.max_show:
            print(f"\n--- {name} (uv {uv_ver}) ---")
            show_diff(path, content, new_content)
        if not args.dry_run:
            path.write_text(new_content, encoding="utf-8")
    action = "DRY-RUN" if args.dry_run else "APPLIED"
    print(f"\n=== {action}: patched {len(targets)} files ===")
    if args.dry_run:
        print("Review the diff above; re-run without --dry-run to apply.")
    else:
        print("To revert: python3 patch_test_sh.py --revert")


if __name__ == "__main__":
    main()
