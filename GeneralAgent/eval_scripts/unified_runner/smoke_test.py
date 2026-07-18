#!/usr/bin/env python3
"""Smoke test for the unified tool layer.

Tests host mode only (no Docker required). Verifies:
  1. tool_schemas imports and defaults to the OpenClaw-full profile
  2. tool_layer host mode: read, write, edit, exec, find, grep, ls, apply_patch, process
  3. base.py TaskResult and RunConfig
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add parent to path so we can import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unified_runner.openclaw_compat import build_openclaw_system_prompt
from unified_runner.prompt_profiles import PROMPT_PROFILE_ENV
from unified_runner.openclaw_probe_full_tools import OPENCLAW_FULL_28_TOOL_NAMES
from unified_runner.tool_schemas import CORE_TOOLS, get_tools, TOOL_REGISTRY
from unified_runner.tool_layer import ToolLayer
from unified_runner.base import TaskResult, RunConfig, AbstractDatasetRunner


def test_schemas():
    print("--- test_schemas ---")
    tools = get_tools()
    assert len(tools) == 28, f"Expected 28 OpenClaw-full tools, got {len(tools)}"
    names = [t["function"]["name"] for t in tools]
    assert names == OPENCLAW_FULL_28_TOOL_NAMES, (
        "OpenClaw-full tool order drift: "
        f"expected={OPENCLAW_FULL_28_TOOL_NAMES}, got={names}"
    )
    assert names.index("web_fetch") < names.index("web_search")
    prompt = build_openclaw_system_prompt(workspace_dir="/workspace")
    assert prompt.index("- web_search:") < prompt.index("- web_fetch:")
    print(f"  OK: {len(tools)} OpenClaw-full tools defined in provider order")

    core_names = {t["function"]["name"] for t in CORE_TOOLS}
    assert core_names == {"read", "write", "edit", "exec", "process", "web_fetch", "web_search"}
    print("  OK: CORE_TOOLS remains the 7-tool implemented subset")
    for legacy_tool in ("apply_patch", "grep", "find", "ls"):
        assert legacy_tool in TOOL_REGISTRY
    print("  OK: legacy convenience tools remain available for migration/tests")

    subset = get_tools(include=["read", "exec"])
    assert len(subset) == 2
    print(f"  OK: get_tools(include=['read','exec']) → {len(subset)} tools")

    subset2 = get_tools(exclude=["process"])
    assert len(subset2) == 27
    print(f"  OK: get_tools(exclude=['process']) → {len(subset2)} tools")

    old_profile = os.environ.get(PROMPT_PROFILE_ENV)
    try:
        os.environ[PROMPT_PROFILE_ENV] = "legacy_11"
        legacy = get_tools()
        legacy_names = {t["function"]["name"] for t in legacy}
        assert len(legacy) == 11, f"Expected 11 legacy tools, got {len(legacy)}"
        assert {"apply_patch", "grep", "find", "ls"} <= legacy_names
        legacy_prompt = build_openclaw_system_prompt(workspace_dir="/workspace")
        assert "inside OpenClaw" not in legacy_prompt
        assert "Your shell tools:" in legacy_prompt
        print("  OK: UNIFIED_PROMPT_PROFILE=legacy_11 → 11 tools + legacy prompt")
    finally:
        if old_profile is None:
            os.environ.pop(PROMPT_PROFILE_ENV, None)
        else:
            os.environ[PROMPT_PROFILE_ENV] = old_profile

    # Verify each schema has valid OpenAI function calling structure
    for tool in tools:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"
    print("  OK: all schemas have valid OpenAI function calling structure")


def test_tool_layer_host():
    print("\n--- test_tool_layer (host mode) ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        layer = ToolLayer(mode="host", workdir=tmpdir)

        # write
        test_file = os.path.join(tmpdir, "hello.txt")
        r = layer.dispatch("write", {"path": test_file, "content": "line1\nline2\nline3\n"})
        assert "error" not in r, f"write failed: {r}"
        assert r["bytes"] == 18
        print(f"  OK: write → {r}")

        # read
        r = layer.dispatch("read", {"path": test_file})
        assert "error" not in r, f"read failed: {r}"
        assert "line1" in r["content"]
        assert r["total_lines"] == 3
        print(f"  OK: read → total_lines={r['total_lines']}")

        # read with offset/limit
        r = layer.dispatch("read", {"path": test_file, "offset": 2, "limit": 1})
        assert "line2" in r["content"]
        assert "line1" not in r["content"]
        print(f"  OK: read(offset=2,limit=1) → {r['content'].strip()}")

        # edit
        r = layer.dispatch("edit", {
            "path": test_file,
            "edits": [{"oldText": "line2", "newText": "LINE_TWO"}],
        })
        assert r.get("replacements") == 1
        print(f"  OK: edit → {r}")

        # verify edit
        r = layer.dispatch("read", {"path": test_file})
        assert "LINE_TWO" in r["content"]
        print(f"  OK: read after edit confirms replacement")

        # exec
        r = layer.dispatch("exec", {"command": "echo hello_world"})
        assert r["exit_code"] == 0
        assert "hello_world" in r["output"]
        print(f"  OK: exec → exit_code={r['exit_code']}")

        # ls
        r = layer.dispatch("ls", {"path": tmpdir})
        assert "hello.txt" in r.get("output", "")
        print(f"  OK: ls → found hello.txt")

        # find
        r = layer.dispatch("find", {"pattern": "*.txt", "path": tmpdir})
        assert any("hello.txt" in f for f in r.get("files", []))
        print(f"  OK: find → {len(r.get('files', []))} files")

        # grep
        r = layer.dispatch("grep", {"pattern": "LINE_TWO", "path": test_file})
        assert r["exit_code"] == 0
        print(f"  OK: grep → found match")

        # apply_patch
        patch_content = """*** Begin Patch
*** Update File: hello.txt
@@
 line1
-LINE_TWO
+PATCHED_LINE
 line3
*** End Patch
"""
        r = layer.dispatch("apply_patch", {"input": patch_content})
        assert r.get("exit_code") == 0, f"apply_patch failed: {r}"
        print(f"  OK: apply_patch → exit_code={r.get('exit_code')}")

        # process (list)
        r = layer.dispatch("process", {"action": "list"})
        assert "processes" in r
        print(f"  OK: process(list) → {len(r['processes'])} tracked")


def test_base_classes():
    print("\n--- test_base_classes ---")
    cfg = RunConfig(model="test-model", api_base="http://localhost:1234/v1")
    assert cfg.model == "test-model"
    print(f"  OK: RunConfig(model={cfg.model})")

    tr = TaskResult(task_id="t1", dataset="test", resolved=True, score=0.85, turns=5, time_sec=30)
    d = tr.to_dict()
    assert d["resolved"] is True
    assert d["score"] == 0.85
    assert "trajectory" not in d
    print(f"  OK: TaskResult.to_dict() → {json.dumps(d, indent=None)[:100]}...")

    print(f"  OK: AbstractDatasetRunner is importable")


if __name__ == "__main__":
    test_schemas()
    test_tool_layer_host()
    test_base_classes()
    print("\n✅ All smoke tests passed!")
