#!/usr/bin/env python3
"""Smoke test for _trim_messages — regression guards for 2026-04-19 fix.

Tests that the trim logic:
  1. No-ops when message count is small
  2. Aggressively trims old tool results to `middle_cap` chars
  3. Keeps system + first-user + last-N intact
  4. Respects env-var overrides (UNIFIED_TRIM_KEEP_LAST, UNIFIED_TRIM_MIDDLE_CHARS)
  5. Actually reduces total chars by >= 50% on realistic inputs

Run:
    python3 GeneralAgent/eval_scripts/unified_runner/test_trim_smoke.py
Returns exit 0 on success, 1 on any failure.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified_runner.agent_loop import UnifiedAgentLoop


def _make_conversation(n_turns: int, tool_result_size: int = 16000) -> list[dict]:
    """Build a realistic messages list: sys + user + N turns of (assistant + tool)."""
    msgs = [
        {"role": "system", "content": "You are a skilled agent. Tools: read, exec..." * 20},
        {"role": "user", "content": "Task prompt here. " * 200},
    ]
    for i in range(n_turns):
        msgs.append({
            "role": "assistant",
            "content": f"Thinking about turn {i}... " * 40,
            "tool_calls": [{
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": "exec", "arguments": json.dumps({"command": f"ls /x{i}"})},
            }],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "name": "exec",
            "content": f"<output turn {i}>\n" + ("x" * tool_result_size),
        })
    return msgs


def _total_chars(msgs: list[dict]) -> int:
    return sum(len(json.dumps(m, default=str)) for m in msgs)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)
    print(f"  OK  : {msg}")


def test_noop_when_small():
    print("Test 1: no-op when <= keep_last + 2 messages")
    msgs = _make_conversation(3)  # 2 + 2*3 = 8 messages
    before = len(msgs)
    out = UnifiedAgentLoop._trim_messages(msgs)
    _assert(len(out) == before, f"preserves all {before} messages when small")


def test_trims_middle_tools():
    print("Test 2: trims middle tool results")
    msgs = _make_conversation(20, tool_result_size=16000)  # 2 + 40 = 42 messages
    before_chars = _total_chars(msgs)
    out = UnifiedAgentLoop._trim_messages(msgs)
    after_chars = _total_chars(out)
    _assert(after_chars < before_chars * 0.5,
            f"reduces chars by >=50%: {before_chars} → {after_chars} ({100*after_chars//before_chars}%)")


def test_preserves_system_and_first_user():
    print("Test 3: system + first user message unchanged")
    msgs = _make_conversation(15)
    orig_sys = msgs[0]["content"]
    orig_user = msgs[1]["content"]
    out = UnifiedAgentLoop._trim_messages(msgs)
    _assert(out[0]["content"] == orig_sys, "system message identical")
    _assert(out[1]["content"] == orig_user, "first user message identical")


def test_keeps_last_n():
    print("Test 4: last N messages intact (default N=6)")
    msgs = _make_conversation(15)
    last_6 = msgs[-6:]
    out = UnifiedAgentLoop._trim_messages(msgs)
    for i, (expected, got) in enumerate(zip(last_6, out[-6:])):
        _assert(expected == got, f"last-6[{i}] preserved ({expected.get('role')})")


def test_env_var_override():
    print("Test 5: UNIFIED_TRIM_KEEP_LAST env var respected")
    os.environ["UNIFIED_TRIM_KEEP_LAST"] = "3"
    try:
        msgs = _make_conversation(15)
        last_3 = msgs[-3:]
        out = UnifiedAgentLoop._trim_messages(msgs)
        _assert(out[-3:] == last_3, "env KEEP_LAST=3 only keeps last 3 intact")
    finally:
        del os.environ["UNIFIED_TRIM_KEEP_LAST"]


def test_middle_cap_env_override():
    print("Test 6: UNIFIED_TRIM_MIDDLE_CHARS env var respected")
    os.environ["UNIFIED_TRIM_MIDDLE_CHARS"] = "500"
    try:
        msgs = _make_conversation(15, tool_result_size=5000)
        out = UnifiedAgentLoop._trim_messages(msgs)
        # middle tool messages should each be ≤ 500 + "[TRIMMED: ...]" suffix
        # The last 6 are kept intact; old tools are index 2..-6
        for m in out[2:-6]:
            if m.get("role") == "tool":
                # content should be <= 500 + suffix (~50 chars)
                _assert(len(m["content"]) <= 600,
                        f"old tool result capped to ~500 chars (got {len(m['content'])})")
                break  # one check is enough for this smoke
    finally:
        del os.environ["UNIFIED_TRIM_MIDDLE_CHARS"]


def test_realistic_131k_ctx_scenario():
    """The exact scenario that was triggering HTTP 400 in tb2/sb."""
    print("Test 7: realistic 50-turn × 16K chars (~simulates tb2 ctx-overflow)")
    # 50 turns × ~16K each ≈ 800K chars, was the actual failure mode
    msgs = _make_conversation(50, tool_result_size=16000)
    before = _total_chars(msgs)
    out = UnifiedAgentLoop._trim_messages(msgs)
    after = _total_chars(out)
    # After trim, total chars should be roughly: sys + user + 2*43 middle (trimmed to 1500 each) + last 6 full
    # ≈ 2000 + 400 + 86 * 1600 + 6 * 16000 = ~240K chars
    # At ~2.5 chars/token for code-heavy EN content, ≈ 96K tokens — below 131K ctx ✓
    _assert(after < 300_000, f"post-trim chars well under ctx-overflow: {before} → {after}")
    # More stringent: should leave ~100K tokens budget = 250K chars
    print(f"       before={before:,}  after={after:,}  reduction={100 - 100*after//before}%")


def main():
    print("=" * 60)
    print("_trim_messages smoke test (regression guard for 2026-04-19 fix)")
    print("=" * 60)
    test_noop_when_small()
    print()
    test_trims_middle_tools()
    print()
    test_preserves_system_and_first_user()
    print()
    test_keeps_last_n()
    print()
    test_env_var_override()
    print()
    test_middle_cap_env_override()
    print()
    test_realistic_131k_ctx_scenario()
    print()
    print("=" * 60)
    print("All 7 smoke tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
