#!/usr/bin/env python3
"""Verify the new PersistentShell behaves like a real interactive bash.

Covers the regressions that broke SETA under the old stateless ToolLayer:
  - cd persistence
  - export persistence
  - source persistence
  - alias persistence
  - rc capture (success / failure)
  - timeout + auto-restart
  - background process tracking

Run:
    python test_persistent_shell.py
"""
import os
import shlex
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified_runner.tool_layer import ToolLayer, PersistentShell


TEST_ROOT = Path(os.environ.get("UNIFIED_TEST_TMPDIR", "/tmp")).resolve()
TEST_ROOT.mkdir(parents=True, exist_ok=True)


def _check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
    return ok


def test_cwd_persistence():
    print("\n=== cwd persistence ===")
    layer = ToolLayer(mode="host", workdir=str(TEST_ROOT))
    try:
        r1 = layer.dispatch("exec", {"command": "cd /etc"})
        r2 = layer.dispatch("exec", {"command": "pwd"})
        out = r2.get("output", "").strip()
        _check("cd /etc; pwd → /etc", out.endswith("/etc"), f"got {out!r}")
    finally:
        layer.close()


def test_env_persistence():
    print("\n=== env persistence ===")
    layer = ToolLayer(mode="host", workdir=str(TEST_ROOT))
    try:
        layer.dispatch("exec", {"command": "export FOO=hello_world_42"})
        r = layer.dispatch("exec", {"command": "echo $FOO"})
        out = r.get("output", "").strip()
        _check("export FOO=...; echo $FOO", out == "hello_world_42", f"got {out!r}")
    finally:
        layer.close()


def test_source_persistence():
    print("\n=== source persistence ===")
    layer = ToolLayer(mode="host", workdir=str(TEST_ROOT))
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, dir=TEST_ROOT) as f:
            f.write("export SOURCED_VAR=found_via_source\n")
            script = f.name
        try:
            layer.dispatch("exec", {"command": f"source {script}"})
            r = layer.dispatch("exec", {"command": "echo $SOURCED_VAR"})
            out = r.get("output", "").strip()
            _check("source script; echo var", out == "found_via_source", f"got {out!r}")
        finally:
            os.unlink(script)
    finally:
        layer.close()


def test_alias_persistence():
    print("\n=== alias persistence ===")
    layer = ToolLayer(mode="host", workdir=str(TEST_ROOT))
    try:
        layer.dispatch("exec", {"command": "shopt -s expand_aliases && alias greet='echo hi_from_alias'"})
        r = layer.dispatch("exec", {"command": "greet"})
        out = r.get("output", "").strip()
        _check("alias greet; greet", out == "hi_from_alias", f"got {out!r}")
    finally:
        layer.close()


def test_exit_codes():
    print("\n=== exit code capture ===")
    layer = ToolLayer(mode="host", workdir=str(TEST_ROOT))
    try:
        r1 = layer.dispatch("exec", {"command": "true"})
        _check("true → rc=0", r1.get("exit_code") == 0, f"got rc={r1.get('exit_code')}")
        r2 = layer.dispatch("exec", {"command": "false"})
        _check("false → rc=1", r2.get("exit_code") == 1, f"got rc={r2.get('exit_code')}")
        r3 = layer.dispatch("exec", {"command": "exit 42"})
        # `exit 42` in the persistent shell would actually kill it. Verify shell self-restarts.
        r4 = layer.dispatch("exec", {"command": "echo recovered"})
        _check("shell self-restart after exit", "recovered" in r4.get("output", ""),
               f"output={r4.get('output', '')!r}")
    finally:
        layer.close()


def test_timeout_recovery():
    print("\n=== timeout + recovery ===")
    layer = ToolLayer(mode="host", workdir=str(TEST_ROOT))
    try:
        t0 = time.time()
        r = layer.dispatch("exec", {"command": "sleep 30", "timeout": 2})
        dt = time.time() - t0
        _check("sleep 30 timeout=2 returns within ~2s", dt < 5, f"took {dt:.1f}s")
        _check("rc == -1 on timeout", r.get("exit_code") == -1, f"rc={r.get('exit_code')}")
        # next exec should still work
        r2 = layer.dispatch("exec", {"command": "echo alive"})
        _check("shell alive after timeout kill", "alive" in r2.get("output", ""),
               f"output={r2.get('output', '')!r}")
    finally:
        layer.close()


def test_multiline_output():
    print("\n=== multiline output ===")
    layer = ToolLayer(mode="host", workdir=str(TEST_ROOT))
    try:
        r = layer.dispatch("exec", {"command": "printf 'a\\nb\\nc\\n'"})
        out = r.get("output", "").strip()
        _check("multiline preserved", out == "a\nb\nc", f"got {out!r}")
    finally:
        layer.close()


def test_state_chain():
    print("\n=== chained state (the SETA scenario) ===")
    layer = ToolLayer(mode="host", workdir=str(TEST_ROOT))
    try:
        state_dir = TEST_ROOT / "persist_test"
        quoted = shlex.quote(str(state_dir))
        layer.dispatch("exec", {"command": f"mkdir -p {quoted} && cd {quoted}"})
        layer.dispatch("exec", {"command": "export PROJECT=demo"})
        layer.dispatch("exec", {"command": "echo hello > $PROJECT.txt"})
        r = layer.dispatch("exec", {"command": "cat demo.txt"})
        out = r.get("output", "").strip()
        ok = out == "hello"
        _check("chained cd + export + write + cat", ok, f"got {out!r}")
    finally:
        layer.dispatch("exec", {"command": f"rm -rf {shlex.quote(str(TEST_ROOT / 'persist_test'))}"})
        layer.close()


def test_background():
    print("\n=== background process tracking ===")
    layer = ToolLayer(mode="host", workdir=str(TEST_ROOT))
    try:
        r = layer.dispatch("exec", {"command": "sleep 1; echo done", "background": True})
        pid = r.get("pid")
        _check("background returns pid", pid is not None, str(r))
        time.sleep(2)
        r2 = layer.dispatch("process", {"action": "list"})
        procs = r2.get("processes", [])
        _check("process list has the bg pid", any(p["pid"] == pid for p in procs), str(procs))
    finally:
        layer.close()


def main():
    tests = [
        test_cwd_persistence,
        test_env_persistence,
        test_source_persistence,
        test_alias_persistence,
        test_exit_codes,
        test_timeout_recovery,
        test_multiline_output,
        test_state_chain,
        test_background,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:
            print(f"  [ERROR] {t.__name__} crashed: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\nDone. ({failed} test(s) raised)")


if __name__ == "__main__":
    main()
