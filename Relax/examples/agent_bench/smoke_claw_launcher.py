# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Unit smoke for the v1.1 real Claw launcher.

Exercises only :class:`ClawLauncher` (no Relax rollout, no env_agent_bench)
so we can verify docker lifecycle + scoring round-trip without burning a
GPU. Uses a fake assistant turn that calls one Claw HTTP tool, so
``score_from_components`` has at least one dispatch to count.

Run::

    DOCKER_HOST=unix:///tmp/local-docker-overlay2.sock \\
    UNIFIED_CLAW_USE_DOCKER_SANDBOX=1 \\
        python -m examples.agent_bench.smoke_claw_launcher \\
            --task-id T002_email_triage
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _fake_messages_with_curl(tool_endpoint_url: str) -> list[dict]:
    """Build a 2-turn trajectory that exec's a curl against ``tool_endpoint_url``.

    ``extract_tool_dispatches`` scans assistant `tool_calls` whose `function.name`
    is ``exec`` or ``process`` and then regexes the ``command`` arg for curl URLs.
    """
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tc_0",
                    "type": "function",
                    "function": {
                        "name": "exec",
                        "arguments": json.dumps(
                            {"command": f"curl -s {tool_endpoint_url} && echo done"}
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "name": "exec", "tool_call_id": "tc_0", "content": "done"},
        {"role": "assistant", "content": "Done. The task is finished."},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task-id", default="T002_email_triage")
    args = parser.parse_args()

    os.environ.setdefault("DOCKER_HOST", "unix:///tmp/local-docker-overlay2.sock")
    os.environ.setdefault("UNIFIED_CLAW_USE_DOCKER_SANDBOX", "1")
    # Make sure we resolve to the *real* ClawLauncher even if some upstream
    # env var has flipped us into mock mode.
    os.environ["UNIFIED_LAUNCHER_MODE"] = "real"

    from examples.agent_bench.launchers.claw_launcher import ClawLauncher  # type: ignore

    launcher = ClawLauncher(args.task_id, task_kwargs={"bench": "claw", "task_id": args.task_id})
    print(f"[smoke] starting launcher for {args.task_id}...")
    t0 = time.time()
    try:
        container = launcher.start()
    except Exception as exc:
        print(f"[FAIL] launcher.start raised: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"[smoke] start ok ({time.time()-t0:.1f}s) container={container}")

    # Find the first HTTP tool endpoint to dispatch against, so the grader
    # sees a non-empty dispatch set.
    task_def = launcher._task_def or {}
    endpoints = task_def.get("tool_endpoints") or []
    if not endpoints:
        print("[FAIL] task has no tool_endpoints; cannot exercise grader", file=sys.stderr)
        launcher.teardown()
        return 1
    first_url = endpoints[0]["url"]
    print(f"[smoke] faking a curl dispatch against {first_url}")
    messages = _fake_messages_with_curl(first_url)

    try:
        score = launcher.grade(final_answer=messages[-1]["content"], messages=messages)
    except Exception as exc:
        print(f"[FAIL] launcher.grade raised: {type(exc).__name__}: {exc}", file=sys.stderr)
        launcher.teardown()
        return 1
    print(f"[smoke] grade ok score={score:.3f}")
    if not (0.0 <= score <= 1.0):
        print(f"[FAIL] score out of [0,1]: {score!r}", file=sys.stderr)
        launcher.teardown()
        return 1

    print("[smoke] tearing down...")
    launcher.teardown()
    print(f"[PASS] Claw real launcher works ({time.time()-t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
