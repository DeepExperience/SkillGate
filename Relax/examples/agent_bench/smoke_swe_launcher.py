# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Unit smoke for SWELauncher (SWE-Gym lite).

Run::

    DOCKER_HOST=unix:///tmp/local-docker-overlay2.sock python -m examples.agent_bench.smoke_swe_launcher \\
        --task-id getmoto__moto-5752
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task-id", default="getmoto__moto-5752")
    args = parser.parse_args()

    os.environ.setdefault("DOCKER_HOST", "unix:///tmp/local-docker-overlay2.sock")
    os.environ["UNIFIED_LAUNCHER_MODE"] = "real"

    from examples.agent_bench.launchers.swe_launcher import SWEGymLauncher  # type: ignore

    launcher = SWEGymLauncher(args.task_id, task_kwargs={"bench": "swe_lite"})
    print(f"[smoke] start launcher: task={args.task_id}")
    t0 = time.time()
    try:
        cname = launcher.start()
    except Exception as exc:
        print(f"[FAIL] start raised: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"[smoke] start ok ({time.time()-t0:.1f}s) container={cname}")

    try:
        score = launcher.grade(container_state=True, messages=[])
    except Exception as exc:
        print(f"[FAIL] grade raised: {type(exc).__name__}: {exc}", file=sys.stderr)
        launcher.teardown()
        return 1
    print(f"[smoke] grade ok score={score:.3f}")

    print("[smoke] tearing down...")
    launcher.teardown()
    print(f"[PASS] SWELauncher works ({time.time()-t0:.1f}s total, score={score:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
