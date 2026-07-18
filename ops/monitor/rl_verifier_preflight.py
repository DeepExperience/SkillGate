#!/usr/bin/env python3
"""Fast verifier preflight for Relax RL agent benches.

Runs launcher start + verifier/grade only; no agent rollout. Intended to
separate image/start failures from verifier dependency/computation timeouts.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELAX_ROOT = PROJECT_ROOT / "Relax"
EVAL_SCRIPTS = PROJECT_ROOT / "GeneralAgent" / "eval_scripts"
for p in (RELAX_ROOT, EVAL_SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from examples.agent_bench.launchers.claw_launcher import ClawLauncher  # noqa: E402
from examples.agent_bench.launchers.harbor_launcher import HarborLauncher  # noqa: E402
from examples.agent_bench.launchers.swe_launcher import SWEGymLauncher  # noqa: E402
from examples.agent_bench.launchers.base import LauncherError, looks_like_infra_failure  # noqa: E402

LOG = logging.getLogger("rl_verifier_preflight")

DEFAULT_TASKS = [
    ("claw", "T001zh_email_triage", "clean_claw_smoke"),
    ("sb_ns", "manufacturing-fjsp-optimization", "current_timeout"),
    ("sb_ns", "exoplanet-detection-period", "current_timeout"),
    ("sb_ns", "financial-modeling-qa", "quarantined_verifier_timeout"),
    ("sb_ns", "latex-formula-extraction", "quarantined_prebuild_timeout"),
    ("seta_synth", "0", "seta_removed_probe"),
    ("seta_synth", "817", "historical_verifier_timeout"),
    ("tb2", "install-windows-3.11", "current_clean_probe"),
    ("tb2", "extract-elf", "quarantined_verifier_timeout"),
    ("swe_lite", "django__django-11049", "swe_probe"),
]


def env_defaults(timeout_cap: int, require_prebuilt: bool) -> None:
    os.environ.setdefault("DOCKER_HOST", "unix:///tmp/local-docker-overlay2.sock")
    os.environ.setdefault("UNIFIED_LAUNCHER_MODE", "real")
    os.environ.setdefault("UNIFIED_CLAW_USE_DOCKER_SANDBOX", "1")
    os.environ.setdefault("UNIFIED_CLAW_SANDBOX_FAIL_HARD", "1")
    os.environ.setdefault("UNIFIED_HARBOR_BUILD_TIMEOUT_SEC", "300")
    os.environ.setdefault("AGENT_BENCH_SETUP_ATTEMPTS", "1")
    os.environ.setdefault("AGENT_BENCH_DOCKER_START_CONCURRENCY", "1")
    os.environ.setdefault("UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS", "1")
    os.environ.setdefault("UNIFIED_SWE_VERIFIER_TIMEOUT_SEC", str(timeout_cap))
    os.environ["UNIFIED_VERIFIER_TIMEOUT_CAP_SEC"] = str(timeout_cap)
    if require_prebuilt:
        os.environ["UNIFIED_HARBOR_REQUIRE_PREBUILT_LOCAL"] = "1"
    os.environ.setdefault("HTTP_PROXY", "http://your-proxy:3128")
    os.environ.setdefault("HTTPS_PROXY", "http://your-proxy:3128")
    os.environ.setdefault("http_proxy", "http://your-proxy:3128")
    os.environ.setdefault("https_proxy", "http://your-proxy:3128")
    os.environ.setdefault(
        "NO_PROXY",
        "127.0.0.1,localhost,0.0.0.0,10.0.0.0/8,172.16.0.0/12,"
        "mirrors.tuna.tsinghua.edu.cn,pypi.tuna.tsinghua.edu.cn,hf-mirror.com",
    )
    os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])


def short_tail(text: str | None, limit: int = 1600) -> str:
    if not text:
        return ""
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", str(text))
    return text[-limit:]


def classify(bench: str, task_id: str, phase: str, ok: bool, error: str, output: str, timings: dict[str, float]) -> str:
    blob = f"{error}\n{output}".lower()
    if ok:
        return "ok"
    if "local build image" in blob and "missing" in blob:
        return "missing_or_unbuilt_image"
    if "copy" in blob and "no such file" in blob or "absent from build context" in blob:
        return "hard_dockerfile_or_build_context_bug"
    if "build" in blob and "timed out" in blob:
        return "heavy_build_timeout_not_proven_broken"
    if "verifier timeout" in blob or "command timed out" in blob:
        dep_markers = [
            "pip install", "uv ", "uv_http_timeout", "download", "pypi", "pytest", "pygments", "numpy", "pandas",
            "installing", "installed", "resolving", "downloading",
        ]
        if any(m in blob for m in dep_markers):
            return "verifier_dependency_install_timeout"
        return "verifier_computation_timeout"
    if "session open refused" in blob or "connection reset" in blob or "cannot connect to the docker daemon" in blob:
        return "docker_daemon_or_concurrency_contention"
    if "traceback" in blob or "keyerror" in blob or "attributeerror" in blob:
        return "runner_or_launcher_bug"
    if phase in {"start", "copy_tests"}:
        return "missing_or_unbuilt_image" if "image" in blob else "unknown_needs_manual_debug"
    return "unknown_needs_manual_debug"


def preflight_harbor(bench: str, task_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {"bench": bench, "task_id": task_id, "kind": "harbor"}
    launcher = HarborLauncher(task_id, task_kwargs={"bench": bench})
    cname = None
    t0 = time.time()
    try:
        t = time.time(); cname = launcher.start(); result["start_sec"] = round(time.time() - t, 3); result["container"] = cname
        t = time.time(); score = launcher.grade(container_state=True, messages=[]); result["verifier_sec"] = round(time.time() - t, 3); result["score"] = score
        result["ok"] = True; result["phase"] = "grade"; result["error"] = ""
    except Exception as exc:
        result["ok"] = False; result["phase"] = "start" if cname is None else "grade"; result["error"] = f"{type(exc).__name__}: {exc}"
        result["trace_tail"] = short_tail(traceback.format_exc(), 1200)
    finally:
        t = time.time()
        try:
            launcher.teardown()
            result["teardown_sec"] = round(time.time() - t, 3)
        except Exception as exc:
            result["teardown_error"] = f"{type(exc).__name__}: {exc}"
    result["total_sec"] = round(time.time() - t0, 3)
    result["class"] = classify(bench, task_id, result.get("phase", ""), bool(result.get("ok")), result.get("error", ""), result.get("trace_tail", ""), result)
    return result


def preflight_claw(bench: str, task_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {"bench": bench, "task_id": task_id, "kind": "claw"}
    launcher = ClawLauncher(task_id, task_kwargs={"bench": "claw"})
    cname = None; t0=time.time()
    try:
        t=time.time(); cname=launcher.start(); result["start_sec"] = round(time.time()-t, 3); result["container"] = cname
        t=time.time(); score=launcher.grade(messages=[]); result["verifier_sec"] = round(time.time()-t, 3); result["score"] = score
        result["ok"] = True; result["phase"] = "grade"; result["error"] = ""
    except Exception as exc:
        result["ok"] = False; result["phase"] = "start" if cname is None else "grade"; result["error"] = f"{type(exc).__name__}: {exc}"; result["trace_tail"] = short_tail(traceback.format_exc(), 1200)
    finally:
        t=time.time()
        try: launcher.teardown(); result["teardown_sec"] = round(time.time()-t, 3)
        except Exception as exc: result["teardown_error"] = f"{type(exc).__name__}: {exc}"
    result["total_sec"] = round(time.time()-t0, 3)
    result["class"] = classify(bench, task_id, result.get("phase", ""), bool(result.get("ok")), result.get("error", ""), result.get("trace_tail", ""), result)
    return result


def preflight_swe(bench: str, task_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {"bench": bench, "task_id": task_id, "kind": "swe"}
    launcher = SWEGymLauncher(task_id, task_kwargs={"bench": "swe_lite"})
    cname = None; t0=time.time()
    try:
        t=time.time(); cname=launcher.start(); result["start_sec"] = round(time.time()-t, 3); result["container"] = cname; result["repo_path"] = launcher._repo_path
        rus = launcher._rus
        inst = launcher._instance
        t=time.time(); ok, out = rus.apply_gold_test_patch(cname, launcher._repo_path, inst.get("test_patch", "") or ""); result["apply_test_patch_sec"] = round(time.time()-t, 3); result["apply_test_patch_ok"] = bool(ok); result["apply_test_patch_tail"] = short_tail(out, 800)
        if not ok:
            result["ok"] = False; result["phase"] = "apply_gold_test_patch"; result["error"] = "apply_gold_test_patch failed"
        else:
            t=time.time(); out = rus.run_tests(cname, launcher._repo_path, inst.get("FAIL_TO_PASS", []) or []); result["verifier_sec"] = round(time.time()-t, 3); result["verifier_tail"] = short_tail(out, 1600); result["score"] = 1.0 if rus.check_test_pass(out) else 0.0
            if looks_like_infra_failure(out):
                result["ok"] = False; result["phase"] = "run_tests"; result["error"] = short_tail(out, 1000)
            else:
                result["ok"] = True; result["phase"] = "run_tests"; result["error"] = ""
    except Exception as exc:
        result["ok"] = False; result["phase"] = "start" if cname is None else result.get("phase", "run_tests"); result["error"] = f"{type(exc).__name__}: {exc}"; result["trace_tail"] = short_tail(traceback.format_exc(), 1200)
    finally:
        t=time.time()
        try: launcher.teardown(); result["teardown_sec"] = round(time.time()-t, 3)
        except Exception as exc: result["teardown_error"] = f"{type(exc).__name__}: {exc}"
    result["total_sec"] = round(time.time()-t0, 3)
    result["class"] = classify(bench, task_id, result.get("phase", ""), bool(result.get("ok")), result.get("error", ""), result.get("trace_tail", "") + result.get("verifier_tail", "") + result.get("apply_test_patch_tail", ""), result)
    return result


def run_one(item: tuple[str, str, str]) -> dict[str, Any]:
    bench, task_id, label = item
    t0=time.time()
    try:
        if bench in {"sb_ns", "tb2", "seta_synth", "seta"}:
            r = preflight_harbor(bench, task_id)
        elif bench == "swe_lite":
            r = preflight_swe(bench, task_id)
        elif bench == "claw":
            r = preflight_claw(bench, task_id)
        else:
            raise ValueError(f"unknown bench {bench}")
    except Exception as exc:
        r = {"bench": bench, "task_id": task_id, "ok": False, "phase": "construct", "error": f"{type(exc).__name__}: {exc}", "trace_tail": short_tail(traceback.format_exc(), 1600)}
        r["class"] = classify(bench, task_id, "construct", False, r["error"], r.get("trace_tail", ""), r)
    r["label"] = label
    r["wall_sec"] = round(time.time()-t0, 3)
    return r


def parse_task_arg(raw: str) -> tuple[str, str, str]:
    # bench/task[:label]
    label = "manual"
    if raw.count(":"):
        raw, label = raw.split(":", 1)
    bench, task = raw.split("/", 1)
    return bench, task, label


def load_task_file(path: str) -> list[tuple[str, str, str]]:
    """Load JSONL task records without relying on huge shell argv strings.

    Expected fields are `bench`, `task_id`, and optional `label`.  This keeps
    full train/eval preflights reproducible and avoids accidentally testing the
    wrong filtered parquet when the active RL resume points elsewhere.
    """
    tasks: list[tuple[str, str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            bench = row.get("bench")
            task_id = row.get("task_id")
            if not bench or task_id is None:
                raise ValueError(f"{path}:{line_no}: requires bench and task_id")
            tasks.append((str(bench), str(task_id), str(row.get("label") or "task_file")))
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", action="append", default=[], help="bench/task_id[:label]; may repeat")
    ap.add_argument("--task-file", help="JSONL records with bench/task_id[/label]; appends before --task")
    ap.add_argument("--out", default="experiments/infra/rl/preflight/verifier_preflight_latest.jsonl")
    ap.add_argument("--timeout-cap", type=int, default=180)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--no-require-prebuilt", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    env_defaults(args.timeout_cap, require_prebuilt=not args.no_require_prebuilt)
    tasks = []
    if args.task_file:
        tasks.extend(load_task_file(args.task_file))
    tasks.extend(parse_task_arg(x) for x in args.task)
    if not tasks:
        tasks = DEFAULT_TASKS
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("preflight tasks=%d jobs=%d out=%s timeout_cap=%s DOCKER_HOST=%s", len(tasks), args.jobs, out, args.timeout_cap, os.environ.get("DOCKER_HOST"))
    results=[]
    with out.open("w") as f:
        if args.jobs <= 1:
            for item in tasks:
                r=run_one(item); results.append(r); f.write(json.dumps(r, ensure_ascii=False)+"\n"); f.flush(); print(json.dumps(r, ensure_ascii=False))
        else:
            with ThreadPoolExecutor(max_workers=args.jobs) as ex:
                futs={ex.submit(run_one,item):item for item in tasks}
                for fut in as_completed(futs):
                    r=fut.result(); results.append(r); f.write(json.dumps(r, ensure_ascii=False)+"\n"); f.flush(); print(json.dumps(r, ensure_ascii=False))
    ok=sum(1 for r in results if r.get("ok"))
    classes={}
    for r in results: classes[r.get("class","unknown")]=classes.get(r.get("class","unknown"),0)+1
    LOG.info("summary ok=%d/%d classes=%s", ok, len(results), classes)
    return 0 if ok == len(results) else 2

if __name__ == "__main__":
    raise SystemExit(main())
