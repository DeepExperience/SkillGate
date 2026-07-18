#!/usr/bin/env python3
"""Unified interface SWE-bench evaluation.

Uses agent_loop + ToolLayer(docker) instead of custom 3-tool agent.
The deployable OpenClaw tool subset (read, write, edit, exec, process,
web_fetch, web_search) replaces the original read_file/write_file/run_command
tools. Shell-native operations such as grep/find/ls are done through exec.
"""

import argparse
import base64
import contextlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified_runner.agent_loop import UnifiedAgentLoop
from unified_runner.tool_layer import ToolLayer
from unified_runner.base import RunConfig
from unified_runner.retrieval_skill_inject import (
    load_retrieval_mapping, build_irrelevant_mapping,
    inject_retrieval_skills, build_retrieval_prompt_hint,
    build_top1_skill_text_prompt,
)
from unified_runner.openclaw_compat import (
    append_runtime_context_to_user_prompt,
    build_openclaw_system_prompt,
    build_swe_runtime_context,
)
from unified_runner.docker_start_gate import docker_start_gate
from unified_runner.docker_lifecycle import docker_label_args, record_lifecycle_event

# Errors worth retrying (flaky infra, not agent fault).
_FLAKY_RETRY_PATTERNS = [
    r"Failed to start container: Command timed out",
    r"Pull failed",
    r"dial .*: i/o timeout",
    r"Conflict\. The container name",
]


def _is_flaky_error(err_msg: str) -> bool:
    if not err_msg:
        return False
    return any(re.search(p, err_msg) for p in _FLAKY_RETRY_PATTERNS)

BASE_DIR = Path(os.environ.get("SKILLRL_ROOT", str(Path(__file__).resolve().parents[3])))
LITE_PARQUET = BASE_DIR / "datasets/swe-gym/lite/data/train-00000-of-00001.parquet"
VERIFIED_PARQUET = BASE_DIR / "datasets/swe-bench-verified/data/data/test-00000-of-00001.parquet"
RESULTS_DIR = BASE_DIR / "experiments"
IMAGE_PREFIX = "xingyaoww/sweb.eval.x86_64."
def _docker_env() -> dict[str, str]:
    """Docker CLI env for single-node tunnel and multi-node Ray workers."""
    env = dict(os.environ)
    env["DOCKER_HOST"] = os.environ.get("DOCKER_HOST", "ssh://your-docker-host")
    return env


DOCKER_ENV = _docker_env()


def _docker_pids_limit_args() -> list[str]:
    """Hard cap per benchmark container to contain runaway fork storms."""
    limit = os.environ.get("UNIFIED_DOCKER_PIDS_LIMIT", "1024").strip()
    if not limit or limit.lower() in {"0", "none", "false", "off"}:
        return []
    return ["--pids-limit", limit]


def _docker_ulimit_fsize_args() -> list[str]:
    """Kernel-enforced single-file size cap inside agent containers.

    Closes the disk-bomb -> kubelet ephemeral-storage eviction path (rl_log
    2026-06-10 20:25): an unbounded single-file writer (dd/fallocate/...) fills
    the node disk and the WHOLE pod gets evicted at ~246GiB free. RLIMIT_FSIZE
    kills only the offending process with SIGXFSZ -- the agent loop just sees
    one failed command (exit 153) and continues. Declared per-task storage tops
    out at 20GB and empirical writable layers at ~6GB, so 32G never touches
    legit tasks. No-op when unset so eval paths are unchanged.
    """
    gb = os.environ.get("UNIFIED_DOCKER_ULIMIT_FSIZE_GB", "").strip()
    if not gb or gb.lower() in {"0", "none", "false", "off"}:
        return []
    return ["--ulimit", f"fsize={int(float(gb) * 1024**3)}"]


def _docker_resource_args() -> list[str]:
    """CPU isolation so an agent-side fork/compile storm cannot starve the trainer
    co-located on the rollout node. UNIFIED_DOCKER_CPUSET pins each container to a
    CPU subset (reserving the complement for the trainer's NCCL threads);
    UNIFIED_DOCKER_BUILD_JOBS caps build parallelism. Both no-op when unset."""
    args: list[str] = []
    cpuset = os.environ.get("UNIFIED_DOCKER_CPUSET", "").strip()
    if cpuset and cpuset.lower() not in {"0", "none", "false", "off"}:
        args += ["--cpuset-cpus", cpuset]
    jobs = os.environ.get("UNIFIED_DOCKER_BUILD_JOBS", "").strip()
    if jobs.isdigit() and int(jobs) > 0:
        args += [
            "-e", f"MAKEFLAGS=-j{jobs}",
            "-e", f"MAX_JOBS={jobs}",
            "-e", f"CMAKE_BUILD_PARALLEL_LEVEL={jobs}",
            "-e", f"NPY_NUM_BUILD_JOBS={jobs}",
        ]
    return args

ALL_IMAGES = [
    "xingyaoww/sweb.eval.x86_64.dask_s_dask-6626",
    "xingyaoww/sweb.eval.x86_64.facebookresearch_s_hydra-2189",
    "xingyaoww/sweb.eval.x86_64.getmoto_s_moto-5752",
    "xingyaoww/sweb.eval.x86_64.getmoto_s_moto-6178",
    "xingyaoww/sweb.eval.x86_64.iterative_s_dvc-4785",
    "xingyaoww/sweb.eval.x86_64.iterative_s_dvc-5822",
    "xingyaoww/sweb.eval.x86_64.mwaskom_s_seaborn-3010",
    "xingyaoww/sweb.eval.x86_64.mwaskom_s_seaborn-3069",
    "xingyaoww/sweb.eval.x86_64.mwaskom_s_seaborn-3187",
    "xingyaoww/sweb.eval.x86_64.pallets_s_flask-5014",
    "xingyaoww/sweb.eval.x86_64.project-monai_s_monai-4688",
    "xingyaoww/sweb.eval.x86_64.psf_s_requests-1142",
    "xingyaoww/sweb.eval.x86_64.psf_s_requests-1724",
    "xingyaoww/sweb.eval.x86_64.pydantic_s_pydantic-8500",
    "xingyaoww/sweb.eval.x86_64.pydata_s_xarray-2905",
    "xingyaoww/sweb.eval.x86_64.pylint-dev_s_pylint-4551",
    "xingyaoww/sweb.eval.x86_64.pylint-dev_s_pylint-4604",
    "xingyaoww/sweb.eval.x86_64.pytest-dev_s_pytest-10051",
    "xingyaoww/sweb.eval.x86_64.pytest-dev_s_pytest-10081",
    "xingyaoww/sweb.eval.x86_64.python_s_mypy-10392",
    "xingyaoww/sweb.eval.x86_64.python_s_mypy-15184",
]

def image_to_instance_id(image):
    name = image.replace(IMAGE_PREFIX, "")
    if name.endswith(":latest"):
        name = name[:-7]
    return name.replace("_s_", "__")


def instance_id_to_image(instance_id):
    name = instance_id.replace("__", "_s_").lower()
    return IMAGE_PREFIX + name + ":latest"


def parse_test_list(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return [val]
    if isinstance(val, list):
        return val
    return []


def load_instances():
    instances = {}
    for parquet_path, dataset_name in [
        (LITE_PARQUET, "swe-gym-lite"),
        (VERIFIED_PARQUET, "swe-bench-verified"),
    ]:
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            iid = row["instance_id"]
            test_patch = row.get("test_patch", "")
            if not isinstance(test_patch, str):
                test_patch = "" if pd.isna(test_patch) else str(test_patch)
            instances[iid] = {
                "instance_id": iid,
                "dataset": dataset_name,
                "problem_statement": row["problem_statement"],
                "repo": row["repo"],
                "base_commit": row["base_commit"],
                "patch": row["patch"],
                "test_patch": test_patch,
                "FAIL_TO_PASS": parse_test_list(row["FAIL_TO_PASS"]),
                "PASS_TO_PASS": parse_test_list(row["PASS_TO_PASS"]),
                "difficulty": row.get("difficulty", "unknown"),
            }
    return instances


def docker_run(cmd, timeout=60):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=_docker_env())
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", -1


def _remove_container_if_exists(cname: str, *, timeout: int = 30) -> bool:
    stdout, stderr, rc = docker_run(["docker", "rm", "-f", cname], timeout=timeout)
    text = f"{stdout}\n{stderr}".lower()
    if rc == 0 or "no such container" in text or "no such object" in text:
        return True
    record_lifecycle_event("rm_failed", container=cname, stderr=stderr[:1000], rc=rc)
    return False


def _remove_containers_by_name_prefix(prefix: str, *, max_remove: int = 8) -> int:
    stdout, stderr, rc = docker_run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name={prefix}",
            "--format",
            "{{.Names}}",
        ],
        timeout=20,
    )
    if rc != 0:
        record_lifecycle_event("prefix_list_failed", container=prefix, stderr=stderr[:500], rc=rc)
        return 0
    removed = 0
    for name in stdout.splitlines():
        name = name.strip()
        if not name.startswith(prefix):
            continue
        if _remove_container_if_exists(name, timeout=20):
            removed += 1
        if removed >= max_remove:
            break
    return removed


def container_exec(container, command, timeout=60):
    stdout, stderr, rc = docker_run(
        ["docker", "exec", container, "bash", "-c", command], timeout=timeout)
    output = stdout
    if stderr:
        output += "\n[STDERR]\n" + stderr
    return output


def start_container(instance_id, image, container_suffix=None):
    # 2026-04-26: include PID to make name per-trial unique (SFT collection
    # runs multiple trials of same instance concurrently in separate procs).
    suffix = container_suffix or f"p{os.getpid()}"
    cname = "swe-unified-" + instance_id.replace("/", "_") + f"-{suffix}"
    _remove_container_if_exists(cname, timeout=30)
    time.sleep(1)
    # Inject the Docker host proxy via --add-host + env (see run_unified_harbor)
    # NO_PROXY includes CN mirrors so apt/pip hit them directly (no clash tolls).
    # Keep in sync with run_unified_harbor.CONTAINER_PROXY_NO + compose-proxy-override.yaml
    proxy = os.environ.get("UNIFIED_CONTAINER_PROXY", "http://your-docker-gateway:8888").strip()
    no_proxy = (
        "localhost,127.0.0.1,0.0.0.0,::1,"
        "mirrors.aliyun.com,mirrors.cloud.aliyuncs.com,"
        "pypi.tuna.tsinghua.edu.cn,mirrors.tuna.tsinghua.edu.cn,"
        "mirrors.ustc.edu.cn,mirror.nju.edu.cn,mirrors.bfsu.edu.cn,"
        "mirrors.163.com,mirrors.huaweicloud.com,mirrors.cernet.edu.cn,"
        "hf-mirror.com,"
        ".aliyun.com,.tsinghua.edu.cn,.ustc.edu.cn,.nju.edu.cn,"
        ".bfsu.edu.cn,.cernet.edu.cn,.163.com,.huaweicloud.com,.hf-mirror.com"
    )
    # unregister_netdevice root fix: host network namespace -> no per-container
    # netns/veth to leak (kernel-5.15 rtnl_lock wedge on teardown). SWE containers
    # only need outbound net (proxy on the docker0 gateway reachable via host routing).
    # `--add-host ...:host-gateway` is bridge-only, dropped under host net.
    # Default ON; UNIFIED_DOCKER_NETWORK_HOST=0 reverts to bridge. rl_log 2026-06-09.
    try:
        from unified_runner.run_unified_harbor import _docker_network_host_enabled
        _use_host_net = _docker_network_host_enabled()
    except Exception:
        _use_host_net = os.environ.get("UNIFIED_DOCKER_NETWORK_HOST", "1").strip().lower() not in {"0", "none", "false", "off", ""}
    _net_args = ["--network", "host"] if _use_host_net else []
    _addhost_args = [] if _use_host_net else ["--add-host", "host.docker.internal:host-gateway"]
    with docker_start_gate(f"swe:{instance_id}"):
        stdout, stderr, rc = docker_run(
            [
                "docker", "run", "-d", "--name", cname,
                *docker_label_args(
                    bench="swe_lite",
                    dataset_tag="swe_lite",
                    task_name=instance_id,
                    container_name=cname,
                    container_suffix=suffix,
                ),
                *_docker_pids_limit_args(),
                *_docker_ulimit_fsize_args(),
                *_docker_resource_args(),
                *_net_args,
                *_addhost_args,
                "-e", f"HTTP_PROXY={proxy}", "-e", f"HTTPS_PROXY={proxy}",
                "-e", f"http_proxy={proxy}", "-e", f"https_proxy={proxy}",
                "-e", f"NO_PROXY={no_proxy}", "-e", f"no_proxy={no_proxy}",
                "-e", "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple",
                "-e", "UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple",
                "-e", "UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple",
                "-e", "HF_ENDPOINT=https://hf-mirror.com",
                "-e", "HF_HUB_ENDPOINT=https://hf-mirror.com",
                "-e", "HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com",
                image, "sleep", "infinity",
            ], timeout=600)
    if rc != 0:
        record_lifecycle_event("start_failed", container=cname, stderr=stderr[:1000], rc=rc)
        _remove_container_if_exists(cname, timeout=30)
        raise RuntimeError("Failed to start container: " + stderr)
    try:
        from unified_runner.run_unified_harbor import _inject_cn_mirrors
        _inject_cn_mirrors(cname)
    except Exception as exc:
        print(f"    [cn-mirror] SWE inject failed: {exc}")
    return cname


def stop_container(cname):
    _, stop_stderr, stop_rc = docker_run(["docker", "stop", "-t", "3", cname], timeout=20)
    _, rm_stderr, rm_rc = docker_run(["docker", "rm", "-f", cname], timeout=30)
    ok = rm_rc == 0
    if not ok:
        record_lifecycle_event(
            "teardown_failed",
            container=cname,
            stop_rc=stop_rc,
            stop_stderr=stop_stderr[:500],
            rm_rc=rm_rc,
            rm_stderr=rm_stderr[:1000],
        )
    else:
        record_lifecycle_event("teardown_ok", container=cname, stop_rc=stop_rc)
    return ok


def get_repo_path(cname):
    for candidate in ["/testbed", "/workspace", "/repo"]:
        _, _, rc = docker_run(
            ["docker", "exec", cname, "test", "-d", candidate], timeout=10)
        if rc == 0:
            return candidate
    return "/testbed"


def check_test_pass(test_output):
    """Check if pytest output indicates all target tests passed."""
    lines = test_output.strip().split("\n")
    for line in reversed(lines[-10:]):
        line = line.strip()
        if " passed" in line and ("failed" not in line) and ("error" not in line.lower().replace("no errors", "")):
            return True
        if line == "OK":
            return True
        if "FAILED" in line or "ERRORS" in line:
            return False
        if "no tests ran" in line.lower():
            return False
    return False


def run_tests(cname, repo_path, tests):
    # Official SWE images are already dependency-pinned. Installing
    # requirements*.txt at evaluation time can silently upgrade/downgrade
    # packages and create false negatives (e.g. old code importing symbols
    # removed from newer moto). Keep dependency installation opt-in.
    python_cmd = (
        "if [ -x /opt/miniconda3/envs/testbed/bin/python ]; then "
        "echo /opt/miniconda3/envs/testbed/bin/python; else echo python; fi"
    )
    # Some images do not put the testbed env first on PATH. Run pytest through
    # the pinned testbed interpreter when it exists, and install pytest only if
    # that interpreter lacks it. Avoid broad dependency installation by default.
    container_exec(cname,
        "PY=$(" + python_cmd + "); "
        "$PY - <<'PYTEST_IMPORT' >/dev/null 2>&1 || "
        "$PY -m pip install -q pytest 2>/dev/null || true\n"
        "import pytest\n"
        "PYTEST_IMPORT",
        timeout=120)
    if os.environ.get("UNIFIED_SWE_INSTALL_TEST_DEPS", "").strip() == "1":
        container_exec(cname,
            "PY=$(" + python_cmd + "); "
            "$PY -m pip install -q pytest sure funcy pytest-xdist pytest-benchmark 2>/dev/null || true",
            timeout=120)
        container_exec(cname,
            "PY=$(" + python_cmd + "); "
            "cd " + repo_path + " && "
            "for f in requirements*.txt test-requirements.txt; do "
            "  [ -f \"$f\" ] && $PY -m pip install -q -r \"$f\" 2>/dev/null || true; "
            "done",
            timeout=120)
    # Normalize test specs: strip leading / if present
    normalized = []
    for t in tests:
        t = t.strip()
        if t.startswith("/"):
            if t.startswith(repo_path + "/"):
                t = t[len(repo_path) + 1:]
            elif t.startswith(repo_path):
                t = t[len(repo_path):]
            else:
                t = t.lstrip("/")
        normalized.append(t)
    test_spec = " ".join("'" + t + "'" for t in normalized)
    # 4. Run pytest with overridden addopts to avoid conftest/pyproject.toml arg conflicts
    cmd = ("PY=$(" + python_cmd + "); cd " + repo_path + " && $PY -m pytest " + test_spec
           + " -x --tb=short --no-header -rN"
           + " -o 'addopts='"
           + " -p no:cacheprovider"
           + " 2>&1 | tail -80")
    verifier_timeout = int(os.environ.get("UNIFIED_SWE_VERIFIER_TIMEOUT_SEC", "300"))
    cap = os.environ.get("UNIFIED_VERIFIER_TIMEOUT_CAP_SEC")
    if cap:
        try:
            verifier_timeout = min(verifier_timeout, max(1, int(float(cap))))
        except ValueError:
            pass
    return container_exec(cname, cmd, timeout=verifier_timeout)


def apply_gold_test_patch(cname: str, repo_path: str, test_patch: str) -> tuple[bool, str]:
    """Apply SWE gold test-only patch after model editing, before verification.

    SWE-bench-style `FAIL_TO_PASS` selectors often refer to tests introduced by
    the gold test patch. The model must never see those tests while solving, but
    the verifier must apply them before running pytest, matching the official
    harness contract.
    """
    if not test_patch or not test_patch.strip():
        return False, "empty"
    encoded = base64.b64encode(test_patch.encode()).decode()
    cmd = f"echo '{encoded}' | base64 -d | git apply 2>&1"
    stdout, stderr, rc = docker_run(
        ["docker", "exec", "-w", repo_path, cname, "bash", "-lc", cmd],
        timeout=60,
    )
    output = stdout + ("\n" + stderr if stderr else "")
    if rc == 0:
        return True, output[-1000:]
    fallback_cmd = f"echo '{encoded}' | base64 -d | git apply --reject 2>&1"
    fb_stdout, fb_stderr, fb_rc = docker_run(
        ["docker", "exec", "-w", repo_path, cname, "bash", "-lc", fallback_cmd],
        timeout=60,
    )
    fallback_output = fb_stdout + ("\n" + fb_stderr if fb_stderr else "")
    applied = fb_rc == 0 or "Applied patch" in fallback_output
    return applied, (output + "\n" + fallback_output)[-2000:]


def format_problem_statement(problem: str, max_problem_chars: int = 0) -> tuple[str, bool]:
    """Return the SWE issue text, optionally truncated for debugging.

    Earlier versions hard-truncated every issue to 3K characters. That is too
    lossy for SWE-style bugs: many instances put reproduction details, expected
    behavior, or traceback context after the first few thousand characters.
    The default is now no truncation; pass a positive cap only for stress tests.
    """
    if max_problem_chars > 0 and len(problem) > max_problem_chars:
        return problem[:max_problem_chars] + "\n... [truncated]", True
    return problem, False


def run_instance(instance, config, retrieval_mapping=None, retrieval_top_n=3, skill_arm="baseline",
                 top1_skill_text_mapping=None,
                 max_problem_chars: int = 0):
    """Run one SWE instance using the unified agent loop.

    retrieval_mapping: optional {instance_id: [skill_path,...]}. When set, top-N
                      skills are docker-cp'd into /root/.claude/skills/ etc and
                      sys prompt gets a hint. skill_arm is a label for trajectory
                      metadata ("baseline", "retrieval", "irrelevant").

    See run_unified_harbor.py::run_task for the SFT-collection env vars
    (UNIFIED_IMPLICIT_MODE, UNIFIED_REFLECTION_CONTEXT) — same semantics here.
    """
    from unified_runner.implicit_instruction import apply_implicit_and_reflection

    instance_id = instance["instance_id"]
    image = instance_id_to_image(instance_id)
    result = {
        "instance_id": instance_id,
        "dataset": instance["dataset"],
        "difficulty": instance.get("difficulty", "unknown"),
        "resolved": False, "turns": 0, "time_sec": 0,
        "patch": "", "test_output": "", "error": "",
        "input_tokens": 0, "output_tokens": 0,
        "skill_arm": skill_arm,
        "retrieval_skills_injected": 0,
        "implicit_mode": os.environ.get("UNIFIED_IMPLICIT_MODE", "").strip(),
        "implicit_text": "",
        "reflection_context": os.environ.get("UNIFIED_REFLECTION_CONTEXT", "").strip(),
        "reflection_text": "",
        "problem_chars": len(instance["problem_statement"]),
        "problem_truncated": False,
        "max_problem_chars": max_problem_chars,
        "test_patch_applied": False,
        "test_patch_output": "",
    }
    start_time = time.time()
    cname = None
    tool_layer = None
    traj = None  # captured for trajectory save outside try/finally
    try:
        with contextlib.nullcontext():
            print("  Starting container for " + instance_id + "...")
            cname = start_container(instance_id, image)
            repo_path = get_repo_path(cname)
            print("  Container: " + cname + ", repo: " + repo_path)

            # Optional: inject retrieval-selected / irrelevant skills
            retrieval_hint = ""
            direct_skill_prompt = ""
            if retrieval_mapping is not None:
                n_injected = inject_retrieval_skills(
                    docker_run, cname, instance_id, retrieval_mapping, top_n=retrieval_top_n,
                )
                result["retrieval_skills_injected"] = n_injected
                retrieval_hint = build_retrieval_prompt_hint(
                    instance_id, retrieval_mapping, retrieval_top_n, arm=skill_arm,
                )
            if top1_skill_text_mapping is not None:
                direct_skill_prompt, skill_name = build_top1_skill_text_prompt(
                    instance_id, top1_skill_text_mapping,
                )
                if direct_skill_prompt:
                    result["retrieval_skills_injected"] = 1
                    result["top1_skill_text_name"] = skill_name

            # Get repo context
            repo_listing = container_exec(cname, "ls " + repo_path, timeout=10)
            git_log = container_exec(cname, "cd " + repo_path + " && git log --oneline -5", timeout=10)

            # Build OpenClaw-compatible system prompt. SWE-specific repo state
            # (path, structure, git log) is inlined into Project Context AGENTS.md
            # so the user message stays just the issue description.
            from unified_runner.bench_workspace_files import build_workspace_files_for_bench
            workspace_files = build_workspace_files_for_bench(
                "swe_lite",
                repo_path=repo_path,
                repo_listing=repo_listing,
                git_log=git_log,
            )
            sys_prompt = build_openclaw_system_prompt(
                workspace_dir=repo_path,
                skills_prompt=retrieval_hint,
                direct_skill_prompt=direct_skill_prompt,
                sandboxed=True,
                runtime_label="unified_runner.swe_lite",
                workspace_files=workspace_files,
            )

            # SFT-collection: optional implicit + reflection appended to sys_prompt.
            sys_prompt, applied_implicit, applied_reflection = apply_implicit_and_reflection(
                sys_prompt,
                implicit_mode=result["implicit_mode"],
                reflection_context=result["reflection_context"],
            )
            result["implicit_text"] = applied_implicit
            result["reflection_text"] = applied_reflection

            # Build task prompt — just the issue, no runtime-context tail.
            problem, problem_truncated = format_problem_statement(
                instance["problem_statement"],
                max_problem_chars=max_problem_chars,
            )
            result["problem_truncated"] = problem_truncated
            task_prompt = "Please fix the following issue:\n\n" + problem

            # Create tool layer in docker mode
            tool_layer = ToolLayer(mode="docker", container=cname, workdir=repo_path)

            # Create agent
            agent = UnifiedAgentLoop(config, tool_layer, max_tool_calls_per_turn=3)

            # Run the agent
            print("  Running unified agent loop...")
            traj = agent.run(task_prompt, system_prompt=sys_prompt)

            result["turns"] = traj.turns
            result["input_tokens"] = traj.total_input_tokens
            result["output_tokens"] = traj.total_output_tokens
            result["error"] = traj.error
            result["finish_reason"] = traj.finish_reason

            print(f"  Agent finished: {traj.finish_reason} "
                  f"(turns={traj.turns}, tokens={traj.total_input_tokens}in/{traj.total_output_tokens}out)")

            # Extract patch
            print("  Extracting patch...")
            patch = container_exec(cname, "cd " + repo_path + " && git diff", timeout=30)
            result["patch"] = patch

            # Run tests
            fail_to_pass = instance.get("FAIL_TO_PASS", [])
            if fail_to_pass and patch.strip():
                test_patch_ok, test_patch_output = apply_gold_test_patch(
                    cname,
                    repo_path,
                    instance.get("test_patch", ""),
                )
                result["test_patch_applied"] = test_patch_ok
                result["test_patch_output"] = test_patch_output
                print(f"  Running {len(fail_to_pass)} FAIL_TO_PASS tests...")
                test_results = run_tests(cname, repo_path, fail_to_pass)
                result["test_output"] = test_results
                result["resolved"] = check_test_pass(test_results)
            elif not patch.strip():
                result["error"] = (result.get("error", "") + " No patch generated").strip()
                result["resolved"] = False

    except Exception as e:
        result["error"] = type(e).__name__ + ": " + str(e)

    finally:
        result["time_sec"] = int(time.time() - start_time)
        if tool_layer is not None:
            tool_layer.close()
        if cname:
            print("  Cleaning up container " + cname + "...")
            stop_container(cname)

    # Attach trajectory (SFT-ready messages) to result so main() can persist it.
    # Attribute agent_loop.UnifiedAgentLoop.run returns `traj` which is captured
    # via local `traj` name in the try: body.
    if traj is not None:
        result["trajectory"] = traj.to_sft_messages()
    return result


def run_instance_with_retry(inst, config, *, max_retries=2, retry_backoff_sec=30, **kw):
    """Wrapper: retry on flaky-infra errors (container-start timeout / pull fail)."""
    last_result = None
    for attempt in range(max_retries + 1):
        result = run_instance(inst, config, **kw)
        err = result.get("error", "")
        if not _is_flaky_error(err):
            return result
        last_result = result
        if attempt < max_retries:
            # Match per-PID naming used in start_container (2026-04-26 fix).
            cname = "swe-unified-" + inst["instance_id"].replace("/", "_") + f"-p{os.getpid()}"
            print(f"    [retry {attempt + 1}/{max_retries}] flaky error, cleaning up "
                  f"and retrying in {retry_backoff_sec}s: {err[:120]}")
            _remove_container_if_exists(cname, timeout=15)
            _remove_containers_by_name_prefix(f"{cname}-", max_remove=8)
            time.sleep(retry_backoff_sec)
    last_result["error"] = (last_result.get("error", "") +
                            f" [exhausted {max_retries} retries]").strip()
    return last_result


def main():
    parser = argparse.ArgumentParser(description="Unified SWE-bench evaluation")
    parser.add_argument("--model", default="qwen3.5-27b", help="Model name")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", "http://localhost:30000/v1"))
    parser.add_argument("--max-turns", type=int,
                        default=int(os.environ.get("UNIFIED_DEFAULT_MAX_TURNS", "70")))
    parser.add_argument(
        "--max-time",
        type=int,
        default=int(os.environ.get("UNIFIED_ROLLOUT_WALLCLOCK_CAP_SEC", "900")),
    )
    parser.add_argument(
        "--max-problem-chars",
        type=int,
        default=int(os.environ.get("UNIFIED_SWE_MAX_PROBLEM_CHARS", "0")),
        help="Maximum issue-description characters to send. <=0 means no truncation "
             "(default; old runner hard-truncated to 3000).",
    )
    parser.add_argument("--instance", type=str, help="Run only this instance_id")
    parser.add_argument("--instance-file", type=str,
                        help="Path to a file with one instance_id per line (# comments OK). "
                             "Overrides the hard-coded ALL_IMAGES list. Use with "
                             "GeneralAgent/eval_scripts/prebake_images/swe_lite_100.txt "
                             "to run the 100-instance stratified subset.")
    parser.add_argument("--retries", type=int, default=2,
                        help="Retries on flaky-infra errors (default 2). "
                             "Agent failures / No-patch are NOT retried.")
    parser.add_argument("--inject-retrieval-skills", type=str, default=None,
                        help="Path to retrieval jsonl (20260418_retrieval_swe_qwen3emb8b.jsonl). "
                             "When set, top-N retrieved skills per instance docker-cp'd into "
                             "/root/.claude/skills/.")
    parser.add_argument("--inject-irrelevant-skills", type=str, default=None,
                        help="Path to retrieval jsonl. Inject top-N *irrelevant* (negative-control) "
                             "skills: random from skill_libraries/merged/ excluding the instance's "
                             "coarse_top20, deterministic per instance_id. "
                             "Mutually exclusive with --inject-retrieval-skills.")
    parser.add_argument("--inject-top1-skill-text", type=str, default=None,
                        help="Path to retrieval jsonl. Inject the top-1 skill's SKILL.md text "
                             "directly into the system prompt. Mutually exclusive with other "
                             "skill injection modes.")
    parser.add_argument("--retrieval-top-n", type=int, default=3)
    parser.add_argument("--arm-tag", type=str, default="",
                        help="Suffix appended to output jsonl filename, e.g. 'retrieval' → "
                             "20260419_swe_27b_retrieval_incremental.jsonl. Empty = baseline.")
    args = parser.parse_args()
    if sum([
        bool(args.inject_retrieval_skills),
        bool(args.inject_irrelevant_skills),
        bool(args.inject_top1_skill_text),
    ]) > 1:
        print("ERROR: --inject-retrieval-skills / --inject-irrelevant-skills / "
              "--inject-top1-skill-text mutually exclusive",
              file=sys.stderr)
        sys.exit(2)

    from unified_runner.base import env_overrides
    config = RunConfig(
        model=args.model,
        api_base=args.api_base,
        max_turns=args.max_turns,
        max_time_sec=args.max_time,
        temperature=0.6,
        max_tokens=4096,
        max_output_chars=8000,
        **env_overrides(),
    )

    print("Loading instances from parquet files...")
    all_instances = load_instances()
    print(f"Loaded {len(all_instances)} instances total")

    # Determine the instance list to run:
    #   --instance-file (from prebake_images/swe_lite_100.txt) ──► overrides ALL_IMAGES
    #   --instance <id>                                         ──► filters to one
    #   (neither)                                               ──► legacy ALL_IMAGES (22)
    if args.instance_file:
        with open(args.instance_file) as f:
            target_iids = [
                line.strip() for line in f
                if line.strip() and not line.lstrip().startswith("#")
            ]
        print(f"[instance-file] loaded {len(target_iids)} instance_ids from {args.instance_file}")
    else:
        target_iids = [image_to_instance_id(image) for image in ALL_IMAGES]

    to_run = []
    canonical_iids = {iid.lower(): iid for iid in all_instances}
    for iid in target_iids:
        if args.instance and iid != args.instance and iid.lower() != args.instance.lower():
            continue
        lookup_iid = iid if iid in all_instances else canonical_iids.get(iid.lower())
        if lookup_iid in all_instances:
            to_run.append(all_instances[lookup_iid])
        else:
            print(f"WARNING: {iid} not found in any parquet")

    print(f"\nWill run {len(to_run)} instances with UNIFIED interface")
    if not to_run:
        print("No instances to run!")
        return

    # Load skill-injection mapping for the selected arm
    retrieval_mapping = None
    top1_skill_text_mapping = None
    skill_arm = "baseline"
    if args.inject_retrieval_skills:
        retrieval_mapping = load_retrieval_mapping(args.inject_retrieval_skills)
        skill_arm = "retrieval"
    elif args.inject_irrelevant_skills:
        retrieval_mapping = build_irrelevant_mapping(args.inject_irrelevant_skills,
                                                      top_n=args.retrieval_top_n)
        skill_arm = "irrelevant"
    elif args.inject_top1_skill_text:
        top1_skill_text_mapping = load_retrieval_mapping(args.inject_top1_skill_text)
        skill_arm = "top1_skill_text"

    # Results files — 2026-04-22 v8 layout:
    #   results/<date>/swe/<experiment>/{incremental.jsonl, trajectories/, summary.md}
    from unified_runner.base import results_subdir, experiment_name
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    date_prefix = os.environ.get("UNIFIED_RESULTS_DATE") or datetime.now().strftime("%Y%m%d")
    tag = args.arm_tag if args.arm_tag else None
    exp_dir = results_subdir(RESULTS_DIR, date_prefix, bench="swe",
                             experiment=experiment_name(skill_arm, tag=tag))
    inc_path = exp_dir / "incremental.jsonl"
    traj_dir = exp_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {exp_dir.relative_to(RESULTS_DIR)}  skill_arm={skill_arm}")

    results = []
    for idx, inst in enumerate(to_run, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(to_run)}] UNIFIED[{skill_arm}]: {inst['instance_id']} ({inst['dataset']})")
        print(f"{'='*60}")

        result = run_instance_with_retry(
            inst, config, max_retries=args.retries,
            retrieval_mapping=retrieval_mapping,
            retrieval_top_n=args.retrieval_top_n,
            skill_arm=skill_arm,
            top1_skill_text_mapping=top1_skill_text_mapping,
            max_problem_chars=args.max_problem_chars,
        )
        results.append(result)

        # Persist trajectory (SFT-ready OpenAI chat messages) to its own file.
        # implicit_text + reflection_text are saved so the SFT collector can
        # strip those exact bytes from the system message at export time.
        traj = result.pop("trajectory", None)
        if traj is not None:
            traj_path = traj_dir / f"{inst['instance_id'].replace('/', '__')}.json"
            traj_path.write_text(json.dumps({
                "instance_id": inst["instance_id"],
                "dataset": inst["dataset"],
                "skill_arm": skill_arm,
                "retrieval_skills_injected": result.get("retrieval_skills_injected", 0),
                "top1_skill_text_name": result.get("top1_skill_text_name", ""),
                "resolved": result.get("resolved", False),
                "finish_reason": result.get("finish_reason", ""),
                "problem_chars": result.get("problem_chars", 0),
                "problem_truncated": result.get("problem_truncated", False),
                "implicit_mode": result.get("implicit_mode", ""),
                "implicit_text": result.get("implicit_text", ""),
                "reflection_context": result.get("reflection_context", ""),
                "reflection_text": result.get("reflection_text", ""),
                "messages": traj,
            }, ensure_ascii=False, default=str, indent=2))

        # Save incremental
        with open(inc_path, "a") as f:
            r = {k: v for k, v in result.items() if k != "agent_log"}
            f.write(json.dumps(r, default=str) + "\n")

        status = "RESOLVED" if result["resolved"] else "FAILED"
        print(f"  [{idx}/{len(to_run)}] {result['instance_id']}: {status} "
              f"(turns={result['turns']}, time={result['time_sec']}s)")

    # Final summary
    total = len(results)
    resolved = sum(1 for r in results if r["resolved"])
    mean_score = resolved / total if total else 0
    pct = 100 * mean_score

    print(f"\n{'='*60}")
    print(f"UNIFIED SWE FINAL: {resolved}/{total} resolved ({pct:.1f}%)")
    print(f"{'='*60}")

    # Write summary — v8 schema (N_total/N_pass/N_error/pass_rate/Mean_score)
    # normalize: SWE has "instance_id" not "task_id"; summary writer handles both
    for r in results:
        if "task_id" not in r and "instance_id" in r:
            r["task_id"] = r["instance_id"]
        if "score" not in r:
            r["score"] = 1.0 if r.get("resolved") else 0.0
    from unified_runner.base import write_summary_md
    summary_path = write_summary_md(exp_dir, "swe-gym", args.model, results,
                                    extra_meta={"skill_arm": skill_arm,
                                                "n_instances": total})
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
