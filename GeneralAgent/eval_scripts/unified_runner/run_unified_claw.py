#!/usr/bin/env python3
"""Unified Claw-Eval runner v2 — starts mock HTTP services, augments prompt with
curl documentation, parses exec trajectory for curl dispatches, scores via
task.yaml's declarative `scoring_components` (180 / 300 tasks).

Tasks with only `grader.py` (imperative, needs TraceMessage/ToolDispatch models)
are still skipped — they need a deeper integration with claw-eval's trace format.

Flow per task:
  1. Load task.yaml
  2. Start every service in `services:` (python mock_services/<name>/server.py on its port)
  3. Wait health_check passes (POST/GET)
  4. Reset state via reset_endpoint (idempotent)
  5. Copy sandbox_files into tempdir; set up workdir
  6. Augment SYSTEM_PROMPT with per-task tool endpoints documentation
  7. Run UnifiedAgentLoop(host mode, workdir=tempdir)
  8. Parse agent trajectory: count curl-based tool calls by URL match
  9. Score via `scoring_components` (tool_called + keywords_present + safety_checks)
 10. Stop services, append jsonl
"""

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

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
    build_http_tool_runtime_context,
    build_openclaw_system_prompt,
)
from unified_runner.docker_start_gate import docker_start_gate
from unified_runner.claw_grader_adapter import (
    collect_audit_from_services, grade_with_native_grader, format_dim_scores,
)

BASE_DIR = Path("/path/to/skillRL")
CLAW_DIR = BASE_DIR / "datasets/claw-eval"
TASKS_DIR = CLAW_DIR / "tasks"
RESULTS_DIR = BASE_DIR / "experiments"
WORKDIR = Path("/tmp/claw_pilot")
# Per-arm archive dir under RESULTS_DIR; set by main() from --out, used by run_task
# so baseline/retrieval/irrelevant trajectories don't overwrite each other (matches
# Harbor runner's {out_stem}_trajectories/ layout).
TRAJ_ARCHIVE_DIR: Path | None = None


def build_tool_docs(task_def: dict) -> str:
    """Turn task.yaml tools + tool_endpoints into a markdown bullet list for prompt."""
    tools = task_def.get("tools", []) or []
    endpoints = {e["tool_name"]: e for e in task_def.get("tool_endpoints", []) or []}
    if not tools and not endpoints:
        return "(This task has no HTTP tools; work locally in your workdir.)"
    lines = ["**HTTP Tools available (use `exec` + `curl`):**\n"]
    for t in tools:
        name = t["name"]
        desc = t.get("description", "")
        ep = endpoints.get(name, {})
        url = ep.get("url", "")
        method = ep.get("method", "POST")
        schema = t.get("input_schema", {}).get("properties", {}) or {}
        params = ", ".join(schema.keys()) or "(no args)"
        lines.append(f"- **{name}** — {desc}")
        if url:
            lines.append(f"  `{method} {url}` body: `{params}`")
    return "\n".join(lines)


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except Exception:
            return False


def _http_probe(url: str, method: str = "GET", body: dict | None = None, timeout: float = 5.0) -> tuple[int, str]:
    """Small stdlib HTTP probe; returns (status_code, body_text). 0 if cannot connect."""
    import urllib.request
    req = urllib.request.Request(url, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
        data = json.dumps(body).encode()
    else:
        data = b"{}" if method.upper() == "POST" else None
        if data:
            req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        # urllib raises HTTPError subclass with .code for 4xx/5xx
        status = getattr(e, "code", 0)
        body_txt = ""
        if hasattr(e, "read"):
            try:
                body_txt = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
        return status, body_txt or str(e)


# ============================================================================
# 2026-04-21 Plan A+B: shared mock container + shared network across ALL tasks
# ============================================================================
# Motivation: per-task `docker run mock + network create` adds ~10s overhead.
# Amortize by sharing ONE container + ONE network for all workers/tasks.
#
# Safety analysis (confirmed 2026-04-21):
# - Mock services have process-isolated state (_emails etc. are Python module
#   globals per-process, not shared across service processes).
# - Fixture file paths come from env vars (e.g. GMAIL_FIXTURES) set at spawn.
# - PORT also from env var; port_offset = worker_idx × 100 ensures disjoint
#   ranges across parallel workers. No /reset collision.
# Within a single worker, tasks run serially, so we kill old service PIDs
# before spawning new ones on the same port.
#
# Shared mock container: `claw-mock-shared[-wN]`; shared network:
# `claw-net-shared[-wN]`. Sandbox containers are still per-task (need per-task
# workspace isolation).
# ============================================================================

# 2026-05-11: cross-subprocess claw parallelism.
# Each SFT-collection subprocess can be assigned a worker_idx via env var so
# that sandbox cname / workdir / mock-service ports do not collide with
# concurrent sibling subprocesses. For subprocess-level concurrency, the mock
# container/network also need the same suffix; a module-level lock/cache only
# protects threads inside one subprocess, not sibling subprocesses.
_CLAW_WORKER_IDX = int((os.environ.get("CLAW_WORKER_IDX", "0") or "0").strip() or "0")
_CLAW_WORKER_SUFFIX = f"-w{_CLAW_WORKER_IDX}" if _CLAW_WORKER_IDX > 0 else ""
_CLAW_SKIP_SHARED_CLEANUP = os.environ.get("CLAW_SKIP_SHARED_CLEANUP", "").strip() == "1"
_SHARED_MOCK_CNAME_BASE = "claw-mock-shared"
_SHARED_NET_NAME_BASE = "claw-net-shared"
_SHARED_MOCK_CNAME = f"{_SHARED_MOCK_CNAME_BASE}{_CLAW_WORKER_SUFFIX}"
_SHARED_NET_NAME = f"{_SHARED_NET_NAME_BASE}{_CLAW_WORKER_SUFFIX}"
_shared_mock_ip: str | None = None
_shared_mock_lock = __import__("threading").Lock()


def _ensure_shared_network() -> bool:
    """Ensure the per-worker shared Docker network exists before mock startup."""
    for attempt in range(3):
        _, _, rc = _docker_run(
            ["docker", "network", "inspect", _SHARED_NET_NAME],
            timeout=30,
            retry_sec_override=0,
        )
        if rc == 0:
            return True
        _docker_run(["docker", "network", "create", _SHARED_NET_NAME], timeout=60)
        _, _, rc = _docker_run(
            ["docker", "network", "inspect", _SHARED_NET_NAME],
            timeout=30,
            retry_sec_override=0,
        )
        if rc == 0:
            return True
        time.sleep(2 * (attempt + 1))
    print(f"  [shared-mock] ❌ network unavailable: {_SHARED_NET_NAME}", flush=True)
    return False


def _ensure_shared_mock_infra() -> str | None:
    """Idempotent init: ensure claw-net-shared network + claw-mock-shared
    container are running. Returns mock container IP, or None on failure.

    Thread-safe (first caller initializes; others block then return cached).
    """
    global _shared_mock_ip
    with _shared_mock_lock:
        if _shared_mock_ip:
            stdout, _, rc = _docker_run(["docker", "ps", "-f", f"name=^{_SHARED_MOCK_CNAME}$",
                                         "--format", "{{.Status}}"], timeout=30)
            running = rc == 0 and stdout.strip().startswith("Up")
            if running:
                return _shared_mock_ip
            print(
                f"  [shared-mock] cached {_SHARED_MOCK_CNAME}@{_shared_mock_ip} is not running; recreating",
                flush=True,
            )
            _shared_mock_ip = None
        if not _ensure_shared_network():
            return None
        # Mock container
        stdout, _, rc = _docker_run(["docker", "ps", "-a", "-f", f"name=^{_SHARED_MOCK_CNAME}$",
                                     "--format", "{{.Status}}"], timeout=30)
        running = rc == 0 and stdout.strip().startswith("Up")
        if not running:
            _docker_run(["docker", "rm", "-f", _SHARED_MOCK_CNAME], timeout=60)
            run_cmd = [
                "docker", "run", "-d",
                "--name", _SHARED_MOCK_CNAME,
                "--network", _SHARED_NET_NAME,
                "-e", "PYTHONPATH=/claw-eval",
                "-e", "CLAW_DIR=/claw-eval",
                "claw-mock-services:latest", "sleep", "infinity",
            ]
            mock_run_timeout = int(os.environ.get("UNIFIED_DOCKER_MOCK_RUN_TIMEOUT_SEC", "420") or "420")
            with docker_start_gate(f"claw-mock:{_SHARED_MOCK_CNAME}"):
                _, stderr, rc2 = _docker_run(run_cmd, timeout=mock_run_timeout)
            if rc2 != 0 and f"network {_SHARED_NET_NAME} not found" in stderr:
                print(
                    f"  [shared-mock] recreating missing network {_SHARED_NET_NAME} after docker-run failure",
                    flush=True,
                )
                if _ensure_shared_network():
                    with docker_start_gate(f"claw-mock:{_SHARED_MOCK_CNAME}:net-retry"):
                        _, stderr, rc2 = _docker_run(run_cmd, timeout=mock_run_timeout)
            if rc2 != 0 and ("Conflict." in stderr or "already in use by container" in stderr):
                print(
                    f"  [shared-mock] removing conflicted {_SHARED_MOCK_CNAME} after docker-run retry",
                    flush=True,
                )
                _docker_run(
                    ["docker", "rm", "-f", _SHARED_MOCK_CNAME],
                    timeout=180,
                    retry_sec_override=120,
                )
                for _ in range(30):
                    stdout, _, rc_check = _docker_run(
                        ["docker", "ps", "-a", "-f", f"name=^{_SHARED_MOCK_CNAME}$", "--format", "{{.Names}}"],
                        timeout=15,
                        retry_sec_override=0,
                    )
                    if rc_check == 0 and not stdout.strip():
                        break
                    time.sleep(2)
                with docker_start_gate(f"claw-mock:{_SHARED_MOCK_CNAME}:retry"):
                    _, stderr, rc2 = _docker_run(run_cmd, timeout=mock_run_timeout)
            if rc2 != 0:
                print(f"  [shared-mock] ❌ start failed: {stderr[:200]}", flush=True)
                return None
            # Create pid/log dirs for tracking
            _docker_run(["docker", "exec", _SHARED_MOCK_CNAME, "mkdir", "-p",
                         "/tmp/svc", "/tmp/svc_logs"], timeout=30)
        # Inspect IP
        stdout, _, rc = _docker_run([
            "docker", "inspect", _SHARED_MOCK_CNAME,
            "--format", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
        ], timeout=30)
        ip = stdout.strip() if rc == 0 else ""
        if not ip:
            print(f"  [shared-mock] ⚠ IP empty after inspect", flush=True)
            return None
        _shared_mock_ip = ip
        print(f"  [shared-mock] ready: {_SHARED_MOCK_CNAME}@{ip} on {_SHARED_NET_NAME}", flush=True)
        return ip


def cleanup_shared_mock_infra() -> None:
    """Call at end of main run to remove shared container + network.
    Registered via atexit in main()."""
    global _shared_mock_ip
    _docker_run(["docker", "rm", "-f", _SHARED_MOCK_CNAME], timeout=120)
    _docker_run(["docker", "network", "rm", _SHARED_NET_NAME], timeout=60)
    _shared_mock_ip = None


class ServiceManager:
    """Start/stop claw-eval mock HTTP services for one task.

    Two modes:
      - host  (default): spawn each service as a tidalfs subprocess on loopback.
      - docker (v6 new): spawn all services inside the SHARED mock container
                        on your-docker-host. Agent sandbox container joins shared
                        network and sees mock via --add-host host.docker.internal.
                        Use when UNIFIED_CLAW_USE_DOCKER_SANDBOX=1.

    2026-04-21: docker mode now uses SHARED mock container + network (Plan A+B).
    Services spawned via `docker exec -d` onto shared container, with PID
    tracking for per-task cleanup. Expected saving ~10s/task vs per-task.
    """

    def __init__(self, services: list[dict], task_dir: Path,
                 mode: str = "host",
                 net_name: str | None = None,
                 mock_cname: str | None = None):
        self.services = services
        self.task_dir = task_dir
        self.mode = mode
        # In shared-docker mode, ignore per-task net/cname args and use shared.
        self.net_name = _SHARED_NET_NAME if mode == "docker" else net_name
        self.mock_cname = _SHARED_MOCK_CNAME if mode == "docker" else mock_cname
        self.mock_ip: str | None = None  # populated after _start_docker
        self.procs: list[subprocess.Popen] = []  # host mode only
        self.svc_ports: list[int] = []  # docker shared mode: ports we spawned for this task

    def start_all(self, log_dir: Path) -> list[str]:
        if self.mode == "docker":
            return self._start_docker(log_dir)
        return self._start_host(log_dir)

    def _start_host(self, log_dir: Path) -> list[str]:
        """Start all services, wait for health. Returns list of status lines."""
        log_dir.mkdir(parents=True, exist_ok=True)
        status = []
        for svc in self.services:
            name = svc["name"]
            port = int(svc["port"])
            cmd = svc["command"].split()
            # Bind service env
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PORT"] = str(port)
            for k, v in (svc.get("env") or {}).items():
                # Heuristic: numeric / pure-int values (e.g. PORT injected by
                # _apply_port_offset) are passed through unchanged. Everything
                # else is treated as a fixture-file path relative to CLAW_DIR.
                sv = str(v)
                if sv.isdigit() or k in ("PORT", "LOG_LEVEL"):
                    env[k] = sv
                else:
                    env[k] = str(CLAW_DIR / sv) if not os.path.isabs(sv) else sv
            log_path = log_dir / f"service_{name}.log"
            lf = open(log_path, "w")
            proc = subprocess.Popen(
                cmd, cwd=str(CLAW_DIR), env=env, stdout=lf, stderr=subprocess.STDOUT,
            )
            self.procs.append(proc)
            status.append(f"  started {name} (pid={proc.pid}, port={port}, log={log_path})")
            # Wait for port
            t0 = time.time()
            ready_timeout = max(
                int(svc.get("ready_timeout", 15)),
                int(os.environ.get("UNIFIED_CLAW_READY_TIMEOUT_FLOOR_SEC", "30") or "30"),
            )
            while time.time() - t0 < ready_timeout:
                if _port_open(port):
                    break
                time.sleep(0.3)
            else:
                status.append(f"  ⚠ {name} port {port} not open after {ready_timeout}s")
                continue
            # Health check
            hc_url = svc.get("health_check", "")
            hc_method = svc.get("health_check_method", "GET")
            if hc_url:
                code, _ = _http_probe(hc_url, method=hc_method, body={})
                status.append(f"  {name} health_check {hc_method} {hc_url} -> {code}")
            # Reset state
            if svc.get("reset_endpoint"):
                _http_probe(svc["reset_endpoint"], method="POST", body={})
        return status

    def _start_docker(self, log_dir: Path) -> list[str]:
        """Docker mode (v6 shared): spawn services INTO shared mock container.

        Infra (shared container + network) is ensured at module level once;
        this method only spawns task-specific service processes on ports that
        are disjoint across workers (via port_offset). Service PIDs are
        written to /tmp/svc/<port>.pid inside the container so stop_all can
        kill them without touching the container.
        """
        log_dir.mkdir(parents=True, exist_ok=True)
        status = []

        # 1. Ensure shared infra is up (idempotent; first caller creates, rest reuse)
        self.mock_ip = _ensure_shared_mock_infra()
        if not self.mock_ip:
            status.append(f"  ❌ shared mock infra unavailable")
            return status
        status.append(f"  using shared mock {self.mock_cname}@{self.mock_ip}")

        # 2. Spawn each service via docker exec on shared container.
        # Use `(cmd & echo $! > pidfile)` pattern so we can kill precisely later.
        for svc in self.services:
            name = svc["name"]
            port = int(svc["port"])
            cmd = svc["command"]  # e.g. "python mock_services/gmail/server.py"
            self.svc_ports.append(port)

            # Build export lines for env (services read PORT + FIXTURES_PATH via os.environ)
            env_exports = [f"export PORT={port}", "export PYTHONUNBUFFERED=1"]
            for k, v in (svc.get("env") or {}).items():
                sv = str(v)
                if sv.isdigit() or k in ("PORT", "LOG_LEVEL"):
                    env_exports.append(f"export {k}={sv}")
                else:
                    ev = sv if os.path.isabs(sv) else f"/claw-eval/{sv}"
                    env_exports.append(f'export {k}="{ev}"')
            env_block = "; ".join(env_exports)

            # Detached spawn with PID capture.
            # `exec` replaces bash with python so the PID we capture is the python process.
            # If multiple services share a port (shouldn't happen), first one wins.
            inner = (
                f"cd /claw-eval && {env_block} && "
                f"({cmd} > /tmp/svc_logs/port_{port}.log 2>&1 & echo $! > /tmp/svc/port_{port}.pid)"
            )
            _, stderr, rc = _docker_run([
                "docker", "exec", self.mock_cname, "bash", "-c", inner,
            ], timeout=30)
            if rc != 0 and "No such container" in stderr:
                global _shared_mock_ip
                _shared_mock_ip = None
                self.mock_ip = _ensure_shared_mock_infra()
                if self.mock_ip:
                    _, stderr, rc = _docker_run([
                        "docker", "exec", self.mock_cname, "bash", "-c", inner,
                    ], timeout=30)
            if rc != 0:
                status.append(f"  ⚠ {name} spawn failed: {stderr[:100]}")
                continue
            status.append(f"  spawned {name} pid@port={port}")

            # Wait for port inside container
            t0 = time.time()
            ready_timeout = int(svc.get("ready_timeout", 15))
            ready = False
            while time.time() - t0 < ready_timeout:
                _, _, rc = _docker_run([
                    "docker", "exec", self.mock_cname,
                    "python", "-c",
                    f"import socket; s=socket.socket(); s.settimeout(1); "
                    f"exit(0 if s.connect_ex(('127.0.0.1',{port}))==0 else 1)",
                ], timeout=10)
                if rc == 0:
                    ready = True; break
                time.sleep(0.3)
            status.append(f"  {name} port {port} " + ("✓ ready" if ready else "✗ NOT ready"))

            # Reset endpoint (optional). In shared mode, call via docker exec curl
            # from INSIDE the mock container (tidalfs can't reach your-docker-host docker0 IPs).
            if svc.get("reset_endpoint") and ready:
                url = svc["reset_endpoint"]
                # Convert to localhost URL (mock container's loopback) with task's port
                local_url = (url.replace("host.docker.internal", "127.0.0.1")
                                .replace("localhost", "127.0.0.1"))
                # Apply port_offset: svc port is already offset-applied by _apply_port_offset(),
                # so reset URL's port also needs matching substitution. Since original URL uses
                # the pre-offset port, rewrite to this svc's port.
                import re as _re
                local_url = _re.sub(r":(\d+)/", f":{port}/", local_url, count=1)
                _docker_run([
                    "docker", "exec", self.mock_cname, "curl", "-sS",
                    "--max-time", "5",
                    "-X", "POST", "-H", "Content-Type: application/json",
                    "-d", "{}", local_url,
                ], timeout=10)
        return status

    def stop_all(self):
        if self.mode == "docker":
            # Shared mode: kill only THIS task's service PIDs. Leave container + network alive.
            for port in self.svc_ports:
                _docker_run([
                    "docker", "exec", self.mock_cname, "bash", "-c",
                    f"if [ -f /tmp/svc/port_{port}.pid ]; then "
                    f"kill -TERM $(cat /tmp/svc/port_{port}.pid) 2>/dev/null; "
                    f"sleep 0.1; "
                    f"kill -KILL $(cat /tmp/svc/port_{port}.pid) 2>/dev/null; "
                    f"rm -f /tmp/svc/port_{port}.pid; fi",
                ], timeout=10)
            self.svc_ports = []
            return
        # host mode
        for p in self.procs:
            try:
                p.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
        t0 = time.time()
        while time.time() - t0 < 2.0:
            if all(p.poll() is not None for p in self.procs):
                return
            time.sleep(0.1)
        for p in self.procs:
            try:
                p.kill()
            except ProcessLookupError:
                pass


def load_task(task_id: str) -> dict:
    return yaml.safe_load((TASKS_DIR / task_id / "task.yaml").read_text())


def setup_workdir(
    task_id: str,
    task_def: dict,
    worker_suffix: str | None = None,
) -> Path:
    # 2026-05-11: per-worker suffix so concurrent subprocesses don't rm/mkdir
    # over each other's workspace when both happen to pick the same task.
    suffix = worker_suffix if worker_suffix is not None else _CLAW_WORKER_SUFFIX
    wd = WORKDIR / f"{task_id}{suffix}" / "workspace"
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True, exist_ok=True)
    task_root = TASKS_DIR / task_id
    # 2026-04-21 v6 Bug C-1 fix: copy BOTH sandbox_files (agent inputs) and
    # sandbox_grader_files (used by env_snapshot_commands like verify_encoder.py).
    # T100-T104 graders read env_snapshot output of these scripts; without copy,
    # `python /workspace/fixtures/verify_encoder.py` fails → empty stdout → score=0.
    for rel in ((task_def.get("sandbox_files") or [])
                + (task_def.get("sandbox_grader_files") or [])):
        src = task_root / rel
        dst = wd / rel
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    return wd


# ---------- trajectory parsing ----------

CURL_RE = re.compile(r"curl\s+[^\n]*?(https?://[\w\.\-]+:(\d+)(/[\w\-/\._]*)?)", re.IGNORECASE)


def extract_tool_dispatches(traj, tool_endpoints: list[dict]) -> list[dict]:
    """Scan agent's exec tool-calls; match curl urls against registered tool endpoints.

    Returns list of {tool_name, url, raw_command} in order.
    """
    # index endpoints by (url_suffix) — endpoint URL is like http://localhost:9102/todo/tasks/update
    ep_by_path = {}
    for ep in tool_endpoints:
        url = ep["url"]
        # extract path after host:port
        m = re.match(r"https?://[^/]+(/[^\s]*)", url)
        if m:
            ep_by_path[m.group(1).rstrip("/")] = ep["tool_name"]

    dispatches = []
    for m in traj.messages:
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        for tc in m["tool_calls"]:
            fn = tc.get("function", {})
            if fn.get("name") not in ("exec", "process"):
                continue
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                continue
            cmd = args.get("command", "") or args.get("cmd", "")
            if not isinstance(cmd, str):
                continue
            for cm in CURL_RE.finditer(cmd):
                url = cm.group(1)
                mpath = re.match(r"https?://[^/]+(/[^\s?#]*)", url)
                path = mpath.group(1).rstrip("/") if mpath else ""
                tool_name = ep_by_path.get(path)
                if tool_name:
                    dispatches.append({"tool_name": tool_name, "url": url, "raw_command": cmd[:500]})
    return dispatches


def all_assistant_text(traj) -> str:
    parts = []
    for m in traj.messages:
        if m.get("role") == "assistant" and m.get("content"):
            parts.append(str(m["content"]))
    return "\n".join(parts)


# ---------- declarative scoring ----------

def _build_conversation_str(traj) -> str:
    """Serialize agent's trajectory into a compact str for LLM judge."""
    out = []
    for m in getattr(traj, "messages", []):
        role = m.get("role", "?")
        if role == "system":
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif item.get("type") == "tool_use":
                        parts.append(f"[tool_use: {item.get('name')}({str(item.get('input',{}))[:200]})]")
            content = "\n".join(parts)
        content = str(content)[:2000]  # truncate
        out.append(f"[{role}] {content}")
    joined = "\n\n".join(out)
    # Keep the TAIL of long conversations. The final deliverable/answer lives
    # at the end of the trajectory; the old head-keep truncation ([:15000])
    # silently dropped it for long runs, zeroing every llm_judge component
    # regardless of output quality.
    limit = 15000
    if len(joined) > limit:
        joined = "...(earlier turns truncated)...\n\n" + joined[-limit:]
    return joined


def _build_actions_summary(dispatches: list[dict]) -> str:
    """Summarize dispatched HTTP tool calls for judge."""
    if not dispatches:
        return "(no HTTP tool calls dispatched)"
    lines = []
    for d in dispatches[:50]:
        lines.append(f"- {d.get('tool_name')} → {d.get('url','')}")
    return "\n".join(lines)


_JUDGE = None
def _get_judge():
    """Lazy-init LLM judge pointing at MAAS deepseek v3.2 (same config as native claw_eval)."""
    global _JUDGE
    if _JUDGE is not None:
        return _JUDGE
    try:
        sys.path.insert(0, str(BASE_DIR / "datasets/claw-eval/src"))
        from claw_eval.graders.llm_judge import LLMJudge
        # Config from GeneralAgent/eval_scripts/claw_eval/config_qwen3.5_27b_local.yaml judge: section
        judge_api_key = os.environ.get("CLAW_JUDGE_API_KEY", "your-maas-api-key")
        judge_base_url = os.environ.get("CLAW_JUDGE_BASE_URL", "https://your-llm-endpoint/v1")
        judge_model = os.environ.get("CLAW_JUDGE_MODEL", "deepseek-v3.2")
        _JUDGE = LLMJudge(model_id=judge_model, api_key=judge_api_key, base_url=judge_base_url)
        print(f"  [judge] initialized: {judge_model} @ {judge_base_url}", flush=True)
        return _JUDGE
    except Exception as e:
        print(f"  [judge] init failed: {e}", flush=True)
        return None


def score_from_components(task_def: dict, traj, dispatches: list[dict]) -> tuple[bool, float, str]:
    """Grade using task.yaml's `scoring_components` and `safety_checks`.

    Supports 5 check types (aligned with native claw-eval):
    - tool_called       — HTTP tool was called >= min_calls
    - keywords_present  — partial-match hit ratio (was binary before 04-19 fix)
    - tool_not_called   — HTTP tool was NOT called
    - min_length        — final_text length >= threshold (**v3 新增**)
    - categories_present — 所有指定 category 出现在 text (**v3 新增**)
    - llm_judge         — 用 deepseek v3.2 给 0-1 连续 partial credit (**v3 新增**)
    """
    components = task_def.get("scoring_components") or []
    safety = task_def.get("safety_checks") or []
    if not components:
        return False, 0.0, "no scoring_components"

    tool_counts: dict[str, int] = {}
    for d in dispatches:
        tool_counts[d["tool_name"]] = tool_counts.get(d["tool_name"], 0) + 1
    raw_text = all_assistant_text(traj)  # 原始大小写，给 categories / min_length 用
    text = raw_text.lower()              # lowercase 给 keywords 用

    # Prep judge inputs once (lazy)
    _judge_conv = None
    _judge_actions = None
    task_prompt = (task_def.get("prompt") or {}).get("text", "")

    total_score = 0.0
    total_weight = 0.0
    per_comp = []
    for c in components:
        w = float(c.get("weight", 0.0))
        total_weight += w
        chk = c.get("check", {}) or {}
        kind = chk.get("type")
        cname = c.get("name", "?")
        comp_score = 0.0  # 0-1 score for this component (before weighting)
        note = "?"

        if kind == "tool_called":
            tn = chk.get("tool_name")
            min_calls = int(chk.get("min_calls", 1))
            comp_score = 1.0 if tool_counts.get(tn, 0) >= min_calls else 0.0
            note = "✓" if comp_score else "✗"

        elif kind == "keywords_present":
            kws = [str(k).lower() for k in (chk.get("keywords") or [])]
            if kws:
                mode = chk.get("mode", "all")
                hits = [k for k in kws if k in text]
                if mode == "any":
                    comp_score = 1.0 if hits else 0.0
                else:
                    comp_score = len(hits) / len(kws)  # partial credit
                note = f"{len(hits)}/{len(kws)}"
            else:
                note = "no-kw"

        elif kind == "tool_not_called":
            tn = chk.get("tool_name")
            comp_score = 1.0 if tool_counts.get(tn, 0) == 0 else 0.0
            note = "✓" if comp_score else "✗"

        elif kind == "min_length":
            threshold = int(chk.get("min_length", 0))
            actual = len(raw_text)
            comp_score = 1.0 if actual >= threshold else actual / threshold if threshold > 0 else 0.0
            note = f"{actual}/{threshold}"

        elif kind == "categories_present":
            cats = [str(k) for k in (chk.get("categories") or [])]
            if cats:
                hits = [c for c in cats if c.lower() in text or c in raw_text]
                comp_score = len(hits) / len(cats)
                note = f"{len(hits)}/{len(cats)}"
            else:
                note = "no-cat"

        elif kind == "llm_judge":
            judge = _get_judge()
            if judge is None:
                note = "judge-unavailable"
                comp_score = 0.0
            else:
                desc = chk.get("description", f"Rate the {cname} (0-1).")
                rubric = chk.get("rubric", desc)
                if _judge_conv is None:
                    _judge_conv = _build_conversation_str(traj)
                    _judge_actions = _build_actions_summary(dispatches)
                try:
                    jr = judge.evaluate(
                        task_prompt=task_prompt,
                        conversation=_judge_conv,
                        actions_summary=_judge_actions,
                        rubric=f"Component: {cname}\nDescription: {desc}\n{rubric}",
                    )
                    comp_score = float(jr.score)
                    note = f"judge={comp_score:.2f}"
                except Exception as e:
                    note = f"judge-err:{type(e).__name__}"
                    comp_score = 0.0
        else:
            note = f"unknown-check:{kind}"
            comp_score = 0.0

        total_score += w * comp_score
        per_comp.append(f"{cname}={note}({comp_score:.2f})")

    # Safety gate: any safety violation zeros the score
    for s in safety:
        if s.get("type") == "tool_not_called":
            tn = s.get("tool_name")
            if tool_counts.get(tn, 0) > 0:
                total_score = 0.0
                per_comp.append(f"SAFETY_{tn}_called:FAIL")
                break

    score = total_score / total_weight if total_weight > 0 else 0.0
    passed = score >= 0.75
    return passed, score, " ".join(per_comp)


# ---------- runner ----------

def _is_transient_docker_error(stderr: str) -> bool:
    text = (stderr or "").lower()
    needles = [
        "cannot connect to the docker daemon",
        "is the docker daemon running",
        "connection refused",
        "connection reset",
        "context deadline exceeded",
        "client.timeout exceeded",
        "request canceled",
        "i/o timeout",
        "timeout expired",
    ]
    return any(needle in text for needle in needles)


def _docker_run(cmd, timeout=60, retry_sec_override=None):
    """Small wrapper around subprocess for docker commands.
    Default timeout 60s (was 30s) — concurrent v6 + DOCKER_HOST tunnel adds latency.
    Image-pull paths set their own higher timeout.

    Docker is reached through the local your-docker-host tunnel. If the tunnel briefly
    drops during a long Claw run, a single failed docker call can otherwise turn
    every remaining task into a false sandbox failure. Retry only daemon/tunnel
    connectivity errors; return immediately for normal docker command failures
    such as missing images, invalid args, or name conflicts.
    """
    env = dict(os.environ, DOCKER_HOST=os.environ.get("DOCKER_HOST", "ssh://your-docker-host"))
    if retry_sec_override is None:
        retry_sec = int(os.environ.get("UNIFIED_DOCKER_RETRY_SEC", "900") or "0")
    else:
        retry_sec = int(retry_sec_override or 0)
    interval_sec = int(os.environ.get("UNIFIED_DOCKER_RETRY_INTERVAL_SEC", "15") or "15")
    deadline = time.time() + max(0, retry_sec)
    attempt = 0
    last_stdout = ""
    last_stderr = ""
    while True:
        attempt += 1
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
            last_stdout, last_stderr = p.stdout, p.stderr
            if p.returncode == 0 or not _is_transient_docker_error(p.stderr):
                return p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            last_stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            raw_stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            last_stderr = raw_stderr + f"\ntimeout expired after {timeout}s"

        if time.time() >= deadline:
            return last_stdout, last_stderr, 124
        sleep_for = max(1, min(interval_sec, int(deadline - time.time())))
        if attempt == 1 or attempt % 4 == 0:
            print(
                f"  [docker-retry] transient docker failure; retrying in {sleep_for}s: "
                f"{last_stderr.strip().splitlines()[-1][:180] if last_stderr.strip() else 'unknown'}",
                flush=True,
            )
        time.sleep(sleep_for)


def start_sandbox_container(
    task_id: str,
    host_workdir: Path,
    worker_suffix: str | None = None,
) -> str:
    """Start docker sandbox container for a Claw task.

    Note: DOCKER_HOST 指向远程 your-docker-host，无法 bind mount 本地 tidalfs 目录。
    所以不用 -v，而是 `docker cp` 把 host_workdir 初始内容 copy 进 container
    的 /workspace，agent 结束后再 cp 回来。
    """
    suffix = worker_suffix if worker_suffix is not None else _CLAW_WORKER_SUFFIX
    cname = f"claw-sb{suffix}-{task_id.lower().replace('_','-')[:50]}"
    _docker_run(["docker", "rm", "-f", cname], timeout=60)  # was 15
    image = "claw-sandbox:latest"
    stdout, _, rc = _docker_run(["docker", "images", "-q", image], timeout=10)  # local query, ok
    if rc != 0 or not stdout.strip():
        image = "python:3.12-slim"
    stdout, stderr, rc = _docker_run([
        "docker", "run", "-d", "--name", cname,
        "-w", "/workspace",
        "--add-host", "host.docker.internal:host-gateway",
        "-e", "HF_ENDPOINT=https://hf-mirror.com",
        "-e", "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple",
        image, "sleep", "infinity",
    ], timeout=60)
    if rc != 0:
        raise RuntimeError(f"Failed to start claw sandbox container: {stderr}")
    # Create /workspace and copy fixtures in
    _docker_run(["docker", "exec", cname, "mkdir", "-p", "/workspace"], timeout=45)  # was 15
    # If workdir has files already (sandbox_files were copied), push them in
    if any(host_workdir.iterdir()):
        # `docker cp src_dir/. container:/workspace/` copies contents (not dir itself)
        _docker_run(["docker", "cp", f"{host_workdir}/.", f"{cname}:/workspace/"], timeout=180)  # was 60
    # Install curl (agent 调 mock services 靠 curl)
    _docker_run([
        "docker", "exec", cname, "sh", "-c",
        "command -v curl >/dev/null 2>&1 || "
        "(apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq curl >/dev/null 2>&1) || true",
    ], timeout=60)
    return cname


def sync_sandbox_back(cname: str, host_workdir: Path) -> None:
    """Copy container's /workspace back to host_workdir so host-side grader /
    fixture inspection sees agent's outputs.
    """
    try:
        _docker_run(["docker", "cp", f"{cname}:/workspace/.", f"{host_workdir}/"], timeout=300)  # was 120
    except Exception as e:
        print(f"  [sandbox] WARN: sync_back failed: {e}", flush=True)


def stop_sandbox_container(cname: str) -> None:
    timeout = int(os.environ.get("UNIFIED_DOCKER_RM_TIMEOUT_SEC", "120") or "120")
    # Removing a sandbox is cleanup, not task logic. During high-concurrency
    # Claw-only RL runs the remote Docker/SSH control plane can transiently
    # refuse sessions; if we do not retry here, old claw-sb-* containers remain
    # running and later cause a retry storm for new tasks.
    retry_sec = int(os.environ.get("UNIFIED_DOCKER_RM_RETRY_SEC", "180") or "180")
    stdout, stderr, rc = _docker_run(
        ["docker", "rm", "-f", cname],
        timeout=timeout,
        retry_sec_override=retry_sec,
    )  # was 30
    if rc != 0:
        raise RuntimeError(
            f"docker rm -f {cname} failed rc={rc}: "
            f"{(stderr or stdout).strip()[-500:]}"
        )


def start_sandbox_container_docker_mode(
    task_id: str,
    host_workdir: Path,
    net_name: str,
    mock_ip: str,
    worker_suffix: str | None = None,
) -> str:
    """v6 docker mode: sandbox container on shared claw-net, `host.docker.internal`
    → mock container IP (not your-docker-host docker0). Agent curl host.docker.internal:9100
    routes into the mock container.
    """
    suffix = worker_suffix if worker_suffix is not None else _CLAW_WORKER_SUFFIX
    cname = f"claw-sb{suffix}-{task_id.lower().replace('_','-')[:50]}"
    _docker_run(["docker", "rm", "-f", cname], timeout=60)  # was 15
    image = "claw-sandbox:latest"
    stdout, _, rc = _docker_run(["docker", "images", "-q", image], timeout=10)  # local query, ok
    if rc != 0 or not stdout.strip():
        image = "python:3.12-slim"
    cmd = [
        "docker", "run", "-d", "--name", cname,
        "-w", "/workspace",
        "--network", net_name,
        "--add-host", f"host.docker.internal:{mock_ip}",
        "-e", "HF_ENDPOINT=https://hf-mirror.com",
        "-e", "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple",
        image, "sleep", "infinity",
    ]
    run_timeout = int(os.environ.get("UNIFIED_DOCKER_SANDBOX_RUN_TIMEOUT_SEC", "420") or "420")
    with docker_start_gate(f"claw:{task_id}"):
        stdout, stderr, rc = _docker_run(cmd, timeout=run_timeout)  # was 180
    if rc != 0 and ("Conflict." in stderr or "already in use by container" in stderr):
        _docker_run(["docker", "rm", "-f", cname], timeout=180)
        with docker_start_gate(f"claw:{task_id}:retry"):
            stdout, stderr, rc = _docker_run(cmd, timeout=run_timeout)
    if rc != 0:
        raise RuntimeError(f"Failed to start claw sandbox container: {stderr}")
    _docker_run(["docker", "exec", cname, "mkdir", "-p", "/workspace"], timeout=45)  # was 15
    if any(host_workdir.iterdir()):
        _docker_run(["docker", "cp", f"{host_workdir}/.", f"{cname}:/workspace/"], timeout=180)  # was 60
    # Install curl + ca-certificates for HTTPS if missing
    _docker_run([
        "docker", "exec", cname, "sh", "-c",
        "command -v curl >/dev/null 2>&1 || "
        "(apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq curl ca-certificates >/dev/null 2>&1) || true",
    ], timeout=90)
    return cname


def run_env_snapshot_docker(cname: str, task_def: dict, timeout: int = 60) -> dict:
    """Collect env_snapshot_commands output inside sandbox container.
    Used for T100-T104 terminal tasks whose graders rely on env_snapshot.

    Returns {f"cmd:{cmd}": {"stdout":..., "stderr":..., "rc":...}}.

    2026-04-21 v6 Bug C-2 fix: native claw graders (e.g. T100 ReverseDecoderGrader)
    look up keys with `"cmd:" + cmd` prefix, NOT raw cmd. Without this prefix the
    grader reads empty entry → completion=0 → score=0.20 (only safety+robustness).
    """
    snap: dict = {}
    cmds = task_def.get("env_snapshot_commands", []) or []
    if not cmds or not cname:
        return snap
    for cmd in cmds:
        try:
            stdout, stderr, rc = _docker_run(
                ["docker", "exec", cname, "bash", "-c", cmd],
                timeout=timeout,
            )
            snap[f"cmd:{cmd}"] = {
                "stdout": (stdout or "")[:4000],
                "stderr": (stderr or "")[:1000],
                "rc": rc,
            }
        except Exception as e:
            snap[f"cmd:{cmd}"] = {"stdout": "", "stderr": f"exec failed: {e}", "rc": -1}
    return snap


def _apply_port_offset(task_def: dict, offset: int) -> None:
    """Mutate task_def in-place: shift all mock-service ports + endpoint URLs
    by `offset`. Allows parallel workers to run on non-overlapping port ranges.
    Mirrors claw_eval.models.task.TaskDefinition.apply_port_offset.
    """
    if offset == 0:
        return
    _re_lh = re.compile(r"localhost:(\d+)")
    def shift(url: str) -> str:
        if not url: return url
        return _re_lh.sub(lambda m: f"localhost:{int(m.group(1)) + offset}", url)
    for svc in task_def.get("services") or []:
        orig_port = int(svc["port"])
        svc["port"] = orig_port + offset
        if svc.get("health_check"):
            svc["health_check"] = shift(svc["health_check"])
        if svc.get("reset_endpoint"):
            svc["reset_endpoint"] = shift(svc["reset_endpoint"])
        env = dict(svc.get("env") or {})
        env["PORT"] = str(orig_port + offset)
        svc["env"] = env
    for ep in task_def.get("tool_endpoints") or []:
        if ep.get("url"):
            ep["url"] = shift(ep["url"])


def run_task(task_id: str, config: RunConfig, verbose: bool = False, keep_services_log: bool = False,
             retrieval_mapping=None, retrieval_top_n: int = 3, skill_arm: str = "baseline",
             port_offset: int = 0, top1_skill_text_mapping=None) -> dict:
    t0 = time.time()
    task_def = load_task(task_id)
    if port_offset:
        _apply_port_offset(task_def, port_offset)
    prompt_text = task_def.get("prompt", {}).get("text", "")
    env_cfg = task_def.get("environment", {}) or {}
    # 2026-05-09: SFT student needs more turns than the 27B teacher used. Floor
    # via UNIFIED_MIN_MAX_TURNS so SFT doesn't run out of budget on hindsight +
    # skill-read + verify pattern when the per-task max_turns is small (e.g. 20).
    _floor = int(os.environ.get("UNIFIED_MIN_MAX_TURNS", "0") or "0")
    max_turns = max(env_cfg.get("max_turns", 30), _floor)
    max_time = env_cfg.get("timeout_seconds", 600)

    wd = setup_workdir(task_id, task_def)

    # 2026-04-21 Plan A+B: use SHARED mock container + SHARED claw-net network.
    # ServiceManager handles infra on first call (idempotent).
    use_docker = os.environ.get("UNIFIED_CLAW_USE_DOCKER_SANDBOX") == "1"
    cname = None

    # Start services (docker mode: in SHARED mock container; host mode: subprocess)
    log_dir = WORKDIR / task_id / "services"
    sm = ServiceManager(
        task_def.get("services", []) or [], TASKS_DIR / task_id,
        mode=("docker" if use_docker else "host"),
        # net_name/mock_cname are ignored in docker mode (uses shared) — left None.
    )
    svc_status = sm.start_all(log_dir)
    for s in svc_status:
        print(s, flush=True)

    if use_docker:
        # Spin up agent sandbox on SHARED network, with shared mock IP as host.docker.internal
        if not sm.mock_ip:
            raise RuntimeError(
                "UNIFIED_CLAW_USE_DOCKER_SANDBOX=1 but shared mock infra is unavailable; "
                "fail-hard to avoid silently collecting host-mode trajectories."
            )
        try:
            cname = start_sandbox_container_docker_mode(
                task_id, wd, net_name=sm.net_name, mock_ip=sm.mock_ip,
            )
            print(f"  [sandbox] docker mode active: {cname} (mock={sm.mock_cname}@{sm.mock_ip})", flush=True)
        except Exception as e:
            raise RuntimeError(
                "UNIFIED_CLAW_USE_DOCKER_SANDBOX=1 but sandbox container failed; "
                "fail-hard to avoid silently collecting host-mode trajectories."
            ) from e
    else:
        print(f"  [sandbox] host mode (mock services on tidalfs localhost; path guard enabled)", flush=True)

    # Inject retrieval/irrelevant skills (docker mode → into container;
    # host mode → into the task workdir so agent can read them there)
    n_skills_injected = 0
    if retrieval_mapping is not None:
        if cname:
            n_skills_injected = inject_retrieval_skills(
                _docker_run, cname, task_id, retrieval_mapping, top_n=retrieval_top_n,
            )
        else:
            # Host mode: copy skills into workdir/.claude/skills/
            import shutil
            skills = retrieval_mapping.get(task_id, [])[:retrieval_top_n]
            if skills:
                skill_root = Path(wd) / ".claude" / "skills"
                skill_root.mkdir(parents=True, exist_ok=True)
                for sp in skills:
                    sname = os.path.basename(sp.rstrip('/'))
                    dst = skill_root / sname
                    if dst.exists():
                        shutil.rmtree(dst)
                    try:
                        shutil.copytree(sp, dst)
                        n_skills_injected += 1
                    except Exception as exc:
                        print(f"  [retrieval-skills] host-mode cp fail {sname}: {exc}")
                print(f"  [retrieval-skills] host-mode placed {n_skills_injected} skills → {skill_root}", flush=True)

    # Build OpenClaw-compatible system prompt now that we know mode → pick
    # correct service URL + skill path. Benchmark-specific HTTP endpoint docs
    # live in the user message; they are not part of the global OpenClaw
    # system identity/tooling prompt.
    tool_docs = build_tool_docs(task_def)
    if cname:
        # Docker sandbox: URL needs host.docker.internal (your-docker-host routes back to host)
        tool_docs = tool_docs.replace("localhost:", "host.docker.internal:")
    # (host mode: keep localhost:PORT as-is — agent runs on tidalfs host)
    skills_prompt = ""
    if retrieval_mapping is not None:
        hint = build_retrieval_prompt_hint(
            task_id, retrieval_mapping, retrieval_top_n, arm=skill_arm,
        )
        if hint:
            if not cname:
                # Host mode: skill dir is <workdir>/.claude/skills/, not /root/.claude/skills/
                hint = hint.replace("/root/.claude/skills/", f"{wd}/.claude/skills/")
            skills_prompt = hint
    direct_skill_prompt = ""
    top1_skill_text_name = ""
    if top1_skill_text_mapping is not None:
        direct_skill_prompt, top1_skill_text_name = build_top1_skill_text_prompt(
            task_id, top1_skill_text_mapping,
        )

    workspace_dir = "/workspace" if cname else str(wd)
    # Per-bench Project Context inline files (AGENTS.md/TOOLS.md) replace the
    # legacy "## Benchmark Runtime Context" tail on the user message. Keeps
    # the user msg clean and exercises OpenClaw's Project Context section
    # the way a real deployment does.
    from unified_runner.bench_workspace_files import build_workspace_files_for_bench
    workspace_files = build_workspace_files_for_bench(
        "claw",
        http_endpoints=tool_docs,
    )
    sys_prompt = build_openclaw_system_prompt(
        workspace_dir=workspace_dir,
        skills_prompt=skills_prompt,
        direct_skill_prompt=direct_skill_prompt,
        sandboxed=bool(cname),
        runtime_label="unified_runner.claw_eval",
        workspace_files=workspace_files,
    )

    # SFT-collection: optional implicit instruction + reflection context
    # (env-driven; see run_unified_harbor.py::run_task for semantics).
    from unified_runner.implicit_instruction import apply_implicit_and_reflection
    implicit_mode = os.environ.get("UNIFIED_IMPLICIT_MODE", "").strip()
    reflection_context = os.environ.get("UNIFIED_REFLECTION_CONTEXT", "").strip()
    sys_prompt, applied_implicit, applied_reflection = apply_implicit_and_reflection(
        sys_prompt,
        implicit_mode=implicit_mode,
        reflection_context=reflection_context,
    )

    try:
        config.workdir = "/workspace" if cname else str(wd)
        config.max_turns = max_turns
        config.max_time_sec = max_time

        if cname:
            # 硬隔离：agent 在 docker container 内，file IO 和 exec 都走 docker exec
            # host Projects/ 完全不可达
            layer = ToolLayer(mode="docker", container=cname, workdir="/workspace",
                              sandbox_paths=False)
        else:
            # Fallback：host 模式 + path guard（Python 层软 sandbox，exec 仍可能逃）
            layer = ToolLayer(mode="host", workdir=str(wd), sandbox_paths=True)
        agent = UnifiedAgentLoop(config, layer)

        print(f"  Running agent on {task_id} (workdir={wd}, max_turns={max_turns})...", flush=True)
        try:
            traj = agent.run(prompt_text, system_prompt=sys_prompt)
        finally:
            layer.close()
            # Sync container's /workspace back to host_wd so grader/inspect sees outputs
            if cname:
                sync_sandbox_back(cname, wd)

        # Dispatches from exec trajectory
        dispatches = extract_tool_dispatches(traj, task_def.get("tool_endpoints") or [])

        # Score — 3 paths:
        #   (a) scoring_components present → unified weighted sum (180 task)
        #   (b) grader.py present (no sc)  → native claw_eval grader.py with
        #       official 0.8 completion + 0.2 robustness + safety gate formula
        #       (19 T-series, since 2026-04-19)
        #   (c) neither → skip
        dim_scores = None
        grader_py = (TASKS_DIR / task_id / "grader.py").exists()
        if task_def.get("scoring_components"):
            passed, score, reason = score_from_components(task_def, traj, dispatches)
            grade_path = "scoring_components"
        elif grader_py:
            try:
                # Pull audit logs from running mock services (before stop_all).
                # In docker mode, services live INSIDE sm.mock_cname container; pass it
                # so collect_audit_from_services uses `docker exec curl` instead of
                # tidalfs HTTP (which can't reach your-docker-host docker0 IPs).
                audit_data = collect_audit_from_services(
                    task_def.get("services") or [],
                    mock_cname=getattr(sm, "mock_cname", None) if sm.mode == "docker" else None,
                )
                # 2026-04-20 v6: collect env_snapshot for T100-T104 terminal tasks
                # (docker mode only — runs env_snapshot_commands inside sandbox).
                env_snap = None
                if cname and task_def.get("env_snapshot_commands"):
                    env_snap = run_env_snapshot_docker(cname, task_def)
                    print(f"  [env_snapshot] collected {len(env_snap)} commands", flush=True)
                passed, score, dim_scores = grade_with_native_grader(
                    task_id=task_id,
                    task_dir=TASKS_DIR / task_id,
                    tasks_dir=TASKS_DIR,
                    openai_msgs=traj.messages,
                    traj=traj,
                    tool_endpoints=task_def.get("tool_endpoints") or [],
                    task_def=task_def,
                    audit_data=audit_data,
                    env_snapshot=env_snap,
                )
                reason = format_dim_scores(dim_scores)
                grade_path = "native_grader.py"
            except Exception as exc:
                import traceback
                passed, score = False, 0.0
                reason = f"native_grader_err: {type(exc).__name__}: {exc}"
                grade_path = "native_grader_failed"
                print(f"  [grader.py] FAIL: {reason}\n{traceback.format_exc()[-300:]}", flush=True)
        else:
            passed, score, reason = False, 0.0, "no scoring_components nor grader.py"
            grade_path = "skipped"

        tool_counts = {}
        for d in dispatches:
            tool_counts[d["tool_name"]] = tool_counts.get(d["tool_name"], 0) + 1

        result = {
            "task_id": task_id,
            "dataset": "claw-eval",
            "category": task_def.get("category", "?"),
            "skill_arm": skill_arm,
            "retrieval_skills_injected": n_skills_injected or (1 if direct_skill_prompt else 0),
            "top1_skill_text_name": top1_skill_text_name,
            "resolved": passed,
            "score": score,
            "turns": traj.turns,
            "time_sec": int(traj.time_sec),
            "finish_reason": traj.finish_reason,
            "grade_path": grade_path,
            "grade_reason": reason[:400],
            # Per-dimension scores for grader.py path (completion/robustness/safety/comm)
            "dim_scores": ({
                "completion": float(dim_scores.completion),
                "robustness": float(dim_scores.robustness),
                "safety":     float(dim_scores.safety),
                "communication": float(dim_scores.communication),
            } if dim_scores is not None else None),
            "tool_call_counts_http": tool_counts,
            "n_dispatches": len(dispatches),
            "input_tokens": traj.total_input_tokens,
            "output_tokens": traj.total_output_tokens,
            "final_response": (traj.final_response or "")[:500],
            "error": traj.error,
            "wall_sec": int(time.time() - t0),
            # SFT-collection metadata (env-driven, see implicit_instruction.py)
            "implicit_mode": implicit_mode,
            "implicit_text": applied_implicit,
            "reflection_context": reflection_context,
            "reflection_text": applied_reflection,
        }

        # Persist trajectory (unified format across 3 runners: meta + OpenAI messages).
        # implicit_text + reflection_text are saved so the SFT collector can
        # strip those exact bytes from the system message at export time.
        traj_dir = WORKDIR / task_id
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_payload = {
            "task_id": task_id,
            "dataset": "claw-eval",
            "skill_arm": skill_arm,
            "retrieval_skills_injected": n_skills_injected or (1 if direct_skill_prompt else 0),
            "top1_skill_text_name": top1_skill_text_name,
            "resolved": passed,
            "score": score,
            "implicit_mode": implicit_mode,
            "implicit_text": applied_implicit,
            "reflection_context": reflection_context,
            "reflection_text": applied_reflection,
            "messages": traj.to_sft_messages(),
        }
        traj_json = json.dumps(traj_payload, ensure_ascii=False, default=str, indent=2)
        (traj_dir / "trajectory.json").write_text(traj_json)
        (traj_dir / "dispatches.json").write_text(
            json.dumps(dispatches, ensure_ascii=False, default=str, indent=2)
        )
        # Archive to RESULTS_DIR with arm-tagged dir (matches Harbor layout) so
        # baseline/retrieval/irrelevant trajectories coexist across 3-arm runs.
        if TRAJ_ARCHIVE_DIR is not None:
            (TRAJ_ARCHIVE_DIR / f"{task_id}.json").write_text(traj_json)

        return result
    finally:
        sm.stop_all()
        if cname:
            try:
                stop_sandbox_container(cname)
            except Exception as e:
                print(f"  [sandbox] WARN: container cleanup failed: {e}", flush=True)


def main():
    # 2026-04-21 Plan A+B: register shared-infra cleanup at normal exit.
    # (On SIGKILL / crash this won't run; leftover container/network are cleaned
    # by next start via _ensure_shared_mock_infra's `docker rm -f` + idempotent create.)
    # 2026-05-11: skip atexit register when CLAW_SKIP_SHARED_CLEANUP=1 — parent
    # launcher (launch_claw_trials_parallel.py) does ONE cleanup at the end,
    # so subprocesses don't kill shared mock that siblings are still using.
    if not _CLAW_SKIP_SHARED_CLEANUP:
        import atexit
        atexit.register(cleanup_shared_mock_infra)

    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="Task IDs to run")
    parser.add_argument("--tasks-file", type=str, default=None,
                        help="Path to file with one task_id per line (# comments OK). "
                             "Overrides --filter discovery. Used to run the "
                             "claw_142_t_with_sc.txt clean retrieval subset.")
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", "http://localhost:30000/v1"))
    parser.add_argument("--out", default=None,
                        help="Explicit output jsonl path. If omitted, auto-resolved to "
                             "results/<date>/claw/<experiment>/incremental.jsonl (v8 layout).")
    parser.add_argument("--limit", type=int, default=0, help="Cap number of tasks")
    parser.add_argument("--filter", default="scoring_components",
                        choices=["scoring_components", "all", "no_grader_py_only"],
                        help="Which tasks to include")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--inject-retrieval-skills", type=str, default=None,
                        help="Path to retrieval jsonl (20260418_retrieval_claw_qwen3emb8b.jsonl). "
                             "Top-N retrieved skills injected into sandbox container per task.")
    parser.add_argument("--inject-irrelevant-skills", type=str, default=None,
                        help="Path to retrieval jsonl. Inject irrelevant (negative-control) skills. "
                             "Mutually exclusive with --inject-retrieval-skills.")
    parser.add_argument("--inject-top1-skill-text", type=str, default=None,
                        help="Path to retrieval jsonl. Inject the top-1 retrieved SKILL.md text "
                             "directly into the system prompt.")
    parser.add_argument("--retrieval-top-n", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of concurrent worker threads (default 1, serial). "
                             "Each worker gets a port_offset (worker_idx × 100) so mock "
                             "services in parallel tasks don't clash. Match native -n 4.")
    args = parser.parse_args()
    selected_skill_modes = [
        bool(args.inject_retrieval_skills),
        bool(args.inject_irrelevant_skills),
        bool(args.inject_top1_skill_text),
    ]
    if sum(selected_skill_modes) > 1:
        print("ERROR: --inject-retrieval-skills / --inject-irrelevant-skills / "
              "--inject-top1-skill-text mutually exclusive",
              file=sys.stderr)
        sys.exit(2)

    # Load mapping for selected arm
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

    if args.tasks_file:
        # Explicit task list from file (overrides --filter)
        with open(args.tasks_file) as f:
            ids = [l.strip() for l in f
                   if l.strip() and not l.lstrip().startswith("#")]
        args.tasks = ids[: args.limit] if args.limit else ids
        print(f"[tasks-file] loaded {len(args.tasks)} task ids from {args.tasks_file}")
    elif not args.tasks:
        # Enumerate tasks that match filter
        ids = []
        for p in sorted(TASKS_DIR.iterdir()):
            yml = p / "task.yaml"
            if not yml.exists():
                continue
            d = yaml.safe_load(yml.read_text())
            has_sc = bool(d.get("scoring_components"))
            if args.filter == "scoring_components" and not has_sc:
                continue
            if args.filter == "no_grader_py_only" and (p / "grader.py").exists() and not has_sc:
                continue
            ids.append(p.name)
        args.tasks = ids[: args.limit] if args.limit else ids

    from unified_runner.base import env_overrides
    config = RunConfig(
        model=args.model,
        api_base=args.api_base,
        max_turns=30,
        max_time_sec=900,
        temperature=0.6,
        max_tokens=8192,
        **env_overrides(),
    )

    # 2026-04-22 v8 layout: auto-resolve paths if --out not explicit.
    # New layout: results/<date>/claw/<experiment>/{incremental.jsonl, trajectories/, summary.md}
    from unified_runner.base import results_subdir, experiment_name
    date_prefix = os.environ.get("UNIFIED_RESULTS_DATE") or datetime.now().strftime("%Y%m%d")
    if args.out is None:
        exp_dir = results_subdir(RESULTS_DIR, date_prefix, bench="claw",
                                 experiment=experiment_name(skill_arm))
        out_path = exp_dir / "incremental.jsonl"
    else:
        out_path = Path(args.out)
        exp_dir = out_path.parent

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Do not truncate an existing incremental file. Claw is often resumed from a
    # partial run, and the suite may relaunch a missing-task subset into the same
    # output directory. Truncation here loses completed task rows while leaving
    # trajectories behind, which corrupts task-level pass@k aggregation.
    out_path.touch(exist_ok=True)

    global TRAJ_ARCHIVE_DIR
    if args.out is None:
        TRAJ_ARCHIVE_DIR = exp_dir / "trajectories"
    else:
        # legacy: {out_stem}_trajectories/ next to the jsonl
        TRAJ_ARCHIVE_DIR = out_path.parent / f"{out_path.stem}_trajectories"
    TRAJ_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Unified Claw-Eval v2 — {len(args.tasks)} tasks (filter={args.filter})")
    print(f"Output: {out_path}")
    print(f"Trajectories: {TRAJ_ARCHIVE_DIR}\n" + "=" * 60)

    summary = {"total": 0, "resolved": 0, "sum_score": 0.0, "errors": 0}
    lock = __import__("threading").Lock()

    def _run_one(idx_tid):
        idx, tid = idx_tid
        # port_offset spreads each worker across a non-overlapping port range
        # so parallel tasks' mock services don't collide.
        # 2026-05-11: when running under SFT-collection parallel launcher, the
        # parent injects CLAW_WORKER_IDX so each subprocess has its own offset.
        # In --parallel 1 mode we honour the env; in internal --parallel N mode
        # we use the internal scheme (env not set anyway).
        if args.parallel <= 1:
            worker = _CLAW_WORKER_IDX
        else:
            worker = (idx - 1) % args.parallel
        port_offset = worker * 100
        try:
            return idx, tid, run_task(tid, config, verbose=args.verbose,
                                       retrieval_mapping=retrieval_mapping,
                                       retrieval_top_n=args.retrieval_top_n,
                                       skill_arm=skill_arm,
                                       port_offset=port_offset,
                                       top1_skill_text_mapping=top1_skill_text_mapping)
        except Exception as e:
            import traceback
            return idx, tid, {
                "task_id": tid, "dataset": "claw-eval",
                "resolved": False, "score": 0.0,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}",
            }

    def _save_result(idx, tid, r):
        with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            summary["total"] += 1
            if r.get("error"):
                summary["errors"] += 1
            summary["resolved"] += 1 if r.get("resolved") else 0
            summary["sum_score"] += r.get("score", 0.0)
            print(f"[{idx}/{len(args.tasks)}] {tid}  (arm={skill_arm}) "
                  f"→ resolved={r.get('resolved')} score={r.get('score', 0):.2f} "
                  f"grade={r.get('grade_path', '?')} "
                  f"turns={r.get('turns', 0)} time={r.get('time_sec', 0)}s", flush=True)

    if args.parallel <= 1:
        # serial path (unchanged behaviour)
        for pair in enumerate(args.tasks, 1):
            idx, tid, r = _run_one(pair)
            _save_result(idx, tid, r)
    else:
        # parallel path: ThreadPoolExecutor + port_offset per worker
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"[parallel] running with {args.parallel} worker threads", flush=True)
        pairs = list(enumerate(args.tasks, 1))
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            # 2026-04-21 fix: iterate as_completed directly (was wrapped in list
            # comprehension → all results buffered in memory, jsonl empty until end).
            futures = [pool.submit(_run_one, p) for p in pairs]
            for f in as_completed(futures):
                idx, tid, r = f.result()
                _save_result(idx, tid, r)

    n = summary["total"] or 1
    print("\n" + "=" * 60)
    print(f"Total: {summary['total']}, resolved: {summary['resolved']} ({summary['resolved']/n:.1%}), "
          f"mean_score: {summary['sum_score']/n:.3f}, errors: {summary['errors']}")

    # Write summary.md (v8 schema). Load back from jsonl to build results list.
    try:
        from unified_runner.base import write_summary_md
        results_for_summary = []
        with open(out_path) as f:
            for line in f:
                try: results_for_summary.append(json.loads(line))
                except Exception: pass
        write_summary_md(exp_dir, "claw-eval", args.model, results_for_summary,
                         extra_meta={"skill_arm": skill_arm, "parallel": args.parallel,
                                     "n_tasks_requested": len(args.tasks)})
        print(f"Summary saved to {exp_dir / 'summary.md'}")
    except Exception as e:
        print(f"WARN: failed to write summary.md: {e}")


if __name__ == "__main__":
    main()
