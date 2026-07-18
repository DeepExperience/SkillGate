"""Unified OpenClaw tool execution layer.

Supports two execution modes:
  - host:   long-lived bash on the host machine
  - docker: long-lived ``docker exec -i container bash`` into a container

Key change vs the previous stateless implementation: each ToolLayer instance
owns ONE persistent bash process (PersistentShell). All exec/grep/find/ls/
apply_patch calls go through the same shell, so cwd / env / aliases / source /
set options persist across calls — matching how terminus-2's tmux session
behaves.

Usage:
    layer = ToolLayer(mode="host")               # or mode="docker", container="..."
    try:
        result = layer.dispatch("exec", {"command": "ls -la"})
        ...
    finally:
        layer.close()
"""

from __future__ import annotations

import base64
import os
import queue
import shlex
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

# Maximum output size before truncation (chars)
MAX_OUTPUT = 16_000
DEFAULT_EXEC_TIMEOUT = 120
# 2026-04-26: bumped 10 → 30 after observing 22/89 tb2 trials timing out on
# `docker exec -i bash` init. Root cause was the remote dockerd slowed by 112
# leaked mysql_<pid> containers from claw mock infra (docker exec is O(N)
# in container count). 10s was too tight; 30s tolerates a moderate dockerd
# slowdown without artificially capping tasks at startup.
SHELL_INIT_TIMEOUT = 30


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2 - 50
    omitted = len(text) - limit
    return text[:half] + f"\n\n... [{omitted} chars truncated] ...\n\n" + text[-half:]


# ---------------------------------------------------------------------------
# Persistent shell
# ---------------------------------------------------------------------------


class PersistentShell:
    """A long-lived bash process with sentinel-delimited command boundaries.

    Why this exists: ``subprocess.run(...)`` per command is stateless — every
    call gets a fresh shell with no cwd / env / aliases. terminus-2's tmux
    session keeps a single bash alive so ``cd /tmp`` then ``pwd`` returns
    ``/tmp``; this class replicates that for the unified agent_loop.

    Implementation:
      - Open ``bash`` (or ``docker exec -i ... bash``) with stdin/stdout pipes.
      - stderr is merged into stdout (matches tmux screen semantics; simpler
        than a second drain thread).
      - A reader thread pushes stdout lines into a thread-safe queue.
      - Each ``exec`` writes ``<cmd>\\n__rc=$?; printf 'SENTINEL:%d\\n' "$__rc"\\n``
        and waits for the sentinel line; everything before is the output.
      - Sentinel is per-call UUID — collision with command output is impossible
        in practice.
      - On timeout, the shell is killed and restarted (state is lost — same
        cost as terminus-2 hitting tmux session timeout).
    """

    def __init__(
        self,
        mode: str = "host",
        container: str | None = None,
        workdir: str = "/workspace",
        docker_env: dict[str, str] | None = None,
    ) -> None:
        assert mode in ("host", "docker"), f"Unknown mode: {mode}"
        if mode == "docker":
            assert container, "container name required in docker mode"
        self.mode = mode
        self.container = container
        self.workdir = workdir
        self._docker_env = docker_env or dict(
            os.environ,
            DOCKER_HOST=os.environ.get("DOCKER_HOST", "ssh://your-docker-host"),
        )
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._lock = threading.Lock()
        self._start()

    # --- lifecycle ---------------------------------------------------------

    def _start(self) -> None:
        # User directive: when docker is congested, QUEUE not fail. Don't
        # count docker-overload failures as model score=0. So we retry
        # generously (up to 30 attempts, growing backoff capped at 20s).
        # asyncio loop is unblocked because rollout.py now wraps env.reset
        # in asyncio.to_thread, so this thread can sleep without stalling
        # other rollouts.
        last_exc = None
        for attempt in range(1, 31):
            try:
                self._start_once()
                return
            except RuntimeError as exc:
                last_exc = exc
                if attempt == 30:
                    raise
                # Backoff: 1, 1, 2, 2, ..., capped at 20s. Total worst case
                # ~250s, which is fine — other rollouts run concurrently.
                time.sleep(min(1.0 + attempt * 0.5, 20.0))
        if last_exc is not None:
            raise last_exc

    def _start_once(self) -> None:
        if self.mode == "docker":
            cmd = ["docker", "exec", "-i", "-w", self.workdir, self.container, "bash"]
            env = self._docker_env
        else:
            cmd = ["bash"]
            env = os.environ.copy()
            # Host mode safety net: point $HOME and $TMPDIR at the task workdir
            # so task prompts that say "save to ~/report/" or "to /tmp/out/"
            # land inside workdir rather than polluting the user's real home
            # (which for this runner is the repository tree on networked storage).
            env["HOME"] = self.workdir
            env["TMPDIR"] = self.workdir
            env["PWD"] = self.workdir

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=self.workdir if self.mode == "host" else None,
            bufsize=0,
        )
        # Drain queue from any previous shell instance.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        # Quiet the prompt and disable history/job-control noise. Use a
        # sentinel here too so we know init is done.
        init_sentinel = f"__INIT_{uuid.uuid4().hex}__"
        init_cmds = (
            "export PS1=''\n"
            "export PS2=''\n"
            "export PROMPT_COMMAND=''\n"
            "set +o history 2>/dev/null || true\n"
            "set +m 2>/dev/null || true\n"
            f"echo '{init_sentinel}'\n"
        )
        try:
            self._proc.stdin.write(init_cmds.encode("utf-8"))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"Failed to init persistent shell: {exc}")

        # Drain init output up to the init sentinel.
        deadline = time.time() + SHELL_INIT_TIMEOUT
        buf = bytearray()
        while time.time() < deadline:
            try:
                line = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._proc.poll() is not None:
                    raise RuntimeError("Persistent shell died during init")
                continue
            buf.extend(line)
            if init_sentinel.encode() in buf:
                return
        raise RuntimeError(
            f"Persistent shell init timed out after {SHELL_INIT_TIMEOUT}s"
        )

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            while True:
                chunk = proc.stdout.readline()
                if not chunk:
                    break
                self._queue.put(chunk)
        except Exception:
            pass

    def _kill_and_restart(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            except Exception:
                pass
        self._proc = None
        try:
            self._start()
        except Exception:
            pass

    def _cleanup_timeout_children(self) -> None:
        """Best-effort cleanup for commands orphaned by docker exec timeout.

        Killing the host-side ``docker exec`` process can leave long-running
        grandchildren alive inside the task container. In practice the harmful
        case is TB2 agents repeatedly trying to ``pip install torch`` or
        recursively grepping the entire filesystem for torch parallelism
        symbols after our command timeout fires; those orphaned processes keep
        consuming network/IO and can hold rollout tail latency open. Keep this
        cleanup narrow so we do not disturb normal task processes.
        """
        if self.mode != "docker" or not self.container:
            return
        if os.environ.get("UNIFIED_TOOL_TIMEOUT_CHILD_CLEANUP", "1").lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return
        cleanup_cmd = (
            "pkill -TERM -f 'pip3? install .*torch|pip .*install .*torch|"
            "python3? .*(/usr/bin/)?pip3? install .*torch|"
            "grep -RIl .*ColumnParallelLinear|grep -RIl .*RowParallelLinear|"
            "grep -RIl .*ColwiseParallel|grep -RIl .*RowwiseParallel' || true; "
            "sleep 0.2; "
            "pkill -KILL -f 'pip3? install .*torch|pip .*install .*torch|"
            "python3? .*(/usr/bin/)?pip3? install .*torch|"
            "grep -RIl .*ColumnParallelLinear|grep -RIl .*RowParallelLinear|"
            "grep -RIl .*ColwiseParallel|grep -RIl .*RowwiseParallel' || true"
        )
        try:
            subprocess.run(
                ["docker", "exec", self.container, "sh", "-lc", cleanup_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                env=self._docker_env,
            )
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            if self._proc is None:
                return
            try:
                if self._proc.poll() is None:
                    self._proc.stdin.write(b"exit\n")
                    self._proc.stdin.flush()
                    try:
                        self._proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self._proc.terminate()
                        try:
                            self._proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            self._proc.kill()
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    # --- main API ----------------------------------------------------------

    def exec(self, command: str, timeout: int = DEFAULT_EXEC_TIMEOUT) -> tuple[str, str, int]:
        """Run a command in the persistent shell. Returns (stdout, stderr, rc).

        stderr is always empty — it's merged into stdout (matches tmux behavior).
        """
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                try:
                    self._start()
                except Exception as exc:
                    return "", f"shell restart failed: {exc}", -1

            sentinel = f"__END_{uuid.uuid4().hex}__"
            full_cmd = (
                f"{command}\n"
                f"__rc=$?; printf '\\n%s:%d\\n' '{sentinel}' \"$__rc\"\n"
            )

            try:
                self._proc.stdin.write(full_cmd.encode("utf-8"))
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._kill_and_restart()
                return "", f"shell write failed: {exc}", -1

            sentinel_bytes = sentinel.encode("utf-8")
            buf = bytearray()
            deadline = time.time() + timeout

            while time.time() < deadline:
                try:
                    line = self._queue.get(timeout=0.5)
                except queue.Empty:
                    if self._proc.poll() is not None:
                        out = buf.decode("utf-8", errors="replace")
                        self._kill_and_restart()
                        return out, "shell died", -1
                    continue

                buf.extend(line)
                if sentinel_bytes in buf:
                    text = buf.decode("utf-8", errors="replace")
                    idx = text.rfind(sentinel)
                    # parse "<sentinel>:<rc>\n"
                    rc_part = text[idx + len(sentinel):].lstrip(":").strip()
                    rc_str = rc_part.split("\n", 1)[0].strip()
                    try:
                        rc = int(rc_str)
                    except ValueError:
                        rc = -1
                    output = text[:idx].rstrip("\n").rstrip("\r")
                    return output, "", rc

            # timeout — kill so the shell doesn't keep accumulating
            self._cleanup_timeout_children()
            self._kill_and_restart()
            return buf.decode("utf-8", errors="replace"), f"timeout after {timeout}s", -1


# ---------------------------------------------------------------------------
# Tool layer
# ---------------------------------------------------------------------------


class ToolLayer:
    """Unified tool execution layer with host/docker support.

    When `sandbox_paths=True` (default for host mode), all file IO tools
    (read/write/edit/apply_patch/grep/find/ls) reject absolute paths outside
    `workdir`. This prevents host-mode Claw/unified runs from polluting or
    deleting files in /mnt/... etc. NOTE: `exec` tool can still use bash to
    escape (cd into absolute path + write) — for *hard* isolation, use
    docker mode (ToolLayer(mode='docker', container=..., workdir='/workspace')).
    """

    def __init__(
        self,
        mode: str = "host",
        container: str | None = None,
        workdir: str = "/workspace",
        docker_env: dict[str, str] | None = None,
        sandbox_paths: bool = True,
    ) -> None:
        assert mode in ("host", "docker"), f"Unknown mode: {mode}"
        self.mode = mode
        self.container = container
        self.workdir = workdir
        self._docker_env = docker_env or dict(
            os.environ, DOCKER_HOST=os.environ.get("DOCKER_HOST", "ssh://your-docker-host")
        )
        self._shell_impl = PersistentShell(mode, container, workdir, self._docker_env)
        self._bg_procs: dict[int, subprocess.Popen] = {}
        # sandbox_paths: host mode 下把 file IO 限制在 workdir 及其子路径。
        # docker mode 不需要（容器本身隔离）。
        self._sandbox_paths = sandbox_paths and mode == "host"
        self._workdir_real = (
            os.path.realpath(workdir) if self._sandbox_paths else None
        )

    def _guard_path(self, path: str) -> str:
        """Reject absolute path outside workdir. Return normalized path for use.

        Relative paths 解释为相对 workdir，不改变。
        Absolute paths 做 realpath 后必须在 _workdir_real 下，否则抛错。
        """
        if not self._sandbox_paths:
            return path
        if not os.path.isabs(path):
            # relative — _file_write/_file_read etc use Path() which resolves
            # relative to Python process cwd. In host mode shell cwd is workdir,
            # but Python Path doesn't use shell cwd. Explicitly prepend workdir.
            candidate = os.path.join(self.workdir, path)
        else:
            candidate = path
        real = os.path.realpath(candidate)
        if not (real == self._workdir_real or real.startswith(self._workdir_real + os.sep)):
            raise PermissionError(
                f"sandbox violation: path {path!r} resolves to {real!r} "
                f"which is outside workdir {self._workdir_real!r}"
            )
        return candidate

    def close(self) -> None:
        """Tear down the persistent shell. Safe to call multiple times."""
        try:
            self._shell_impl.close()
        except Exception:
            pass
        for pid, proc in list(self._bg_procs.items()):
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        self._bg_procs.clear()

    def __enter__(self) -> "ToolLayer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- public dispatch ----------------------------------------------------

    _HANDLERS = {
        "read": "_handle_read",
        "write": "_handle_write",
        "edit": "_handle_edit",
        "apply_patch": "_handle_apply_patch",
        "grep": "_handle_grep",
        "find": "_handle_find",
        "ls": "_handle_ls",
        "exec": "_handle_exec",
        "process": "_handle_process",
        "web_fetch": "_handle_web_fetch",
        "web_search": "_handle_web_search",
    }

    # OpenClaw tools that are advertised in the openclaw_full prompt but not
    # implemented in this benchmark runtime. Each maps to a short suggestion
    # the agent can use as a fallback. When the model calls one, it gets a
    # clear "not implemented" error so it can route around the missing tool.
    _UNIMPLEMENTED_OPENCLAW_TOOLS: dict[str, str] = {
        "browser": "Browser automation is not available in this benchmark runtime. Use exec with curl + a headless tool such as `wget`/`pyppeteer` only when the task explicitly needs page rendering; otherwise prefer web_fetch.",
        "canvas": "Canvas tool is not available in this benchmark runtime; the task does not require Canvas presentation.",
        "nodes": "Paired-node tools are not available in this benchmark runtime; act on the local workspace directly.",
        "cron": "Cron/wake events are not available in this benchmark runtime; perform the task in this turn instead of scheduling.",
        "message": "Cross-channel messaging is not available in this benchmark runtime; reply directly in this session instead.",
        "gateway": "Gateway/config tools are not available in this benchmark runtime; do not attempt OpenClaw self-management actions.",
        "agents_list": "Sub-agent orchestration is not available in this benchmark runtime; complete the task directly in this session.",
        "sessions_list": "Sub-agent / session listing is not available in this benchmark runtime.",
        "sessions_history": "Cross-session history is not available in this benchmark runtime.",
        "sessions_send": "Cross-session messaging is not available in this benchmark runtime; reply directly.",
        "subagents": "Sub-agent management is not available in this benchmark runtime; complete the task directly.",
        "session_status": "Status card is not available in this benchmark runtime; current model and time are fixed for this run.",
        "dir_fetch": "Node-based directory transfer is not available in this benchmark runtime; use exec(`ls -la <path>`) or write_then_inspect locally.",
        "dir_list": "Node-based directory listing is not available in this benchmark runtime; use exec(`ls -la <path>`) instead.",
        "file_fetch": "Node-based file transfer is not available in this benchmark runtime; use read(<path>) or exec(`cat <path>`) locally.",
        "file_write": "Node-based file write is not available in this benchmark runtime; use write(path=..., content=...) on the local workspace.",
        "memory_get": "Persistent memory tools are not available in this benchmark runtime; reason from the current conversation context.",
        "memory_search": "Persistent memory tools are not available in this benchmark runtime; reason from the current conversation context.",
        "sessions_spawn": "Sub-agent spawning is not available in this benchmark runtime; complete the task directly in this session.",
        "sessions_yield": "Session yield is not available in this benchmark runtime; finalize your reply in this session.",
        "tts": "Text-to-speech is not available in this benchmark runtime; respond with text only.",
    }

    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler_name = self._HANDLERS.get(tool_name)
        if handler_name is not None:
            handler = getattr(self, handler_name)
            try:
                return handler(arguments)
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}"}
        # Advertised-but-unimplemented OpenClaw tool: return a clear,
        # actionable error so the agent learns to fall back. We do NOT raise;
        # the agent loop treats the tool result as observation and retries.
        unimpl_hint = self._UNIMPLEMENTED_OPENCLAW_TOOLS.get(tool_name)
        if unimpl_hint is not None:
            return {
                "error": (
                    f"Tool '{tool_name}' is advertised by the OpenClaw prompt but "
                    f"is not implemented in this benchmark runtime. {unimpl_hint}"
                ),
                "tool": tool_name,
                "implemented": False,
            }
        return {"error": f"Unknown tool: {tool_name}"}

    # --- internal shell helper ----------------------------------------------

    def _shell(
        self,
        command: str,
        timeout: int = DEFAULT_EXEC_TIMEOUT,
        workdir: str | None = None,
    ) -> tuple[str, str, int]:
        """Run a command in the persistent shell.

        ``workdir`` is honored only as a one-shot hint via ``(cd X && cmd)``
        subshell — the persistent shell's own cwd is NOT changed.
        """
        if workdir:
            command = f"(cd {shlex.quote(workdir)} && {{ {command}; }})"
        return self._shell_impl.exec(command, timeout=timeout)

    def _file_read(self, path: str) -> str:
        if self.mode == "docker":
            stdout, stderr, rc = self._shell(f"cat {shlex.quote(path)}", timeout=30)
            if rc != 0:
                raise FileNotFoundError(stderr.strip() or f"Cannot read {path}")
            return stdout
        path = self._guard_path(path)
        return Path(path).read_text(encoding="utf-8", errors="replace")

    def _file_write(self, path: str, content: str) -> None:
        if self.mode == "docker":
            b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
            cmd = (
                f"mkdir -p $(dirname {shlex.quote(path)}) && "
                f"echo {shlex.quote(b64)} | base64 -d > {shlex.quote(path)}"
            )
            _, stderr, rc = self._shell(cmd, timeout=30)
            if rc != 0:
                raise IOError(f"Write failed: {stderr.strip()}")
        else:
            path = self._guard_path(path)
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

    def _file_exists(self, path: str) -> bool:
        if self.mode == "docker":
            _, _, rc = self._shell(f"test -e {shlex.quote(path)}", timeout=10)
            return rc == 0
        try:
            path = self._guard_path(path)
        except PermissionError:
            return False
        return Path(path).exists()

    # --- tool handlers ------------------------------------------------------

    def _handle_read(self, args: dict) -> dict:
        path = args["path"]
        content = self._file_read(path)
        offset = args.get("offset")
        limit = args.get("limit")
        lines = content.splitlines(keepends=True)
        start = (offset - 1) if offset and offset >= 1 else 0
        end = (start + limit) if limit else len(lines)
        selected = lines[start:end]
        numbered = [f"  {i}\t{line.rstrip()}" for i, line in enumerate(selected, start=start + 1)]
        return {"content": _truncate("\n".join(numbered)), "total_lines": len(lines)}

    def _handle_write(self, args: dict) -> dict:
        path = args["path"]
        content = args["content"]
        self._file_write(path, content)
        return {"written": path, "bytes": len(content.encode("utf-8"))}

    def _handle_edit(self, args: dict) -> dict:
        path = args["path"]
        content = self._file_read(path)
        if "edits" in args:
            edits = args.get("edits") or []
        else:
            # Backward-compatible execution for trajectories collected before
            # the OpenClaw schema migration.
            edits = [
                {
                    "oldText": args.get("old_string"),
                    "newText": args.get("new_string"),
                    "replaceAll": args.get("replace_all", False),
                }
            ]
        if not isinstance(edits, list) or not edits:
            return {"error": "edits must be a non-empty list"}
        new_content = content
        total_replacements = 0
        for edit in edits:
            if not isinstance(edit, dict):
                return {"error": "each edit must be an object"}
            old_string = edit.get("oldText")
            new_string = edit.get("newText")
            replace_all = bool(edit.get("replaceAll", False))
            if not isinstance(old_string, str) or old_string == "":
                return {"error": "edit.oldText must be a non-empty string"}
            if not isinstance(new_string, str):
                return {"error": "edit.newText must be a string"}
            count = new_content.count(old_string)
            if count == 0:
                return {"error": f"oldText not found in {path}"}
            if count > 1 and not replace_all:
                return {"error": f"oldText found {count} times in {path}; make it more specific"}
            new_content = (
                new_content.replace(old_string, new_string)
                if replace_all
                else new_content.replace(old_string, new_string, 1)
            )
            total_replacements += count if replace_all else 1
        self._file_write(path, new_content)
        return {"edited": path, "replacements": total_replacements}

    def _handle_apply_patch(self, args: dict) -> dict:
        patch_content = args.get("input", args.get("patch", ""))
        if not isinstance(patch_content, str) or not patch_content.strip():
            return {"error": "input is required"}
        if patch_content.lstrip().startswith("*** Begin Patch"):
            return self._handle_openclaw_apply_patch(patch_content)
        strip = args.get("strip", 1)
        if self.mode == "docker":
            b64 = base64.b64encode(patch_content.encode("utf-8")).decode("ascii")
            cmd = f"echo {shlex.quote(b64)} | base64 -d | patch -p{strip} --no-backup-if-mismatch"
        else:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False)
            tmp.write(patch_content)
            tmp.close()
            cmd = f"patch -p{strip} --no-backup-if-mismatch < {shlex.quote(tmp.name)}"
        stdout, stderr, rc = self._shell(cmd, timeout=30)
        output = stdout
        if stderr:
            output += "\n" + stderr
        return {"exit_code": rc, "output": _truncate(output)}

    def _handle_openclaw_apply_patch(self, patch_content: str) -> dict:
        """Small OpenClaw apply_patch-format executor.

        This intentionally handles the common Add/Delete/Update cases used by
        agents. It is not a full reimplementation of OpenClaw's TypeScript
        parser, but it keeps unified_runner callable with the OpenClaw schema.
        """
        lines = patch_content.splitlines()
        if not lines or lines[0].strip() != "*** Begin Patch":
            return {"error": "patch must start with *** Begin Patch"}
        changed: list[str] = []
        i = 1
        while i < len(lines):
            line = lines[i]
            if line.strip() == "*** End Patch":
                break
            if line.startswith("*** Add File: "):
                rel = line[len("*** Add File: "):].strip()
                i += 1
                added: list[str] = []
                while i < len(lines) and not lines[i].startswith("*** "):
                    if not lines[i].startswith("+"):
                        return {"error": f"add file line must start with + for {rel}"}
                    added.append(lines[i][1:])
                    i += 1
                self._file_write(rel, "\n".join(added) + ("\n" if added else ""))
                changed.append(f"added {rel}")
                continue
            if line.startswith("*** Delete File: "):
                rel = line[len("*** Delete File: "):].strip()
                target = self._guard_path(rel) if self.mode == "host" else rel
                if self.mode == "docker":
                    stdout, stderr, rc = self._shell(f"rm -f {shlex.quote(target)}", timeout=30)
                    if rc != 0:
                        return {"exit_code": rc, "output": _truncate(stdout + stderr)}
                else:
                    Path(target).unlink(missing_ok=True)
                changed.append(f"deleted {rel}")
                i += 1
                continue
            if line.startswith("*** Update File: "):
                rel = line[len("*** Update File: "):].strip()
                i += 1
                original = self._file_read(rel)
                updated = original
                old_lines: list[str] = []
                new_lines: list[str] = []
                saw_change = False

                def flush_chunk() -> str | None:
                    nonlocal updated, old_lines, new_lines, saw_change
                    if not old_lines and not new_lines:
                        return None
                    old = "\n".join(old_lines)
                    new = "\n".join(new_lines)
                    if old and old not in updated:
                        if old + "\n" in updated:
                            old += "\n"
                            new += "\n"
                        else:
                            return f"update context not found in {rel}"
                    if old:
                        updated = updated.replace(old, new, 1)
                    old_lines = []
                    new_lines = []
                    saw_change = True
                    return None

                while i < len(lines) and not lines[i].startswith("*** "):
                    cur = lines[i]
                    if cur.startswith("@@"):
                        err = flush_chunk()
                        if err:
                            return {"error": err}
                    elif cur.startswith(" "):
                        old_lines.append(cur[1:])
                        new_lines.append(cur[1:])
                    elif cur.startswith("-"):
                        old_lines.append(cur[1:])
                    elif cur.startswith("+"):
                        new_lines.append(cur[1:])
                    else:
                        # tolerate empty context lines as unchanged blank lines
                        old_lines.append(cur)
                        new_lines.append(cur)
                    i += 1
                err = flush_chunk()
                if err:
                    return {"error": err}
                if not saw_change:
                    return {"error": f"no update hunks for {rel}"}
                self._file_write(rel, updated)
                changed.append(f"modified {rel}")
                continue
            return {"error": f"unsupported patch line: {line}"}
        return {"exit_code": 0, "output": "Applied patch\n" + "\n".join(changed)}

    def _handle_grep(self, args: dict) -> dict:
        pattern = args["pattern"]
        path = args.get("path", ".")
        if self.mode == "host":
            try: path = self._guard_path(path)
            except PermissionError as e: return {"error": str(e)}
        cmd_parts = ["grep", "-rP"]
        if args.get("case_insensitive"):
            cmd_parts.append("-i")
        mode = args.get("output_mode", "files_with_matches")
        if mode == "files_with_matches":
            cmd_parts.append("-l")
        elif mode == "count":
            cmd_parts.append("-c")
        ctx = args.get("context_lines")
        if ctx:
            cmd_parts.extend(["-C", str(ctx)])
        glob_filter = args.get("glob")
        if glob_filter:
            cmd_parts.extend(["--include", glob_filter])
        cmd_parts.extend(["--", shlex.quote(pattern), shlex.quote(path)])
        cmd = " ".join(cmd_parts)
        stdout, stderr, rc = self._shell(cmd, timeout=30)
        return {"output": _truncate(stdout), "exit_code": rc}

    def _handle_find(self, args: dict) -> dict:
        pattern = args["pattern"]
        path = args.get("path", ".")
        if self.mode == "host":
            try: path = self._guard_path(path)
            except PermissionError as e: return {"error": str(e)}
        if "**" in pattern:
            name_part = pattern.split("/")[-1]
            cmd = f"find {shlex.quote(path)} -type f -name {shlex.quote(name_part)} 2>/dev/null | head -100"
        else:
            cmd = f"find {shlex.quote(path)} -type f -name {shlex.quote(pattern)} 2>/dev/null | head -100"
        stdout, _, _ = self._shell(cmd, timeout=30)
        files = [f for f in stdout.strip().split("\n") if f]
        return {"files": files[:100]}

    def _handle_ls(self, args: dict) -> dict:
        path = args.get("path", ".")
        if self.mode == "host":
            try: path = self._guard_path(path)
            except PermissionError as e: return {"error": str(e)}
        flags = "-la" if args.get("all") else "-l"
        stdout, stderr, rc = self._shell(f"ls {flags} {shlex.quote(path)}", timeout=15)
        if rc != 0:
            return {"error": stderr.strip() or stdout.strip()}
        return {"output": _truncate(stdout)}

    # Paths outside workdir that must never be written to in host mode.
    # Applied as a command-string pre-filter (not perfect — bash is too flexible
    # — but catches the common cases that produced ~30 file leaks pre-2026-04-19).
    _EXEC_WRITE_DENYLIST = (
        os.environ.get("SKILLRL_ROOT", str(Path(__file__).resolve().parents[3])).rstrip("/") + "/",
    )

    def _check_exec_safe(self, command: str) -> str | None:
        """Return reason string if command is deemed to write outside workdir.
        Host-mode sandbox_paths-only check. Returns None if OK.
        """
        if not self._sandbox_paths:
            return None
        for denied in self._EXEC_WRITE_DENYLIST:
            # Accept reads (cat, ls, grep) since those don't leak; only block
            # clear write patterns referencing the denylist path.
            if denied in command:
                # Detect write intents: >, >>, tee, cp/mv/rsync DST, mkdir, rm -, chmod +w
                # Use a conservative check: reject if the denied path appears
                # in the right-hand side of a redirection or destination arg.
                import re as _re
                write_markers = [
                    _re.compile(rf"[>]\s*[\"']?{_re.escape(denied)}"),
                    _re.compile(rf"\|\s*tee\s+[\"']?{_re.escape(denied)}"),
                    _re.compile(rf"\b(?:cp|mv|rsync|install|ln|mkdir)\b[^\n|]*{_re.escape(denied)}"),
                    _re.compile(rf"\brm\b[^\n|]*{_re.escape(denied)}"),
                    _re.compile(rf"\bchattr\b[^\n|]*{_re.escape(denied)}"),
                    _re.compile(rf"\b(?:touch|truncate)\b[^\n|]*{_re.escape(denied)}"),
                    _re.compile(rf"open\([\"']{_re.escape(denied)}[^\"']*[\"']\s*,\s*[\"']?[waxWAX]"),
                ]
                for pat in write_markers:
                    if pat.search(command):
                        return f"refused: command would write outside workdir ({denied})"
        return None

    def _handle_exec(self, args: dict) -> dict:
        command = args["command"]
        timeout = args.get("timeout", DEFAULT_EXEC_TIMEOUT)
        background = args.get("background", False)
        workdir = args.get("workdir")

        # Host-mode sandbox: refuse writes targeting paths outside workdir.
        # Python-layer guard for file-IO tools was already in place (_guard_path),
        # but exec could bypass via bash redirection until this filter was added.
        refusal = self._check_exec_safe(command)
        if refusal:
            return {
                "exit_code": 126,
                "output": f"[SANDBOX] {refusal}\n"
                          f"Use the {self.workdir!r} workdir for all outputs.",
            }

        if background and self.mode == "host":
            # Background processes are tracked separately — they don't run in
            # the persistent shell so they survive across exec calls.
            proc = subprocess.Popen(
                command, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self._bg_procs[proc.pid] = proc
            return {
                "pid": proc.pid,
                "sessionId": str(proc.pid),
                "status": "started in background",
            }

        stdout, stderr, rc = self._shell(command, timeout=timeout, workdir=workdir)
        output = stdout
        if stderr:
            output += "\n[STDERR]\n" + stderr
        return {"exit_code": rc, "output": _truncate(output)}

    def _handle_process(self, args: dict) -> dict:
        action = args["action"]
        if action == "read":
            action = "log"
        elif action == "signal":
            action = "kill"

        if action == "list":
            if self.mode == "host":
                info = []
                for pid, proc in list(self._bg_procs.items()):
                    poll = proc.poll()
                    info.append({
                        "pid": pid,
                        "status": "running" if poll is None else f"exited ({poll})",
                    })
                return {"processes": info}
            stdout, _, _ = self._shell("ps aux --sort=-rss | head -20", timeout=10)
            return {"output": _truncate(stdout)}

        elif action in {"poll", "log"}:
            pid = args.get("pid") or args.get("sessionId")
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                pid = None
            if not pid:
                return {"error": "sessionId required for process log/poll"}
            if self.mode == "host" and pid in self._bg_procs:
                proc = self._bg_procs[pid]
                if proc.poll() is not None:
                    stdout, stderr = proc.communicate(timeout=5)
                    return {
                        "pid": pid, "status": f"exited ({proc.returncode})",
                        "stdout": _truncate(stdout.decode("utf-8", errors="replace")),
                        "stderr": _truncate(stderr.decode("utf-8", errors="replace")),
                    }
                return {"pid": pid, "sessionId": str(pid), "status": "running"}
            return {"error": f"Process {pid} not tracked"}

        elif action == "kill":
            pid = args.get("pid") or args.get("sessionId")
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                pid = None
            sig_name = args.get("signal", "SIGTERM")
            if not pid:
                return {"error": "sessionId required for kill action"}
            if self.mode == "docker":
                self._shell(f"kill -{sig_name} {pid}", timeout=5)
                return {"pid": pid, "signal": sig_name, "sent": True}
            try:
                sig = getattr(signal, sig_name, signal.SIGTERM)
                os.kill(pid, sig)
                return {"pid": pid, "signal": sig_name, "sent": True}
            except ProcessLookupError:
                return {"error": f"Process {pid} not found"}

        elif action in {"write", "send-keys", "submit", "paste", "clear", "remove"}:
            return {
                "error": (
                    f"process action {action!r} is OpenClaw-compatible in schema "
                    "but not implemented by unified_runner's lightweight executor"
                )
            }

        return {"error": f"Unknown action: {action}"}

    # ---- web tools (runner-side HTTP; independent of container state) ------

    def _handle_web_fetch(self, args: dict) -> dict:
        import urllib.error
        import urllib.request
        url = args.get("url", "").strip()
        if not url.startswith(("http://", "https://")):
            return {"error": "url must start with http:// or https://"}
        extract_text = args.get("extract_text", True)
        max_chars = min(int(args.get("max_chars", 8000)), 32000)
        timeout = min(int(args.get("timeout", 30)), 60)
        fetch_url = f"https://r.jina.ai/{url}" if (extract_text and not url.startswith("https://r.jina.ai/")) else url
        headers = {"User-Agent": "Mozilla/5.0 (unified-runner web_fetch)"}
        jina_key = os.environ.get("JINA_API_KEY", "").strip()
        if jina_key and "r.jina.ai" in fetch_url:
            headers["Authorization"] = f"Bearer {jina_key}"
        try:
            req = urllib.request.Request(
                fetch_url,
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                body = raw.decode("utf-8", errors="replace")
                n = len(body)
                truncated = body[:max_chars]
                if n > max_chars:
                    truncated += f"\n\n[... truncated {n - max_chars} chars]"
                return {"url": url, "status_code": resp.status, "content": truncated}
        except urllib.error.HTTPError as e:
            return {"url": url, "status_code": e.code, "error": f"HTTP {e.code}: {e.reason}"}
        except Exception as e:
            return {"url": url, "error": f"{type(e).__name__}: {e}"}

    def _handle_web_search(self, args: dict) -> dict:
        import json as _json
        import urllib.error
        import urllib.parse
        import urllib.request
        query = args.get("query", "").strip()
        if not query:
            return {"error": "query is required"}
        # OpenClaw's native web_search schema uses `count`; legacy unified
        # runner schema used `num_results`. Accept both so OpenClaw-aligned
        # train/eval data and older trajectories execute identically.
        raw_count = args.get("count", args.get("num_results", 5))
        num_results = max(1, min(int(raw_count), 10))

        exa_key = os.environ.get("EXA_API_KEY")
        if exa_key:
            try:
                data = _json.dumps({
                    "query": query, "numResults": num_results,
                    "contents": {"text": {"maxCharacters": 500}},
                }).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.exa.ai/search",
                    data=data,
                    headers={
                        "x-api-key": exa_key,
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    r = _json.loads(resp.read())
                return {"provider": "exa", "results": [
                    {"title": it.get("title",""), "url": it.get("url",""),
                     "snippet": (it.get("text") or "")[:300]}
                    for it in (r.get("results") or [])[:num_results]
                ]}
            except Exception as e:
                return {"provider": "exa", "error": f"{type(e).__name__}: {e}"}

        gcse_key = os.environ.get("GOOGLE_CSE_API_KEY")
        gcse_id = os.environ.get("GOOGLE_CSE_ID")
        if gcse_key and gcse_id:
            try:
                qs = urllib.parse.urlencode({
                    "key": gcse_key, "cx": gcse_id, "q": query, "num": num_results,
                })
                with urllib.request.urlopen(
                    f"https://www.googleapis.com/customsearch/v1?{qs}", timeout=30
                ) as resp:
                    r = _json.loads(resp.read())
                return {"provider": "google_cse", "results": [
                    {"title": it.get("title",""), "url": it.get("link",""),
                     "snippet": it.get("snippet","")}
                    for it in (r.get("items") or [])[:num_results]
                ]}
            except Exception as e:
                return {"provider": "google_cse", "error": f"{type(e).__name__}: {e}"}

        return {"error": (
            "web_search: no API key configured. Set EXA_API_KEY or "
            "GOOGLE_CSE_API_KEY+GOOGLE_CSE_ID. Fallback: use web_fetch with a "
            "known URL (e.g. arxiv.org, github.com), or `exec curl https://r.jina.ai/<URL>`."
        )}
