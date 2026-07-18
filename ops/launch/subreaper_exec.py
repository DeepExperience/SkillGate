#!/usr/bin/env python3
"""Subreaper wrapper for the local RL dockerd.

Runs a command (dockerd) as a child while marking THIS process as a
PR_SET_CHILD_SUBREAPER. Orphaned grandchildren - notably containerd-shim
processes that daemonize (double-fork + setsid) and reparent - then reparent to
THIS subreaper instead of to PID1. KubeRay worker pods have PID1 == `ray start
--block`, which does NOT reap children, so without this wrapper dead/orphaned
shims pile up (zombies + lock-blocked), saturate the kernel cgroup/netlink/rtnl
locks, and hard-wedge the node (root cause, docs/rl_log 2026-06-08). This loop
reaps them immediately so they can never accumulate.

Exits with the main child's exit status; forwards termination signals to it.
"""
import os
import sys
import signal
import ctypes

PR_SET_CHILD_SUBREAPER = 36


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: subreaper_exec.py <cmd> [args...]\n")
        sys.exit(2)
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        sys.stderr.write("[subreaper] WARN prctl(PR_SET_CHILD_SUBREAPER) failed; "
                         "orphans may still escape to PID1\n")
    else:
        sys.stderr.write("[subreaper] PR_SET_CHILD_SUBREAPER set; will reap orphaned shims\n")
    sys.stderr.flush()

    cmd = sys.argv[1:]
    pid = os.fork()
    if pid == 0:
        try:
            os.execvp(cmd[0], cmd)
        except OSError as exc:
            sys.stderr.write(f"[subreaper] exec failed: {exc!r}\n")
            os._exit(127)

    def _forward(signum, _frame):
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
        signal.signal(s, _forward)

    while True:
        try:
            wpid, status = os.waitpid(-1, 0)
        except ChildProcessError:
            break
        except InterruptedError:
            continue
        if wpid == pid:
            if os.WIFEXITED(status):
                sys.exit(os.WEXITSTATUS(status))
            if os.WIFSIGNALED(status):
                sys.exit(128 + os.WTERMSIG(status))
            sys.exit(1)
        # else: a reparented orphan (e.g. dead containerd-shim) was reaped -> continue
    sys.exit(0)


if __name__ == "__main__":
    main()
