#!/usr/bin/env python3
"""Bring up the rollout node (node B) after pod recreation.

Phase 1: spawn ray_worker_init.sh in tmux; start subreaper dockerd; verify.
Run on the surviving node; all node-side work happens via Ray NodeAffinity tasks.

Set SKILLRL_ROLLOUT_NODE_IP to the rollout node's Ray NodeManagerAddress and
SKILLRL_ROOT to the repository root before running.
"""
import os
import subprocess
import sys

import ray

TARGET_IP = os.environ.get("SKILLRL_ROLLOUT_NODE_IP", "your-rollout-node-ip")
ROOT = os.environ.get("SKILLRL_ROOT", "/path/to/skillRL")

ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)


def node_id(ip):
    nodes = [n for n in ray.nodes() if n["Alive"] and n["NodeManagerAddress"] == ip]
    if not nodes:
        raise SystemExit(f"no alive node {ip}")
    return nodes[0]["NodeID"]


@ray.remote(num_cpus=0.1)
def run(cmd, timeout=120):
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    return f"rc={r.returncode}\n{r.stdout}\n{r.stderr[-2000:] if r.stderr else ''}"


from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

strat = NodeAffinitySchedulingStrategy(node_id=node_id(TARGET_IP), soft=False)


def remote(cmd, timeout=120):
    return ray.get(run.options(scheduling_strategy=strat).remote(cmd, timeout), timeout=timeout + 30)


step = sys.argv[1] if len(sys.argv) > 1 else "phase1"

if step == "phase1":
    # 1. worker init in tmux (idempotent, long: apt/ShellCrash/harbor)
    print("== spawn worker-init tmux ==")
    print(remote(
        "tmux kill-session -t worker-init 2>/dev/null; "
        "tmux new-session -d -s worker-init "
        "'bash ~/ray_worker_init.sh "
        ">/tmp/ray_worker_init_rerun.log 2>&1; echo DONE_rc=$? >> /tmp/ray_worker_init_rerun.log'; "
        "tmux ls | grep worker-init"
    ))

    # 2. local dockerd under subreaper on the clean 3.3T nvme (/tmp/ray)
    print("== start local dockerd (subreaper, /tmp/ray) ==")
    print(remote(
        "cd %s && "
        "LOCAL_DOCKER_USE_SUBREAPER=1 "
        "LOCAL_DOCKER_DATA_ROOT=/tmp/ray/local-docker-overlay2-root "
        "LOCAL_DOCKER_EXEC_ROOT=/tmp/ray/local-docker-overlay2-exec "
        "bash ops/launch/start_local_overlay2_docker.sh 2>&1 | tail -5" % ROOT,
        timeout=180,
    ))

    # 3. verify dockerd + subreaper + squid reachability
    print("== verify ==")
    print(remote(
        "sleep 5; "
        "DOCKER_HOST=unix:///tmp/local-docker-overlay2.sock docker info --format "
        "'Server={{.ServerVersion}} Root={{.DockerRootDir}} Driver={{.Driver}}' 2>&1; "
        "pgrep -f subreaper_exec >/dev/null && echo SUBREAPER_OK || echo SUBREAPER_MISSING; "
        "timeout 8 curl -sx http://your-proxy:3128 -o /dev/null -w 'squid_pip=%{http_code}\\n' "
        "https://pypi.tuna.tsinghua.edu.cn/simple/ 2>&1 | tail -1",
        timeout=60,
    ))

elif step == "restore":
    print("== spawn img-restore tmux (551 tars, ~730G from networked storage) ==")
    print(remote(
        "tmux kill-session -t img-restore 2>/dev/null; "
        "tmux new-session -d -s img-restore "
        "'cd %s && DOCKER_HOST=unix:///tmp/local-docker-overlay2.sock "
        "/usr/bin/python3 ops/launch/restore_local_docker_images_from_cache.py "
        "--parquet %s/datasets/rl/parquet_4bench_factual_20260602 "
        "--local-docker-host unix:///tmp/local-docker-overlay2.sock "
        "--workers 8 "
        ">/tmp/img_restore.log 2>&1; echo DONE_rc=$? >> /tmp/img_restore.log'; "
        "tmux ls | grep img-restore" % (ROOT, ROOT)
    ))

elif step == "wait_restore":
    # blocking waiter: returns when img-restore tmux session exits
    print(remote(
        "while tmux has-session -t img-restore 2>/dev/null; do sleep 60; done; "
        "echo '== restore finished =='; tail -15 /tmp/img_restore.log",
        timeout=14400,
    ))

elif step == "status":
    print(remote(
        "echo '--- tmux ---'; tmux ls 2>&1; "
        "echo '--- worker-init tail ---'; tail -3 /tmp/ray_worker_init_rerun.log 2>/dev/null; "
        "echo '--- restore tail ---'; tail -5 /tmp/img_restore.log 2>/dev/null; "
        "echo '--- docker ---'; DOCKER_HOST=unix:///tmp/local-docker-overlay2.sock docker images -q 2>/dev/null | wc -l; "
        "df -h /tmp/ray | tail -1",
        timeout=60,
    ))
