#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -exo pipefail

echo "=== Cleaning up residual python/sglang processes ==="

ps -eo ppid=,pid=,cmd= | python3 -c '
import re
import sys

protected = re.compile(
    r"(ray start|raylet|gcs_server|plasma|log_monitor\\.py|dashboard|"
    r"runtime_env|RuntimeEnvAgent|DashboardAgent|JobSupervisor|"
    r"setup_worker\\.py|default_worker\\.py|ray::IDLE|ray::run_command|"
    r"gpustat|grep|kill_for_ray\\.sh)"
)
targets = re.compile(
    r"(sglang|relax\\.entrypoints\\.train|ray::ServeReplica|"
    r"ray::(Actor|Rollout|Reference|Advantages|TransferQueue|Megatron|SGLang))"
)

for line in sys.stdin:
    parts = line.strip().split(None, 2)
    if len(parts) < 3:
        continue
    ppid, pid, cmd = parts
    try:
        if int(ppid) <= 1:
            continue
    except ValueError:
        continue
    if protected.search(cmd):
        continue
    if targets.search(cmd):
        print(pid)
' | xargs -r kill -9 2>/dev/null || true
