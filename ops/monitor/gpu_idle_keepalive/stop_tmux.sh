#!/usr/bin/env bash
set -euo pipefail

SESSION=${GPU_GUARD_TMUX_SESSION:-gpu-idle-keepalive}

if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    echo "stopped tmux session: $SESSION"
else
    echo "tmux session not running: $SESSION"
fi
