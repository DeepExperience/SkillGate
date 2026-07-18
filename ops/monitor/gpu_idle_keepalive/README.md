# GPU Idle Keepalive

Purpose: prevent a container from being reclaimed when a model is loaded but GPU utilization stays near zero for too long.

Default policy:

- Samples `nvidia-smi` every 60 seconds.
- If max selected-GPU util stays `<= 3%` for 5 continuous hours, enters keepalive mode.
- In keepalive mode, calls the local OpenAI-compatible endpoint every 120 seconds.
- If no local endpoint/model is available, runs a short CUDA matmul fallback so
  MaaS-only runs still produce real GPU utilization.
- If max selected-GPU util is `>= 20%`, skips keepalive calls so real experiments are not disturbed.
- `start_tmux.sh` runs the monitor under a small supervisor loop, so if the
  Python monitor is killed the tmux session restarts it.
- Writes no persistent log files; output only goes to tmux scrollback/stdout.

## Start

```bash
cd /path/to/skillRL
bash ops/monitor/gpu_idle_keepalive/start_tmux.sh
```

For the two-Qwen-27B 8-card setup, start one guard per 4-GPU endpoint:

```bash
bash ops/monitor/start_qwen27b_dual_keepalive.sh
```

View:

```bash
tmux attach -t gpu-idle-keepalive
```

Stop:

```bash
bash ops/monitor/gpu_idle_keepalive/stop_tmux.sh
```

## Defaults

The default endpoint is:

```text
http://127.0.0.1:30000/v1
```

The model id is auto-detected from `/v1/models`.

`start_tmux.sh` uses a safer operational default of `GPU_GUARD_IDLE_HOURS=0.25`
(15 minutes) so a freshly started container does not wait most of the 8-hour
reclaim window before proving GPU activity.

## Manual Run With Overrides

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -u ops/monitor/gpu_idle_keepalive/gpu_idle_keepalive.py \
  --threshold 3 \
  --gpu-indices 0,1,2,3 \
  --idle-hours 5 \
  --keepalive-sec 120 \
  --api-base http://127.0.0.1:30000/v1
```

Use a custom command instead of the HTTP call:

```bash
GPU_GUARD_CALL_COMMAND='curl -sS --max-time 120 http://127.0.0.1:30000/v1/models >/dev/null' \
PYTHONDONTWRITEBYTECODE=1 python3 -B -u ops/monitor/gpu_idle_keepalive/gpu_idle_keepalive.py
```

For a real keepalive, the custom command should perform inference, not just query metadata.
