#!/usr/bin/env python3
"""
Monitor GPU utilization and keep an already-mounted local model endpoint warm
after a long idle period.

Default behavior:
- sample nvidia-smi every 60s;
- optionally restrict the idle decision to a GPU subset;
- if max selected-GPU utilization stays <= 3% for 5 hours, enter keepalive mode;
- in keepalive mode, call the local OpenAI-compatible endpoint every 120s;
- if the endpoint is absent or unhealthy, run a short local CUDA matmul probe;
- if selected-GPU utilization is >= 20%, skip keepalive calls to avoid
  competing with real experiments.

No files are written by this script; log output goes to stdout/stderr only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable


@dataclass(frozen=True)
class GpuSample:
    index: int
    util: int


def ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{ts()}] {message}", flush=True)


def query_gpu_util() -> list[GpuSample]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    samples: list[GpuSample] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        samples.append(GpuSample(index=int(parts[0]), util=int(parts[1])))
    if not samples:
        raise RuntimeError("nvidia-smi returned no GPU samples")
    return samples


def parse_gpu_indices(raw: str) -> set[int] | None:
    if not raw.strip():
        return None
    indices: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        indices.add(int(item))
    if not indices:
        return None
    return indices


def filter_gpu_samples(samples: list[GpuSample], indices: set[int] | None) -> list[GpuSample]:
    if indices is None:
        return samples
    selected = [sample for sample in samples if sample.index in indices]
    if not selected:
        available = ", ".join(str(sample.index) for sample in samples)
        requested = ", ".join(str(index) for index in sorted(indices))
        raise RuntimeError(f"no selected GPU samples; requested={requested} available={available}")
    return selected


def max_util(samples: Iterable[GpuSample]) -> int:
    return max(sample.util for sample in samples)


def format_samples(samples: Iterable[GpuSample]) -> str:
    return ", ".join(f"gpu{sample.index}={sample.util}%" for sample in samples)


def post_json(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'sk-local-anything')}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body.decode("utf-8"))


def get_json(url: str, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'sk-local-anything')}"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body.decode("utf-8"))


def detect_model(api_base: str, timeout: int) -> str:
    data = get_json(f"{api_base.rstrip('/')}/models", timeout=timeout)
    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"no models returned by {api_base}/models")
    model = models[0].get("id")
    if not model:
        raise RuntimeError(f"first model has no id: {models[0]!r}")
    return str(model)


def call_openai_compatible_endpoint(args: argparse.Namespace, model: str) -> bool:
    prompt = (
        "Generate a compact deterministic keepalive response: write 384 comma-separated "
        "integers between 0 and 9999. No explanation, no markdown."
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a local GPU keepalive probe. Keep the response plain text.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
        "stream": False,
    }
    url = f"{args.api_base.rstrip('/')}/chat/completions"
    started = time.monotonic()
    try:
        response = post_json(url, payload, timeout=args.call_timeout_sec)
    except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        log(f"keepalive HTTP call failed: {exc}")
        return False

    elapsed = time.monotonic() - started
    usage = response.get("usage") or {}
    completion_tokens = usage.get("completion_tokens", "?")
    log(f"keepalive HTTP call ok: elapsed={elapsed:.1f}s completion_tokens={completion_tokens}")
    return True


def call_command(args: argparse.Namespace) -> bool:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            args.call_command,
            shell=True,
            timeout=args.call_timeout_sec,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log(f"keepalive command timed out after {args.call_timeout_sec}s")
        return False

    elapsed = time.monotonic() - started
    if completed.returncode == 0:
        log(f"keepalive command ok: elapsed={elapsed:.1f}s")
        return True
    log(f"keepalive command failed: exit={completed.returncode} elapsed={elapsed:.1f}s")
    return False


def call_torch_gpu_probe(args: argparse.Namespace) -> bool:
    """Generate real GPU utilization without requiring a served model endpoint.

    This is the safety net for GLM/MaaS-only nights: the old keepalive could
    enter keepalive mode but still do no GPU work if no local `/v1` endpoint was
    running.  A short CUDA matmul loop is enough to make GPU util non-zero while
    avoiding persistent files or model downloads.
    """
    started = time.monotonic()
    try:
        import torch

        if not torch.cuda.is_available():
            log("torch gpu probe skipped: cuda is not available")
            return False
        dtype = torch.float16
        size = args.gpu_probe_size
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            log("torch gpu probe skipped: no visible cuda devices")
            return False
        iterations_by_device = [0 for _ in range(device_count)]
        errors: list[str] = []
        deadline = time.monotonic() + args.gpu_probe_sec

        def worker(device_index: int) -> None:
            try:
                torch.cuda.set_device(device_index)
                device = torch.device(f"cuda:{device_index}")
                a = torch.randn((size, size), device=device, dtype=dtype)
                b = torch.randn((size, size), device=device, dtype=dtype)
                iterations = 0
                while time.monotonic() < deadline:
                    c = a @ b
                    a = c * 0.999 + a * 0.001
                    iterations += 1
                torch.cuda.synchronize(device)
                iterations_by_device[device_index] = iterations
            except Exception as exc:  # pragma: no cover - depends on CUDA runtime
                errors.append(f"cuda:{device_index}: {exc}")

        threads = [threading.Thread(target=worker, args=(index,), daemon=True) for index in range(device_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise RuntimeError("; ".join(errors))
        elapsed = time.monotonic() - started
        log(
            "torch gpu probe ok: "
            f"elapsed={elapsed:.1f}s devices={device_count} "
            f"iterations={iterations_by_device} size={size}"
        )
        return True
    except Exception as exc:
        log(f"torch gpu probe failed: {exc}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=int, default=3, help="Idle threshold for max GPU util percentage.")
    parser.add_argument("--busy-threshold", type=int, default=20, help="Skip keepalive calls above this util.")
    parser.add_argument("--idle-hours", type=float, default=5.0, help="Required continuous idle hours before keepalive.")
    parser.add_argument("--sample-sec", type=int, default=60, help="nvidia-smi sampling interval.")
    parser.add_argument("--keepalive-sec", type=int, default=120, help="Call interval after keepalive mode starts.")
    parser.add_argument("--status-sec", type=int, default=600, help="Periodic status log interval.")
    parser.add_argument(
        "--gpu-indices",
        default=os.environ.get("GPU_GUARD_GPU_INDICES", ""),
        help="Comma-separated physical GPU indices used for idle/busy decisions; default all GPUs.",
    )
    parser.add_argument("--api-base", default=os.environ.get("GPU_GUARD_API_BASE", "http://127.0.0.1:30000/v1"))
    parser.add_argument("--model", default=os.environ.get("GPU_GUARD_MODEL", ""), help="Model id; default auto-detect.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens for HTTP keepalive call.")
    parser.add_argument("--call-timeout-sec", type=int, default=300)
    parser.add_argument("--gpu-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--always-gpu-probe",
        action="store_true",
        help="Run the CUDA matmul probe even after a successful HTTP keepalive call.",
    )
    parser.add_argument("--gpu-probe-sec", type=int, default=20, help="Seconds of CUDA matmul when HTTP keepalive is unavailable.")
    parser.add_argument("--gpu-probe-size", type=int, default=4096, help="Square matrix size for CUDA matmul fallback.")
    parser.add_argument(
        "--call-command",
        default=os.environ.get("GPU_GUARD_CALL_COMMAND", ""),
        help="Optional shell command used instead of HTTP call. stdout/stderr are discarded.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Log intended calls without calling anything.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_gpu_indices = parse_gpu_indices(args.gpu_indices)
    idle_required_sec = int(args.idle_hours * 3600)
    low_since: float | None = None
    keepalive_mode = False
    next_keepalive_at = 0.0
    next_status_at = 0.0
    model: str | None = args.model or None

    log(
        "starting gpu_idle_keepalive "
        f"threshold={args.threshold}% idle_hours={args.idle_hours} "
        f"keepalive_sec={args.keepalive_sec} busy_threshold={args.busy_threshold}%"
    )
    if selected_gpu_indices is None:
        log("gpu selection: all physical GPUs")
    else:
        log(f"gpu selection: physical indices {','.join(str(index) for index in sorted(selected_gpu_indices))}")
    if args.call_command:
        log("keepalive target: custom command")
    else:
        log(f"keepalive target: {args.api_base.rstrip('/')}/chat/completions model={model or 'auto'}")
    if args.gpu_fallback:
        log(f"gpu fallback enabled: torch matmul {args.gpu_probe_sec}s size={args.gpu_probe_size}")
    if args.always_gpu_probe:
        log("always_gpu_probe enabled: every keepalive cycle will include CUDA matmul")

    while True:
        try:
            all_samples = query_gpu_util()
            samples = filter_gpu_samples(all_samples, selected_gpu_indices)
        except Exception as exc:
            log(f"gpu query failed: {exc}; retrying")
            time.sleep(args.sample_sec)
            continue

        now = time.monotonic()
        current_max = max_util(samples)
        if now >= next_status_at:
            low_for = 0 if low_since is None else int(now - low_since)
            mode = "keepalive" if keepalive_mode else "monitor"
            log(f"status mode={mode} max_util={current_max}% low_for={low_for}s samples=({format_samples(samples)})")
            next_status_at = now + args.status_sec

        if not keepalive_mode:
            if current_max <= args.threshold:
                if low_since is None:
                    low_since = now
                    log(f"low-util window started: max_util={current_max}%")
                low_elapsed = now - low_since
                if low_elapsed >= idle_required_sec:
                    keepalive_mode = True
                    next_keepalive_at = 0.0
                    log(f"entering keepalive mode after {int(low_elapsed)}s continuous low util")
            else:
                if low_since is not None:
                    log(f"low-util window reset: max_util={current_max}%")
                low_since = None
            # If this iteration just flipped into keepalive mode, loop back
            # almost immediately so the first keepalive call is not delayed by
            # a full sample interval after the idle threshold has already been
            # reached.
            time.sleep(1 if keepalive_mode else args.sample_sec)
            continue

        if current_max >= args.busy_threshold:
            log(f"GPU busy; skip keepalive call this cycle: max_util={current_max}%")
            time.sleep(args.sample_sec)
            continue

        if now >= next_keepalive_at:
            if args.dry_run:
                log("dry-run: would issue keepalive call")
            elif args.call_command:
                ok = call_command(args)
                if not ok and args.gpu_fallback:
                    call_torch_gpu_probe(args)
            else:
                ok = False
                if model is None:
                    try:
                        model = detect_model(args.api_base, timeout=args.call_timeout_sec)
                        log(f"auto-detected model={model}")
                    except Exception as exc:
                        log(f"model auto-detect failed: {exc}")
                if model is not None:
                    ok = call_openai_compatible_endpoint(args, model)
                if args.always_gpu_probe or (not ok and args.gpu_fallback):
                    call_torch_gpu_probe(args)
            next_keepalive_at = time.monotonic() + args.keepalive_sec

        sleep_for = max(1, min(args.sample_sec, int(next_keepalive_at - time.monotonic())))
        time.sleep(sleep_for)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("stopped by KeyboardInterrupt")
        raise SystemExit(130)
