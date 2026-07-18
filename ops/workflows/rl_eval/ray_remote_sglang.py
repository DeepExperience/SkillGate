#!/usr/bin/env python3
"""Launch or stop SGLang servers on a Ray worker node without Ray Jobs.

This avoids running a job driver on the small Ray head node. The launcher itself
runs locally, schedules tiny Ray tasks onto the requested worker node, and those
tasks start/stop SGLang processes with explicit CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

import ray


ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[3])).resolve()


def q(value: object) -> str:
    return shlex.quote(str(value))


def parse_engine(value: str) -> tuple[str, int]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("engine must be GPUS:PORT, e.g. 0,1,2,3:30000")
    gpus, port = value.rsplit(":", 1)
    return gpus, int(port)


def init_ray() -> None:
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")


def launch(args: argparse.Namespace) -> None:
    init_ray()
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    refs = []
    for gpus, port in args.engine:
        log = log_dir / f"remote_{args.target_node}_{port}.log"

        @ray.remote(num_cpus=0.5, num_gpus=0, resources={f"node:{args.target_node}": 0.01})
        def _launch(gpus: str = gpus, port: int = port, log: str = str(log)) -> str:
            import socket
            import subprocess

            cmd = (
                f"cd {q(ROOT)} && setsid env "
                f"CUDA_VISIBLE_DEVICES={q(gpus)} "
                f"MODEL_PATH={q(args.model_path)} "
                f"SERVED_NAME={q(args.served_name)} "
                f"PORT={port} TP_SIZE={args.tp_size} "
                f"CONTEXT_LENGTH={args.context_length} "
                f"MEM_FRACTION={q(args.mem_fraction)} "
                f"RANDOM_SEED={q(args.seed)} "
                f"bash ops/launch/run_qwen35_sglang_server.sh "
                f"> {q(log)} 2>&1 &"
            )
            subprocess.run(["bash", "-lc", cmd], check=True)
            return f"{socket.gethostname()} launched port={port} gpus={gpus} log={log}"

        refs.append(_launch.remote())

    for line in ray.get(refs, timeout=args.timeout_sec):
        print(line)


def stop(args: argparse.Namespace) -> None:
    init_ray()

    @ray.remote(num_cpus=0.5, num_gpus=0, resources={f"node:{args.target_node}": 0.01})
    def _stop() -> str:
        import socket
        import subprocess
        import time

        pattern = f"sglang.launch_server.*{args.model_path}"
        subprocess.run(["bash", "-lc", f"pkill -f {q(pattern)} || true"], check=False)
        time.sleep(5)
        left = subprocess.run(
            ["bash", "-lc", f"pgrep -af {q(pattern)} | wc -l"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        gpu = subprocess.run(
            ["bash", "-lc", "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\\n' '|'"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return f"{socket.gethostname()} sglang_matching_left={left} gpu={gpu}"

    print(ray.get(_stop.remote(), timeout=args.timeout_sec))


def status(args: argparse.Namespace) -> None:
    init_ray()

    @ray.remote(num_cpus=0.5, num_gpus=0, resources={f"node:{args.target_node}": 0.01})
    def _status() -> str:
        import socket
        import subprocess

        procs = subprocess.run(
            ["bash", "-lc", "pgrep -af 'sglang.launch_server' || true"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        gpu = subprocess.run(
            ["bash", "-lc", "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\\n' '|'"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return f"host={socket.gethostname()}\ngpu={gpu}\nprocs=\n{procs}"

    print(ray.get(_status.remote(), timeout=args.timeout_sec))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["launch", "stop", "status"], required=True)
    parser.add_argument("--target-node", default=os.environ.get("SKILLRL_TARGET_NODE", "your-gpu-node-ip"))
    parser.add_argument("--engine", type=parse_engine, action="append", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-name", required=True)
    parser.add_argument("--log-dir", default=str(ROOT / "z_cc_terminal_imgs/.eval_queues/remote_sglang"))
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=65536)
    parser.add_argument("--mem-fraction", default="0.88")
    parser.add_argument("--seed", default="1063810697")
    parser.add_argument("--timeout-sec", type=int, default=120)
    args = parser.parse_args()
    if args.engine is None:
        args.engine = [("0,1,2,3", 30000), ("4,5,6,7", 30001)]

    if args.action == "launch":
        launch(args)
    elif args.action == "stop":
        stop(args)
    else:
        status(args)


if __name__ == "__main__":
    main()
