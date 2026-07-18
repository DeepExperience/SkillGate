#!/usr/bin/env python3
"""Traffic monitor for remote-Docker-host/ShellCrash/Docker experiments.

This is intentionally read-only.  It samples the remote Docker host over SSH,
records ShellCrash cumulative traffic, proxy clients, host/container network
counters, and writes JSONL plus a compact human log for next-day attribution.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))
REMOTE_COLLECTOR_PATH = "/tmp/apex_traffic_collect.py"


REMOTE_COLLECTOR = r'''
import json
import os
import re
import subprocess
import time
import urllib.request
from datetime import datetime


def sh(cmd, timeout=8):
    try:
        p = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as exc:
        return exc.stdout or "", exc.stderr or "timeout", 124


def read_secret():
    for path in ("/tmp/ShellCrash/config.yaml", "/etc/ShellCrash/config.yaml"):
        try:
            with open(path, encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.startswith("secret:"):
                        return line.split(":", 1)[1].strip().strip("\"'")
        except OSError:
            pass
    return ""


def fetch_json(path, timeout=3, stream_first_line=False):
    secret = read_secret()
    headers = {"Authorization": "Bearer " + secret} if secret else {}
    req = urllib.request.Request("http://127.0.0.1:56789" + path, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        if stream_first_line:
            line = response.readline().decode("utf-8", errors="replace").strip()
            return json.loads(line) if line else {}
        return json.loads(response.read().decode("utf-8", errors="replace"))


def read_netdev():
    out = {}
    try:
        with open("/proc/net/dev") as handle:
            for line in handle.read().splitlines()[2:]:
                if ":" not in line:
                    continue
                name, rest = line.split(":", 1)
                vals = rest.split()
                if len(vals) < 16:
                    continue
                out[name.strip()] = {"rx": int(vals[0]), "tx": int(vals[8])}
    except OSError:
        pass
    return out


def ifindex_map(netdev):
    idx = {}
    for name in netdev:
        path = f"/sys/class/net/{name}/ifindex"
        try:
            with open(path) as handle:
                idx[int(handle.read().strip())] = name
        except OSError:
            pass
        except ValueError:
            pass
    return idx


def docker_containers(netdev):
    stdout, stderr, rc = sh("docker ps -q", timeout=8)
    if rc != 0:
        return [], {"docker_ps_error": stderr[-500:]}
    ids = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not ids:
        return [], {}

    stdout, stderr, rc = sh("docker inspect " + " ".join(ids), timeout=20)
    if rc != 0:
        return [], {"docker_inspect_error": stderr[-500:]}
    try:
        inspected = json.loads(stdout)
    except Exception as exc:
        return [], {"docker_inspect_parse_error": str(exc)}

    host_by_ifindex = ifindex_map(netdev)
    containers = []
    for info in inspected:
        name = (info.get("Name") or "").lstrip("/")
        state = info.get("State") or {}
        pid = state.get("Pid") or 0
        config = info.get("Config") or {}
        networks = (info.get("NetworkSettings") or {}).get("Networks") or {}
        ips = []
        for net_name, net_info in networks.items():
            ip = net_info.get("IPAddress")
            if ip:
                ips.append({"network": net_name, "ip": ip})

        host_veths = []
        root = f"/proc/{pid}/root/sys/class/net"
        if pid and os.path.isdir(root):
            for eth in os.listdir(root):
                if eth == "lo":
                    continue
                try:
                    with open(f"{root}/{eth}/iflink") as handle:
                        peer = int(handle.read().strip())
                    host = host_by_ifindex.get(peer)
                    if host:
                        counters = netdev.get(host, {})
                        host_veths.append(
                            {
                                "container_if": eth,
                                "host_if": host,
                                "rx": counters.get("rx", 0),
                                "tx": counters.get("tx", 0),
                            }
                        )
                except OSError:
                    pass
                except ValueError:
                    pass

        containers.append(
            {
                "id": info.get("Id", "")[:12],
                "name": name,
                "image": config.get("Image") or info.get("Image") or "",
                "pid": pid,
                "ips": ips,
                "host_veths": host_veths,
            }
        )
    return containers, {}


def proxy_connections(ip_to_container):
    stdout, stderr, rc = sh("ss -tanp 2>/dev/null | grep ':8888' || true", timeout=5)
    rows = []
    endpoint_re = re.compile(r"(\[?[0-9a-fA-F:.]+\]?):(\d+)$")
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0] not in {"ESTAB", "SYN-SENT", "TIME-WAIT", "CLOSE-WAIT"}:
            continue
        local = parts[3]
        peer = parts[4]

        def split_ep(ep):
            ep = ep.strip()
            match = endpoint_re.search(ep)
            if not match:
                return ep, ""
            return match.group(1).strip("[]"), match.group(2)

        lip, lport = split_ep(local)
        pip, pport = split_ep(peer)
        if lport == "8888":
            client_ip = pip
        elif pport == "8888":
            client_ip = lip
        else:
            client_ip = ""
        rows.append(
            {
                "state": parts[0],
                "local": local,
                "peer": peer,
                "client_ip": client_ip,
                "client_container": ip_to_container.get(client_ip, ""),
                "raw": line[-500:],
            }
        )
    return rows


def docker_image_summary():
    stdout, stderr, rc = sh("docker image ls --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.CreatedSince}} {{.Size}}' | head -n 200", timeout=8)
    if rc != 0:
        return {"error": stderr[-500:]}
    prefixes = {}
    for line in stdout.splitlines():
        repo = line.split()[0] if line.split() else ""
        if repo.startswith("unified-seta-"):
            key = "unified-seta-*"
        elif repo.startswith("unified-skillsbench-no-skills-"):
            key = "unified-skillsbench-no-skills-*"
        elif repo.startswith("unified-skillsbench-with-skills-"):
            key = "unified-skillsbench-with-skills-*"
        elif repo.startswith("alexgshaw/"):
            key = "alexgshaw/*"
        elif "sweb.eval" in repo:
            key = "swe/*"
        elif repo.startswith("claw-"):
            key = "claw-*"
        elif repo.startswith("mysql"):
            key = "mysql"
        else:
            key = repo.split("/", 1)[0] if repo else "<none>"
        prefixes[key] = prefixes.get(key, 0) + 1
    return {"prefix_counts_head200": prefixes}


def main():
    sample = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "epoch": time.time(),
        "host": os.uname().nodename,
    }
    errors = {}
    try:
        sample["clash_connections"] = fetch_json("/connections", timeout=3)
    except Exception as exc:
        errors["clash_connections"] = repr(exc)
    try:
        sample["clash_traffic_rate"] = fetch_json("/traffic", timeout=3, stream_first_line=True)
    except Exception as exc:
        errors["clash_traffic_rate"] = repr(exc)

    netdev = read_netdev()
    sample["netdev"] = {
        name: counters
        for name, counters in netdev.items()
        if name in {"enp1s0", "docker0", "lo"} or name.startswith("br-") or name.startswith("veth")
    }

    containers, container_errors = docker_containers(netdev)
    errors.update(container_errors)
    sample["containers"] = containers
    ip_to_container = {}
    for container in containers:
        for ip in container.get("ips", []):
            ip_to_container[ip["ip"]] = container["name"]
    sample["proxy_connections_8888"] = proxy_connections(ip_to_container)
    sample["docker_images"] = docker_image_summary()
    sample["errors"] = errors
    print(json.dumps(sample, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
'''


def run(cmd: list[str], *, input_text: str | None = None, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def install_remote_collector(remote: str) -> None:
    proc = run(["ssh", remote, f"cat > {REMOTE_COLLECTOR_PATH}"], input_text=REMOTE_COLLECTOR, timeout=20)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to install remote collector: {proc.stderr[-1000:]}")


def collect_once(remote: str) -> dict[str, Any]:
    proc = run(["ssh", remote, "python3", REMOTE_COLLECTOR_PATH], timeout=45)
    if proc.returncode != 0:
        raise RuntimeError(f"collector failed: {proc.stderr[-1000:]}")
    return json.loads(proc.stdout)


def total_container_bytes(container: dict[str, Any]) -> int:
    total = 0
    for item in container.get("host_veths") or []:
        total += int(item.get("rx") or 0) + int(item.get("tx") or 0)
    return total


def netdev_total(sample: dict[str, Any], name: str) -> int:
    counters = (sample.get("netdev") or {}).get(name) or {}
    return int(counters.get("rx") or 0) + int(counters.get("tx") or 0)


def bytes_mb(num: int | float) -> float:
    return float(num) / 1_000_000.0


def short_bytes(num: int | float) -> str:
    value = float(num)
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}GB"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}MB"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}KB"
    return f"{int(value)}B"


def infer_date(value: str | None) -> str:
    if value:
        match = re.search(r"(20\d{6})", value)
        if match:
            return match.group(1)
    for env_name in ("DATE", "EXPERIMENT_DATE"):
        env_value = os.environ.get(env_name, "")
        if re.fullmatch(r"20\d{6}", env_value):
            return env_value
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def resolve_default_log_dir() -> Path:
    explicit_root = os.environ.get("EXPERIMENT_ROOT") or os.environ.get("RUN_ROOT")
    if explicit_root:
        root = Path(explicit_root)
        if not root.is_absolute():
            root = PROJECT / root
        return root / "logs" / "monitor" / "traffic"

    run_id = os.environ.get("RUN_ID") or os.environ.get("UNIFIED_RUN_ID")
    date = infer_date(run_id)
    if run_id:
        return PROJECT / "experiments" / date / run_id / "logs" / "monitor" / "traffic"

    run_id = f"{date}_ops_monitor"
    return PROJECT / "experiments" / date / run_id / "logs" / "monitor" / "traffic"


def sample_deltas(prev: dict[str, Any] | None, cur: dict[str, Any]) -> dict[str, Any]:
    if not prev:
        return {}

    elapsed = max(1.0, float(cur.get("epoch", 0)) - float(prev.get("epoch", 0)))
    prev_conn = prev.get("clash_connections") or {}
    cur_conn = cur.get("clash_connections") or {}
    deltas: dict[str, Any] = {
        "elapsed_sec": elapsed,
        "clash_down": int(cur_conn.get("downloadTotal") or 0) - int(prev_conn.get("downloadTotal") or 0),
        "clash_up": int(cur_conn.get("uploadTotal") or 0) - int(prev_conn.get("uploadTotal") or 0),
        "netdev": {},
        "containers": {},
    }

    for name in sorted(set((prev.get("netdev") or {}).keys()) | set((cur.get("netdev") or {}).keys())):
        before = netdev_total(prev, name)
        after = netdev_total(cur, name)
        if after >= before:
            deltas["netdev"][name] = after - before

    prev_containers = {c.get("name"): total_container_bytes(c) for c in prev.get("containers") or []}
    cur_containers = {c.get("name"): total_container_bytes(c) for c in cur.get("containers") or []}
    for name, after in cur_containers.items():
        before = prev_containers.get(name)
        if before is not None and after >= before:
            deltas["containers"][name] = after - before
    return deltas


def summarize(cur: dict[str, Any], deltas: dict[str, Any], start_clash_down: int) -> str:
    ts = datetime.now().strftime("%H:%M:%S")
    conn = cur.get("clash_connections") or {}
    total_down = int(conn.get("downloadTotal") or 0)
    total_up = int(conn.get("uploadTotal") or 0)
    since_start = total_down - start_clash_down

    elapsed = float(deltas.get("elapsed_sec") or 0)
    clash_delta = int(deltas.get("clash_down") or 0)
    rate = clash_delta / elapsed if elapsed else 0

    proxy_clients: dict[str, int] = {}
    for row in cur.get("proxy_connections_8888") or []:
        key = row.get("client_container") or row.get("client_ip") or "unknown"
        proxy_clients[key] = proxy_clients.get(key, 0) + 1
    proxy_top = ", ".join(f"{name}:{count}" for name, count in sorted(proxy_clients.items(), key=lambda x: x[1], reverse=True)[:5]) or "-"

    container_delta = deltas.get("containers") or {}
    top_containers = ", ".join(
        f"{name}={short_bytes(delta)}"
        for name, delta in sorted(container_delta.items(), key=lambda x: x[1], reverse=True)[:5]
        if delta > 0
    ) or "-"

    netdev_delta = deltas.get("netdev") or {}
    registry_delta = container_delta.get("registry-mirror", 0)
    docker0_delta = netdev_delta.get("docker0", 0)
    enp_delta = netdev_delta.get("enp1s0", 0)

    return (
        f"[{ts}] clash +{short_bytes(clash_delta)} ({short_bytes(rate)}/s) "
        f"since_start={short_bytes(since_start)} total={short_bytes(total_down)} up_total={short_bytes(total_up)} | "
        f"proxy_clients={proxy_top} | top_container_delta={top_containers} | "
        f"registry={short_bytes(registry_delta)} docker0={short_bytes(docker0_delta)} enp1s0={short_bytes(enp_delta)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor remote-Docker-host/ShellCrash/Docker traffic overnight.")
    parser.add_argument("--remote", default="your-docker-host", help="SSH host for the remote Docker host.")
    parser.add_argument("--interval", type=int, default=60, help="Sampling interval in seconds.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=resolve_default_log_dir(),
        help="Output log directory. Defaults inside experiments/<date>/<run_id>/logs/monitor/traffic.",
    )
    parser.add_argument("--prefix", default=None, help="Log filename prefix; default uses current UTC date.")
    parser.add_argument("--once", action="store_true", help="Collect one sample and exit.")
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    prefix = args.prefix or f"{date}_overnight_traffic"
    jsonl_path = args.log_dir / f"{prefix}.jsonl"
    text_path = args.log_dir / f"{prefix}.log"

    install_remote_collector(args.remote)

    previous: dict[str, Any] | None = None
    start_clash_down: int | None = None
    start_epoch = time.time()
    last_summary_epoch = start_epoch
    sample_index = 0

    with jsonl_path.open("a", encoding="utf-8") as jsonl, text_path.open("a", encoding="utf-8") as text:
        header = {
            "event": "monitor_start",
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "remote": args.remote,
            "interval": args.interval,
            "jsonl": str(jsonl_path),
            "text": str(text_path),
            "remote_collector": REMOTE_COLLECTOR_PATH,
        }
        print(json.dumps(header, ensure_ascii=False), file=jsonl, flush=True)
        print(f"# monitor_start {header}", file=text, flush=True)
        print(f"[monitor] writing JSONL={jsonl_path}")
        print(f"[monitor] writing text={text_path}")

        while True:
            try:
                cur = collect_once(args.remote)
                conn = cur.get("clash_connections") or {}
                if start_clash_down is None:
                    start_clash_down = int(conn.get("downloadTotal") or 0)
                deltas = sample_deltas(previous, cur)
                record = {
                    "event": "sample",
                    "sample_index": sample_index,
                    "sample": cur,
                    "deltas": deltas,
                }
                print(json.dumps(record, ensure_ascii=False, sort_keys=True), file=jsonl, flush=True)
                line = summarize(cur, deltas, start_clash_down)
                print(line)
                print(line, file=text, flush=True)

                now = time.time()
                if now - last_summary_epoch >= 1800:
                    summary = {
                        "event": "summary",
                        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "runtime_min": round((now - start_epoch) / 60, 1),
                        "clash_down_since_start": int(conn.get("downloadTotal") or 0) - start_clash_down,
                        "clash_total_down": int(conn.get("downloadTotal") or 0),
                        "clash_total_up": int(conn.get("uploadTotal") or 0),
                    }
                    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), file=jsonl, flush=True)
                    summary_line = (
                        f"=== SUMMARY runtime={summary['runtime_min']}min "
                        f"clash_since_start={short_bytes(summary['clash_down_since_start'])} "
                        f"clash_total={short_bytes(summary['clash_total_down'])} ==="
                    )
                    print(summary_line)
                    print(summary_line, file=text, flush=True)
                    last_summary_epoch = now

                previous = cur
                sample_index += 1
                if args.once:
                    break
            except Exception as exc:
                error = {
                    "event": "error",
                    "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "error": repr(exc),
                }
                print(json.dumps(error, ensure_ascii=False), file=jsonl, flush=True)
                print(f"[monitor] ERROR {error['error']}", file=text, flush=True)
                print(f"[monitor] ERROR {error['error']}", file=sys.stderr)
                if args.once:
                    return 1

            if not args.once:
                time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
