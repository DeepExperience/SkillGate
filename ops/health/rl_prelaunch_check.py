#!/usr/bin/env python3
"""Pre-launch health gate for Relax RL runs.

This check is intentionally local and cheap. It catches the failure mode where
Docker-over-SSH leaves many orphaned ssh/nc zombies under PID 1 and the worker
is close to its pod pids limit; in that state Ray Serve / SGLang startup will
fail with pthread_create/fork errors and should not be attempted.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except Exception:
        return None


def count_processes() -> dict[str, Any]:
    total_proc = 0
    total_threads = 0
    zombie_by_name: dict[str, int] = {}
    zombie_ppid: dict[str, int] = {}
    live_ssh_nc = 0
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            total_proc += 1
            comm = (item / "comm").read_text().strip()
            status = (item / "status").read_text().splitlines()
            state = next((line for line in status if line.startswith("State:")), "")
            ppid = next((line.split()[1] for line in status if line.startswith("PPid:")), "?")
            threads = int(next((line.split()[1] for line in status if line.startswith("Threads:")), "0"))
            total_threads += threads
            if comm in {"ssh", "nc"}:
                if "\tZ" in state:
                    zombie_by_name[comm] = zombie_by_name.get(comm, 0) + 1
                    key = f"{comm}:ppid={ppid}"
                    zombie_ppid[key] = zombie_ppid.get(key, 0) + 1
                else:
                    live_ssh_nc += 1
        except Exception:
            continue
    return {
        "total_proc": total_proc,
        "total_threads": total_threads,
        "zombie_by_name": zombie_by_name,
        "zombie_ssh_nc_total": sum(zombie_by_name.get(k, 0) for k in ("ssh", "nc")),
        "zombie_ppid_top": sorted(zombie_ppid.items(), key=lambda kv: kv[1], reverse=True)[:10],
        "live_ssh_nc": live_ssh_nc,
    }


def cgroup_pids() -> dict[str, Any]:
    rows = []
    for root, _, files in os.walk("/sys/fs/cgroup"):
        if "pids.current" not in files or "pids.max" not in files:
            continue
        cur_s = _read(Path(root) / "pids.current")
        max_s = _read(Path(root) / "pids.max")
        if cur_s is None or max_s is None:
            continue
        try:
            cur = int(cur_s)
        except ValueError:
            continue
        max_value = None if max_s == "max" else int(max_s)
        ratio = None if not max_value else cur / max_value
        rows.append({"path": root, "current": cur, "max": max_s, "ratio": ratio})
    rows.sort(key=lambda row: row["current"], reverse=True)
    limited = [row for row in rows if row["max"] != "max"]
    limited.sort(key=lambda row: (row["ratio"] or 0), reverse=True)
    return {"top_current": rows[:10], "top_limited_ratio": limited[:10]}


def ray_gpu_usage() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["ray", "status"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        out = proc.stdout
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}
    lines = []
    for line in out.splitlines():
        if any(token in line for token in ("Total Usage:", "GPU", "Pending Demands", "Active:", "Idle:", "Pending:")):
            lines.append(line)
    return {"ok": proc.returncode == 0, "summary_lines": lines[-40:]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Relax RL pre-launch health check")
    parser.add_argument("--max-zombie-ssh-nc", type=int, default=1000)
    parser.add_argument("--max-pids-ratio", type=float, default=0.70)
    parser.add_argument("--skip-ray", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit non-zero when unhealthy")
    args = parser.parse_args()

    proc = count_processes()
    cg = cgroup_pids()
    ray = None if args.skip_ray else ray_gpu_usage()

    failures = []
    if proc["zombie_ssh_nc_total"] > args.max_zombie_ssh_nc:
        failures.append(
            f"ssh/nc zombie count {proc['zombie_ssh_nc_total']} exceeds {args.max_zombie_ssh_nc}"
        )
    for row in cg["top_limited_ratio"]:
        ratio = row.get("ratio")
        if ratio is not None and ratio > args.max_pids_ratio and row["current"] > 1000:
            failures.append(
                f"cgroup pids ratio {ratio:.3f} exceeds {args.max_pids_ratio:.3f}: "
                f"{row['current']}/{row['max']} at {row['path']}"
            )
            break

    result = {"ok": not failures, "failures": failures, "processes": proc, "cgroups": cg, "ray": ray}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("OK" if result["ok"] else "UNHEALTHY")
        for failure in failures:
            print(f"- {failure}")
        print(f"- total_proc={proc['total_proc']} total_threads={proc['total_threads']}")
        print(f"- zombie_ssh_nc_total={proc['zombie_ssh_nc_total']} by_name={proc['zombie_by_name']}")
        top = cg["top_limited_ratio"][:3]
        for row in top:
            ratio = row.get("ratio")
            ratio_s = "n/a" if ratio is None else f"{ratio:.3f}"
            print(f"- cgroup pids {row['current']}/{row['max']} ratio={ratio_s} path={row['path']}")
        if ray:
            print(f"- ray_ok={ray.get('ok')} ray_summary={ray.get('summary_lines', [])[-6:]}")
    return 1 if failures and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
