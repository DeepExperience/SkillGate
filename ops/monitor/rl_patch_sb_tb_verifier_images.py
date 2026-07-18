#!/usr/bin/env python3
"""Patch SB/TB verifier images so test.sh no longer installs at grading time.

The RL verifier preflight inventory records runtime dependency commands from
each task's `tests/test.sh`.  This script applies those dependencies once to
the already-prebuilt local image and commits the image back to the same tag:

  unified-skillsbench-no-skills-<task_id>:latest
  unified-tb2-<task_id>:latest

It intentionally does not build missing images.  Missing/unpatchable images are
logged so RL can run with `UNIFIED_HARBOR_REQUIRE_PREBUILT_LOCAL=1` and skip
them instead of spending rollout time on Docker builds or verifier downloads.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any
try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = PROJECT_ROOT / "experiments/infra/rl/preflight/sb_tb_verifier_static_inventory_20260524.json"
DEFAULT_OUT = PROJECT_ROOT / "experiments/infra/rl/preflight/sb_tb_verifier_image_patch_latest.jsonl"

DATASET_PREFIX = {
    "sb_ns": "unified-skillsbench-no-skills",
    "tb2": "unified-tb2",
}
DATASET_DIR = {
    "sb_ns": PROJECT_ROOT / "datasets/skillsbench/tasks",
    "tb2": PROJECT_ROOT / "datasets/terminal-bench-v2",
}

APT_ALWAYS = {"ca-certificates", "curl", "python3", "python3-pip", "python3-venv"}
APT_DENY = {">", "2>&1", "2>/dev/null", "/dev/null", "||", "true", "\\", "install", "apt-get", "apt"}
PIP_DENY = {
    "||",
    "true",
    "\\",
    "install",
    "pip",
    "pip3",
    "uv",
    "2>/dev/null",
    "2>&1",
    "/dev/null",
}
PIP_OPTION_WITH_ARG = {
    "-r",
    "--requirement",
    "-c",
    "--constraint",
    "-i",
    "--index-url",
    "--extra-index-url",
    "--find-links",
}
UV_OPTION_WITH_ARG = {
    "--with",
    "-w",
    "--from",
    "-p",
    "--python",
    "--index",
    "--extra-index-url",
    "--index-strategy",
}
PIP_SKIP_PREFIXES = ("-", ">", "2>")


def run(cmd: list[str], *, docker_host: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["DOCKER_HOST"] = docker_host
    return subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=timeout)


def docker(args: list[str], *, docker_host: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return run(["docker", "-H", docker_host, *args], docker_host=docker_host, timeout=timeout)


def tail(text: str | None, limit: int = 2000) -> str:
    return "" if not text else str(text)[-limit:]


def shell_tokens(cmd: str) -> list[str]:
    try:
        return shlex.split(cmd)
    except ValueError:
        return cmd.split()


def clean_pkg_token(token: str) -> str | None:
    token = token.strip().strip(";")
    if not token or token in PIP_DENY or token.startswith(PIP_SKIP_PREFIXES):
        return None
    if token in {"quiet", "pytest", "python", "python3"}:
        return token
    if re.match(r"^[A-Za-z0-9_.-]+(==|>=|<=|~=|>|<).+", token):
        return token
    if token.startswith("git+"):
        return token
    if re.match(r"^[A-Za-z0-9_.-]+$", token):
        return token
    return None


def extract_pip_packages(cmd: str) -> list[str]:
    toks = shell_tokens(cmd)
    if "install" not in toks:
        return []
    out: list[str] = []
    i = toks.index("install") + 1
    while i < len(toks):
        tok = toks[i]
        if tok in PIP_OPTION_WITH_ARG:
            i += 2
            continue
        if tok.startswith("--") or tok.startswith("-"):
            i += 1
            continue
        pkg = clean_pkg_token(tok)
        if pkg:
            out.append(pkg)
        i += 1
    return out


def extract_uv_with_packages(cmd: str) -> list[str]:
    toks = shell_tokens(cmd)
    if "uvx" not in toks:
        return []
    toks = toks[toks.index("uvx") + 1 :]
    out: list[str] = []
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok in {"--with", "-w"} and i + 1 < len(toks):
            pkg = clean_pkg_token(toks[i + 1])
            if pkg:
                out.append(pkg)
            i += 2
            continue
        if tok.startswith("--with=") or tok.startswith("-w="):
            pkg = clean_pkg_token(tok.split("=", 1)[1])
            if pkg:
                out.append(pkg)
            i += 1
            continue
        if tok in UV_OPTION_WITH_ARG:
            i += 2
            continue
        if any(tok.startswith(prefix + "=") for prefix in UV_OPTION_WITH_ARG if prefix.startswith("--")):
            i += 1
            continue
        if tok.startswith("-"):
            i += 1
            continue
        break
    return out


def script_text_for(row: dict[str, Any]) -> str:
    if row["bench"] == "tb2":
        path = DATASET_DIR["tb2"] / row["task_id"] / "tests" / "test.sh"
    else:
        path = DATASET_DIR["sb_ns"] / row["task_id"] / "tests" / "test.sh"
    try:
        return path.read_text()
    except Exception:
        return ""


def shell_assignments(script: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for match in re.finditer(r"(?m)^([A-Za-z_][A-Za-z0-9_]*)=(['\"])(.*?)\2", script):
        assignments[match.group(1)] = match.group(3)
    return assignments


def expand_shell_vars(spec: str, assignments: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        return assignments.get(name, match.group(0))

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", repl, spec)


def extract_apt_packages(cmd: str) -> list[str]:
    toks = shell_tokens(cmd)
    if "install" not in toks:
        return []
    out: list[str] = []
    for tok in toks[toks.index("install") + 1 :]:
        tok = tok.strip().strip(";")
        if not tok or tok.startswith("-") or tok in APT_DENY:
            continue
        if re.match(r"^[A-Za-z0-9+_.-]+$", tok):
            out.append(tok)
    return out


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = package_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def package_key(spec: str) -> str:
    """Best-effort package identity for deconflicting per-task install specs."""
    if spec.startswith("git+"):
        return spec.lower()
    name = re.split(r"==|>=|<=|~=|>|<|\[", spec, maxsplit=1)[0]
    return name.strip().lower().replace("_", "-")


def package_plan(row: dict[str, Any]) -> dict[str, Any]:
    apt = set(APT_ALWAYS)
    pip: list[str] = []
    script = script_text_for(row)
    assignments = shell_assignments(script)
    for cmd in row.get("apt_cmds") or []:
        apt.update(extract_apt_packages(cmd))
    for cmd in row.get("pip_cmds") or []:
        pip.extend(extract_pip_packages(cmd))
    for cmd in row.get("uvx_cmds") or []:
        pip.extend(extract_uv_with_packages(cmd))
    pip = [expand_shell_vars(pkg, assignments) for pkg in pip]
    # pytest-json-ctrf provides the --ctrf pytest option used by most tasks.
    existing = {package_key(p) for p in pip}
    if row.get("bench") in {"sb_ns", "tb2"}:
        if "pytest" not in existing:
            pip.append("pytest==8.4.1")
        if ("--ctrf" in script or "pytest-json-ctrf" in script) and "pytest-json-ctrf" not in existing:
            pip.append("pytest-json-ctrf==0.3.5")
    return {
        "apt": sorted(apt),
        "pip": dedupe_keep_order([p for p in pip if p not in {"pytest", "pytest-json-ctrf"}]),
        "playwright": bool(row.get("flags") and "playwright_install" in row["flags"]),
        "npm": bool(row.get("flags") and "npm_install" in row["flags"]),
    }


def image_for(row: dict[str, Any]) -> str:
    if row["bench"] == "tb2":
        task_toml = DATASET_DIR["tb2"] / row["task_id"] / "task.toml"
        if task_toml.exists():
            try:
                meta = tomllib.loads(task_toml.read_text())
                declared = meta.get("environment", {}).get("docker_image")
                if declared:
                    return str(declared)
            except Exception:
                pass
    return f"{DATASET_PREFIX[row['bench']]}-{row['task_id']}:latest"


def patch_one(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    image = image_for(row)
    cname = f"verifier-patch-{row['bench']}-{row['task_id']}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    plan = package_plan(row)
    result: dict[str, Any] = {
        "bench": row["bench"],
        "task_id": row["task_id"],
        "image": image,
        "container": cname,
        "flags": row.get("flags", []),
        "apt_count": len(plan["apt"]),
        "pip_count": len(plan["pip"]),
        "playwright": plan["playwright"],
        "status": "unknown",
        "started_at": time.strftime("%FT%T%z"),
    }
    t0 = time.time()
    try:
        r = docker(["images", "-q", image], docker_host=args.docker_host, timeout=30)
        if r.returncode != 0:
            result.update(status="docker_error", error=tail(r.stderr or r.stdout))
            return result
        if not r.stdout.strip():
            result.update(status="missing_image")
            return result
        if args.dry_run:
            result.update(status="dry_run", apt=plan["apt"], pip=plan["pip"][:80])
            return result

        r = docker(
            ["run", "-d", "--name", cname, image, "sleep", "infinity"],
            docker_host=args.docker_host,
            timeout=args.run_timeout,
        )
        if r.returncode != 0:
            result.update(status="run_failed", error=tail(r.stderr or r.stdout))
            return result
        result["run_sec"] = round(time.time() - t0, 3)

        apt_list = " ".join(shlex.quote(x) for x in plan["apt"])
        setup_cmd = f"""
set -eu
if command -v apt-get >/dev/null 2>&1; then
  if [ -f /etc/apt/sources.list ]; then
    sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://security.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list || true
  fi
  find /etc/apt/sources.list.d -type f -name '*.sources' -exec sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://security.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' {{}} + 2>/dev/null || true
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends {apt_list}
elif command -v dnf >/dev/null 2>&1; then
  dnf -y install python3 python3-pip ca-certificates curl || true
elif command -v yum >/dev/null 2>&1; then
  yum -y install python3 python3-pip ca-certificates curl || true
elif command -v apk >/dev/null 2>&1; then
  apk add --no-cache python3 py3-pip ca-certificates curl || true
fi
mkdir -p /etc/pip /root/.config/pip /root/.pip
cat >/etc/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120
retries = 3
EOF
cp /etc/pip.conf /root/.config/pip/pip.conf
cp /etc/pip.conf /root/.pip/pip.conf
if python3 -m pip install --help 2>/dev/null | grep -q -- '--break-system-packages'; then
  python3 -m pip install -q --break-system-packages --ignore-installed setuptools wheel || true
else
  python3 -m pip install -q --ignore-installed setuptools wheel || true
fi
"""
        t = time.time()
        r = docker(["exec", cname, "bash", "-lc", setup_cmd], docker_host=args.docker_host, timeout=args.apt_timeout)
        result["apt_sec"] = round(time.time() - t, 3)
        if r.returncode != 0:
            result.update(status="apt_failed", error=tail(r.stderr or r.stdout))
            return result

        if plan["pip"]:
            chunks = [plan["pip"][i : i + args.pip_chunk_size] for i in range(0, len(plan["pip"]), args.pip_chunk_size)]
            pip_times: list[float] = []
            for idx, chunk in enumerate(chunks, 1):
                pkg_list = " ".join(shlex.quote(x) for x in chunk)
                cmd = (
                    "if python3 -m pip install --help 2>/dev/null | grep -q -- '--break-system-packages'; then "
                    f"python3 -m pip install -q --break-system-packages --ignore-installed {pkg_list}; "
                    "else "
                    f"python3 -m pip install -q --ignore-installed {pkg_list}; "
                    "fi"
                )
                t = time.time()
                r = docker(["exec", cname, "bash", "-lc", cmd], docker_host=args.docker_host, timeout=args.pip_timeout)
                pip_times.append(round(time.time() - t, 3))
                if r.returncode != 0:
                    result.update(status="pip_failed", pip_chunk=idx, error=tail(r.stderr or r.stdout))
                    return result
            result["pip_sec"] = round(sum(pip_times), 3)

        if plan["playwright"]:
            t = time.time()
            r = docker(
                ["exec", cname, "bash", "-lc", "python3 -m playwright install --with-deps chromium || python3 -m playwright install chromium"],
                docker_host=args.docker_host,
                timeout=args.playwright_timeout,
            )
            result["playwright_sec"] = round(time.time() - t, 3)
            if r.returncode != 0:
                result.update(status="playwright_failed", error=tail(r.stderr or r.stdout))
                return result

        cleanup_cmd = "rm -rf /var/lib/apt/lists/* /root/.cache/pip /tmp/* || true"
        docker(["exec", cname, "bash", "-lc", cleanup_cmd], docker_host=args.docker_host, timeout=60)

        t = time.time()
        r = docker(["commit", cname, image], docker_host=args.docker_host, timeout=args.commit_timeout)
        result["commit_sec"] = round(time.time() - t, 3)
        if r.returncode != 0:
            result.update(status="commit_failed", error=tail(r.stderr or r.stdout))
            return result
        result.update(status="patched")
        return result
    except subprocess.TimeoutExpired as exc:
        result.update(status="timeout", error=f"{exc.cmd} exceeded {exc.timeout}s")
        return result
    except Exception as exc:  # noqa: BLE001
        result.update(status="exception", error=f"{type(exc).__name__}: {exc}")
        return result
    finally:
        try:
            cleanup = docker(["rm", "-f", cname], docker_host=args.docker_host, timeout=args.cleanup_timeout)
            if cleanup.returncode != 0 and cleanup.stderr:
                result["cleanup_error"] = tail(cleanup.stderr or cleanup.stdout, 500)
        except subprocess.TimeoutExpired:
            result["cleanup_error"] = f"docker rm -f exceeded {args.cleanup_timeout}s"
        except Exception as exc:  # noqa: BLE001
            result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        result["total_sec"] = round(time.time() - t0, 3)


def load_success_keys(paths: list[str]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if not path or not Path(path).exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("status") in {"patched", "already_patched"}:
                    keys.add(f"{row.get('bench')}/{row.get('task_id')}")
    return keys


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = json.loads(Path(args.inventory).read_text())
    benches = set(args.bench or DATASET_PREFIX)
    only = {x for item in args.only for x in item.split(",") if x}
    skip_success = load_success_keys(args.skip_success_from)
    selected = []
    for row in rows:
        if row.get("bench") not in benches:
            continue
        if f"{row['bench']}/{row['task_id']}" in skip_success:
            continue
        if only and f"{row['bench']}/{row['task_id']}" not in only and str(row["task_id"]) not in only:
            continue
        if not row.get("exists"):
            continue
        if args.only_runtime_installs and not row.get("flags"):
            continue
        selected.append(row)
    selected.sort(key=lambda r: (r["bench"], r["task_id"]))
    if args.limit:
        selected = selected[: args.limit]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--docker-host", default="unix:///tmp/local-docker-overlay2.sock")
    parser.add_argument("--bench", action="append", choices=sorted(DATASET_PREFIX))
    parser.add_argument("--only", action="append", default=[], help="task id or bench/task id; comma-separated; repeatable")
    parser.add_argument("--skip-success-from", action="append", default=[], help="JSONL output(s) with successful prior patches to skip")
    parser.add_argument("--only-runtime-installs", action="store_true", default=True)
    parser.add_argument("--include-no-runtime-installs", dest="only_runtime_installs", action="store_false")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--apt-timeout", type=int, default=600)
    parser.add_argument("--run-timeout", type=int, default=180)
    parser.add_argument("--pip-timeout", type=int, default=1200)
    parser.add_argument("--playwright-timeout", type=int, default=1200)
    parser.add_argument("--commit-timeout", type=int, default=300)
    parser.add_argument("--cleanup-timeout", type=int, default=20)
    parser.add_argument("--pip-chunk-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[sb-tb-patch] rows={len(rows)} jobs={args.jobs} dry_run={args.dry_run} out={out}")
    counts: dict[str, int] = {}
    with out.open("a", buffering=1) as fh, futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        future_to_row = {pool.submit(patch_one, row, args): row for row in rows}
        for i, fut in enumerate(futures.as_completed(future_to_row), 1):
            row = fut.result()
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{i}/{len(rows)}] {row['status']:18s} "
                f"{row['bench']}/{row['task_id']} {row.get('total_sec', 0):7.1f}s "
                f"apt={row.get('apt_count', 0)} pip={row.get('pip_count', 0)} "
                f"{row.get('error','')[:140]}"
            )
    print("[sb-tb-patch] summary", json.dumps(counts, ensure_ascii=False, sort_keys=True))
    return 0 if not any(k in counts for k in ("exception", "docker_error")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
