#!/usr/bin/env python3
"""Pre-build Docker images for SkillsBench / SETA tasks (local Dockerfile, no clash).

Usage:
    python prebuild_images.py --dataset skillsbench               # w/skills + no-skills
    python prebuild_images.py --dataset skillsbench --variant with-skills
    python prebuild_images.py --dataset seta
    python prebuild_images.py --dataset seta --limit 30           # first 30 only
"""
import argparse
import os
import subprocess
import sys
import shutil
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(os.environ.get("SKILLRL_ROOT", str(Path(__file__).resolve().parents[3])))


def _docker_env():
    env = dict(os.environ)
    env["DOCKER_HOST"] = os.environ.get("DOCKER_HOST", "ssh://your-docker-host")
    return env


DOCKER_ENV = _docker_env()

DATASETS = {
    "skillsbench-with-skills": BASE_DIR / "datasets/skillsbench/tasks",
    "skillsbench-no-skills": BASE_DIR / "datasets/skillsbench/tasks-no-skills",
    "seta": BASE_DIR / "datasets/seta/dataset/seta_baseline_30",
    "seta-synth": BASE_DIR / "datasets/seta/dataset/synth_data_harbor",
}


def list_tasks(root):
    tasks = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        dockerfile = d / "environment" / "Dockerfile"
        if dockerfile.exists():
            tasks.append((d.name, d / "environment", dockerfile))
    return tasks


def image_exists(tag):
    r = subprocess.run(
        ["docker", "images", "-q", tag],
        capture_output=True, text=True, timeout=15, env=_docker_env(),
    )
    return bool(r.stdout.strip())


def _prepare_build_dockerfile(dockerfile: Path) -> tuple[Path, str | None]:
    """Patch Ubuntu apt sources to a CN mirror for generated task images.

    This mirrors `run_unified_harbor.py`: task Dockerfiles are left untouched,
    while the build uses a temporary Dockerfile when it contains apt updates.
    """
    text = dockerfile.read_text(encoding="utf-8", errors="replace")
    if "apt-get update" not in text or "ubuntu" not in text.lower():
        return dockerfile, None

    snippet = (
        "RUN set -eux; \\\n"
        "    sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; "
        "s|http://security.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' "
        "/etc/apt/sources.list 2>/dev/null || true; \\\n"
        "    find /etc/apt/sources.list.d -type f -name '*.sources' -exec sed -i "
        "'s|http://archive.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; "
        "s|http://security.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' "
        "{} + 2>/dev/null || true\n"
    )
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        out.append(line)
        if line.lstrip().upper().startswith("FROM "):
            out.append(snippet + "\n")

    tmp_dir = tempfile.mkdtemp(prefix="unified-prebuild-dockerfile-")
    patched = Path(tmp_dir) / "Dockerfile"
    patched.write_text("".join(out), encoding="utf-8")
    return patched, tmp_dir


def build_one(dataset_tag, name, env_dir, dockerfile, timeout=1200):
    tag = f"unified-{dataset_tag}-{name}:latest"
    t0 = time.time()
    if image_exists(tag):
        return tag, "cached", time.time() - t0, ""
    build_dockerfile, tmp_dir = _prepare_build_dockerfile(dockerfile)
    try:
        r = subprocess.run(
            ["docker", "build", "-t", tag, "-f", str(build_dockerfile), str(env_dir)],
            capture_output=True, text=True, timeout=timeout, env=_docker_env(),
        )
        dt = time.time() - t0
        if r.returncode == 0:
            return tag, "ok", dt, ""
        return tag, "FAIL", dt, (r.stderr or r.stdout)[-600:]
    except subprocess.TimeoutExpired:
        return tag, "TIMEOUT", time.time() - t0, f"exceeded {timeout}s"
    except Exception as e:
        return tag, "ERR", time.time() - t0, str(e)[:300]
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def run(dataset_tag, dataset_path, parallel, limit, log_path):
    tasks = list_tasks(dataset_path)
    if limit:
        tasks = tasks[:limit]
    print(f"[{dataset_tag}] {len(tasks)} tasks -> parallel={parallel}")
    print(f"[{dataset_tag}] log: {log_path}")

    ok = cached = failed = 0
    fails = []
    log_f = open(log_path, "a", buffering=1)
    log_f.write(f"\n=== {dataset_tag} ({len(tasks)} tasks) {time.strftime('%F %T')} ===\n")

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futs = {
            ex.submit(build_one, dataset_tag, n, e, d): n
            for (n, e, d) in tasks
        }
        for i, fut in enumerate(as_completed(futs), 1):
            name = futs[fut]
            tag, status, dt, err = fut.result()
            line = f"[{i}/{len(tasks)}] {status:8s} {dt:6.1f}s {tag}"
            if status == "ok":
                ok += 1
            elif status == "cached":
                cached += 1
            else:
                failed += 1
                fails.append((tag, status, err[:200]))
                line += f" | {err[:200]}"
            print(line)
            log_f.write(line + "\n")

    log_f.write(f"--- summary {dataset_tag}: ok={ok} cached={cached} failed={failed} ---\n")
    log_f.close()
    print(f"\n[{dataset_tag}] ok={ok} cached={cached} failed={failed}")
    if fails:
        print("failures:")
        for tag, st, err in fails[:10]:
            print(f"  {st:8s} {tag}: {err[:150]}")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["skillsbench", "skillsbench-with-skills", "skillsbench-no-skills",
                             "seta", "seta-synth"])
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log", default="/tmp/prebuild_images.log")
    args = ap.parse_args()

    if args.dataset == "skillsbench":
        targets = [("skillsbench-with-skills", DATASETS["skillsbench-with-skills"]),
                   ("skillsbench-no-skills", DATASETS["skillsbench-no-skills"])]
    else:
        targets = [(args.dataset, DATASETS[args.dataset])]

    total_failed = 0
    for tag, path in targets:
        total_failed += run(tag, path, args.parallel, args.limit, args.log)
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
