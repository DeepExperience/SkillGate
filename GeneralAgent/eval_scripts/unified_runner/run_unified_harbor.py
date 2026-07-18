#!/usr/bin/env python3
"""Unified interface evaluation for Harbor-format datasets (SkillsBench, SETA).

Builds Docker images from task environments, runs agent_loop with docker ToolLayer,
then executes the verifier (tests/test.sh) inside the container.

Usage:
    # SkillsBench with skills
    python run_unified_harbor.py --dataset skillsbench --variant with-skills
    # SkillsBench without skills
    python run_unified_harbor.py --dataset skillsbench --variant no-skills
    # SETA
    python run_unified_harbor.py --dataset seta
"""

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified_runner.agent_loop import UnifiedAgentLoop
from unified_runner.tool_layer import ToolLayer
from unified_runner.base import RunConfig
from unified_runner.retrieval_skill_inject import (
    load_retrieval_mapping, build_irrelevant_mapping,
    inject_retrieval_skills, build_retrieval_prompt_hint,
    build_top1_skill_text_prompt,
)
from unified_runner.openclaw_compat import (
    append_runtime_context_to_user_prompt,
    build_harbor_runtime_context,
    build_openclaw_system_prompt,
)
from unified_runner.docker_start_gate import docker_start_gate
from unified_runner.docker_lifecycle import docker_label_args, record_lifecycle_event

BASE_DIR = Path(os.environ.get("SKILLRL_ROOT", str(Path(__file__).resolve().parents[3])))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from GeneralAgent.task_exclusions import bad_task_ids

RESULTS_DIR = BASE_DIR / "experiments"
def _docker_env() -> dict[str, str]:
    """Docker CLI env for both single-node tunnel and multi-node Ray actors.

    Historically this file forced ``tcp://127.0.0.1:2375`` because evaluation
    ran from one container with a local tunnel.  The tunnel has proven less
    stable than Docker-over-SSH on the current hosts, so default to the stable
    ``ssh://your-docker-host`` path unless the caller explicitly overrides it.
    """
    env = dict(os.environ)
    env["DOCKER_HOST"] = os.environ.get("DOCKER_HOST", "ssh://your-docker-host")
    return env


DOCKER_ENV = _docker_env()


def _docker_pids_limit_args() -> list[str]:
    """Hard cap per benchmark container to contain runaway fork storms."""
    limit = os.environ.get("UNIFIED_DOCKER_PIDS_LIMIT", "1024").strip()
    if not limit or limit.lower() in {"0", "none", "false", "off"}:
        return []
    return ["--pids-limit", limit]


def _docker_ulimit_fsize_args() -> list[str]:
    """Kernel-enforced single-file size cap inside agent containers.

    Closes the disk-bomb -> kubelet ephemeral-storage eviction path (rl_log
    2026-06-10 20:25): an unbounded single-file writer (dd/fallocate/...) fills
    the node disk and the WHOLE pod gets evicted at ~246GiB free. RLIMIT_FSIZE
    kills only the offending process with SIGXFSZ -- the agent loop just sees
    one failed command (exit 153) and continues. Declared per-task storage tops
    out at 20GB and empirical writable layers at ~6GB, so 32G never touches
    legit tasks. No-op when unset so eval paths are unchanged.
    """
    gb = os.environ.get("UNIFIED_DOCKER_ULIMIT_FSIZE_GB", "").strip()
    if not gb or gb.lower() in {"0", "none", "false", "off"}:
        return []
    return ["--ulimit", f"fsize={int(float(gb) * 1024**3)}"]


def _docker_resource_args() -> list[str]:
    """CPU isolation so an agent-side fork/compile storm (e.g. heavy `pip install`
    spawning nproc-way builds) cannot starve the trainer that is CO-LOCATED on the
    rollout node (reference/actor_fwd NCCL threads, SGLang scheduler).

    UNIFIED_DOCKER_CPUSET pins every benchmark container to a CPU subset (kernel-
    enforced), leaving the complement cores exclusively available to the trainer
    so its NCCL collective threads can never be crowded off-core -> no 30-min
    watchdog DistBackendError under load.  UNIFIED_DOCKER_BUILD_JOBS additionally
    caps build parallelism so one pip/cmake build cannot fan out to nproc jobs.
    Both are no-ops when unset, so eval paths are unaffected.
    """
    args: list[str] = []
    cpuset = os.environ.get("UNIFIED_DOCKER_CPUSET", "").strip()
    if cpuset and cpuset.lower() not in {"0", "none", "false", "off"}:
        args += ["--cpuset-cpus", cpuset]
    jobs = os.environ.get("UNIFIED_DOCKER_BUILD_JOBS", "").strip()
    if jobs.isdigit() and int(jobs) > 0:
        args += [
            "-e", f"MAKEFLAGS=-j{jobs}",
            "-e", f"MAX_JOBS={jobs}",
            "-e", f"CMAKE_BUILD_PARALLEL_LEVEL={jobs}",
            "-e", f"NPY_NUM_BUILD_JOBS={jobs}",
        ]
    return args


def _docker_network_host_enabled() -> bool:
    """unregister_netdevice mitigation (root fix). Run agent containers on the
    HOST network namespace so there is NO per-container netns/veth pair to leak.
    The kernel-5.15 netns refcount leak (IPv6 `lo` "waiting for lo to become
    free") that wedges rtnl_lock on container teardown CANNOT occur without a
    per-container netns. Agent tasks only need OUTBOUND network (clash proxy /
    CN mirrors), which host networking provides via host routing to the docker0 gateway.
    Default ON; set UNIFIED_DOCKER_NETWORK_HOST=0 to revert to docker0 bridge.
    See rl_log 2026-06-09. NOTE: sidecar tasks (fix-visual-stability) keep their
    custom bridge network; host net is skipped for them by the caller.
    """
    v = os.environ.get("UNIFIED_DOCKER_NETWORK_HOST", "1").strip().lower()
    return v not in {"0", "none", "false", "off", ""}


DATASET_PATHS = {
    "skillsbench": BASE_DIR / "datasets/skillsbench/tasks",
    "skillsbench-no-skills": BASE_DIR / "datasets/skillsbench/tasks-no-skills",
    "seta": BASE_DIR / "datasets/seta/dataset/seta_baseline_30",
    "seta-synth": BASE_DIR / "datasets/seta/dataset/synth_data_harbor",
    "tb2": BASE_DIR / "datasets/terminal-bench-v2",
}

# Errors that are flaky-infra (not agent-fault). Worth retrying after cleanup.
# See archive/overnight/logs/migrated_20260428/logs/reports/plan_analysis/20260419_error_distribution_deep.md §5
_FLAKY_RETRY_PATTERNS = [
    r"Failed to start container: Command timed out",
    r"Pull failed for .*",
    r"Conflict\. The container name .* is already in use",
    r"Docker compose command failed",
    r"failed to send non-blocking keys",
    r"dial .*: i/o timeout",
]

# Tasks excluded from evaluation (structural reasons; documented per entry).
# dataset_tag -> set of task_name. When filtering, excluded tasks are removed
# from the N_total denominator entirely (not counted as N_error).
_EXCLUDED_TASKS: dict[str, set[str]] = {
    "skillsbench-with-skills": {
        # Requires Google Cloud / Gmail / Calendar OAuth credentials (task design).
        # See datasets/skillsbench/tasks/scheduling-email-assistant/environment/docker-compose.yaml
        "scheduling-email-assistant",
    },
    "skillsbench-no-skills": {
        "scheduling-email-assistant",
    },
}
_EXCLUDED_TASKS.setdefault("seta-synth", set()).update(bad_task_ids("seta_synth"))
_EXCLUDED_TASKS.setdefault("skillsbench-no-skills", set()).update(bad_task_ids("sb_ns"))


def _is_flaky_error(err_msg: str) -> bool:
    if not err_msg:
        return False
    return any(re.search(p, err_msg) for p in _FLAKY_RETRY_PATTERNS)


def _remove_container_if_exists(cname: str, *, attempts: int = 5) -> bool:
    """Force-remove a named container, waiting until Docker releases the name."""
    for attempt in range(1, attempts + 1):
        inspect_stdout, inspect_stderr, inspect_rc = docker_run(["docker", "inspect", cname], timeout=15)
        if inspect_rc != 0:
            inspect_text = f"{inspect_stdout}\n{inspect_stderr}".lower()
            if "no such object" in inspect_text or "no such container" in inspect_text:
                return True
            print(
                f"    [cleanup] docker inspect {cname} failed attempt {attempt}/{attempts}: "
                f"{inspect_stderr[:200]}"
            )
            time.sleep(min(2 * attempt, 10))
            continue

        _, stderr, rm_rc = docker_run(["docker", "rm", "-f", cname], timeout=60)
        if rm_rc != 0:
            print(f"    [cleanup] docker rm -f {cname} failed attempt {attempt}/{attempts}: {stderr[:200]}")
            record_lifecycle_event("rm_failed", container=cname, attempt=attempt, stderr=stderr[:500])
        time.sleep(min(2 * attempt, 10))

    inspect_stdout, inspect_stderr, inspect_rc = docker_run(["docker", "inspect", cname], timeout=15)
    if inspect_rc == 0:
        print(f"    [cleanup] stale container still exists after {attempts} attempts: {cname}")
        record_lifecycle_event("rm_still_exists", container=cname, attempts=attempts)
        return False
    inspect_text = f"{inspect_stdout}\n{inspect_stderr}".lower()
    if inspect_text.strip() and "no such object" not in inspect_text and "no such container" not in inspect_text:
        print(f"    [cleanup] final docker inspect {cname} failed as infra error: {inspect_stderr[:200]}")
        record_lifecycle_event("inspect_failed_after_rm", container=cname, stderr=inspect_stderr[:500])
        return False
    return True


def _remove_containers_by_name_prefix(prefix: str, *, max_remove: int = 8) -> int:
    """Remove stale containers whose names start with a known RL prefix.

    RL container names include per-sample salts such as
    ``...-p<PID>-i<idx>-...-a<attempt>``.  Older retry cleanup used the shorter
    ``...-p<PID>`` name and therefore missed real retry attempts.
    """
    stdout, stderr, rc = docker_run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name={prefix}",
            "--format",
            "{{.Names}}",
        ],
        timeout=20,
    )
    if rc != 0:
        print(f"    [cleanup] docker ps for prefix {prefix} failed: {stderr[:200]}")
        record_lifecycle_event("prefix_list_failed", container=prefix, stderr=stderr[:500])
        return 0
    removed = 0
    for name in stdout.splitlines():
        name = name.strip()
        if not name.startswith(prefix):
            continue
        if _remove_container_if_exists(name, attempts=2):
            removed += 1
        if removed >= max_remove:
            break
    return removed


def _cleanup_stale(task_name: str, dataset_tag: str) -> None:
    """Best-effort cleanup before retrying a flaky task.

    Removes any stale container matching this task's name pattern, plus
    any leftover docker compose project (for tasks with docker-compose.yaml).
    Failures are ignored — this runs before a retry that will re-create.
    """
    prefix = f"u-{dataset_tag}-" if dataset_tag else "unified-"
    # Match the per-PID naming used in start_container (2026-04-26 fix).
    cname = f"{prefix}{task_name}-p{os.getpid()}"
    _remove_container_if_exists(cname)
    _remove_containers_by_name_prefix(f"{cname}-", max_remove=8)
    # compose project: lowercase, hyphen-preserving, strip underscores (docker compose convention)
    proj = task_name.lower().replace("_", "").replace(".", "")
    docker_run(
        ["docker", "compose", "-p", proj, "down", "--remove-orphans", "--volumes"],
        timeout=30)

def _coerce_proc_output(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def docker_run(cmd, timeout=120):
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_docker_env(),
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        return stdout, stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        timeout_stdout = _coerce_proc_output(
            getattr(exc, "stdout", None) or getattr(exc, "output", None)
        )
        timeout_stderr = _coerce_proc_output(getattr(exc, "stderr", None))
        if proc is not None:
            try:
                os.killpg(proc.pid, 15)
                final_stdout, final_stderr = proc.communicate(timeout=5)
                timeout_stdout = _coerce_proc_output(final_stdout) or timeout_stdout
                timeout_stderr = _coerce_proc_output(final_stderr) or timeout_stderr
            except subprocess.TimeoutExpired as exc2:
                timeout_stdout = _coerce_proc_output(
                    getattr(exc2, "stdout", None) or getattr(exc2, "output", None)
                ) or timeout_stdout
                timeout_stderr = _coerce_proc_output(
                    getattr(exc2, "stderr", None)
                ) or timeout_stderr
                try:
                    os.killpg(proc.pid, 9)
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
            except Exception as kill_exc:
                timeout_stderr = (
                    timeout_stderr
                    + f"\nFailed to collect output after timeout: {kill_exc!r}"
                ).strip()
                try:
                    os.killpg(proc.pid, 9)
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
        timeout_stderr = (timeout_stderr + "\nCommand timed out").strip()
        return timeout_stdout, timeout_stderr, -1


_DOCKER_ENDPOINT_ERROR_PATTERNS = (
    "Cannot connect to the Docker daemon",
    "error during connect",
    "Connection timed out during banner exchange",
    "Connection to UNKNOWN port",
    "context deadline exceeded",
    "connection reset by peer",
    "broken pipe",
)


def _is_docker_endpoint_error(stdout: str = "", stderr: str = "") -> bool:
    text = f"{stdout}\n{stderr}"
    return any(pattern in text for pattern in _DOCKER_ENDPOINT_ERROR_PATTERNS)


def _docker_image_id(image_tag: str, *, attempts: int = 4, timeout: int = 20) -> str:
    """Return local image id, separating Docker endpoint failures from absence.

    In high-concurrency RL the tunnel to the remote Docker host can flap for a few seconds.
    Treating that as "image missing" causes false ABORTED samples and refill
    storms. Only an available Docker endpoint with empty output means missing.
    """
    last_stdout = ""
    last_stderr = ""
    for attempt in range(1, attempts + 1):
        stdout, stderr, rc = docker_run(
            ["docker", "images", "-q", image_tag], timeout=timeout
        )
        last_stdout, last_stderr = stdout, stderr
        if rc == 0:
            return stdout.strip()
        if _is_docker_endpoint_error(stdout, stderr):
            time.sleep(min(2 * attempt, 8))
            continue
        raise RuntimeError(
            f"docker images failed for {image_tag}: {(stderr or stdout)[-300:]}"
        )
    raise RuntimeError(
        f"Docker endpoint unavailable while checking local image {image_tag}: "
        f"{(last_stderr or last_stdout)[-300:]}"
    )


def list_tasks(dataset_path):
    """List tasks in a dataset directory."""
    tasks = []
    for d in sorted(dataset_path.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        # Check for task definition
        if (d / "task.toml").exists() or (d / "instruction.md").exists():
            tasks.append(d.name)
    return tasks


def get_instruction(task_dir):
    """Read task instruction from instruction.md."""
    instruction_file = task_dir / "instruction.md"
    if instruction_file.exists():
        return instruction_file.read_text(encoding="utf-8")
    # Fallback: look for README or any .md file
    for md in task_dir.glob("*.md"):
        if md.name != "task.toml":
            return md.read_text(encoding="utf-8")
    return "Complete the task as described in the files within the working directory."


def _read_task_toml(task_dir):
    toml_path = task_dir / "task.toml"
    if not toml_path.exists():
        return {}
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        return tomllib.loads(toml_path.read_text())
    except Exception:
        return {}


def resolve_image(task_dir, task_name, dataset):
    """Resolve the Docker image for a task.

    If task.toml declares `environment.docker_image` (TB 2.0 style), use it
    directly (assume pre-pulled; if missing, pull-on-demand). Otherwise fall
    back to building from `environment/Dockerfile`.
    """
    meta = _read_task_toml(task_dir)
    declared = meta.get("environment", {}).get("docker_image")
    if declared:
        # Ensure image exists locally; if not, pull (costs clash bandwidth for
        # alexgshaw/* etc; the caller is responsible for pre-pulling to save).
        image_id = _docker_image_id(declared)
        if not image_id:
            print(f"    Pulling {declared} (not cached)...")
            _, stderr, rc = docker_run(
                ["docker", "pull", declared], timeout=600)
            if rc != 0:
                raise RuntimeError(f"Pull failed for {declared}: {stderr[-300:]}")
        return declared
    return build_image(task_dir, task_name, dataset)


def build_image(task_dir, task_name, dataset):
    """Build Docker image for a task."""
    env_dir = task_dir / "environment"
    dockerfile = env_dir / "Dockerfile"
    if not dockerfile.exists():
        raise FileNotFoundError(f"No Dockerfile in {env_dir}")

    image_tag = f"unified-{dataset}-{task_name}:latest"

    # Check if image already exists
    image_id = _docker_image_id(image_tag)
    if image_id:
        print(f"    Image {image_tag} already exists, reusing")
        return image_tag
    require_prebuilt = os.environ.get(
        "UNIFIED_HARBOR_REQUIRE_PREBUILT_LOCAL", "0"
    ).lower() in {"1", "true", "yes"}
    if require_prebuilt:
        raise RuntimeError(
            f"Local build image {image_tag} is missing and "
            "UNIFIED_HARBOR_REQUIRE_PREBUILT_LOCAL=1; aborting setup instead of "
            "building online during RL rollout"
        )

    print(f"    Building image {image_tag}...")
    # Inject clash proxy build-args so RUN wget/curl/apt can reach external
    # resources during image build (fonts.gstatic, github, apache, etc).
    # The proxy is on the Docker host :8888; use host-gateway so build container can reach it.
    # build 阶段 buildkit 有独立 network namespace。默认 Docker 主机 docker0 代理
    # (your-docker-gateway:8888);节点/拓扑变更后用 UNIFIED_CONTAINER_PROXY 覆盖成可达代理
    # (本节点 = squid http://your-proxy:3128,外网 IP,buildkit netns 经 host 路由可达)。
    proxy_url = os.environ.get("UNIFIED_CONTAINER_PROXY", "http://your-docker-gateway:8888").strip()
    no_proxy = CONTAINER_PROXY_NO
    build_dockerfile, tmp_dir = _prepare_build_dockerfile(dockerfile)
    try:
        build_cmd = [
            "docker", "build",
            "--build-arg", f"HTTP_PROXY={proxy_url}",
            "--build-arg", f"HTTPS_PROXY={proxy_url}",
            "--build-arg", f"http_proxy={proxy_url}",
            "--build-arg", f"https_proxy={proxy_url}",
            "--build-arg", f"NO_PROXY={no_proxy}",
            "--build-arg", f"no_proxy={no_proxy}",
            "-t", image_tag, "-f", str(build_dockerfile), str(env_dir),
        ]
        build_timeout = int(os.environ.get("UNIFIED_HARBOR_BUILD_TIMEOUT_SEC", "300"))
        stdout, stderr, rc = docker_run(build_cmd, timeout=build_timeout)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    if rc != 0:
        raise RuntimeError(f"Docker build failed (timeout={build_timeout}s): {stderr[-500:]}")
    print(f"    Image built successfully")
    return image_tag


def _prepare_build_dockerfile(dockerfile: Path) -> tuple[Path, str | None]:
    """Return a Dockerfile path patched for China apt mirrors, if useful.

    Some generated SETA/SB/TB Dockerfiles run ``apt-get update`` against
    archive.ubuntu.com. During Docker build, the remote host's proxy can return
    502 for those apt endpoints. We keep the task Dockerfiles untouched and
    build from a temporary Dockerfile that rewrites Ubuntu apt sources to the
    HTTP Tsinghua mirror immediately after each FROM. HTTP is deliberate:
    minimal Ubuntu images may not have CA certificates before apt runs.
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
    if "RUN python3" in text and "install -y" not in text[text.find("FROM "):text.find("RUN python3")]:
        snippet += (
            "\nRUN set -eux; \\\n"
            "    apt-get update; \\\n"
            "    apt-get install -y --no-install-recommends python3 ca-certificates; \\\n"
            "    rm -rf /var/lib/apt/lists/*\n"
        )
    out: list[str] = []
    heredoc_end: str | None = None
    heredoc_start_re = re.compile(
        r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_-]*)['\"]?"
    )
    from_re = re.compile(r"^\s*FROM\s+\S+", re.IGNORECASE)
    for line in text.splitlines(keepends=True):
        out.append(line)

        stripped = line.strip()
        if heredoc_end is not None:
            if stripped == heredoc_end:
                heredoc_end = None
            continue

        heredoc_match = heredoc_start_re.search(line)
        if heredoc_match:
            heredoc_end = heredoc_match.group(1)
            continue

        if from_re.match(line):
            out.append(snippet + "\n")

    tmp_dir = tempfile.mkdtemp(prefix="unified-harbor-dockerfile-")
    patched = Path(tmp_dir) / "Dockerfile"
    patched.write_text("".join(out), encoding="utf-8")
    return patched, tmp_dir


PKG_CACHE_DIR = BASE_DIR / "ops/cache/pkg"
APT_SOURCES_DIR = PKG_CACHE_DIR / "apt_sources"
PIP_CONF_PATH = PKG_CACHE_DIR / "pip.conf"


def _inject_cn_apt_sources(cname):
    """Replace container's apt sources with Tsinghua mirror when distro is known.

    Direct archive.ubuntu.com / deb.debian.org traffic often falls through
    Clash. Use per-container mirror injection so task images stay untouched and
    external-network fallback remains available for non-mirror domains.
    """
    # Detect Debian/Ubuntu version inside container.
    stdout, _, rc = docker_run(
        ["docker", "exec", cname, "bash", "-c", ". /etc/os-release && echo $VERSION_CODENAME"],
        timeout=10)
    codename = stdout.strip() if rc == 0 else ""

    if codename == "noble":  # 24.04
        src = APT_SOURCES_DIR / "ubuntu.sources.noble"
        dst = "/etc/apt/sources.list.d/ubuntu.sources"
        # noble uses .sources format; also blank out the default /etc/apt/sources.list
        docker_run(["docker", "exec", cname, "sh", "-c", "true > /etc/apt/sources.list"], timeout=5)
    elif codename in ("focal", "jammy"):  # 20.04 / 22.04
        fname = f"sources.list.{codename}" if codename != "focal" else "sources.list.focal"
        src = APT_SOURCES_DIR / fname
        if not src.exists():
            src = APT_SOURCES_DIR / "sources.list.focal"
        dst = "/etc/apt/sources.list"
    elif codename in ("bookworm", "trixie"):  # Debian 12 / 13
        src = APT_SOURCES_DIR / f"sources.list.{codename}"
        dst = "/etc/apt/sources.list"
        # Debian images commonly use deb822 `/etc/apt/sources.list.d/debian.sources`.
        # Disable it before adding our mirror list; otherwise apt still contacts
        # deb.debian.org in addition to the mirror.
        docker_run(
            ["docker", "exec", cname, "sh", "-c",
             "for f in /etc/apt/sources.list.d/debian.sources; do "
             "[ -f \"$f\" ] && mv \"$f\" \"$f.disabled-cn-mirror\" || true; done"],
            timeout=5)
    else:
        # Unknown / non-Debian; skip
        print(f"    [cn-apt] skipping unknown codename={codename!r}")
        return
    if not src.exists():
        print(f"    [cn-apt] source file {src} missing, skipping")
        return
    for attempt in range(1, 4):
        _, stderr, rc = docker_run(["docker", "cp", str(src), f"{cname}:{dst}"], timeout=120)
        if rc == 0:
            break
        print(f"    [cn-apt] docker cp failed attempt {attempt}/3: {stderr[:200]}")
        time.sleep(min(3 * attempt, 10))
    else:
        return
    print(f"    [cn-apt] injected Tsinghua mirror ({codename}) -> {dst}")


def _inject_language_mirror_defaults(cname):
    """Best-effort defaults for package managers that respect env/profile.

    These are conservative: they change default mirrors only. Explicit URLs in
    task commands still work through Clash, preserving task score when a task
    genuinely needs a non-mirror external resource.
    """
    script = r'''
set -eu
mkdir -p /etc/pip /root/.config/pip /root/.pip
cat >/etc/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
timeout = 120
retries = 3
EOF
cp /etc/pip.conf /root/.config/pip/pip.conf
cp /etc/pip.conf /root/.pip/pip.conf

cat >/etc/profile.d/agent-cn-mirrors.sh <<'EOF'
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENDPOINT="${HF_HUB_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HUGGINGFACE_HUB_ENDPOINT:-https://hf-mirror.com}"
EOF

for f in /etc/R/Rprofile.site /usr/lib/R/etc/Rprofile.site /usr/local/lib/R/etc/Rprofile.site; do
  d=$(dirname "$f")
  if [ -d "$d" ]; then
    touch "$f"
    if ! grep -q "mirrors.tuna.tsinghua.edu.cn/CRAN" "$f"; then
      cat >>"$f" <<'EOF'
local({
  r <- getOption("repos")
  r["CRAN"] <- "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"
  options(repos = r)
})
EOF
    fi
  fi
done
'''
    _, stderr, rc = docker_run(["docker", "exec", cname, "sh", "-lc", script], timeout=120)
    if rc != 0:
        print(f"    [cn-mirror-env] inject failed: {stderr[:200]}")
        return
    print("    [cn-mirror-env] default pip/uv/HF/CRAN mirrors configured")


def _inject_cn_pip_config(cname):
    """Inject Tsinghua pip index into container at /etc/pip.conf (system-wide)."""
    if not PIP_CONF_PATH.exists():
        print(f"    [cn-pip] {PIP_CONF_PATH} missing, skipping")
        return
    # Create /etc for images that don't have it (rare)
    docker_run(["docker", "exec", cname, "mkdir", "-p", "/etc"], timeout=5)
    _, stderr, rc = docker_run(
        ["docker", "cp", str(PIP_CONF_PATH), f"{cname}:/etc/pip.conf"], timeout=15)
    if rc != 0:
        print(f"    [cn-pip] docker cp failed: {stderr[:200]}")
        return
    # Also put it in root's home for pip versions that ignore /etc/pip.conf
    docker_run(
        ["docker", "exec", cname, "mkdir", "-p", "/root/.config/pip"], timeout=5)
    docker_run(
        ["docker", "cp", str(PIP_CONF_PATH), f"{cname}:/root/.config/pip/pip.conf"],
        timeout=15)
    # Legacy location (pip <20)
    docker_run(["docker", "exec", cname, "mkdir", "-p", "/root/.pip"], timeout=5)
    docker_run(
        ["docker", "cp", str(PIP_CONF_PATH), f"{cname}:/root/.pip/pip.conf"], timeout=15)
    print(f"    [cn-pip] injected Tsinghua pypi mirror -> /etc/pip.conf + ~/.config/pip + ~/.pip")


TB2_UV_CACHE_TARBALL = PKG_CACHE_DIR / "tb2_uv_cache.tar.gz"
HARBOR_UV_BINARIES_TARBALL = PKG_CACHE_DIR / "harbor_uv_binaries.tar.gz"
TB2_UV_CACHE_REMOTE_DIR = os.environ.get(
    "TB2_UV_CACHE_REMOTE_DIR",
    "/ext1/tmp/tb2_uv_cache/tb2-uv",
)


def _tb2_uv_bind_mount_enabled() -> bool:
    """Whether TB2 containers should bind-mount the remote prebaked uv cache.

    The Docker daemon runs remotely, so a `-v` source path must exist on the
    remote Docker host, not on this client.  Keep this behind an explicit env
    switch so fresh hosts without the prepared `/ext1/tmp/...` cache fall back
    to the older docker-cp path instead of silently mounting an empty directory.
    """
    return os.environ.get("TB2_UV_CACHE_BIND_MOUNT", "0").lower() in {"1", "true", "yes", "on"}


def _install_tb2_pip_guard(cname: str) -> None:
    """Install a narrow pip wrapper that blocks slow agent-side torch installs."""
    wrapper = r"""if [ -x /usr/bin/pip3 ] && [ ! -e /usr/bin/pip3.real ]; then
  cp -a /usr/bin/pip3 /usr/bin/pip3.real || true
fi
cat >/tmp/tb2_pip3_wrapper <<'SH'
#!/bin/sh
case " $* " in
  *" install "*" torch"*|*" install "*" torch=="*)
    echo "[tb2-uv] slow pip install torch is disabled in this benchmark container." >&2
    echo "[tb2-uv] Use /opt/tb2-uv/uvx -p 3.13 -w torch==2.7.0 ... or write the solution and let /tests/test.sh run with the prebaked torch cache." >&2
    exit 1
    ;;
esac
if [ -x /usr/bin/pip3.real ]; then
  exec /usr/bin/pip3.real "$@"
fi
exec python3 -m pip "$@"
SH
install -m 0755 /tmp/tb2_pip3_wrapper /usr/local/bin/pip3
install -m 0755 /tmp/tb2_pip3_wrapper /usr/bin/pip3
chmod +x /usr/local/bin/pip3
ln -sf /usr/local/bin/pip3 /usr/local/bin/pip
ln -sf /usr/bin/pip3 /usr/bin/pip || true
"""
    docker_run(["docker", "exec", cname, "sh", "-c", wrapper], timeout=15)


def _inject_harbor_uv_binaries(cname, dataset_tag=""):
    """Inject small uv/uvx binaries for SkillsBench verifier scripts.

    Many SB verifier scripts install uv from `https://astral.sh/uv/...` before
    running `uvx`.  Under high RL concurrency that network hop is a frequent
    502/SSL failure source.  We avoid the network install but still let `uvx`
    resolve task-specific dependencies normally through the configured mirrors.

    This is intentionally much smaller than the TB2 uv cache: the full TB2
    cache contains torch wheels and is too expensive to copy into every SB
    container.  Heavy SB tasks still need targeted image/cache work; this fixes
    the common uv bootstrap failure.
    """
    if not dataset_tag.startswith("skillsbench"):
        return
    if not HARBOR_UV_BINARIES_TARBALL.exists():
        print(f"    [harbor-uv] tarball missing at {HARBOR_UV_BINARIES_TARBALL}; verifier may install uv online")
        return
    for attempt in range(1, 4):
        _, stderr, rc = docker_run(
            ["docker", "cp", str(HARBOR_UV_BINARIES_TARBALL), f"{cname}:/harbor_uv_binaries.tar.gz"],
            timeout=120)
        if rc == 0:
            break
        print(f"    [harbor-uv] docker cp failed attempt {attempt}/3: {stderr[:200]}")
        time.sleep(min(3 * attempt, 10))
    else:
        return
    _, stderr, rc = docker_run(
        ["docker", "exec", cname, "sh", "-lc",
         "mkdir -p /opt && cd /opt && tar xzf /harbor_uv_binaries.tar.gz && "
         "rm /harbor_uv_binaries.tar.gz && chmod +x /opt/tb2-uv/uv /opt/tb2-uv/uvx"],
        timeout=120)
    if rc != 0:
        print(f"    [harbor-uv] extract failed: {stderr[:200]}")
        return
    print("    [harbor-uv] local uv/uvx injected at /opt/tb2-uv/")


def _inject_tb2_uv_cache(cname, dataset_tag=""):
    """Inject prebaked uv+cpython-3.13.9+deps cache into TB2 containers.

    Only activates for tb2 tasks (dataset_tag=='tb2'). Tarball contains:
      - /opt/tb2-uv/uv, /opt/tb2-uv/uvx
      - /opt/tb2-uv/uv-data/python/cpython-3.13.9-...  (via UV_PYTHON_INSTALL_DIR)
      - /opt/tb2-uv/uv-data/cache/...                   (via UV_CACHE_DIR)

    Patched test.sh (see datasets/terminal-bench-v2/*/tests/test.sh 32 patched files)
    checks for /opt/tb2-uv/uv and uses it directly. If tarball missing or inject
    fails, the test.sh fallback kicks in (re-downloads via clash, slow but works).

    Run `bash GeneralAgent/eval_scripts/tb2_eval/prebake_uv_cache.sh` once to
    create the tarball.
    """
    if dataset_tag != "tb2":
        return
    stdout, _, rc = docker_run(
        ["docker", "exec", cname, "sh", "-lc", "test -x /opt/tb2-uv/uv && test -x /opt/tb2-uv/uvx && test -d /opt/tb2-uv/data"],
        timeout=15,
    )
    if rc == 0:
        _install_tb2_pip_guard(cname)
        print(f"    [tb2-uv] prebaked uv+python+deps mounted at /opt/tb2-uv/")
        return
    if not TB2_UV_CACHE_TARBALL.exists():
        print(f"    [tb2-uv] tarball missing at {TB2_UV_CACHE_TARBALL}; test.sh will fall back to online install")
        return
    for attempt in range(1, 4):
        _, stderr, rc = docker_run(
            ["docker", "cp", str(TB2_UV_CACHE_TARBALL), f"{cname}:/tb2_uv_cache.tar.gz"],
            timeout=600)
        if rc == 0:
            break
        print(f"    [tb2-uv] docker cp failed attempt {attempt}/3: {stderr[:200]}")
        time.sleep(min(5 * attempt, 15))
    else:
        return
    for attempt in range(1, 4):
        _, stderr, rc = docker_run(
            ["docker", "exec", cname, "sh", "-c",
             "cd /opt && tar xzf /tb2_uv_cache.tar.gz && rm /tb2_uv_cache.tar.gz"],
            timeout=600)
        if rc == 0:
            break
        print(f"    [tb2-uv] extract failed attempt {attempt}/3: {stderr[:200]}")
        time.sleep(min(5 * attempt, 15))
    else:
        return
    _install_tb2_pip_guard(cname)
    print(f"    [tb2-uv] prebaked uv+python+deps injected at /opt/tb2-uv/")


def _inject_cn_mirrors(cname, dataset_tag=""):
    """Inject apt + pip China mirrors + (for tb2) prebaked uv cache."""
    try:
        _inject_language_mirror_defaults(cname)
    except Exception as exc:
        print(f"    [cn-mirror-env] inject failed: {exc}")
    try:
        _inject_cn_apt_sources(cname)
    except Exception as exc:
        print(f"    [cn-apt] inject failed: {exc}")
    try:
        _inject_cn_pip_config(cname)
    except Exception as exc:
        print(f"    [cn-pip] inject failed: {exc}")
    try:
        _inject_tb2_uv_cache(cname, dataset_tag=dataset_tag)
    except Exception as exc:
        print(f"    [tb2-uv] inject failed: {exc}")
    try:
        _inject_harbor_uv_binaries(cname, dataset_tag=dataset_tag)
    except Exception as exc:
        print(f"    [harbor-uv] inject failed: {exc}")


# Container outbound proxy. Default is the Docker host's docker0 proxy (legacy topology).
# Override with UNIFIED_CONTAINER_PROXY when the node's reachable clash differs
# (e.g. after a node/IP change your-docker-gateway:8888 is unroutable -> use the host's
# ShellCrash clash; with --network host that is http://127.0.0.1:7890). rl_log 2026-06-09.
HOST_CLASH_PROXY = os.environ.get("UNIFIED_CONTAINER_PROXY", "http://your-docker-gateway:8888").strip()
# NO_PROXY includes CN mirrors + internal so they bypass clash (saves ~20-25 GB/run).
# Keep in sync with ops/cache/pkg/compose-proxy-override.yaml
CONTAINER_PROXY_NO = (
    "localhost,127.0.0.1,0.0.0.0,::1,api,"
    "mirrors.aliyun.com,mirrors.cloud.aliyuncs.com,"
    "pypi.tuna.tsinghua.edu.cn,mirrors.tuna.tsinghua.edu.cn,"
    "mirrors.ustc.edu.cn,mirror.nju.edu.cn,mirrors.bfsu.edu.cn,"
    "mirrors.163.com,mirrors.huaweicloud.com,mirrors.cernet.edu.cn,"
    "hf-mirror.com,"
    ".aliyun.com,.tsinghua.edu.cn,.ustc.edu.cn,.nju.edu.cn,"
    ".bfsu.edu.cn,.cernet.edu.cn,.163.com,.huaweicloud.com,.hf-mirror.com"
)


def start_container(
    image_tag,
    task_name,
    dataset_tag="",
    inject_cn_mirror=True,
    container_suffix=None,
):
    """Start a Docker container for a task.

    Injects the Docker host proxy (http://host.docker.internal:8888) via env +
    --add-host so container can reach external GitHub / PyPI / HuggingFace.
    The proxy instance is bound to 0.0.0.0:8888 on the Docker host.

    2026-04-26: container name now includes our PID. SFT data collection
    spawns one runner subprocess per trial; multiple trials of the same task
    can run concurrently, and a fixed name like `u-tb2-foo` would collide
    ("Conflict. The container name ... is already in use"). Adding `-p<PID>`
    makes the name per-trial unique. Legacy single-process eval is unaffected
    (task_name already varied between calls).
    """
    prefix = f"u-{dataset_tag}-" if dataset_tag else "unified-"
    suffix = container_suffix or f"p{os.getpid()}"
    cname = f"{prefix}{task_name}-{suffix}"
    needs_visual_api = dataset_tag == "skillsbench-no-skills" and task_name == "fix-visual-stability"
    network_name = f"{cname}-net"
    api_cname = f"{cname}-api"
    if not _remove_container_if_exists(cname):
        raise RuntimeError(f"Failed to cleanup stale container before start: {cname}")
    if needs_visual_api:
        _remove_container_if_exists(api_cname)
        docker_run(["docker", "network", "rm", network_name], timeout=30)
        api_image = "skillsbench-visual-stability-api:latest"
        _, image_stderr, image_rc = docker_run(["docker", "image", "inspect", api_image], timeout=30)
        if image_rc != 0:
            raise RuntimeError(
                f"Missing required sidecar image {api_image} for {task_name}; "
                f"prebuild/load it before running RL/eval. docker inspect stderr={image_stderr[:300]}"
            )
        _, net_stderr, net_rc = docker_run(["docker", "network", "create", network_name], timeout=30)
        if net_rc != 0:
            raise RuntimeError(f"Failed to create sidecar network {network_name}: {net_stderr[:300]}")
        _, api_stderr, api_rc = docker_run(
            [
                "docker", "run", "-d", "--name", api_cname,
                *docker_label_args(
                    bench="harbor-sidecar",
                    dataset_tag=dataset_tag,
                    task_name=task_name,
                    container_name=api_cname,
                    container_suffix=f"{suffix}-api",
                ),
                *_docker_pids_limit_args(),
                *_docker_ulimit_fsize_args(),
                *_docker_resource_args(),
                "--network", network_name,
                "--network-alias", "api",
                api_image,
            ],
            timeout=120,
        )
        if api_rc != 0:
            docker_run(["docker", "network", "rm", network_name], timeout=30)
            raise RuntimeError(f"Failed to start sidecar {api_cname}: {api_stderr[:300]}")
        api_ip, api_ip_stderr, api_ip_rc = docker_run(
            [
                "docker", "inspect", "-f",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                api_cname,
            ],
            timeout=30,
        )
        api_ip = (api_ip or "").strip()
        if api_ip_rc != 0 or not api_ip:
            _remove_container_if_exists(api_cname)
            docker_run(["docker", "network", "rm", network_name], timeout=30)
            raise RuntimeError(
                f"Failed to inspect sidecar IP for {api_cname}: {api_ip_stderr[:300]}"
            )
    volume_args = []
    if dataset_tag == "tb2" and _tb2_uv_bind_mount_enabled():
        volume_args = ["-v", f"{TB2_UV_CACHE_REMOTE_DIR}:/opt/tb2-uv:ro"]
    # unregister_netdevice root fix: host network -> no per-container netns/veth.
    use_host_net = (not needs_visual_api) and _docker_network_host_enabled()
    network_args = []
    addhost_args = ["--add-host", "host.docker.internal:host-gateway"]
    if needs_visual_api:
        network_args = [
            "--network", network_name,
            "--network-alias", "main",
            "--add-host", f"api:{api_ip}",
        ]
    elif use_host_net:
        # host netns: nothing to leak. `--add-host ...:host-gateway` is bridge-only
        # and rejected under `--network host`; the clash proxy (your-docker-gateway:8888) is
        # still reachable from the host network namespace via host routing.
        network_args = ["--network", "host"]
        addhost_args = []
    with docker_start_gate(f"harbor:{dataset_tag}:{task_name}"):
        stdout, stderr, rc = docker_run(
            [
                "docker", "run", "-d", "--name", cname,
                *docker_label_args(
                    bench="harbor",
                    dataset_tag=dataset_tag,
                    task_name=task_name,
                    container_name=cname,
                    container_suffix=suffix,
                ),
                *_docker_pids_limit_args(),
                *_docker_ulimit_fsize_args(),
                *_docker_resource_args(),
                *volume_args,
                *network_args,
                *addhost_args,
                "-e", f"HTTP_PROXY={HOST_CLASH_PROXY}",
                "-e", f"HTTPS_PROXY={HOST_CLASH_PROXY}",
                "-e", f"http_proxy={HOST_CLASH_PROXY}",
                "-e", f"https_proxy={HOST_CLASH_PROXY}",
                "-e", f"NO_PROXY={CONTAINER_PROXY_NO}",
                "-e", f"no_proxy={CONTAINER_PROXY_NO}",
                "-e", "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple",
                "-e", "UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple",
                "-e", "UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple",
                # HuggingFace 国内镜像 (不走 clash)
                "-e", "HF_ENDPOINT=https://hf-mirror.com",
                "-e", "HF_HUB_ENDPOINT=https://hf-mirror.com",
                "-e", "HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com",
                image_tag, "sleep", "infinity",
            ],
            timeout=600)
    if rc != 0:
        record_lifecycle_event("start_failed", container=cname, stderr=stderr[:1000], rc=rc)
        _remove_container_if_exists(cname, attempts=2)
        if needs_visual_api:
            _remove_container_if_exists(api_cname)
            docker_run(["docker", "network", "rm", network_name], timeout=30)
        raise RuntimeError(f"Failed to start container: {stderr}")
    if inject_cn_mirror:
        _inject_cn_mirrors(cname, dataset_tag=dataset_tag)
    return cname


def stop_container(cname):
    # Graceful teardown (③, unregister_netdevice mitigation): `docker stop -t 3`
    # (SIGTERM -> 3s -> SIGKILL) before remove, so the container's sockets/conntrack
    # release before docker tears down its netns -- lowers the rtnl_lock wedge risk
    # that immediate `rm -f` can trigger on kernel 5.15. Bounded to 3s so teardown
    # stays fast; the persistent `docker exec` shell is already closed by env.close()
    # before this runs, so stop won't hang on it. Host-networked containers have no
    # own netns (UNIFIED_DOCKER_NETWORK_HOST=1), so this is belt-and-suspenders.
    # See rl_log 2026-06-09.
    docker_run(["docker", "stop", "-t", "3", cname], timeout=20)
    _, stderr, rc = docker_run(["docker", "rm", "-f", cname], timeout=60)
    ok = rc == 0
    if not ok:
        print(f"    [cleanup] docker rm -f {cname} failed during teardown: {stderr[:200]}")
        record_lifecycle_event("teardown_failed", container=cname, stderr=stderr[:1000], rc=rc)
    if "fix-visual-stability" in cname:
        _, api_stderr, api_rc = docker_run(["docker", "rm", "-f", f"{cname}-api"], timeout=60)
        if api_rc != 0:
            ok = False
            record_lifecycle_event("teardown_sidecar_failed", container=f"{cname}-api", stderr=api_stderr[:1000], rc=api_rc)
        docker_run(["docker", "network", "rm", f"{cname}-net"], timeout=30)
    if ok:
        record_lifecycle_event("teardown_ok", container=cname)
    return ok


def copy_tests(task_dir, cname):
    """Copy test files into the container."""
    tests_dir = task_dir / "tests"
    if not tests_dir.exists():
        return False
    # Copy tests directory to container
    last_stdout = ""
    last_stderr = ""
    for attempt in range(1, 4):
        stdout, stderr, rc = docker_run(
            ["docker", "cp", str(tests_dir), f"{cname}:/tests"],
            timeout=60)
        last_stdout, last_stderr = stdout, stderr
        if rc == 0:
            return True
        if _is_docker_endpoint_error(stdout, stderr):
            time.sleep(min(2 * attempt, 8))
            continue
        raise RuntimeError(
            f"docker cp tests failed for {cname}: {(stderr or stdout)[-300:]}"
        )
    raise RuntimeError(
        f"Docker endpoint unavailable while copying tests into {cname}: "
        f"{(last_stderr or last_stdout)[-300:]}"
    )


def _read_verifier_timeout(task_dir, default=600):
    """Read verifier.timeout_sec from task.toml; fall back to default if not present."""
    toml_path = task_dir / "task.toml"
    timeout = default
    if not toml_path.exists():
        pass
    else:
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore
            data = tomllib.loads(toml_path.read_text())
            v = data.get("verifier", {}).get("timeout_sec")
            if v:
                timeout = int(float(v))
        except Exception:
            pass
    cap = os.environ.get("UNIFIED_VERIFIER_TIMEOUT_CAP_SEC")
    if cap:
        try:
            timeout = min(timeout, max(1, int(float(cap))))
        except ValueError:
            pass
    return timeout


def run_verifier(cname, timeout_sec=600):
    """Run the verifier (test.sh) inside the container and return (reward, output, ok).

    `ok` is False if verifier itself timed out / errored — distinguishes
    "verifier crashed" from "agent solution scored 0".
    """
    # Check for test.sh.  Under Relax's multi-env concurrency, the remote Docker daemon can
    # occasionally spend >60s just accepting an otherwise trivial docker exec.
    # Retry before classifying it as verifier infra failure.
    test_stderr = ""
    rc = -1
    for attempt in range(1, 4):
        _, test_stderr, rc = docker_run(
            ["docker", "exec", cname, "test", "-f", "/tests/test.sh"], timeout=120)
        if rc == 0:
            break
        if rc == -1 and attempt < 3:
            time.sleep(2 * attempt)
            continue
        break
    if rc != 0:
        if rc == -1:
            return 0.0, "test.sh existence check timed out after retries: Command timed out", False
        return 0.0, f"No test.sh found{': ' + test_stderr[-500:] if test_stderr else ''}", False

    # Run the verifier with the per-task timeout (or default 600s).
    #
    # TB2 test.sh files were patched to use a prebaked uv cache at /opt/tb2-uv,
    # but some scripts still copy /opt/tb2-uv/uv over /usr/local/bin/uv. Under
    # high Relax concurrency, that can fail with "Text file busy" while another
    # verifier process is already running uv. Patch the copied test script
    # inside the container just before execution: prefer PATH=/opt/tb2-uv and
    # keep the same UV_* cache envs, without mutating /usr/local/bin.
    verifier_cmd = ("SETA_CONTINUOUS_REWARD=%s\n" % (
        "1" if os.environ.get("SETA_CONTINUOUS_REWARD") == "1" else "0")) + r'''
set -e
mkdir -p /logs/verifier
rm -f /logs/verifier/reward.txt /logs/verifier/ctrf.json /logs/verifier/timing.log
_vts(){ date +%s.%N; }
_vlog(){ printf '%s\t%s\n' "$(_vts)" "$*" >> /logs/verifier/timing.log; }
_vlog "verifier_wrapper_start"
if [ -f /opt/tb2-uv/uv ] && [ -f /tests/test.sh ]; then
  _vlog "tb2_uv_patch_start"
  sed -i \
    -e 's#^[[:space:]]*cp /opt/tb2-uv/uv /usr/local/bin/uv[[:space:]]*$#    export PATH=/opt/tb2-uv:$PATH#' \
    -e 's#^[[:space:]]*cp /opt/tb2-uv/uvx /usr/local/bin/uvx[[:space:]]*$#    :#' \
    /tests/test.sh || true
  _vlog "tb2_uv_patch_done"
fi

if [ -f /opt/tb2-uv/uv ] && [ -f /tests/test.sh ]; then
  _vlog "harbor_uv_bootstrap_patch_start"
  sed -i \
    -e 's#^[[:space:]]*curl -LsSf https://astral.sh/uv/[^|]*install.sh | sh[[:space:]]*$#export PATH=/opt/tb2-uv:$PATH#' \
    -e 's#^[[:space:]]*source[[:space:]]\\+.*\\.local/bin/env[[:space:]]*$#export PATH=/opt/tb2-uv:$PATH#' \
    /tests/test.sh || true
  export PATH=/opt/tb2-uv:$PATH
  export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
  export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
  _vlog "harbor_uv_bootstrap_patch_done"
fi

# RL verifier images are prepatched with their test dependencies.  With this
# guard enabled, verifier scripts may still contain original setup lines, but
# apt/pip/uvx install/download operations are blocked at runtime.  Missing
# dependencies then fail quickly in preflight instead of silently downloading
# during rollout grading.
if [ "${UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS:-0}" = "1" ]; then
  _vlog "runtime_install_guard_start"
  apt-get() {
    case "${1:-}" in
      update) return 0 ;;
      install) return 0 ;;
    esac
    command apt-get "$@"
  }
  apt() {
    case "${1:-}" in
      update) return 0 ;;
      install) return 0 ;;
    esac
    command apt "$@"
  }
  pip() {
    if [ "${1:-}" = "install" ]; then return 0; fi
    command pip "$@"
  }
  pip3() {
    if [ "${1:-}" = "install" ]; then return 0; fi
    command pip3 "$@"
  }
  python() {
    if [ "${1:-}" = "-m" ] && [ "${2:-}" = "pip" ] && [ "${3:-}" = "install" ]; then return 0; fi
    command python "$@"
  }
  python3() {
    if [ "${1:-}" = "-m" ] && [ "${2:-}" = "pip" ] && [ "${3:-}" = "install" ]; then return 0; fi
    command python3 "$@"
  }
  uv() {
    if [ "${1:-}" = "pip" ] && [ "${2:-}" = "install" ]; then return 0; fi
    if [ "${1:-}" = "tool" ] && [ "${2:-}" = "install" ]; then return 0; fi
    command uv "$@"
  }
  uvx() {
    # Drop dependency-selection options and run the requested command against
    # the prepatched image environment.  Supported verifier usages are pytest
    # and small python snippets; unexpected commands fall through to PATH.
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --with|-w|--from|-p|--python|--index|--extra-index-url|--index-strategy)
          shift 2 || return 127
          ;;
        --with=*|-w=*|--from=*|-p=*|--python=*|--index=*|--extra-index-url=*|--index-strategy=*)
          shift
          ;;
        --*)
          shift
          ;;
        *)
          break
          ;;
      esac
    done
    if [ "$#" -eq 0 ]; then return 0; fi
    case "$1" in
      pytest)
        shift
        python3 -m pytest "$@"
        ;;
      python|python3)
        shift
        python3 "$@"
        ;;
      *)
        command "$@"
        ;;
    esac
  }
  npm() {
    if [ "${1:-}" = "install" ] || [ "${1:-}" = "ci" ]; then return 0; fi
    command npm "$@"
  }
  export -f apt-get apt pip pip3 python python3 uv uvx npm
  _vlog "runtime_install_guard_done"
fi

# SETA synthetic tasks use a generated pytest wrapper that installs
# python3-pip + pytest on every verifier invocation. Under RL concurrency this
# repeatedly hits apt/pip/network and causes verifier timeouts even though the
# actual tests are plain Python asserts. If the test file does not use pytest
# fixtures/APIs, run it with a tiny in-process test runner instead. This keeps
# scoring equivalent for these generated tests and removes per-rollout package
# installation from the verifier path.
if [ -f /tests/test_outputs.py ] && [ -f /tests/test.sh ] \
   && grep -q 'pytest /tests/test_outputs.py' /tests/test.sh; then
  if ! command -v python3 >/dev/null 2>&1; then
    _vlog "seta_python3_install_start"
    apt-get update -qq
    apt-get install -y -qq python3
    _vlog "seta_python3_install_done rc=$?"
  fi
  if ! grep -Eq 'pytest\.|tmp_path|capsys|monkeypatch|parametrize' /tests/test_outputs.py; then
    _vlog "seta_mini_runner_start"
    set +e
    python3 - <<'PY'
import importlib.util
import sys
import traceback
import types

sys.modules.setdefault("pytest", types.SimpleNamespace())
path = "/tests/test_outputs.py"
spec = importlib.util.spec_from_file_location("test_outputs", path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)

failures = []
count = 0
for name in sorted(dir(module)):
    if not name.startswith("test_"):
        continue
    fn = getattr(module, name)
    if not callable(fn):
        continue
    count += 1
    try:
        fn()
        print(f"PASS {name}")
    except BaseException:
        failures.append(name)
        print(f"FAIL {name}")
        traceback.print_exc()

if count == 0:
    print("No test_* functions found", file=sys.stderr)
    sys.exit(1)
try:
    # Always record the fraction of passing tests. The shell below keeps this
    # value only when SETA_CONTINUOUS_REWARD=1; otherwise it overwrites with the
    # legacy binary reward, so default behaviour is unchanged.
    with open("/logs/verifier/reward.txt", "w") as _rf:
        _rf.write(str((count - len(failures)) / count))
except BaseException:
    pass
if failures:
    print(f"{len(failures)}/{count} tests failed: {failures}", file=sys.stderr)
    sys.exit(1)
print(f"{count}/{count} tests passed")
PY
    rc=$?
    set -e
    _vlog "seta_mini_runner_done rc=$rc"
    if [ "${SETA_CONTINUOUS_REWARD:-0}" = "1" ]; then
      :  # mini-runner already wrote the continuous (test-pass-fraction) reward
    elif [ $rc -eq 0 ]; then
      echo 1 > /logs/verifier/reward.txt
    else
      echo 0 > /logs/verifier/reward.txt
    fi
    exit 0
  fi
fi
_vlog "test_sh_start"
export PS4='+VERIFIER_TRACE $(date +%s.%N) '
set +e
bash -x /tests/test.sh
rc=$?
set -e
_vlog "test_sh_done rc=$rc"
exit $rc
'''
    exec_args = ["docker", "exec"]
    block_runtime_installs = os.environ.get("UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS")
    if block_runtime_installs is not None:
        exec_args.extend([
            "-e",
            f"UNIFIED_VERIFIER_BLOCK_RUNTIME_INSTALLS={block_runtime_installs}",
        ])
    stdout, stderr, rc = docker_run(
        [*exec_args, cname, "bash", "-lc", verifier_cmd],
        timeout=timeout_sec)

    verifier_ok = (rc != -1)  # -1 means docker_run hit subprocess.TimeoutExpired

    # Read reward (only meaningful if verifier finished)
    reward_out, _, reward_rc = docker_run(
        ["docker", "exec", cname, "cat", "/logs/verifier/reward.txt"],
        timeout=60)
    timing_out, _, _ = docker_run(
        ["docker", "exec", cname, "cat", "/logs/verifier/timing.log"],
        timeout=20)

    try:
        reward = float(reward_out.strip())
    except (ValueError, TypeError):
        reward = 0.0

    output = stdout[-2000:] if stdout else ""
    if stderr:
        output += "\n[STDERR] " + stderr[-1000:]
    if timing_out:
        output += "\n[VERIFIER_TIMING]\n" + timing_out[-2000:]
    if not verifier_ok:
        output = f"[VERIFIER TIMEOUT after {timeout_sec}s]\n" + output

    return reward, output, verifier_ok


def run_task(task_dir, task_name, dataset, config, use_skills=True, dataset_tag="",
             verifier_timeout_multiplier=1.2,
             retrieval_mapping=None, retrieval_top_n=3,
             top1_skill_text_mapping=None):
    """Run one task using the unified agent loop.

    retrieval_mapping: optional {task_id: [skill_path,...]} dict. When set,
                      top-N skills are docker-cp'd into /root/.claude/skills/
                      (and siblings), and the sys prompt gets a hint listing them.

    Two env vars (used by SFT data collection) further modify the system prompt:
      UNIFIED_IMPLICIT_MODE      ∈ {"use_skill", "no_skill", ""}
      UNIFIED_REFLECTION_CONTEXT  free text from a previous failed attempt
    Both get appended via implicit_instruction.apply_implicit_and_reflection
    and the exact appended text is recorded in the result row so the SFT
    collector can strip it before training.
    """
    from unified_runner.implicit_instruction import apply_implicit_and_reflection

    result = {
        "task_id": task_name,
        "dataset": dataset,
        "resolved": False,
        "score": 0.0,
        "turns": 0,
        "time_sec": 0,
        "error": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "verifier_output": "",
        "retrieval_skills_injected": 0,
        # SFT-collection metadata (populated below if env vars are set):
        "implicit_mode": os.environ.get("UNIFIED_IMPLICIT_MODE", "").strip(),
        "implicit_text": "",      # exact text appended; collector strips this
        "reflection_context": os.environ.get("UNIFIED_REFLECTION_CONTEXT", "").strip(),
        "reflection_text": "",    # exact text appended (incl. wrapper); collector strips this
    }
    start_time = time.time()
    cname = None
    tool_layer = None
    try:
        with contextlib.nullcontext():
            # Build or resolve Docker image (TB 2.0 uses prebuilt alexgshaw/*)
            image_tag = resolve_image(task_dir, task_name, dataset)

            # Start container
            cname = start_container(image_tag, task_name, dataset_tag=dataset_tag)
            print(f"    Container: {cname}")

            # Inject retrieval-selected skills if a mapping was supplied.
            retrieval_hint = ""
            direct_skill_prompt = ""
            if retrieval_mapping is not None:
                n_injected = inject_retrieval_skills(
                    docker_run, cname, task_name, retrieval_mapping,
                    top_n=retrieval_top_n,
                )
                result["retrieval_skills_injected"] = n_injected
                retrieval_hint = build_retrieval_prompt_hint(
                    task_name, retrieval_mapping, retrieval_top_n,
                )
            if top1_skill_text_mapping is not None:
                direct_skill_prompt, skill_name = build_top1_skill_text_prompt(
                    task_name, top1_skill_text_mapping,
                )
                if direct_skill_prompt:
                    result["retrieval_skills_injected"] = 1
                    result["top1_skill_text_name"] = skill_name

            # Read instruction
            instruction = get_instruction(task_dir)
            if len(instruction) > 5000:
                instruction = instruction[:5000] + "\n... [truncated]"

            # Build OpenClaw-compatible system prompt. Harbor/TB/SETA/SB runtime
            # instructions are inlined as Project Context AGENTS.md/TOOLS.md,
            # so the user message stays just the task description.
            from unified_runner.bench_workspace_files import build_workspace_files_for_bench
            # Map runner --dataset → SFT bench-id (used by AGENTS.md/TOOLS.md
            # selectors). All harbor benches share the same generic content, so
            # any of (tb2 / sb_ns / seta_synth) yields identical files.
            if dataset_tag.startswith("skillsbench"):
                bench_label = "sb_ns"
            elif dataset_tag == "tb2":
                bench_label = "tb2"
            else:
                bench_label = "seta_synth"
            workspace_files = build_workspace_files_for_bench(bench_label)
            sys_prompt = build_openclaw_system_prompt(
                workspace_dir="/root",
                skills_prompt=retrieval_hint if (use_skills or retrieval_hint) else "",
                direct_skill_prompt=direct_skill_prompt,
                sandboxed=True,
                runtime_label="unified_runner.harbor",
                workspace_files=workspace_files,
            )

            # SFT-collection: optional implicit instruction + reflection context.
            # Both come from env vars (set by sft_data_collection/launch_trials.py
            # via the plan record's "env" field). Returns the EXACT bytes appended
            # so the collector can strip them when exporting SFT data.
            sys_prompt, applied_implicit, applied_reflection = apply_implicit_and_reflection(
                sys_prompt,
                implicit_mode=result["implicit_mode"],
                reflection_context=result["reflection_context"],
            )
            result["implicit_text"] = applied_implicit
            result["reflection_text"] = applied_reflection

            # Create tool layer in docker mode (one persistent shell per task)
            tool_layer = ToolLayer(mode="docker", container=cname, workdir="/root")

            # Create and run agent
            agent = UnifiedAgentLoop(config, tool_layer, max_tool_calls_per_turn=10)
            print(f"    Running unified agent loop...")
            # NB: per-bench runtime guidance is now inlined via Project Context
            # (AGENTS.md/TOOLS.md), not appended to the user message.
            task_prompt = instruction
            traj = agent.run(task_prompt, system_prompt=sys_prompt)

            result["turns"] = traj.turns
            result["input_tokens"] = traj.total_input_tokens
            result["output_tokens"] = traj.total_output_tokens
            result["finish_reason"] = traj.finish_reason
            result["trajectory"] = traj.to_sft_messages()
            if traj.error:
                result["error"] = traj.error

            print(f"    Agent finished: {traj.finish_reason} "
                  f"(turns={traj.turns}, tokens={traj.total_input_tokens}in/{traj.total_output_tokens}out)")

            # Copy tests and run verifier
            print(f"    Running verifier...")
            if copy_tests(task_dir, cname):
                v_base = _read_verifier_timeout(task_dir, default=600)
                v_timeout = int(v_base * verifier_timeout_multiplier)
                reward, verifier_output, verifier_ok = run_verifier(cname, timeout_sec=v_timeout)
                result["score"] = reward
                result["resolved"] = reward >= 1.0
                result["verifier_output"] = verifier_output
                if not verifier_ok:
                    result["error"] = (result.get("error", "") + f" verifier_timeout({v_timeout}s)").strip()
            else:
                result["error"] = (result.get("error", "") + " No tests found").strip()

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    finally:
        result["time_sec"] = int(time.time() - start_time)
        if tool_layer is not None:
            tool_layer.close()
        if cname:
            print(f"    Cleaning up container {cname}...")
            stop_container(cname)

    return result


def run_task_with_retry(task_dir, task_name, dataset, config, *,
                         max_retries=2, retry_backoff_sec=15, **kw):
    """Run task; on flaky-infra error, cleanup + retry up to `max_retries` times.

    Only retries on errors matching _FLAKY_RETRY_PATTERNS. Agent-timeout,
    HTTP 4xx, verifier failures are NOT retried — those reflect agent ability
    or upstream task state and re-running wastes budget.
    """
    last_result = None
    for attempt in range(max_retries + 1):
        result = run_task(task_dir, task_name, dataset, config, **kw)
        err = result.get("error", "")
        if not _is_flaky_error(err):
            return result
        last_result = result
        if attempt < max_retries:
            print(f"    [retry {attempt + 1}/{max_retries}] flaky error; "
                  f"cleaning up and retrying in {retry_backoff_sec}s: {err[:120]}")
            _cleanup_stale(task_name, kw.get("dataset_tag", dataset))
            time.sleep(retry_backoff_sec)
    # All retries exhausted — mark the attempt count in result for analysis.
    last_result["error"] = (last_result.get("error", "") +
                            f" [exhausted {max_retries} retries]").strip()
    return last_result


def main():
    parser = argparse.ArgumentParser(description="Unified Harbor evaluation")
    parser.add_argument("--dataset", required=True,
                        choices=["skillsbench", "seta", "seta-synth", "tb2"],
                        help="Dataset to evaluate. `seta` = 30-task baseline; "
                             "`seta-synth` = 1376-task synth_data_harbor (use with --tasks-file).")
    parser.add_argument("--variant", default="with-skills",
                        choices=["with-skills", "no-skills"],
                        help="SkillsBench variant (ignored for SETA)")
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", "http://localhost:30000/v1"))
    parser.add_argument("--max-turns", type=int,
                        default=int(os.environ.get("UNIFIED_DEFAULT_MAX_TURNS", "70")))
    parser.add_argument(
        "--max-time",
        type=int,
        default=int(os.environ.get("UNIFIED_ROLLOUT_WALLCLOCK_CAP_SEC", "1800")),
    )
    parser.add_argument("--task", type=str, help="Run only this task")
    parser.add_argument("--tasks-file", type=str, help="File with one task name per line; filter to these")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent tasks (default 1)")
    parser.add_argument("--verifier-timeout-multiplier", type=float, default=1.2,
                        help="Multiply task.toml verifier.timeout_sec (default 1.2). "
                             "Lowered from 3.0 on 2026-04-19 because tb2_uv_cache prebake "
                             "removes the 50-400MB uv+python+deps download per task; 1.2x "
                             "gives enough slack for remote-docker exec overhead.")
    parser.add_argument("--retries", type=int, default=2,
                        help="Retries on flaky-infra errors (container-start timeout, "
                             "image-pull fail, compose-name conflict, etc). Default 2.")
    parser.add_argument("--no-exclude", action="store_true",
                        help="Disable the _EXCLUDED_TASKS filter (run every task, "
                             "even ones structurally requiring external credentials).")
    parser.add_argument("--inject-retrieval-skills", type=str, default=None,
                        help="Path to retrieval jsonl. When set, top-N retrieved "
                             "skills per task are docker-cp'd into /root/.claude/skills/.")
    parser.add_argument("--inject-irrelevant-skills", type=str, default=None,
                        help="Path to retrieval jsonl. When set, top-N *irrelevant* "
                             "(negative-control) skills are injected: random draw "
                             "from skill_libraries/merged/ excluding anything in the "
                             "task's coarse_top20. Deterministic (seeded by task_id). "
                             "Mutually exclusive with --inject-retrieval-skills.")
    parser.add_argument("--inject-top1-skill-text", type=str, default=None,
                        help="Path to retrieval jsonl. When set, the top-1 retrieved "
                             "SKILL.md text is injected directly into the system prompt "
                             "without requiring the agent to read the skill file.")
    parser.add_argument("--retrieval-top-n", type=int, default=3,
                        help="How many skills to inject per task (default 3).")
    args = parser.parse_args()
    selected_skill_modes = [
        bool(args.inject_retrieval_skills),
        bool(args.inject_irrelevant_skills),
        bool(args.inject_top1_skill_text),
    ]
    if sum(selected_skill_modes) > 1:
        print("ERROR: --inject-retrieval-skills / --inject-irrelevant-skills / "
              "--inject-top1-skill-text are mutually exclusive",
              file=sys.stderr)
        sys.exit(2)

    # Determine dataset path
    if args.dataset == "skillsbench":
        if args.variant == "no-skills":
            dataset_path = DATASET_PATHS["skillsbench-no-skills"]
        else:
            dataset_path = DATASET_PATHS["skillsbench"]
        dataset_tag = f"skillsbench-{args.variant}"
    elif args.dataset == "tb2":
        dataset_path = DATASET_PATHS["tb2"]
        dataset_tag = "tb2"
    elif args.dataset == "seta-synth":
        dataset_path = DATASET_PATHS["seta-synth"]
        dataset_tag = "seta-synth"
    else:
        dataset_path = DATASET_PATHS["seta"]
        dataset_tag = "seta"

    if not dataset_path.exists():
        print(f"ERROR: Dataset path not found: {dataset_path}")
        sys.exit(1)

    from unified_runner.base import env_overrides
    config = RunConfig(
        model=args.model,
        api_base=args.api_base,
        max_turns=args.max_turns,
        max_time_sec=args.max_time,
        temperature=0.6,
        max_tokens=8192,
        max_output_chars=16000,
        **env_overrides(),
    )

    # Load skill-injection mapping for the selected arm (retrieval / irrelevant / none).
    retrieval_mapping = None
    top1_skill_text_mapping = None
    skill_arm = "baseline"
    if args.inject_retrieval_skills:
        retrieval_mapping = load_retrieval_mapping(args.inject_retrieval_skills)
        skill_arm = "retrieval"
        print(f"[retrieval] {len(retrieval_mapping)} entries from {args.inject_retrieval_skills} "
              f"(top_n={args.retrieval_top_n})")
    elif args.inject_irrelevant_skills:
        retrieval_mapping = build_irrelevant_mapping(args.inject_irrelevant_skills,
                                                      top_n=args.retrieval_top_n)
        skill_arm = "irrelevant"
        print(f"[irrelevant] {len(retrieval_mapping)} entries built from "
              f"{args.inject_irrelevant_skills} (excluding coarse_top20; seed=hash(task_id))")
    elif args.inject_top1_skill_text:
        top1_skill_text_mapping = load_retrieval_mapping(args.inject_top1_skill_text)
        skill_arm = "top1_skill_text"
        print(f"[top1-skill-text] {len(top1_skill_text_mapping)} entries from "
              f"{args.inject_top1_skill_text}")

    # List tasks
    tasks = list_tasks(dataset_path)
    if args.task:
        tasks = [t for t in tasks if t == args.task]
    if getattr(args, "tasks_file", None):
        wanted = {l.strip() for l in open(args.tasks_file) if l.strip()}
        tasks = [t for t in tasks if t in wanted]
        missing = wanted - set(tasks)
        if missing:
            print(f"WARN: {len(missing)} tasks in --tasks-file not found in dataset: {sorted(missing)[:5]}...")
    # Structural exclusions (can be bypassed with --no-exclude for debugging).
    excluded = _EXCLUDED_TASKS.get(dataset_tag, set())
    if excluded and not args.no_exclude:
        before = set(tasks)
        tasks = [t for t in tasks if t not in excluded]
        removed = before & excluded
        if removed:
            print(f"[exclude] skipping {len(removed)} structural task(s): {sorted(removed)}")
    print(f"Dataset: {dataset_tag}")
    print(f"Tasks: {len(tasks)}")
    print(f"Model: {args.model}")
    print(f"Interface: Unified OpenClaw deploy-tool subset")

    if not tasks:
        print("No tasks found!")
        sys.exit(1)

    # Results files — 2026-04-22 v8 layout:
    #   results/<date>/<bench>/<experiment>/{incremental.jsonl, trajectories/, summary.md}
    # experiment = "<version>_<arm>" where version from env UNIFIED_EXP_VERSION (default "v8")
    from unified_runner.base import results_subdir, experiment_name
    date_prefix = os.environ.get("UNIFIED_RESULTS_DATE") or datetime.now().strftime("%Y%m%d")
    exp_dir = results_subdir(RESULTS_DIR, date_prefix, bench=dataset_tag,
                             experiment=experiment_name(skill_arm))
    inc_path = exp_dir / "incremental.jsonl"
    traj_dir = exp_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {exp_dir.relative_to(RESULTS_DIR)}  skill_arm={skill_arm}")
    use_skills = args.variant == "with-skills" if args.dataset == "skillsbench" else True

    results = []
    for idx, task_name in enumerate(tasks, 1):
        task_dir = dataset_path / task_name
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(tasks)}] UNIFIED {dataset_tag}: {task_name}")
        print(f"{'='*60}")

        result = run_task_with_retry(
            task_dir, task_name, dataset_tag, config,
            max_retries=args.retries,
            use_skills=use_skills,
            dataset_tag=dataset_tag,
            verifier_timeout_multiplier=args.verifier_timeout_multiplier,
            retrieval_mapping=retrieval_mapping,
            retrieval_top_n=args.retrieval_top_n,
            top1_skill_text_mapping=top1_skill_text_mapping,
        )
        results.append(result)

        # Save trajectory to its own file (SFT-ready, unified format across 3 runners).
        # implicit_text + reflection_text are saved so the SFT collector can strip
        # those exact bytes from messages[0]['content'] (the system message)
        # without needing to also load the incremental.jsonl row.
        traj = result.pop("trajectory", None)
        if traj is not None:
            (traj_dir / f"{task_name}.json").write_text(
                json.dumps({
                    "task_id": task_name,
                    "dataset": dataset_tag,
                    "skill_arm": skill_arm,
                    "retrieval_skills_injected": result.get("retrieval_skills_injected", 0),
                    "top1_skill_text_name": result.get("top1_skill_text_name", ""),
                    "resolved": result.get("resolved", False),
                    "score": result.get("score", 0.0),
                    "implicit_mode": result.get("implicit_mode", ""),
                    "implicit_text": result.get("implicit_text", ""),
                    "reflection_context": result.get("reflection_context", ""),
                    "reflection_text": result.get("reflection_text", ""),
                    "messages": traj,
                }, ensure_ascii=False, default=str, indent=2)
            )

        # Save incremental — keep verifier_output for debugging (truncate large fields)
        with open(inc_path, "a") as f:
            r = dict(result)
            if "verifier_output" in r and r["verifier_output"]:
                r["verifier_output"] = r["verifier_output"][-2000:]
            f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")

        status = "RESOLVED" if result["resolved"] else "FAILED"
        score_str = f", score={result['score']:.3f}" if result["score"] > 0 else ""
        print(f"  [{idx}/{len(tasks)}] {task_name}: {status}{score_str} "
              f"(turns={result['turns']}, time={result['time_sec']}s)")

    # Summary
    total = len(results)
    resolved = sum(1 for r in results if r["resolved"])
    scores = [r["score"] for r in results]
    mean_score = sum(scores) / total if total else 0
    resolve_rate = resolved / total if total else 0

    print(f"\n{'='*60}")
    print(f"UNIFIED {dataset_tag} FINAL:")
    print(f"  resolve_rate: {resolve_rate:.3f} ({resolved}/{total})")
    print(f"  mean_score: {mean_score:.3f}")
    print(f"{'='*60}")

    # Write summary — 2026-04-22 v8 schema (full metrics: N_total/N_pass/N_error/pass_rate/Mean_score)
    from unified_runner.base import write_summary_md
    summary_path = write_summary_md(exp_dir, dataset_tag, args.model, results,
                                    extra_meta={"skill_arm": skill_arm,
                                                "n_tasks_requested": len(tasks)})
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
