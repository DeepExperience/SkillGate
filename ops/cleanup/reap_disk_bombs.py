#!/usr/bin/env python3
"""Two-tier disk-bomb reaper for the local RL dockerd.

Root cause it mitigates: an agent on a disk-operations task (e.g. seta_synth/210
"secure wipe") can emit an UNBOUNDED `dd if=/dev/urandom of=...img bs=1M` (no
count=), writing until the docker disk fills -> dockerd dies -> run crashes.
The task itself is fine (1MB images, official solution bounds dd with count=);
this is an agent command error, amplified by concurrency. storage_mb declared per
task (max 20GB) is NOT enforced by docker run, so writes are unbounded.

Design: cheap df-watermark check every cycle; only when free space drops below
WATERMARK do we run the expensive per-container `docker ps -s` scan and force-remove
any container whose WRITABLE layer exceeds KILL_GB. KILL_GB=40 is safely above the
max declared storage (20GB) and the empirical max writable (~6GB), far below a
runaway (100s of GB), so legit tasks are never touched.
"""
import os, re, time, subprocess

DOCKER_HOST = os.environ.get("DOCKER_HOST", "unix:///tmp/local-docker-overlay2.sock")
DISK_PATH   = os.environ.get("DISK_REAP_PATH", "/mnt/docker-overlay2")
WATERMARK_GB= int(os.environ.get("DISK_REAP_WATERMARK_GB", "400"))   # hunt below this free
KILL_GB     = float(os.environ.get("DISK_REAP_KILL_GB", "40"))       # kill writable-layer above this
INTERVAL    = int(os.environ.get("DISK_REAP_INTERVAL_SEC", "30"))
# Early-warning + language-agnostic kill knobs (2026-06-14 seta_synth/210 python-urandom bomb):
# a 1.16TB writable layer sat unkilled because (a) the writer was python, not `dd`, and
# (b) free was still above WATERMARK so hunt_and_kill never ran. dirty memory spikes long
# before free drops, and /proc/<pid>/io write_bytes catches any-language runaway writers.
WRITER_IO_KILL_GB = float(os.environ.get("DISK_REAP_WRITER_IO_KILL_GB", "20"))  # kill urandom/zero writer past this cumulative write
DIRTY_TRIGGER_GB  = float(os.environ.get("DISK_REAP_DIRTY_TRIGGER_GB", "50"))   # also hunt when dirty pages exceed this
# Unconditional periodic big-layer scan: catch slow bombs that read neither
# /dev/urandom nor /dev/zero (so the writer killer misses) AND grow while free is
# still above WATERMARK (so the free-triggered hunt never runs) -- e.g. the
# seta_synth/1366 227GB layer on 2026-06-14 that fell between both rules. KILL_GB
# (40GB) is far above any legit task (~6GB), so an unconditional sweep is safe.
BIG_LAYER_SCAN_SEC = float(os.environ.get("DISK_REAP_BIG_LAYER_SCAN_SEC", "60"))
ENV = {**os.environ, "DOCKER_HOST": DOCKER_HOST}

def dirty_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("Dirty:"):
                    return float(line.split()[1]) / (1024*1024)  # kB -> GB
    except Exception:
        pass
    return 0.0

def _proc_write_gb(pid):
    try:
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                if line.startswith("write_bytes:"):
                    return float(line.split()[1]) / 1e9
    except Exception:
        pass
    return 0.0

def free_gb():
    try:
        out = subprocess.run(["df","--output=avail",DISK_PATH], capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[-1]) / (1024*1024)
    except Exception:
        return None

_SZ = re.compile(r'^([0-9.]+)\s*([kKMGTP]?B)')
def to_gb(s):
    m = _SZ.match(s.strip())
    if not m: return 0.0
    v=float(m.group(1)); u=m.group(2).upper()
    # TB/PB matter: a 3.08TB runaway layer must not parse as 0.0 (2026-06-13 incident)
    return {"B":v/1e9,"KB":v/1e6,"MB":v/1e3,"GB":v,"TB":v*1e3,"PB":v*1e6}.get(u, 0.0)

def hunt_and_kill():
    # writable-layer size is the part BEFORE "(virtual ...)"; that is what a runaway write grows
    try:
        out = subprocess.run(["docker","ps","-s","--format","{{.ID}}\t{{.Names}}\t{{.Size}}"],
                             capture_output=True, text=True, timeout=120, env=ENV)
    except Exception as e:
        print(f"[disk-reap] ps -s failed: {e!r}", flush=True); return 0
    killed=0
    for line in out.stdout.splitlines():
        parts=line.split("\t")
        if len(parts)<3: continue
        cid,name,size=parts[0],parts[1],parts[2]
        writable=size.split("(")[0]  # "5.74GB " from "5.74GB (virtual 24.1GB)"
        gb=to_gb(writable)
        if gb>=KILL_GB:
            # kill FIRST (stops the writer within seconds even for TB-scale layers),
            # then best-effort rm. 2026-06-10 17:56 eviction post-mortem: rm -f of a
            # runaway layer timed out at 30s and the node hit kubelet's
            # ephemeral-storage eviction threshold (~246GiB free) ~90s later.
            try:
                subprocess.run(["docker","kill",cid], capture_output=True, timeout=15, env=ENV)
                killed+=1
                print(f"[disk-reap] KILLED {name} writable={gb:.1f}GB (>{KILL_GB}GB runaway)", flush=True)
            except Exception as e:
                print(f"[disk-reap] kill {name} failed: {e!r}", flush=True)
                continue
            try:
                subprocess.run(["docker","rm","-f",cid], capture_output=True, timeout=120, env=ENV)
            except Exception as e:
                print(f"[disk-reap] rm {name} deferred (stale cleaner will reap): {e!r}", flush=True)
    return killed

def kill_runaway_writers():
    """Every cycle, kill disk-bomb writers regardless of language/command form.
    Two rules, both fire independent of free space (the bomb spikes load/writeback
    long before the disk reaches the watermark):
      1) `dd if=/dev/{urandom,zero}` with NO count= -> unbounded by construction, kill on sight.
      2) ANY process whose cmdline references /dev/urandom or /dev/zero AND whose cumulative
         write_bytes exceeds WRITER_IO_KILL_GB (default 20GB, far above the ~6GB max a legit
         task writes). This covers python/perl/etc. loop writers that the dd-only matcher
         missed -- e.g. the seta_synth/210 `python -c "... open('/dev/urandom') ... f.write"`
         bomb that grew a 1.16TB layer unkilled on 2026-06-14."""
    killed = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\x00", b" ").decode("utf-8", "ignore")
        except Exception:
            continue
        if "/dev/urandom" not in cmd and "/dev/zero" not in cmd:
            continue
        try:
            comm = open(f"/proc/{pid}/comm").read().strip()
        except Exception:
            comm = ""
        unbounded_dd = comm == "dd" and "count=" not in cmd
        big_writer = _proc_write_gb(pid) >= WRITER_IO_KILL_GB
        if unbounded_dd or big_writer:
            try:
                os.kill(int(pid), 9)
                killed += 1
            except Exception:
                pass
    if killed:
        print(f"[disk-reap] {time.strftime('%H:%M:%S')} killed {killed} runaway writer(s) (urandom/zero, dd-or-IO>{WRITER_IO_KILL_GB:.0f}GB)", flush=True)
    return killed


def main():
    print(f"[disk-reap] start watermark={WATERMARK_GB}GB kill_gb={KILL_GB} interval={INTERVAL}s "
          f"writer_io_kill={WRITER_IO_KILL_GB}GB dirty_trigger={DIRTY_TRIGGER_GB}GB "
          f"big_layer_scan={BIG_LAYER_SCAN_SEC:.0f}s (+per-cycle runaway-writer killer)", flush=True)
    last_big_scan = 0.0
    while True:
        kill_runaway_writers()
        fg = free_gb()
        dg = dirty_gb()
        now = time.monotonic()
        low_free = fg is not None and fg < WATERMARK_GB
        high_dirty = dg >= DIRTY_TRIGGER_GB
        periodic = (now - last_big_scan) >= BIG_LAYER_SCAN_SEC
        if low_free or high_dirty or periodic:
            why = (f"free={fg:.0f}GB<{WATERMARK_GB}" if low_free
                   else f"dirty={dg:.0f}GB>={DIRTY_TRIGGER_GB:.0f}" if high_dirty
                   else "periodic")
            k = hunt_and_kill()
            last_big_scan = now
            if k or low_free or high_dirty:
                print(f"[disk-reap] {time.strftime('%H:%M:%S')} {why} -> hunt killed={k} free_now={ (free_gb() or 0):.0f}GB", flush=True)
        time.sleep(INTERVAL)

if __name__=="__main__":
    main()
