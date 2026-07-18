#!/usr/bin/env python3
"""Periodic orphan containerd-shim reaper for the RL rollout node.

Root cause it mitigates: the KubeRay worker pod's PID1 is `ray start --block`, which
does NOT reap children. Under sustained high-concurrency container churn, docker
teardowns occasionally time out, leaking containerd-shim processes that reparent to
PID1 and never get reaped. They pile up (live -> D-state on cgroup/netlink/rtnl locks)
and saturate the kernel control plane -> container create/teardown stalls -> rollout
wedges. This loop kills shims whose container id is NOT in `docker ps` (orphans),
converting them to harmless zombies BEFORE thousands accumulate live and jam the locks.

Safe: only signals processes whose comm contains 'containerd-shim' AND whose -id is not
a currently-running container. Never touches ray/raylet/gcs/python/sglang.
"""
import os, glob, time, subprocess, sys

INTERVAL = int(os.environ.get("REAP_INTERVAL_SEC", "120"))
SOCK = os.environ.get("DOCKER_HOST", "unix:///tmp/local-docker-overlay2.sock")

def live_cids():
    try:
        out = subprocess.run(["docker","ps","-q","--no-trunc"], capture_output=True,
                             text=True, timeout=30, env={**os.environ,"DOCKER_HOST":SOCK})
        return set(out.stdout.split())
    except Exception as e:
        return None  # docker unhealthy -> skip this cycle

def shim_pid_cid():
    res=[]
    for d in glob.glob('/proc/[0-9]*'):
        pid=d.rsplit('/',1)[-1]
        try:
            if 'containerd-shim' not in open(d+'/comm').read(): continue
            cmd=open(d+'/cmdline','rb').read().split(b'\x00')
            cid=None
            for i,a in enumerate(cmd):
                if a==b'-id' and i+1<len(cmd): cid=cmd[i+1].decode(); break
            res.append((pid,cid))
        except Exception: pass
    return res

def main():
    while True:
        live=live_cids()
        if live is None:
            print(f"[reap] {time.strftime('%H:%M:%S')} docker unhealthy; skip", flush=True)
            time.sleep(INTERVAL); continue
        shims=shim_pid_cid()
        orphan=[p for p,c in shims if (c is None or c not in live)]
        killed=0
        for p in orphan:
            try:
                st=open(f'/proc/{p}/stat').read().split()[2]
                if st=='Z': continue   # already dead, can't reap (PID1 won't), skip
                os.kill(int(p),9); killed+=1
            except Exception: pass
        la=open('/proc/loadavg').read().split()[0]
        print(f"[reap] {time.strftime('%H:%M:%S')} shims={len(shims)} live_ctr={len(live)} orphan={len(orphan)} killed={killed} load1={la}", flush=True)
        time.sleep(INTERVAL)

if __name__=="__main__":
    main()
