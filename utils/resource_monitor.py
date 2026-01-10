#!/usr/bin/env python3

"""
Run a command while collecting container-accurate CPU/memory (via cgroups)
and best-effort GPU metrics (via nvidia-smi), then emit sentinel JSON lines:

  #PROC_METRICS#{...}
  #GPU_METRICS#{...}

Designed to be called from train.sh so suffix scripts remain untouched.
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple


CGROUP_ROOT = "/sys/fs/cgroup"


def _file_exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False

def _read_text(p: str) -> str:
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()

def _read_int(p: str) -> int:
    return int(_read_text(p))

def _is_cgroup_v2() -> bool:
    return _file_exists(os.path.join(CGROUP_ROOT, "cgroup.controllers"))

def _read_cpu_stat_v2() -> Dict[str, int]:
    # cpu.stat example lines: usage_usec 123, user_usec 45, system_usec 67
    out: Dict[str, int] = {}
    txt = _read_text(os.path.join(CGROUP_ROOT, "cpu.stat"))
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) == 2:
            k, v = parts
            try:
                out[k] = int(v)
            except ValueError:
                pass
    return out

def _read_mem_current_v2() -> int:
    return _read_int(os.path.join(CGROUP_ROOT, "memory.current"))

def _read_mem_peak_v2() -> Optional[int]:
    p = os.path.join(CGROUP_ROOT, "memory.peak")
    if _file_exists(p):
        try:
            return _read_int(p)
        except Exception:
            return None
    return None

def _read_io_stat_v2() -> Tuple[int, int]:
    """
    Returns (rbytes, wbytes) summed across devices from io.stat.
    If unavailable, returns (0,0).
    """
    p = os.path.join(CGROUP_ROOT, "io.stat")
    if not _file_exists(p):
        return (0, 0)
    rbytes = 0
    wbytes = 0
    txt = _read_text(p)
    for line in txt.splitlines():
        # device line format varies; tokens like rbytes=..., wbytes=...
        for tok in line.split():
            if tok.startswith("rbytes="):
                try:
                    rbytes += int(tok.split("=", 1)[1])
                except ValueError:
                    pass
            elif tok.startswith("wbytes="):
                try:
                    wbytes += int(tok.split("=", 1)[1])
                except ValueError:
                    pass
    return (rbytes, wbytes)

def _run_nvidia_smi(query: str) -> Optional[str]:
    """
    Return raw output of nvidia-smi query or None if not available.
    """
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None

def _gpu_static_info() -> Dict[str, Any]:
    # best-effort single-GPU fields (you can extend to multi-GPU later)
    out = {"name": None, "total_mb": None}
    raw = _run_nvidia_smi("gpu=name,memory.total")
    if raw:
        # take first GPU line
        first = raw.splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) >= 2:
            out["name"] = parts[0]
            try:
                out["total_mb"] = int(parts[1])
            except ValueError:
                out["total_mb"] = None
    return out

def _gpu_util_and_mem_used() -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Returns (util_gpu_pct, util_mem_pct, mem_used_mb) for GPU0 (best-effort).
    Device-level, not per-process.
    """
    raw = _run_nvidia_smi("gpu=utilization.gpu,utilization.memory,memory.used")
    if not raw:
        return (None, None, None)
    first = raw.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 3:
        return (None, None, None)
    try:
        ug = int(float(parts[0]))
    except Exception:
        ug = None
    try:
        um = int(float(parts[1]))
    except Exception:
        um = None
    try:
        mu = int(float(parts[2]))
    except Exception:
        mu = None
    return (ug, um, mu)

def _container_pid_to_host_pid(container_pid: int) -> Optional[int]:
    """
    Map container PID -> host PID via /proc/<pid>/status NSpid line.
    Works when kernel exposes NSpid.
    """
    try:
        status = _read_text(f"/proc/{container_pid}/status")
    except Exception:
        return None
    for line in status.splitlines():
        if line.startswith("NSpid:"):
            nums = [int(x) for x in line.split()[1:] if x.isdigit()]
            # NSpid lists pids from outermost (host) to innermost (container)
            return nums[0] if nums else None
    return None

def _gpu_proc_mem_used_mb_by_hostpid(host_pid: int) -> Optional[int]:
    """
    Best-effort per-process used GPU memory via nvidia-smi compute-apps.
    Returns MB or None if not found.
    """
    raw = _run_nvidia_smi("compute-apps=pid,used_memory")
    if not raw:
        return None
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            if pid != host_pid:
                continue
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None

def _pump(src, dst):
    try:
        while True:
            buf = src.read(8192)
            if not buf:
                break
            dst.write(buf)
            dst.flush()
    except Exception:
        pass

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=0.5, help="sampling interval seconds")
    ap.add_argument("--", dest="cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cmd = args.cmd
    if not cmd:
        print("resource_monitor.py: missing command after --", file=sys.stderr)
        return 2

    # --- baseline cgroup stats (preferred: container-wide, includes workers)
    use_cgv2 = _is_cgroup_v2()
    if not use_cgv2:
        print("resource_monitor.py: WARNING: cgroup v2 not detected; metrics may be degraded.", file=sys.stderr)

    cpu0 = _read_cpu_stat_v2() if use_cgv2 else {}
    io_r0, io_w0 = _read_io_stat_v2() if use_cgv2 else (0, 0)

    mem_peak_bytes = 0
    mem_peak_file = _read_mem_peak_v2() if use_cgv2 else None
    if mem_peak_file is not None:
        mem_peak_bytes = mem_peak_file
    else:
        # will track peak by sampling memory.current
        if use_cgv2:
            try:
                mem_peak_bytes = _read_mem_current_v2()
            except Exception:
                mem_peak_bytes = 0

    gpu_info = _gpu_static_info()
    cuda_available = False
    try:
        import torch  # type: ignore
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        # fall back to nvidia-smi presence
        cuda_available = _run_nvidia_smi("gpu=name") is not None

    # --- run child
    t0 = time.perf_counter()
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    # forward output live
    t_out = threading.Thread(target=_pump, args=(p.stdout, sys.stdout.buffer), daemon=True)  # type: ignore
    t_err = threading.Thread(target=_pump, args=(p.stderr, sys.stderr.buffer), daemon=True)  # type: ignore
    t_out.start()
    t_err.start()

    host_pid = _container_pid_to_host_pid(p.pid)
    # GPU time series
    util_sum = 0.0
    util_cnt = 0
    util_peak = None
    mem_used_sum = 0.0
    mem_used_cnt = 0
    mem_used_peak = None

    proc_mem_peak = None  # per-process peak GPU mem if we can resolve
    proc_mem_sum = 0.0
    proc_mem_cnt = 0

    try:
        while True:
            rc = p.poll()
            # sample memory peak (cgroup)
            if use_cgv2 and mem_peak_file is None:
                try:
                    cur = _read_mem_current_v2()
                    if cur > mem_peak_bytes:
                        mem_peak_bytes = cur
                except Exception:
                    pass

            # sample GPU (best-effort)
            ug, _um, mu = _gpu_util_and_mem_used()
            if ug is not None:
                util_sum += ug
                util_cnt += 1
                util_peak = ug if util_peak is None else max(util_peak, ug)

            if mu is not None:
                mem_used_sum += mu
                mem_used_cnt += 1
                mem_used_peak = mu if mem_used_peak is None else max(mem_used_peak, mu)

            if host_pid is not None:
                pm = _gpu_proc_mem_used_mb_by_hostpid(host_pid)
                if pm is not None:
                    proc_mem_sum += pm
                    proc_mem_cnt += 1
                    proc_mem_peak = pm if proc_mem_peak is None else max(proc_mem_peak, pm)

            if rc is not None:
                break
            time.sleep(args.interval)
    finally:
        t_out.join(timeout=5)
        t_err.join(timeout=5)

    duration = time.perf_counter() - t0

    # --- final cgroup stats
    cpu1 = _read_cpu_stat_v2() if use_cgv2 else {}
    io_r1, io_w1 = _read_io_stat_v2() if use_cgv2 else (0, 0)

    # CPU deltas in seconds
    def _delta_usec(k: str) -> float:
        return max(0, cpu1.get(k, 0) - cpu0.get(k, 0)) / 1e6

    cpu_user_s = _delta_usec("user_usec") if use_cgv2 else None
    cpu_sys_s = _delta_usec("system_usec") if use_cgv2 else None

    # IO deltas in MB
    disk_read_mb = max(0, io_r1 - io_r0) / 1e6
    disk_write_mb = max(0, io_w1 - io_w0) / 1e6

    max_rss_kb = int(mem_peak_bytes // 1024) if mem_peak_bytes else None

    proc_metrics = {
        "cpu_seconds_user": round(cpu_user_s, 3) if cpu_user_s is not None else None,
        "cpu_seconds_sys": round(cpu_sys_s, 3) if cpu_sys_s is not None else None,
        "disk_read_mb": round(disk_read_mb, 3),
        "disk_write_mb": round(disk_write_mb, 3),
        "max_rss_kb": max_rss_kb,
        "training_time_s": round(duration, 3),
        "source": "cgroup_v2" if use_cgv2 else "unknown",
    }
    print("#PROC_METRICS#" + json.dumps(proc_metrics))

    avg_util = (util_sum / util_cnt) if util_cnt else None
    avg_mem_used = (mem_used_sum / mem_used_cnt) if mem_used_cnt else None
    avg_proc_used = (proc_mem_sum / proc_mem_cnt) if proc_mem_cnt else None

    gpu_metrics = {
        "cuda_available": bool(cuda_available),
        "name": gpu_info.get("name"),
        "total_mb": gpu_info.get("total_mb"),
        # Backwards-compatible key (your pipeline expects gpu_peak_alloc_mb)
        # Here we treat "peak used by process" as the meaningful quantity.
        "peak_used_mb": int(proc_mem_peak) if proc_mem_peak is not None else (int(mem_used_peak) if mem_used_peak is not None else None),
        "avg_used_mb": round(avg_proc_used, 3) if avg_proc_used is not None else (round(avg_mem_used, 3) if avg_mem_used is not None else None),
        "avg_util_pct": round(avg_util, 3) if avg_util is not None else None,
        "peak_util_pct": util_peak,
        "note": "peak_used_mb is per-process if host PID mapping succeeds; otherwise device-level memory.used peak.",
    }
    print("#GPU_METRICS#" + json.dumps(gpu_metrics))

    return int(p.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
