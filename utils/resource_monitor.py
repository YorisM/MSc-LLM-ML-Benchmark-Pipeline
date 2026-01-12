# utils.resource_monitor.py

"""
Run a command while collecting container-accurate CPU/memory (via cgroups)
and best-effort GPU metrics (via nvidia-smi), then emit sentinel JSON lines:

  #PROC_METRICS#{...}
  #GPU_METRICS#{...}

Designed to be called from train.sh so suffix scripts remain untouched.
"""

from __future__ import annotations
import os, sys, json, argparse, subprocess, threading, time
from typing import Any, Dict, Optional, Tuple


CGROUP_ROOT = "/sys/fs/cgroup"
_CLK_TCK = os.sysconf(os.sysconf_names["SC_CLK_TCK"])


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

def _read_proc_stat(pid: int):
    # /proc/<pid>/stat: utime is field 14, stime is field 15 (1-indexed)
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            s = f.read().strip()
        # comm is inside parentheses and may contain spaces
        rparen = s.rfind(")")
        rest = s[rparen+2:].split()
        utime = int(rest[11])  # 14th field overall
        stime = int(rest[12])  # 15th field overall
        return utime, stime
    except Exception:
        return None

def _read_proc_rss_kb(pid: int):
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])  # already kB
    except Exception:
        pass
    return None

def _read_children_pids(pid: int) -> list[int]:
    # /proc/<pid>/task/<pid>/children contains space-separated child PIDs
    try:
        with open(f"/proc/{pid}/task/{pid}/children", "r") as f:
            txt = f.read().strip()
        if not txt:
            return []
        return [int(x) for x in txt.split()]
    except Exception:
        return []
    
def _collect_proc_tree_ticks(root_pid: int) -> tuple[dict[int, tuple[int, int]], int | None]:
    """
    Return (ticks_map, rss_kb_sum) for root + descendants currently visible.
    ticks_map: pid -> (uticks, sticks) in clock ticks.
    rss_kb_sum: sum of VmRSS over tree or None if nothing readable.
    """
    seen: set[int] = set()
    stack: list[int] = [root_pid]
    ticks: dict[int, tuple[int, int]] = {}

    rss_kb_sum = 0
    rss_seen_any = False

    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)

        st = _read_proc_stat(pid)
        if st is not None:
            ticks[pid] = st  # (uticks, sticks)

        rk = _read_proc_rss_kb(pid)
        if rk is not None:
            rss_kb_sum += rk
            rss_seen_any = True

        stack.extend(_read_children_pids(pid))

    return ticks, (rss_kb_sum if rss_seen_any else None)

def main() -> int:
    def _parse_args():
        ap = argparse.ArgumentParser()
        ap.add_argument("--interval", type=float, default=0.5, help="Sampling interval seconds")
        # Everything after options is the command to run
        ap.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to execute")
        args = ap.parse_args()

        cmd = list(args.cmd)
        # Be tolerant if caller includes an explicit '--'
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        if not cmd:
            ap.error("missing command to run (e.g. ... -- python -u script.py)")

        return args.interval, cmd
    
    interval, cmd = _parse_args()

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

    peak_rss_kb = 0
    peak_rss_valid = False

    cpu_max_ticks: dict[int, tuple[int, int]] = {}

    try:
        while True:
            rc = p.poll()

            # ---- sample proc tree (CPU ticks + RSS) ----
            ticks_now, rss_kb = _collect_proc_tree_ticks(p.pid)

            # update per-pid max ticks so we don't lose CPU time after PIDs exit
            for pid, (u, s) in ticks_now.items():
                prev = cpu_max_ticks.get(pid)
                if prev is None:
                    cpu_max_ticks[pid] = (u, s)
                else:
                    pu, ps = prev
                    cpu_max_ticks[pid] = (max(pu, u), max(ps, s))

            if rss_kb is not None:
                peak_rss_kb = max(peak_rss_kb, rss_kb)
                peak_rss_valid = True

            # ---- sample cgroup memory peak (if available) ----
            if use_cgv2 and mem_peak_file is None:
                try:
                    cur = _read_mem_current_v2()
                    if cur > mem_peak_bytes:
                        mem_peak_bytes = cur
                except Exception:
                    pass

            # ---- sample GPU (best-effort) ----
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

            # ---- exit condition ----
            if rc is not None:
                break

            time.sleep(interval)
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

    total_uticks = 0
    total_sticks = 0
    for (u, s) in cpu_max_ticks.values():
        total_uticks += u
        total_sticks += s

    cpu_user_s_ptree = total_uticks / _CLK_TCK if cpu_max_ticks else None
    cpu_sys_s_ptree = total_sticks / _CLK_TCK if cpu_max_ticks else None

    proc_metrics = {
        "cpu_seconds_user": round(cpu_user_s_ptree, 3) if cpu_user_s_ptree is not None else None,
        "cpu_seconds_sys": round(cpu_sys_s_ptree, 3) if cpu_sys_s_ptree is not None else None,
        "max_rss_kb": int(peak_rss_kb) if peak_rss_valid else None,
        "training_time_s": round(duration, 3),
        "source": "proc_tree_ticks",
    }

    try:
        print("#PROC_METRICS#" + json.dumps(proc_metrics), flush=True)
    except Exception as e:
        print(f"resource_monitor.py: WARNING: failed to emit CPU metrics: {e}", file=sys.stderr, flush=True)

    avg_util = (util_sum / util_cnt) if util_cnt else None
    avg_mem_used = (mem_used_sum / mem_used_cnt) if mem_used_cnt else None
    avg_proc_used = (proc_mem_sum / proc_mem_cnt) if proc_mem_cnt else None

    per_process_gpu_mem = (proc_mem_cnt > 0)

    # note: peak_used_mb is per-process if host PID mapping succeeds; otherwise device-level memory.used peak
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
        "per_process_gpu_mem": bool(per_process_gpu_mem)
    }

    try:
        print("#GPU_METRICS#" + json.dumps(gpu_metrics), flush=True)
    except Exception as e:
        print(f"resource_monitor.py: WARNING: failed to emit GPU metrics: {e}", file=sys.stderr, flush=True)
    # ALWAYS exit with child rc
    return int(p.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
