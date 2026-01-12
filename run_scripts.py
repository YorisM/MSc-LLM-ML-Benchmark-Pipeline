# run_scripts.py

import os, sys, re, config, subprocess, logging, time, argparse, psutil, json, pathlib
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.utils import append_to_response_json, iter_input_dir, SKIP_DIRS
from utils.run_id import get_active_run_id

# Execution Flow
# (main.py)  ─▶ execute_scripts_in_batch(...)
#                 └─▶ run_single_script(..., dryrun=?, use_docker=?)
#                        └─▶ execute_script(...)


def _find_scripts(base_folder: str | Path) -> list[str]:
    base = Path(base_folder)

    # Case 1: user points directly to a script file
    if base.is_file() and base.suffix.lower() == ".py":
        if base.parent.name in SKIP_DIRS:
            return []
        return [str(base)]

    if not base.is_dir():
        logging.warning("Input path does not exist or is not a directory: %s", base)
        return []

    scripts: list[str] = []

    # Case 2: user points to a model folder containing scripts directly
    direct = sorted(base.glob("script_*.py"))
    if direct:
        for py in direct:
            if py.parent.name in SKIP_DIRS:
                continue
            scripts.append(str(py))
        logging.info("Collected %d valid scripts for execution", len(scripts))
        return scripts

    # Case 3: original behaviour (input-dir is above Q dirs / model dirs)
    for q_dir in iter_input_dir(base):
        for py in q_dir.rglob("script_*.py"):
            if py.parent.name in SKIP_DIRS:
                continue
            scripts.append(str(py))

    logging.info("Collected %d valid scripts for execution", len(scripts))
    return scripts

def _challenge_from_script(script_path: str) -> str:
    """
    Derive the challenge name from a script path without hardcoding names.
    Works for:
      - .../challenges/<CHALLENGE>/...
      - .../outputs/<DATE>/<CHALLENGE>/Q<NUM>/...
      - any path containing .../<CHALLENGE>/Q<NUM>/...
    Case-insensitive; Windows/Unix separators supported.
    """

    s = script_path.replace("\\", "/")

    # .../challenges/FOURTOPS/Q1/script_x.py  ->  FOURTOPS
    m = re.search(r"/challenges/([^/]+)/", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    # 2) Outputs layout: outputs/<DATE>/<CHALLENGE>/(Q\d+|anything)/
    m = re.search(r"/outputs/[^/]+/([^/]+)/", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    # 3) Generic: capture the segment preceding Q<number>
    m = re.search(r"/([^/]+)/Q\d+/", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    # 4) Fallback: try the parent directory name (second last segment)
    parts = [p for p in s.split("/") if p]
    if len(parts) >= 2:
        cand = parts[-2]
        logging.warning("Falling back to parent folder as challenge: %s (path=%s)", cand, script_path)
        return cand

    raise ValueError("Cannot derive challenge from path " + script_path)

def _mount_args(challenge: str, host_root: str) -> list[str]:
    base = pathlib.Path(host_root) / "challenges" / challenge / "data"
    return [
        "-v", f"{host_root}:/workspace:rw",
        "-v", f"{base/'train'}:/data/train:ro",
        "-v", f"{host_root}/outputs:/workspace/out:rw"
    ]

def execute_script(script_path: str, timeout: float, dryrun: bool = False, use_docker: bool = True):
    """
    Unified runner:

    Returns:
      (script_path: str,
       success: bool,
       return_code: Optional[int],
       stdout: str,
       stderr: str,
       duration_s: float,
       max_rss_kb: float)
    """

    logging.info(f"{'Dry-run' if dryrun else 'Run'} start: {script_path}")
    
    # Debugging Block
    logging.debug(f"Script to run inside container: {script_path}")
    rel_script = os.path.relpath(script_path, os.getcwd())
    logging.debug("Relative script path: %s", rel_script)
    rel_script = rel_script.replace("\\", "/")
    logging.debug("Relative script path: %s", rel_script)
    mount_src = os.path.abspath(os.getcwd()).replace("\\", "/")
    logging.debug("Docker mount_src: %s", mount_src)

    cmd = []
    if use_docker:

        cmd = ["docker", "run", "--rm"]

        # Only add constraints if not 0
        if config.CPU_LIMIT != 0:
            cmd.append(f"--cpus={config.CPU_LIMIT}")
        if config.MEMORY_LIMIT_GB != 0:
            cmd.append(f"--memory={config.MEMORY_LIMIT_GB}g")
        if config.PIDS_LIMIT != 0:
            cmd.append(f"--pids-limit={config.PIDS_LIMIT}")

        llm_io_py           = (Path(mount_src) / "utils" / "llm_io.py").resolve()
        loaderspec_py       = (Path(mount_src) / "utils" / "loaderspec.py").resolve()
        suffix_utils_py     = (Path(mount_src) / "utils" / "suffix_utils.py").resolve()
        resource_monitor_py = (Path(mount_src) / "utils" / "resource_monitor.py").resolve() 

        cmd += [
            "--gpus", "all",
            "--network", "none", 
            "--read-only", 
            "--cap-drop", "ALL",
            "--security-opt", f"seccomp=docker/seccomp_profile.json",
            "--tmpfs", "/tmp:rw,noexec,nosuid",
            "--tmpfs", "/dev/shm:rw",
            "-e", f"DRYRUN_TIMEOUT_S={config.DRYRUN_TIMEOUT_S}",
            "-e", f"TRAIN_TIMEOUT_S={config.TRAIN_TIMEOUT_S}",
            "-e", f"EVAL_TIMEOUT_S={config.EVAL_TIMEOUT_S}",
            "-e", "PYTHONPATH=/workspace",
            "-e", "MPLCONFIGDIR=/tmp/mpl",
            "-w", "/workspace",

            # mount volume
            "-v", f"{llm_io_py}:/workspace/utils/llm_io.py:ro",
            "-v", f"{loaderspec_py}:/workspace/utils/loaderspec.py:ro",
            "-v", f"{suffix_utils_py}:/workspace/utils/suffix_utils.py:ro",
            "-v", f"{resource_monitor_py}:/workspace/utils/resource_monitor.py:ro"
        ]

        # add volumes
        logging.debug("Deriving challenge from path: %s", script_path)
        challenge = _challenge_from_script(script_path)
        cmd += _mount_args(challenge, mount_src)

        # select entrypoint + image
        cmd += [
            "--entrypoint", "/usr/local/bin/train.sh",
            "llm-sandbox:latest",
            rel_script
        ]

        if dryrun:
            cmd.append("--dryrun")

        logging.debug("DOCKER CMD: %s", " ".join(cmd))

        proc = psutil.Process()
        mem_before = proc.memory_info().rss
        t0 = time.perf_counter()

    else:
        cmd = [sys.executable, script_path]
        if dryrun:
            cmd.append("--dryrun")

    logging.debug("Final CMD list: %r", cmd)

    # Actually run the script
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                                timeout=config.TRAIN_TIMEOUT_S)
        ret = result.returncode
        out, err = result.stdout, result.stderr
        success = (ret == 0)
        
    except subprocess.TimeoutExpired as e:
        logging.error("Timeout (%.2fs) while running %s", e.timeout, cmd)
        success, ret, out, err = False, None, "", "TimeoutExpired"
    except Exception as e:
        logging.exception("Exception while running %s", cmd)
        success, ret, out, err = False, None, "", f"Exception: {e}"

    # Pull metrics from output (prefer container-emitted sentinels)
    metrics_json = None
    combined = out + "\n" + err
    for line in reversed(combined.splitlines()):
        if line.startswith("#TRAIN_METRICS#"):
            try:
                metrics_json = json.loads(line[len("#TRAIN_METRICS#"):])
            except json.JSONDecodeError:
                logging.warning("Malformed TRAIN_METRICS in %s", script_path)
            break

    gpu_metrics = None
    for line in reversed(combined.splitlines()):
        if line.startswith("#GPU_METRICS#"):
            try:
                gpu_metrics = json.loads(line[len("#GPU_METRICS#"):])
            except json.JSONDecodeError:
                logging.warning("Malformed GPU_METRICS in %s", script_path)
            break

    proc_metrics = None
    for line in reversed(combined.splitlines()):
        if line.startswith("#PROC_METRICS#"):
            try:
                proc_metrics = json.loads(line[len("#PROC_METRICS#"):])
            except json.JSONDecodeError:
                logging.warning("Malformed PROC_METRICS in %s", script_path)
            break    
    
    # Process metrics
    duration = time.perf_counter() - t0
    cpu_user = proc_metrics.get("cpu_seconds_user") if proc_metrics else None
    cpu_sys  = proc_metrics.get("cpu_seconds_sys") if proc_metrics else None
    disk_read_mb  = proc_metrics.get("disk_read_mb") if proc_metrics else None
    disk_write_mb = proc_metrics.get("disk_write_mb") if proc_metrics else None
    max_rss = proc_metrics.get("max_rss_kb") if proc_metrics else None

    # Logging
    logging.info("Subprocess finished (rc=%s, duration=%.2fs)", ret, duration)
    logging.debug("STDOUT: %s", out)
    logging.debug("STDERR: %s", err)

    # Create metrics dictionary
    wall_time_s = round(duration, 3)
    training_time_s = proc_metrics.get("training_time_s") if proc_metrics else None

    resources = {
        "wall_time_s": wall_time_s,
        "training_time_s": training_time_s,
        "cpu_seconds_user": cpu_user,
        "cpu_seconds_sys":  cpu_sys,
        "disk_read_mb":     disk_read_mb,
        "disk_write_mb":    disk_write_mb,
        "max_rss_kb":       max_rss,
        "proc_source":      proc_metrics.get("source") if proc_metrics else None,

        "cuda_available":   gpu_metrics.get("cuda_available") if gpu_metrics else None,
        "gpu_name":         gpu_metrics.get("name") if gpu_metrics else None,
        "gpu_total_mb":     gpu_metrics.get("total_mb") if gpu_metrics else None,
        "per_process_gpu_mem": gpu_metrics.get("per_process_gpu_mem") if gpu_metrics else None,

        "gpu_peak_alloc_mb": (
            gpu_metrics.get("peak_alloc_mb")
            if (gpu_metrics and "peak_alloc_mb" in gpu_metrics)
            else (gpu_metrics.get("peak_used_mb") if gpu_metrics else None)
        ),
        "gpu_avg_util_pct": gpu_metrics.get("avg_util_pct") if gpu_metrics else None,
        "gpu_peak_util_pct":gpu_metrics.get("peak_util_pct") if gpu_metrics else None,
        "gpu_avg_used_mb":  gpu_metrics.get("avg_used_mb") if gpu_metrics else None,
    }

    # derive companion JSON path: script_X.py -> response_X.json
    base = os.path.basename(script_path).replace("script_", "response_")
    json_file = os.path.join(os.path.dirname(script_path),
                                os.path.splitext(base)[0] + ".json")

    STD = {
        "stdout": out, 
        "stderr": err
    }
    
    if not dryrun:
        append_to_response_json(json_file, "Training",
            {   
                "__timestamp": datetime.now(timezone.utc).isoformat(),
                "passed": bool(success),
                "resources": resources,
                "metrics": metrics_json,
                "STD": STD
            })

    mem_str = f"{max_rss}KB" if max_rss is not None else "n/a"
    logging.info("%s %s → code=%s time=%.2fs mem=%s", "Dry-run" if dryrun else "Run", script_path, ret, duration, mem_str)

    return (script_path, success, ret, out, err, duration, max_rss)

def run_single_script(script_path: str, *, dryrun: bool = False, 
                      use_docker: bool = True, timeout: float | None = None,
                      ) -> tuple:
    t = timeout or (config.DRYRUN_TIMEOUT_S if dryrun else config.TRAIN_TIMEOUT_S)
    return execute_script(
        script_path   = script_path,
        timeout       = t,
        dryrun        = dryrun,
        use_docker    = use_docker,
    )

def execute_scripts_in_batch(base_folder, max_workers=1, *, dryrun=False, use_docker=True):
    """
    Finds all valid scripts, runs them (in parallel if max_workers>1),
    shows a tqdm progress bar, and writes summary.csv at base_folder.
    Returns list of results.
    """

    scripts = _find_scripts(base_folder)
    logging.info("Found %d scripts to run under %s", len(scripts), base_folder)
    results = []

    if max_workers == 1:
        # serial
        for script in tqdm(scripts, desc="Running scripts", unit="script"):
            results.append(run_single_script(script, dryrun = dryrun, 
                                                     use_docker = use_docker))
    else:
        # parallel
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(run_single_script, s, config.TRAIN_TIMEOUT_S,
                                dryrun, use_docker): s for s in scripts}
            for fut in tqdm(as_completed(futures),
                            total=len(scripts),
                            desc="Running scripts", unit="script"):
                try:
                    results.append(fut.result())
                except Exception as e:
                    script = futures[fut]
                    logging.error("Unhandled exception in %s: %s", script, e)
                    results.append((script, None, "", f"Exception: {e}", 0.0, 0))

    # write summary CSV
    df = pd.DataFrame(
        results,
        columns=[
            "script",          # absolute path
            "success",         # True / False
            "return_code",     # int | None
            "stdout",          # str
            "stderr",          # str
            "duration_s",      # float
            "max_rss_kb",      # int
        ],
    )

    summary_path = os.path.join(base_folder, "summary.csv")
    df.to_csv(summary_path, index=False)
    logging.info("Wrote summary to %s", summary_path)
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", nargs="?", default=None,
        help="Base output folder to scan/run. If omitted, uses outputs/<run-id>/<challenge>[/<question>].",)
    
    p.add_argument("--run-id", dest="run_id", default=None,
        help="Run id under outputs/ (e.g. 01-06 or 01-06(2)). Defaults to the active run id.",)
    
    p.add_argument("--challenge", "-c", default=None,
        help="Challenge name (e.g. FOURTOPS, TRACKFORMERS). Required if folder is omitted.",)
    
    p.add_argument("-q", default=None, help="Optional question id (e.g. Q1) if folder is omitted.",)

    p.add_argument("--max-workers", type=int, default=1, help="1 = serial; >1 = parallel threads")

    args = p.parse_args()

    folder = args.folder
    if folder is None:
        run_id = args.run_id or get_active_run_id()
        if not run_id:
            p.error("No folder provided and no active run id. Provide folder or --run-id.")

        if not args.challenge:
            p.error("When folder is omitted you must pass --challenge (e.g. --challenge FOURTOPS).")

        folder_path = Path("outputs") / run_id / args.challenge
        if args.question:
            folder_path = folder_path / args.question

        folder = str(folder_path)

    execute_scripts_in_batch(folder, args.max_workers)
