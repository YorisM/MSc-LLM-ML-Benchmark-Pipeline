# run_scripts.py

# Imports
import os, sys, subprocess, logging, time, argparse, psutil
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import execution_timeout, DOCKER_IMAGE
from utils import WSL_path


# - - - - - TODO - - - - - 
#   properly containerize scripts using docker
# - - - - - - - - - - - - -


def naive_safety_check(script_path):
    """
    Perform a basic (naive) safety check on the script by scanning for disallowed keywords.
    """

    dangerous_keywords = [
        'os.system',        # executing system commands
        'subprocess.Popen', # launching subprocesses
        # 'eval(',          # evaluating arbitrary expressions
        'exec(',            # executing arbitrary code
        '__import__',       # dynamic import of modules
        'import socket',    # network operations
        'shutil.rmtree'     # potentially dangerous file operations
    ]
    
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        logging.error(f"Error reading {script_path}: {e}")
        return False

    for keyword in dangerous_keywords:
        if keyword in code:
            logging.error(f"Script {script_path} contains dangerous keyword: {keyword}")
            return False
    logging.info("Successfully passed naïve safety check.")    
    return True

def collect_valid_scripts(base_folder):
    scripts = []
    for root, _, files in os.walk(base_folder):
        if os.path.basename(root) == "Failed Dry-run Scripts":
            continue
        for f in files:
            if f.endswith(".py"):
                scripts.append(os.path.join(root, f))
    logging.info(f"Collected valid scripts for execution: {scripts}")
    return scripts

def execute_script(
    script_path: str,
    timeout: float,
    dryrun: bool = False,
    use_docker: bool = True,
    safety_check: callable = None
):
    
    """
    Unified runner: does an optional naive_safety_check(), then runs the script
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
    
    # safety
    if safety_check and not safety_check(script_path):
        logging.error("Safety check failed: %s", script_path)
        return False, None, "", "Safety check", 0.0, 0.0
    
    rel_script = os.path.relpath(script_path, os.getcwd())
    logging.info("Relative script path: %s", rel_script)

    rel_script = rel_script.replace("\\", "/")
    logging.info("Relative script path: %s", rel_script)

    cmd = []

    """
    # build command
    if use_docker:
        mount_src = os.path.abspath(os.getcwd()).replace("\\", "/")

        volume_arg = f'{mount_src}:/workspace'
        logging.info("Docker volume arg: %s", volume_arg)

        WSL_arg = WSL_path(volume_arg)
        logging.info("WSL path: %s", WSL_arg)

        cmd = [
            "docker", "run", "--rm",
            "-v", WSL_arg,
            "-w", "/workspace",
            DOCKER_IMAGE,
            "python", rel_script
        ]
        logging.info("Docker CMD: %s", " ".join(cmd))
        if dryrun:
            cmd.append("--dryrun")
    """

    # path to mount on host, with forward slashes
    mount_src = os.path.abspath(os.getcwd()).replace("\\", "/")
    logging.info("Docker mount_src: %s", mount_src)

    # build the plain “host:container” volume spec—no embedded quotes!
    volume_arg = f"{mount_src}:/workspace"
    logging.info("Docker volume_arg: %s", volume_arg)

    # convert to WSL path
    volume_arg = WSL_path(volume_arg)
    logging.info("WSL volume_arg: %s", volume_arg)

    cmd = [
        "docker", "run", "--rm",
        "-v", volume_arg,            # exactly one colon in this string
        "-w", "/workspace",
        DOCKER_IMAGE,
        rel_script,        # rel_script already uses forward slashes
    ]

    if dryrun:
        cmd.append("--dryrun")
    else:
        cmd = [sys.executable, script_path]
        if dryrun:
            cmd.append("--dryrun")
    
    logging.info("Final CMD list: %r", cmd)
    proc = psutil.Process()
    mem_before = proc.memory_info().rss
    t0 = time.perf_counter()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=execution_timeout)
        ret = result.returncode
        out, err = result.stdout, result.stderr
        success = (ret == 0)
    except subprocess.TimeoutExpired:
        success, ret, out, err = False, None, "", "TimeoutExpired"
    except Exception as e:
        success, ret, out, err = False, None, "", f"Exception: {e}"

    mem_after = proc.memory_info().rss
    duration = time.perf_counter() - t0
    max_rss = (mem_after - mem_before) // 1024

    logging.info("%s %s → code=%s time=%.2fs mem=%dKB", "Dry-run" if dryrun else "Run", script_path, ret, duration, max_rss)
    return (script_path, success, ret, out, err, duration, max_rss)


def run_single_script(script_path, timeout):
    """
    Runs one script, returns a tuple:
      (path, returncode, stdout, stderr, duration_s, max_rss_kb)
    """

    start = time.perf_counter()
    proc = psutil.Process()
    before_mem = proc.memory_info().rss
    logging.info(f"Executing script: {script_path}...")

    # Implement Safety Measures
    logging.info("Performing naive safety check on script: %s", script_path)
    if not naive_safety_check(script_path):
        logging.error(f"Failed naive safety check on script: {script_path}")
        return (script_path, None, "", "Failed naive safety check", 0.0, 0.0)
  
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True,
            timeout=timeout
        )
        retval = result.returncode
        out, err = result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        retval, out, err = None, "", "Timeout"
    except Exception as e:
        retval, out, err = None, "", f"Exception: {e}"

    after_mem = proc.memory_info().rss
    duration = time.perf_counter() - start
    max_rss = after_mem - before_mem  # Kilobytes

    logging.info("Ran %s → code=%s, time=%.1fs, mem=%sKB", script_path, retval, duration, max_rss)
    return (script_path, retval, out, err, duration, max_rss)

def execute_scripts_in_batch(base_folder, max_workers=1):
    """
    Finds all valid scripts, runs them (in parallel if max_workers>1),
    shows a tqdm progress bar, and writes summary.csv at base_folder.
    Returns list of results.
    """

    scripts = collect_valid_scripts(base_folder)
    logging.info("Found %d scripts to run under %s", len(scripts), base_folder)
    results = []

    if max_workers == 1:
        # serial
        for script in tqdm(scripts, desc="Running scripts", unit="script"):
            results.append(run_single_script(script, execution_timeout))
    else:
        # parallel
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(run_single_script, s, execution_timeout): s
                       for s in scripts}
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
    df = pd.DataFrame(results,
        columns=["script","return_code","stdout","stderr","duration_s","max_rss_kb"])
    summary_path = os.path.join(base_folder, "summary.csv")
    df.to_csv(summary_path, index=False)
    logging.info("Wrote summary to %s", summary_path)
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="Base output folder to scan/run")
    p.add_argument("--max-workers", type=int, default=1,
                   help="1 = serial; >1 = parallel threads")
    args = p.parse_args()
    execute_scripts_in_batch(args.folder, args.max_workers)