# run_scripts.py

# Imports
import os
import sys
import subprocess
import logging
import time
import argparse
import psutil
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import execution_timeout


# - - - - - TODO - - - - - 
#   
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
    return True
 

def collect_valid_scripts(base_folder):
    scripts = []
    for root, _, files in os.walk(base_folder):
        if os.path.basename(root) == "Failed Dry-run Scripts":
            continue
        for f in files:
            if f.endswith(".py"):
                scripts.append(os.path.join(root, f))
    return scripts


def run_single_script(script_path, timeout):
    """
    Runs one script, returns a tuple:
      (path, returncode, stdout, stderr, duration_s, max_rss_kb)
    """
    start = time.perf_counter()
    proc = psutil.Process()
    before_mem = proc.memory_info().rss

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
    max_rss = after_mem - before_mem  # Kilobytes on Linux, bytes on macOS; approximate

    logging.info("Ran %s → code=%s, time=%.1fs, mem=%sKB",
                 script_path, retval, duration, max_rss)
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