# run_scripts.py

# Imports
import os
import sys
import logging
import subprocess
import time
import config


def script_safety_check(script_path):
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


def script_dryrun(script_path, timeout=300):
    """
    Run the given Python script using subprocess with a timeout.
    Returns (stdout, stderr) if successful, or None if it times out or fails the safety check.
    """

    logging.info("Running dry run for script: %s", script_path)
    try:
        result = subprocess.run(
            [sys.executable, script_path, "--dryrun"],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0:
            logging.info("Dry run succeeded for script: %s", script_path)
            return True, result.stdout, result.stderr
        else:
            logging.error("Dry run failed for script: %s (return code %d). STDERR:\n%s",
                          script_path, result.returncode, result.stderr)
            return False, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        logging.error("Dry run for script %s timed out after %d seconds.", script_path, timeout)
        return False, None, None
    except Exception as e:
        logging.error("Error during dry run for script %s: %s", script_path, e)
        return False, None, None


def main():
    output_dir = "./outputs/21-03/"  # example directory; adjust as needed
    script_filename = f"{output_dir}/generated_script_openai_chatgpt-4o-latest_21031640_1_a.py" #example script

    # Run the generated script with a 10-minute timeout.
    result = script_dryrun(script_filename, timeout=3600)
    if result:
        stdout, stderr = result
        print("Script Output:")
        print(stdout)
        if stderr:
            print("Script Errors:")
            print(stderr)
    else:
        print("Script failed to run or timed out.")
        return

if __name__ == "__main__":
    main()