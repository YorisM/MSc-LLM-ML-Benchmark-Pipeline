# static_checks.py

import subprocess, logging, json, sys

# ---------- Config ----------
PYLINT_SCORE_FLOOR = 6.0      # out-of-10  (tweak)
BANDIT_MAX_ISSUES  = 0        # 0 medium/high allowed
BANDIT_CFG         = "bandit.yml"

# ---------- Helpers ----------
def run_pylint(py_file: str) -> bool:
    """Return True if pylint global score >= floor."""
    logging.info("Running Pylint on %s", py_file)
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pylint", "--score", "y", "--output-format=json", py_file],
            text=True, stderr=subprocess.DEVNULL,
        )
        reports = json.loads(out)
        score   = reports[0]["score"]  # global evaluation
        logging.info("PyLint check passed with score: %.2f", score)
        return score >= PYLINT_SCORE_FLOOR
    except subprocess.CalledProcessError as e:
        logging.error("PyLint failed: %s", e)
        return False

def run_bandit(py_file: str) -> bool:
    """Return True if bandit finds <= BANDIT_MAX_ISSUES violating policy."""
    logging.info("Running Bandit on %s", py_file)
    cmd = [
        "bandit", "-q", "-c", BANDIT_CFG,
        "-f", "json", "-r", py_file
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
        report = json.loads(out)
        issues = len(report.get("results", []))
        logging.info(f'Bandit check passed. Found {issues} issues.')
        return issues <= BANDIT_MAX_ISSUES
    except subprocess.CalledProcessError as e:
        logging.error("Bandit failed: %s", e)
        return False
