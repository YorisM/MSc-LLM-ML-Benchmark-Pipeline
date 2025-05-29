# static_checks.py

import subprocess, logging, json, sys, textwrap

PYLINT_SCORE_FLOOR = 6.0      # out-of-10  (tweak)
BANDIT_MAX_ISSUES  = 0        # 0 high allowed

def run_pylint(py_file: str) -> bool:
    """
    Run PyLint.

    RETURNS
    Tuple : [bool, JSON] : [True / False, diagnostics report]
    """

    logging.info("Running Pylint on %s", py_file)
    cmd = [
        sys.executable, "-m", "pylint",
        "--output-format=json",
        "--score=n",            # no aggregate score calculation
        py_file,
    ]

    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        out = e.output or ""

    # guard empty JSON
    if not out.strip():
        logging.error("Pylint produced no JSON output.")
        return False, {}

    reports = json.loads(out)
    # collect blocking messages
    blocking = [r for r in reports if r.get("type") in {"fatal", "error"}]

    if not blocking:
        logging.info("Pylint OK (no fatal/error messages).")
        return True, reports

    preview = "\n".join(
        f"{r.get('type')}:{r.get('symbol')}:{r.get('message', '')} @L{r.get('line', '?')}"
        for r in blocking[:5]
    )

    logging.error("Pylint failed (%d blocking issues):\n%s", len(blocking), preview or "<none>")

    return False, reports

def run_bandit(py_file: str) -> bool:
    """
    Run bandit with project policy.  We parse JSON even on non-zero
    exit status to report concrete problems.

    RETURNS
    Tuple : [bool, JSON] : [True / False, diagnostics report]
    """

    logging.info("Running Bandit on %s", py_file)
    cmd = [
        "bandit", "-q",
        "--severity-level", "high",          # keep only HIGH
        "-s", "B101,B311,B403",              # skips: 
        "-x", "tests",                       # exclude dirs
        "-f", "json", "-r", py_file,
    ]
    
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        out = e.output or ""
    except Exception as e:
        logging.error("Bandit invocation failed: %s", e)
        return False, {"Bandit invocation failed: %s", e}

    if not out.strip():
        logging.error("Bandit produced no JSON output.")
        return False, {"Bandit produced no JSON output."}
   
    try:
        report  = json.loads(out)
    except json.JSONDecodeError as e:
        logging.error("Bandit produced invalid JSON: %s", e)
        return False, {"Bandit produced invalid JSON: %s", e}

    # Guard against potential future schema changes
    issues = report.get("results", [])
    if not isinstance(issues, list):
        logging.error("Bandit JSON has unexpected structure - no 'results' list")
        return False, {"Bandit JSON has unexpected structure - no 'results' list"}
   
    n_iss   = len(issues)
    if n_iss <= BANDIT_MAX_ISSUES:
        logging.info("Bandit OK (issues=%d).", n_iss)
        return True, report
    
    # Build short preview for logging
    preview_lines = []
    for iss in issues[:5]:
        severity = iss.get("issue_severity", "?")
        test     = iss.get("test_name", "?")
        line     = iss.get("line_number", "?")
        preview_lines.append(f"{severity}: {test} @ L{line}")
    preview = "\n".join(preview_lines)

    logging.error(textwrap.dedent(f"""
        Bandit failed on {py_file}
        ─ issues : {n_iss}
        ─ preview:\n{preview}
    """))
    return False, report
