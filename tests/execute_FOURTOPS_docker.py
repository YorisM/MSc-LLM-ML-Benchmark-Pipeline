# tests/test_execute_real_fourtops.py

import os, sys, glob, pytest

# insert project root (one level up) to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from run_scripts import execute_script, naive_safety_check

# adjust this pattern to point at your real training script(s)
SCRIPT_PATTERN = "outputs/23-04/FOURTOPS/Q1/deepseek_deepseek-r1/script_deepseek_deepseek-r1_1739_3.py"
# SCRIPT_PATTERN = "G:\My Drive\Neurophysics Master\Internship\Notebooks\Benchmark\outputs\23-04\FOURTOPS\Q1\deepseek_deepseek-r1\script_deepseek_deepseek-r1_1739_3.py"

@pytest.fixture(scope="module")
def real_scripts():
    scripts = glob.glob(SCRIPT_PATTERN)
    if not scripts:
        pytest.skip(f"No FOURTOPS scripts matching {SCRIPT_PATTERN} found")
    # we'll just test the first one
    return scripts[0]

def test_real_fourtops_script_dryrun(real_scripts):
    script = real_scripts
    # run in dryrun mode so it only does 1 epoch
    script_path, success,  returncode, stdout, stderr, duration_s, max_rss_kb = execute_script(
        script_path=SCRIPT_PATTERN, 
        timeout=300,                             # give it up to 5 minutes
        dryrun=True,                             # pass --dryrun under the hood
        use_docker=True,                         # assuming local exec
        safety_check=naive_safety_check          # or your naive_safety_check
    )

    # assertions:
    assert success, f"Script failed: code={returncode} stderr={stderr}"
    assert returncode == 0
    # (optionally) check that it printed something expected
    assert "Epoch 1" in stdout or "dry-run complete" in stdout
    # resource usage metrics should be non-negative
    assert duration_s >= 0
    assert max_rss_kb >= 0

if __name__ == "__main__":
    test_real_fourtops_script_dryrun(SCRIPT_PATTERN)
    