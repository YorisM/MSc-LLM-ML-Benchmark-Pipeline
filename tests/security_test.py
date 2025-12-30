# tests/security_test.py

"""
Security smoke-test for the Docker-Desktop setup (seccomp only).

Usage:
    python tests/security_test.py FOURTOPS
Returns exit-code 0 if every check passes.
"""

import sys, subprocess, textwrap, tempfile, shutil
from pathlib import Path

# ----------------------------------------------------------------------
# 0.  CLI & paths
# ----------------------------------------------------------------------
if len(sys.argv) != 2:
    sys.exit("Usage: python security_smoke_test.py <CHALLENGE_NAME>")
challenge = sys.argv[1]

proj_root = Path(__file__).resolve().parents[2]          # …/Benchmark
print(f"Project root: {proj_root}")

data_root = proj_root / "Benchmark" / "challenges" / challenge / "data"
out_root  = proj_root / "Benchmark" / "outputs"
seccomp   = proj_root / "Benchmark" / "docker" / "seccomp_profile.json"

# dummy files so the mounts aren’t empty
for sub in ("train", "val"):
    (data_root / sub).mkdir(parents=True, exist_ok=True)
    (data_root / sub / "dummy.txt").write_text("dummy\n")
out_root.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 1.  create the inner test script (ASCII only)
# ----------------------------------------------------------------------
inner = textwrap.dedent(r"""
    import os, sys, errno
    from pathlib import Path
    from datetime import datetime as dt

    OK,BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
    res=[]
    def log(ok,msg,e=None):
        print(f"{OK if ok else BAD} - {msg}" + (f": {e}" if e else ""))
        res.append(ok)
    def ok(msg,f):   # expect success
        try: f(); log(True,msg)
        except Exception as e: log(False,msg,e)
    def fail(msg,f): # expect failure
        try: f(); log(False,msg,"unexpected success")
        except Exception as e: log(True,msg,e)

    # 1. read checks
    ok("list /data/train", lambda: list(Path('/data/train').iterdir()))
    ok("list /data/val",   lambda: list(Path('/data/val').iterdir()))
    fail("access /data/test", lambda: list(Path('/data/test').iterdir()))

    # 2. write checks
    stamp=dt.now().strftime('%Y%m%d_%H%M%S')
    out = Path('/workspace/out')
    ok("write artefact", lambda: (out/f"chk_{stamp}_state.pt").write_text("ok"))
    ok("write extra file (allowed)", lambda: (out/f"extra_{stamp}.txt").write_text("ok"))
    fail("write in train dir", lambda: Path('/data/train/evil.txt').write_text("no"))
    fail("write in $HOME", lambda: Path.home().joinpath('evil.txt').write_text("no"))

    print(f"\nSummary: {sum(res)}/{len(res)} PASS")
    sys.exit(0 if all(res) else 1)
""").strip()

tmpdir     = Path(tempfile.mkdtemp(prefix="sec_smoke_"))
inner_path = tmpdir / "inner_test.py"
inner_path.write_text(inner, encoding="utf-8")

# ----------------------------------------------------------------------
# 2.  docker run command
# ----------------------------------------------------------------------
cmd = [
    "docker","run","--rm",
    "--read-only",
    "--cap-drop","ALL",
    "--security-opt", f"seccomp={seccomp}",
    "--network","none",
    "--tmpfs","/tmp:rw,noexec,nosuid",
    "--tmpfs","/dev/shm:rw",
    "-v",f"{data_root/'train'}:/data/train:ro",
    "-v",f"{data_root/'val'}:/data/val:ro",
    "-v",f"{out_root}:/workspace/out:rw",
    "-v",f"{inner_path}:/workspace/inner_test.py:ro",
    "--entrypoint","python",               # skip wrapper; test raw sandbox
    "llm-training-sandbox:latest",
    "/workspace/inner_test.py"
]

print("Running docker command:\n", " \\\n ".join(map(str, cmd)),"\n")
rc = subprocess.call(cmd)
print("\nContainer exited with", rc)

# ----------------------------------------------------------------------
# 3.  cleanup
# ----------------------------------------------------------------------
shutil.rmtree(tmpdir)
sys.exit(rc)
