# ./docker/run_docker.py

import os, subprocess, time, logging
from pathlib import Path

# small helper script to automatically run docker desktop when initialising the benchmark

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)

def _docker_ready() -> bool:
    p = _run(["docker", "info"])
    return p.returncode == 0

def _start_docker_desktop_windows() -> bool:
    # Common install paths:
    candidates = [
        r"C:\Program Files\Docker\Docker\Docker Desktop.exe",
        r"C:\Program Files\Docker\Docker\DockerDesktop.exe",
    ]
    exe = next((p for p in candidates if Path(p).exists()), None)
    if exe is None:
        logging.error("Docker Desktop executable not found in default locations.")
        return False

    logging.warning("Docker daemon not reachable. Launching Docker Desktop...")
    # Launch detached
    subprocess.Popen([exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True

def ensure_docker_running(*, timeout_s: int = 120, poll_s: float = 1.0) -> None:
    """
    Ensure Docker daemon is reachable. On Windows, tries to start Docker Desktop if needed.
    Raises RuntimeError if Docker is still not ready after timeout.
    """
    if _docker_ready():
        return

    if os.name == "nt":
        if not _start_docker_desktop_windows():
            raise RuntimeError("Docker not reachable and Docker Desktop could not be started.")
    else:
        raise RuntimeError("Docker not reachable. Start Docker daemon and retry.")

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _docker_ready():
            logging.info("Docker is up.")
            return
        time.sleep(poll_s)

    raise RuntimeError("Docker did not become ready in time. Is Docker Desktop starting correctly?")
