# utils/run_id.py

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import re


TZ = ZoneInfo("Europe/Amsterdam")
DATE_FMT = "%m-%d"
OUT_ROOT = Path("./outputs")
ACTIVE_FILE = OUT_ROOT / ".active_run_id"
_RUN_RE = re.compile(r"^(?P<base>\d{2}-\d{2})(?:\((?P<n>\d+)\))?$")


def _today_base() -> str:
    return datetime.now(TZ).strftime(DATE_FMT)

def get_active_run_id() -> str | None:
    try:
        txt = ACTIVE_FILE.read_text(encoding="utf-8").strip()
        return txt or None
    except FileNotFoundError:
        return None

def set_active_run_id(run_id: str) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ACTIVE_FILE.write_text(run_id, encoding="utf-8")

def _next_available_run_id(base: str) -> str:
    """
    Returns base, or base(2), base(3), ... such that outputs/<run_id>/ does not exist.
    """
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    candidate = base
    k = 2
    while (OUT_ROOT / candidate).exists():
        candidate = f"{base}({k})"
        k += 1
    return candidate

def get_or_create_run_id(*, create_new: bool = False) -> str:
    """
    - If create_new=True: always create a fresh run for today (12-19, 12-19(2), ...).
    - If create_new=False:
        - If active run is today: reuse it.
        - Otherwise (missing / old date): create today's run and set active.
    """
    today = _today_base()

    if not create_new:
        active = get_active_run_id()
        if active:
            m = _RUN_RE.match(active)
            if m and m.group("base") == today:
                return active

    run_id = _next_available_run_id(today)
    set_active_run_id(run_id)
    return run_id

def new_run_id() -> str:
    """Convenience: always create a new run id for today and set it active."""
    return get_or_create_run_id(create_new=True)
