# utils/utils.py

import os, pathlib, platform, json, datetime, logging
from typing import Union, TextIO, Optional, Iterator
from sys import stdout


# Config
SKIP_DIRS: set[str] = {"Failed Dry-run Scripts", "StaticFail"}


class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        # Local imports to avoid hard deps at module import time
        try:
            import numpy as np
        except Exception:
            np = None
        try:
            import torch
        except Exception:
            torch = None
        try:
            from pathlib import Path
        except Exception:
            Path = None
        try:
            from enum import Enum
        except Exception:
            Enum = None

        # ----- numpy scalars/arrays -----
        if np is not None:
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()

        # ----- torch tensors -----
        if torch is not None and isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()

        # ----- pathlib -----
        if Path is not None and isinstance(obj, Path):
            return str(obj)

        # ----- enums -----
        if Enum is not None and isinstance(obj, Enum):
            return obj.value

        # ----- exceptions (e.g., FileNotFoundError) -----
        if isinstance(obj, BaseException):
            return {"error": obj.__class__.__name__, "message": str(obj)}

        # ----- containers not handled by default -----
        # sets/frozensets: convert to list with stable, comparable keys
        if isinstance(obj, (set, frozenset)):
            def _coerce(e):
                # Keep JSON-native scalars as-is; stringify everything else
                if isinstance(e, (str, int, float, bool)) or e is None:
                    return e
                if isinstance(e, BaseException):
                    return {"error": e.__class__.__name__, "message": str(e)}
                # Avoid deep recursion for arbitrary objects
                return str(e)
            items = [_coerce(e) for e in obj]
            # Deterministic order without comparing unlike types
            return sorted(items, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))

        # tuples: represent as JSON arrays
        if isinstance(obj, tuple):
            return list(obj)

        # Fallback
        return super().default(obj)

def WSL_path(host_path: str) -> str:
    """
    Convert Windows 'G:\\foo\\bar' to '/mnt/g/foo/bar'.
    On Linux/macOS it returns the original path unchanged.
    """
    if platform.system() == 'Windows':
        # Use pathlib to split the drive and the rest
        p = pathlib.Path(host_path).resolve()
        drive = p.drive.rstrip(':').lower()            # 'g'
        rest  = '/'.join(p.parts[1:])                  # 'foo/bar'
        return f"/mnt/{drive}/{rest}"
    else:
        return host_path

def rename_file_with_suffix(old_file, suffix):
    base, ext = os.path.splitext(old_file)
    new_file = f"{base}{suffix}{ext}"
    os.rename(old_file, new_file)
    return new_file

def _tail(text: str, max_chars: int = 4000) -> str:
    """Return the last *max_chars* of text."""
    return text[-max_chars:] if len(text) > max_chars else text

def append_to_response_json(json_path: str, section: str, payload: dict, *, trim_output: bool = True,
                            ts: bool = True) -> None:

    p = pathlib.Path(json_path)

    if not p.exists() or p.stat().st_size == 0:
        blob = {}
    else:
        blob = json.loads(p.read_text(encoding="utf-8"))

    if trim_output:
        for k in ("stdout", "stderr", "stdout_tail", "stderr_tail"):
            if k in payload and isinstance(payload[k], str):
                payload[k] = _tail(payload[k])

    if ts:
        payload = {"__timestamp": datetime.datetime.utcnow()
                                      .isoformat(timespec="seconds")+'Z',
                   **payload}
    
    blob[section] = payload

    p.write_text(json.dumps(blob, ensure_ascii=False, indent=2, cls=_NpEncoder), 
                encoding="utf-8")
    
    logging.debug("Updated %s -> %s", p.name, section)

def iter_input_dir(base: pathlib.Path | str) -> Iterator[pathlib.Path]:
    """
    Yield every outputs/<DATE>/<CHALLENGE>/<QUESTION> directory that lives
    *under* `base`, no matter whether `base` itself is
      • outputs/<DATE>/…
      • outputs/<DATE>/<CHALLENGE>/…
      • outputs/<DATE>/<CHALLENGE>/<QUESTION>
    """
    root = pathlib.Path(base).resolve()
    try:
        out_idx = root.parts.index("outputs")
    except ValueError:
        raise ValueError(f"{root} is not inside an 'outputs' tree")

    depth = len(root.parts) - out_idx - 1          # after “outputs/”

    if depth == 3:                                 # already at question depth
        yield root
        return

    for p in root.rglob("*"):
        if p.is_dir() and len(p.parts) - out_idx - 1 == 3:
            yield p