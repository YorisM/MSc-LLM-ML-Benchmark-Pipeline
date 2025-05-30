# utils.py

import os, pathlib, platform, json, datetime, logging

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
    blob = {}

    if p.exists():
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
    p.write_text(json.dumps(blob, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    logging.debug("Updated %s -> %s", p.name, section)