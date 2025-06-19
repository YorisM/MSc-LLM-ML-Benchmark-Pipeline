# utils.py

import os, pathlib, platform, json, datetime, logging
from typing import Union, TextIO, Optional
from sys import stdout

def print_repo_tree(
    root: Union[str, pathlib.Path] = ".",
    max_depth: int = 2,
    show_files: bool = True,
    ignore_hidden: bool = True,
    out: Optional[TextIO] = None,
) -> None:
    """
    Pretty–prints the folder / file hierarchy of a repository.

    Parameters
    ----------
    root : str | Path, default "."
        Directory to start from.  Relative paths are resolved against CWD.
    max_depth : int, default 2
        How many directory levels to descend (0 = just `root`).
    show_files : bool, default True
        If False, only directories are listed.
    ignore_hidden : bool, default True
        Skip entries whose names start with '.'.
    out : TextIO | None, default None
        Stream to write to (e.g. open file handle).  `None` → `sys.stdout`.
    """
    
    root = pathlib.Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    out_stream = out or stdout
    spacer = "    "

    def _is_hidden(p: pathlib.Path) -> bool:
        return p.name.startswith(".")

    def _recurse(dir_path: pathlib.Path, depth: int) -> None:
        if depth > max_depth:
            return

        entries = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        for entry in entries:
            if ignore_hidden and _is_hidden(entry):
                continue

            prefix = spacer * depth + ("└── " if depth else "")
            print(f"{prefix}{entry.name}{'/' if entry.is_dir() else ''}", file=out_stream)

            if entry.is_dir():
                _recurse(entry, depth + 1)
            elif not show_files:
                continue

    print(f"{root.name}/", file=out_stream)
    _recurse(root, 1)

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