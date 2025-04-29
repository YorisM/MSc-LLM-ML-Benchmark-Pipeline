# utils.py

# Imports
import os, sys, pathlib, platform, subprocess

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