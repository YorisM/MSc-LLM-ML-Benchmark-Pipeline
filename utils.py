# utils.py

# Imports
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

def append_to_response_json(json_path: str, section: str, payload: dict) -> None:
    """
    Append (or overwrite) a top-level section inside the *.json file that lives
    next to each script.  Automatically time-stamps the entry.

        >>> append_to_response_json("…/response_gpt4o_1745_1.json",
                                    "DryRun",
                                    {"success": True, "runtime_s": 1.32})
    """
    with open(json_path, encoding="utf-8") as fh:
        blob = json.load(fh)

    # include an ISO-timestamp so you can see *when* the section was produced
    blob[section] = {
        "__timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds")+'Z',
        **payload
    }

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, ensure_ascii=False, indent=2)
    logging.debug("Appended %s section to %s", section, json_path)