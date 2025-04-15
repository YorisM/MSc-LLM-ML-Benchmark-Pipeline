# utils.py


# region - - - - - Imports - - - - - 
import os
import sys
# endregion


def rename_file_with_suffix(old_file, suffix):
    base, ext = os.path.splitext(old_file)
    new_file = f"{base}{suffix}{ext}"
    os.rename(old_file, new_file)
    return new_file