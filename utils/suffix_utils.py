# utils/suffix_utils.py
from __future__ import annotations

import os, sys, pickle, torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Optional, Sequence, Mapping
from utils.loaderspec import write_loaderspec

def base_from_argv0(script_prefix: str = "script_") -> str:
    """
    Derive the artefact base name from the invoked script filename.
    Example:  sys.argv[0] = ".../script_mymodel.py"  ->  "mymodel"
    """
    name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    if name.startswith(script_prefix):
        name = name[len(script_prefix):]
    return name

def to_python(obj: Any) -> Any:
    """
    Recursively convert common ML types into JSON-safe / plotting-safe Python types:
      - torch.Tensor -> float/int or list
      - numpy scalars/arrays -> float/int or list
      - dict/list/tuple -> recursively converted
    """

    # torch
    if torch.is_tensor(obj):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()

    # numpy
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # containers
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(x) for x in obj]

    return obj

def plot_train_val(train_series: Optional[Sequence[float]], val_series: Optional[Sequence[float]], title: str, out_path: str, xlabel: str = "Epoch") -> None:
    """
    Safe plot helper: only plots if both series are provided and non-empty.
    """

    if train_series is None or val_series is None:
        return
    if len(train_series) == 0 or len(val_series) == 0:
        return
    
    train_series = to_python(train_series)
    val_series   = to_python(val_series)

    plt.figure()
    epochs = range(1, len(train_series) + 1)
    plt.plot(epochs, train_series, label=f"Train {title}")
    plt.plot(epochs, val_series, label=f"Val {title}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(out_path)
    plt.close()

def persist_artefacts(base: str, script_dir: str, model: torch.nn.Module, preproc: Any, spec: Any, save_state_dict: bool = True, save_pickled_model: bool = True, save_pickled_preproc: bool = True, extra_files: Optional[Mapping[str, bytes]] = None) -> None:
    """
    Persist the standard artefacts:
      {base}_state.pt
      {base}_model.pkl
      {base}_preproc.pkl
      {base}_loaderspec.json
    """

    os.makedirs(script_dir, exist_ok=True)

    if save_state_dict:
        torch.save(model.state_dict(), os.path.join(script_dir, f"{base}_state.pt"))

    if save_pickled_model:
        with open(os.path.join(script_dir, f"{base}_model.pkl"), "wb") as f:
            pickle.dump(model, f)

    if save_pickled_preproc:
        with open(os.path.join(script_dir, f"{base}_preproc.pkl"), "wb") as f:
            pickle.dump(preproc, f)

    write_loaderspec(base, spec, script_dir)

    if extra_files:
        for rel_name, blob in extra_files.items():
            out = os.path.join(script_dir, rel_name)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "wb") as f:
                f.write(blob)
