# utils/suffix_utils.py
from __future__ import annotations

import os, sys, pickle, torch
import numpy as np
from numbers import Number
from typing import Any, Optional, Sequence, Mapping, List
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

def _as_series(x: Any) -> List[float]:
    """
    Coerce many possible 'metric' shapes into a 1D list[float].

    Accepted:
      - None -> []
      - scalar -> [scalar]
      - list/tuple/np.ndarray -> flattened list
      - torch tensor -> via to_python
    Rejects:
      - dict -> [] (not meaningful to plot)
    """
    x = to_python(x)

    if x is None:
        return []

    # dict isn't meaningful as a series
    if isinstance(x, dict):
        return []

    # numpy arrays already converted to list by to_python, but keep safe
    if isinstance(x, np.ndarray):
        x = x.reshape(-1).tolist()

    # scalar
    if isinstance(x, (np.generic, Number, int, float)):
        try:
            return [float(x)]
        except Exception:
            return []

    # list/tuple/iterable
    if isinstance(x, (list, tuple)):
        out: List[float] = []
        for v in x:
            v = to_python(v)
            if v is None:
                continue
            if isinstance(v, dict):
                # skip nested dict entries
                continue
            try:
                out.append(float(v))
            except Exception:
                # if it's nested list like [[...]], flatten one level
                if isinstance(v, (list, tuple)) and len(v) == 1:
                    try:
                        out.append(float(v[0]))
                    except Exception:
                        pass
        return out

    # last resort: try iterating
    try:
        out = []
        for v in list(x):
            try:
                out.append(float(to_python(v)))
            except Exception:
                pass
        return out
    except Exception:
        return []

def plot_train_val(train_series: Optional[Sequence[float]], val_series: Optional[Sequence[float]], title: str, out_path: str, xlabel: str = "Epoch") -> None:
    """
    Safe plot helper: never raises. Accepts scalars, numpy scalars, lists, tensors, or None.

    - If either series is missing/empty after coercion, it returns without plotting.
    - If lengths differ, it plots the common prefix.
    """

    try:
        tr = _as_series(train_series)
        va = _as_series(val_series)

        if len(tr) == 0 or len(va) == 0:
            return

        n = min(len(tr), len(va))
        tr = tr[:n]
        va = va[:n]

        # ensure output dir exists
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        import matplotlib.pyplot as plt  # local import to avoid hard dependency at module import

        plt.figure()
        epochs = range(1, n + 1)
        plt.plot(epochs, tr, label=f"Train {title}")
        plt.plot(epochs, va, label=f"Val {title}")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        # plotting should never kill a run
        print(f"WARNING: plot_train_val failed for {title}: {e}", file=sys.stderr)

def persist_artefacts(base: str, script_dir: str, model: torch.nn.Module, preproc: Any, spec: Any, save_state_dict: bool = True, save_pickled_model: bool = True, save_pickled_preproc: bool = True, extra_files: Optional[Mapping[str, bytes]] = None) -> None:
    """
    Persist the standard artefacts:
      {base}_state.pt
      {base}_model.pkl
      {base}_preproc.pkl
      {base}_loaderspec.json
    """

    os.makedirs(script_dir, exist_ok=True)

    # Always try to save state dict (most robust)
    if save_state_dict:
        try:
            torch.save(model.state_dict(), os.path.join(script_dir, f"{base}_state.pt"))
        except Exception as e:
            print(f"WARNING: failed to save state_dict: {e}", file=sys.stderr)

    # Pickled model may fail for many valid models; don't crash the run.
    if save_pickled_model:
        try:
            with open(os.path.join(script_dir, f"{base}_model.pkl"), "wb") as f:
                pickle.dump(model, f)
        except Exception as e:
            print(f"WARNING: failed to pickle model: {e}", file=sys.stderr)

    if save_pickled_preproc:
        try:
            with open(os.path.join(script_dir, f"{base}_preproc.pkl"), "wb") as f:
                pickle.dump(preproc, f)
        except Exception as e:
            print(f"WARNING: failed to pickle preproc: {e}", file=sys.stderr)

    # loaderspec is small and important; but still don't crash if something weird happens
    try:
        write_loaderspec(base, spec, script_dir)
    except Exception as e:
        print(f"WARNING: failed to write loaderspec: {e}", file=sys.stderr)

    if extra_files:
        for rel_name, blob in extra_files.items():
            try:
                out = os.path.join(script_dir, rel_name)
                os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
                with open(out, "wb") as f:
                    f.write(blob)
            except Exception as e:
                print(f"WARNING: failed to write extra file {rel_name}: {e}", file=sys.stderr)

# --- Backwards-compatibility re-exports (TRACKFORMERS) ------------------------

def build_trackformers_model(*args, **kwargs):
    from challenges.TRACKFORMERS.utils_trackformers import build_trackformers_model as _impl
    return _impl(*args, **kwargs)
