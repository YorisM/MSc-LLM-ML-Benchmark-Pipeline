# utils.llm_io.py

from __future__ import annotations
import os, sys, torch, importlib.util, pickle, importlib, inspect
import numpy as np
from torch.utils.data import Dataset
from utils.loaderspec import LoaderSpec
from typing import Any

def _collate_ragged(batch):
    if len(batch) == 0:
        return [], []
    first = batch[0]
    # expected sample shape: (X_i, y_i)
    if isinstance(first, (tuple, list)) and len(first) == 2:
        xs = [xy[0] for xy in batch]
        ys = [xy[1] for xy in batch]
        return xs, ys
    # fallback (shouldn't happen for your default EventDataset)
    return batch

def _collate_identity(batch):
    return batch


BUILTIN_COLLATES = {
    "ragged_xy": _collate_ragged,
    "identity": _collate_identity,
}


def _apply_preproc(preproc, x: torch.Tensor) -> torch.Tensor:
    if callable(preproc):
        return preproc(x)
    if hasattr(preproc, "transform"):
        return preproc.transform(x)
    raise TypeError("Pre-processor is neither callable nor has .transform()")

def _mount_llm_script(model_dir: str) -> None:
    """
    Import the LLM-generated script that lives next to the artefacts and
    register it *also* as sys.modules['__main__'] so that
    __main__.MyPreprocessor can be resolved during unpickling.
    Safe to call more than once per process.
    """
    
    # find the script_<model>_*.py file
    script_path = next(
        f for f in os.listdir(model_dir)
        if f.startswith("script_") and f.endswith(".py")
    )
    script_path = os.path.join(model_dir, script_path)

    # If we already loaded *this* script, nothing to do
    if "__main__" in sys.modules and getattr(sys.modules["__main__"], "__file__", None) == script_path:
        return

    spec = importlib.util.spec_from_file_location("llm_script", script_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)           # type: ignore[attr-defined]

    sys.modules["llm_script"] = mod        # real name
    sys.modules["__main__"]   = mod        # alias used inside pickle

def _initialize_artefacts(model_path: str):
    model_dir = os.path.dirname(model_path)

    # Make sure MyPreprocessor lives in sys.modules['__main__']
    _mount_llm_script(model_dir)

    # Now unpickle safely
    with open(model_path.replace("_model.pkl", "_preproc.pkl"), "rb") as f:
        preproc = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model, preproc

def assert_batch_xy(batch):
    if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
        raise TypeError(f"Batch must be (xs, ys). Got {type(batch).__name__} with len={getattr(batch,'__len__',None)}")

    xs, ys = batch
    if not (isinstance(xs, list) and isinstance(ys, list)):
        raise TypeError(f"(xs, ys) must be (list, list). Got xs={type(xs).__name__}, ys={type(ys).__name__}")

    if len(xs) != len(ys):
        raise ValueError(f"xs and ys length mismatch: {len(xs)} vs {len(ys)}")

    if len(xs) == 0:
        raise ValueError("Empty batch.")

    for i, (x, y) in enumerate(zip(xs, ys)):
        if not torch.is_tensor(x) or not torch.is_tensor(y):
            raise TypeError(f"xs[i], ys[i] must be tensors. At i={i}: {type(x).__name__}, {type(y).__name__}")
        if x.ndim != 2:
            raise ValueError(f"xs[i] must be [N_i,F]. At i={i}: shape={tuple(x.shape)}")
        if y.ndim != 1:
            raise ValueError(f"ys[i] must be [N_i]. At i={i}: shape={tuple(y.shape)}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"N mismatch at i={i}: x has {x.shape[0]}, y has {y.shape[0]}")

def _to_device(obj: Any, device: torch.device) -> Any:
    """
    Recursively move tensors (and PyG Data/Batch objects) to device.
    """
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.to(device)
    # PyG Data/Batch objects implement .to(device)
    if hasattr(obj, "to") and callable(getattr(obj, "to")) and not isinstance(obj, (str, bytes)):
        try:
            return obj.to(device)
        except Exception:
            # fall through to recursive handling
            pass
    if isinstance(obj, tuple):
        return tuple(_to_device(x, device) for x in obj)
    if isinstance(obj, list):
        return [_to_device(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj

def resolve_path(path: str):
    # "module:symbol"
    mod_name, sym = path.split(":", 1)
    mod = importlib.import_module(mod_name)
    obj = mod
    for part in sym.split("."):
        obj = getattr(obj, part)
    return obj

def build_dataset(spec: LoaderSpec, events, preproc, train: bool):
    builder = resolve_path(spec.dataset.builder)
    kwargs = dict(spec.dataset.kwargs)

    if inspect.isclass(builder):
        return builder(events, preproc, train=train, **kwargs)
    if callable(builder):
        return builder(events, preproc, train=train, **kwargs)

    raise TypeError(f"Dataset builder {spec.dataset.builder!r} is not callable/class.")

def build_dataloader(spec: LoaderSpec, dataset, *, is_eval: bool = True):
    LoaderCls = resolve_path(spec.loader.class_path)

    cfg = {
        "batch_size": spec.loader.batch_size,
        "shuffle": spec.loader.shuffle,
        "num_workers": spec.loader.num_workers,
        "pin_memory": spec.loader.pin_memory,
        **(spec.loader.extra_kwargs or {}),
    }

    if is_eval and spec.eval_overrides:
        cfg.update(spec.eval_overrides)

    is_pyg = "torch_geometric" in spec.loader.class_path

    # ---- PyG loader ----
    if is_pyg:
        # For PyG, collate must be None
        if spec.loader.collate is not None:
            raise ValueError("PyG DataLoader selected but collate is not None. Set collate=None in cfg/spec.")

        # Optional: force eval to batch_size=1 (if that's your rule)
        if is_eval and not (spec.eval_overrides and "batch_size" in spec.eval_overrides):
            cfg["batch_size"] = 1

        return LoaderCls(dataset, **cfg)

    # ---- torch DataLoader ----
    if spec.loader.collate is None:
        return LoaderCls(dataset, **cfg)

    collate_fn = BUILTIN_COLLATES[spec.loader.collate.builtin]
    return LoaderCls(dataset, collate_fn=collate_fn, **cfg)

def split_X_y(evt):
    # When modifying make sure to update split_X_y in prefix.trackformers.py 
    X = np.column_stack([
        evt["hit_r"].astype(np.float32),
        evt["hit_theta"].astype(np.float32),
        evt["hit_z"].astype(np.float32),
        evt["layer_id"].astype(np.float32)
    ])
    y = evt["track_id"].astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)

class EventDataset(Dataset):
    # When modifying make sure to update EventDataset in prefix.trackformers.py
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, labels = split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        return (X, labels)
    

# --- Backwards-compatibility re-exports (FOURTOPS) ----------------------------

def assert_binary_output(*args, **kwargs):
    from challenges.FOURTOPS.utils_fourtops import assert_binary_output as _impl
    return _impl(*args, **kwargs)

def detect_and_assert_lane_fourtops(*args, **kwargs):
    from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops as _impl
    return _impl(*args, **kwargs)

def make_view_by_lane_fourtops(*args, **kwargs):
    from challenges.FOURTOPS.utils_fourtops import make_view_by_lane_fourtops as _impl
    return _impl(*args, **kwargs)

# --- Backwards-compatibility re-exports (TRACKFORMERS) ------------------------

def detect_and_assert_lane(*args, **kwargs):
    from challenges.TRACKFORMERS.utils_trackformers import detect_and_assert_lane as _impl
    return _impl(*args, **kwargs)

def assert_label_output_by_lane(*args, **kwargs):
    from challenges.TRACKFORMERS.utils_trackformers import assert_label_output_by_lane as _impl
    return _impl(*args, **kwargs)
