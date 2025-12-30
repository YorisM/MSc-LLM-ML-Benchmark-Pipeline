# utils.llm_io.py

from __future__ import annotations
import os, sys, torch, importlib.util, pickle, importlib, inspect
import numpy as np
from torch.utils.data import Dataset
from utils.loaderspec import LoaderSpec
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass


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

def assert_binary_output(view, out: Any, *, tol: float = 1e-6, accept_logits: bool = True, accept_probs: bool = True, require_finite: bool = True,) -> Tuple[torch.Tensor, str]:
    """
    Enforce: model returns one float score per sample for the FOURTOPS challenge.

    Accepted:
      - Tensor(B,) or Tensor(B,1)
      - (optionally) probabilities in [0,1] OR logits in (-inf, +inf)

    Returns:
      (scores, kind) where kind is "probs" or "logits".
    """

    # Unwrap common "aux outputs" patterns: (scores, aux) / [scores, ...]
    if isinstance(out, (tuple, list)):
        if len(out) == 0:
            raise TypeError("Model output is an empty tuple/list; expected Tensor(B,) or Tensor(B,1).")
        out0 = out[0]
        if not torch.is_tensor(out0):
            raise TypeError(f"Model output[0] must be a Tensor, got {type(out0)}")
        out = out0

    if not torch.is_tensor(out):
        raise TypeError(f"Model output must be a Tensor, got {type(out)}")

    if not torch.is_floating_point(out):
        raise TypeError(f"Expected float outputs (logits or probabilities), got dtype={out.dtype}")

    # Shape: allow (B,) or (B,1)
    scores = out.squeeze(-1)
    if scores.ndim != 1:
        raise ValueError(f"Expected output with 1 dimension after squeeze, got shape {tuple(scores.shape)}")

    # Determine batch size B from labels if possible, else from batch_x
    B: Optional[int] = None
    y = getattr(view, "batch_y", None)
    x = getattr(view, "batch_x", None)

    if torch.is_tensor(y):
        B = int(y.shape[0])
    elif isinstance(y, list):
        B = len(y)
    elif torch.is_tensor(x):
        B = int(x.shape[0])
    elif isinstance(x, list):
        B = len(x)

    if B is None:
        raise ValueError("Could not infer batch size from view.batch_y or view.batch_x.")

    if scores.shape[0] != B:
        raise ValueError(f"Expected output length B={B}, got {scores.shape[0]}")

    if require_finite and not torch.isfinite(scores).all():
        raise ValueError("Model output contains NaN/Inf")

    # Classify as probs vs logits
    if accept_probs:
        in_range = ((scores >= -tol) & (scores <= 1.0 + tol)).all().item()
        if bool(in_range):
            return scores, "probs"

    if accept_logits:
        return scores, "logits"

    raise ValueError("Output is neither valid probabilities nor logits under current settings.")

def _is_int_tensor(t: torch.Tensor) -> bool:
    return torch.is_tensor(t) and t.dtype in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.long
    )

def assert_label_output(batch_x, out, *, allow_noise_label=True):
    """
    Enforce: model returns integer labels per hit for the TRACKFORMERS challenge.

    Accepted outputs:
      - list[Tensor(Ni)]          aligned with ragged batch_x=list[Xi]
      - Tensor(sumNi,)            aligned with concatenation of ragged batch_x
      - (optionally) Tensor(B,N)  if batch_x is padded Tensor(B,N,F) (rare in your new design)

    Also checks that per-event lengths match.
    """
    if torch.is_tensor(out):
        if not _is_int_tensor(out):
            raise TypeError(
                f"Model output must be integer labels (torch.long preferred). "
                f"Got tensor dtype={out.dtype}, shape={tuple(out.shape)}"
            )
    
        # Shape sanity: must be (N,) or (N,1) for single-tensor outputs
        if out.dim() == 2 and out.size(1) == 1:
            out = out.squeeze(1)
        elif out.dim() != 1:
            raise ValueError(
                f"Model output tensor must be 1-D labels (N,) (or (N,1)). Got shape={tuple(out.shape)}"
            )

    elif isinstance(out, list):
        if len(out) == 0:
            raise TypeError("Model output list is empty; expected list of label tensors per event.")
        if not all(torch.is_tensor(t) for t in out):
            bad = [type(t).__name__ for t in out if not torch.is_tensor(t)]
            raise TypeError(f"Model output list must contain only tensors. Bad elements: {bad}")
        if not all(_is_int_tensor(t) for t in out):
            dtypes = [t.dtype for t in out]
            raise TypeError(f"Model output labels must be integer tensors. Got dtypes={dtypes}")
    else:
        raise TypeError(
            f"Model output must be a Tensor or list[Tensor] of integer labels. "
            f"Got type={type(out).__name__}"
        )

    # ---- length checks ----
    # Case A: ragged batch_x is list[Tensor(Ni,F)] or list[Data] (PyG)
    if isinstance(batch_x, list):
        # infer per-event lengths
        lens = []
        for item in batch_x:
            if torch.is_tensor(item):
                if item.dim() == 1:
                    lens.append(int(item.shape[0]))
                else:
                    lens.append(int(item.shape[0]))  # [Ni,F] -> Ni
            elif hasattr(item, "num_nodes"):
                lens.append(int(item.num_nodes))
            elif hasattr(item, "x") and torch.is_tensor(item.x):
                lens.append(int(item.x.shape[0]))
            else:
                raise TypeError(
                    "Cannot infer number of hits for an item in batch_x list. "
                    f"Got element type={type(item).__name__}"
                )

        total = int(sum(lens))

        if isinstance(out, list):
            if len(out) != len(lens):
                raise ValueError(
                    f"Model returned {len(out)} label tensors, but batch has {len(lens)} events."
                )
            for i, (li, yi) in enumerate(zip(lens, out)):
                if yi.dim() != 1:
                    raise ValueError(
                        f"Output labels for event {i} must be 1-D (Ni,). Got shape={tuple(yi.shape)}"
                    )
                if int(yi.shape[0]) != li:
                    raise ValueError(
                        f"Length mismatch in event {i}: expected {li} labels (Ni), got {int(yi.shape[0])}."
                    )
        else:
            # single concatenated tensor
            if out.dim() != 1:
                raise ValueError(f"Concatenated label output must be 1-D (sumNi,). Got shape={tuple(out.shape)}")
            if int(out.shape[0]) != total:
                raise ValueError(
                    f"Concatenated label length mismatch: expected sumNi={total}, got {int(out.shape[0])}."
                )

        if allow_noise_label and torch.is_tensor(out):
            # optional sanity: allow -1 as noise, but labels should be >= -1
            if out.numel() > 0 and out.min().item() < -1:
                raise ValueError("Label tensor contains values < -1. Use -1 for noise/unassigned.")

    # Case B: padded batch_x as Tensor(B,N,F) with a mask elsewhere (not your default now)
    # (You can extend here if you ever reintroduce padded collate.)

@dataclass
class BatchView:
    """
    Normalised view of an arbitrary DataLoader batch.

    batch_x:
      - list[Tensor]   (ragged: one Tensor per event)
      - Tensor         (padded/batched: e.g. [B, N, F] or [B, F])
      - tuple/list     (multi-input models: e.g. (x, edge_index, ...))
      - dict           (rare; only if you explicitly allow)
    batch_y:
      - Tensor or list[Tensor] or None
    """

    batch_x: Any
    batch_y: Any = None
    meta: Dict[str, Any] = None

    def __eq__(self, other: object) -> bool:
        # Allow test ergonomics: compare a BatchView to a mode string.
        if isinstance(other, str):
            return (self.meta or {}).get("mode") == other

        # Avoid dataclass/tensor equality pitfalls; default to identity equality for non-string comparisons.
        return object.__eq__(self, other)

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

def _is_pyg_batchlike(obj: Any) -> bool:
    """
    Duck-type check for torch_geometric.data.Data / Batch.
    Avoids importing torch_geometric on module import.
    """
    if obj is None or torch.is_tensor(obj):
        return False
    # Must have .to() and at least one typical PyG attribute
    if not (hasattr(obj, "to") and callable(getattr(obj, "to"))):
        return False
    return any(hasattr(obj, a) for a in ("x", "edge_index", "batch", "pos"))

def _to_tensor_labels(y: Any) -> Any:
    """
    Convert common label containers to a 1D torch tensor when possible.
    """
    if y is None:
        return None
    if torch.is_tensor(y):
        # allow (B,1) -> (B,)
        return y.view(-1) if (y.ndim == 2 and y.shape[1] == 1) else y
    if isinstance(y, (list, tuple)):
        if len(y) == 0:
            return torch.empty((0,), dtype=torch.long)
        if all(torch.is_tensor(t) for t in y):
            ys = [t.view(1) if t.ndim == 0 else t.view(-1) for t in y]
            return torch.cat(ys, dim=0)
        # list of ints/bools
        return torch.tensor(y)
    return y  # leave as-is (some exotic cases)

def _pyg_batch_from_list(data_list: list[Any]) -> Any | None:
    """
    If torch_geometric is available, batch a list[Data] into a Batch.
    """
    try:
        from torch_geometric.data import Batch  # type: ignore
        return Batch.from_data_list(data_list)
    except Exception:
        return None

def normalise_batch(batch: Any, device: torch.device | None = None) -> BatchView:
    """
    Convert arbitrary DataLoader batches into a BatchView(batch_x, batch_y).

    Key goals:
      1) Accept many batch "shapes" (tensor / tuple / list / dict / PyG).
      2) Move *all* contained tensors to `device` if provided (recursively).
      3) Avoid common LLM mistakes: manual unpacking, calling .to(device) on raw batch, etc.
      4) NEW GAP FIX:
         Handle "collated list of fields" where the first element is NOT a Tensor,
         e.g. [dict_of_batched_tensors, y_batched] or [(x1_batched, x2_batched), y_batched].

    Supported high-level inputs:
      - A0) list of tensors fields: [X_batch, y_batch] or [x1_batch,...,y_batch]
      - A)  list batch: ragged samples [(Xi,yi), ...], [Xi, ...], PyG lists, etc.
      - B)  tuple: (X, y) or (X1, X2, ..., y)
      - C)  tensor: X
      - D)  mapping: {"x": X, "y": y} (and variants)
      - E0) object with attributes: obj.x / obj.y etc (non-PyG)
      - E)  PyG Data/Batch: has .to() and (.x or .pos)
    """
    
    meta: Dict[str, Any] = {"source_type": type(batch).__name__}
    dev = device

    def _to_dev(x: Any) -> Any:
        return _to_device(x, dev) if dev is not None else x

    def _stack_fixed(seq: Any) -> Tuple[Any, bool]:
        """
        Try to stack a list/tuple of fixed-size items into a single tensor.
        Returns (stacked_or_original, did_stack).
        FOURTOPS convenience: if samples are fixed-size, avoid accidental ragged handling.
        """
        if not isinstance(seq, (list, tuple)) or len(seq) == 0:
            return seq, False

        # list[Tensor] with identical shapes -> stack
        if all(torch.is_tensor(t) for t in seq):
            shapes = [tuple(t.shape) for t in seq]
            if all(s == shapes[0] for s in shapes):
                try:
                    return torch.stack(list(seq), dim=0), True
                except Exception:
                    return seq, False

        # list of Python scalars -> tensor
        if all(isinstance(v, (int, float, bool, np.number)) for v in seq):
            return torch.tensor(list(seq)), True

        return seq, False

    def _is_pyg_obj(obj: Any) -> bool:
        # Duck-typed PyG Data/Batch: has .to() and (.x or .pos)
        return (
            hasattr(obj, "to")
            and callable(getattr(obj, "to"))
            and (hasattr(obj, "x") or hasattr(obj, "pos"))
        )

    def _infer_leading_B(obj: Any) -> int | None:
        """
        Best-effort: infer batch-size B from an object by finding a Tensor with ndim>=1
        and returning its shape[0]. Works for:
          - Tensor: shape[0]
          - Mapping/tuple/list: first tensor found within
          - PyG Batch: try num_graphs / batch vector / y
        Returns None if cannot infer.
        """
        if obj is None:
            return None
        if torch.is_tensor(obj):
            return int(obj.shape[0]) if obj.ndim >= 1 else None
        if _is_pyg_obj(obj):
            # PyG Batch often has num_graphs; Data doesn't necessarily.
            if hasattr(obj, "num_graphs"):
                try:
                    return int(obj.num_graphs)
                except Exception:
                    pass
            bvec = getattr(obj, "batch", None)
            if torch.is_tensor(bvec) and bvec.numel() > 0:
                try:
                    return int(bvec.max().item()) + 1
                except Exception:
                    pass
            y = getattr(obj, "y", None)
            if torch.is_tensor(y) and y.ndim >= 1:
                return int(y.shape[0])
            return None
        if isinstance(obj, Mapping):
            for v in obj.values():
                b = _infer_leading_B(v)
                if b is not None:
                    return b
            return None
        if isinstance(obj, (tuple, list)):
            for v in obj:
                b = _infer_leading_B(v)
                if b is not None:
                    return b
            return None
        if hasattr(obj, "__dict__"):
            # look for common input attrs
            for attr in ("x", "X", "inputs", "input", "batch_x"):
                if hasattr(obj, attr):
                    return _infer_leading_B(getattr(obj, attr))
        return None

    def _labelish_for_B(y: Any, B: int) -> bool:
        """
        Decide whether `y` looks like labels for batch-size B.
        Important fence: refuse "sample-like dicts" as labels.
        """
        if y is None:
            return False
        if torch.is_tensor(y):
            if y.ndim == 0:
                return True
            if y.ndim == 1 and int(y.shape[0]) == B:
                return True
            if y.ndim == 2 and int(y.shape[0]) == B and int(y.shape[1]) == 1:
                return True
            return False
        if isinstance(y, (list, tuple)):
            # Only accept per-sample labels if:
            #   - length matches batch size AND
            #   - elements are scalar-like (python scalars or scalar tensors)
            if len(y) != B:
                return False
            if all(isinstance(v, (int, float, bool, np.number)) for v in y):
                return True
            if all(torch.is_tensor(v) and (v.ndim == 0 or (v.ndim == 1 and v.numel() == 1)) for v in y):
                return True
            # Anything else (e.g., tensors shaped [Ni, F]) is *not* labels
            return False
        if isinstance(y, Mapping):
            # If it has "x"/"inputs"/etc, it's likely a *sample*, not labels.
            if any(k in y for k in ("x", "X", "inputs", "input", "batch_x")):
                return False
            # If it has only label-ish keys, allow.
            if any(k in y for k in ("y", "Y", "labels", "label", "batch_y")):
                return True
            return False
        # python scalar labels are acceptable
        if isinstance(y, (int, float, bool, np.number)):
            return True
        return False

    def _normalize_y(y: Any) -> Any:
        """
        Gentle normalization of y:
          - stack python scalar lists
          - stack list[Tensor] only if identical shapes (FOURTOPS-friendly)
          - flatten (B,1) to (B,)
        Avoid aggressive concatenation that could destroy ragged boundaries.
        """
        if isinstance(y, (list, tuple)):
            y2, did = _stack_fixed(y)
            y = y2
        if torch.is_tensor(y) and y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        return y

    # -------------------------
    # A0) Torch default_collate often returns a *list* of fields:
    #     [X_batch, y_batch] or [x1_batch, x2_batch, ..., y_batch]
    #     This path only triggers when ALL elements are tensors.
    # -------------------------
    if isinstance(batch, list) and 2 <= len(batch) <= 16 and all(torch.is_tensor(x) for x in batch):

        seq = list(batch)  # normalize container type

        # Helper: all "batched" tensors share the same leading dimension B
        def _shares_leading_dim(ts):
            if any(t.ndim == 0 for t in ts):
                return False
            B = int(ts[0].shape[0])
            return all(t.ndim >= 1 and int(t.shape[0]) == B for t in ts)

        def _looks_like_labels(y, B: int) -> bool:
            if not torch.is_tensor(y):
                return False
            if y.ndim == 0:
                return True
            if y.ndim == 1 and int(y.shape[0]) == B:
                return True
            if y.ndim == 2 and int(y.shape[0]) == B and int(y.shape[1]) == 1:
                return True
            return False

        if _shares_leading_dim(seq):
            B = int(seq[0].shape[0])
            y_last = seq[-1]

            # If last field is batch-shaped, treat it as labels.
            if _looks_like_labels(y_last, B):
                xs = seq[0] if len(seq) == 2 else seq[:-1]   # <-- list for multi-input
                ys = y_last
                if ys.ndim == 2 and ys.shape[1] == 1:
                    ys = ys.reshape(-1)
                return BatchView(
                    batch_x=_to_dev(xs),
                    batch_y=_to_dev(ys),
                    meta={**meta, "mode": "collated_list_xy", "arity": len(seq)},
                )

            # Otherwise: multi-input X only (still keep as list, not tuple)
            return BatchView(
                batch_x=_to_dev(seq),
                batch_y=None,
                meta={**meta, "mode": "collated_list_x", "arity": len(seq)},
            )

    # -------------------------
    # A) list batch (ragged, PyG list shapes, or "singleton tensor in list")
    # -------------------------
    if isinstance(batch, list):
        if len(batch) == 0:
            return BatchView(batch_x=batch, batch_y=None, meta={**meta, "mode": "empty_list"})

        first = batch[0]

        # --- PyG list shape #1: [Batch/Data, y] ---
        if len(batch) == 2 and _is_pyg_obj(first) and not _is_pyg_obj(batch[1]):
            b = first.to(device) if device is not None else first
            y = _normalize_y(batch[1])
            return BatchView(batch_x=b, batch_y=_to_dev(y), meta={**meta, "mode": "pyg_collated_list_xy"})

        # --- PyG list shape #2: [(Data, y), ...] ---
        if isinstance(first, (tuple, list)) and len(first) == 2 and _is_pyg_obj(first[0]):
            data_list = [xy[0] for xy in batch]
            y_list = [xy[1] for xy in batch]
            try:
                from torch_geometric.data import Batch as PyGBatch
                b = PyGBatch.from_data_list(data_list)
                b = b.to(device) if device is not None else b
                y_stacked, _ = _stack_fixed(y_list)
                y_stacked = _normalize_y(y_stacked)
                return BatchView(
                    batch_x=b,
                    batch_y=_to_dev(y_stacked),
                    meta={**meta, "mode": "pyg_ragged_xy_list_batched"},
                )
            except Exception:
                return BatchView(
                    batch_x=_to_dev(data_list),
                    batch_y=_to_dev(y_list),
                    meta={**meta, "mode": "pyg_ragged_xy_list"},
                )

        # --- PyG list shape #3: [Data, Data, ...] ---
        if _is_pyg_obj(first) and all(_is_pyg_obj(x) for x in batch):
            try:
                from torch_geometric.data import Batch as PyGBatch
                b = PyGBatch.from_data_list(batch)
                b = b.to(device) if device is not None else b
                y = _normalize_y(getattr(b, "y", None))
                return BatchView(
                    batch_x=b,
                    batch_y=_to_dev(y),
                    meta={**meta, "mode": "pyg_data_list_batched"},
                )
            except Exception:
                return BatchView(batch_x=_to_dev(batch), batch_y=None, meta={**meta, "mode": "pyg_data_list"})

        # --- NEW GAP FIX: structured collated list of fields [X_batch_struct, y_batch] ---
        # Example: batch == [ {"g": Tensor(B,2), "obj": Tensor(B,18,5)}, Tensor(B,) ]
        # This is *not* covered by A0 because elements aren't all tensors.
        if 2 <= len(batch) <= 16:
            # Only attempt to interpret "fields list" if it does NOT look like a list-of-samples.
            # (List-of-samples is typically [(Xi, yi), ...] which is handled below.)
            if not (isinstance(first, (tuple, list)) and len(first) == 2):
                y_last = batch[-1]
                x_fields = batch[:-1]

                # Infer B from x_fields (preferred), else from y_last.
                B = None
                for xf in x_fields:
                    b = _infer_leading_B(xf)
                    if b is not None:
                        B = b
                        break
                if B is None:
                    B = _infer_leading_B(y_last)

                if B is not None and B > 0 and _labelish_for_B(y_last, B):
                    xs = x_fields[0] if len(x_fields) == 1 else tuple(x_fields)
                    ys = _normalize_y(y_last)
                    return BatchView(
                        batch_x=_to_dev(xs),
                        batch_y=_to_dev(ys),
                        meta={**meta, "mode": "collated_list_xy", "arity": len(batch), "structured": True},
                    )

        # [(X,y), ...]
        if isinstance(first, (tuple, list)) and len(first) == 2:
            xs = [xy[0] for xy in batch]
            ys = [xy[1] for xy in batch]
            return BatchView(batch_x=_to_dev(xs), batch_y=_to_dev(ys), meta={**meta, "mode": "ragged_xy_list"})
        
        # [(X1, X2, ..., y), ...]
        if isinstance(first, (tuple, list)) and len(first) > 2:
            xs = [tuple(xy[:-1]) for xy in batch]
            ys = [xy[-1] for xy in batch]
            return BatchView(
                batch_x=_to_dev(xs),
                batch_y=_to_dev(ys),
                meta={**meta, "mode": "ragged_multi_xy_list", "arity": len(first)},
            )

        # [X, ...] where X are tensors
        if all(torch.is_tensor(x) for x in batch):
            if len(batch) == 1:
                return BatchView(batch_x=_to_dev(batch[0]), batch_y=None, meta={**meta, "mode": "tensor_in_list"})
            return BatchView(batch_x=_to_dev(list(batch)), batch_y=None, meta={**meta, "mode": "ragged_x_list"})

        # fallback: keep as-is
        return BatchView(batch_x=_to_dev(batch), batch_y=None, meta={**meta, "mode": "list"})

    # -------------------------
    # B) (X, y) tuple only
    # -------------------------
    if isinstance(batch, tuple) and len(batch) == 2:
        x, y = batch
        x_stacked, did_x = _stack_fixed(x)
        y_stacked, did_y = _stack_fixed(y)
        if did_x or did_y:
            y_stacked = _normalize_y(y_stacked)
            return BatchView(batch_x=_to_dev(x_stacked), batch_y=_to_dev(y_stacked), meta={**meta, "mode": "stacked_xy_pair"})
        return BatchView(batch_x=_to_dev(x), batch_y=_to_dev(_normalize_y(y)), meta={**meta, "mode": "xy_pair"})

    # -------------------------
    # B2) (X1, X2, ..., y) tuple
    # -------------------------
    if isinstance(batch, tuple) and len(batch) > 2:
        x = tuple(batch[:-1])
        y = _normalize_y(batch[-1])
        return BatchView(batch_x=_to_dev(x), batch_y=_to_dev(y), meta={**meta, "mode": "multi_xy_tuple", "arity": len(batch)})

    # -------------------------
    # C) plain tensor
    # -------------------------
    if torch.is_tensor(batch):
        return BatchView(batch_x=_to_dev(batch), batch_y=None, meta={**meta, "mode": "tensor"})

    # -------------------------
    # D) mapping batch
    # -------------------------
    if isinstance(batch, Mapping):
        x = None
        y = None
        for k in ("x", "X", "inputs", "input", "batch_x"):
            if k in batch:
                x = batch[k]
                break
        for k in ("y", "Y", "labels", "label", "batch_y"):
            if k in batch:
                y = batch[k]
                break
        if x is None:
            raise TypeError(f"Dict batch missing an input key. Keys={list(batch.keys())}")
        return BatchView(batch_x=_to_dev(x), batch_y=_to_dev(_normalize_y(y)), meta={**meta, "mode": "dict"})

    # -------------------------
    # E0) object batch with attributes (non-dict containers), non-PyG
    # -------------------------
    if hasattr(batch, "__dict__") and not _is_pyg_obj(batch):
        x = None
        y = None
        for attr in ("x", "X", "inputs", "input", "batch_x"):
            if hasattr(batch, attr):
                x = getattr(batch, attr)
                break
        for attr in ("y", "Y", "labels", "label", "batch_y"):
            if hasattr(batch, attr):
                y = getattr(batch, attr)
                break
        if x is not None:
            return BatchView(batch_x=_to_dev(x), batch_y=_to_dev(_normalize_y(y)), meta={**meta, "mode": "attr_object"})

    # -------------------------
    # E) PyG Data / Batch (duck-typed)
    # -------------------------
    if _is_pyg_obj(batch):
        b = batch.to(device) if device is not None else batch
        y = _normalize_y(getattr(b, "y", None))
        mode = "pyg_batch" if hasattr(b, "batch") else "pyg_data"
        return BatchView(batch_x=b, batch_y=_to_dev(y), meta={**meta, "mode": mode})

    raise TypeError(f"Unsupported batch type: {type(batch).__name__}")

def extract_batch_x(batch: Any, device: Optional[torch.device] = None) -> Any:
    """
    Convenience: returns only batch_x, optionally moved to device.
    """
    view = normalise_batch(batch)
    bx = view.batch_x
    if device is not None:
        bx = _to_device(bx, device)
    return bx

def extract_xy(batch: Any, device: Optional[torch.device] = None) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Returns (batch_x, batch_y, meta), optionally moved to device.
    """
    view = normalise_batch(batch)
    bx, by = view.batch_x, view.batch_y
    if device is not None:
        bx = _to_device(bx, device)
        by = _to_device(by, device)
    return bx, by, (view.meta or {})

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
        if is_eval:
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