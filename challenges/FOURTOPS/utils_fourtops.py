# ./challenges/FOURTOPS/utils_fourtops.py

import logging, torch, itertools
import torch.nn.functional as F 
import numpy as np
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass


_BOOL_INT_DTYPES = {torch.bool, torch.int8, torch.int16, torch.int32, torch.int64}


def _is_pyg_obj(obj) -> bool:
    # Duck-typed PyG Data/Batch
    return hasattr(obj, "to") and callable(getattr(obj, "to")) and (hasattr(obj, "x") or hasattr(obj, "pos"))

def _assert_torch_xy_batch_fourtops(batch):
    if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
        raise TypeError(f"FOURTOPS TORCH lane requires batch == (xb, yb). Got {type(batch).__name__}.")

    xb, yb = batch
    if not torch.is_tensor(xb) or not torch.is_tensor(yb):
        raise TypeError(f"FOURTOPS TORCH lane requires tensors. Got xb={type(xb).__name__}, yb={type(yb).__name__}")

    if xb.ndim < 2:
        raise ValueError(f"FOURTOPS TORCH lane: xb must be batched with ndim>=2 (B, ...). Got shape={tuple(xb.shape)}")

    # allow y: (B,) or (B,1)
    if yb.ndim == 2 and yb.shape[1] == 1:
        yb = yb.view(-1)

    if yb.ndim != 1:
        raise ValueError(f"FOURTOPS TORCH lane: yb must be (B,) or (B,1). Got shape={tuple(yb.shape)}")

    if yb.shape[0] != xb.shape[0]:
        raise ValueError(f"FOURTOPS TORCH lane: batch size mismatch: xb[0]={xb.shape[0]} vs yb[0]={yb.shape[0]}")

    if yb.dtype not in _BOOL_INT_DTYPES:
        raise TypeError(f"FOURTOPS TORCH lane: yb must be integer/bool dtype. Got {yb.dtype}")
    
    # Enforce binary labels {0,1}
    if yb.numel() == 0:
        raise ValueError("FOURTOPS TORCH lane: yb is empty.")
    ymin = int(yb.min().item())
    ymax = int(yb.max().item())
    if ymin < 0 or ymax > 1:
        raise ValueError(
            f"FOURTOPS TORCH lane: labels must be binary 0/1. "
            f"Got min={ymin}, max={ymax}."
        )

def _infer_num_graphs_pyg(G) -> int:
    if hasattr(G, "num_graphs"):
        try:
            return int(G.num_graphs)
        except Exception:
            pass
    bvec = getattr(G, "batch", None)
    if torch.is_tensor(bvec) and bvec.numel() > 0:
        return int(bvec.max().item()) + 1
    return 1

def _assert_pyg_graph_batch_fourtops(batch):
    if not _is_pyg_obj(batch):
        raise TypeError(f"FOURTOPS PyG lane requires a PyG Data/Batch object. Got {type(batch).__name__}")

    x = getattr(batch, "x", None)
    if not (torch.is_tensor(x) and x.ndim == 2):
        raise ValueError("FOURTOPS PyG lane requires batch.x to be rank-2 [num_nodes, F].")

    y = getattr(batch, "y", None)
    if y is None or not torch.is_tensor(y):
        raise ValueError("FOURTOPS PyG lane requires batch.y to be a Tensor of graph labels.")

    B = _infer_num_graphs_pyg(batch)

    # allow y: (B,) or (B,1)
    y_flat = y.view(-1) if (y.ndim == 2 and y.shape[1] == 1) else y
    if y_flat.ndim != 1 or int(y_flat.shape[0]) != B:
        # common LLM mistake: node-level y
        if y_flat.ndim == 1 and int(y_flat.shape[0]) == int(x.shape[0]):
            raise ValueError(
                "FOURTOPS PyG lane: batch.y looks like node-level labels (len == num_nodes). "
                "FOURTOPS requires graph-level labels: y shape == (num_graphs,)."
            )
        raise ValueError(f"FOURTOPS PyG lane: batch.y must be graph labels of shape (num_graphs,) or (num_graphs,1). Got {tuple(y.shape)}; num_graphs={B}")

    if y_flat.dtype not in _BOOL_INT_DTYPES:
        raise TypeError(f"FOURTOPS PyG lane: y must be integer/bool dtype. Got {y.dtype}")
    
    # Enforce binary graph labels {0,1}
    if y_flat.numel() == 0:
        raise ValueError("FOURTOPS PyG lane: y is empty.")
    ymin = int(y_flat.min().item())
    ymax = int(y_flat.max().item())
    if ymin < 0 or ymax > 1:
        raise ValueError(
            f"FOURTOPS PyG lane: labels must be binary 0/1. "
            f"Got min={ymin}, max={ymax}."
        )

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
    
def detect_and_assert_lane_fourtops(spec, first_batch):
    """
    Returns: "torch_xy" or "pyg_graph".
    Uses spec.loader.class_path to decide lane, then asserts batch matches FOURTOPS contract.
    """
    is_pyg = "torch_geometric" in spec.loader.class_path

    if is_pyg:
        _assert_pyg_graph_batch_fourtops(first_batch)
        mode = "pyg_graph"
        B = _infer_num_graphs_pyg(first_batch)
        logging.info("BATCH_LANE=%s | num_graphs=%d | x=%s y=%s", mode, B, tuple(first_batch.x.shape), tuple(first_batch.y.shape))
        return mode

    _assert_torch_xy_batch_fourtops(first_batch)
    xb, yb = first_batch
    logging.info("BATCH_LANE=%s | B=%d | xb=%s yb=%s", "torch_xy", int(xb.shape[0]), tuple(xb.shape), tuple(yb.shape))
    return "torch_xy"

def make_view_by_lane_fourtops(mode: str, batch, device: torch.device) -> BatchView:
    if mode == "torch_xy":
        xb, yb = batch
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if yb.ndim == 2 and yb.shape[1] == 1:
            yb = yb.view(-1)
        return BatchView(batch_x=xb, batch_y=yb, meta={"mode": mode})

    if mode == "pyg_graph":
        G = batch.to(device)
        y = G.y
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.view(-1)
        return BatchView(batch_x=G, batch_y=y, meta={"mode": mode})

    raise ValueError(f"Unknown FOURTOPS lane mode: {mode}")

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

    # Shape: accept (B,) or (B,1); also tolerate scalar only when B==1 later
    if out.ndim == 2 and out.shape[1] == 1:
        scores = out[:, 0]              # (B,)
    elif out.ndim == 1:
        scores = out                    # (B,)
    elif out.ndim == 0:
        scores = out.view(1)            # (1,)  (will be rejected later if B != 1)
    else:
        raise ValueError(f"Expected output (B,) or (B,1), got shape {tuple(out.shape)}")

    if scores.ndim != 1:
        raise ValueError(f"Expected 1D scores, got shape {tuple(scores.shape)}")

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

def _finite_or_raise(name: str, x):
    """
    Raise with a clear error if x contains NaN/Inf."""
    if x is None:
        return
    if torch.is_tensor(x):
        if not torch.isfinite(x).all():
            # show a small diagnostic
            bad = x[~torch.isfinite(x)]
            ex = bad.flatten()[0].item() if bad.numel() else None
            raise FloatingPointError(f"{name} contains non-finite values. Example={ex}")
        return
    # handle numpy/scalars/lists
    arr = np.asarray(x)
    if not np.isfinite(arr).all():
        bad = arr[~np.isfinite(arr)]
        ex = bad.flatten()[0] if bad.size else None
        raise FloatingPointError(f"{name} contains non-finite values. Example={ex}")

def _extract_logits_tensor(out):
    """
    Best-effort: unwrap common model outputs into a tensor of logits/scores.
    """

    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
        return out[0]
    # dict output
    if isinstance(out, dict):
        for k in ("logits", "scores", "y", "out"):
            v = out.get(k, None)
            if torch.is_tensor(v):
                return v
    return out  # fallback (may be numpy/scalar)

def _compute_binary_loss_from_logits(logits, y):
    """
    Robust-ish binary loss:
    - If logits look like [B,2], use CE
    - else use BCEWithLogits on flattened tensors
    """

    if not torch.is_tensor(logits) or not torch.is_tensor(y):
        return None

    # move shapes into something sane
    if logits.ndim == 2 and logits.shape[1] == 2:
        return F.cross_entropy(logits, y.long().view(-1))
    return F.binary_cross_entropy_with_logits(
        logits.view(-1),
        y.float().view(-1),
    )

def dryrun_finite_check_fourtops(trained_model, spec, val_loader, device, batches: int = 10):
    """
    Dry-run-only numerics guard:
    - checks params finite
    - runs eval forward on several val batches
    - asserts logits + loss finite
    """

    trained_model.eval()

    # 1) parameters finite
    with torch.no_grad():
        for n, p in trained_model.named_parameters():
            if p is None:
                continue
            if torch.is_tensor(p) and not torch.isfinite(p).all():
                raise FloatingPointError(f"Parameter '{n}' contains NaN/Inf.")

    # 2) get lane once
    first_val = next(iter(val_loader))
    mode = detect_and_assert_lane_fourtops(spec, first_val)

    # 3) scan multiple batches (fp32)
    with torch.no_grad():
        trained_model.eval()
        for i, batch in enumerate(itertools.islice(val_loader, batches)):
            view = make_view_by_lane_fourtops(mode, batch, device)

            out = trained_model(view.batch_x)
            logits = _extract_logits_tensor(out)
            _finite_or_raise(f"val_logits[batch={i}]", logits)

            if torch.is_tensor(logits) and torch.is_tensor(view.batch_y):
                loss = _compute_binary_loss_from_logits(logits.float(), view.batch_y)
                if loss is not None:
                    _finite_or_raise(f"val_loss(fp32)[batch={i}]", loss)

    return True
