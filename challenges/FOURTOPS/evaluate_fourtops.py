# challenges/FOURTOPS/evaluate_fourtops.py

import torch, logging, importlib 
import numpy as np, pandas as pd
from typing import Tuple
from torch.utils.data import TensorDataset
from sklearn.metrics import roc_curve, accuracy_score, auc
from utils.llm_io import _initialize_artefacts, _apply_preproc
from torch_geometric.data import Data, Batch


def load_FOURTOPS_test(model_path):
    """
    Build the SAME DataLoader config the LLM used in training

    RETURNS
    test_loader
    """
    
    # Load preproc
    _, preproc = _initialize_artefacts(model_path)
    cfg        = getattr(preproc, "make_loader_cfg", lambda: {})() or {}
    collate_fn = getattr(preproc, "_collate_fn", None)

    if "loader_class" in cfg:
        mod_path, cls_name = cfg["loader_class"].rsplit(".", 1)
        LoaderCls = getattr(importlib.import_module(mod_path), cls_name)
    else:
        from torch.utils.data import DataLoader as LoaderCls

    # raw test tensors
    X = pd.read_csv("./challenges/FOURTOPS/data/test/X_test.csv",
                    dtype=np.float32).to_numpy(copy=False)
    Y = pd.read_csv("./challenges/FOURTOPS/data/test/Y_test.csv",
                    dtype=np.int64 ).to_numpy(copy=False).ravel()
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).long()
    base_ds = TensorDataset(X, Y)

    test_loader = LoaderCls(
        base_ds,
        batch_size = cfg.get("batch_size", 512),
        shuffle    = cfg.get("shuffle", False),
        num_workers= cfg.get("num_workers", 0),
        collate_fn = collate_fn,
    )

    return test_loader

def evaluate_FOURTOPS(model_path: str, test_loader) \
    -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    PARAMETERS
    model_path  : path to `<MODEL>_model.pkl`
    test_loader : DataLoader yielding (features, labels)

    RETURNS
    auc      : float
    acc      : float
    fpr, tpr : np.ndarray
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Evaluating %s on %s", model_path, device)

    model, preproc = _initialize_artefacts(model_path)
    model = model.to(device)

    # Helper to move arbitrarily nested tensors
    def _to(t, dev):
        if isinstance(t, torch.Tensor):
            return t.to(dev)
        elif isinstance(t, (tuple, list)):
            return tuple(_to(x, dev) for x in t)
        else:
            raise TypeError("Unexpected type in batch:", type(t))

    # Determine output mode
    it  = iter(test_loader(model_path))
    xb0, yb0 = next(it)
    xb0, yb0 = _to(xb0, device), yb0.to(device)
    xb0      = _apply_preproc(preproc, xb0)
    if isinstance(xb0, list) and xb0 and isinstance(xb0[0], Data):
        xb0 = Batch.from_data_list(xb0)

    out0 = model(*xb0) if isinstance(xb0, tuple) else model(xb0)
    out0 = out0.detach()

    mn, mx = out0.min().item(), out0.max().item()
    mode   = "prob" if 0.0 <= mn <= mx <= 1.0 else "logit"

    if mode == "prob":
        act = (lambda o: o[:, 1])                          if out0.ndim == 2 and out0.size(1) == 2 else \
              (lambda o: o.squeeze())
    else:  # logits
        act = (lambda o: torch.sigmoid(o).squeeze(1))      if out0.ndim == 2 and out0.size(1) == 1 else \
              (lambda o: torch.softmax(o, 1)[:, 1])        if out0.ndim == 2 and out0.size(1) == 2 else \
              (lambda o: torch.sigmoid(o))
    logging.info("Output mode: %s", mode)

    # Collect Predictions
    all_probs  = act(out0).cpu().numpy().tolist()
    all_labels = yb0.cpu().numpy().tolist()

    with torch.no_grad():
        for xb, yb in it:
            xb, yb = _to(xb, device), yb.to(device)
            xb = _apply_preproc(preproc, xb)
            if isinstance(xb, list) and xb and isinstance(xb[0], Data):
                xb = Batch.from_data_list(xb)
            logits = model(*xb) if isinstance(xb, tuple) else model(xb)
            all_probs.extend(act(logits).cpu().numpy().tolist())
            all_labels.extend(yb.cpu().numpy().tolist())

    # Metrics
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc     = auc(fpr, tpr)
    acc         = accuracy_score(all_labels,
                                 (np.array(all_probs) >= 0.5).astype(int))

    logging.info("Evaluation finished: AUC %.4f  ACC %.4f", roc_auc, acc)

    metrics = {
        "auc"       : roc_auc,
        "accuracy"  : acc,
        "fpr"       : fpr,
        "tpr"       : tpr
    }
    
    return metrics