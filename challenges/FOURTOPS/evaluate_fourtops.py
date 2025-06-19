# challenges/FOURTOPS/evaluate.py


import torch, logging 
import numpy as np, pandas as pd
from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, accuracy_score, auc
from utils.llm_io import _initialize_artefacts, _apply_preproc


def load_FOURTOPS_test():
    X = pd.read_csv('./challenges/FOURTOPS/data/X_test.csv',
                          dtype=np.float32).to_numpy(copy=False)
    Y = pd.read_csv('./challenges/FOURTOPS/data/Y_test.csv',
                          dtype=np.int64).to_numpy(copy=False).ravel()
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).long()
    test_ds = TensorDataset(X, Y)
    return (DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0))

def evaluate_FOURTOPS(model_path: str, test_loader) \
    -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    PARAMETERS
    model_path  : path to `<MODEL>_model.pkl`
    test_loader : DataLoader yielding (features, labels)

    RETURNS
    fpr, tpr : np.ndarray
    auc      : float
    acc      : float
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
    it  = iter(test_loader)
    xb0, yb0 = next(it)
    xb0, yb0 = _to(xb0, device), yb0.to(device)
    xb0      = _apply_preproc(preproc, xb0)      # ← fixed: xb0 not xb
    out0 = model(*xb0) if isinstance(xb0, (tuple, list)) else model(xb0)
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
            logits = model(*xb) if isinstance(xb,(tuple,list)) else model(xb)
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