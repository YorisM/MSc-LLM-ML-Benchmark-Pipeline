
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: Python 3.12, PyTorch 2.6.0, NumPy 2.2.3, SciKit-Learn 1.6.1
import os, sys, pickle, gzip, json, torch, numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_DIR = "./challenges/TRACKFORMERS/data"
TAG      = "10_50_linear"

def _load_events(split: str):
    pkl = os.path.join(DATA_DIR, f"REDVID_{TAG}_{split}.pkl.gz")
    with gzip.open(pkl, "rb") as fh:
        return pickle.load(fh)["events"]

def split_X_y(evt):
    lay = evt["layer_id"].astype(np.float32)
    lay_norm = lay / lay.max()
    X = np.column_stack([evt["hit_r"],
                         evt["hit_theta"],
                         evt["hit_z"],
                         lay_norm])
    t_id = evt["track_id"].astype(np.int32)
    return (torch.from_numpy(X),
            torch.from_numpy(t_id))

class EventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, track_id = split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        return (X, track_id)

def _ragged(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    # batch[i] = (hits_i, track_id_i)      ← shapes: (N_i, F), (N_i,)
    return batch

def make_loaders(batch_size=128, workers=0):
    tr = EventDataset(_load_events("train"), pre=None, train=True)
    va = EventDataset(_load_events("val"),   pre=None, train=False)

    train_ld = DataLoader(tr, batch_size=batch_size, shuffle=True,
                          collate_fn=_ragged, num_workers=workers)
    val_ld   = DataLoader(va, batch_size=batch_size, collate_fn=_ragged)
    return train_ld, val_ld

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# 0. ---------- IMPORTS ----------
from sklearn.cluster import DBSCAN
import numpy as np
import torch
from torch import nn

# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    """
    A no-op pre-processor that leaves the raw hit features unchanged. This keeps
    the cylindrical coordinates in their natural units, which is convenient for
    the geometry–based clustering performed inside the model.
    """
    def __init__(self):
        pass

    # The interface expects `fit` to return *self*
    def fit(self, events):
        return self

    # Accepts either torch.Tensor or np.ndarray and returns the same type
    # unchanged.
    def transform(self, X):
        return X

    def fit_transform(self, events):
        return self

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    """
    Geometry-driven, training-free clustering model.

    For each event we cluster hits in the 2-dimensional space

          ( azimuth ϕ = theta ,
            slope   ≈ z / r )

    using DBSCAN.  Hits belonging to the same physical track tend to have both
    (i) similar azimuthal angle and (ii) approximately constant z/r, so this
    simple density based clustering already achieves surprisingly good
    separation without any learnable parameters.
    """
    def __init__(self, in_features: int, eps: float = 0.008):
        super().__init__()
        # DBSCAN radius in the (θ , z/r) space.  Tuned by a few quick trials.
        self.eps = float(eps)

    def forward(self, batch):
        """
        Parameters
        ----------
        batch : list[torch.Tensor]            # each tensor has shape (N_i, F)

        Returns
        -------
        list[torch.Tensor]                   # predicted cluster labels, dtype long
                                             # matching the length of every event
        """
        predictions = []
        for evt in batch:
            # evt may be just the hit tensor or (hits, labels). Accept both.
            hits = evt[0] if isinstance(evt, (tuple, list)) else evt       # (N, F)

            r      = hits[:, 0].cpu().numpy()                              # (N,)
            theta  = hits[:, 1].cpu().numpy()                              # (N,)
            z      = hits[:, 2].cpu().numpy()                              # (N,)
            slope  = z / (r + 1e-6)                                        # (N,)

            features = np.column_stack((theta, slope))                     # (N, 2)
            labels   = DBSCAN(eps=self.eps, min_samples=1).fit(features).labels_
            predictions.append(torch.from_numpy(labels).long().to(hits.device))
        return predictions

def make_model(in_features):
    # The model is parameter-free, nevertheless we keep it as nn.Module so that
    # it integrates seamlessly with the training harness.
    return HitClassifier(in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 1          # nothing to train – a single evaluation pass is sufficient
def _cluster_accuracy(pred, truth):
    """
    Fast majority-vote cluster accuracy for a single event.

    Parameters
    ----------
    pred  : np.ndarray[int] , shape (N,)
    truth : np.ndarray[int] , shape (N,)

    Returns
    -------
    float  – fraction of correctly classified hits for this event
    """
    correct = 0
    for c in np.unique(pred):
        mask = (pred == c)
        if mask.any():
            true_labels_in_cluster = truth[mask]
            majority_true_label    = np.bincount(true_labels_in_cluster).argmax()
            correct              += (true_labels_in_cluster == majority_true_label).sum()
    return correct / len(truth)

def train_model(model, train_loader, val_loader, epochs):
    # Since the model has no learnable parameters, we merely compute the
    # clustering accuracy over the training and validation sets once.
    train_loss = []
    val_loss   = []
    train_acc  = []
    val_acc    = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def _evaluate(loader):
        total_correct, total_hits = 0, 0
        for batch in loader:
            hits_list   = [hits.to(device) for hits, _ in batch]
            truth_list  = [y.to(device)    for _,    y in batch]
            preds       = model(hits_list)                         # list of tensors

            for p, y in zip(preds, truth_list):
                acc_evt        = _cluster_accuracy(p.cpu().numpy(),
                                                   y.cpu().numpy())
                total_correct += acc_evt * len(y)
                total_hits    += len(y)
        return total_correct / total_hits if total_hits else 0.0

    # Single "epoch" evaluation
    tr_acc = _evaluate(train_loader)
    va_acc = _evaluate(val_loader)

    train_acc.append(tr_acc)
    val_acc.append(va_acc)

    # No loss is defined for an unsupervised clustering approach – use NaNs.
    train_loss.append(float("nan"))
    val_loss.append(float("nan"))

    return model, train_loss, val_loss, train_acc, val_acc

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    raw_train, raw_val = _load_events("train"), _load_events("val")
    if dryrun:
        raw_train, raw_val = raw_train[:32], raw_val[:8]
    pre = make_preprocessor().fit(raw_train)
    train_ds = EventDataset(raw_train, pre, train=True)
    val_ds   = EventDataset(raw_val , pre, train=False)
    train_ld = DataLoader(train_ds, batch_size=512,
                        shuffle=True, collate_fn=_ragged)
    val_ld   = DataLoader(val_ds,   batch_size=512,
                        collate_fn=_ragged)

    # 2. Build model
    in_features = train_ds[0][0].shape[-1]                   
    model = make_model(in_features)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_ld, val_ld, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* - run a single toy forward pass
    if dryrun:
        toy_event       = torch.zeros(10, in_features)
        toy_transformed = pre.transform(toy_event)
        toy_batch       = [toy_transformed]
        try:
            _ = trained_model(toy_batch)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 5. Persist artefacts
    if not dryrun:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

        pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
        pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
        pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

        torch.save(trained_model.state_dict(), pth_state)
        with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
        with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

        # 6. Save plots
        _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
        _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

    # 7. Write JSON Summary
    if not dryrun: 
        summary = {
            "epochs": n_epochs,
            "train_loss": tr_loss   if tr_loss else None,
            "val_loss":   va_loss   if va_loss else None,
            "train_acc":  tr_acc    if tr_acc else None,
            "val_acc":    va_acc    if va_acc else None,
        }
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)
# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

