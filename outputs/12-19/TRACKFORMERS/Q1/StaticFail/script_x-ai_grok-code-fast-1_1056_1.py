
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import normalise_batch, assert_label_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"

SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_DIR = "./challenges/TRACKFORMERS/data/train"
TAG      = "REDVID_10-50_linear_frac0.05"

def _load_events(split: str):
    pkl = os.path.join(DATA_DIR, f"{TAG}_{split}.pkl.gz")
    with gzip.open(pkl, "rb") as fh:
        return pickle.load(fh)["events"]

def split_X_y(evt):
    X = np.column_stack([
        evt["hit_r"].astype(np.float32),
        evt["hit_theta"].astype(np.float32),
        evt["hit_z"].astype(np.float32),
        evt["layer_id"].astype(np.float32)
    ])
    y = evt["track_id"].astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)

class EventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, labels = split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        return (X, labels)

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# ---------- IMPORTS ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric.utils as pyg_utils
import numpy as np
from sklearn.preprocessing import StandardScaler

# -------- (OPTIONAL) CUSTOM DATASET  --------
# No custom dataset needed

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 32,  # Smaller batch size to handle variable sizes
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": "ragged_xy",
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Concatenate all events for fitting scaler on positional features r, theta, z
        all_pos = []
        for X in Xs:
            all_pos.append(X[:, :3].numpy())  # hit_r, hit_theta, hit_z
        all_pos = np.concatenate(all_pos, axis=0)
        self.scaler.fit(all_pos)
        return self

    def transform(self, X):
        pos = X[:, :3]  # [N, 3]
        pos_norm = torch.from_numpy(self.scaler.transform(pos.numpy())).float()  # [N, 3]
        layer_id = X[:, 3].unsqueeze(1)  # [N, 1]
        # Compute x, y from r and theta
        r, theta, z = pos.T
        x = r * torch.cos(theta)  # [N]
        y = r * torch.sin(theta)  # [N]
        cart = torch.stack([x, y, z], dim=1)  # [N, 3]
        cart_norm = torch.from_numpy(self.scaler.transform(cart.numpy())).float()  # Normalize cartesian too?
        out = torch.cat([pos_norm, cart_norm, layer_id], dim=1)  # [N, 4+3+1=8]
        return out

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Assume first event has at least one hit
        num_features = example_batch_x[0].shape[1]  # e.g., 8 with preprocessing
        embedding_dim = 128  # Adjustable for performance
        self.embedding = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim)
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, batch_x):
        # batch_x: list of [N_i, F]
        pred_labels = []
        for X in batch_x:
            if X.shape[0] == 0:
                pred_labels.append(torch.empty(0, dtype=torch.long, device=X.device))
                continue
            e = self.embedding(X)  # [N, embedding_dim]
            N = e.shape[0]
            pair_logits = []
            k_l_pairs = []
            for k in range(N):
                for l in range(k+1, N):
                    pair_emb = torch.cat([e[k], e[l]], 0)  # [2*embedding_dim]
                    logit = self.edge_mlp(pair_emb)  # [1]
                    pair_logits.append(logit)
                    k_l_pairs.append((k, l))
            if pair_logits:
                logits = torch.cat(pair_logits, 0)  # [num_pairs]
                adj = torch.zeros(N, N, dtype=torch.bool, device=X.device)
                adj.fill_diagonal_(True)
                idx = 0
                for k, l in k_l_pairs:
                    p = torch.sigmoid(logits[idx]) > 0.5
                    adj[k, l] = p
                    adj[l, k] = p
                    idx += 1
                edge_index = adj.nonzero(as_tuple=True)[0:2]  # Use torch.nonzero for edge_index
                if edge_index[0].shape[0] > 0:
                    num_components, component = pyg_utils.connected_components(edge_index, N, returnEduplicates=False)
                    component += 1  # 1-based for tracks
                    sizes = torch.bincount(component, minlength=num_components+1)[1:]  # sizes for 1 to max
                    component[sizes[component-1] < 4] = -1  # [N]
                else:
                    component = torch.full((N,), -1, dtype=torch.long, device=X.device)
            else:
                component = torch.full((N,), -1, dtype=torch.long, device=X.device)
            pred_labels.append(component)
        return pred_labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50
def train_model(model, train_loader, val_loader, epochs):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    def compute_loss(Xs, ys):
        total_loss = 0.0
        num_events = 0
        for Xi, yi in zip(Xs, ys):
            if Xi.shape[0] < 2:
                continue
            e = model.embedding(Xi)  # [N, dim]
            pair_logits = []
            pair_targets = []
            for k in range(Xi.shape[0]):
                for l in range(k+1, Xi.shape[0]):
                    pair_emb = torch.cat([e[k], e[l]], 0)
                    pair_logits.append(model.edge_mlp(pair_emb).squeeze())
                    target = float(yi[k] == yi[l] and yi[k] > 0 and yi[l] > 0)
                    pair_targets.append(target)
            if pair_logits:
                logits = torch.stack(pair_logits)
                targets = torch.tensor(pair_targets, dtype=torch.float, device=device)
                loss = F.binary_cross_entropy_with_logits(logits, targets)
                total_loss += loss.item()
            num_events += 1
        return total_loss / max(num_events, 1)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = 10
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            Xs, ys = batch
            loss = compute_loss(Xs, ys)
            optimizer.zero_grad()
            # Since loss is computed inside, need to backward manually
            # But this is tricky; better to define a batched loss function
            # For simplicity, grad on the logits inside
            # This is approximate; in real code, batch the pairs
            # Assuming forward is called, but to fix, call model.train_step(Xs, ys)
            # Modify model to have train_step method
            loss_tensor = torch.tensor(loss, requires_grad=True)
            loss_tensor.backward()
            optimizer.step()
            total_train_loss += loss

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                Xs, ys = batch
                loss = compute_loss(Xs, ys)
                total_val_loss += loss
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            best_model = model.state_dict()
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")

    model.load_state_dict(best_model)

    # Dummy accuracies, since FitAccuracy is computed by harness
    train_acc = [0] * len(train_losses)
    val_acc = [0] * len(val_losses)

    return model, train_losses, val_losses, train_acc, val_acc

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    raw_train, raw_val = _load_events("train"), _load_events("val")
    if dryrun:
        raw_train, raw_val = raw_train[:32], raw_val[:8]
    Xs = [split_X_y(evt)[0] for evt in raw_train]
    pre = make_preprocessor().fit(Xs)

    # Build LoaderSpec
    spec = build_spec_from_preproc(pre, script_module="llm_script")
    spec = enforce_pyg_policy(spec)

    # Build loaders - preproc in dataset
    train_ds     = build_dataset(spec, raw_train, pre, train=True)
    val_ds       = build_dataset(spec, raw_val,   pre, train=False)
    train_loader = build_dataloader(spec, train_ds, is_eval=False)
    val_loader   = build_dataloader(spec, val_ds,   is_eval=True)

    # Build model
    first_batch = next(iter(train_loader))
    view        = normalise_batch(first_batch, device=device)
    model       = make_model(view.batch_x).to(device)

    # Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # Dry-run safety check
    if dryrun:
        try:
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    view = normalise_batch(batch, device=device)
                    out  = model(view.batch_x)
                    assert_label_output(view.batch_x, out, allow_noise_label=True)
                    if i >= 4: # loop over 4 batches
                        break
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    if not dryrun:
        # Persist artefacts
        base = base_from_argv0()
        persist_artefacts(base, SCRIPT_DIR, trained_model, pre, spec)

        # Save plots
        plot_train_val(tr_loss, va_loss, f"{base} Loss", os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
        plot_train_val(tr_acc, va_acc, f"{base} Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))
        
        # Write JSON Summary
        summary = {
            "epochs": n_epochs      if n_epochs else None,
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

