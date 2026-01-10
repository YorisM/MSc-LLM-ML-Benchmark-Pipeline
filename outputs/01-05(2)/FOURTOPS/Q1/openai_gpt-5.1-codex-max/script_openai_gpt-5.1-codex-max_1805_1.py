
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
                        
DATASET = {
    "X_train": "./challenges/FOURTOPS/data/train/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/train/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/train/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/train/Y_val.csv"
}
                       
def load_data():
    X_train = pd.read_csv(DATASET["X_train"], dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv(DATASET["Y_train"], dtype=np.int64).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv(DATASET["X_val"], dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv(DATASET['Y_val'], dtype=np.int64).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train), torch.from_numpy(Y_train),
            torch.from_numpy(X_val), torch.from_numpy(Y_val))

class FourTopsDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        X2 = pre.transform(X) if pre is not None else X
        if not torch.is_tensor(X2):
            X2 = torch.as_tensor(X2)
        self.X = X2.float()
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        self.y = y.long()
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

import math
from copy import deepcopy
from typing import List

class MyPreprocessor:
    def __init__(self):
        self.id_indices: List[int] = []
        self.non_id_indices: List[int] = []
        self.mean = None
        self.std = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        F = X.shape[1]
        self.id_indices = [i for i in range(F) if i >= 2 and (i - 2) % 5 == 0]
        self.non_id_indices = [i for i in range(F) if i not in self.id_indices]
        mean_list = torch.zeros(F)
        std_list = torch.ones(F)
        with torch.no_grad():
            for idx in range(F):
                if idx in self.id_indices:
                    mean_list[idx] = 0.0
                    std_list[idx] = 1.0
                else:
                    xi = X[:, idx]
                    if torch.is_tensor(xi):
                        mask = xi != 0
                        if mask.any():
                            vals = xi[mask].float()
                            mean_list[idx] = vals.mean()
                            std = vals.std(unbiased=False)
                            std_list[idx] = std if std > 1e-6 else 1.0
                        else:
                            mean_list[idx] = 0.0
                            std_list[idx] = 1.0
        self.mean = mean_list
        self.std = std_list
        return self

    def transform(self, X):
        if not torch.is_tensor(X):
            X = torch.as_tensor(X)
        X2 = X.clone()
        for idx in self.non_id_indices:
            xi = X2[:, idx]
            mask = xi != 0
            if mask.any():
                X2[:, idx] = (xi - self.mean[idx]) / self.std[idx]
            else:
                X2[:, idx] = 0.0
        return X2.float()

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[1]
        self.num_objects = 18
        self.obj_feat_dim = 5
        self.global_dim = 2
        self.embedding_dim = 8
        self.max_id = 100  # safe upper bound for object id embedding
        hidden_obj = 128
        hidden_final = 128
        self.id_emb = nn.Embedding(self.max_id, self.embedding_dim, padding_idx=0)
        self.obj_mlp = nn.Sequential(
            nn.Linear(4 + self.embedding_dim, hidden_obj),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_obj),
            nn.Dropout(0.1),
            nn.Linear(hidden_obj, hidden_obj),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_obj),
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_obj * 2 + self.global_dim + 1, hidden_final),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_final),
            nn.Dropout(0.2),
            nn.Linear(hidden_final, hidden_final // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_final // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_final // 2, 1)
        )

    def forward(self, batch_x):
        # batch_x: [B, F]
        B = batch_x.shape[0]
        global_feat = batch_x[:, :self.global_dim]  # [B,2]
        objs = batch_x[:, self.global_dim:]  # [B,90]
        objs = objs.view(B, self.num_objects, self.obj_feat_dim)  # [B,18,5]
        obj_ids = objs[:, :, 0].long().clamp(0, self.max_id - 1)  # [B,18]
        cont_feats = objs[:, :, 1:]  # [B,18,4]
        emb = self.id_emb(obj_ids)  # [B,18,embedding_dim]
        obj_input = torch.cat([cont_feats, emb], dim=-1)  # [B,18,4+emb]
        obj_input_flat = obj_input.view(B * self.num_objects, -1)  # [B*18, F_obj]
        obj_repr = self.obj_mlp(obj_input_flat)  # [B*18, hidden_obj]
        obj_repr = obj_repr.view(B, self.num_objects, -1)  # [B,18,hidden_obj]
        mask = (obj_ids > 0)  # [B,18]
        mask_float = mask.float()
        mask_sum = mask_float.sum(dim=1, keepdim=True)  # [B,1]
        mask_sum_clamped = mask_sum.clamp(min=1.0)
        masked_repr = obj_repr * mask_float.unsqueeze(-1)  # [B,18,H]
        mean_pool = masked_repr.sum(dim=1) / mask_sum_clamped  # [B,H]
        masked_repr_neginf = obj_repr.masked_fill(~mask.unsqueeze(-1), -1e9)
        max_pool = masked_repr_neginf.max(dim=1).values  # [B,H]
        final_input = torch.cat([mean_pool, max_pool, global_feat, mask_sum], dim=-1)  # [B, 2H+3]
        out = self.final_mlp(final_input).squeeze(-1)  # [B]
        return out

def make_model(example_object):
    return BinaryClassifier(example_object)

EPOCHS = 10
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    best_val_loss = float('inf')
    best_state = None
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    patience = 5
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float()
            optimizer.zero_grad()
            logits = model(xb)  # [B]
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).long()
                correct += (preds.view(-1) == yb.long()).sum().item()
                total += xb.size(0)
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        train_loss_list.append(epoch_train_loss)
        train_acc_list.append(epoch_train_acc)
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).float()
                logits = model(xb)
                loss = criterion(logits, yb)
                val_running_loss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(logits) > 0.5).long()
                val_correct += (preds.view(-1) == yb.long()).sum().item()
                val_total += xb.size(0)
        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total
        val_loss_list.append(epoch_val_loss)
        val_acc_list.append(epoch_val_acc)
        scheduler.step(epoch_val_loss)
        if epoch_val_loss < best_val_loss - 1e-4:
            best_val_loss = epoch_val_loss
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_loss_list, val_loss_list, train_acc_list, val_acc_list

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    if dryrun:
        idx = torch.randperm(X_train.shape[0])[:400]
        X_train, Y_train = X_train[idx], Y_train[idx]
        idx = torch.randperm(X_val.shape[0])[:20]
        X_val, Y_val = X_val[idx], Y_val[idx]
    pre     = make_preprocessor().fit(X_train, Y_train)
    
    # Build LoaderSpec
    spec = build_spec_from_preproc(pre, script_module="llm_script")
    spec = enforce_pyg_policy(spec, require_torch_collate=False)

    # Build loaders - preproc in dataset
    train_ds     = build_dataset(spec, (X_train, Y_train), pre, train=True)
    val_ds       = build_dataset(spec, (X_val,   Y_val),   pre, train=False)
    train_loader = build_dataloader(spec, train_ds, is_eval=False)
    val_loader   = build_dataloader(spec, val_ds,   is_eval=True)

    # Build batch and check
    first_batch = next(iter(train_loader))
    mode = detect_and_assert_lane_fourtops(spec, first_batch)
    view = make_view_by_lane_fourtops(mode, first_batch, device)

    # Build model
    model = make_model(view.batch_x).to(device)

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
                mode = detect_and_assert_lane_fourtops(spec, first_batch)
                view = make_view_by_lane_fourtops(mode, first_batch, device)
                out  = trained_model(view.batch_x)
                scores, kind = assert_binary_output(view, out)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e

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

