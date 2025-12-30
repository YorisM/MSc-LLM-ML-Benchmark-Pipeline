
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import normalise_batch, assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts

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
        self.X = pre.transform(X) if pre is not None else X
        self.y = y
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

import numpy as np
from sklearn.metrics import roc_auc_score
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader

class MyPreprocessor:
    def __init__(self):
        self.means = None
        self.stds = None

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
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = np.asarray(X, dtype=np.float32)
        n_features = X_np.shape[1]
        means = np.zeros(n_features, dtype=np.float32)
        stds = np.ones(n_features, dtype=np.float32)
        # ETmiss magnitude
        means[0] = X_np[:, 0].mean()
        stds[0] = X_np[:, 0].std() + 1e-6
        # phi missing left unscaled
        means[1] = 0.0
        stds[1] = 1.0
        for i in range(18):
            base = 2 + i * 5
            id_col = base
            mask = X_np[:, id_col] != 0
            if np.any(mask):
                e_col = base + 1
                pt_col = base + 2
                eta_col = base + 3
                means[e_col] = X_np[mask, e_col].mean()
                stds[e_col] = X_np[mask, e_col].std() + 1e-6
                means[pt_col] = X_np[mask, pt_col].mean()
                stds[pt_col] = X_np[mask, pt_col].std() + 1e-6
                means[eta_col] = X_np[mask, eta_col].mean()
                stds[eta_col] = X_np[mask, eta_col].std() + 1e-6
            # leave id and phi unscaled
            means[id_col] = 0.0
            stds[id_col] = 1.0
            means[base + 4] = 0.0
            stds[base + 4] = 1.0
        self.means = means
        self.stds = stds
        return self

    def transform(self, X):
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = np.asarray(X, dtype=np.float32)
        X_out = X_np.copy()
        # scale ETmiss magnitude
        X_out[:, 0] = (X_np[:, 0] - self.means[0]) / self.stds[0]
        # phi missing unchanged
        for i in range(18):
            base = 2 + i * 5
            id_col = base
            mask = X_np[:, id_col] != 0
            if np.any(mask):
                e_col = base + 1
                pt_col = base + 2
                eta_col = base + 3
                X_out[mask, e_col] = (X_np[mask, e_col] - self.means[e_col]) / self.stds[e_col]
                X_out[mask, pt_col] = (X_np[mask, pt_col] - self.means[pt_col]) / self.stds[pt_col]
                X_out[mask, eta_col] = (X_np[mask, eta_col] - self.means[eta_col]) / self.stds[eta_col]
            # id and phi unchanged
        return torch.from_numpy(X_out.astype(np.float32))

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.embed_dim = 8
        self.num_embeddings = 256
        self.id_embedding = nn.Embedding(self.num_embeddings, self.embed_dim, padding_idx=0)
        obj_feat_dim = self.embed_dim + 6  # E, pT, eta, sinphi, cosphi, mask
        self.obj_mlp = nn.Sequential(
            nn.Linear(obj_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.attention_net = nn.Sequential(
            nn.Linear(obj_feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        combined_dim = 64 * 2 + 64  # pooled, max, mean
        global_extra = 3 + 4  # etmiss+sin/cos, count,sum_pt,sum_E,mean_eta? wait mean_eta included below
        global_extra += 1  # mean_eta
        self.final_mlp = nn.Sequential(
            nn.Linear(combined_dim + global_extra, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        if isinstance(batch_x, (list, tuple)):
            x = batch_x[0]
        else:
            x = batch_x
        x = x.float()
        B = x.shape[0]
        etmiss = x[:, 0]  # (B,)
        phi_miss = x[:, 1]  # (B,)
        global_feats = torch.stack([etmiss, torch.sin(phi_miss), torch.cos(phi_miss)], dim=1)  # (B,3)
        objs = x[:, 2:].view(B, 18, 5)  # (B,18,5)
        ids = objs[:, :, 0].long()  # (B,18)
        mask = (ids != 0).float()  # (B,18)
        E = objs[:, :, 1]  # (B,18)
        pT = objs[:, :, 2]  # (B,18)
        eta = objs[:, :, 3]  # (B,18)
        phi = objs[:, :, 4]  # (B,18)
        id_embed = self.id_embedding(torch.clamp(ids, max=self.num_embeddings - 1))  # (B,18,embed_dim)
        sinphi = torch.sin(phi)
        cosphi = torch.cos(phi)
        obj_features = torch.cat([
            id_embed,
            E.unsqueeze(-1),
            pT.unsqueeze(-1),
            eta.unsqueeze(-1),
            sinphi.unsqueeze(-1),
            cosphi.unsqueeze(-1),
            mask.unsqueeze(-1)
        ], dim=-1)  # (B,18,obj_feat_dim)
        obj_latent = self.obj_mlp(obj_features)  # (B,18,64)
        obj_latent = obj_latent * mask.unsqueeze(-1)
        # Attention pooling
        att_scores = self.attention_net(obj_features).squeeze(-1)  # (B,18)
        att_scores = att_scores.masked_fill(mask == 0, -1e9)
        att_weights = torch.softmax(att_scores, dim=1)  # (B,18)
        pooled = torch.sum(att_weights.unsqueeze(-1) * obj_latent, dim=1)  # (B,64)
        # max pooling
        masked_latent = obj_latent + (mask.unsqueeze(-1) - 1) * 1e9  # (B,18,64)
        max_pooled = masked_latent.max(dim=1).values  # (B,64)
        # mean pooling
        sum_latent = torch.sum(obj_latent, dim=1)  # (B,64)
        count = mask.sum(dim=1, keepdim=True)  # (B,1)
        mean_pooled = sum_latent / (count + 1e-6)  # (B,64)
        # additional aggregate features
        sum_pt = (pT * mask).sum(dim=1, keepdim=True)  # (B,1)
        sum_E = (E * mask).sum(dim=1, keepdim=True)  # (B,1)
        mean_eta = (eta * mask).sum(dim=1, keepdim=True) / (count + 1e-6)  # (B,1)
        global_agg = torch.cat([global_feats, count, sum_pt, sum_E, mean_eta], dim=1)  # (B,7)
        combined = torch.cat([pooled, max_pooled, mean_pooled, global_agg], dim=1)  # (B,64*3+7)
        out = self.final_mlp(combined).squeeze(-1)  # (B,)
        return out

def make_model(example_object):
    return BinaryClassifier(example_object)

EPOCHS = 15
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_state = copy.deepcopy(model.state_dict())
    best_val_auc = -np.inf
    patience = 5
    no_improve = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        correct = 0
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y
            optimizer.zero_grad()
            logits = model(xb)  # (B,)
            loss = criterion(logits, yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item() * yb.size(0)
            total_samples += yb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == yb).sum().item()
        epoch_loss = running_loss / max(total_samples, 1)
        epoch_acc = correct / max(total_samples, 1)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_total = 0
        val_correct = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y
                logits = model(xb)
                loss = criterion(logits, yb.float())
                val_running_loss += loss.item() * yb.size(0)
                val_total += yb.size(0)
                preds = torch.sigmoid(logits)
                pred_labels = (preds > 0.5).long()
                val_correct += (pred_labels == yb).sum().item()
                all_preds.append(preds.detach().cpu())
                all_labels.append(yb.detach().cpu())
        val_loss = val_running_loss / max(val_total, 1)
        val_losses.append(val_loss)
        val_acc = val_correct / max(val_total, 1)
        # Compute AUC
        try:
            y_true = torch.cat(all_labels).numpy()
            y_score = torch.cat(all_preds).numpy()
            val_auc = roc_auc_score(y_true, y_score)
        except Exception:
            val_auc = val_acc
        val_accs.append(val_auc)
        # Track best model based on val_auc
        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    model.load_state_dict(best_state)
    return model, train_losses, val_losses, train_accs, val_accs

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
                view = normalise_batch(first_batch, device=device)
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

