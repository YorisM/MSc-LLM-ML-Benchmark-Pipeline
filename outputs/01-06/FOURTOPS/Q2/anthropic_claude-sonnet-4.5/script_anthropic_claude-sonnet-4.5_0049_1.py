
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

# ---------- IMPORTS ----------
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import copy

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.obj_mean = None
        self.obj_std = None

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
        # X: [N, 92] = [E_T_miss, phi_ET_miss, obj1_features(5), ..., obj18_features(5)]

        # Global features (indices 0-1)
        global_feats = X[:, :2]  # [N, 2]
        self.global_mean = global_feats.mean(dim=0)
        self.global_std = global_feats.std(dim=0).clamp(min=1e-8)

        # Object features (indices 2-91, reshape to [N, 18, 5])
        obj_feats = X[:, 2:].reshape(-1, 18, 5)  # [N, 18, 5]

        # Collect statistics only from valid objects (obj_id != 0)
        valid_mask = (obj_feats[:, :, 0] != 0)  # [N, 18]

        # Gather valid kinematic features (E, pT, eta, phi at indices 1-4)
        valid_kinematics = []
        for i in range(obj_feats.shape[0]):
            for j in range(18):
                if valid_mask[i, j]:
                    valid_kinematics.append(obj_feats[i, j, 1:5])

        if len(valid_kinematics) > 0:
            valid_kinematics = torch.stack(valid_kinematics)  # [M, 4]
            self.obj_mean = valid_kinematics.mean(dim=0)  # [4]
            self.obj_std = valid_kinematics.std(dim=0).clamp(min=1e-8)  # [4]
        else:
            self.obj_mean = torch.zeros(4)
            self.obj_std = torch.ones(4)

        return self

    def transform(self, X):
        # X: [N, 92]
        X_norm = X.clone()

        # Normalize global features
        X_norm[:, :2] = (X[:, :2] - self.global_mean) / self.global_std

        # Normalize object features
        obj_feats = X_norm[:, 2:].reshape(-1, 18, 5)  # [N, 18, 5]
        valid_mask = (obj_feats[:, :, 0] != 0)  # [N, 18]

        # Normalize kinematic features (indices 1-4: E, pT, eta, phi)
        for k in range(1, 5):
            normalized = (obj_feats[:, :, k] - self.obj_mean[k-1]) / self.obj_std[k-1]
            obj_feats[:, :, k] = torch.where(valid_mask, normalized, obj_feats[:, :, k])

        X_norm[:, 2:] = obj_feats.reshape(-1, 90)

        return X_norm

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Model dimensions
        self.d_model = 128
        self.nhead = 4
        self.num_layers = 3
        self.dim_ff = 256
        self.dropout = 0.1

        # Embedding layers
        self.obj_embed = nn.Linear(5, self.d_model)
        self.global_embed = nn.Linear(2, self.d_model)

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, 18, self.d_model) * 0.02)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_ff,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, batch_x):
        # batch_x: [B, 92]
        B = batch_x.shape[0]

        # Split into global and object features
        global_feats = batch_x[:, :2]  # [B, 2]
        obj_feats = batch_x[:, 2:].reshape(B, 18, 5)  # [B, 18, 5]

        # Padding mask (True = padding, should be ignored)
        pad_mask = (obj_feats[:, :, 0] == 0)  # [B, 18]

        # Embed objects
        obj_h = self.obj_embed(obj_feats) + self.pos_embed  # [B, 18, d_model]

        # Transform
        obj_h = self.transformer(obj_h, src_key_padding_mask=pad_mask)  # [B, 18, d_model]

        # Pool (mean over non-padded positions)
        mask_pool = (~pad_mask).unsqueeze(-1).float()  # [B, 18, 1]
        obj_pooled = (obj_h * mask_pool).sum(dim=1) / (mask_pool.sum(dim=1) + 1e-9)  # [B, d_model]

        # Embed global
        global_h = self.global_embed(global_feats)  # [B, d_model]

        # Combine and classify
        combined = torch.cat([obj_pooled, global_h], dim=1)  # [B, 2*d_model]
        logits = self.classifier(combined).squeeze(-1)  # [B]

        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 25

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

    # History
    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    # Early stopping
    best_val_auc = 0.0
    best_model = None
    patience = 0
    max_patience = 7

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_labels = []

        for X, y in train_loader:
            X, y = X.to(device), y.to(device).float()

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * X.size(0)
            probs = torch.sigmoid(logits)
            train_correct += ((probs > 0.5).float() == y).sum().item()
            train_total += y.size(0)

            train_preds.extend(probs.detach().cpu().numpy())
            train_labels.extend(y.cpu().numpy())

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total
        train_auc = roc_auc_score(train_labels, train_preds)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).float()

                logits = model(X)
                loss = criterion(logits, y)

                val_loss_sum += loss.item() * X.size(0)
                probs = torch.sigmoid(logits)
                val_correct += ((probs > 0.5).float() == y).sum().item()
                val_total += y.size(0)

                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_labels, val_preds)

        # Record history
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        # Update scheduler
        scheduler.step(val_auc)

        # Early stopping check
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        print(f"Epoch {epoch+1}/{epochs}: "
              f"TrLoss={train_loss:.4f} TrAcc={train_acc:.4f} TrAUC={train_auc:.4f} | "
              f"ValLoss={val_loss:.4f} ValAcc={val_acc:.4f} ValAUC={val_auc:.4f}")

        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

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

