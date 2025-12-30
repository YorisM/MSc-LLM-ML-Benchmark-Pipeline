
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
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

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ----------------                        
# -------------------------- START OF LLM BLOCK ------------------------------

# ---------- IMPORTS ----------
import torch
import torch.nn.functional as F
from torch import nn, optim

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # Must implement:
    #   - fit() 
    #   - transform()

    # DATA SPECIFICS
    # Total flat length per event (X_train & X_val): 92
    # Index  0 :  missing-ET magnitude  (E_T_miss)
    # Index  1 :  missing-ET azimuth    (phi_Et_miss)
    # Indices  2-6  : object 1  ->  obj_1, E_1, p_T1, eta_1, phi_1
    # Indices  7-11 : object 2  ->  obj_2, E_2 , p_T_2 , eta_2 , phi_2
    # ...
    # Indices 87-91 : object 18 ->  obj_18, E_18 , p_T_18 , eta_18 , phi_18
    # Global features       = 2
    # Per-object slice size = 5
    # Max objects encoded   = 18

    def __init__(self):
        # Store normalization statistics
        self.mean = None
        self.std = None

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",     # or torch_geometric.loader:DataLoader
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed. Choose one: 
            "collate": None, # (or "ragged_xy" or "identity" - If loader_class is torch_geometric.loader:DataLoader, set "collate": None.)

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Compute mean and std for standardization
        Xf = X.float()
        self.mean = Xf.mean(dim=0)  # (92,)
        self.std = Xf.std(dim=0) + 1e-6  # avoid div by zero
        return self

    def transform(self, X):
        # Standardize using stored mean and std
        Xf = X.float()
        return (Xf - self.mean) / self.std  # (N,92)

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Object embedding
        self.obj_mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attn = nn.Linear(64, 1)
        # Global / summary features
        self.global_mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3 + 64, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, batch_x):
        # batch_x: (B,92)
        B = batch_x.shape[0]
        global_feats = batch_x[:, :2]  # (B,2)
        obj_feats = batch_x[:, 2:]  # (B,90)
        obj_feats = obj_feats.view(B, 18, 5)  # (B,18,5)
        obj_ids = obj_feats[..., 0]  # (B,18)
        # Mask: padded objects have all zeros; use obj_id==0
        mask = (obj_ids != 0)  # (B,18) bool
        mask_float = mask.unsqueeze(-1).float()  # (B,18,1)
        obj_feats = obj_feats * mask_float  # (B,18,5)
        # Embed objects
        obj_emb = self.obj_mlp(obj_feats)  # (B,18,64)
        # Transformer encoder with padding mask (True for pad)
        pad_mask = ~mask  # (B,18)
        obj_enc = self.transformer(obj_emb, src_key_padding_mask=pad_mask)  # (B,18,64)
        obj_enc = obj_enc * mask_float  # (B,18,64)
        # Pooling
        sum_pool = obj_enc.sum(dim=1)  # (B,64)
        denom = mask_float.sum(dim=1).clamp(min=1e-6)  # (B,1)
        mean_pool = sum_pool / denom  # (B,64)
        att_scores = self.attn(obj_enc).squeeze(-1)  # (B,18)
        att_scores = att_scores.masked_fill(~mask, -1e9)  # mask pads
        att_weights = torch.softmax(att_scores, dim=1)  # (B,18)
        att_weights = att_weights.unsqueeze(-1)  # (B,18,1)
        att_pool = (obj_enc * att_weights).sum(dim=1)  # (B,64)
        # Additional global summary features
        pT = obj_feats[:, :, 2]  # (B,18)
        total_pT = pT.sum(dim=1, keepdim=True)  # (B,1)
        max_pT, _ = pT.max(dim=1, keepdim=True)  # (B,1)
        num_obj = mask_float.sum(dim=1)  # (B,1)
        glob_input = torch.cat([global_feats, total_pT, max_pT, num_obj], dim=1)  # (B,5)
        glob_emb = self.global_mlp(glob_input)  # (B,64)
        # Combine
        combined = torch.cat([sum_pool, mean_pool, att_pool, glob_emb], dim=1)  # (B,256)
        logits = self.classifier(combined).squeeze(-1)  # (B,)
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 12
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            bx, by = batch
            bx = bx.to(device)
            by = by.float().to(device)
            optimizer.zero_grad()
            logits = model(bx)  # (B,)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds.cpu() == by.cpu().long()).sum().item()
            total += bx.size(0)
        avg_loss = total_loss / total
        avg_acc = correct / total
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)
        # Validation
        model.eval()
        val_total_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                bx, by = batch
                bx = bx.to(device)
                by = by.float().to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_total_loss += loss.item() * bx.size(0)
                preds = (torch.sigmoid(logits) > 0.5).long()
                val_correct += (preds.cpu() == by.cpu().long()).sum().item()
                val_total += bx.size(0)
        val_loss = val_total_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        scheduler.step(val_loss)
        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    # Load best state if saved
    if 'best_state' in locals():
        model.load_state_dict(best_state)
    return model, train_losses, val_losses, train_accs, val_accs

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

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


