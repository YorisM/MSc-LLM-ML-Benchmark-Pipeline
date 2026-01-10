
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
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

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
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 256}
        }

    def fit(self, X, y=None):
        # X shape: [N, 92]
        # First 2 features: E_T_miss, phi_E_T_miss
        # Next 90 features: 18 objects × 5 features each

        global_feats = X[:, :2]  # [N, 2]
        self.global_mean = global_feats.mean(dim=0)
        self.global_std = global_feats.std(dim=0) + 1e-6

        obj_feats = X[:, 2:].reshape(-1, 18, 5)  # [N, 18, 5]
        obj_ids = obj_feats[:, :, 0]  # [N, 18]
        valid_mask = obj_ids != 0  # [N, 18]

        obj_stats = []
        for i in range(5):
            feat_values = obj_feats[:, :, i]  # [N, 18]
            valid_values = feat_values[valid_mask]  # [M]
            if len(valid_values) > 0:
                mean_val = valid_values.mean()
                std_val = valid_values.std() + 1e-6
            else:
                mean_val = 0.0
                std_val = 1.0
            obj_stats.append((mean_val, std_val))

        self.obj_mean = torch.tensor([s[0] for s in obj_stats])
        self.obj_std = torch.tensor([s[1] for s in obj_stats])

        return self

    def transform(self, X):
        # X shape: [N, 92]
        X_norm = X.clone()
        X_norm[:, :2] = (X[:, :2] - self.global_mean) / self.global_std  # [N, 2]
        obj_feats = X[:, 2:].reshape(-1, 18, 5)  # [N, 18, 5]
        obj_feats_norm = (obj_feats - self.obj_mean) / self.obj_std  # [N, 18, 5]
        X_norm[:, 2:] = obj_feats_norm.reshape(-1, 90)  # [N, 90]
        return X_norm

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        self.obj_feature_dim = 5
        self.max_objects = 18
        self.d_model = 128

        self.global_embed = nn.Linear(2, 64)
        self.obj_embed = nn.Linear(self.obj_feature_dim, self.d_model)
        self.pos_encoding = nn.Parameter(torch.randn(self.max_objects, self.d_model))

        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=4)

        self.classifier = nn.Sequential(
            nn.Linear(self.d_model + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, batch_x):
        # batch_x shape: [B, 92]
        batch_size = batch_x.shape[0]

        global_feats = batch_x[:, :2]  # [B, 2]
        global_embed = torch.relu(self.global_embed(global_feats))  # [B, 64]

        obj_feats = batch_x[:, 2:].reshape(batch_size, self.max_objects, self.obj_feature_dim)  # [B, 18, 5]
        obj_ids = obj_feats[:, :, 0]  # [B, 18]
        padding_mask = obj_ids == 0  # [B, 18]

        obj_embed = self.obj_embed(obj_feats)  # [B, 18, d_model]
        obj_embed = obj_embed + self.pos_encoding.unsqueeze(0)  # [B, 18, d_model]

        transformer_out = self.transformer(obj_embed, src_key_padding_mask=padding_mask)  # [B, 18, d_model]

        mask_expanded = (~padding_mask).unsqueeze(-1).float()  # [B, 18, 1]
        sum_embed = (transformer_out * mask_expanded).sum(dim=1)  # [B, d_model]
        count = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
        pooled = sum_embed / count  # [B, d_model]

        combined = torch.cat([pooled, global_embed], dim=1)  # [B, d_model + 64]
        logits = self.classifier(combined)  # [B, 1]

        return logits.squeeze(-1)  # [B]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 25

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_val_auc = 0
    patience = 7
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = binary_cross_entropy_with_logits(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float()

                logits = model(batch_x)
                loss = binary_cross_entropy_with_logits(logits, batch_y)

                val_losses.append(loss.item())
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_acc = ((np.array(train_preds) > 0.5) == np.array(train_labels)).mean()
        val_acc = ((np.array(val_preds) > 0.5) == np.array(val_labels)).mean()
        val_auc = roc_auc_score(val_labels, val_preds)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        scheduler.step(val_auc)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
              f"val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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

