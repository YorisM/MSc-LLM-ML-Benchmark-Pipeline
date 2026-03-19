
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts, to_python
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops, dryrun_finite_check_fourtops

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
from torch import nn
from torch.utils.data import Dataset
import numpy as np
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
        # X: [N, 92]
        # Indices 0-1: global features (MET magnitude, MET phi)
        # Indices 2-91: 18 objects × 5 features (obj_id, E, p_T, eta, phi)

        # Global features
        global_feats = X[:, :2]  # [N, 2]
        self.global_mean = global_feats.mean(dim=0)
        self.global_std = global_feats.std(dim=0) + 1e-8

        # Object features
        obj_feats = X[:, 2:].reshape(-1, 18, 5)  # [N, 18, 5]

        # Mask for valid objects (obj_id != 0)
        valid_mask = obj_feats[:, :, 0] != 0  # [N, 18]

        # Kinematic features: E, p_T, eta, phi (indices 1-4 within each object)
        kinematics = obj_feats[:, :, 1:]  # [N, 18, 4]

        # Collect valid kinematics only
        kinematics_flat = kinematics.reshape(-1, 4)  # [N*18, 4]
        valid_flat = valid_mask.reshape(-1)  # [N*18]
        valid_kinematics = kinematics_flat[valid_flat]  # [num_valid, 4]

        self.obj_mean = valid_kinematics.mean(dim=0)
        self.obj_std = valid_kinematics.std(dim=0) + 1e-8

        return self

    def transform(self, X):
        # X: [N, 92] -> [N, 92] normalized
        X_out = X.clone()

        # Normalize global features
        X_out[:, :2] = (X[:, :2] - self.global_mean) / self.global_std

        # Normalize object kinematics
        obj_feats = X[:, 2:].reshape(-1, 18, 5).clone()  # [N, 18, 5]
        obj_feats[:, :, 1:] = (obj_feats[:, :, 1:] - self.obj_mean) / self.obj_std
        X_out[:, 2:] = obj_feats.reshape(-1, 90)

        return X_out

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Global feature embedding
        self.global_embed = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Object kinematic embedding (E, p_T, eta, phi)
        self.obj_embed = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Learnable positional embeddings for objects
        self.pos_embed = nn.Parameter(torch.randn(1, 18, 128) * 0.02)

        # Transformer encoder for learning object interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, batch_x):
        # batch_x: [B, 92]
        B = batch_x.shape[0]

        # Extract global features
        global_feats = batch_x[:, :2]  # [B, 2]
        global_embed = self.global_embed(global_feats)  # [B, 128]

        # Extract object features
        obj_feats = batch_x[:, 2:].reshape(B, 18, 5)  # [B, 18, 5]
        obj_id = obj_feats[:, :, 0]  # [B, 18]
        obj_kinematics = obj_feats[:, :, 1:]  # [B, 18, 4]

        # Create padding mask (True for padding positions)
        mask = (obj_id == 0)  # [B, 18]

        # Embed object kinematics
        obj_embed = self.obj_embed(obj_kinematics)  # [B, 18, 128]

        # Add positional embeddings
        obj_embed = obj_embed + self.pos_embed  # [B, 18, 128]

        # Apply transformer encoder
        obj_encoded = self.transformer(obj_embed, src_key_padding_mask=mask)  # [B, 18, 128]

        # Mean pooling over valid objects
        mask_expanded = mask.unsqueeze(-1).expand_as(obj_encoded)  # [B, 18, 128]
        obj_encoded_masked = obj_encoded.masked_fill(mask_expanded, 0.0)  # [B, 18, 128]
        num_valid = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1.0)  # [B, 1]
        obj_pooled = obj_encoded_masked.sum(dim=1) / num_valid  # [B, 128]

        # Concatenate global and pooled object features
        combined = torch.cat([global_embed, obj_pooled], dim=1)  # [B, 256]

        # Classify
        logits = self.classifier(combined)  # [B, 1]

        return logits.squeeze(1)  # [B]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model, train_loader, val_loader, epochs):
    device = next(model.parameters()).device

    # Optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # Tracking metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Early stopping variables
    best_val_auc = 0.0
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss_sum = 0.0
        train_preds_list = []
        train_labels_list = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()

            # Forward pass
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate metrics
            train_loss_sum += loss.item() * batch_x.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            train_preds_list.extend(probs)
            train_labels_list.extend(batch_y.cpu().numpy())

        # Compute training metrics
        train_loss_avg = train_loss_sum / len(train_loader.dataset)
        train_auc = roc_auc_score(train_labels_list, train_preds_list)

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_preds_list = []
        val_labels_list = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float()

                # Forward pass
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                # Accumulate metrics
                val_loss_sum += loss.item() * batch_x.size(0)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds_list.extend(probs)
                val_labels_list.extend(batch_y.cpu().numpy())

        # Compute validation metrics
        val_loss_avg = val_loss_sum / len(val_loader.dataset)
        val_auc = roc_auc_score(val_labels_list, val_preds_list)

        # Store metrics
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accs.append(train_auc)
        val_accs.append(val_auc)

        # Update learning rate
        scheduler.step()

        # Print progress
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss_avg:.4f}, Train AUC={train_auc:.4f}, "
              f"Val Loss={val_loss_avg:.4f}, Val AUC={val_auc:.4f}")

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    return model, train_losses, val_losses, train_accs, val_accs

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    X_fit, Y_fit = X_train, Y_train
    if dryrun:
        idx = torch.randperm(X_train.shape[0])[:400]
        X_train, Y_train = X_train[idx], Y_train[idx]
        idx = torch.randperm(X_val.shape[0])[:200]
        X_val, Y_val = X_val[idx], Y_val[idx]
    pre = make_preprocessor().fit(X_fit, Y_fit)
    
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
    n_epochs = 10 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # Dry-run safety check
    if dryrun:
        try:
            dryrun_finite_check_fourtops(trained_model, spec, val_loader, device, batches=10)
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
        summary = to_python(summary)
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

