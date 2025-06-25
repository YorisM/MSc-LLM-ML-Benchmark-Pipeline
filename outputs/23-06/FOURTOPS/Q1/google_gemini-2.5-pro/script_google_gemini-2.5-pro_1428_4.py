
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: Python 3.12, PyTorch 2.6.0, Torch_Geometric 2.6.1, NumPy 2.2.3, SciPy v1.15.2, SciKit-Learn 1.6.1
import os, sys, pickle, torch, torch_geometric, gc, json, importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
                        
DATASET = {
    "X_train": "./challenges/FOURTOPS/data/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/Y_val.csv"
}
                       
def load_data():
    X_train = pd.read_csv(DATASET["X_train"], dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv(DATASET["Y_train"], dtype=np.int64).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv(DATASET["X_val"], dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv(DATASET['Y_val'], dtype=np.int64).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train), torch.from_numpy(Y_train),
            torch.from_numpy(X_val), torch.from_numpy(Y_val))

class PairDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
    
        if isinstance(self.x, (tuple, list)) and all(torch.is_tensor(t) for t in self.x):
            return (tuple(t[idx] for t in self.x), self.y[idx])
        else:
            return (self.x[idx], self.y[idx])

def _make_dataset(x, y):
    custom = globals().get("make_dataset", None)
    if callable(custom):
        ds = custom(x, y)
        if ds is not None:
            return ds
    return PairDataset(x, y)

def make_loaders(X_train, Y_train, X_val, Y_val, *, batch=512, collate_fn=None, loader_cls=None):
    train_ds = _make_dataset(X_train, Y_train)
    val_ds   = _make_dataset(X_val , Y_val)

    if loader_cls is None: 
        loader_cls = DataLoader

    train_ld = loader_cls(train_ds, batch_size=batch, shuffle=True, num_workers=0, 
                        collate_fn=collate_fn)
    val_ld   = loader_cls(val_ds, batch_size=batch, shuffle=False, num_workers=0,
                        collate_fn=collate_fn)

    return train_ld, val_ld

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ----------------                        
# -------------------------- START OF LLM BLOCK ------------------------------

# 0. ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch, scipy, sklearn (sub-)modules you actually use.
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy


# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    #    Must implement:
    #   - fit(...)               -> self
    #   - transform(X: ???)      -> ???

    # DATA SPECIFICS
    # Total flat length per event (X_train & X_val): 92
    # Index  0 :  missing-ET magnitude  (E_T_miss)
    # Index  1 :  missing-ET azimuth    (phi_Et_miss)
    # Indices  2-6  : object 1  ->  obj_1, E_1, p_T1, eta_1, phi_1
    # Indices  7-11 : object 2  ->  obj_2, E_2 , p_T_2 , eta_2 , phi_2
    # ...
    # Indices 88-92 : object 18 ->  obj_18, E_18 , p_T_18 , eta_18 , phi_18
    # Global features       = 2
    # Per-object slice size = 5
    # Max objects encoded   = 18

    # TIPS
    # When modifying data features or feature engineering: annotate tensor size as comments after 
    # each tensor operation to reduce dimension mismatches.

    # REQUIREMENTS
    # IMPORTANT: All state must be picklable with the std-lib pickle module.
    # May allocate NumPy arrays or Torch tensors internally, but:
    # transform() must be deterministic.
    # Store only derived parameters needed for transform i.e. do not store the raw data
    # itself in the preprocessor object.

    def __init__(self):
        # Define and initialize any stateful components here
        self.global_scaler = StandardScaler()
        self.object_scaler = StandardScaler()
        # Engineered features:
        # globals: log(1+E_T_miss), met_x, met_y
        self.n_global_feats = 3
        # objects: log(1+E), log(1+pT), eta, phi, px, py, pz
        self.n_obj_cont_feats = 7 

    def _reshape_and_feature_engineer(self, X: torch.Tensor):           
        # Apply optional raw data reshaping logic here

        # X: [N, 92]
        global_features_raw = X[:, :2]                       # [N, 2]
        objects_raw = X[:, 2:].reshape(-1, 18, 5)             # [N, 18, 5]

        # Create mask for valid objects (p_T > 0), pT is at index 2 of the 5-tuple
        mask = objects_raw[:, :, 2] > 1e-6 # Use a small epsilon for float comparison # [N, 18]

        # --- Feature engineering for global features ---
        E_T_miss = global_features_raw[:, 0]
        phi_E_T_miss = global_features_raw[:, 1]

        log_E_T_miss = torch.log1p(E_T_miss)
        met_x = E_T_miss * torch.cos(phi_E_T_miss)
        met_y = E_T_miss * torch.sin(phi_E_T_miss)

        global_features = torch.stack([log_E_T_miss, met_x, met_y], dim=1) # [N, 3]

        # --- Feature engineering for object features ---
        # The first feature is obj_id, which we treat as categorical
        obj_ids = objects_raw[:, :, 0]                       # [N, 18]
        E = objects_raw[:, :, 1]
        pT = objects_raw[:, :, 2]
        eta = objects_raw[:, :, 3]
        phi = objects_raw[:, :, 4]

        log_E = torch.log1p(E)
        log_pT = torch.log1p(pT)

        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)

        object_continuous_features = torch.stack([
            log_E, log_pT, eta, phi, px, py, pz
        ], dim=-1) # [N, 18, 7]

        # apply mask to zero out padded values
        object_continuous_features[~mask] = 0.0

        return global_features, obj_ids, object_continuous_features, mask


    def fit(self, X, y=None):
        # Extract statistics for fit transformers
        global_features, _, object_cont_features, mask = self._reshape_and_feature_engineer(X)

        # Fit global scaler
        self.global_scaler.fit(global_features.cpu().numpy())

        # Fit object scaler on valid objects only
        if mask.any():
            valid_object_features = object_cont_features[mask]
            self.object_scaler.fit(valid_object_features.cpu().numpy())

        return self

    def transform(self, X):
        # Apply pre-processing logic
        global_features, obj_ids, object_cont_features, mask = self._reshape_and_feature_engineer(X)

        # Scale global features
        global_features_np = global_features.cpu().numpy()
        scaled_global_features_np = self.global_scaler.transform(global_features_np)
        scaled_global_features = torch.from_numpy(scaled_global_features_np).to(global_features.dtype)

        # Scale object features
        scaled_object_features = torch.zeros_like(object_cont_features)
        if mask.any():
            valid_features = object_cont_features[mask]
            if valid_features.shape[0] > 0:
                scaled_valid_np = self.object_scaler.transform(valid_features.cpu().numpy())
                scaled_valid = torch.from_numpy(scaled_valid_np).to(valid_features.dtype)
                scaled_object_features[mask] = scaled_valid

        # Return a tuple of tensors to be handled by the default DataLoader
        return (
            scaled_global_features.float(),
            obj_ids.long(),
            scaled_object_features.float(),
            mask
        )

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Define and initialize any stateful components here
        globals, obj_ids, obj_feats, mask = sample_object

        n_global_features = globals.shape[1]
        n_obj_continuous_features = obj_feats.shape[2]

        # Infer max object ID from the first batch. This is a pragmatic approach.
        self.max_obj_id = int(obj_ids.max().item())

        d_model = 128
        n_head = 4
        n_layers = 4
        d_ff = 256
        dropout = 0.1
        self.obj_id_embedding_dim = 16

        self.obj_id_embedding = nn.Embedding(
            num_embeddings=self.max_obj_id + 2, # Add padding_idx and one for safety
            embedding_dim=self.obj_id_embedding_dim,
            padding_idx=0
        )

        self.continuous_input_proj = nn.Linear(
            n_obj_continuous_features, 
            d_model - self.obj_id_embedding_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation=nn.GELU()
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.classifier_head = nn.Sequential(
            nn.Linear(d_model + n_global_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, *data):
        # Define your model's forward pass here
        globals, obj_ids, obj_feats, mask = data

        B = globals.shape[0]  # Batch size

        # Project object features to d_model
        # obj_ids: [B, 18], obj_feats: [B, 18, 7]
        id_embeds = self.obj_id_embedding(obj_ids)                  # [B, 18, 16]
        cont_embeds = self.continuous_input_proj(obj_feats)        # [B, 18, 112]
        object_embeddings = torch.cat([id_embeds, cont_embeds], dim=-1) # [B, 18, 128]

        # Prepend CLS token for aggregation
        cls_tokens = self.cls_token.expand(B, -1, -1)               # [B, 1, 128]
        full_seq = torch.cat([cls_tokens, object_embeddings], dim=1) # [B, 19, 128]

        # Create transformer mask. Mask should be True for padded values.
        # Our input mask is True for valid values.
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device) # CLS token is never padded
        transformer_mask = torch.cat([cls_mask, ~mask], dim=1) # [B, 19], True for padded element

        transformer_output = self.transformer_encoder(
            src=full_seq, 
            src_key_padding_mask=transformer_mask
        ) # [B, 19, 128]

        # Get CLS token output (the aggregated representation of all objects)
        cls_output = transformer_output[:, 0, :]                   # [B, 128]

        # Concatenate with global features
        # globals: [B, 3]
        combined_features = torch.cat([cls_output, globals], dim=1) # [B, 128 + 3]

        # Final classification head
        logits = self.classifier_head(combined_features).squeeze(-1) # [B]

        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30   # Adjust if you wish
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS 
    # Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    # Must return trained_model, train_loss, val_loss, train_acc, val_acc
    # Implement early-stopping.
    # Forward signature must match.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

    best_val_loss = float('inf')
    early_stopping_patience = 5
    patience_counter = 0
    best_model_state_dict = None

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss, correct_preds, total_preds = 0.0, 0, 0
        for data_tuple, labels in train_loader:
            inputs = [item.to(device) for item in data_tuple]
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(*inputs)
            loss = criterion(outputs, labels.float())

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = torch.round(torch.sigmoid(outputs))
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

        epoch_train_loss = running_loss / total_preds
        epoch_train_acc = correct_preds / total_preds
        train_loss_hist.append(epoch_train_loss)
        train_acc_hist.append(epoch_train_acc)

        # --- Validation Phase ---
        model.eval()
        running_loss, correct_preds, total_preds = 0.0, 0, 0
        with torch.no_grad():
            for data_tuple, labels in val_loader:
                inputs = [item.to(device) for item in data_tuple]
                labels = labels.to(device)

                outputs = model(*inputs)
                loss = criterion(outputs, labels.float())

                running_loss += loss.item() * labels.size(0)
                preds = torch.round(torch.sigmoid(outputs))
                correct_preds += (preds == labels).sum().item()
                total_preds += labels.size(0)

        epoch_val_loss = running_loss / total_preds
        epoch_val_acc = correct_preds / total_preds
        val_loss_hist.append(epoch_val_loss)
        val_acc_hist.append(epoch_val_acc)

        scheduler.step(epoch_val_loss)

        # --- Early Stopping Logic ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state_dict = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    # Load best model weights before returning
    if best_model_state_dict:
        model.load_state_dict(best_model_state_dict)

    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

def _import_dotted(path: str):
    mod, name = path.rsplit(".", 1)
    module = importlib.import_module(mod)
    return getattr(module, name)

def _plot(series_train, series_val, name, out_path):
    plt.figure()
    epochs = range(1, len(series_train) + 1)
    plt.plot(epochs, series_train, label=f"Train {name}")
    plt.plot(epochs, series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    if dryrun:
        X_train, Y_train, X_val, Y_val = X_train[:200], Y_train[:200], X_val[:20], Y_val[:20]
    pre     = make_preprocessor().fit(X_train, Y_train)
    X_train = pre.transform(X_train)
    X_val   = pre.transform(X_val)

    collate = getattr(pre, "_collate_fn", None)
    cfg     = getattr(pre, "make_loader_cfg", lambda: None)() or {}
    loader_cls = _import_dotted(cfg["loader_class"]) if "loader_class" in cfg else None
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val, 
                                            batch      = cfg.get("batch_size", 512), 
                                            collate_fn = collate,
                                            loader_cls = loader_cls)

    # 2. Build model
    first_batch    = next(iter(train_loader))
    example_sample = first_batch[0]
    model          = make_model(example_sample)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. Dry-run safety check
    if dryrun:
        sample, _ = first_batch
        try:
            _ = trained_model(*sample) if isinstance(sample, (tuple, list)) else trained_model(sample)
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

