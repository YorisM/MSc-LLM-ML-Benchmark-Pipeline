
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

# <start code template>
# 0. ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch, scipy, sklearn (sub-)modules you actually use.
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
        self.global_mean, self.global_std = None, None
        self.object_mean, self.object_std = None, None
        self.is_fitted = False

    def _raw_reshape(self, X: torch.Tensor):
        # Input X: (N, 92)
        global_features = X[:, :2] # (N, 2)
        object_features = X[:, 2:].reshape(X.shape[0], 18, 5) # (N, 18, 5)
        return global_features, object_features

    def fit(self, X, y=None):
        X_torch = torch.as_tensor(X, dtype=torch.float32)
        globals_raw, objects_raw = self._raw_reshape(X_torch)

        # Global features: E_T_miss, phi -> E_T_miss, cos(phi), sin(phi)
        g_phi = globals_raw[:, 1]
        g_feats = torch.stack([globals_raw[:, 0], torch.cos(g_phi), torch.sin(g_phi)], dim=1) # (N, 3)
        self.global_mean = g_feats.mean(dim=0)
        self.global_std = g_feats.std(dim=0)
        self.global_std[self.global_std < 1e-8] = 1.0

        # Object features: id, E, pT, eta, phi
        # We normalize E, pT, eta
        mask = objects_raw[:, :, 0] > 0.5
        o_feats_to_norm = objects_raw[:, :, 1:4] # E, pT, eta

        active_objects_feats = o_feats_to_norm[mask] # (n_active, 3)
        self.object_mean = active_objects_feats.mean(dim=0)
        self.object_std = active_objects_feats.std(dim=0)
        self.object_std[self.object_std < 1e-8] = 1.0

        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming.")

        X_torch = torch.as_tensor(X, dtype=torch.float32)
        globals_raw, objects_raw = self._raw_reshape(X_torch)
        mask = objects_raw[:, :, 0] > 0.5 # (N, 18)

        # Process global features
        g_phi = globals_raw[:, 1]
        g_feats = torch.stack([globals_raw[:, 0], torch.cos(g_phi), torch.sin(g_phi)], dim=1) # (N, 3)
        norm_g_feats = (g_feats - self.global_mean) / self.global_std # (N, 3)

        # Process object features
        o_phi = objects_raw[:, :, 4]
        o_feats_normed_part = (objects_raw[:, :, 1:4] - self.object_mean) / self.object_std
        o_feats_trig_part = torch.stack([torch.cos(o_phi), torch.sin(o_phi)], dim=-1) # (N, 18, 2)

        # Combine object features: [norm(E), norm(pT), norm(eta), cos(phi), sin(phi)]
        processed_objects = torch.cat([o_feats_normed_part, o_feats_trig_part], dim=-1) # (N, 18, 5)

        # Zero out padded objects after normalization
        processed_objects = processed_objects * mask.unsqueeze(-1).float()

        return norm_g_feats, processed_objects, mask

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Infer dimensions from preprocessed sample
        global_features_sample, object_features_sample, _ = sample_object
        global_dim = global_features_sample.shape[-1]
        object_dim = object_features_sample.shape[-1]

        d_model = 64
        nhead = 4
        num_encoder_layers = 3
        dim_feedforward = 128
        dropout = 0.1

        # Object embedding layer
        self.obj_embed = nn.Linear(object_dim, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True, # Process as (Batch, Seq, Feature)
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d_model + global_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, *data):
        # Unpack preprocessed data tuple
        global_features, object_features, object_mask = data[0], data[1], data[2] # Bx3, Bx18x5, Bx18

        # Embed object features
        obj_embeddings = self.obj_embed(object_features) # (B, 18, d_model)

        # Transformer needs mask where True indicates a masked (padded) position.
        # Our mask is True for valid data, so we invert it.
        padding_mask = ~object_mask
        transformer_output = self.transformer_encoder(
            obj_embeddings, src_key_padding_mask=padding_mask
        ) # (B, 18, d_model)

        # Aggregate object features (masked average pooling)
        # Ensure we don't divide by zero for events with no valid objects
        num_valid_objects = object_mask.sum(dim=1, keepdim=True).clamp(min=1)
        # Apply mask before summing to zero out contributions from padded tokens
        masked_output = transformer_output * object_mask.unsqueeze(-1)
        obj_summary = masked_output.sum(dim=1) / num_valid_objects # (B, d_model)

        # Concatenate with global features
        combined_features = torch.cat([obj_summary, global_features], dim=1) # (B, d_model + global_dim)

        # Final classification
        logits = self.classifier(combined_features)
        return logits.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 15
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS
    # Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    # Must return trained_model, train_loss, val_loss, train_acc, val_acc
    # Implement early-stopping.
    # Forward signature must match.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 4 # Stop after 4 epochs of no improvement
    best_model_state = None

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        epoch_train_loss = 0.0
        correct_train, total_train = 0, 0
        for data_tuple, labels in train_loader:
            data = [d.to(device) for d in data_tuple]
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(*data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_acc = correct_train / total_train
        train_loss.append(avg_train_loss)
        train_acc.append(avg_train_acc)

        # --- Validation Phase ---
        model.eval()
        epoch_val_loss = 0.0
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for data_tuple, labels in val_loader:
                data = [d.to(device) for d in data_tuple]
                labels = labels.to(device).float()

                outputs = model(*data)
                loss = loss_fn(outputs, labels)

                epoch_val_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_acc = correct_val / total_val
        val_loss.append(avg_val_loss)
        val_acc.append(avg_val_acc)

        scheduler.step(avg_val_loss)

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load the best model state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_loss, val_loss, train_acc, val_acc

# IMPORTANT: DO NOT execute the pipeline here – the harness will do that.
# <end code template>

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

