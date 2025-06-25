
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
        # We will scale E_T_miss, E, pT, eta.
        # Phi will be transformed to (sin(phi), cos(phi)) to handle periodicity.
        # Globals to scale: E_T_miss (1 feature)
        self.global_scaler = StandardScaler()
        # Objects to scale: E, p_T, eta (3 features)
        self.object_scaler = StandardScaler()
        self.is_fit = False

    def _raw_reshape(self, X):
        # Reshape the flat tensor into global and object features
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()

        global_feats = X[:, :2] # Shape: (N, 2)
        object_feats = X[:, 2:].reshape(-1, 18, 5) # Shape: (N, 18, 5)
        return global_feats, object_feats

    def fit(self, X, y=None):
        global_feats, object_feats = self._raw_reshape(X)

        # Fit global scaler on E_T_miss
        # global_feats: [E_T_miss, phi_miss]
        self.global_scaler.fit(global_feats[:, 0:1].numpy())

        # Create mask for valid (non-padded) objects
        # Valid objects are those where obj_id is not 0
        mask = object_feats[:, :, 0] != 0 # Shape: (N, 18)

        # Extract features for valid objects to fit the scaler
        # object_feats: [obj_id, E, p_T, eta, phi]
        valid_object_kinematics = object_feats[mask][:, 1:4] # Select E, p_T, eta
        self.object_scaler.fit(valid_object_kinematics.numpy())

        self.is_fit = True
        return self

    def transform(self, X):
        if not self.is_fit:
            raise RuntimeError("Preprocessor must be fit before transforming data.")

        global_feats_raw, object_feats_raw = self._raw_reshape(X) # (N, 2), (N, 18, 5)

        # ---- Process Global Features ----
        et_miss = global_feats_raw[:, 0:1] # (N, 1)
        phi_miss = global_feats_raw[:, 1:2] # (N, 1)

        # Scale E_T_miss
        scaled_et_miss = self.global_scaler.transform(et_miss.numpy())
        scaled_et_miss_t = torch.from_numpy(scaled_et_miss).float() # (N, 1)

        # Transform phi to (sin(phi), cos(phi))
        sin_phi_miss = torch.sin(phi_miss) # (N, 1)
        cos_phi_miss = torch.cos(phi_miss) # (N, 1)

        # Final global features
        processed_globals = torch.cat([scaled_et_miss_t, sin_phi_miss, cos_phi_miss], dim=1) # (N, 3)

        # ---- Process Object Features ----
        obj_ids = object_feats_raw[:, :, 0].long() # (N, 18)
        mask = obj_ids != 0 # (N, 18)

        # Kinematics: E, p_T, eta, phi
        kinematics = object_feats_raw[:, :, 1:] # (N, 18, 4)

        # Features to scale: E, p_T, eta
        feats_to_scale = kinematics[:, :, :3] # (N, 18, 3)
        # Feature to transform: phi
        phi = kinematics[:, :, 3:4] # (N, 18, 1)

        # Apply scaling. We reshape to 2D, scale all, then reshape back
        n_samples, n_objects, _ = feats_to_scale.shape
        flat_feats_to_scale = feats_to_scale.reshape(-1, 3)
        scaled_flat_feats = self.object_scaler.transform(flat_feats_to_scale.numpy())
        scaled_kin = torch.from_numpy(scaled_flat_feats).float().reshape(n_samples, n_objects, 3)

        # Transform phi
        sin_phi = torch.sin(phi) # (N, 18, 1)
        cos_phi = torch.cos(phi) # (N, 18, 1)

        # Combine scaled E, pT, eta with transformed phi
        processed_kinematics = torch.cat([scaled_kin, sin_phi, cos_phi], dim=-1) # (N, 18, 5)

        # Zero out the padded objects using the mask
        processed_kinematics[~mask] = 0.0

        return (processed_globals, obj_ids, processed_kinematics, mask)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # From EDA on the dataset, max particle ID is 15. Vocab size is 16.
        vocab_size = 16 
        d_model = 128
        n_head = 8
        n_layers = 4
        dim_feedforward = 256
        dropout = 0.1

        # Unpack sample object to get dimensions
        sample_globals, _, sample_kin, _ = sample_object
        n_global_feats = sample_globals.shape[-1] # Should be 3
        n_kin_feats = sample_kin.shape[-1] # Should be 5

        # Input processing layers
        # Embedding for categorical particle IDs. padding_idx=0 ensures padded items get zero vector.
        self.obj_embedding = nn.Embedding(vocab_size, d_model // 2, padding_idx=0)
        # Linear projection for continuous kinematics
        self.obj_kin_projection = nn.Linear(n_kin_feats, d_model // 2)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model + n_global_feats, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        global_feats, obj_ids, obj_kinematics, mask = data
        # global_feats: (B, 3)
        # obj_ids: (B, 18)
        # obj_kinematics: (B, 18, 5)
        # mask: (B, 18)

        # 1. Process object features
        embedded_ids = self.obj_embedding(obj_ids)  # (B, 18, d_model/2)
        projected_kin = self.obj_kin_projection(obj_kinematics)  # (B, 18, d_model/2)

        # Combine to form the input to the transformer
        obj_features = torch.cat([embedded_ids, projected_kin], dim=-1) # (B, 18, d_model)

        # 2. Apply Transformer Encoder
        # The mask for transformer needs True for padded elements
        padding_mask = ~mask
        transformer_output = self.transformer_encoder(
            obj_features, src_key_padding_mask=padding_mask
        ) # (B, 18, d_model)

        # 3. Aggregate object representations (masked mean pooling)
        # Zero out the padded outputs before summing
        transformer_output = transformer_output * mask.unsqueeze(-1)
        # Summing over the sequence dimension and dividing by the number of valid particles
        num_valid_objects = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        aggregated_objects = transformer_output.sum(dim=1) / num_valid_objects # (B, d_model)

        # 4. Concatenate with global features
        combined_features = torch.cat([aggregated_objects, global_feats], dim=1) # (B, d_model + n_global_feats)

        # 5. Classifier Head
        logits = self.classifier(combined_features).squeeze(-1) # (B,)

        return logits


def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=False)

    # Early stopping parameters
    early_stopping_patience = 6
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for data, targets in train_loader:
            # Move data to device
            data = tuple(d.to(device) for d in data)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, targets.float())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Accuracy
            preds = torch.sigmoid(outputs) > 0.5
            total_train += targets.size(0)
            correct_train += (preds == targets).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data = tuple(d.to(device) for d in data)
                targets = targets.to(device)

                outputs = model(data)
                loss = criterion(outputs, targets.float())

                running_val_loss += loss.item()

                preds = torch.sigmoid(outputs) > 0.5
                total_val += targets.size(0)
                correct_val += (preds == targets).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_acc = correct_val / total_val
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        scheduler.step(epoch_val_loss)

        # --- Early Stopping ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            break

    # Load the best model state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_loss, val_loss, train_acc, val_acc

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

