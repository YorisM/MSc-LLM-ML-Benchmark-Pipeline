
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
import copy
from sklearn.metrics import roc_auc_score # For optional logging, not required by harness
import torch.nn.functional as F

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
        # Store normalization statistics
        self.global_mean = None
        self.global_std = None
        self.particle_mean = None
        self.particle_std = None
        # Number of particle features after engineering
        self.particle_feature_dim = 7

    def _feature_engineer(self, X: torch.Tensor):
        # X: [N, 92]

        # --- Global Features ---
        met = X[:, 0:1]  # [N, 1]
        met_phi = X[:, 1:2]  # [N, 1]
        # Convert from polar to Cartesian coordinates for better geometric representation
        met_x = met * torch.cos(met_phi) # [N, 1]
        met_y = met * torch.sin(met_phi) # [N, 1]
        globals_eng = torch.cat([met_x, met_y], dim=1)  # [N, 2]

        # --- Particle Features ---
        particles_raw = X[:, 2:].reshape(-1, 18, 5)  # [N, 18, 5]

        # Create mask based on original pT > 0, to identify real particles vs zero-padding
        # pT is the 3rd feature in the 5-tuple (index 2)
        pt_original = particles_raw[:, :, 2]  # [N, 18]
        mask = pt_original > 1e-9  # Use a small epsilon for float comparison. [N, 18]

        # Deconstruct particle 5-vector for feature engineering
        obj_id = particles_raw[:, :, 0:1]  # [N, 18, 1]
        E = particles_raw[:, :, 1:2]       # [N, 18, 1]
        pT = particles_raw[:, :, 2:3]      # [N, 18, 1]
        eta = particles_raw[:, :, 3:4]     # [N, 18, 1]
        phi = particles_raw[:, :, 4:5]     # [N, 18, 1]

        # Engineer new features: Cartesian coordinates and log-transformed energies
        px = pT * torch.cos(phi)  # [N, 18, 1]
        py = pT * torch.sin(phi)  # [N, 18, 1]
        pz = pT * torch.sinh(eta) # [N, 18, 1]

        # Using log1p on relu'd inputs handles large dynamic range and ensures non-negativity
        log_E = torch.log1p(torch.relu(E))
        log_pT = torch.log1p(torch.relu(pT))

        particles_eng = torch.cat([
            obj_id,
            log_E,
            log_pT,
            eta,
            px,
            py,
            pz
        ], dim=-1)  # [N, 18, 7]

        assert particles_eng.shape[-1] == self.particle_feature_dim

        return globals_eng, particles_eng, mask

    def fit(self, X, y=None):
        globals_eng, particles_eng, mask = self._feature_engineer(X)

        # Calculate and store statistics for normalization
        # For global features
        self.global_mean = globals_eng.mean(dim=0)
        self.global_std = globals_eng.std(dim=0)

        # For particle features, using only valid (non-padded) particles
        valid_particles = particles_eng[mask]  # [num_valid_particles, 7]
        self.particle_mean = valid_particles.mean(dim=0)
        self.particle_std = valid_particles.std(dim=0)
        # Prevent division by zero if a feature has no variance
        self.particle_std[self.particle_std < 1e-9] = 1.0

        return self

    def transform(self, X):
        globals_eng, particles_eng, mask = self._feature_engineer(X)

        # Apply normalization using stored statistics
        globals_scaled = (globals_eng - self.global_mean) / self.global_std
        particles_scaled = (particles_eng - self.particle_mean) / self.particle_std

        # Re-apply mask to zero out padded entries after normalization
        particles_scaled *= mask.unsqueeze(-1)  # [N, 18, 7]

        # PyTorch transformer mask: True for positions to be MASKED (ignored)
        # Our mask is True for valid particles, so we must invert it.
        padding_mask = ~mask  # [N, 18]

        # Return a tuple of tensors to be consumed by the model
        return (globals_scaled, particles_scaled, padding_mask)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Infer input shapes from a sample object provided by the preprocessor
        global_sample, particle_sample, _ = sample_object
        global_dim = global_sample.shape[-1]    # Expected: 2
        particle_dim = particle_sample.shape[-1]  # Expected: 7 (from preprocessor)

        # Model hyperparameters
        embed_dim = 128
        n_heads = 8
        n_layers = 4
        ff_dim = 256
        dropout = 0.1

        # Particle feature encoder: an MLP to project particle features into an embedding space
        self.particle_encoder = nn.Sequential(
            nn.Linear(particle_dim, embed_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
        )

        # Standard Transformer Encoder to find relationships between particles
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,  # Input format is [batch, seq, feature]
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(embed_dim)
        )

        # Classifier head: an MLP to process aggregated features for final classification
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + global_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, global_feats, particle_feats, padding_mask):
        # global_feats: [N, global_dim], particle_feats: [N, 18, particle_dim], padding_mask: [N, 18]

        # 1. Encode particle features into a higher-dimensional space
        particle_embeddings = self.particle_encoder(particle_feats)  # [N, 18, embed_dim]

        # 2. Pass through transformer. The padding_mask ensures attention ignores padded particles.
        transformer_output = self.transformer_encoder(
            src=particle_embeddings,
            src_key_padding_mask=padding_mask
        )  # [N, 18, embed_dim]

        # 3. Aggregate particle features using masked mean pooling for a permutation-invariant representation
        valid_mask = ~padding_mask  # Invert mask: True for valid particles. [N, 18]
        masked_output = transformer_output * valid_mask.unsqueeze(-1)  # Zero out padded outputs. [N, 18, embed_dim]

        num_valid_particles = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [N, 1]
        summed_output = masked_output.sum(dim=1)  # [N, embed_dim]
        aggregated_particles = summed_output / num_valid_particles  # [N, embed_dim]

        # 4. Concatenate aggregated particle vector with global features
        combined_features = torch.cat([aggregated_particles, global_feats], dim=1)  # [N, embed_dim + global_dim]

        # 5. Get final classification score (logits) from the classifier head
        logits = self.classifier(combined_features)  # [N, 1]

        return logits.squeeze(-1)  # Return shape [N] for BCEWithLogitsLoss

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30   # Increased epochs for convergence, with early stopping
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3, verbose=False)

    # Early stopping settings
    patience = 7
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_train_loss, train_correct, train_total = 0, 0, 0
        for data_tuple, targets in train_loader:
            data_tuple = tuple(d.to(device) for d in data_tuple)
            targets = targets.to(device).float()

            optimizer.zero_grad(set_to_none=True)
            outputs = model(*data_tuple)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_train_loss += loss.item() * targets.size(0)
            preds = (torch.sigmoid(outputs) > 0.5)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        epoch_train_loss = running_train_loss / train_total
        epoch_train_acc = train_correct / train_total
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # --- Validation Phase ---
        model.eval()
        running_val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for data_tuple, targets in val_loader:
                data_tuple = tuple(d.to(device) for d in data_tuple)
                targets = targets.to(device)

                outputs = model(*data_tuple)
                loss = criterion(outputs, targets.float())

                running_val_loss += loss.item() * targets.size(0)
                preds = (torch.sigmoid(outputs) > 0.5)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        epoch_val_loss = running_val_loss / val_total
        epoch_val_acc = val_correct / val_total
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        scheduler.step(epoch_val_loss)

        # --- Early Stopping Check ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

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

