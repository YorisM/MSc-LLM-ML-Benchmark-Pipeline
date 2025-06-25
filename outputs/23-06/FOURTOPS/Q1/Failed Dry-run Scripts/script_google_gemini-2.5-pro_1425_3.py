
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
import torch.optim as optim

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

    # REQUIREMENTS
    # IMPORTANT: All state must be picklable with the std-lib pickle module.
    # May allocate NumPy arrays or Torch tensors internally, but:
    # transform() must be deterministic.
    # Store only derived parameters needed for transform i.e. do not store the raw data
    # itself in the preprocessor object.

    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.particle_mean = None
        self.particle_std = None

    def _reshape_and_feature_eng(self, X: torch.Tensor):
        """Reshapes flat data, performs feature engineering, and creates a mask."""
        # X shape: [N, 92]
        X_global = X[:, :2]  # [N, 2]
        X_particles_flat = X[:, 2:]  # [N, 90]

        # Reshape particles
        # [N, 18, 5]
        X_particles = X_particles_flat.reshape(-1, 18, 5)

        # Create mask from obj_id (feature 0 of each particle). obj_id == 0 indicates padding.
        mask = X_particles[:, :, 0] != 0  # [N, 18]

        # --- Particle feature engineering ---
        # Original features: obj_id, E, pT, eta, phi
        E = X_particles[:, :, 1]
        pT = X_particles[:, :, 2]
        eta = X_particles[:, :, 3]
        phi = X_particles[:, :, 4]

        # Add a small epsilon for numerical stability in log
        pT_safe = pT + 1e-8
        E_safe = E + 1e-8

        # Cartesian coordinates
        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)

        # Log-transformed kinematics (often helps with large dynamic ranges)
        log_pT = torch.log(pT_safe)
        log_E = torch.log(E_safe)

        # New particle features: [log_E, log_pT, eta, phi, px, py, pz]
        X_particles_eng = torch.stack([
            log_E, log_pT, eta, phi, px, py, pz
        ], dim=-1)  # [N, 18, 7]

        # --- Global feature engineering ---
        MET, MET_phi = X_global[:, 0], X_global[:, 1]
        MET_safe = MET + 1e-8

        MET_x = MET * torch.cos(MET_phi)
        MET_y = MET * torch.sin(MET_phi)
        log_MET = torch.log(MET_safe)

        # New global features: [log_MET, MET_x, MET_y]
        X_global_eng = torch.stack([
            log_MET, MET_x, MET_y
        ], dim=-1)  # [N, 3]

        return X_global_eng, X_particles_eng, mask

    def fit(self, X, y=None):
        """Fits the preprocessor by calculating mean and std for scaling."""
        X_global, X_particles, mask = self._reshape_and_feature_eng(X)

        self.global_mean = X_global.mean(dim=0, keepdim=True)
        self.global_std = X_global.std(dim=0, keepdim=True)

        # Use only real particles (not padding) for statistics
        if mask.any():
            real_particles = X_particles[mask]
            self.particle_mean = real_particles.mean(dim=0, keepdim=True)
            self.particle_std = real_particles.std(dim=0, keepdim=True)
        else: # Handle case with no particles in the batch
            self.particle_mean = torch.zeros(1, X_particles.shape[-1], device=X.device)
            self.particle_std = torch.ones(1, X_particles.shape[-1], device=X.device)


        # Prevent division by zero
        self.particle_std[self.particle_std < 1e-6] = 1.0
        self.global_std[self.global_std < 1e-6] = 1.0

        return self

    def transform(self, X):
        """Applies the learned transformation to the data."""
        X_global, X_particles, mask = self._reshape_and_feature_eng(X)

        # Standardize global features
        X_global_scaled = (X_global - self.global_mean) / self.global_std

        # Standardize particle features (only the real ones)
        X_particles_scaled = torch.zeros_like(X_particles)
        p_mask_expanded = mask.unsqueeze(-1)  # [N, 18, 1]

        # Apply scaling transformation only to locations indicated by the mask
        scaled_values = (X_particles - self.particle_mean) / self.particle_std
        X_particles_scaled = torch.where(p_mask_expanded, scaled_values, X_particles_scaled)

        # Returns a tuple of tensors to be consumed by the model
        return (X_global_scaled, X_particles_scaled, mask)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object is a tuple (global_feats, particle_feats, mask)
        global_sample, particle_sample, _ = sample_object

        # Dynamically get feature sizes from the preprocessed sample
        n_particle_features = particle_sample.shape[-1] # Shape: [B, 18, 7] -> 7
        n_global_features = global_sample.shape[-1]     # Shape: [B, 3] -> 3

        # Hyperparameters for the Transformer model
        d_model = 128  # Embedding dimension
        n_head = 8     # Number of attention heads
        n_layers = 4   # Number of transformer layers
        dim_feedforward = 256 # Dimension of the feedforward network in transformer
        dropout = 0.1

        # Particle processing branch
        self.particle_embed = nn.Linear(n_particle_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Crucial for [B, S, E] shaped tensors
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Global feature processing branch
        self.global_embed = nn.Sequential(
            nn.Linear(n_global_features, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32)
        )

        # Classifier head combining particle and global information
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 32, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, *data):
        X_global, X_particles, mask = data
        # X_global shape: [B, 3]
        # X_particles shape: [B, 18, 7]
        # mask shape: [B, 18] (True for real particles, False for padding)

        # The transformer mask requires True for positions to be IGNORED.
        attn_mask = ~mask  # [B, 18]

        # 1. Process particles
        particle_embeddings = self.particle_embed(X_particles) # [B, 18, 128]

        # 2. Apply transformer encoder with padding mask
        transformer_output = self.transformer_encoder(
            particle_embeddings, src_key_padding_mask=attn_mask
        ) # [B, 18, 128]

        # 3. Aggregate particle representations using masked average pooling
        mask_expanded = mask.unsqueeze(-1).expand_as(transformer_output) # [B, 18, 128]
        transformer_output_masked = transformer_output * mask_expanded

        summed_output = transformer_output_masked.sum(dim=1) # [B, 128]

        num_particles = mask.sum(dim=1, keepdim=True).float() # [B, 1]
        num_particles = torch.max(num_particles, torch.ones_like(num_particles))

        mean_pooled_output = summed_output / num_particles # [B, 128]

        # 4. Process global features
        global_processed = self.global_embed(X_global) # [B, 32]

        # 5. Combine and classify
        combined_features = torch.cat([mean_pooled_output, global_processed], dim=1) # [B, 160]
        logits = self.classifier(combined_features) # [B, 1]

        return logits.squeeze(-1) # Return shape [B]

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 35 
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer, Loss, and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=4, verbose=False)

    # Early stopping parameters
    best_val_loss = float('inf')
    best_model_state = None
    patience = 8
    patience_counter = 0

    # History tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for data, labels in train_loader:
            # Move data tuple and labels to the correct device
            data = [d.to(device) for d in data]
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(*data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(outputs) > 0.5)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data = [d.to(device) for d in data]
                labels = labels.to(device).float()

                outputs = model(*data)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * labels.size(0)
                preds = (torch.sigmoid(outputs) > 0.5)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        # --- Epoch Statistics ---
        avg_train_loss = running_train_loss / max(1, total_train)
        train_accuracy = correct_train / max(1, total_train)
        train_losses.append(avg_train_loss)
        train_accs.append(train_accuracy)

        avg_val_loss = running_val_loss / max(1, total_val)
        val_accuracy = correct_val / max(1, total_val)
        val_losses.append(avg_val_loss)
        val_accs.append(val_accuracy)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # print(f"Early stopping at epoch {epoch+1} as validation loss did not improve for {patience} epochs.")
                break

    # Load the best model weights before returning
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Return the best model on CPU for harness compatibility
    model.to("cpu")

    return model, train_losses, val_losses, train_accs, val_accs

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

