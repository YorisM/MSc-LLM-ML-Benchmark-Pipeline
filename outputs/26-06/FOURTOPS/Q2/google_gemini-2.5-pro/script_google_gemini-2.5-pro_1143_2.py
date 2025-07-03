
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
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
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
        self.global_scaler = StandardScaler()
        self.particle_scaler = StandardScaler()
        self.is_fit = False

    def _raw_reshape(self, X: torch.Tensor):
        # X shape: [n_events, 92]
        globals_ = X[:, :2]  # -> [n_events, 2]
        particles_ = X[:, 2:].reshape(-1, 18, 5) # -> [n_events, 18, 5]
        return globals_, particles_

    def _feature_engineer_particles(self, particles: torch.Tensor):
        # particles shape: [n_events, 18, 5] (obj_id, E, pT, eta, phi)
        obj_id = particles[..., 0:1]
        E = particles[..., 1:2]
        pT = particles[..., 2:3]
        eta = particles[..., 3:4]
        phi = particles[..., 4:5]

        # Mask for valid particles to avoid division by zero or large sinh values
        valid_mask = (pT > 1e-9).squeeze(-1) # -> [n_events, 18]

        px = torch.zeros_like(pT)
        py = torch.zeros_like(pT)
        pz = torch.zeros_like(pT)

        # Calculate on valid particles only
        px[valid_mask] = pT[valid_mask] * torch.cos(phi[valid_mask])
        py[valid_mask] = pT[valid_mask] * torch.sin(phi[valid_mask])
        # Clamp eta to prevent overflow in sinh
        eta_clamped = torch.clamp(eta[valid_mask], -5.0, 5.0)
        pz[valid_mask] = pT[valid_mask] * torch.sinh(eta_clamped)

        # New feature set: (obj_id, E, px, py, pz)
        engineered_particles = torch.cat([obj_id, E, px, py, pz], dim=-1) # -> [n_events, 18, 5]
        return engineered_particles, valid_mask

    def fit(self, X, y=None):
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        globals_, particles_raw = self._raw_reshape(X_tensor)

        # Fit global scaler
        self.global_scaler.fit(globals_.numpy())

        # Feature engineer particles and get mask
        particles_eng, mask = self._feature_engineer_particles(particles_raw)

        # Fit particle scaler only on valid (non-padded) particles
        valid_particles = particles_eng[mask]
        if valid_particles.shape[0] > 0:
            self.particle_scaler.fit(valid_particles.numpy())

        self.is_fit = True
        return self

    def transform(self, X):
        if not self.is_fit:
            raise RuntimeError("Preprocessor must be fitted before transforming!")

        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        globals_, particles_raw = self._raw_reshape(X_tensor)

        # Transform global features (numpy -> numpy -> tensor)
        globals_scaled = self.global_scaler.transform(globals_.numpy())
        globals_tensor = torch.from_numpy(globals_scaled).float()

        # Feature engineer particles
        particles_eng, mask = self._feature_engineer_particles(particles_raw) # mask shape [n_events, 18]

        # Transform particle features
        n_events, n_particles, n_features = particles_eng.shape
        # Reshape for scaler: [n_events * 18, 5]
        particles_flat = particles_eng.reshape(-1, n_features) 

        # Scale (numpy -> numpy -> tensor)
        particles_scaled_flat = self.particle_scaler.transform(particles_flat.numpy())
        particles_scaled = torch.from_numpy(particles_scaled_flat).float().reshape(n_events, n_particles, n_features)

        # Apply mask to zero out padding after scaling
        particles_scaled[~mask] = 0.0

        # Return tuple of tensors
        return (globals_tensor, particles_scaled, mask)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object is a tuple from the preprocessor: (globals, particles, mask)
        _, sample_particles, _ = sample_object

        global_dim = 2 # From problem spec
        particle_feature_dim = sample_particles.shape[-1] # From preprocessor

        # Hyperparameters
        self.d_model = 128
        nhead = 8
        num_encoder_layers = 6
        dim_feedforward = 512
        dropout = 0.1

        # Particle input projection
        self.particle_embedding = nn.Linear(particle_feature_dim, self.d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model + global_dim),
            nn.Linear(self.d_model + global_dim, 256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1)
        )

    def forward(self, *data):
        globals_features, particle_features, mask = data
        # globals: [B, 2], particles: [B, 18, 5], mask: [B, 18]

        # Project particle features into d_model dimension
        particle_embeddings = self.particle_embedding(particle_features) # -> [B, 18, d_model]

        # Prepare mask for transformer: it wants True for padding
        padding_mask = ~mask # -> [B, 18]

        # Pass through transformer
        # transformer_output: [B, 18, d_model]
        transformer_output = self.transformer_encoder(
            particle_embeddings,
            src_key_padding_mask=padding_mask
        ) 

        # Aggregate particle information (masked mean pooling)
        # Zero out the padded values to not affect the sum
        transformer_output = transformer_output.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        # Sum over the particle dimension
        summed_output = torch.sum(transformer_output, dim=1) # -> [B, d_model]
        # Count number of non-padded particles per event
        n_particles = mask.sum(dim=1, keepdim=True) # -> [B, 1]
        # .clamp(min=1) to avoid division by zero for events with no particles
        mean_output = summed_output / n_particles.clamp(min=1) # -> [B, d_model]

        # Concatenate aggregated particle features with global features
        combined_features = torch.cat([globals_features, mean_output], dim=1) # -> [B, 2 + d_model]

        # Final classification
        logits = self.classifier(combined_features).squeeze(-1) # -> [B]

        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 40
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS 
    # Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    # Must return trained_model, train_loss, val_loss, train_acc, val_acc
    # Implement early-stopping.
    # Forward signature must match.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 7
    best_model_state = None

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_data, targets in train_loader:
            data_dev = [d.to(device) for d in batch_data]
            targets_dev = targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(*data_dev)
            loss = criterion(outputs, targets_dev)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).long()
            total_train += targets.size(0)
            correct_train += (predicted == targets_dev.long()).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # Validation loop
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_data, targets in val_loader:
                data_dev = [d.to(device) for d in batch_data]
                targets_dev = targets.to(device).float()

                outputs = model(*data_dev)
                loss = criterion(outputs, targets_dev)

                running_val_loss += loss.item() * targets.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).long()
                total_val += targets.size(0)
                correct_val += (predicted == targets_dev.long()).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        # Scheduler Step
        scheduler.step(epoch_val_loss)

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # A more robust way to copy state_dict
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_loss, val_loss, train_acc, val_acc

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

