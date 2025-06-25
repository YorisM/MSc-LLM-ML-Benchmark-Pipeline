
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
import torch.nn.functional as F
import numpy as np


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
        # We will scale global features and particle-level features separately.
        # This is more robust than scaling all 92 features together.
        self.global_scaler = StandardScaler()
        self.particle_scaler = StandardScaler()

    def _engineer_features(self, X: torch.Tensor):
        # Reshape the flat input into global and particle-level features.
        # X shape: [N, 92]
        global_features_raw = X[:, :2]              # shape: [N, 2]
        particles_raw = X[:, 2:].view(-1, 18, 5)     # shape: [N, 18, 5]

        # --- Engineer Global Features ---
        E_T_miss = global_features_raw[:, 0]
        phi_miss = global_features_raw[:, 1]

        # log1p is more stable than log for values near 0.
        # Representing angles as (cos, sin) removes circularity.
        log_E_T_miss = torch.log1p(E_T_miss)
        cos_phi_miss = torch.cos(phi_miss)
        sin_phi_miss = torch.sin(phi_miss)

        global_features = torch.stack([log_E_T_miss, cos_phi_miss, sin_phi_miss], dim=1) # shape: [N, 3]

        # --- Engineer Particle Features ---
        obj_id = particles_raw[:, :, 0] # shape: [N, 18]
        E = particles_raw[:, :, 1]
        p_T = particles_raw[:, :, 2]
        eta = particles_raw[:, :, 3]
        phi = particles_raw[:, :, 4]

        # Create a mask for real (non-padded) particles. We assume p_T>0 for real particles.
        mask = p_T > 1e-6 # shape: [N, 18]

        # Use log-transformed kinematics and Cartesian coordinates for angles.
        log_E = torch.log1p(E)
        log_pT = torch.log1p(p_T)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        # We treat obj_id as a continuous feature to be scaled. This is simpler than
        # using an embedding layer and works well in practice.
        particle_features = torch.stack([
            obj_id, log_E, log_pT, eta, cos_phi, sin_phi
        ], dim=2) # shape: [N, 18, 6]

        # Zero out features of padded particles to ensure they don't affect scaling.
        particle_features[~mask] = 0.0

        return global_features, particle_features, mask

    def fit(self, X, y=None):
        # This method learns the scaling parameters from the training data.
        global_features, particle_features, mask = self._engineer_features(X)

        # Fit scaler on global features.
        self.global_scaler.fit(global_features.numpy())

        # Fit particle scaler only on the *active* particles to avoid statistical
        # bias from the zero-padded entries.
        active_particle_features = particle_features[mask]
        if active_particle_features.shape[0] > 0:
            self.particle_scaler.fit(active_particle_features.numpy())

        return self

    def transform(self, X):
        # This method applies the learned transformations to the data.
        global_features, particle_features, mask = self._engineer_features(X)

        # Apply global scaler.
        global_features_scaled = torch.from_numpy(
            self.global_scaler.transform(global_features.numpy())
        ).float()

        # Apply particle scaler only to active particles.
        particle_features_scaled = torch.zeros_like(particle_features)

        active_mask_np = mask.numpy()
        if np.any(active_mask_np):
            # Select only active particles for scaling
            active_features = particle_features[mask]
            scaled_active = self.particle_scaler.transform(active_features.numpy())
            # Place the scaled features back into the tensor at the correct positions
            particle_features_scaled[mask] = torch.from_numpy(scaled_active).float()

        # The output is a tuple of tensors, which will be handled by the custom dataset
        # and data loader provided in the harness.
        # This structured output is ideal for models that distinguish between
        # global and per-particle information, like Transformers or GNNs.
        return (global_features_scaled, particle_features_scaled, mask)

    def fit_transform(self, X, y=None):
        # Convenience method to fit and transform in one step.
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Infer input feature dimensions from a sample batch.
        # The sample_object is a tuple: (global_features, particle_features, mask)
        global_feature_dim = sample_object[0].shape[-1]
        particle_feature_dim = sample_object[1].shape[-1]

        # Model hyperparameters, chosen for a high-capacity Transformer.
        d_model = 128          # Core embedding dimension
        nhead = 8              # Number of attention heads (must be divisor of d_model)
        num_encoder_layers = 6 # Depth of the model
        dim_feedforward = 512  # Hidden dimension in the MLP inside Transformer blocks
        dropout = 0.1

        # A linear layer to project input particle features into the model's dimension.
        self.particle_embed = nn.Linear(particle_feature_dim, d_model)

        # The core of the model: a stack of Transformer Encoder layers.
        # `batch_first=True` is crucial for using [Batch, Sequence, Features] tensors.
        # `GELU` is a common activation function in modern transformers.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True,
            activation=F.gelu
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=num_encoder_layers
        )

        # A small MLP to process global event-level features.
        self.global_mlp = nn.Sequential(
            nn.Linear(global_feature_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )

        # The final classifier head. Takes the combined particle and global information
        # and outputs a single logit for classification.
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2), # Input is concatenated particle + global embeddings
            nn.Linear(d_model * 2, dim_feedforward),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(dim_feedforward // 2, 1)
        )

    def forward(self, *data):
        globals_data, particles_data, mask_data = data

        # The mask from our preprocessor is True for VALID particles.
        # PyTorch's `src_key_padding_mask` expects True for PADDED elements.
        # Therefore, we must invert the mask.
        padding_mask = ~mask_data  # shape: [B, 18]

        # 1. Project particle features to the model's dimension.
        particle_embeddings = self.particle_embed(particles_data) # shape: [B, 18, d_model]

        # 2. Pass through the Transformer encoder.
        # The mask ensures that attention is not paid to padded elements.
        transformer_output = self.transformer_encoder(
            src=particle_embeddings, 
            src_key_padding_mask=padding_mask
        ) # shape: [B, 18, d_model]

        # 3. Aggregate particle information via masked average pooling.
        # This creates a single vector representing all particles in the event.
        # First, zero out the outputs for padded elements to not affect the sum.
        transformer_output = transformer_output * mask_data.unsqueeze(-1)
        # Sum over the sequence dimension and divide by the number of real particles.
        num_particles = mask_data.sum(dim=1, keepdim=True).clamp(min=1) # Avoid division by zero
        aggregated_particles = transformer_output.sum(dim=1) / num_particles # shape: [B, d_model]

        # 4. Process global features through its own MLP.
        global_embedding = self.global_mlp(globals_data) # shape: [B, d_model]

        # 5. Concatenate the aggregated particle and global representations.
        combined_features = torch.cat([aggregated_particles, global_embedding], dim=1) # shape: [B, d_model * 2]

        # 6. Pass through the final classifier to get logits.
        logits = self.classifier(combined_features) # shape: [B, 1]

        # Return a 1D tensor of logits for `BCEWithLogitsLoss`.
        return logits.squeeze(-1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 40   # Set a generous number of epochs; early stopping will handle convergence.
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS 
    # Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    # Must return trained_model, train_loss, val_loss, train_acc, val_acc
    # Implement early-stopping.
    # Forward signature must match.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss, Optimizer, and Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

    # Early stopping parameters
    patience = 7
    best_val_loss = float('inf')
    epochs_no_improve = 0
    # Store the best model state to restore it at the end
    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # History tracking
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0

        for data, labels in train_loader:
            inputs = [d.to(device) for d in data]
            labels = labels.to(device).float()

            optimizer.zero_grad(set_to_none=True)

            outputs = model(*inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct_train += (preds == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / len(train_loader.dataset)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0

        with torch.no_grad():
            for data, labels in val_loader:
                inputs = [d.to(device) for d in data]
                labels = labels.to(device).float()

                outputs = model(*inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * labels.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                correct_val += (preds == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / len(val_loader.dataset)
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Learning rate scheduler step
        scheduler.step(epoch_val_loss)

        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # Load the best performing model state before returning
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

