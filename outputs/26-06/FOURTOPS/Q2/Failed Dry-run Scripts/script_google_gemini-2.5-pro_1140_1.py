
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
from sklearn.metrics import roc_auc_score
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import collections

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

    def _raw_reshape(self, X: torch.Tensor):
        # Reshapes the flat input tensor into global and particle features.
        global_features = X[:, :2]  # Shape: [N, 2]
        # Shape: [N, 90] -> [N, 18, 5]
        particle_features = X[:, 2:].reshape(-1, 18, 5)
        return global_features, particle_features

    def fit(self, X, y=None):
        # Fit scalers to the training data.
        # X is a numpy array here.
        X_tensor = torch.from_numpy(X).float()
        global_feats, particles = self._raw_reshape(X_tensor)

        # Fit scaler for global features.
        self.global_scaler.fit(global_feats.numpy())

        # Create a mask for real (non-padded) particles.
        mask = particles[:, :, 0] != 0  # Shape: [N, 18]

        # Isolate continuous features for particles (E, pT, eta, phi).
        continuous_particles = particles[:, :, 1:]  # Shape: [N, 18, 4]

        # Fit scaler only on the active particles to avoid distorting stats with padding.
        active_continuous_particles = continuous_particles[mask] # Shape: [n_active, 4]
        self.particle_scaler.fit(active_continuous_particles.numpy())

        return self

    def transform(self, X):
        # Apply the fitted transformations.
        # X is expected to be a torch tensor here from the harness.
        if not torch.is_tensor(X):
            X = torch.from_numpy(X).float()

        device = X.device

        global_feats, particles = self._raw_reshape(X)

        # Transform global features.
        scaled_globals_np = self.global_scaler.transform(global_feats.cpu().numpy())
        scaled_globals = torch.from_numpy(scaled_globals_np).float().to(device)

        # Separate particle features.
        obj_ids = particles[:, :, 0].long() # Shape: [N, 18]
        continuous_particles = particles[:, :, 1:] # Shape: [N, 18, 4]
        mask = (obj_ids != 0).bool() # Shape: [N, 18]

        # Transform particle features.
        # Reshape for scaler, apply, then reshape back.
        n_events, n_steps, n_features = continuous_particles.shape
        # Shape: [N * 18, 4]
        reshaped_particles = continuous_particles.reshape(-1, n_features).cpu().numpy()
        scaled_reshaped_particles = self.particle_scaler.transform(reshaped_particles)
        scaled_particles = torch.from_numpy(scaled_reshaped_particles).float().reshape(n_events, n_steps, n_features).to(device)

        # Zero out the padded values after scaling.
        scaled_particles = scaled_particles * mask.unsqueeze(-1)

        # Return a tuple of tensors. The default DataLoader and PairDataset can handle this.
        return (scaled_globals, obj_ids, scaled_particles, mask)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        if torch.is_tensor(X):
            return self.transform(X)
        else: # Harness flow calls fit with numpy, so transform needs to handle it.
            return self.transform(torch.from_numpy(X).float())


def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Infer dimensions from the sample object provided by the harness
        global_dim = sample_object[0].shape[1]
        particle_continuous_dim = sample_object[2].shape[2]

        # Use a safe, hardcoded number of object types.
        # padding_idx=0 will keep the embedding for padded objects as zero.
        num_obj_types = 16 

        # Model hyperparameters
        d_model = 128
        nhead = 8
        num_encoder_layers = 4
        dim_feedforward = 512
        dropout = 0.1

        # Particle feature embedding
        # Project continuous features to d_model
        self.continuous_particle_proj = nn.Linear(particle_continuous_dim, d_model)
        # Embed categorical object IDs to d_model
        self.obj_id_embedding = nn.Embedding(
            num_embeddings=num_obj_types, embedding_dim=d_model, padding_idx=0
        )

        # Transformer Encoder to process particle sequences
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

        # Global feature processing
        self.global_feat_proj = nn.Sequential(
            nn.Linear(global_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU()
        )

        # Final classifier head
        self.classifier_head = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 1)
        )

    def forward(self, global_feats, obj_ids, particle_feats, mask):
        # Input shapes:
        # global_feats:   [N, 2]
        # obj_ids:        [N, 18]
        # particle_feats: [N, 18, 4]
        # mask:           [N, 18] (boolean, True for valid particles)

        # 1. Create combined particle embeddings
        # Project continuous features and add to object ID embeddings
        continuous_emb = self.continuous_particle_proj(particle_feats) #-> [N, 18, d_model]
        id_emb = self.obj_id_embedding(obj_ids) #-> [N, 18, d_model]
        particle_embeddings = continuous_emb + id_emb #-> [N, 18, d_model]

        # 2. Process particles with Transformer
        # Transformer expects padding mask where True indicates a value to be ignored
        padding_mask = ~mask  # Invert mask. #-> [N, 18]
        transformer_out = self.transformer_encoder(
            src=particle_embeddings,
            src_key_padding_mask=padding_mask
        ) #-> [N, 18, d_model]

        # 3. Aggregate particle information (masked mean pooling)
        # Mask out the padding tokens before summing
        transformer_out = transformer_out.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Sum of non-padded item embeddings
        summed_out = transformer_out.sum(dim=1) # -> [N, d_model]

        # Number of actual particles in each event for averaging
        n_particles = mask.sum(dim=1, keepdim=True).clamp(min=1e-9) #-> [N, 1]

        # Calculate the mean
        aggregated_particles = summed_out / n_particles #-> [N, d_model]

        # 4. Process global features
        projected_globals = self.global_feat_proj(global_feats) #-> [N, d_model/2]

        # 5. Combine and classify
        # Concatenate aggregated particle vector and global features vector
        final_representation = torch.cat([
            aggregated_particles,
            projected_globals
        ], dim=1) #-> [N, d_model + d_model/2]

        # Pass through the final classification head
        logits = self.classifier_head(final_representation).squeeze(-1) #-> [N]

        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 40
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Optimizer, Loss, and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    # Scheduler will reduce LR if validation AUC stagnates
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Early stopping parameters
    patience = 8
    patience_counter = 0
    best_val_auc = -1.0
    best_model_state = None

    # History tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            data, labels = batch
            # Unpack data and move to device
            global_feats, obj_ids, particle_feats, mask = data
            global_feats = global_feats.to(device)
            obj_ids = obj_ids.to(device)
            particle_feats = particle_feats.to(device)
            mask = mask.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()

            outputs = model(global_feats, obj_ids, particle_feats, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.sigmoid(outputs) >= 0.5
            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in val_loader:
                data, labels = batch
                global_feats, obj_ids, particle_feats, mask = data
                global_feats = global_feats.to(device)
                obj_ids = obj_ids.to(device)
                particle_feats = particle_feats.to(device)
                mask = mask.to(device)
                labels = labels.to(device).float()

                outputs = model(global_feats, obj_ids, particle_feats, mask)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                # Store predictions and labels for AUC calculation
                all_labels.append(labels.cpu().numpy())
                all_preds.append(torch.sigmoid(outputs).cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        # Calculate AUC and accuracy for the whole validation set
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        val_auc = roc_auc_score(all_labels, all_preds)
        val_acc = np.mean((all_preds >= 0.5) == all_labels)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

        # LR Scheduler Step
        scheduler.step(val_auc)

        # Early Stopping Check
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save the best model state
            best_model_state = model.state_dict()
            print(f"New best validation AUC: {best_val_auc:.4f}. Saving model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break

    # Load the best performing model
    if best_model_state:
        model.load_state_dict(best_model_state)

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

