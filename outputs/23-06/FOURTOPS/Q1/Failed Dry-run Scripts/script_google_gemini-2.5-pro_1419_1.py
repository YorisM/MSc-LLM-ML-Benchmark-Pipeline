
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
        # Initialize state for storing scaling parameters
        self.global_mean = None
        self.global_std = None
        self.object_mean = None
        self.object_std = None

    def _raw_reshape(self, X):
        # Reshape the flat input tensor into global and object features
        X_torch = torch.as_tensor(X, dtype=torch.float32)
        global_feats = X_torch[:, :2]                     # Shape: [B, 2]
        object_feats = X_torch[:, 2:].reshape(-1, 18, 5)  # Shape: [B, 18, 5]
        return global_feats, object_feats

    def _feature_engineer(self, global_feats, object_feats):
        # Global features: (ET_miss, phi_miss) -> (METx, METy)
        et_miss = global_feats[:, 0:1]
        phi_miss = global_feats[:, 1:2]
        met_x = et_miss * torch.cos(phi_miss)
        met_y = et_miss * torch.sin(phi_miss)
        engineered_global = torch.cat([met_x, met_y], dim=1)  # Shape: [B, 2]

        # Object features
        obj_id = object_feats[..., 0:1]  # Shape: [B, 18, 1]
        mask = (obj_id != 0).float()  # Shape: [B, 18, 1] padding mask

        E = object_feats[..., 1:2]
        pT = object_feats[..., 2:3]
        eta = object_feats[..., 3:4]
        phi = object_feats[..., 4:5]

        # Derived kinematic features
        log_E = torch.log(E + 1.0)
        log_pT = torch.log(pT + 1.0)

        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)

        # Mass calculation
        p_sq = px**2 + py**2 + pz**2
        m_sq = E**2 - p_sq
        m = torch.sqrt(torch.relu(m_sq)) # ensure non-negative mass^2
        log_m = torch.log(m + 1.0)

        # Concatenate all engineered features for objects
        cont_features = [log_E, log_pT, eta, px, py, pz, log_m]
        engineered_objects = torch.cat([obj_id] + cont_features, dim=-1) # Shape: [B, 18, 8]

        # Apply mask to continuous features to avoid affecting stats of padded objects
        engineered_objects[:, :, 1:] *= mask

        return engineered_global, engineered_objects, mask.squeeze(-1) # return mask as [B, 18]

    def fit(self, X, y=None):
        global_feats, object_feats = self._raw_reshape(X)
        engineered_global, engineered_objects, mask = self._feature_engineer(global_feats, object_feats)

        # Calculate and store mean/std for global features
        self.global_mean = engineered_global.mean(dim=0)
        self.global_std = engineered_global.std(dim=0)
        self.global_std[self.global_std < 1e-6] = 1.0 # Prevent division by zero

        # Calculate and store mean/std for continuous object features, respecting padding
        cont_objects = engineered_objects[:, :, 1:] # Exclude obj_id
        mask_expanded = mask.unsqueeze(-1).expand_as(cont_objects)
        num_real_objects = mask.sum()

        if num_real_objects > 0:
            total_sum = (cont_objects * mask_expanded).sum(dim=(0, 1))
            self.object_mean = total_sum / num_real_objects

            sum_sq_diff = (((cont_objects - self.object_mean) * mask_expanded)**2).sum(dim=(0, 1))
            self.object_std = torch.sqrt(sum_sq_diff / num_real_objects)
            self.object_std[self.object_std < 1e-6] = 1.0
        else: # Fallback for empty batch
            self.object_mean = torch.zeros(cont_objects.shape[-1])
            self.object_std = torch.ones(cont_objects.shape[-1])

        return self

    def transform(self, X):
        global_feats, object_feats = self._raw_reshape(X)
        engineered_global, engineered_objects, mask = self._feature_engineer(global_feats, object_feats)

        # Apply scaling to global features
        device = engineered_global.device
        scaled_global = (engineered_global - self.global_mean.to(device)) / self.global_std.to(device)

        # Apply scaling to continuous object features
        cont_objects = engineered_objects[:, :, 1:]
        scaled_cont_objects = (cont_objects - self.object_mean.to(device)) / self.object_std.to(device)

        # Re-apply mask to ensure padded values are zero after scaling
        scaled_cont_objects *= mask.unsqueeze(-1)

        # Recombine with non-scaled obj_id
        final_objects = torch.cat([engineered_objects[:, :, 0:1], scaled_cont_objects], dim=-1)

        # Return a tuple that will be handled by the DataLoader
        return (scaled_global, final_objects, mask)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, global_dim, object_dim, num_embeddings, embedding_dim=16, latent_dim=128, dropout_rate=0.25):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        # Continuous features are all but the first (obj_id)
        encoder_input_dim = embedding_dim + (object_dim - 1)

        # Object Encoder MLP (phi network in DeepSets)
        self.obj_mlp1 = nn.Linear(encoder_input_dim, latent_dim)
        self.obj_bn1 = nn.BatchNorm1d(latent_dim)
        self.obj_drop1 = nn.Dropout(dropout_rate)
        self.obj_mlp2 = nn.Linear(latent_dim, latent_dim)
        self.obj_bn2 = nn.BatchNorm1d(latent_dim)

        # Classifier MLP (rho network in DeepSets)
        classifier_input_dim = latent_dim + global_dim
        self.clf_mlp1 = nn.Linear(classifier_input_dim, 128)
        self.clf_bn1 = nn.BatchNorm1d(128)
        self.clf_drop1 = nn.Dropout(dropout_rate)
        self.clf_mlp2 = nn.Linear(128, 64)
        self.clf_bn2 = nn.BatchNorm1d(64)
        self.clf_drop2 = nn.Dropout(dropout_rate)
        self.clf_out = nn.Linear(64, 1)

    def forward(self, *data):
        # Unpack the tuple of tensors. Input comes from our custom preprocessor.
        global_feats, object_feats, object_mask = data[0], data[1], data[2]
        # Shapes: global_feats [B, G], object_feats [B, N_obj, F_obj], object_mask [B, N_obj]
        B, N_obj, _ = object_feats.shape

        # --- Object Encoding ---
        obj_ids = object_feats[:, :, 0].long()           # [B, N_obj]
        cont_obj_feats = object_feats[:, :, 1:]          # [B, N_obj, F_obj - 1]

        embedded_ids = self.embedding(obj_ids)           # [B, N_obj, E_dim]

        encoder_input = torch.cat([embedded_ids, cont_obj_feats], dim=-1) # [B, N_obj, E_dim + F_obj-1]

        # Reshape for BatchNorm1d: (B, N_obj, Feat) -> (B * N_obj, Feat)
        x_obj = encoder_input.view(B * N_obj, -1)

        x_obj = self.obj_mlp1(x_obj)
        x_obj = self.obj_bn1(torch.relu(x_obj))
        x_obj = self.obj_drop1(x_obj)
        x_obj = self.obj_mlp2(x_obj)
        x_obj = self.obj_bn2(torch.relu(x_obj))

        object_latents = x_obj.view(B, N_obj, -1)     # [B, N_obj, latent_dim]

        # --- Aggregation ---
        # Mask out padded objects before summing for permutation-invariance
        object_latents = object_latents * object_mask.unsqueeze(-1)
        aggregated_objects = torch.sum(object_latents, dim=1) # [B, latent_dim]

        # --- Classification ---
        classifier_input = torch.cat([global_feats, aggregated_objects], dim=1) # [B, latent_dim + G]

        x_clf = self.clf_mlp1(classifier_input)
        x_clf = self.clf_bn1(torch.relu(x_clf))
        x_clf = self.clf_drop1(x_clf)
        x_clf = self.clf_mlp2(x_clf)
        x_clf = self.clf_bn2(torch.relu(x_clf))
        x_clf = self.clf_drop2(x_clf)
        logits = self.clf_out(x_clf)                     # [B, 1]

        return logits.squeeze(-1) # [B]

def make_model(example_object):
    # example_object from the loader is a tuple of features: (global, objects, mask)
    global_dim = example_object[0].shape[1]
    object_dim = example_object[1].shape[2]

    # Hardcode a safe upper bound for number of particle types, as inspecting the full
    # dataset during model creation is not feasible in the harness.
    num_embeddings = 20

    return BinaryClassifier(global_dim, object_dim, num_embeddings)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 35
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss, optimizer, and scheduler setup
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

    # Early stopping parameters
    patience = 7
    patience_counter = 0
    best_val_loss = float('inf')
    best_model_state = None

    # History tracking
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss, running_acc, total_samples = 0.0, 0.0, 0
        for inputs, labels in train_loader:
            inputs = [i.to(device) for i in inputs]
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            running_acc += (preds == labels.long()).float().sum().item()
            total_samples += labels.size(0)

        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = running_acc / total_samples
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # --- Validation Phase ---
        model.eval()
        running_val_loss, running_val_acc, total_val_samples = 0.0, 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = [i.to(device) for i in inputs]
                labels = labels.to(device).float()

                outputs = model(*inputs)
                loss = loss_fn(outputs, labels)

                running_val_loss += loss.item() * labels.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).long()
                running_val_acc += (preds == labels.long()).float().sum().item()
                total_val_samples += labels.size(0)

        epoch_val_loss = running_val_loss / total_val_samples
        epoch_val_acc = running_val_acc / total_val_samples
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} - "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # --- Scheduler and Early Stopping ---
        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    # Load best model state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)

    trained_model = model.cpu()
    return trained_model, train_loss, val_loss, train_acc, val_acc

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

