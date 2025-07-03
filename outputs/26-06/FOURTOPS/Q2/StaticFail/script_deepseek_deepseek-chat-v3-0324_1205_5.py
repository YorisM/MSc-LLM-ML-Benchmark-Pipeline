
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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from abc import abstractmethod

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None
        self.max_particles = 18
        self.particle_feature_size = 5
        self.global_feature_size = 2

    def _raw_reshape(self, X):
        batch_size = X.shape[0]
        global_features = X[:, :self.global_feature_size]
        particle_features = X[:, self.global_feature_size:].reshape(batch_size, self.max_particles, self.particle_feature_size)

        # Remove zero-padded particles
        mask = particle_features[:, :, 0] != 0  # obj_id != 0

        # Compute pairwise features
        pairwise_features = []
        for i in range(batch_size):
            active_particles = particle_features[i][mask[i]]
            if len(active_particles) < 2:
                pairwise_features.append(torch.zeros(1, 3))  # dummy features
                continue

            # Compute delta R and invariant mass
            pts = active_particles[:, 2]
            etas = active_particles[:, 3]
            phis = active_particles[:, 4]
            energies = active_particles[:, 1]

            # Delta R matrix
            eta_diff = etas.unsqueeze(1) - etas.unsqueeze(0)
            phi_diff = phis.unsqueeze(1) - phis.unsqueeze(0)
            phi_diff = (phi_diff + np.pi) % (2 * np.pi) - np.pi
            delta_r = torch.sqrt(eta_diff.pow(2) + phi_diff.pow(2))

            # Invariant mass
            px = pts * torch.cos(phis)
            py = pts * torch.sin(phis)
            pz = pts * torch.sinh(etas)
            p = torch.stack([px, py, pz, energies], dim=1)
            p1 = p.unsqueeze(1)
            p2 = p.unsqueeze(0)
            m2 = (p1[:, :, 3] + p2[:, :, 3])**2 - ((p1[:, :, :3] + p2[:, :, :3])**2).sum(2)
            m2 = torch.where(m2 > 0, m2, torch.zeros_like(m2))
            m = torch.sqrt(m2)

            # Take upper triangular without diagonal
            idx = torch.triu_indices(delta_r.shape[0], delta_r.shape[1], offset=1)
            pairwise = torch.stack([
                delta_r[idx[0], idx[1]],
                m[idx[0], idx[1]],
                pts.unsqueeze(0) * pts.unsqueeze(1) / (pts.sum()**2 + 1e-8)  # pT correlations
            ], dim=1)
            pairwise_features.append(pairwise)

        # Pad pairwise features
        max_pairs = max(p.shape[0] for p in pairwise_features) if len(pairwise_features) > 0 else 1
        paired_features_padded = torch.zeros(batch_size, max_pairs, 3)
        for i, pf in enumerate(pairwise_features):
            paired_features_padded[i, :pf.shape[0]] = pf

        return (
            global_features,  # [batch_size, 2]
            particle_features,  # [batch_size, 18, 5]
            mask.to(torch.float32),  # [batch_size, 18]
            paired_features_padded,  # [batch_size, max_pairs, 3]
            torch.tensor([p.shape[0] for p in pairwise_features])  # [batch_size]
        )

    def fit(self, X, y=None):
        X_transformed = self._raw_reshape(X)
        global_features = X_transformed[0]
        self.mean = global_features.mean(0)
        self.std = global_features.std(0)
        return self

    def transform(self, X):
        global_features, particle_features, mask, paired_features, pair_counts = self._raw_reshape(X)

        # Normalize global features
        global_features = (global_features - self.mean) / (self.std + 1e-8)

        # Normalize particle features (except object IDs)
        particle_features[:, :, 1:] = (particle_features[:, :, 1:] - particle_features[:, :, 1:].mean(dim=(0,1), keepdim=True)) / (particle_features[:, :, 1:].std(dim=(0,1), keepdim=True) + 1e-8)

        # Normalize pairwise features
        paired_features = (paired_features - paired_features.mean(dim=(0,1), keepdim=True)) / (paired_features.std(dim=(0,1), keepdim=True) + 1e-8)

        mask = mask.unsqueeze(-1)  # [batch_size, 18, 1]
        pair_counts = pair_counts.unsqueeze(-1)  # [batch_size, 1]

        return (global_features, particle_features, mask, paired_features, pair_counts)

# 2. ---------- MODEL DEFINITION ----------
class ParticleAttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.linear = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask.squeeze(-1)==0 if mask is not None else None)
        x = x + attn_output
        x = x + self.linear(x)
        return x

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Extract input dimensions from sample
        global_feats, particle_feats, mask, paired_feats, pair_counts = sample_object

        # Parameters
        self.global_dim = global_feats.shape[-1]
        self.particle_dim = particle_feats.shape[-1]
        self.pair_dim = paired_feats.shape[-1]
        self.hidden_dim = 64

        # Global feature processor
        self.global_processor = nn.Sequential(
            nn.Linear(self.global_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Particle feature processor
        self.particle_embedding = nn.Sequential(
            nn.Linear(self.particle_dim - 1, self.hidden_dim),  # exclude object ID
            nn.ReLU()
        )

        # Pairwise feature processor
        self.pair_processor = nn.Sequential(
            nn.Linear(self.pair_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Attention layers
        self.particle_attention = ParticleAttentionLayer(self.hidden_dim, self.hidden_dim, num_heads=4)
        self.pair_attention = ParticleAttentionLayer(self.hidden_dim, self.hidden_dim, num_heads=4)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(2 * self.hidden_dim),
            nn.Linear(2 * self.hidden_dim, 1)
        )

    def forward(self, *data):
        global_features, particle_features, mask, paired_features, pair_counts = data

        # Process global features
        global_encoded = self.global_processor(global_features)

        # Process particle features
        particle_hidden = self.particle_embedding(particle_features[..., 1:])  # exclude object ID
        particle_hidden = particle_hidden * mask  # apply mask
        particle_encoded = self.particle_attention(particle_hidden, mask)
        particle_encoded = torch.sum(particle_encoded * mask, dim=1) / (torch.sum(mask, dim=1) + 1e-8)

        # Process pairwise features
        batch_indices = torch.arange(paired_features.shape[0]).unsqueeze(1).expand(-1, paired_features.shape[1])
        batch_indices = batch_indices.flatten()

        valid_pairs = torch.cat([torch.arange(n) for n in pair_counts.flatten().tolist()]).long()
        batch_indices = batch_indices[valid_pairs]
        selected_pairs = paired_features[batch_indices, valid_pairs]

        pair_hidden = self.pair_processor(selected_pairs)
        pair_encoded = torch.zeros(paired_features.shape[0], self.hidden_dim, device=paired_features.device)
        pair_encoded = pair_encoded.scatter_add(0, batch_indices.unsqueeze(1).expand(-1, self.hidden_dim), pair_hidden)
        pair_encoded = pair_encoded / (pair_counts.float() + 1e-8)

        # Combine features
        combined = torch.cat([global_encoded, particle_encoded, pair_encoded], dim=-1)

        # Final prediction
        output = self.output_head(combined)
        return torch.sigmoid(output.squeeze())

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    best_val_auc = -1
    best_model_weights = None
    patience = 5
    epochs_no_improve = 0

    train_loss_history = []
    val_loss_history = []
    train_auc_history = []
    val_auc_history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            X_batch = [x.to(device) for x in X_batch]
            y_batch = y_batch.to(device).float()

            outputs = model(*X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            all_train_preds.append(outputs.detach().cpu())
            all_train_labels.append(y_batch.cpu())

        # Calculate train metrics
        train_loss /= len(train_loader)
        train_preds = torch.cat(all_train_preds)
        train_labels = torch.cat(all_train_labels)
        train_auc = roc_auc_score(train_labels.numpy(), train_preds.numpy())

        # Validation
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = [x.to(device) for x in X_batch]
                y_batch = y_batch.to(device).float()

                outputs = model(*X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                all_val_preds.append(outputs.cpu())
                all_val_labels.append(y_batch.cpu())

        val_loss /= len(val_loader)
        val_preds = torch.cat(all_val_preds)
        val_labels = torch.cat(all_val_labels)
        val_auc = roc_auc_score(val_labels.numpy(), val_preds.numpy())
        scheduler.step(val_auc)

        # Logging
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_auc_history.append(train_auc)
        val_auc_history.append(val_auc)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_weights = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} with best val AUC: {best_val_auc:.4f}")
                break

    # Load best model weights
    model.load_state_dict(best_model_weights)

    return model, train_loss_history, val_loss_history, train_auc_history, val_auc_history

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

