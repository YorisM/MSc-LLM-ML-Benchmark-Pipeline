
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

# -------------------------- START OF LLM BLOCK ------------------------------
# 0. ---------- IMPORTS ----------
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
import math

# 2. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None
        self.max_objects = 18
        self.feature_size = 5  # per-object features: obj_id, E, pT, eta, phi

    def _raw_reshape(self, X):
        # Reshape to (batch_size, max_objects, feature_size)
        # Also include MET features (indices 0-1)
        batch_size = X.shape[0]

        # MET features
        met_features = X[:, :2]

        # Reshape object features
        obj_features = X[:, 2:].reshape(batch_size, self.max_objects, self.feature_size)

        return met_features, obj_features

    def fit(self, X, y=None):
        met_features, obj_features = self._raw_reshape(X)

        # Calculate mean and std for normalization
        combined = torch.cat([
            met_features, 
            obj_features.reshape(-1, self.feature_size)
        ], dim=0)

        # Filter out padding (obj_id = 0)
        valid_mask = combined[:, 0] != 0
        valid_features = combined[valid_mask]

        self.mean = valid_features.mean(dim=0)
        self.std = valid_features.std(dim=0)

        # Handle zero std (for obj_id)
        self.std[self.std == 0] = 1

        return self

    def transform(self, X):
        met_features, obj_features = self._raw_reshape(X)
        batch_size = met_features.shape[0]

        # Normalize features
        norm_met = (met_features - self.mean[:2]) / self.std[:2]

        # Normalize objects, preserving the object ID
        obj_ids = obj_features[..., 0]  # shape: (batch_size, max_objects)
        normalized_obj_features = (obj_features[..., 1:] - self.mean[1:]) / self.std[1:]

        # Combine back with object IDs
        normalized_obj_features = torch.cat([
            obj_ids.unsqueeze(-1),  # shape: (batch_size, max_objects, 1)
            normalized_obj_features  # shape: (batch_size, max_objects, 4)
        ], dim=-1)

        # Compute additional features
        combined_features = []

        # For each object, compute pairwise features with all other objects
        for i in range(batch_size):
            objects = normalized_obj_features[i]  # shape: (max_objects, 5)
            met = norm_met[i]  # shape: (2,)

            # Filter non-padded objects (where obj_id != 0)
            mask = objects[:, 0] != 0
            valid_objs = objects[mask]  # shape: (n_valid_objects, 5)
            n_valid = valid_objs.shape[0]

            # We'll use 5 features per object:
            # 1. Original normalized features (5)
            # 2. MET-related features (3)
            # 3. Pairwise features (mean of top 5 deltaR and m) (10)
            # Total per object: 5 + 3 + 10 = 18 features

            # Prepare output tensor
            output_features = torch.zeros((self.max_objects, 18))

            if n_valid > 0:
                # Original features
                output_features[:n_valid, :5] = valid_objs

                # MET-related features
                delta_phi_met = valid_objs[:, 4] - met[1]  # phi_obj - phi_met
                met_rel_pT = valid_objs[:, 2] / met[0] if met[0] > 0 else 0  # pT / MET
                output_features[:n_valid, 5:8] = torch.stack([
                    delta_phi_met,
                    met_rel_pT,
                    met[0]  # MET magnitude
                ], dim=-1)

                # Pairwise features
                if n_valid > 1:
                    # Calculate delta R and invariant mass between all pairs
                    etas = valid_objs[:, 3]  # shape: (n_valid,)
                    phis = valid_objs[:, 4]  # shape: (n_valid,)
                    energies = valid_objs[:, 1]  # shape: (n_valid,)
                    pTs = valid_objs[:, 2]  # shape: (n_valid,)

                    # Calculate deltaR between all pairs
                    delta_eta = etas.unsqueeze(0) - etas.unsqueeze(1)  # shape: (n_valid, n_valid)
                    delta_phi = phis.unsqueeze(0) - phis.unsqueeze(1)  # shape: (n_valid, n_valid)
                    delta_phi = (delta_phi + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi]
                    delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)

                    # Calculate invariant mass (approximate)
                    # m^2 ≈ 2*pT1*pT2(cosh(Δη) - cos(Δφ))
                    cosh_deta = torch.cosh(delta_eta)
                    cos_dphi = torch.cos(delta_phi)
                    m2_approx = 2 * pTs.unsqueeze(0) * pTs.unsqueeze(1) * (cosh_deta - cos_dphi)
                    m2_approx = torch.clamp(m2_approx, min=0)
                    m_approx = torch.sqrt(m2_approx)

                    # For each object, get top 5 deltaR and m values with other objects
                    top_deltaR = []
                    top_m = []

                    for j in range(n_valid):
                        # Get all pairs except self (set diagonal to very small value)
                        mask = torch.ones(n_valid, dtype=torch.bool)
                        mask[j] = False

                        obj_deltaR = delta_R[j, mask]
                        obj_m = m_approx[j, mask]

                        # Sort and get top 5
                        sorted_dR, _ = torch.sort(obj_deltaR, descending=True)
                        sorted_m, _ = torch.sort(obj_m, descending=True)

                        top_deltaR.append(sorted_dR[:5])
                        top_m.append(sorted_m[:5])

                    if top_deltaR and top_m:
                        top_deltaR = torch.stack(top_deltaR)  # shape: (n_valid, 5)
                        top_m = torch.stack(top_m)  # shape: (n_valid, 5)

                        # Mean of top 5 deltaR and m
                        output_features[:n_valid, 8:13] = top_deltaR
                        output_features[:n_valid, 13:18] = top_m

            # Add to batch
            combined_features.append(output_features)

        # Combine all batches
        combined_features = torch.stack(combined_features)  # shape: (batch_size, max_objects, 18)

        # Finally, combine with MET features
        met_features_expanded = met.unsqueeze(0).expand(batch_size, -1)

        return met_features_expanded, combined_features

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class ParticleAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_proj(x)  # (batch_size, seq_len, hidden_dim)

        if mask is not None:
            # Convert to bool and invert (False for padded elements)
            mask = ~(mask.bool())

        x = self.transformer(x, src_key_padding_mask=mask)

        # Global pooling
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)
        x = self.pooling(x)  # (batch_size, hidden_dim, 1)
        x = x.squeeze(-1)  # (batch_size, hidden_dim)

        return x

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        met_features, obj_features = sample_object
        input_dim = obj_features.shape[-1]  # Should be 18
        hidden_dim = 256
        num_heads = 8
        num_layers = 4

        self.met_proj = nn.Linear(met_features.shape[-1], hidden_dim)
        self.particle_attn = ParticleAttention(input_dim, hidden_dim, num_heads, num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, *data):
        met_features, obj_features = data
        batch_size = met_features.shape[0]

        # Create mask for padded objects (where obj_id = 0)
        mask = obj_features[:, :, 0] == 0  # shape: (batch_size, max_objects)

        # Project MET features
        met_embed = self.met_proj(met_features)  # shape: (batch_size, hidden_dim)

        # Process objects with attention
        obj_embed = self.particle_attn(obj_features, mask)  # shape: (batch_size, hidden_dim)

        # Combine features
        combined = torch.cat([met_embed, obj_embed], dim=-1)  # shape: (batch_size, hidden_dim * 2)

        # Classify
        output = self.classifier(combined)  # shape: (batch_size, 1)

        return output.squeeze(-1)  # shape: (batch_size,)

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    best_auc = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        all_preds = []
        all_targets = []

        for batch in train_loader:
            data, target = batch
            data = tuple(d.to(device) for d in data)
            target = target.float().to(device)

            optimizer.zero_grad()
            output = model(*data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            all_preds.append(output.detach().cpu())
            all_targets.append(target.detach().cpu())

        # Calculate training metrics
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        train_acc = ((all_preds > 0.5) == all_targets).float().mean().item()
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                data, target = batch
                data = tuple(d.to(device) for d in data)
                target = target.float().to(device)

                output = model(*data)
                loss = criterion(output, target)
                val_loss += loss.item()

                val_preds.append(output.cpu())
                val_targets.append(target.cpu())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)

        val_acc = ((val_preds > 0.5) == val_targets).float().mean().item()
        val_accs.append(val_acc)

        # Calculate AUC
        val_auc = roc_auc_score(val_targets.numpy(), val_preds.numpy())

        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
              f'Val AUC: {val_auc:.4f}')

        # Early stopping based on AUC
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()

        scheduler.step(val_auc)

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs
# ---------------------------  END OF LLM-CODE BLOCK ---------------------------

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

