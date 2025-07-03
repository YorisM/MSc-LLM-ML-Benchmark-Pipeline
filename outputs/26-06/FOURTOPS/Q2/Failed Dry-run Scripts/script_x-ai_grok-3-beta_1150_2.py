
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
from torch.nn import Linear, LayerNorm, MultiheadAttention, Dropout
from sklearn.preprocessing import StandardScaler

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler_global = StandardScaler()
        self.scaler_obj = StandardScaler()
        self.max_objects = 18
        self.obj_features = 4  # E, p_T, eta, phi

    def _raw_reshape(self, X):
        return X  # Return as-is, reshaping done in transform

    def fit(self, X, y=None):
        # Extract global features (E_T_miss, phi_Et_miss)
        global_features = X[:, :2].numpy()  # Shape: [N, 2]
        self.scaler_global.fit(global_features)

        # Extract object features
        obj_data = X[:, 2:].reshape(-1, self.max_objects, 5)  # Shape: [N, max_obj, 5]
        obj_kinematics = obj_data[:, :, 1:].reshape(-1, self.obj_features)  # Shape: [N*max_obj, 4]
        mask = obj_data[:, :, 0].reshape(-1) > 0  # Shape: [N*max_obj], valid objects
        valid_kinematics = obj_kinematics[mask]
        if len(valid_kinematics) > 0:
            self.scaler_obj.fit(valid_kinematics)
        return self

    def transform(self, X):
        device = X.device
        N = X.shape[0]

        # Global features
        global_features = X[:, :2].numpy()  # Shape: [N, 2]
        global_features = torch.from_numpy(self.scaler_global.transform(global_features)).float().to(device)  # Shape: [N, 2]

        # Object features
        obj_data = X[:, 2:].reshape(N, self.max_objects, 5)  # Shape: [N, max_obj, 5]
        obj_kinematics = obj_data[:, :, 1:]  # Shape: [N, max_obj, 4]
        obj_ids = obj_data[:, :, 0]  # Shape: [N, max_obj]
        mask = obj_ids > 0  # Shape: [N, max_obj], valid objects

        # Scale kinematics
        flat_kinematics = obj_kinematics.reshape(-1, self.obj_features).numpy()  # Shape: [N*max_obj, 4]
        flat_scaled = self.scaler_obj.transform(flat_kinematics)
        obj_kinematics = torch.from_numpy(flat_scaled).float().reshape(N, self.max_objects, self.obj_features).to(device)  # Shape: [N, max_obj, 4]

        # Compute pairwise features (delta_R and invariant mass)
        pairwise_features = self._compute_pairwise_features(obj_kinematics, mask)  # Shape: [N, max_obj, max_obj, 2]

        return {"global": global_features, "objects": obj_kinematics, "pairwise": pairwise_features, "mask": mask}

    def _compute_pairwise_features(self, obj_kinematics, mask):
        N = obj_kinematics.shape[0]
        dR = torch.zeros(N, self.max_objects, self.max_objects, device=obj_kinematics.device)  # Shape: [N, max_obj, max_obj]
        inv_mass = torch.zeros(N, self.max_objects, self.max_objects, device=obj_kinematics.device)  # Shape: [N, max_obj, max_obj]

        for i in range(self.max_objects):
            for j in range(i + 1, self.max_objects):
                eta_i = obj_kinematics[:, i, 2]  # Shape: [N]
                eta_j = obj_kinematics[:, j, 2]  # Shape: [N]
                phi_i = obj_kinematics[:, i, 3]  # Shape: [N]
                phi_j = obj_kinematics[:, j, 3]  # Shape: [N]
                d_eta = eta_i - eta_j  # Shape: [N]
                d_phi = phi_i - phi_j  # Shape: [N]
                dR[:, i, j] = dR[:, j, i] = torch.sqrt(d_eta**2 + d_phi**2)  # Shape: [N]

                E_i = obj_kinematics[:, i, 0]  # Shape: [N]
                E_j = obj_kinematics[:, j, 0]  # Shape: [N]
                pt_i = obj_kinematics[:, i, 1]  # Shape: [N]
                pt_j = obj_kinematics[:, j, 1]  # Shape: [N]
                inv_mass[:, i, j] = inv_mass[:, j, i] = torch.sqrt(
                    (E_i + E_j)**2 - (pt_i * torch.cos(phi_i) + pt_j * torch.cos(phi_j))**2 -
                    (pt_i * torch.sin(phi_i) + pt_j * torch.sin(phi_j))**2
                )  # Simplified approximation, Shape: [N]

        pairwise = torch.stack([dR, inv_mass], dim=-1)  # Shape: [N, max_obj, max_obj, 2]
        return pairwise

    def make_loader_cfg(self):
        return {
            "loader_class": "torch.utils.data.DataLoader",
            "batch_size": 256,
            "shuffle": False,
            "num_workers": 0
        }

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.max_objects = 18
        self.obj_dim = 4
        self.pairwise_dim = 2
        self.global_dim = 2
        self.d_model = 64

        # Embeddings
        self.obj_embed = Linear(self.obj_dim, self.d_model)
        self.pairwise_embed = Linear(self.pairwise_dim, self.d_model // 2)
        self.global_embed = Linear(self.global_dim, self.d_model)

        # Transformer layers for objects
        self.attention = MultiheadAttention(embed_dim=self.d_model, num_heads=8)
        self.norm1 = LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            Linear(self.d_model, 256),
            nn.ReLU(),
            Dropout(0.1),
            Linear(256, self.d_model)
        )
        self.norm2 = LayerNorm(self.d_model)
        self.dropout = Dropout(0.1)

        # Pairwise interaction layer
        self.pairwise_attn = MultiheadAttention(embed_dim=self.d_model // 2, num_heads=4)

        # Final classifier
        self.classifier = nn.Sequential(
            Linear(self.d_model * (self.max_objects + 1), 512),
            nn.ReLU(),
            Dropout(0.1),
            Linear(512, 128),
            nn.ReLU(),
            Linear(128, 1)
        )

    def forward(self, data):
        global_feat = data["global"]  # Shape: [batch, 2]
        objects = data["objects"]  # Shape: [batch, max_obj, 4]
        pairwise = data["pairwise"]  # Shape: [batch, max_obj, max_obj, 2]
        mask = data["mask"]  # Shape: [batch, max_obj]
        batch_size = global_feat.size(0)

        # Embed global features
        global_emb = self.global_embed(global_feat).unsqueeze(1)  # Shape: [batch, 1, d_model]

        # Embed object features
        obj_emb = self.obj_embed(objects)  # Shape: [batch, max_obj, d_model]

        # Transformer on objects
        obj_emb = obj_emb.permute(1, 0, 2)  # Shape: [max_obj, batch, d_model]
        attn_mask = ~mask.unsqueeze(1).repeat(1, self.max_objects, 1)  # Shape: [batch, max_obj, max_obj]
        attn_output, _ = self.attention(obj_emb, obj_emb, obj_emb, key_padding_mask=mask)  # Shape: [max_obj, batch, d_model]
        obj_emb = self.norm1(obj_emb + self.dropout(attn_output))  # Shape: [max_obj, batch, d_model]
        ffn_output = self.ffn(obj_emb)  # Shape: [max_obj, batch, d_model]
        obj_emb = self.norm2(obj_emb + self.dropout(ffn_output))  # Shape: [max_obj, batch, d_model]
        obj_emb = obj_emb.permute(1, 0, 2)  # Shape: [batch, max_obj, d_model]

        # Pairwise features processing
        pairwise_emb = self.pairwise_embed(pairwise)  # Shape: [batch, max_obj, max_obj, d_model//2]
        pairwise_emb = pairwise_emb.view(batch_size, self.max_objects**2, -1).permute(1, 0, 2)  # Shape: [max_obj^2, batch, d_model//2]
        pairwise_output, _ = self.pairwise_attn(pairwise_emb, pairwise_emb, pairwise_emb)  # Shape: [max_obj^2, batch, d_model//2]
        pairwise_emb = pairwise_output.permute(1, 0, 2).view(batch_size, self.max_objects, self.max_objects, -1)  # Shape: [batch, max_obj, max_obj, d_model//2]
        pairwise_summary = pairwise_emb.mean(dim=2)  # Shape: [batch, max_obj, d_model//2]

        # Combine features
        obj_combined = torch.cat([obj_emb, pairwise_summary], dim=-1)  # Shape: [batch, max_obj, d_model + d_model//2]
        obj_summary = obj_combined.view(batch_size, -1)  # Shape: [batch, max_obj * (d_model + d_model//2)]
        full_feat = torch.cat([global_emb.view(batch_size, -1), obj_summary], dim=-1)  # Shape: [batch, d_model + max_obj * (d_model + d_model//2)]

        # Classification
        logits = self.classifier(full_feat)  # Shape: [batch, 1]
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    patience = 5
    counter = 0
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    from sklearn.metrics import roc_auc_score

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        train_preds, train_labels = [], []
        for batch in train_loader:
            data, labels = batch
            if isinstance(data, dict):
                data = {k: v.to(device) for k, v in data.items()}
            labels = labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            train_preds.extend(preds.ravel())
            train_labels.extend(labels.cpu().numpy().ravel())

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_auc = roc_auc_score(train_labels, train_preds) if len(set(train_labels)) > 1 else 0.5
        train_loss.append(avg_train_loss)
        train_acc.append(train_auc)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                data, labels = batch
                if isinstance(data, dict):
                    data = {k: v.to(device) for k, v in data.items()}
                labels = labels.to(device).float().view(-1, 1)
                outputs = model(data)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                preds = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(preds.ravel())
                val_labels.extend(labels.cpu().numpy().ravel())

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0.5
        val_loss.append(avg_val_loss)
        val_acc.append(val_auc)

        scheduler.step(val_auc)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train AUC: {train_auc:.4f}, Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

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

