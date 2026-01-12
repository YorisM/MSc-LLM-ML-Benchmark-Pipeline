
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts, to_python
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops, dryrun_finite_check_fourtops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
                        
DATASET = {
    "X_train": "./challenges/FOURTOPS/data/train/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/train/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/train/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/train/Y_val.csv"
}
                       
def load_data():
    X_train = pd.read_csv(DATASET["X_train"], dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv(DATASET["Y_train"], dtype=np.int64).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv(DATASET["X_val"], dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv(DATASET['Y_val'], dtype=np.int64).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train), torch.from_numpy(Y_train),
            torch.from_numpy(X_val), torch.from_numpy(Y_val))

class FourTopsDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        X2 = pre.transform(X) if pre is not None else X
        if not torch.is_tensor(X2):
            X2 = torch.as_tensor(X2)
        self.X = X2.float()
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        self.y = y.long()
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

# ---------- IMPORTS ----------
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.global_scaler = StandardScaler()
        self.obj_scaler = StandardScaler()

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 256}
        }

    def fit(self, X, y=None):
        # Separate global and object features
        global_feats = X[:, :2]  # E_T_miss, phi_Et_miss
        obj_feats = X[:, 2:].reshape(-1, 18, 5)[:, :, 1:]  # Skip obj_id, keep E, pT, eta, phi

        # Flatten for scaling
        global_flat = global_feats.reshape(-1, 2)
        obj_flat = obj_feats.reshape(-1, 4)

        # Fit scalers
        self.global_scaler.fit(global_flat)
        self.obj_scaler.fit(obj_flat)
        return self

    def transform(self, X):
        # Apply scaling
        global_feats = X[:, :2]
        obj_feats = X[:, 2:].reshape(-1, 18, 5)

        # Scale global features
        global_scaled = self.global_scaler.transform(global_feats)

        # Extract and scale object features (skip obj_id)
        obj_ids = obj_feats[:, :, 0].long()
        obj_vals = obj_feats[:, :, 1:].reshape(-1, 4)
        obj_vals_scaled = self.obj_scaler.transform(obj_vals).reshape(-1, 18, 4)

        # Reconstruct full tensor
        obj_feats_scaled = torch.cat([
            obj_ids.unsqueeze(-1).float(),
            torch.from_numpy(obj_vals_scaled).float()
        ], dim=-1).reshape(X.shape[0], -1)

        # Combine with global features
        X_scaled = torch.cat([
            torch.from_numpy(global_scaled).float(),
            obj_feats_scaled
        ], dim=1)

        return X_scaled

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Extract dimensions from sample
        if isinstance(sample_object, Data):
            # PyG mode
            self.node_dim = sample_object.x.shape[1]
            self.global_dim = 2  # E_T_miss, phi_Et_miss
            self.use_pyg = True
        else:
            # Dense mode
            self.node_dim = 5  # obj_id + 4 features
            self.global_dim = 2
            self.use_pyg = False

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Object feature encoder
        self.obj_encoder = nn.Sequential(
            nn.Linear(4, 32),  # Skip obj_id
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Transformer layers for object interactions
        self.transformer_conv1 = TransformerConv(16, 32, heads=4)
        self.transformer_conv2 = TransformerConv(32, 32, heads=4)

        # Attention for global-object interaction
        self.global_attention = nn.MultiheadAttention(embed_dim=16, num_heads=2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, batch_x):
        if self.use_pyg:
            return self._forward_pyg(batch_x)
        else:
            return self._forward_dense(batch_x)

    def _forward_dense(self, x):
        # x shape: [B, 92]
        B = x.shape[0]

        # Extract global features
        global_feats = x[:, :2]  # [B, 2]
        global_encoded = self.global_encoder(global_feats)  # [B, 16]

        # Extract object features
        obj_feats = x[:, 2:].reshape(B, 18, 5)  # [B, 18, 5]
        obj_ids = obj_feats[:, :, 0].long()  # [B, 18]
        obj_vals = obj_feats[:, :, 1:]  # [B, 18, 4]

        # Encode objects
        obj_encoded = self.obj_encoder(obj_vals)  # [B, 18, 16]

        # Create mask for valid objects (non-zero obj_id)
        mask = (obj_ids != 0).float().unsqueeze(-1)  # [B, 18, 1]

        # Compute pairwise features
        pairwise_feats = self._compute_pairwise_features(obj_vals, mask)  # [B, 18, 18, 4]

        # Transformer processing
        edge_index = self._get_dense_edges(18).to(obj_encoded.device)
        x_transformed = F.relu(self.transformer_conv1(obj_encoded, edge_index))
        x_transformed = F.relu(self.transformer_conv2(x_transformed, edge_index))

        # Global attention
        global_expanded = global_encoded.unsqueeze(1).expand(-1, 18, -1)  # [B, 18, 16]
        attn_output, _ = self.global_attention(
            global_expanded.transpose(0, 1),  # [18, B, 16]
            x_transformed.transpose(0, 1),    # [18, B, 32]
            x_transformed.transpose(0, 1)     # [18, B, 32]
        )
        attn_output = attn_output.transpose(0, 1)  # [B, 18, 16]

        # Aggregate features
        obj_agg = torch.mean(x_transformed * mask, dim=1)  # [B, 32]
        attn_agg = torch.mean(attn_output * mask, dim=1)  # [B, 16]

        # Combine and classify
        combined = torch.cat([obj_agg, attn_agg], dim=1)
        return self.classifier(combined).squeeze(-1)

    def _forward_pyg(self, data):
        # data is a Batch object
        x = data.x  # [N, 5] where N is total nodes in batch
        batch = data.batch  # [N] batch indices
        global_feats = data.global_feats  # [B, 2]

        # Encode global features
        global_encoded = self.global_encoder(global_feats)  # [B, 16]

        # Extract object features
        obj_ids = x[:, 0].long()
        obj_vals = x[:, 1:]

        # Encode objects
        obj_encoded = self.obj_encoder(obj_vals)  # [N, 16]

        # Create mask
        mask = (obj_ids != 0).float().unsqueeze(-1)  # [N, 1]

        # Get edge index for full graph
        edge_index = data.edge_index

        # Transformer processing
        x_transformed = F.relu(self.transformer_conv1(obj_encoded, edge_index))
        x_transformed = F.relu(self.transformer_conv2(x_transformed, edge_index))

        # Global attention
        # Expand global features to match nodes
        global_expanded = global_encoded[batch]  # [N, 16]
        attn_output, _ = self.global_attention(
            global_expanded.unsqueeze(0),  # [1, N, 16]
            x_transformed.unsqueeze(0),    # [1, N, 32]
            x_transformed.unsqueeze(0)     # [1, N, 32]
        )
        attn_output = attn_output.squeeze(0)  # [N, 16]

        # Aggregate features
        obj_agg = global_mean_pool(x_transformed * mask, batch)  # [B, 32]
        attn_agg = global_mean_pool(attn_output * mask, batch)  # [B, 16]

        # Combine and classify
        combined = torch.cat([obj_agg, attn_agg], dim=1)
        return self.classifier(combined).squeeze(-1)

    def _compute_pairwise_features(self, obj_vals, mask):
        # obj_vals: [B, 18, 4]
        B, N, F = obj_vals.shape

        # Expand for pairwise computation
        obj_i = obj_vals.unsqueeze(2)  # [B, 18, 1, 4]
        obj_j = obj_vals.unsqueeze(1)  # [B, 1, 18, 4]

        # Compute invariant mass
        E_i = obj_i[..., 0]
        E_j = obj_j[..., 0]
        pT_i = obj_i[..., 1]
        pT_j = obj_j[..., 1]
        eta_i = obj_i[..., 2]
        eta_j = obj_j[..., 2]
        phi_i = obj_i[..., 3]
        phi_j = obj_j[..., 3]

        # Compute pz from pT and eta
        pz_i = pT_i * torch.sinh(eta_i)
        pz_j = pT_j * torch.sinh(eta_j)

        # Compute px and py
        px_i = pT_i * torch.cos(phi_i)
        py_i = pT_i * torch.sin(phi_i)
        px_j = pT_j * torch.cos(phi_j)
        py_j = pT_j * torch.sin(phi_j)

        # Total energy and momentum
        E_total = E_i + E_j
        px_total = px_i + px_j
        py_total = py_i + py_j
        pz_total = pz_i + pz_j

        # Invariant mass
        m_inv = torch.sqrt(E_total**2 - (px_total**2 + py_total**2 + pz_total**2))

        # Angular distance
        deta = eta_i - eta_j
        dphi = torch.atan2(torch.sin(phi_i - phi_j), torch.cos(phi_i - phi_j))
        delta_R = torch.sqrt(deta**2 + dphi**2)

        # Stack features
        pairwise_feats = torch.stack([
            m_inv,
            delta_R,
            torch.abs(eta_i - eta_j),
            torch.abs(phi_i - phi_j)
        ], dim=-1)  # [B, 18, 18, 4]

        return pairwise_feats

    def _get_dense_edges(self, num_nodes):
        # Create complete graph edges
        adj = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        edge_index = adj.nonzero().t()
        return edge_index

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)

    best_val_auc = 0
    best_model = None

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            if isinstance(batch, Data):
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y.float())
            else:
                X, y = batch
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(X)
                loss = criterion(out, y.float())

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            pred = (out > 0).float()
            correct += (pred == y.float()).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, Data):
                    batch = batch.to(device)
                    out = model(batch)
                    loss = criterion(out, batch.y.float())
                    preds = torch.sigmoid(out)
                else:
                    X, y = batch
                    X, y = X.to(device), y.to(device)
                    out = model(X)
                    loss = criterion(out, y.float())
                    preds = torch.sigmoid(out)

                val_loss += loss.item() * y.size(0)
                pred = (preds > 0.5).float()
                correct += (pred == y.float()).sum().item()
                total += y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss = val_loss / total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(all_labels, all_preds)
        except:
            val_auc = 0

        # Update best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model.state_dict().copy()

        # Update scheduler
        scheduler.step(val_auc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

    # Load best model
    model.load_state_dict(best_model)

    return model, train_losses, val_losses, train_accs, val_accs

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    X_fit, Y_fit = X_train, Y_train
    if dryrun:
        idx = torch.randperm(X_train.shape[0])[:400]
        X_train, Y_train = X_train[idx], Y_train[idx]
        idx = torch.randperm(X_val.shape[0])[:200]
        X_val, Y_val = X_val[idx], Y_val[idx]
    pre = make_preprocessor().fit(X_fit, Y_fit)
    
    # Build LoaderSpec
    spec = build_spec_from_preproc(pre, script_module="llm_script")
    spec = enforce_pyg_policy(spec, require_torch_collate=False)

    # Build loaders - preproc in dataset
    train_ds     = build_dataset(spec, (X_train, Y_train), pre, train=True)
    val_ds       = build_dataset(spec, (X_val,   Y_val),   pre, train=False)
    train_loader = build_dataloader(spec, train_ds, is_eval=False)
    val_loader   = build_dataloader(spec, val_ds,   is_eval=True)

    # Build batch and check
    first_batch = next(iter(train_loader))
    mode = detect_and_assert_lane_fourtops(spec, first_batch)
    view = make_view_by_lane_fourtops(mode, first_batch, device)

    # Build model
    model = make_model(view.batch_x).to(device)

    # Train model
    n_epochs = 10 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # Dry-run safety check
    if dryrun:
        try:
            dryrun_finite_check_fourtops(trained_model, spec, val_loader, device, batches=10)
            with torch.no_grad():
                mode = detect_and_assert_lane_fourtops(spec, first_batch)
                view = make_view_by_lane_fourtops(mode, first_batch, device)
                out  = trained_model(view.batch_x)
                scores, kind = assert_binary_output(view, out)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e

    if not dryrun:
        # Persist artefacts
        base = base_from_argv0()
        persist_artefacts(base, SCRIPT_DIR, trained_model, pre, spec)

        # Save plots
        plot_train_val(tr_loss, va_loss, f"{base} Loss", os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
        plot_train_val(tr_acc, va_acc, f"{base} Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))
        
        # Write JSON Summary
        summary = {
            "epochs": n_epochs      if n_epochs else None,
            "train_loss": tr_loss   if tr_loss else None,
            "val_loss":   va_loss   if va_loss else None,
            "train_acc":  tr_acc    if tr_acc else None,
            "val_acc":    va_acc    if va_acc else None,
        }
        summary = to_python(summary)
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

