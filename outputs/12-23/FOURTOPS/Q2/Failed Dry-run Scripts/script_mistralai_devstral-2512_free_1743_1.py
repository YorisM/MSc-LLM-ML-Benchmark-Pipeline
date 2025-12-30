
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import normalise_batch, assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts

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
        self.X = pre.transform(X) if pre is not None else X
        self.y = y
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ----------------                        
# -------------------------- START OF LLM BLOCK ------------------------------

# ---------- IMPORTS ----------
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.loader import DataLoader

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
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Separate global and object features
        global_features = X[:, :2]  # [n_events, 2]
        obj_features = X[:, 2:].reshape(-1, 18, 5)  # [n_events, 18, 5]

        # Flatten object features for scaling (excluding obj_id)
        obj_features_flat = obj_features[:, :, 1:].reshape(-1, 4)  # [n_events*18, 4]

        # Fit scalers
        self.global_scaler.fit(global_features)
        self.obj_scaler.fit(obj_features_flat)
        return self

    def transform(self, X):
        # Separate global and object features
        global_features = X[:, :2]  # [n_events, 2]
        obj_features = X[:, 2:].reshape(-1, 18, 5)  # [n_events, 18, 5]

        # Scale global features
        global_scaled = self.global_scaler.transform(global_features)  # [n_events, 2]

        # Scale object features (excluding obj_id)
        obj_features_flat = obj_features[:, :, 1:].reshape(-1, 4)  # [n_events*18, 4]
        obj_scaled_flat = self.obj_scaler.transform(obj_features_flat)  # [n_events*18, 4]
        obj_scaled = obj_scaled_flat.reshape(-1, 18, 4)  # [n_events, 18, 4]

        # Combine obj_id with scaled features
        obj_id = obj_features[:, :, 0:1]  # [n_events, 18, 1]
        obj_processed = torch.cat([obj_id, torch.from_numpy(obj_scaled).float()], dim=2)  # [n_events, 18, 5]

        # Create graph data
        data_list = []
        for i in range(X.shape[0]):
            # Node features: [num_nodes, node_feature_dim]
            node_features = obj_processed[i]  # [18, 5]

            # Global features: [global_feature_dim]
            global_feat = torch.from_numpy(global_scaled[i]).float()  # [2]

            # Create edge indices for complete graph (excluding self-loops)
            num_nodes = 18
            edge_index = []
            for j in range(num_nodes):
                for k in range(num_nodes):
                    if j != k:
                        edge_index.append([j, k])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]

            # Compute pairwise features
            eta = node_features[:, 3].unsqueeze(1)  # [18, 1]
            phi = node_features[:, 4].unsqueeze(1)  # [18, 1]

            # Delta R and invariant mass
            delta_eta = eta - eta.t()  # [18, 18]
            delta_phi = phi - phi.t()  # [18, 18]
            delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)  # [18, 18]

            # Energy and pT for invariant mass
            E = node_features[:, 1].unsqueeze(1)  # [18, 1]
            pT = node_features[:, 2].unsqueeze(1)  # [18, 1]

            # Approximate invariant mass (simplified)
            m_ij = torch.sqrt(2 * pT * pT.t() * (torch.cosh(delta_eta) - torch.cos(delta_phi)))  # [18, 18]

            # Flatten pairwise features for edge attributes
            edge_attr = torch.stack([
                delta_R[edge_index[0], edge_index[1]],
                m_ij[edge_index[0], edge_index[1]]
            ], dim=1)  # [num_edges, 2]

            # Create data object
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                u=global_feat,
                y=torch.tensor([0], dtype=torch.float)  # placeholder
            )
            data_list.append(data)

        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Extract dimensions from sample
        node_dim = sample_object.x.shape[1]  # 5
        edge_dim = sample_object.edge_attr.shape[1] if hasattr(sample_object, 'edge_attr') else 0  # 2
        global_dim = sample_object.u.shape[0]  # 2

        # Node embedding
        self.node_embed = nn.Linear(node_dim, 64)

        # Edge embedding
        if edge_dim > 0:
            self.edge_embed = nn.Linear(edge_dim, 64)
        else:
            self.edge_embed = None

        # Global embedding
        self.global_embed = nn.Linear(global_dim, 64)

        # Transformer layers
        self.conv1 = TransformerConv(64, 64, heads=4, edge_dim=64 if edge_dim > 0 else None)
        self.conv2 = TransformerConv(64 * 4, 64, heads=4, edge_dim=64 if edge_dim > 0 else None)

        # Global attention
        self.global_att = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # Unpack batch
        x = batch_x.x  # [num_nodes, node_dim]
        edge_index = batch_x.edge_index  # [2, num_edges]
        edge_attr = batch_x.edge_attr if hasattr(batch_x, 'edge_attr') else None  # [num_edges, edge_dim]
        u = batch_x.u  # [batch_size, global_dim]

        # Embed nodes
        h = F.relu(self.node_embed(x))  # [num_nodes, 64]

        # Embed edges if available
        if edge_attr is not None and self.edge_embed is not None:
            edge_emb = F.relu(self.edge_embed(edge_attr))  # [num_edges, 64]
        else:
            edge_emb = None

        # First transformer layer
        h = self.conv1(h, edge_index, edge_emb)  # [num_nodes, 64*4]
        h = F.relu(h)

        # Second transformer layer
        h = self.conv2(h, edge_index, edge_emb)  # [num_nodes, 64*4]
        h = F.relu(h)

        # Global pooling
        h_global = global_mean_pool(h, batch_x.batch)  # [batch_size, 64*4]

        # Embed global features
        u_emb = F.relu(self.global_embed(u))  # [batch_size, 64]

        # Global attention
        att_input = torch.cat([h_global, u_emb], dim=1)  # [batch_size, 64*4 + 64]
        att_weights = torch.softmax(self.global_att(att_input), dim=0)  # [batch_size, 1]
        h_att = (h_global * att_weights).sum(dim=0, keepdim=True)  # [1, 64*4]

        # Combine with global features
        combined = torch.cat([h_att.expand(u_emb.shape[0], -1), u_emb], dim=1)  # [batch_size, 64*4 + 64]

        # Classifier
        out = self.classifier(combined)  # [batch_size, 1]
        return torch.sigmoid(out)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)
    criterion = nn.BCELoss()
    best_val_auc = 0
    patience = 10
    patience_counter = 0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch).squeeze()
            targets = batch.y.float()

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs
            train_total += batch.num_graphs

            # Compute accuracy
            preds = (outputs > 0.5).float()
            train_correct += (preds == targets).sum().item()

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                outputs = model(batch).squeeze()
                targets = batch.y.float()

                loss = criterion(outputs, targets)
                val_loss += loss.item() * batch.num_graphs
                val_total += batch.num_graphs

                preds = (outputs > 0.5).float()
                val_correct += (preds == targets).sum().item()

                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calculate metrics
        train_loss = train_loss / train_total
        val_loss = val_loss / val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(all_targets, all_preds)

        # Store history
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Early stopping and learning rate scheduling
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        scheduler.step(val_auc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

    # Load best model
    model.load_state_dict(best_model)

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    if dryrun:
        idx = torch.randperm(X_train.shape[0])[:400]
        X_train, Y_train = X_train[idx], Y_train[idx]
        idx = torch.randperm(X_val.shape[0])[:20]
        X_val, Y_val = X_val[idx], Y_val[idx]
    pre     = make_preprocessor().fit(X_train, Y_train)
    
    # Build LoaderSpec
    spec = build_spec_from_preproc(pre, script_module="llm_script")
    spec = enforce_pyg_policy(spec, require_torch_collate=False)

    # Build loaders - preproc in dataset
    train_ds     = build_dataset(spec, (X_train, Y_train), pre, train=True)
    val_ds       = build_dataset(spec, (X_val,   Y_val),   pre, train=False)
    train_loader = build_dataloader(spec, train_ds, is_eval=False)
    val_loader   = build_dataloader(spec, val_ds,   is_eval=True)

    # Build model
    first_batch = next(iter(train_loader))
    view        = normalise_batch(first_batch, device=device)
    model       = make_model(view.batch_x).to(device)

    # Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # Dry-run safety check
    if dryrun:
        try:
            with torch.no_grad():
                view = normalise_batch(first_batch, device=device)
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
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 


