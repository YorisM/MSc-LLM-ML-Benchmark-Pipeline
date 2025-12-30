
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
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, NNConv, global_add_pool
from torch_geometric.data import Data, Batch
from typing import List, Tuple

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class GraphDataset:
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.graphs = pre.transform(X) if pre is not None else X
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.y[idx]

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.obj_mean = None
        self.obj_std = None
        self.obj_types = None

    def make_loader_cfg(self):
        return {
            "dataset_builder": "llm_script:GraphDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
            "collate_fn": self.collate_fn,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def collate_fn(self, batch):
        graphs = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

        if isinstance(graphs[0], Data):
            batch_graph = Batch.from_data_list(graphs)
            return batch_graph, labels
        return torch.stack(graphs), labels

    def fit(self, X, y=None):
        # Extract statistics from training data
        X_np = X.numpy()

        # Global features (first 2)
        self.global_mean = np.mean(X_np[:, :2], axis=0)
        self.global_std = np.std(X_np[:, :2], axis=0) + 1e-8

        # Object features (rest)
        obj_features = X_np[:, 2:].reshape(-1, 5)
        # Mask out zero-padded objects (obj_id == 0)
        mask = obj_features[:, 0] != 0
        valid_obj_features = obj_features[mask, 1:]  # Exclude obj_id
        self.obj_mean = np.mean(valid_obj_features, axis=0)
        self.obj_std = np.std(valid_obj_features, axis=0) + 1e-8

        # Collect object types
        self.obj_types = np.unique(obj_features[mask, 0].astype(int))

        return self

    def transform(self, X):
        X_np = X.numpy()
        batch_graphs = []

        for i in range(X_np.shape[0]):
            # Extract global features
            global_feats = X_np[i, :2]
            global_feats = (global_feats - self.global_mean) / self.global_std

            # Extract objects
            objects = X_np[i, 2:].reshape(-1, 5)

            # Create mask for valid objects (non-zero padded)
            valid_mask = objects[:, 0] != 0
            valid_objs = objects[valid_mask]

            if len(valid_objs) == 0:
                # Empty graph fallback
                node_features = torch.zeros((1, 6), dtype=torch.float32)
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                global_tensor = torch.tensor(global_feats, dtype=torch.float32)
                graph = Data(x=node_features, edge_index=edge_index, global_feats=global_tensor)
                batch_graphs.append(graph)
                continue

            # Normalize object features
            obj_ids = valid_objs[:, 0].astype(int)
            obj_kinematics = valid_objs[:, 1:]
            obj_kinematics = (obj_kinematics - self.obj_mean) / self.obj_std

            # Create node features: [obj_id (one-hot), E, pT, eta, phi, obj_type_embed]
            n_nodes = len(obj_ids)

            # One-hot encode object IDs
            obj_one_hot = np.zeros((n_nodes, len(self.obj_types)))
            for j, obj_id in enumerate(obj_ids):
                idx = np.where(self.obj_types == obj_id)[0]
                if len(idx) > 0:
                    obj_one_hot[j, idx[0]] = 1

            # Combine features
            node_features = np.concatenate([
                obj_one_hot,
                obj_kinematics,
                np.sqrt(obj_kinematics[:, 1:2]**2 + obj_kinematics[:, 2:3]**2),  # pT magnitude feature
            ], axis=1)

            # Create fully connected graph
            edge_index = []
            for u in range(n_nodes):
                for v in range(n_nodes):
                    if u != v:
                        edge_index.append([u, v])

            if edge_index:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            # Edge features: deltaR, deltaPhi, deltaEta, pT ratio
            edge_attr = []
            if len(edge_index) > 0:
                for e in range(edge_index.shape[1]):
                    src, dst = edge_index[:, e]
                    eta1, phi1 = obj_kinematics[src, 1], obj_kinematics[src, 2]
                    eta2, phi2 = obj_kinematics[dst, 1], obj_kinematics[dst, 2]

                    dphi = abs(phi1 - phi2)
                    dphi = min(dphi, 2*np.pi - dphi)
                    deta = eta1 - eta2
                    deltaR = np.sqrt(dphi**2 + deta**2)

                    pT_ratio = obj_kinematics[src, 0] / (obj_kinematics[dst, 0] + 1e-8)

                    edge_attr.append([deltaR, dphi, deta, pT_ratio])

                edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
            else:
                edge_attr = torch.zeros((0, 4), dtype=torch.float32)

            # Convert to tensors
            node_features = torch.tensor(node_features, dtype=torch.float32)
            global_tensor = torch.tensor(global_feats, dtype=torch.float32)

            graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                global_feats=global_tensor
            )
            batch_graphs.append(graph)

        return batch_graphs

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class ParticleAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.scale = in_channels ** -0.5

    def forward(self, x):
        # x: [batch_size, n_particles, features]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        return out

class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim=4):
        super().__init__()
        self.conv = NNConv(
            in_channels, 
            out_channels,
            nn.Sequential(
                nn.Linear(edge_dim, 64),
                nn.ReLU(),
                nn.Linear(64, in_channels * out_channels)
            )
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = self.norm(x)
        x = self.act(x)
        return x

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        if isinstance(sample_object, Data):
            node_dim = sample_object.x.shape[1]
            global_dim = sample_object.global_feats.shape[0]
        else:
            # Fallback dimensions
            node_dim = 20
            global_dim = 2

        # GNN layers
        self.gnn1 = GNNLayer(node_dim, 128)
        self.gnn2 = GNNLayer(128, 256)
        self.gnn3 = GNNLayer(256, 512)

        # Attention pooling
        self.attn_pool = ParticleAttention(512)

        # Dense layers for global features
        self.global_net = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Interaction network
        self.interaction = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Auxiliary outputs for intermediate features
        self.aux_classifier1 = nn.Linear(256, 1)
        self.aux_classifier2 = nn.Linear(512, 1)

    def forward(self, batch_x):
        if hasattr(batch_x, 'batch'):
            # Graph input
            x, edge_index, edge_attr = batch_x.x, batch_x.edge_index, batch_x.edge_attr
            batch = batch_x.batch
            global_feats = batch_x.global_feats

            # GNN processing
            x1 = self.gnn1(x, edge_index, edge_attr)
            x2 = self.gnn2(x1, edge_index, edge_attr)
            x3 = self.gnn3(x2, edge_index, edge_attr)

            # Pooling
            batch_size = batch.max().item() + 1
            pooled = []

            for i in range(batch_size):
                mask = batch == i
                if mask.sum() > 0:
                    # Attention pooling
                    node_feats = x3[mask].unsqueeze(0)  # [1, n_nodes, features]
                    attn_out = self.attn_pool(node_feats)
                    pooled.append(attn_out.mean(dim=1))
                else:
                    pooled.append(torch.zeros(1, 512, device=x3.device))

            graph_embed = torch.cat(pooled, dim=0)  # [batch_size, 512]

            # Process global features
            global_embed = self.global_net(global_feats)

            # Combine
            combined = torch.cat([graph_embed, global_embed], dim=1)
            features = self.interaction(combined)
            logits = self.classifier(features).squeeze(-1)

            # Auxiliary outputs for deep supervision
            aux1 = self.aux_classifier1(x2.mean(dim=0, keepdim=True)).squeeze()
            aux2 = self.aux_classifier2(x3.mean(dim=0, keepdim=True)).squeeze()

            return logits, aux1, aux2

        else:
            # Fallback MLP for flat input
            x = batch_x.view(batch_x.size(0), -1)
            x = self.classifier(x)
            return x.squeeze(-1), None, None

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 100

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.001,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # Scheduler with patience
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=10, 
        verbose=True
    )

    criterion = nn.BCEWithLogitsLoss()

    # Tracking variables
    best_val_auc = 0.0
    patience_counter = 0
    patience = 25
    best_model_state = None

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for batch in train_loader:
            if isinstance(batch, tuple):
                data, labels = batch
                data = data.to(device)
                labels = labels.to(device).float()
            else:
                data = batch.to(device)
                labels = data.y.float()

            optimizer.zero_grad()

            logits, aux1, aux2 = model(data)
            loss = criterion(logits, labels)

            # Auxiliary losses for deep supervision
            if aux1 is not None and aux2 is not None:
                aux_loss1 = criterion(aux1.expand_as(labels), labels) * 0.3
                aux_loss2 = criterion(aux2.expand_as(labels), labels) * 0.3
                loss = loss + aux_loss1 + aux_loss2

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * len(labels)
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss = train_loss / len(train_loader.dataset)
        train_auc = roc_auc_score(train_labels, train_preds)
        train_acc = np.mean((np.array(train_preds) > 0.5) == np.array(train_labels))

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, tuple):
                    data, labels = batch
                    data = data.to(device)
                    labels = labels.to(device).float()
                else:
                    data = batch.to(device)
                    labels = data.y.float()

                logits, _, _ = model(data)
                loss = criterion(logits, labels)

                val_loss += loss.item() * len(labels)
                val_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = np.mean((np.array(val_preds) > 0.5) == np.array(val_labels))

        # Update scheduler
        scheduler.step(val_auc)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping and model checkpointing
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
            print(f"  Best Val AUC: {best_val_auc:.4f}, Patience: {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

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


