
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ---------- IMPORTS ----------
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_add_pool
import math

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.edge_scaler = StandardScaler()

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomGraphDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        # Flatten event features for scaling (global features + object kinematics)
        # We'll scale the continuous features: E_T^miss, phi_Et_miss, E, pT, eta, phi
        # Object IDs will be one-hot encoded and not scaled
        n_events = X.shape[0]
        continuous_features = []

        for i in range(n_events):
            event = X[i]
            # Global features
            continuous_features.append(event[:2].reshape(1, -1))

            # Object features (skip object IDs which are at indices 0, 5, 10, ... in object blocks)
            obj_block = event[2:].reshape(18, 5)
            # Take E, pT, eta, phi (skip obj_id at column 0)
            if obj_block[:, 1].sum() > 0:  # if there are real objects
                mask = obj_block[:, 1] > 0  # E > 0 indicates real object
                cont_obj = obj_block[mask, 1:5]  # shape [n_real, 4]
                continuous_features.append(cont_obj)

        all_cont = np.vstack(continuous_features)
        self.scaler.fit(all_cont)

        # Also fit edge feature scaler for pairwise features
        edge_features = []
        for i in range(min(10000, n_events)):  # sample for efficiency
            event = X[i]
            obj_block = event[2:].reshape(18, 5)
            mask = obj_block[:, 1] > 0
            real_objs = obj_block[mask]
            n_real = real_objs.shape[0]

            if n_real >= 2:
                # Compute pairwise features for a subset of pairs
                for j in range(n_real):
                    for k in range(j+1, min(j+6, n_real)):  # limit pairs for efficiency
                        dr, mass = self._compute_pairwise_features(
                            real_objs[j], real_objs[k])
                        edge_features.append([dr, mass])

        if edge_features:
            edge_features = np.array(edge_features)
            self.edge_scaler.fit(edge_features)

        return self

    def _compute_pairwise_features(self, obj1, obj2):
        # obj: [id, E, pT, eta, phi]
        eta1, phi1 = obj1[3], obj1[4]
        eta2, phi2 = obj2[3], obj2[4]

        # DeltaR
        deta = eta1 - eta2
        dphi = (phi1 - phi2 + math.pi) % (2*math.pi) - math.pi
        dr = math.sqrt(deta**2 + dphi**2)

        # Invariant mass (in GeV)
        E1, E2 = obj1[1] / 1000., obj2[1] / 1000.  # MeV to GeV

        # Convert to Cartesian momentum
        px1 = obj1[2] * math.cos(phi1) / 1000.
        py1 = obj1[2] * math.sin(phi1) / 1000.
        pz1 = obj1[2] * math.sinh(eta1) / 1000.

        px2 = obj2[2] * math.cos(phi2) / 1000.
        py2 = obj2[2] * math.sin(phi2) / 1000.
        pz2 = obj2[2] * math.sinh(eta2) / 1000.

        # Sum four-vectors
        E = E1 + E2
        px = px1 + px2
        py = py1 + py2
        pz = pz1 + pz2

        inv_mass = math.sqrt(max(0, E**2 - (px**2 + py**2 + pz**2)))
        return dr, inv_mass

    def transform(self, X):
        graphs = []
        for i in range(X.shape[0]):
            event = X[i]

            # Global features
            global_feats = event[:2]  # [E_T^miss, phi]
            global_feats_scaled = self.scaler.transform(global_feats.reshape(1, -1)).flatten()

            # Object features
            obj_block = event[2:].reshape(18, 5)  # [18, 5]

            # Find real objects (E > 0)
            mask = obj_block[:, 1] > 0
            real_objs = obj_block[mask]  # [n_real, 5]
            n_real = real_objs.shape[0]

            if n_real == 0:
                # Create dummy graph with one node
                node_features = torch.zeros((1, 11), dtype=torch.float32)
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                edge_attr = torch.zeros((1, 2), dtype=torch.float32)
            else:
                # Node features: [E, pT, eta, phi (scaled), one-hot ID (6 dim), global_feats (2 dim)]
                cont_features = real_objs[:, 1:5]  # [n_real, 4]
                cont_scaled = self.scaler.transform(cont_features)

                # One-hot encode object IDs (0-11 are possible particle types)
                obj_ids = real_objs[:, 0].long()
                one_hot = torch.zeros((n_real, 6), dtype=torch.float32)
                valid_ids = torch.clamp(obj_ids, 0, 5)
                one_hot.scatter_(1, valid_ids.unsqueeze(1), 1)

                # Global features repeated per node
                global_repeated = torch.from_numpy(global_feats_scaled).unsqueeze(0).repeat(n_real, 1)

                node_features = torch.cat([
                    torch.from_numpy(cont_scaled).float(),
                    one_hot,
                    global_repeated
                ], dim=1)  # [n_real, 4+6+2=12]

                # Build fully connected graph edges
                edge_list = []
                edge_features = []

                for j in range(n_real):
                    for k in range(n_real):
                        if j != k:
                            edge_list.append([j, k])
                            dr, mass = self._compute_pairwise_features(
                                real_objs[j], real_objs[k])
                            edge_features.append([dr, mass])

                if edge_features:
                    edge_features = np.array(edge_features)
                    edge_features_scaled = self.edge_scaler.transform(edge_features)
                    edge_attr = torch.from_numpy(edge_features_scaled).float()
                else:
                    edge_attr = torch.zeros((1, 2), dtype=torch.float32)
                    edge_list = [[0, 0]]

                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

            # Create PyG Data object
            graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr
            )
            graphs.append(graph)

        return graphs

def make_preprocessor():
    return MyPreprocessor()

# Custom dataset for PyG graphs
class CustomGraphDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.graphs = pre.transform(X)  # list of Data objects
        self.y = torch.as_tensor(y).long()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.y[idx]

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object is a Data object from the dataset
        node_dim = sample_object.x.shape[1]
        edge_dim = sample_object.edge_attr.shape[1]

        # GIN layers with edge features
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(node_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 128)
            ), train_eps=True)

        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256)
            ), train_eps=True)

        self.conv3 = GINConv(
            nn.Sequential(
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256)
            ), train_eps=True)

        # Edge feature processing
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        # Attention mechanism for node aggregation
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # batch_x is a PyG Batch object
        x, edge_index, edge_attr, batch = batch_x.x, batch_x.edge_index, batch_x.edge_attr, batch_x.batch

        # Process edge features and incorporate into node features
        if edge_attr.shape[1] > 0:
            edge_features = self.edge_mlp(edge_attr)
            # Aggregate edge features to nodes
            row, col = edge_index
            edge_to_node = torch.zeros_like(x[:, :128])
            edge_to_node = edge_to_node.scatter_add(0, row.unsqueeze(-1).expand(-1, 128), edge_features)
            edge_to_node = edge_to_node.scatter_add(0, col.unsqueeze(-1).expand(-1, 128), edge_features)
            x = torch.cat([x, edge_to_node], dim=1)

        # Graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)  # [total_nodes, 256]

        # Attention-based global pooling
        attn_weights = torch.softmax(self.attention(x), dim=0)
        graph_emb = global_add_pool(x * attn_weights, batch)  # [batch_size, 256]

        # Classification
        out = self.classifier(graph_emb).squeeze(-1)  # [batch_size]
        return out

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 100

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0
    best_model_state = None
    patience_counter = 0
    patience = 15

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch = batch.to(device)
            y = batch.y.float()

            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * y.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                y = batch.y.float()

                logits = model(batch)
                loss = criterion(logits, y)

                val_loss += loss.item() * y.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu())
                all_labels.append(y.cpu())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Compute AUC
        from sklearn.metrics import roc_auc_score
        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()
        val_auc = roc_auc_score(all_labels, all_probs)

        # Update scheduler based on AUC
        scheduler.step(val_auc)

        # Early stopping based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

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

    # Build batch and check
    first_batch = next(iter(train_loader))
    mode = detect_and_assert_lane_fourtops(spec, first_batch)
    view = make_view_by_lane_fourtops(mode, first_batch, device)

    # Build model
    model = make_model(view.batch_x).to(device)

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
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

