
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
from sklearn.preprocessing import RobustScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_mean
import math

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.obj_scalers = [RobustScaler() for _ in range(18)]
        self.max_objects = 18
        self.obj_feature_size = 5
        self.global_feature_size = 2

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
        # Fit global features (E_T_miss, phi_Et_miss)
        global_features = X[:, :self.global_feature_size]
        self.scaler.fit(global_features)

        # Fit per-object features
        for i in range(self.max_objects):
            start_idx = self.global_feature_size + i * self.obj_feature_size
            end_idx = start_idx + self.obj_feature_size
            obj_features = X[:, start_idx:end_idx]
            # Only fit on non-zero objects (not padding)
            mask = obj_features[:, 0] != 0  # obj_id != 0
            if mask.any():
                self.obj_scalers[i].fit(obj_features[mask, 1:])  # Skip obj_id

        return self

    def transform(self, X):
        # Transform global features
        global_features = X[:, :self.global_feature_size]
        global_features = self.scaler.transform(global_features)

        # Transform per-object features and build graph data
        data_list = []
        batch_size = X.shape[0]

        for i in range(batch_size):
            event = X[i]
            global_feat = global_features[i]

            # Extract objects
            objects = []
            obj_ids = []
            for j in range(self.max_objects):
                start_idx = self.global_feature_size + j * self.obj_feature_size
                end_idx = start_idx + self.obj_feature_size
                obj_data = event[start_idx:end_idx]
                obj_id = int(obj_data[0].item())
                if obj_id != 0:  # Not padding
                    obj_ids.append(obj_id)
                    # Scale kinematic features (E, p_T, eta, phi)
                    kinematic = obj_data[1:].reshape(1, -1)
                    kinematic = self.obj_scalers[j].transform(kinematic)
                    objects.append(kinematic.squeeze(0))

            if not objects:
                # Handle empty events (shouldn't happen in this dataset)
                objects = [np.zeros(4)]
                obj_ids = [1]

            objects = np.array(objects)  # [num_objects, 4]
            num_objects = objects.shape[0]

            # Create node features: [obj_id, E, p_T, eta, phi]
            node_features = np.zeros((num_objects, 5))
            node_features[:, 0] = obj_ids
            node_features[:, 1:] = objects

            # Create edge_index (fully connected graph)
            edge_index = []
            for src in range(num_objects):
                for dst in range(num_objects):
                    if src != dst:
                        edge_index.append([src, dst])
            edge_index = np.array(edge_index).T  # [2, num_edges]

            # Compute pairwise features for edges
            edge_attr = []
            for src, dst in edge_index.T:
                src_feat = node_features[src, 1:]
                dst_feat = node_features[dst, 1:]

                # Invariant mass (approximate)
                E1, pt1, eta1, phi1 = src_feat
                E2, pt2, eta2, phi2 = dst_feat

                # Convert to 4-vector
                px1 = pt1 * math.cos(phi1)
                py1 = pt1 * math.sin(phi1)
                pz1 = pt1 * math.sinh(eta1)
                m1_sq = E1**2 - (px1**2 + py1**2 + pz1**2)

                px2 = pt2 * math.cos(phi2)
                py2 = pt2 * math.sin(phi2)
                pz2 = pt2 * math.sinh(eta2)
                m2_sq = E2**2 - (px2**2 + py2**2 + pz2**2)

                # Combined 4-vector
                E = E1 + E2
                px = px1 + px2
                py = py1 + py2
                pz = pz1 + pz2
                m_inv_sq = E**2 - (px**2 + py**2 + pz**2)
                m_inv = math.sqrt(max(0, m_inv_sq)) if m_inv_sq > 0 else 0

                # Delta R
                delta_eta = eta1 - eta2
                delta_phi = (phi1 - phi2 + math.pi) % (2 * math.pi) - math.pi
                delta_R = math.sqrt(delta_eta**2 + delta_phi**2)

                edge_attr.append([m_inv, delta_R])

            edge_attr = np.array(edge_attr)

            # Create PyG Data object
            data = Data(
                x=torch.from_numpy(node_features).float(),
                edge_index=torch.from_numpy(edge_index).long(),
                edge_attr=torch.from_numpy(edge_attr).float(),
                global_feat=torch.from_numpy(global_feat).float(),
                num_nodes=num_objects
            )
            data_list.append(data)

        return data_list

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.global_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.conv1 = GATv2Conv(64 + 32, 128, heads=4, concat=True, edge_dim=32)
        self.conv2 = GATv2Conv(128 * 4, 128, heads=4, concat=True, edge_dim=32)
        self.conv3 = GATv2Conv(128 * 4, 128, heads=4, concat=False, edge_dim=32)

        self.readout = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, batch_x):
        # batch_x is a Batch object from PyG
        x = batch_x.x  # [total_nodes, 5]
        edge_index = batch_x.edge_index  # [2, total_edges]
        edge_attr = batch_x.edge_attr  # [total_edges, 2]
        batch = batch_x.batch  # [total_nodes]
        global_feat = batch_x.global_feat  # [batch_size, 2]

        # Encode nodes
        node_feat = self.node_encoder(x)  # [total_nodes, 64]

        # Encode edges
        edge_feat = self.edge_encoder(edge_attr)  # [total_edges, 32]

        # Encode global features
        global_feat = self.global_encoder(global_feat)  # [batch_size, 32]

        # GNN layers
        x = torch.cat([node_feat, edge_feat[batch_x.edge_index[0]]], dim=1)  # [total_nodes, 64+32]
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_feat))
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_feat))
        x = F.relu(self.conv3(x, edge_index, edge_attr=edge_feat))  # [total_nodes, 128]

        # Global pooling
        graph_emb = scatter_mean(x, batch, dim=0)  # [batch_size, 128]

        # Combine with global features
        combined = torch.cat([graph_emb, global_feat], dim=1)  # [batch_size, 128+32]

        # Readout
        out = self.readout(combined)  # [batch_size, 1]

        return out.squeeze(1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    criterion = nn.BCELoss()

    best_auc = 0
    best_model = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.y.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == batch.y.float()).sum().item()
            train_total += batch.y.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

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
                outputs = model(batch)
                loss = criterion(outputs, batch.y.float())
                val_loss += loss.item()

                predicted = (outputs > 0.5).float()
                val_correct += (predicted == batch.y.float()).sum().item()
                val_total += batch.y.size(0)

                all_probs.extend(outputs.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc_score = roc_auc_score(all_labels, all_probs)
        except:
            auc_score = 0.5

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, AUC: {auc_score:.4f}")

        # Early stopping based on AUC
        scheduler.step(auc_score)
        if auc_score > best_auc:
            best_auc = auc_score
            best_model = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                print("Early stopping triggered")
                break

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, train_losses, val_losses, train_accs, val_accs

def make_preprocessor():
    return MyPreprocessor()

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


