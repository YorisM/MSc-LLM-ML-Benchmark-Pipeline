
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

# <start code template>
# ---------- IMPORTS ----------
import torch
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import GENConv, global_mean_pool
from sklearn.metrics import roc_auc_score
import numpy as np
import itertools

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.data_list = pre.transform(X, y)
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.etmiss_mean = None
        self.etmiss_std = None
        # Normalization for kinematics: E, pT (log scaled for better) but simple std
        self.kin_mean = None
        self.kin_std = None
        self.phi_mean = 0.0  # approximate
        self.phi_std = 1.0
        self.eta_mean = 0.0
        self.eta_std = 1.0

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",  # custom for PyG Data
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        # Compute norms from training data
        # Globals: Etmiss, phi_etmiss
        etmiss = X[:, 0:1]  # [N,1]
        phi_et = X[:, 1:2]
        self.etmiss_mean = etmiss.mean().item()
        self.etmiss_std = etmiss.std().item()
        # Kins: for each object, E, pT, eta, phi
        # Collect all valid E, pT, eta, phi
        Es, pTs, etas, phis = [], [], [], []
        for i in range(18):
            start = 2 + i*5
            obj_id = X[:, start]
            mask = obj_id != 0
            if mask.sum() > 0:
                Es.append(X[mask, start+1])  # E
                pTs.append(X[mask, start+2])  # pT
                etas.append(X[mask, start+3])  # eta
                phis.append(X[mask, start+4])  # phi, but toroidal, approx normal
        if Es:
            all_E = torch.cat(Es)
            all_pT = torch.cat(pTs)
            all_eta = torch.cat(etas)
            all_phi = torch.cat(phis)
            self.kin_mean = torch.cat([all_E, all_pT]).mean()
            self.kin_std = torch.cat([all_E, all_pT]).std()
            self.eta_mean = all_eta.mean()
            self.eta_std = all_eta.std()
            self.phi_mean = all_phi.mean()
            self.phi_std = all_phi.std()
        return self

    def transform(self, X, y):
        data_list = []
        for i in range(X.shape[0]):
            row = X[i]
            y_i = y[i]
            etmiss, phi_etmiss = row[0], row[1]
            objects = []
            for k in range(18):
                start = 2 + k*5
                obj_id = row[start].item()
                E = row[start+1].item()
                pT = row[start+2].item()
                eta = row[start+3].item()
                phi = row[start+4].item()
                if obj_id <= 0 or E <= 0:
                    break
                objects.append({'id': obj_id, 'E': E, 'pT': pT, 'eta': eta, 'phi': phi})
            if len(objects) < 2:
                # Pad or skip, but assume min 2
                continue
            num_nodes = len(objects)
            # Normalize
            g_et = (etmiss - self.etmiss_mean) / self.etmiss_std
            g_phi = (phi_etmiss - self.phi_mean) / self.phi_std
            g_vec = torch.tensor([g_et, g_phi], dtype=torch.float32)
            node_list = []
            for o in objects:
                e_n = (o['E'] - self.kin_mean) / self.kin_std
                pt_n = (o['pT'] - self.kin_mean) / self.kin_std
                eta_n = (o['eta'] - self.eta_mean) / self.eta_std
                phi_n = (o['phi'] - self.phi_mean) / self.phi_std
                node_list.append(torch.tensor([o['id'], e_n, pt_n, eta_n, phi_n, g_et, g_phi], dtype=torch.float32))
            x = torch.stack(node_list)  # [num_nodes, 7]
            # Edges: all pairs
            pairs = list(itertools.combinations(range(num_nodes), 2))
            edge_list = []
            edge_attr_list = []
            for idx, (i, j) in enumerate(pairs):
                p1 = objects[i]
                p2 = objects[j]
                # 4-vectors
                p1_x = p1['pT'] * np.cos(p1['phi'])
                p1_y = p1['pT'] * np.sin(p1['phi'])
                p1_z = p1['pT'] * np.sinh(p1['eta'])
                p2_x = p2['pT'] * np.cos(p2['phi'])
                p2_y = p2['pT'] * np.sin(p2['phi'])
                p2_z = p2['pT'] * np.sinh(p2['eta'])
                m_ij = np.sqrt((p1['E'] + p2['E'])**2 - (p1_x + p2_x)**2 - (p1_y + p2_y)**2 - (p1_z + p2_z)**2)
                # Normalize m_ij perhaps, but skip for now
                delta_phi = np.abs(p1['phi'] - p2['phi'])
                delta_phi = min(delta_phi, 2*np.pi - delta_phi)
                dR_ij = np.sqrt((p1['eta'] - p2['eta'])**2 + delta_phi**2)
                edge_list.append([i, j])
                edge_attr_list.append(torch.tensor([m_ij, dR_ij], dtype=torch.float32))
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()  # [2, num_edges]
                edge_attr = torch.stack(edge_attr_list)  # [num_edges, 2]
                # Add reverse? If undirected, add [j,i] with same attr
                edge_index_rev = torch.tensor([[j, i] for i, j in pairs], dtype=torch.long).t()
                edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
            else:
                edge_index = torch.empty(2, 0, dtype=torch.long)
                edge_attr = torch.empty(0, 2, dtype=torch.float32)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([y_i], dtype=torch.long))
            data_list.append(data)
        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object is Batch, x.shape [total_nodes, 7]
        in_features = 7
        hidden = 256
        self.conv1 = GENConv(in_features, hidden, aggr='softmax', learn_t=True, learn_p=True, edge_dim=2, msg_norm=True)
        self.conv2 = GENConv(hidden, hidden, aggr='softmax', learn_t=True, learn_p=True, edge_dim=2, msg_norm=True)
        self.conv3 = GENConv(hidden, hidden, aggr='softmax', learn_t=True, learn_p=True, edge_dim=2, msg_norm=True)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden, 1)

    def forward(self, G):
        x, edge_index, edge_attr, batch = G.x, G.edge_index, G.edge_attr, G.batch
        h = self.conv1(x, edge_index, edge_attr)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.conv3(h, edge_index, edge_attr)
        h = self.relu(h)
        out = global_mean_pool(h, batch)
        out = self.fc(out).squeeze(-1)
        return out

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20  # Increased for better AUC
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()
    best_auc = 0.0
    best_model_state = None
    patience = 5
    counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for G in train_loader:
            G = G.to(device)
            optimizer.zero_grad()
            out = model(G).squeeze()
            loss = criterion(out, G.y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * G.num_graphs
            pred = (out > 0).float()
            train_correct += (pred == G.y.float()).sum().item()
            train_total += G.num_graphs
        train_loss /= train_total
        train_acc = train_correct / train_total
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_out = []
        all_y = []
        with torch.no_grad():
            for G in val_loader:
                G = G.to(device)
                out = model(G).squeeze()
                loss = criterion(out, G.y.float())
                val_loss += loss.item() * G.num_graphs
                pred = (out > 0).float()
                val_correct += (pred == G.y.float()).sum().item()
                val_total += G.num_graphs
                all_out.extend(out.cpu().numpy())
                all_y.extend(G.y.cpu().numpy())
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(all_y, all_out) if len(set(all_y)) > 1 else 0.5
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f}; Val Loss {val_loss:.4f}, Acc {val_acc:.4f}, AUC {val_auc:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        scheduler.step()
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, train_accs, val_accs

# DO NOT execute the pipeline here – the harness will do that.
# <end code template>

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

