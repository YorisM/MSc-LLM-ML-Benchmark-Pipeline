
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as pyg_DataLoader
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import math

# ----------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(FourTopsDataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        super().__init__(events, pre, train, **kwargs)

# ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.particle_mean = None
        self.particle_std = None
        self.max_obj_id = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
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
        # Compute means and stds for normalization
        # Globals: E_T_miss, phi_Et_miss
        self.global_mean = X[:, :2].mean(dim=0)
        self.global_std = X[:, :2].std(dim=0)
        # Particles: for each obj, E, pt, eta, phi
        particle_features = []
        obj_ids = []
        for j in range(18):
            start = 2 + j * 5
            particle_features.append(X[:, start + 1:start + 5])  # E, pt, eta, phi
            obj_ids.append(X[:, start])  # obj_id
        particle_features = torch.cat(particle_features, dim=0)  # [18*N, 4]
        obj_ids = torch.cat(obj_ids, dim=0)
        # Mask zeros (padded, assume E=0 means padded)
        mask = particle_features[:, 0] != 0  # E != 0
        particle_features_masked = particle_features[mask]
        self.particle_mean = particle_features_masked.mean(dim=0)
        self.particle_std = particle_features_masked.std(dim=0)
        masked_obj_ids = obj_ids[mask]
        self.max_obj_id = masked_obj_ids.max() + 1  # for embedding
        return self

    def transform(self, X):
        datas = []
        for i in range(X.shape[0]):
            event = X[i]
            # Extract globals
            globals = (event[:2] - self.global_mean) / self.global_std
            # Extract particles (up to 18, skip if E==0 or pt<=0)
            particles = []
            for j in range(18):
                start = 2 + j * 5
                obj_id = event[start].int()
                features = event[start + 1:start + 5]  # E, pt, eta, phi
                if features[0] == 0 or features[1] <= 0:
                    continue
                particles.append((obj_id, features[0], features[1], features[2], features[3]))
            if len(particles) < 2:
                # If less than 2 particles, add a dummy or handle as single node, but skip for now
                continue
            num_particles = len(particles)
            # Normalize particle features
            node_features = []
            for p in particles:
                obj_id, E, pt, eta, phi = p
                norm_features = torch.tensor([obj_id.float() / self.max_obj_id, (E - self.particle_mean[0]) / self.particle_std[0],
                                             (pt - self.particle_mean[1]) / self.particle_std[1],
                                             (eta - self.particle_mean[2]) / self.particle_std[2],
                                             (phi - self.particle_mean[3]) / self.particle_std[3],
                                             globals[0], globals[1]], dtype=torch.float32)
                node_features.append(norm_features)  # [8]
            node_features = torch.stack(node_features)  # [num_particles, 8]
            # Build edges (fully connected)
            edge_index = []
            edge_attr = []
            for k in range(num_particles):
                for l in range(k + 1, num_particles):
                    pt1, eta1, phi1 = particles[k][1], particles[k][3], particles[k][4]  # pt, eta, phi
                    pt2, eta2, phi2 = particles[l][1], particles[l][3], particles[l][4]
                    delta_eta = eta1 - eta2
                    delta_phi = torch.remainder(phi1 - phi2 + math.pi, 2 * math.pi) - math.pi
                    delta_R = torch.sqrt(delta_eta ** 2 + delta_phi ** 2)
                    cosh_delta_eta = torch.cosh(delta_eta)
                    cos_delta_phi = torch.cos(delta_phi)
                    m_ij_sq = 2 * pt1 * pt2 * (cosh_delta_eta - cos_delta_phi)
                    m_ij = torch.sqrt(torch.clamp(m_ij_sq, min=0))  # [1]
                    edge_index.append([k, l])
                    edge_index.append([l, k])
                    edge_attr.append([delta_R, m_ij])
                    edge_attr.append([delta_R, m_ij])
            edge_index = torch.tensor(edge_index).t()  # [2, num_edges]
            edge_attr = torch.stack(edge_attr)  # [num_edges, 2]
            datas.append(Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr))
        return datas  # list of Data

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        from torch_geometric.nn import GATConv, global_mean_pool
        hidden_dim = 64
        num_heads = 8
        self.conv1 = GATConv(sample_object.x.size(1), hidden_dim, heads=num_heads, edge_dim=2)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, edge_dim=2)
        self.pool = global_mean_pool
        self.fc1 = nn.Linear(hidden_dim * num_heads, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, batch_x):
        x, edge_index, edge_attr, batch = batch_x.x, batch_x.edge_index, batch_x.edge_attr, batch_x.batch
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = self.pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze()

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model, train_loader, val_loader, epochs):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    best_auc = 0.0
    patience = 5
    counter = 0
    train_loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            batch_x, batch_y = batch.batch_x.to(device), batch.batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        scheduler.step()
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch_x, batch_y = batch.batch_x.to(device), batch.batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.float())
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        val_loss /= len(val_loader)
        val_acc = ((torch.tensor(val_preds) > 0.5).long() == torch.tensor(val_labels)).float().mean().item()
        val_auc = roc_auc_score(val_labels, val_preds)
        if val_auc > best_auc:
            best_auc = val_auc
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
        if counter >= patience:
            break
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
    model.load_state_dict(torch.load('best_model.pth'))
    return model, train_loss_list, val_loss_list, train_acc_list, val_acc_list

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


