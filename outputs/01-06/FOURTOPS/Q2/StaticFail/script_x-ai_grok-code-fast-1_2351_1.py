
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
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
from torch_geometric.data import Data
from torch_geometric import nn as pyg_nn
from sklearn.metrics import roc_auc_score
import math

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        self.data_list, self.labels = events
        self.labels = torch.as_tensor(self.labels) if not torch.is_tensor(self.labels) else self.labels
        for i, data in enumerate(self.data_list):
            data.y = self.labels[i]
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        pass

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",   
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 128,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, 
                                "batch_size": 128}
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data_list = []
        for i in range(X.shape[0]):
            x = X[i]
            et_miss = x[0]
            phi_et = x[1]
            objects = []
            for j in range(18):
                start = 2 + 5 * j
                obj_id = x[start]
                if obj_id == 0.0: continue
                E = x[start + 1]
                pT = x[start + 2]
                eta = x[start + 3]
                phi = x[start + 4]
                objects.append((obj_id, E, pT, eta, phi))
            num_nodes = len(objects)
            node_features = torch.tensor([[obj_id, E, pT, eta, phi] for obj_id, E, pT, eta, phi in objects], dtype=torch.float32)
            if num_nodes < 2:
                edge_index = torch.empty(2, 0, dtype=torch.long)
                edge_attr = torch.empty(0, 2, dtype=torch.float32)
            else:
                idx = torch.arange(num_nodes)
                edge_index_u = torch.combinations(idx, r=2, with_replacement=False).t()
                edge_index = torch.cat([edge_index_u, edge_index_u[[1, 0]]], dim=1)
                edge_attr_list = []
                for pair_idx in range(edge_index_u.shape[1]):
                    i, j = edge_index_u[0, pair_idx].item(), edge_index_u[1, pair_idx].item()
                    _, Ei, pTi, etai, phii = objects[i]
                    _, Ej, pTj, etaj, phij = objects[j]
                    pxi = pTi * math.cos(phii)
                    pyi = pTi * math.sin(phii)
                    pzi = pTi * math.sinh(etai)
                    pxj = pTj * math.cos(phij)
                    pyj = pTj * math.sin(phij)
                    pzj = pTj * math.sinh(etaj)
                    p_tot_x = pxi + pxj
                    p_tot_y = pyi + pyj
                    p_tot_z = pzi + pzj
                    E_tot = Ei + Ej
                    m2 = E_tot**2 - p_tot_x**2 - p_tot_y**2 - p_tot_z**2
                    m_ij = math.sqrt(max(0.0, m2.item()))
                    deta = etai - etaj
                    dphi = phii - phij
                    dphi_wrapped = math.atan2(math.sin(dphi), math.cos(dphi))
                    dr = math.sqrt(deta**2 + dphi_wrapped**2)
                    edge_attr_list.append([m_ij, dr])
                edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
            data_list.append(Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr))
        return data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.node_input_dim = sample_object.x.shape[1]
        self.edge_dim = sample_object.edge_attr.shape[1]
        self.conv1 = pyg_nn.TransformerConv(in_channels=self.node_input_dim, out_channels=128, edge_dim=self.edge_dim, heads=8)
        self.conv2 = pyg_nn.TransformerConv(128, 128, edge_dim=self.edge_dim, heads=8)
        self.conv3 = pyg_nn.TransformerConv(128, 128, edge_dim=self.edge_dim, heads=8)
        self.pool = pyg_nn.global_mean_pool
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, batch):
        x = batch.x
        x = self.conv1(x, batch.edge_index, batch.edge_attr)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, batch.edge_index, batch.edge_attr)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, batch.edge_index, batch.edge_attr)
        x = torch.relu(x)
        x = self.pool(x, batch.batch)
        x = torch.relu(self.fc1(x))
        out = self.fc2(x).squeeze(-1)
        return out

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()
    best_auc = 0.0
    patience = 15
    wait = 0
    best_model = None
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            probs = torch.sigmoid(out)
            preds = probs > 0.5
            correct += (preds == batch.y).sum().item()
            total += batch.y.size(0)
            all_preds.extend(probs.detach().cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
        acc = correct / total if total > 0 else 0.0
        auc = roc_auc_score(all_labels, all_preds)
        train_loss_list.append(total_loss / len(train_loader))
        train_acc_list.append(acc)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y.float())
                val_loss += loss.item()
                probs = torch.sigmoid(out)
                preds = probs > 0.5
                val_correct += (preds == batch.y).sum().item()
                val_total += batch.y.size(0)
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(batch.y.cpu().numpy())
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_auc = roc_auc_score(val_labels, val_preds) if len(val_labels) > 1 and len(set(val_labels)) > 1 else 0.5
        val_loss_list.append(val_loss / len(val_loader))
        val_acc_list.append(val_acc)
        scheduler.step()
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict().copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_model)
    return model, train_loss_list, val_loss_list, train_acc_list, val_acc_listoprene.deserialize

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

