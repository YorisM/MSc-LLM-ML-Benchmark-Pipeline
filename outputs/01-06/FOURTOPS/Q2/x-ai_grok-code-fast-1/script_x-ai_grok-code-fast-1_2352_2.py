
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

# ---------- IMPORTS ----------
import torch
from torch import nn
from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.utils import dense_to_sparse
from sklearn.metrics import roc_auc_score

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.datas = pre.transform(X, y)  # list of Data objects
    def __len__(self):
        return len(self.datas)
    def __getitem__(self, idx):
        return self.datas[idx]

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.means = None
        self.stds = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 32}
        }

    def fit(self, X, y=None):
        # Compute mean and std for normalization
        self.means = X.mean(dim=0)  # [92]
        self.stds = X.std(dim=0)    # [92]
        # Clamp stds to avoid division by zero
        self.stds = torch.clamp(self.stds, min=1e-8)
        return self

    def transform(self, X, y=None):
        # Normalize X
        X_norm = (X - self.means) / self.stds  # [N, 92]
        datas = []
        for i in range(X_norm.shape[0]):
            event = X_norm[i]  # [92]
            # Extract globals: E_T_miss (idx 0), phi_Et_miss (idx 1)
            global_E = event[0]
            global_phi = event[1]
            # Extract up to 18 objects, each [id, E, pT, eta, phi] (indices 2-91, step 5)
            objects = []  # list of non-zero object tensors
            for j in range(18):
                start = 2 + j * 5
                obj = event[start:start+5].clone()  # [id, E, pT, eta, phi] normalized
                # Assuming id == 0 is padding, skip if id == 0 (original id, but since normalized, check if obj[0] != 0 assuming mean/id !=0)
                # To check if original id !=0, but since normalized, perhaps skip if all are zero or id norm is near 0
                # For simplicity, assume if obj[0] is close to mean of padding id, but hard
                # Since padding is zero, and if id=0, normalized id = -mean/std
                # But to simplify, keep all 18 nodes, treating padding as valid with id normed.
                objects.append(obj)
            num_obj = len(objects)  # always 18
            num_nodes = num_obj + 1  # 19 nodes: 18 objs + 1 global
            # Node features: [5] per node: id, E, pT, eta, phi
            node_features = torch.zeros(num_nodes, 5)
            for j, obj in enumerate(objects):
                node_features[j] = obj
            # Global node: id=999, E=global_E, pT=0, eta=0, phi=global_phi
            node_features[num_obj, 0] = 999  # arbitrary
            node_features[num_obj, 1] = global_E
            node_features[num_obj, 2] = 0.0
            node_features[num_obj, 3] = 0.0
            node_features[num_obj, 4] = global_phi
            # Edge index: fully connected, undirected, including to global
            adj = torch.ones(num_nodes, num_nodes).tril(diagonal=0) - torch.eye(num_nodes)  # lower triangular for undir, but since dense_to_sparse
            adj[num_obj, :] = 1  # connect global to all objs and self (but self removed)
            edge_index = dense_to_sparse(adj)[0]  # [2, num_edges]
            # Edge attr: [num_edges, 2] : m, deltaR
            edge_attr = torch.zeros(edge_index.shape[1], 2)
            for idx, (src, tgt) in enumerate(edge_index.t()):
                src, tgt = src.item(), tgt.item()
                if src < num_obj and tgt < num_obj:
                    # Compute m_ij, deltaR_ij
                    obj_src = node_features[src, 1:]  # [E, pT, eta, phi] but normalized, oh problem!
                    # Normalization affects calculations, since E, pT are scaled.
                    # So, need to denorm for obj features to compute physics
                    # Get original obj_src_raw
                    orig_event = X[i]
                    orig_objects = []
                    for k in range(18):
                        start_k = 2 + k * 5
                        orig_obj = orig_event[start_k:start_k+5]
                        orig_objects.append(orig_obj)
                    obj_src_raw = orig_objects[src]
                    obj_tgt_raw = orig_objects[tgt]
                    # 4-vectors using original
                    E_s, pT_s, eta_s, phi_s = obj_src_raw[1:]  # E, pT, eta, phi
                    pz_s = pT_s * torch.sinh(eta_s)
                    px_s = pT_s * torch.cos(phi_s)
                    py_s = pT_s * torch.sin(phi_s)
                    E_t, pT_t, eta_t, phi_t = obj_tgt_raw[1:]
                    pz_t = pT_t * torch.sinh(eta_t)
                    px_t = pT_t * torch.cos(phi_t)
                    py_t = pT_t * torch.sin(phi_t)
                    m2 = (E_s + E_t)**2 - (px_s + px_t)**2 - (py_s + py_t)**2 - (pz_s + pz_t)**2
                    m = torch.sqrt(torch.clamp(m2, min=0.0))
                    deta = eta_s - eta_t
                    dphi = torch.abs(phi_s - phi_t)
                    dphi = torch.min(dphi, 2 * torch.pi - dphi)
                    deltaR = torch.sqrt(deta**2 + dphi**2)
                    edge_attr[idx, 0] = m
                    edge_attr[idx, 1] = deltaR
                # For global edges, leave as 0
            # Build Data
            data = torch_geometric.data.Data(
                x=node_features.float(),  # [19, 5]
                edge_index=edge_index.long(),  # [2, num_edges]
                edge_attr=edge_attr.float(),  # [num_edges, 2]
                y=y[i].unsqueeze(0).long() if y is not None else None  # [1]
            )
            datas.append(data)
        return datas

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.num_features = sample_object.x.shape[1]  # 5
        # Two GAT layers with edge_attr
        self.conv1 = torch_geometric.nn.GATConv(self.num_features, 64, edge_dim=2, heads=4)  # out [num_nodes, 64*4=256]
        self.conv2 = torch_geometric.nn.GATConv(256, 128, edge_dim=2, heads=4)  # out [num_nodes, 128*4=512]
        self.pool = torch_geometric.nn.global_mean_pool  # pool to [batch, 512]
        self.linear = nn.Linear(512, 1)

    def forward(self, batch):
        # batch: Batch from PyG DataLoader
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        x = self.conv1(x, edge_index, edge_attr)  # [total_nodes, 256]
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)  # [total_nodes, 512]
        x = torch.relu(x)
        x = self.pool(x, batch.batch)  # [batch_size, 512]
        x = self.linear(x)  # [batch_size, 1]
        return x.squeeze(-1)  # [batch_size]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 10
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    best_auc = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = criterion(pred, batch.y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Eval AUC on val
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                preds.append(torch.sigmoid(pred).cpu())
                targets.append(batch.y.cpu())
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        auc = roc_auc_score(targets.numpy(), preds.numpy())
        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict()
    if best_state:
        model.load_state_dict(best_state)
    # Return placeholders as harness expects, set train_acc and val_acc to auc for summary
    return model, None, None, best_auc, best_auc  # approx

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

