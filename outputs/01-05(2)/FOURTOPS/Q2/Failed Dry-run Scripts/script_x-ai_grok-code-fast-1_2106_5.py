
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
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
# <LLM: Import modules>
import torch
import torch.nn as nn
import torch_geometric
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(torch.utils.data.Dataset):
    # REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:CustomDataset"
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.graph_data = pre.transform(X)
        self.y = y
    def __len__(self):
        return len(self.graph_data)
    def __getitem__(self, idx):
        data = self.graph_data[idx]
        return torch_geometric.data.Data(**data, y=self.y[idx])

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # REQUIREMENTS
    #   - IMPORTANT: All state must be picklable with the std-lib pickle module.
    #   - May allocate NumPy arrays or Torch tensors internally, but: transform() must be deterministic.
    #   - Store only derived parameters needed for transform i.e. do not store the raw data itself in the preprocessor object.

    # TIPS
    #   - When modifying data features or feature engineering: annotate tensor size as comments after 
    #   - each tensor operation to reduce dimension mismatches.

    # DATA SPECIFICS
    #    Total flat length per event (X_train & X_val): 92
    #    Index  0 :  missing-ET magnitude  (E_T_miss)
    #    Index  1 :  missing-ET azimuth    (phi_Et_miss)
    #    Indices  2-6  : object 1  ->  obj_1, E_1, p_T1, eta_1, phi_1
    #    Indices  7-11 : object 2  ->  obj_2, E_2 , p_T_2 , eta_2 , phi_2
    #    ...
    #    Indices 87-91 : object 18 ->  obj_18, E_18 , p_T_18 , eta_18 , phi_18
    #    Global features       = 2
    #    Per-object slice size = 5
    #    Max objects encoded   = 18

    # <LLM: Write code to preprocess the data> 

    def __init__(self):
        # <LLM: Define and initialize any stateful components here>
        self.scaler = StandardScaler()

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this. Configure as you please.
        return {
            "dataset_builder": "llm_script:CustomDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",     # or torch_geometric.loader:DataLoader
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed.
            "collate": None,

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False, 
                                "batch_size": 512} # Or whatever you want
        }

    def fit(self, X, y=None):
        # <LLM: Extract statistics for transform>
        all_node_features = []
        for event in X:
            event = event.view(-1)  # [92]
            objs = event[2:].view(-1, 5)  # [18, 5]
            valid_mask = objs[:, 0] != 0  # obj_id != 0
            if valid_mask.any():
                feats = objs[valid_mask, 1:].numpy()  # [n, 4] kinematic features
                all_node_features.extend(feats.tolist())
        self.scaler.fit(np.array(all_node_features))
        return self

    def transform(self, X):
        # <LLM: Apply pre-processing logic>
        graphs = []
        for event in X:
            event = event.view(-1)  # [92]
            global_feat = event[:2]  # [2]
            objs = event[2:].view(-1, 5)  # [18, 5]
            valid_mask = objs[:, 0] != 0  # obj_id != 0
            n_nodes = valid_mask.sum().item()

            if n_nodes == 0:
                # Skip empty events, though unlikely
                graphs.append({'x': torch.zeros(1, 4).float(), 'edge_index': torch.empty(2, 0, dtype=torch.long), 'edge_attr': torch.empty(0, 2).float()})
                continue

            # Node features: kinematic [E, pT, eta, phi], scaled
            x_unscaled = objs[valid_mask, 1:]  # [n_nodes, 4]
            x = torch.tensor(self.scaler.transform(x_unscaled.numpy())).float()  # [n_nodes, 4]

            # Add global node: [E_T_miss, phi_E_T_miss, 0, 0]
            x_global = torch.zeros(1, 4).float()
            x_global[0, 0] = global_feat[0]
            x_global[0, 1] = global_feat[1]
            x = torch.cat([x, x_global], dim=0)  # [n_nodes+1, 4]
            n_nodes_final = n_nodes + 1
            global_idx = n_nodes

            # Edges: fully connected among objects + global connections
            edge_list = []
            edge_attr_list = []

            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:  # Fully connected, remove self-loops for objects, but global has self?
                        eta1, phi1 = x_unscaled[i, 2], x_unscaled[i, 3]
                        eta2, phi2 = x_unscaled[j, 2], x_unscaled[j, 3]
                        deta = eta1 - eta2
                        dphi = self._phi_to_pi(phi1 - phi2)
                        deltaR = torch.sqrt(deta**2 + dphi**2)

                        # Invariant mass
                        p1 = self._four_momenta(x_unscaled[i])
                        p2 = self._four_momenta(x_unscaled[j])
                        m_ij = torch.sqrt((p1[0] + p2[0])**2 - torch.sum((p1[1:] + p2[1:])**2))

                        edge_list.append([i, j])
                        edge_attr_list.append([m_ij.item(), deltaR.item()])

            # Global connections to all objects
            for k in range(n_nodes):
                edge_list.append([k, global_idx])
                edge_list.append([global_idx, k])
                edge_attr_list.append([0.0, 0.0])  # Dummy edge attrs for global

            if edge_list:
                edge_index = torch.tensor(edge_list).t().contiguous()  # [2, E]
                edge_attr = torch.tensor(edge_attr_list).float()  # [E, 2]
            else:
                edge_index = torch.empty(2, 0, dtype=torch.long)
                edge_attr = torch.empty(0, 2).float()

            graphs.append({'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr})

        return graphs  # List of dicts

    def _phi_to_pi(self, dphi):
        while dphi > np.pi:
            dphi -= 2 * np.pi
        while dphi < -np.pi:
            dphi += 2 * np.pi
        return dphi

    def _four_momenta(self, feat):
        # feat: [E, pT, eta, phi]
        E, pT, eta, phi = feat
        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)
        return torch.tensor([E, px, py, pz])

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
# MODEL I/O BATCH CONTRACT (CHOOSE ONE LANE)
# You MUST choose exactly one of the two supported input lanes and keep it consistent:
#
# --- LANE A: Torch dense batch (default) ---
# Loader:
#   - loader_class: "torch.utils.data:DataLoader"
#   - collate: None
# Batch from DataLoader:
#   (Xb, yb) where
#     Xb: FloatTensor[B, F]
#     yb: LongTensor[B] (or [B,1])
# Model forward:
#   out = model(Xb)
#   out must be FloatTensor[B] or FloatTensor[B,1] (logits or probabilities)
#
# --- LANE B: PyTorch Geometric (PyG) graphs ---
# Loader:
#   - loader_class: "torch_geometric.loader:DataLoader"
#   - collate: None
# Dataset samples MUST be torch_geometric.data.Data with at least:
#   data.x : FloatTensor[N_i, F]
#   data.edge_index : LongTensor[2, E_i]   (or equivalent; your model can build edges too)
#   data.y : LongTensor[1]                (GRAPH-LEVEL label for the event!)
# Batch from DataLoader:
#   G : torch_geometric.data.Batch (has G.x, G.edge_index, G.batch, and G.y)
# Model forward:
#   out = model(G)
#   out must be FloatTensor[num_graphs] or FloatTensor[num_graphs,1] (logits or probabilities)
#
# Any other batch shapes are NOT supported.

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # <LLM: Define and initialize any stateful components here>
        in_channels = 4  # Node features: 4
        self.conv1 = torch_geometric.nn.GATConv(in_channels, 64, edge_dim=2)
        self.conv2 = torch_geometric.nn.GATConv(64, 64, edge_dim=2)
        self.linear = nn.Linear(64, 1)
        self.pool = torch_geometric.nn.global_mean_pool

    # <LLM: optionally build extra layers here>

    def forward(self, batch):
        # IMPORTANT output must be logits/probabilities per event
        # Batch is torch_geometric.data.Batch
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = torch.relu(x)
        # Global pooling [B,64]
        x = self.pool(x, batch_idx)  # [num_graphs, 64]
        out = self.linear(x).squeeze(-1)  # [num_graphs]
        return out

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 10   # <LLM: adjust if you wish>
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS
    #   - Must return: trained_model, train_loss, val_loss, train_acc, val_acc
    #   - Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).

    # <LLM: Write code to define training loop, use the code above>
    # <LLM: Implement early stopping if possible>
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_auc = 0.0
    patience = 5
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_train_loss = 0.0
        train_preds = []
        train_labels = []
        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y.float())
            epoch_train_loss += loss.item()

            probs = torch.sigmoid(out).detach().cpu().numpy()
            train_preds.extend(probs)
            train_labels.extend(batch.y.cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_labels, train_preds)
        train_acc = np.mean((np.array(train_preds) > 0.5).astype(int) == train_labels)

        # Val
        model.eval()
        epoch_val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y.float())
                epoch_val_loss += loss.item()

                probs = torch.sigmoid(out).detach().cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(batch.y.cpu().numpy())

        epoch_val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = np.mean((np.array(val_preds) > 0.5).astype(int) == val_labels)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        scheduler.step()

    # Load best model
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    return model, train_losses[-1], val_losses[-1], train_acc, val_acc

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

