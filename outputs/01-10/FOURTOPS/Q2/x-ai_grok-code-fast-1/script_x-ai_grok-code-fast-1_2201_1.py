
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

# -------------------------- START OF LLM BLOCK ------------------------------
# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
# <LLM: Import modules>
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        if pre is not None:
            self.datas = pre.transform(X, y)
        else:
            self.datas = [torch_geometric.data.Data(x=torch.empty(0, 8), edge_index=torch.empty(2,0), edge_attr=torch.empty(0,2, dtype=torch.float), y=torch.tensor(y_i, dtype=torch.long)) for y_i in y]
    def __len__(self):
        return len(self.datas)
    def __getitem__(self, idx):
        return self.datas[idx]

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
        pass

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this. Configure as you please.
        return {
            "dataset_builder": "llm_script:CustomDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",     # or torch.utils.data:DataLoader
            "batch_size": 64,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed.
            "collate": None,

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False, 
                                "batch_size": 64} # Or whatever you want
        }

    def fit(self, X, y=None):
        # <LLM: Extract statistics for transform>
        return self

    def transform(self, X, y):
        # <LLM: Apply pre-processing logic>
        # X: [N, 92], y: [N]
        datas = []
        for i in range(X.shape[0]):
            event = X[i]  # [92]
            y_i = y[i]
            global_et = event[0]
            global_phi = event[1]
            objs = []
            for j in range(18):
                start = 2 + j * 5
                obj_type_int = int(event[start].item())
                if obj_type_int == 0:
                    continue
                E = event[start + 1]
                pT = event[start + 2]
                eta = event[start + 3]
                phi = event[start + 4]
                objs.append((obj_type_int, E, pT, eta, phi))
            num_objs = len(objs)
            if num_objs == 0:
                # Handle rare empty event
                node_features = torch.zeros(1, 5)  # [1, 5] node for global
                edge_index = torch.empty(2, 0, dtype=torch.long)  # [2, 0]
                edge_attr = torch.empty(0, 2)  # [0, 2]
            else:
                # Node features: [num_objs, 5] - obj_type, E, pT, eta, phi
                node_features = torch.zeros(num_objs, 5)  # [num_objs, 5]
                for k, (t, E, pT, eta, phi) in enumerate(objs):
                    node_features[k] = torch.tensor([t, E, pT, eta, phi])  # [5]
                # Add global node: [1, 5]
                global_feat = torch.tensor([100.0, global_et, 0.0, 0.0, global_phi])  # type 100 for global
                node_features = torch.cat([node_features, global_feat.unsqueeze(0)], dim=0)  # [num_objs+1, 5]
                num_nodes = num_objs + 1
                global_idx = num_nodes - 1
                # Edges: full between objs, and global to objs, with directions
                edges = []
                edge_features = []
                # Between objs: for each pair a<b, add undirected with feat
                for a in range(num_objs):
                    for b in range(a + 1, num_objs):
                        # Compute edge features
                        a_feat = objs[a]
                        b_feat = objs[b]
                        t1, E1, pT1, eta1, phi1 = a_feat
                        t2, E2, pT2, eta2, phi2 = b_feat
                        # 4-momenta components
                        px1 = pT1 * torch.cos(phi1)
                        py1 = pT1 * torch.sin(phi1)
                        pz1 = pT1 * torch.sinh(eta1)
                        px2 = pT2 * torch.cos(phi2)
                        py2 = pT2 * torch.sin(phi2)
                        pz2 = pT2 * torch.sinh(eta2)
                        sum_E = E1 + E2
                        sum_px = px1 + px2
                        sum_py = py1 + py2
                        sum_pz = pz1 + pz2
                        m_sq = sum_E**2 - sum_px**2 - sum_py**2 - sum_pz**2
                        m_ij = torch.sqrt(torch.clamp(m_sq, min=0.0))
                        delta_eta = eta1 - eta2
                        delta_phi = torch.remainder(phi1 - phi2, 2 * torch.pi)
                        delta_phi = torch.where(delta_phi > torch.pi, delta_phi - 2 * torch.pi, delta_phi)
                        delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)
                        edge_feat = torch.tensor([delta_R, m_ij])  # [2]
                        edges.append([a, b])
                        edge_features.append(edge_feat)
                        edges.append([b, a])
                        edge_features.append(edge_feat)
                # Global to objs: undirected with dummy feat [0,0]
                for a in range(num_objs):
                    edges.append([global_idx, a])
                    edge_features.append(torch.tensor([0.0, 0.0]))  # [2]
                    edges.append([a, global_idx])
                    edge_features.append(torch.tensor([0.0, 0.0]))  # [2]
                edge_index = torch.tensor(edges, dtype=torch.long).t()  # [2, num_edges]
                edge_attr = torch.stack(edge_features)  # [num_edges, 2]
            data = torch_geometric.data.Data(x=node_features.clone().detach(), edge_index=edge_index.clone().detach(), edge_attr=edge_attr.clone().detach(), y=torch.tensor(y_i, dtype=torch.long))
            datas.append(data)
        return datas

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
        # sample_object is Batch, sample_object.x.shape[1] = 5 (obj_type, E, pT, eta, phi)
        in_dim = 5
        embed_dim = 3
        self.embedding = nn.Embedding(101, embed_dim)  # obj_type 0-100, global 100
        hidden_dim = 32
        self.conv1 = GATConv(in_dim - 1 + embed_dim, hidden_dim, edge_dim=2)
        self.conv2 = GATConv(hidden_dim, hidden_dim*2, edge_dim=2)
        self.pool = global_mean_pool
        self.fc = nn.Linear(hidden_dim*2, 1)

    # <LLM: optionally build extra layers here>

    def forward(self, batch):
        # IMPORTANT output must be logits/probabilities per event
        # batch is torch_geometric.data.Batch
        x = batch.x  # [total_nodes, 5]
        emb = self.embedding(x[:, 0].long())  # [total_nodes, embed_dim]
        feat = torch.cat([emb, x[:, 1:]], dim=1)  # [total_nodes, embed_dim + 4]
        x = feat
        x = self.conv1(x, batch.edge_index, batch.edge_attr)  # [total_nodes, hidden_dim]
        x = F.relu(x)
        x = self.conv2(x, batch.edge_index, batch.edge_attr)  # [total_nodes, hidden_dim*2]
        x = F.relu(x)
        x = self.pool(x, batch.batch)  # [batch_size, hidden_dim*2]
        out = self.fc(x).squeeze(1)  # [batch_size]
        return out

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20   # <LLM: adjust if you wish>
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS
    #   - Must return: trained_model, train_loss, val_loss, train_acc, val_acc
    #   - Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).

    # <LLM: Write code to define training loop, use the code above>
    # <LLM: Implement early stopping if possible>
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
    wait = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == batch.y.float()).sum().item()
            total_samples += batch.num_graphs
        train_loss = total_loss / len(train_loader.dataset)
        train_acc = total_correct / total_samples
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        model.eval()
        total_val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = criterion(logits, batch.y.float())
                total_val_loss += loss.item() * batch.num_graphs
                preds = (torch.sigmoid(logits) > 0.5).float()
                total_val_correct += (preds == batch.y.float()).sum().item()
                total_val_samples += batch.num_graphs
        val_loss = total_val_loss / len(val_loader.dataset)
        val_acc = total_val_correct / total_val_samples
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, train_loss_list, val_loss_list, train_acc_list, val_acc_list

# DO NOT execute the pipeline here – the harness will do that.
# <end code template>
# ---------------------------  END OF LLM-CODE BLOCK  ---------------------------

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

