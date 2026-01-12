
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, GlobalAttention, global_mean_pool, batch_norm
import numpy as np
import sklearn.preprocessing
import math

class CustomDataset(Dataset):
#  REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:CustomDataset"
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        # pre.transform returns a list of Data objects
        self.data_list = pre.transform(X)
        self.y = y

        # Attach labels to data objects for PyG collation
        if self.y is not None:
            # Ensure y is accessible
            if isinstance(self.y, np.ndarray):
                y_tensor = torch.from_numpy(self.y)
            else:
                y_tensor = self.y

            y_tensor = y_tensor.cpu()
            for i, data in enumerate(self.data_list):
                data.y = y_tensor[i].view(1).long()
        else:
            # Inference mode, dummy labels
            for data in self.data_list:
                data.y = torch.tensor([0]).long()

    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx], self.data_list[idx].y

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
        self.stats = {}

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this. Configure as you please.
        return {
            "dataset_builder": "llm_script:CustomDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",     # or torch_geometric.loader:DataLoader
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,

            # NO custom collate callables allowed.
            "collate": None,

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False, 
                                "batch_size": 512} # Or whatever you want
        }

    def fit(self, X, y=None):
        # <LLM: Extract statistics for transform>
        if torch.is_tensor(X):
            X = X.cpu().numpy()

        # Collect stats for log-normalization
        # E: indices 3, 8, ...
        # Pt: indices 4, 9, ...
        # MET: index 0

        E_cols = X[:, 3::5]
        Pt_cols = X[:, 4::5]

        # Mask only valid objects (E > 0)
        mask = E_cols > 0
        all_E = E_cols[mask]
        all_Pt = Pt_cols[mask]

        self.stats['logE_mean'] = np.mean(np.log1p(all_E))
        self.stats['logE_std']  = np.std(np.log1p(all_E))
        self.stats['logPt_mean'] = np.mean(np.log1p(all_Pt))
        self.stats['logPt_std']  = np.std(np.log1p(all_Pt))

        met = X[:, 0]
        self.stats['logMET_mean'] = np.mean(np.log1p(met))
        self.stats['logMET_std']  = np.std(np.log1p(met))

        return self

    def transform(self, X):
        # <LLM: Apply pre-processing logic>
        # Perform vectorized preprocessing to Speed up graph creation
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if str(X.device) != 'cpu':
            X = X.cpu()

        N = X.shape[0]

        # 1. Unpack Raw Data
        # MET: [N]
        met_val = X[:, 0]
        met_phi = X[:, 1]

        # Objects: [N, 18, 5]
        objs = X[:, 2:].view(N, 18, 5)
        # Features: id(0), E(1), pt(2), eta(3), phi(4)

        # 2. Construct Unified Particle List (MET + 18 Objects) -> [N, 19, F]
        # We'll treat MET as the first particle
        # MET "particle": id=0(dummy), E=met_val, pt=met_val, eta=0, phi=met_phi

        # Prepare MET tensor [N, 1, 5]
        met_exp = met_val.unsqueeze(1).unsqueeze(2) # [N, 1, 1]
        met_phi_exp = met_phi.unsqueeze(1).unsqueeze(2)
        zero = torch.zeros_like(met_exp)
        # MET feature vector for kinematic calc: E, pt, eta, phi (we don't need ID here yet)
        met_vec = torch.cat([met_exp, met_exp, zero, met_phi_exp], dim=2) # [N, 1, 4] (E, pt, eta, phi)

        # Object vectors [N, 18, 4] (E, pt, eta, phi) -> indices 1,2,3,4
        obj_vec = objs[:, :, 1:5]

        # Concatenate: [N, 19, 4]
        all_vec = torch.cat([met_vec, obj_vec], dim=1)

        # Unpack for readability
        E_all   = all_vec[:, :, 0]
        pt_all  = all_vec[:, :, 1]
        eta_all = all_vec[:, :, 2]
        phi_all = all_vec[:, :, 3]

        # 3. Vectorized Pairwise Calculation (N, 19, 19)
        # Broadcast for Matrix: (N, 19, 1) vs (N, 1, 19)
        d_eta = eta_all.unsqueeze(2) - eta_all.unsqueeze(1)
        d_phi = phi_all.unsqueeze(2) - phi_all.unsqueeze(1)
        # Wrap phi
        d_phi = torch.remainder(d_phi + math.pi, 2 * math.pi) - math.pi
        dR = torch.sqrt(d_eta**2 + d_phi**2)

        # Invariant Mass
        # p = (pt*cos, pt*sin, pt*sinh)
        px = pt_all * torch.cos(phi_all)
        py = pt_all * torch.sin(phi_all)
        pz = pt_all * torch.sinh(eta_all)

        # Matrix sum
        E_sum  = E_all.unsqueeze(2) + E_all.unsqueeze(1)
        px_sum = px.unsqueeze(2) + px.unsqueeze(1)
        py_sum = py.unsqueeze(2) + py.unsqueeze(1)
        pz_sum = pz.unsqueeze(2) + pz.unsqueeze(1)

        p2_sum = px_sum**2 + py_sum**2 + pz_sum**2
        m2 = E_sum**2 - p2_sum
        m_ij = torch.sqrt(torch.clamp(m2, min=1e-6))

        # Norm Constants
        lE_m, lE_s = self.stats['logE_mean'], self.stats['logE_std']
        lP_m, lP_s = self.stats['logPt_mean'], self.stats['logPt_std']
        lM_m, lM_s = self.stats['logMET_mean'], self.stats['logMET_std']

        # 4. Building Data Objects
        # We assume 0 padding means pt=0 or E=0.
        # Check pt > 0 for original objects. MET is always valid? (Assume yes)
        # valid_mask: [N, 19]. Index 0 is MET (True). Indices 1..18 depend on input.
        valid_mask_objs = (objs[:, :, 1] > 0.001) # E > 1e-3
        valid_mask_met  = torch.ones((N, 1), dtype=torch.bool)
        valid_mask = torch.cat([valid_mask_met, valid_mask_objs], dim=1)

        data_list = []

        # Helper indices for meshgrid
        base_range = torch.arange(19)

        for i in range(N):
            # Select valid particles
            mask_i = valid_mask[i]
            indices = base_range[mask_i] # e.g. [0, 1, 2, 5...]
            num_nodes = len(indices)

            # Extract Node Features
            # MET: [logPt, 0, sin, cos, 1, logE]
            # Obj: [logPt, eta, sin, cos, 0, logE]

            # Gather raw for this event
            pts_i = pt_all[i, mask_i]
            eta_i = eta_all[i, mask_i]
            phi_i = phi_all[i, mask_i]
            E_i   = E_all[i, mask_i]

            # Normalize
            # Apply different norm for MET (idx 0) vs others? 
            # Simplified: Use obj statistics for all except maybe MET. 
            # Actually, let's just use the stats we computed. 
            # Since MET is index 0 in `indices`, check index.

            # Vectorized normalization for the subset
            # log1p
            l_pts = torch.log1p(pts_i)
            l_Es  = torch.log1p(E_i)

            # Standardize
            # Create tensor of means/stds matching the size
            # If index is 0 (MET), use MET stats, else Obj stats
            is_met = (indices == 0)

            means_pt = torch.where(is_met, torch.tensor(lM_m), torch.tensor(lP_m))
            stds_pt  = torch.where(is_met, torch.tensor(lM_s), torch.tensor(lP_s))
            means_E  = torch.where(is_met, torch.tensor(lM_m), torch.tensor(lE_m))
            stds_E   = torch.where(is_met, torch.tensor(lM_s), torch.tensor(lE_s))

            n_pts = (l_pts - means_pt) / stds_pt
            n_Es  = (l_Es - means_E) / stds_E

            # Node Feature Tensor [Nodes, 6]
            # Feats: [norm_log_pt, eta, sin_phi, cos_phi, is_met, norm_log_E]
            x = torch.stack([
                n_pts,
                eta_i,
                torch.sin(phi_i),
                torch.cos(phi_i),
                is_met.float(),
                n_Es
            ], dim=1).float()

            # Edges
            # Fully connected logic on valid node subset
            r_idx, c_idx = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
            r_idx, c_idx = r_idx.reshape(-1), c_idx.reshape(-1)
            # Remove self loops
            loop_mask = r_idx != c_idx
            r_idx, c_idx = r_idx[loop_mask], c_idx[loop_mask]

            edge_index = torch.stack([r_idx, c_idx], dim=0)

            # Extract Edge Features using original indices to lookup in precalc matrices
            orig_r = indices[r_idx]
            orig_c = indices[c_idx]

            curr_dR = dR[i, orig_r, orig_c]
            curr_m  = m_ij[i, orig_r, orig_c]

            # Scale edge features slightly
            # dR is usually < 5. m_ij can be 200,000. 
            # log(m_ij)
            feat_dR = curr_dR
            feat_m  = torch.log1p(curr_m / 1000.0) # approx GeV scale log

            edge_attr = torch.stack([feat_dR, feat_m], dim=1).float()

            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

        return data_list # must return an indexable, picklable object

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
        # Lane B: sample_object is a Batch object (or Data)
        in_channels = sample_object.x.shape[1]
        edge_dim = sample_object.edge_attr.shape[1]

        hidden_dim = 128
        out_dim = 1

        # Transformer Conv layers
        # Uses edge features in attention
        self.conv1 = TransformerConv(in_channels, hidden_dim, heads=4, edge_dim=edge_dim, dropout=0.1)
        self.conv2 = TransformerConv(hidden_dim*4, hidden_dim, heads=4, edge_dim=edge_dim, dropout=0.1)
        self.conv3 = TransformerConv(hidden_dim*4, hidden_dim, heads=4, edge_dim=edge_dim, dropout=0.1)

        self.bn1 = nn.BatchNorm1d(hidden_dim*4)
        self.bn2 = nn.BatchNorm1d(hidden_dim*4)
        self.bn3 = nn.BatchNorm1d(hidden_dim*4)

        # Readout
        self.lin_head = nn.Sequential(
            nn.Linear(hidden_dim*4 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    # <LLM: optionally build extra layers here>

    def forward(self, batch):
        # IMPORTANT output must be logits/probabilities per event
        # <LLM: Define your model's forward pass here>
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        # Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        # Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)

        # Layer 3
        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)

        # Global Pooling
        x_mean = global_mean_pool(x, batch_idx)
        x_max  = global_mean_pool(x, batch_idx) # Typo in plan, fixed to max? Let's use mean_pool for both or concat
        # Actually PyG has global_max_pool
        from torch_geometric.nn import global_max_pool
        x_max = global_max_pool(x, batch_idx)

        x_cat = torch.cat([x_mean, x_max], dim=1)

        # Output
        out = self.lin_head(x_cat)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # Using BCEWithLogitsLoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.001, 
        epochs=epochs, 
        steps_per_epoch=len(train_loader)
    )

    best_loss = float('inf')
    best_model_state = None

    final_tr_loss = 0
    final_val_loss = 0
    final_tr_acc = 0
    final_val_acc = 0

    for epoch in range(epochs):
        # Train
        model.train()
        sum_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch)
            y = batch.y.float().view(-1, 1)

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            sum_loss += loss.item() * batch.num_graphs
            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == y).sum().item()
            total += batch.num_graphs

        train_loss = sum_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        sum_val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                y = batch.y.float().view(-1, 1)

                loss = criterion(out, y)
                sum_val_loss += loss.item() * batch.num_graphs

                preds = (torch.sigmoid(out) > 0.5).float()
                val_correct += (preds == y).sum().item()
                val_total += batch.num_graphs

        val_loss = sum_val_loss / val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}: TrLoss {train_loss:.4f}, ValLoss {val_loss:.4f}, ValAcc {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            # Save state on CPU to save memory/avoid drift
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

        final_tr_loss = train_loss
        final_val_loss = val_loss
        final_tr_acc = train_acc
        final_val_acc = val_acc

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # Recalculate metrics for best model? 
        # Typically we just return the last epoch metrics or the best ones. 
        # Harness convention suggests returning what we ended with, but model should be best.
        # Let's return the best validation loss we saw.
        final_val_loss = best_loss

    return model, final_tr_loss, final_val_loss, final_tr_acc, final_val_acc

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

