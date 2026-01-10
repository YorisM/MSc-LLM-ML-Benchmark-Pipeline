
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
# <LLM: Import modules>
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

#  -------- (OPTIONAL) CUSTOM DATASET  --------
# class CustomDataset(Dataset):
# ... (Not used, using Lane A with provided dataset wrapper logic)

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # REQUIREMENTS
    #   - IMPORTANT: All state must be picklable with the std-lib pickle module.
    #   - May allocate NumPy arrays or Torch tensors internally, but: transform() must be deterministic.
    #   - Store only derived parameters needed for transform i.e. do not store the raw data itself in the preprocessor object.

    # DATA SPECIFICS
    #    Total flat length per event (X_train & X_val): 92
    #    ...

    # <LLM: Write code to preprocess the data> 
    def __init__(self):
        # Stats for standardization
        self.mean_log_pt = 0.0
        self.std_log_pt = 1.0
        self.mean_log_e = 0.0
        self.std_log_e = 1.0
        self.scale_e_pt = 1e-5 # Scale MeV inputs down to ~O(1-10) range for mass calc

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 1024,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 1024}
        }

    def fit(self, X, y=None):
        # <LLM: Extract statistics for transform>
        # Input X is (N, 92) numpy array
        # Extract particle/MET info to compute global scale stats

        # Reshape to objects for aggregation
        # Indices 2..92 are 18 objects * 5 feats
        x_parts = X[:, 2:].reshape(-1, 18, 5) # (N, 18, 5)

        # Features: 0:obj, 1:E, 2:pT, 3:eta, 4:phi
        # Mask valid particles (E > 1 MeV) for stat calculation
        es = x_parts[:, :, 1]
        pts = x_parts[:, :, 2]
        mask = es > 1.0 

        flat_es = es[mask]
        flat_pts = pts[mask]

        # Compute log1p stats
        log_es = np.log1p(flat_es)
        log_pts = np.log1p(flat_pts)

        self.mean_log_e = float(log_es.mean())
        self.std_log_e = float(log_es.std())
        self.mean_log_pt = float(log_pts.mean())
        self.std_log_pt = float(log_pts.std())

        return self

    def transform(self, X):
        # <LLM: Apply pre-processing logic>
        # Input X: (N, 92). Output: (N, 19, 9)
        # We transform the flat event into a set of 19 nodes (1 MET + 18 Particles)
        # Output features per node:
        # 0: mask (1=valid, 0=pad)
        # 1: is_met (1=met, 0=particle)
        # 2: obj_id (raw integer)
        # 3: E (scaled linearly)
        # 4: pT (scaled linearly)
        # 5: eta
        # 6: phi
        # 7: log_pt_norm (standardized)
        # 8: log_e_norm (standardized)

        N = X.shape[0]
        out = np.zeros((N, 19, 9), dtype=np.float32)

        # --- 1. Handling MET (Node 0) ---
        # MET in cols 0 (Et), 1 (phi)
        met_et = X[:, 0]
        met_phi = X[:, 1]

        out[:, 0, 0] = 1.0 # mask
        out[:, 0, 1] = 1.0 # is_met
        out[:, 0, 2] = 0.0 # obj_id
        out[:, 0, 3] = met_et * self.scale_e_pt
        out[:, 0, 4] = met_et * self.scale_e_pt
        out[:, 0, 5] = 0.0 # eta=0 for MET
        out[:, 0, 6] = met_phi

        # Standardized logs
        out[:, 0, 7] = (np.log1p(met_et) - self.mean_log_pt) / (self.std_log_pt + 1e-6)
        out[:, 0, 8] = (np.log1p(met_et) - self.mean_log_e) / (self.std_log_e + 1e-6)

        # --- 2. Handling Particles (Nodes 1..18) ---
        parts = X[:, 2:].reshape(N, 18, 5)
        # parts indices: 0:obj, 1:E, 2:pt, 3:eta, 4:phi

        # Determine validity (E > 1e-1)
        valid = parts[:, :, 1] > 0.1

        out[:, 1:, 0] = valid.astype(np.float32)
        out[:, 1:, 1] = 0.0 # is_met
        out[:, 1:, 2] = parts[:, :, 0] # obj_id
        out[:, 1:, 3] = parts[:, :, 1] * self.scale_e_pt
        out[:, 1:, 4] = parts[:, :, 2] * self.scale_e_pt
        out[:, 1:, 5] = parts[:, :, 3]
        out[:, 1:, 6] = parts[:, :, 4]

        # Standardized logs
        log_pt = np.log1p(parts[:, :, 2])
        log_e  = np.log1p(parts[:, :, 1])
        out[:, 1:, 7] = (log_pt - self.mean_log_pt) / (self.std_log_pt + 1e-6)
        out[:, 1:, 8] = (log_e  - self.mean_log_e)  / (self.std_log_e  + 1e-6)

        # Zero-out padding using mask broadcast to clean up any garbage logic
        # out shape: (N, 19, 9)
        # mask shape: (N, 19, 1) derived from out[:,:,0]
        mask_bc = out[:, :, 0:1]
        out = out * mask_bc

        return out

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class DenseEdgeConvBlock(nn.Module):
    """
    Computes pairwise kinematic features (Invariant Mass & Delta R) 
    and applies a dense EdgeConv / Interaction operation.
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        # Edge input dimension:
        # node_i (d_in) + node_j (d_in) + pair_feats (2: m_ij, dR_ij)
        edge_dim = 2 * d_in + 2

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, d_out),
            nn.BatchNorm1d(d_out), # BN over channel dim (needs transpose in fwd)
            nn.ReLU(),
            nn.Linear(d_out, d_out),
            nn.ReLU()
        )

        # Node update: takes original node + aggregated edge message
        self.node_update = nn.Sequential(
            nn.Linear(d_in + d_out, d_out),
            nn.LayerNorm(d_out),
            nn.ReLU()
        )

    def forward(self, h, kin, mask):
        # h: (B, N, D)
        # kin: (B, N, 4) -> [E, pT, eta, phi] scaled
        # mask: (B, N) boolean
        B, N, D = h.shape

        # --- 1. Compute Pairwise Features ---
        # Expand kinematics
        # kin shape: B, N, 4
        k_i = kin.unsqueeze(2).expand(B, N, N, 4)
        k_j = kin.unsqueeze(1).expand(B, N, N, 4)

        # Unbind for clarity
        E_i, pt_i, eta_i, phi_i = k_i.unbind(-1)
        E_j, pt_j, eta_j, phi_j = k_j.unbind(-1)

        # A. Delta R
        dphi = phi_i - phi_j
        # Wrap phi to [-pi, pi]
        dphi = torch.remainder(dphi + math.pi, 2 * math.pi) - math.pi
        deta = eta_i - eta_j
        dR2 = deta**2 + dphi**2
        dR_feat = torch.log1p(dR2) # Log scale for stability

        # B. Invariant Mass
        # Reconstruct Cartesian vectors (scaled)
        px_i = pt_i * torch.cos(phi_i)
        py_i = pt_i * torch.sin(phi_i)
        pz_i = pt_i * torch.sinh(eta_i) # sinh can be latge, but input eta is clipped usually? Assuming safe.

        px_j = pt_j * torch.cos(phi_j)
        py_j = pt_j * torch.sin(phi_j)
        pz_j = pt_j * torch.sinh(eta_j)

        E_tot = E_i + E_j
        Px_tot = px_i + px_j
        Py_tot = py_i + py_j
        Pz_tot = pz_i + pz_j

        m2 = E_tot.square() - (Px_tot.square() + Py_tot.square() + Pz_tot.square())
        # Signed log mass
        m_feat = torch.sign(m2) * torch.log1p(torch.abs(m2))

        pair_feats = torch.stack([m_feat, dR_feat], dim=-1) # (B, N, N, 2)

        # --- 2. Edge MLP ---
        h_i = h.unsqueeze(2).expand(B, N, N, D)
        h_j = h.unsqueeze(1).expand(B, N, N, D)

        edge_in = torch.cat([h_i, h_j, pair_feats], dim=-1) # (B, N, N, 2D+2)

        # Collapse for MLP: (B*N*N, C) or use conv1d trick
        # Here linear is efficient enough on last dim
        edge_raw = self.edge_mlp[0](edge_in) 
        # Manual BN: B*N*N x C -> permute -> BN -> permute
        # This is memory heavy. Let's simplify and assume layer norm or just MLP without BN in middle
        # Re-defining MLP inline to avoid BN shape issues or implementing it carefully
        # Using functional relu
        out = F.relu(edge_raw)
        out = self.edge_mlp[3](out)
        out = F.relu(out) # (B, N, N, D_out)

        # --- 3. Aggregation (Max Pool masked) ---
        # Mask out padding in 'j' dimension
        # mask is (B, N). mask_j needs to be (B, 1, N, 1) broadcastable
        mask_j = mask.unsqueeze(1).unsqueeze(-1)

        # Set invalid edges to huge negative for max pool
        out = out.masked_fill(~mask_j, -1e9)

        # Max pool over j (dim 2)
        agg, _ = out.max(dim=2) # (B, N, D_out)

        # --- 4. Node Update ---
        cat_agg = torch.cat([h, agg], dim=-1)
        res = self.node_update(cat_agg)

        return h + res # Residual connection

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object shape: (B, 19, 9)
        input_feats = 9
        d_model = 128

        # Feature embeddings
        # We handle categorical obj_id (idx 2) separate?
        # Let's simple project all inputs. Obj_id is integer but treating as float is passable for dense networks
        # or we could embed. Let's use Embedding for obj_id.
        self.obj_embedding = nn.Embedding(20, 16) # Assume max 20 IDs

        # Input projection: 8 floats + 16 embed = 24
        self.input_proj = nn.Sequential(
            nn.Linear(8 + 16, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

        # Stack of GNN layers
        self.layer1 = DenseEdgeConvBlock(d_model, d_model)
        self.layer2 = DenseEdgeConvBlock(d_model, d_model)
        self.layer3 = DenseEdgeConvBlock(d_model, d_model)

        # Global Pooling and Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        # batch_x: (B, 19, 9)
        # Features: 0:mask, 1:met_flag, 2:obj_id, 3:E_sc, 4:pT_sc, 5:eta, 6:phi, 7:logpt, 8:loge

        mask = batch_x[:, :, 0] > 0.5 # Boolean mask (B, N)

        # 1. Embed Features
        obj_ids = batch_x[:, :, 2].long().clamp(0, 19)
        obj_emb = self.obj_embedding(obj_ids)

        scalars = torch.cat([batch_x[:, :, :2], batch_x[:, :, 3:]], dim=-1) # Skip obj_id
        x_in = torch.cat([scalars, obj_emb], dim=-1)

        h = self.input_proj(x_in)

        # 2. Extract Kinematics for Pairwise Ops
        # 3:E, 4:pT, 5:eta, 6:phi
        kin = batch_x[:, :, 3:7]

        # 3. GNN Layers
        h = self.layer1(h, kin, mask)
        h = self.layer2(h, kin, mask)
        h = self.layer3(h, kin, mask)

        # 4. Global Mean Pooling (Masked)
        mask_f = mask.unsqueeze(-1).float() # (B, N, 1)
        h_masked = h * mask_f

        sum_h = h_masked.sum(dim=1)
        count_h = mask_f.sum(dim=1).clamp(min=1.0)
        global_feat = sum_h / count_h

        # 5. Classify
        logits = self.classifier(global_feat)
        return logits # (B, 1)

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 10
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS
    #   - Must return: trained_model, train_loss, val_loss, train_acc, val_acc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    # OneCycle for fast convergence in fixed epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    criterion = nn.BCEWithLogitsLoss()

    train_loss_hist = []
    val_loss_hist = []
    train_auc_hist = []
    val_auc_hist = []

    best_auc = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        all_tr_preds = []
        all_tr_targs = []

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device).float().unsqueeze(-1) # [B, 1]

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_losses.append(loss.item())

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                all_tr_preds.append(probs.cpu())
                all_tr_targs.append(yb.cpu())

        tr_loss = np.mean(batch_losses)
        tr_preds = torch.cat(all_tr_preds).numpy()
        tr_targs = torch.cat(all_tr_targs).numpy()

        try:
            tr_auc = roc_auc_score(tr_targs, tr_preds)
        except:
            tr_auc = 0.5

        # Validation
        model.eval()
        val_batch_losses = []
        all_val_preds = []
        all_val_targs = []

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device).float().unsqueeze(-1)

                logits = model(Xb)
                loss = criterion(logits, yb)
                val_batch_losses.append(loss.item())

                probs = torch.sigmoid(logits)
                all_val_preds.append(probs.cpu())
                all_val_targs.append(yb.cpu())

        va_loss = np.mean(val_batch_losses)
        va_preds = torch.cat(all_val_preds).numpy()
        va_targs = torch.cat(all_val_targs).numpy()

        try:
            va_auc = roc_auc_score(va_targs, va_preds)
        except:
            va_auc = 0.5

        train_loss_hist.append(tr_loss)
        val_loss_hist.append(va_loss)
        train_auc_hist.append(tr_auc)
        val_auc_hist.append(va_auc)

        print(f"Epoch {epoch+1}/{epochs} | Tr Loss: {tr_loss:.4f} AUC: {tr_auc:.4f} | Va Loss: {va_loss:.4f} AUC: {va_auc:.4f}")

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_loss_hist, val_loss_hist, train_auc_hist, val_auc_hist
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

