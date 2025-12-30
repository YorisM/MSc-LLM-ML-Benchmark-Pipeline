
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
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

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

class MyPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None

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
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        # Fit on visible particles: assume padded with 0
        visible_mask = X_np[:, 2::5] > 1e-3  # E > 1e-3, shape [batch, 18]
        visible_X = X_np[visible_mask.any(axis=1)]
        # Normalize per particle: obj_id, log(E), log(pT), eta, phi
        # But obj_id is categorical, skip normalization for it
        feats_to_norm = []
        for i in range(18):
            start = 2 + i*5
            feats = visible_X[:, start:start+5]  # obj_id, E, pT, eta, phi
            E_pt = np.log(np.clip(feats[:, 1:3], 1e-3, None))  # log E, log pT
            eta_phi = feats[:, 3:5]
            feats_to_norm.append(np.concatenate([feats[:, 0:1], E_pt, eta_phi], axis=1))
        all_feats = np.concatenate(feats_to_norm, axis=0)
        self.mean = np.mean(all_feats, axis=0)
        self.std = np.std(all_feats, axis=0) + 1e-8
        return self

    def transform(self, X):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        data_list = []
        for event in X_np:
            # Extract globals: ET_miss, phi_ET_miss
            globals = event[:2].astype(np.float32)  # [2]

            # Extract particles
            particles = []
            for i in range(18):
                start = 2 + i*5
                obj_id, E, pT, eta, phi = event[start:start+5]
                if E < 1e-3:  # Skip padding
                    continue
                # Compute log(E), log(pT), eta, phi; obj_id as is
                logE = np.log(max(E, 1e-3))
                logpT = np.log(max(pT, 1e-3))
                particles.append([obj_id, logE, logpT, eta, phi])
            particles = np.array(particles)
            n_particles = len(particles)
            if n_particles == 0:
                # Handle empty events, but assume not
                node_features = torch.zeros(1, 5)
                edge_index = torch.zeros(2, 0, dtype=torch.long)
                edge_attr = torch.zeros(0, 2)
                global_feat = torch.tensor(globals, dtype=torch.float32)
            else:
                # Normalize
                particle_norm = (particles - self.mean) / self.std
                node_features = torch.tensor(particle_norm, dtype=torch.float32)  # [n_particles, 5]

                # Edges: all pairs i<j
                edge_index_list = []
                edge_attr_list = []
                for i in range(n_particles):
                    for j in range(i+1, n_particles):
                        edge_index_list.append([i, j])
                        edge_index_list.append([j, i])  # Undirected
                        # Compute ΔR
                        eta_i, eta_j = particles[i, 3], particles[j, 3]
                        phi_i, phi_j = particles[i, 4], particles[j, 4]
                        delta_eta = eta_i - eta_j
                        delta_phi = torch.atan2(torch.sin(phi_i - phi_j), torch.cos(phi_i - phi_j)).numpy()
                        delta_R = np.sqrt(delta_eta**2 + delta_phi**2)

                        # Invariant mass
                        # First, compute four-momenta
                        E_i, E_j = particles[i, 1], particles[j, 1]  # E
                        pT_i, pT_j = particles[i, 2], particles[j, 2]  # pT
                        eta_i, eta_j = particles[i, 3], particles[j, 3]
                        phi_i, phi_j = particles[i, 4], particles[j, 4]
                        # Assume massless, but compute properly
                        p_x_i = pT_i * np.cos(phi_i)
                        p_y_i = pT_i * np.sin(phi_i)
                        p_z_i = pT_i * np.sinh(eta_i)
                        p_x_j = pT_j * np.cos(phi_j)
                        p_y_j = pT_j * np.sin(phi_j)
                        p_z_j = pT_j * np.sinh(eta_j)
                        E_tot = E_i + E_j
                        p_x_tot = p_x_i + p_x_j
                        p_y_tot = p_y_i + p_y_j
                        p_z_tot = p_z_i + p_z_j
                        m_ij_squared = E_tot**2 - (p_x_tot**2 + p_y_tot**2 + p_z_tot**2)
                        m_ij = np.sqrt(max(m_ij_squared, 0))

                        edge_attr_list.append([delta_R, m_ij])
                        edge_attr_list.append([delta_R, m_ij])

                edge_index = torch.tensor(np.array(edge_index_list).T, dtype=torch.long) if edge_index_list else torch.zeros(2, 0, dtype=torch.long)
                edge_attr = torch.tensor(np.array(edge_attr_list), dtype=torch.float32) if edge_attr_list else torch.zeros(0, 2, dtype=torch.float32)

                global_feat = torch.tensor(globals, dtype=torch.float32)

            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, global_feat=global_feat)
            data_list.append(data)
        return data_list  # List of Data objects

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        from torch_geometric.nn import GINConv, GraphConv
        self.conv1 = GINConv(nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 64)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)))
        self.pool = global_mean_pool
        self.global_mlp = nn.Sequential(nn.Linear(64 + 2, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, batch_x):
        # batch_x is a Batch
        out = self.conv1(batch_x.x, batch_x.edge_index, batch_x.edge_attr)
        out = F.relu(out)
        out = self.conv2(out, batch_x.edge_index, batch_x.edge_attr)
        out = F.relu(out)
        pooled = self.pool(out, batch_x.batch)  # [batch_size, 64]
        # Concat global features: assume batch_x.global_feat is batched
        if hasattr(batch_x, 'global_feat') and batch_x.global_feat is not None:
            global_in = torch.cat([pooled, batch_x.global_feat], dim=1)  # [batch_size, 66]
        else:
            global_in = pooled
        logits = self.global_mlp(global_in).squeeze(-1)
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, mode='max')
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    best_model_state = None
    patience = 5
    counter = 0

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_logits = []
        train_ys = []
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * yb.size(0)
            preds = torch.sigmoid(out) > 0.5
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)
            train_logits.extend(out.cpu().detach().numpy())
            train_ys.extend(yb.cpu().numpy())
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        train_auc = roc_auc_score(train_ys, train_logits)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_logits = []
        val_ys = []
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y
                out = model(xb)
                loss = criterion(out, yb.float())
                val_loss += loss.item() * yb.size(0)
                preds = torch.sigmoid(out) > 0.5
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)
                val_logits.extend(out.cpu().numpy())
                val_ys.extend(yb.cpu().numpy())
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_ys, val_logits)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        scheduler.step(val_auc)

        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Train AUC {train_auc:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}, Val AUC {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_loss_list, val_loss_list, train_acc_list, val_acc_list
```

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

