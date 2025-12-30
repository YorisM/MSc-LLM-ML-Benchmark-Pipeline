
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

# ---------- IMPORTS ----------
from torch.utils.data import Dataset

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.X = pre.transform(X) if pre is not None else X
        self.y = y
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # No stateful components needed for graph building
        pass

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
        return self

    def transform(self, X):
        # X shape: [num_events, 92]
        events = []
        for x in X:  # x shape: [92]
            et_miss, phi_miss = x[0], x[1]  # Global features
            objects = x[2:].reshape(18, 5)  # [18, 5]: obj_id, E, pT, eta, phi

            # Valid objects where E > 0 (assuming energy > 0 for real particles)
            valid_mask = objects[:, 1] > 0  # E is index 1 in object
            objs = objects[valid_mask]  # [n, 5], n <= 18
            num_nodes = objs.shape[0]

            if num_nodes == 0:
                # If no objects, create a dummy graph with one node for safety
                dummy_obj = torch.tensor([[1.0, 1e-10, 0.0, 0.0, 0.0]], dtype=torch.float32)  # [1,5]
                events.append(torch_geometric.data.Data(
                    x=dummy_obj,
                    edge_index=torch.empty(2, 0, dtype=torch.long),
                    edge_attr=torch.empty(0, 4, dtype=torch.float32),
                    global_features=torch.tensor([et_miss, phi_miss], dtype=torch.float32)
                ))
                continue

            # Node features: [n, 5]: obj_id, E, pT, eta, phi
            x_nodes = objs.clone()  # [n, 5]

            # Edge index: all pairs i < j
            if num_nodes > 1:
                edge_index = torch.triu_indices(num_nodes, num_nodes, offset=1)  # [2, num_edges]
                num_edges = edge_index.shape[1]
                edge_features = []  # [num_edges, 4]: delta_eta, delta_phi, delta_R, m
                for i, j in zip(edge_index[0], edge_index[1]):
                    # Extract o1 and o2: [5]: obj_id, E, pT, eta, phi
                    E1, pT1, eta1, phi1 = objs[i, 1], objs[i, 2], objs[i, 3], objs[i, 4]
                    E2, pT2, eta2, phi2 = objs[j, 1], objs[j, 2], objs[j, 3], objs[j, 4]
                    delta_eta = eta1 - eta2
                    delta_phi = phi1 - phi2
                    delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)
                    # Invariant mass calculation
                    px1 = pT1 * torch.cos(phi1)
                    py1 = pT1 * torch.sin(phi1)
                    pz1 = pT1 * torch.sinh(eta1)
                    px2 = pT2 * torch.cos(phi2)
                    py2 = pT2 * torch.sin(phi2)
                    pz2 = pT2 * torch.sinh(eta2)
                    E_tot = E1 + E2
                    px_tot = px1 + px2
                    py_tot = py1 + py2
                    pz_tot = pz1 + pz2
                    m2 = E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2
                    m = torch.sqrt(torch.clamp(m2, min=0.0))  # Clamp to avoid sqrt of negative due to numerical errors
                    edge_features.append([delta_eta.item(), delta_phi.item(), delta_R.item(), m.item()])
                edge_attr = torch.tensor(edge_features, dtype=torch.float32)  # [num_edges, 4]
            else:
                edge_index = torch.empty(2, 0, dtype=torch.long)  # No edges if only one node
                edge_attr = torch.empty(0, 4, dtype=torch.float32)

            # Global features: [2]
            global_feat = torch.tensor([et_miss, phi_miss], dtype=torch.float32)

            # Create Data object
            data = torch_geometric.data.Data(
                x=x_nodes,  # [n, 5]
                edge_index=edge_index,  # [2, num_edges]
                edge_attr=edge_attr,  # [num_edges, 4]
                global_features=global_feat  # [2]
            )
            events.append(data)

        return events  # List of Data objects, picklable

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object is a DataBatch with .x, .edge_index, .edge_attr, .global_features, .batch
        self.node_emb = nn.Linear(5, 64)  # node features: [5] -> [64]
        self.edge_emb = nn.Linear(4, 32)  # edge features: [4] -> [32]
        self.gnn1 = torch_geometric.nn.GATConv(64, 64, heads=4, edge_dim=32)  # 4 heads, [64] -> [64*4], but output [64*4 / heads] wait, heads=4, out_channels=64, so out_dim per head 64/4=16? No:
        # GATConv(in_channels, out_channels, heads=1, edge_dim=None)
        # Here: in=64, out=64, heads=4: output [num_nodes, 64*4] then need to sum or something, but typically for multihead,出身 often mean/concat
        # Standard is to leave as is, the graph module handles aggregation.
        # To simplify, use heads=4, out=64: final node feat [num_nodes, 64]
        self.gnn2 = torch_geometric.nn.GATConv(64*4 if heads==4 else 64, 64, heads=1, edge_dim=32)  # After head concat, to 64
        self.pool = torch_geometric.nn.global_mean_pool  # [batch_size, 64] after pool
        self.global_emb = nn.Linear(2, 64)  # global [2] -> [64]
        self.classifier = nn.Linear(64, 1)  # [64] -> logits [1]

    def forward(self, batch_x):
        # batch_x is DataBatch
        x = self.node_emb(batch_x.x)  # [total_nodes, 64]
        edge_attr = self.edge_emb(batch_x.edge_attr)  # [total_edges, 32]
        x = self.gnn1(x, batch_x.edge_index, edge_attr=edge_attr)  # [total_nodes, 64*4]
        x = torch.relu(x)
        x = self.gnn2(x, batch_x.edge_index, edge_attr=edge_attr)  # [total_nodes, 64]
        x = torch.relu(x)
        pool = self.pool(x, batch_x.batch)  # [batch_size, 64]
        if hasattr(batch_x, 'global_features') and batch_x.global_features is not None:
            glob = self.global_emb(batch_x.global_features)  # [batch_size, 64]
            pool += glob  # Element-wise add
        out = self.classifier(pool).squeeze(-1)  # [batch_size] logits
        return out

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20  # Increased for better training
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # Some regularization
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    best_val_auc = 0
    patience = 5  # Early stopping patience
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        scheduler.step()
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # Validation
        model.eval()
        total_loss_val = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y
                out = model(xb)
                loss = criterion(out, yb.float())
                total_loss_val += loss.item()
                preds = (torch.sigmoid(out) > 0.5).float()
                correct_val += (preds == yb).sum().item()
                total_val += yb.size(0)
        val_loss = total_loss_val / len(val_loader)
        val_acc = correct_val / total_val

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping on AUC proxy (val_loss decrease)
        if val_loss < best_val_auc:  # Since AUC correlates with loss, use as proxy
            best_val_auc = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break

    return model, train_loss_list, val_loss_list, train_acc_list, val_acc_list

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

