
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
# No additional imports needed

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        pass

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:MyDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader.DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, 
                               "batch_size": 512}
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        import torch
        from torch_geometric.data import Data

        data_list = []
        for event in range(X.shape[0]):
            event_data = X[event]
            E_T_miss = event_data[0]
            phi_E_T_miss = event_data[1]
            global_features = torch.tensor([0.0, E_T_miss, E_T_miss, 0.0, phi_E_T_miss])

            obj_list = []
            for k in range(18):
                idx = 2 + k * 5
                obj, E, p_T, eta, phi = event_data[idx:idx+5]
                if obj == 0.0 or E == 0.0:
                    break
                obj_list.append([obj, E, p_T, eta, phi])
            if not obj_list:
                # If no objects, add dummy object to avoid empty graph
                obj_list.append([1.0, 1.0, 1.0, 0.0, 0.0])
            n_obj = len(obj_list)
            node_features = torch.tensor(obj_list + [global_features.tolist()])  # [n_obj+1, 5]

            edge_index = []
            edge_attr = []
            for i in range(n_obj):
                for j in range(i+1, n_obj):
                    obj_i, E_i, p_T_i, eta_i, phi_i = obj_list[i]
                    obj_j, E_j, p_T_j, eta_j, phi_j = obj_list[j]
                    delta_eta = eta_i - eta_j
                    delta_phi = phi_i - phi_j
                    delta_phi = torch.remainder(delta_phi, 2 * torch.pi)
                    delta_phi = delta_phi - 2 * torch.pi * (delta_phi > torch.pi)
                    delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)

                    p_x_i = p_T_i * torch.cos(phi_i)
                    p_y_i = p_T_i * torch.sin(phi_i)
                    p_z_i = p_T_i * torch.sinh(eta_i)
                    p_x_j = p_T_j * torch.cos(phi_j)
                    p_y_j = p_T_j * torch.sin(phi_j)
                    p_z_j = p_T_j * torch.sinh(eta_j)
                    p_tot_x = p_x_i + p_x_j
                    p_tot_y = p_y_i + p_y_j
                    p_tot_z = p_z_i + p_z_j
                    E_tot = E_i + E_j
                    m_2 = E_tot**2 - p_tot_x**2 - p_tot_y**2 - p_tot_z**2
                    m_ij = torch.sqrt(torch.relu(m_2))

                    edge_index.extend([[i, j], [j, i]])
                    edge_attr.extend([[delta_R, m_ij], [delta_R, m_ij]])

            # Edges from/to global node
            global_idx = n_obj
            for j in range(n_obj):
                edge_index.extend([[global_idx, j], [j, global_idx]])
                edge_attr.extend([[0.0, 0.0], [0.0, 0.0]])

            dataset = Data(
                x=node_features.float(),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                y=torch.tensor(y[event], dtype=torch.long)
            )
            data_list.append(dataset)
        return data_list

def make_preprocessor():
    return MyPreprocessor()

class MyDataset:
    def __init__(self, events, pre, train: bool = True, **kwargs):
        self.data_list = pre.transform(*events)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

# ---------- MODEL ARCHITECTURE ----------
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_mean_pool

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.num_obj_types = 10  # Assumed max obj type
        embed_dim = 8
        self.obj_embed = nn.Embedding(self.num_obj_types + 1, embed_dim)  # +1 for 0-index
        node_in = embed_dim + 4  # obj_embed + E, p_T, eta, phi
        edge_in = 2  # delta_R, m_ij
        hidden = 64
        self.conv1 = pyg_nn.TransformerConv(node_in, hidden, heads=2, edge_dim=edge_in)
        self.bn1 = nn.LayerNorm(hidden * 2)
        self.conv2 = pyg_nn.TransformerConv(hidden * 2, hidden, heads=2, edge_dim=edge_in)
        self.bn2 = nn.LayerNorm(hidden * 2)
        self.final = nn.Linear(hidden * 2, 1)

    def forward(self, G):
        x_embed = self.obj_embed(G.x[:, 0].long())  # [n_nodes, embed_dim]
        x_cont = G.x[:, 1:]  # [n_nodes, 4]
        x = torch.cat([x_embed, x_cont], dim=1).float()  # [n_nodes, embed_dim+4]
        x = self.conv1(x, G.edge_index, G.edge_attr)
        x = self.bn1(x).relu()
        x = self.conv2(x, G.edge_index, G.edge_attr)
        x = self.bn2(x).relu()
        graph_feat = global_mean_pool(x, G.batch)  # [n_graphs, hidden*2]
        out = self.final(graph_feat)  # [n_graphs, 1]
        return out.squeeze(-1)  # [n_graphs]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    import torch.optim as optim
    from sklearn.metrics import roc_auc_score

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            targets = batch.y.float()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == targets).sum().item()
            train_total += batch.num_graphs
        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds_list = []
        val_targets_list = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                targets = batch.y.float()
                loss = criterion(outputs, targets)
                val_loss += loss.item() * batch.num_graphs
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == targets).sum().item()
                val_total += batch.num_graphs
                val_preds_list.extend(outputs.cpu().numpy())
                val_targets_list.extend(targets.cpu().numpy())
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_targets_list, val_preds_list)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}, Val AUC {val_auc:.4f}")

        # Early stopping on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        scheduler.step()

    # Return the last (or best, but simplify to last)
    return model, train_losses, val_losses, train_accs, val_accs

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

