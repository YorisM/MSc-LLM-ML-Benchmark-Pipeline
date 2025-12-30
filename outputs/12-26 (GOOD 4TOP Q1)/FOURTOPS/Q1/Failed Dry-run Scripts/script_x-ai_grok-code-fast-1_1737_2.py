
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
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
import torch_geometric
from torch_geometric.nn import GATConv, global_mean_pool

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
 # REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:CustomDataset"
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.datas = pre.transform(X, y)
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

    def __init__(self):
        self.max_obj = 18
        self.num_types = 22  # assumed max obj_id
        self.dR_threshold = 1.5  # for edges between objects and MET
        # node features: typ (embedding), log(E+1), log(pT+1), eta, phi

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "llm_script:CustomDataset",   # custom
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",     # torch_geometric for PyG batches
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed. Choose one: 
            "collate": None,  # for PyG DataLoader

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not torch.is_tensor(X):
            X = torch.tensor(X)
        datas = []
        for i in range(X.shape[0]):
            X_event = X[i]  # [92]
            E_miss = X_event[0]
            phi_miss = X_event[1]
            objects = []
            for j in range(18):
                start = 2 + j * 5
                typ = X_event[start].item()
                E = X_event[start + 1].item()
                pT = X_event[start + 2].item()
                eta = X_event[start + 3].item()
                phi = X_event[start + 4].item()
                if typ == 0.0 or E <= 0:  # skip padding or invalid
                    continue
                objects.append((typ, E, pT, eta, phi))
            node_list = []
            # add MET node
            met_node = torch.tensor([self.num_types + 1, torch.log(E_miss + 1), torch.log(E_miss + 1), 1000.0, phi_miss])  # [5]
            node_list.append(met_node)
            # add object nodes if any
            if objects:
                obj_nodes = torch.tensor(objects)  # [num_obj, 5]
                obj_nodes = obj_nodes.float()
                # transform to log for E and pT
                obj_nodes[:, 1] = torch.log(obj_nodes[:, 1] + 1)
                obj_nodes[:, 2] = torch.log(obj_nodes[:, 2] + 1)
                node_list.append(obj_nodes)
                nodes = torch.cat(node_list, dim=0)  # [1 + num_obj, 5]
            else:
                nodes = met_node.unsqueeze(0)
            num_nodes = nodes.shape[0]
            # build edges: connect MET to all, and objects within dR
            edge_list = []
            for ii in range(num_nodes):
                for jj in range(ii + 1, num_nodes):
                    eta1, phi1 = nodes[ii, 3], nodes[ii, 4]
                    eta2, phi2 = nodes[jj, 3], nodes[jj, 4]
                    deta = abs(eta1 - eta2)
                    dphi = abs(phi1 - phi2)
                    dphi = min(dphi, 2 * 3.14159 - dphi)
                    dR = torch.sqrt(deta**2 + dphi**2)
                    if dR < self.dR_threshold or ii == 0 or jj == 0:  # always connect if involving MET, or within dR
                        edge_list.extend([[ii, jj], [jj, ii]])
            edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list else torch.empty(2, 0, dtype=torch.long).contiguous()
            data = torch_geometric.data.Data(x=nodes, edge_index=edge_index, y=y[i] if y is not None else None)
            datas.append(data)
        return datas  # list of Data objects

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
# Model batch contract:
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.num_types = 22  # match pre
        emb_dim = 8
        node_feat_without_emb = 4  # logE, logpT, eta, phi
        self.typ_emb = nn.Embedding(self.num_types + 2, emb_dim)  # 0 to 22
        input_dim = node_feat_without_emb + emb_dim
        self.conv1 = GATConv(input_dim, 64, heads=4, dropout=0.3)
        self.conv2 = GATConv(64 * 4, 128, heads=4, dropout=0.3)  # output 128*4
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 4, 1)  # output logit

    def forward(self, batch):
        x = batch.x  # [total_nodes, 5]
        typ = x[:, 0].long()
        features = x[:, 1:5]  # [total_nodes, 4]: E, pT, eta, phi
        typ_emb = self.typ_emb(typ)  # [total_nodes, emb_dim]
        x_emb = torch.cat([typ_emb, features], dim=-1)  # [total_nodes, input_dim]
        edge_index = batch.edge_index
        x1 = self.conv1(x_emb, edge_index)  # [total_nodes, 64*4]
        x1 = torch.relu(x1)
        x2 = self.conv2(x1, edge_index)  # [total_nodes, 128*4]
        x2 = torch.relu(x2)
        out = self.dropout(x2)  # [total_nodes, 128*4]
        out = global_mean_pool(out, batch.batch)  # [batch_size, 128*4]
        out = self.fc(out)  # [batch_size, 1]
        return out

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    best_val_acc = 0.0
    best_model_state = None
    patience = 5
    trigger = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb = view.batch_x
            yb = view.batch_y.float().view(-1, 1)  # for BCE
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * yb.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds.float() == yb).sum().item()
            total += yb.size(0)
        train_loss = total_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb = view.batch_x
                yb = view.batch_y.float().view(-1, 1)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * yb.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds.float() == yb).sum().item()
                total += yb.size(0)
        val_loss /= total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            trigger = 0
        else:
            trigger += 1
        if trigger >= patience:
            print("Early stopping")
            break
        scheduler.step()
    model.load_state_dict(best_model_state)
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

