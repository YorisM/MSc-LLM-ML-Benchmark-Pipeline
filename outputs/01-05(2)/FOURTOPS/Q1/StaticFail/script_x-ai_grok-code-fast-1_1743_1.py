
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
from torch_geometric.data import Data
from torch.nn import functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool
import sys

# Global for number of object types
num_obj_types = 0

# -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        X_processed = pre.transform(X) if pre is not None else X
        self.datas = []
        for i in range(len(X_processed)):
            data = self.data_from_X(X_processed[i], y[i])
            if data is not None:
                self.datas.append(data)

    def data_from_X(self, x, y):
        # x is torch.Tensor [92]
        # Extract global features
        global_feat = x[0:2]  # [2]
        # Global node feature: [global_feat[0], global_feat[1], 0.0, 0.0]
        global_node = torch.tensor([global_feat[0], global_feat[1], 0.0, 0.0], dtype=torch.float32)

        nodes = [global_node]
        obj_ids = []

        for j in range(18):
            start = 2 + j * 5
            obj_id = int(x[start].item())
            e = x[start + 1]
            pt = x[start + 2]
            eta = x[start + 3]
            phi = x[start + 4]
            if e != 0.0:  # Filter out padded zeros
                nodes.append(torch.tensor([e, pt, eta, phi], dtype=torch.float32))
                obj_ids.append(obj_id)

        if len(nodes) == 0:
            return None  # No objects, skip event

        x_tensor = torch.stack(nodes)  # [num_nodes, 4]
        obj_types = [len(obj_ids)] + obj_ids  # Global has type = len(obj_ids), arbitrary
        obj_types_tensor = torch.tensor(obj_types, dtype=torch.long)

        # Fully connect all nodes (including global)
        num_nodes = len(nodes)
        if num_nodes > 1:
            edges = torch.combinations(torch.arange(num_nodes), 2).t()  # [2, num_edges]
            # Add reverse edges for undirected
            edge_index = torch.cat([edges, edges.flip(0)], dim=1)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)  # No edges if only one node

        data = Data(x=x_tensor, edge_index=edge_index, y=y, obj_types=obj_types_tensor)
        return data

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.num_obj_types = 0

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
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        obj_ids = set()
        for x in X:
            for j in range(18):
                start = 2 + j * 5
                oid = x[start].item()
                e = x[start + 1]
                if e > 0.0:
                    obj_ids.add(oid)
        self.num_obj_types = len(obj_ids)
        global num_obj_types
        num_obj_types = self.num_obj_types
        return self

    def transform(self, X):
        return X

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        embed_dim = 4
        input_dim = 4 + embed_dim  # 8
        self.embedding = nn.Embedding(num_obj_types + 1, embed_dim)
        self.convs = nn.ModuleList([
            GINConv(nn.Linear(input_dim, input_dim), train_eps=True) for _ in range(3)
        ])
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, G):
        x = G.x.float()  # [N, 4]
        obj_emb = self.embedding(G.obj_types)  # [N, embed_dim]
        node_feat = torch.cat([x, obj_emb], dim=-1)  # [N, 8]

        edge_index = G.edge_index
        for conv in self.convs:
            node_feat = conv(node_feat, edge_index)

        out = global_mean_pool(node_feat, G.batch)  # [batch_size, 8]
        out = self.classifier(out).squeeze(-1)  # [batch_size]
        return out

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_auc = 0.0
    best_model_state = None
    patience = 5
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        all_train_logits = []
        all_train_labels = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (logits.sigmoid() > 0.5).long()
            train_correct += (preds == batch.y).sum().item()
            total_train += batch.y.size(0)

            all_train_logits.append(logits.cpu().detach())
            all_train_labels.append(batch.y.cpu().detach())

        # Compute train AUC
        train_logits = torch.cat(all_train_logits)
        train_labels = torch.cat(all_train_labels)
        if len(torch.unique(train_labels)) == 1:
            train_auc = 0.5
        else:
            train_auc = self.compute_auc(train_logits, train_labels)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        all_val_logits = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = criterion(logits, batch.y.float())
                val_loss += loss.item()

                preds = (logits.sigmoid() > 0.5).long()
                val_correct += (preds == batch.y).sum().item()
                total_val += batch.y.size(0)

                all_val_logits.append(logits.cpu())
                all_val_labels.append(batch.y.cpu())

        # Compute val AUC
        val_logits = torch.cat(all_val_logits)
        val_labels = torch.cat(all_val_labels)
        if len(torch.unique(val_labels)) == 1:
            val_auc = 0.5
        else:
            val_auc = self.compute_auc(val_logits, val_labels)

        scheduler.step()

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accs.append(train_correct / total_train)
        val_accs.append(val_auc)  # Use AUC as "accuracy" for scoring

        # Early stopping based on val AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_losses[-1], val_losses[-1], train_accs[-1], val_accs[-1]

    @staticmethod
    def compute_auc(logits, labels):
        from sklearn.metrics import roc_auc_score
        probs = torch.sigmoid(logits).numpy()
        auc = roc_auc_score(labels.numpy(), probs)
        return auc

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

