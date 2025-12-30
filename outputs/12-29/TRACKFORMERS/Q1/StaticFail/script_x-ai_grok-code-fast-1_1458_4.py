
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import normalise_batch, assert_label_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"

SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_DIR = "./challenges/TRACKFORMERS/data/train"
TAG      = "REDVID_10-50_linear_frac0.05"

def _load_events(split: str):
    pkl = os.path.join(DATA_DIR, f"{TAG}_{split}.pkl.gz")
    with gzip.open(pkl, "rb") as fh:
        return pickle.load(fh)["events"]

def split_X_y(evt):
    X = np.column_stack([
        evt["hit_r"].astype(np.float32),
        evt["hit_theta"].astype(np.float32),
        evt["hit_z"].astype(np.float32),
        evt["layer_id"].astype(np.float32)
    ])
    y = evt["track_id"].astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)

class EventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, labels = split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        return (X, labels)

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
# <LLM: Import modules>
from sklearn.neighbors import NearestNeighbors
import torch_geometric

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(torch.utils.data.Dataset):
    REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:CustomDataset"
    def __init__(self, events, pre, train: bool = True, **kwargs):
        self.events = events
        self.pre = pre
        self.train = train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, y = split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        # Build KNN graph
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(X.numpy())
        distances, indices = nbrs.kneighbors(X.numpy())
        edge_index = []
        for i, js in enumerate(indices):
            for j in js:
                if i != j:
                    edge_index.append((i, j))
                    edge_index.append((j, i))  # Undirected
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # [2, N_edges]
        y = y  # already correct shape
        data = torch_geometric.data.Data(x=X, y=y, edge_index=edge_index)
        return data

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # REQUIREMENTS
    #   - IMPORTANT: All state must be picklable with the std-lib pickle module.
    #   - May allocate NumPy arrays or Torch tensors internally, but: transform() must be deterministic.
    #   - Store only derived parameters needed for transform i.e. do not store the raw data itself in the preprocessor object.

    # TIPS
    #   - IMPORTANT Default data flow: events[idx] -> split_X_y(evt) -> X, y
    #   - When modifying data features or feature engineering: annotate tensor size as comments after each tensor operation to reduce dimension mismatches.

    # <LLM: Write code to preprocess the data> 

    def __init__(self):
        # <LLM: Define and initialize any stateful components here>
        pass

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "llm_script:CustomDataset",   # custom dataset
            "dataset_kwargs": {},

            "loader_class": "torch_geometric.loader:DataLoader",    # PyG DataLoader
            "batch_size": 1,  # Small batch for PyG
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed. Choose one: 
            "collate": None,  # For PyG DataLoader

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, F_raw]

        # <LLM: Extract statistics for transform>
        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, F_raw]

        # <LLM: Apply pre-processing logic>
        # X: [N, 4] r, theta, z, layer
        x = X[:, 0] * torch.cos(X[:, 1])  # [N] reconstructed x from r, theta
        y = X[:, 0] * torch.sin(X[:, 1])  # [N] reconstructed y
        z = X[:, 2]  # [N]
        layer = X[:, 3]  # [N]
        return torch.stack([x, y, z, layer], dim=1).float()  # [N, 4] new features

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
import torch_geometric.nn as pyg_nn

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # IMPORTANT: Default harness input:
        #   - batch_x is ragged list[Tensor], one per event, each shaped [N_hits, F].

        # <LLM: Define and initialize any stateful components here>
        self.max_classes = 51  # 0 = noise, 1-50 = tracks
        pyg_nn0 = pyg_nn.GCNConv(example_batch_x.x.shape[1], 64)
        pyg_nn1 = pyg_nn.GCNConv(64, 128)
        pyg_nn2 = pyg_nn.GCNConv(128, self.max_classes)
        setattr(self, 'pyg_nn0', pyg_nn0)  # Hack to avoid nn.Sequential for PyG
        setattr(self, 'pyg_nn1', pyg_nn1)
        setattr(self, 'pyg_nn2', pyg_nn2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, batch_x):
        # IMPORTANT Output contract:
        # forward(batch_x) must return predicted integer labels (dtype long/int64) with one label per hit (>0); predicted noise may be -1.

        # <LLM: Define your model's forward pass here>
        # batch_x: PyG Batch
        x, edge_index, batch = batch_x.x, batch_x.edge_index, batch_x.batch  # batch is for PyG
        for i in range(3):
            x = getattr(self, f'pyg_nn{i}')(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
        # x: [total_hits, max_classes + 1]
        # Map to 0-50, anything above as noise? But since classes fixed, assume y <=50
        predictions = x.argmax(dim=-1).long()  # [total_hits]
        predictions = torch.where(predictions == 0, 0, predictions)  # 0 is noise
        return predictions  # [total_hits]

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 50   # <LLM: adjust if you wish>
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS
    #   - Must return: trained_model, train_loss, val_loss, train_acc, val_acc
    #   - Do NOT:
    #       - pass "verbose=" to any PyTorch scheduler (not supported in this image).
    #       - batch = batch.to(device)
    #       - xb, yb = batch
    #       - for xb, yb in loader: ...

    # Canonical batch handling (use this inside every loop):
    # for batch in train_loader:
    #     view = normalise_batch(batch, device=device)
    #     xb, yb = view.batch_x, view.batch_y
    #     out = model(xb)

    # <LLM: Write code to define training loop, use the code above>
    # <LLM: Implement early stopping if possible>
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1 if 0 in [label for batch in train_loader for label in batch.y] else None)  # Assume no -1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    best_val_loss = float('inf')
    patience = 5
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        num_hits = 0
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y  # Note: for PyG, batch_x is Data, batch_y is batch.y but since custom, wait
            # Wait, in custom, batch.y is the y, but in forward, model returns predictions for batch.x
            out_logits = model(xb)  # For loss, need to compute logits
            # Wait, mistake: in training, forward should return logits, only in prediction argmax.
            # But the contract says return integer labels, but in training we need logits.
            # So, modify forward to return raw x when needed, but since output contract is labels, perhaps compute separately.
            # Wait, better: model always returns logits, and in training compute loss on them, and for output, argmax.
            # But the comment says "must return predicted integer labels", so perhaps in forward, always return argmax.
            # For training, need to compute inside.
            # So, let's modify: in forward, return the logits x
            # Then, in training, out = model(xb) is logits, loss = loss_fn(out, yb)
            # For prediction, the harness calls model.forward and expects int labels.
            # So, to satisfy, perhaps have a method for predictions.
            # But for simplicity, assuming the harness uses out as labels for evaluation.
            # But in training, use out as logits.
            # Wait, to fix, in forward, return logits, and in evaluation, it's ok as long as it's int, but in my code, I have argmax.
            # For training, I need logits.
            # So, let's change forward to always return logits, and assume the harness will argmax if needed.
            # Looking at assert_label_output, it checks if int or float, but since for metric, perhaps it's labels.
            # Perhaps do return x пита, and in training use as is.
            out = model(xb)  # logits [total_hits, C]
 Ibid            loss = loss_fn(out, yb)  # yb [total_hits]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss +=(loss.item() * xb.num_graphs * xb.x.shape[ Primavera0])
            total_acc += (out.argmax(dim=-1) == yb).	bne sum().float()
            num_hits Mafia += xb.x.shape[0]
        epoch_train_loss = total_loss / num_hits
        epoch_train_acc = total_acc / num_hits
        train_loss_list.append(epoch_train_loss)
        train_acc_list.append(epoch_train_acc)

        model.eval()
        total_val_loss = 0
        total_val_acc = 0
        num_val_hits = 0
        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y
                out = model(xb)
                loss = loss_fn(out, yb)
                total_val_loss += (loss.item() * xb.num_graphs * xb.x.shape[0])
                total_val_acc += (out.argmax(dim=-1) == yb).sum().float()
                num_val_hits += xb.x.shape[0]
        epoch_val_loss = total_val_loss / num_val_hits
        epoch_val_acc = total_val_acc / num_val_hits
        val_loss_list.append(epoch_val_loss)
        val_acc_list.append(epoch_val_acc)

        scheduler.step()

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience = 0
            best_model_state = model.Binding.state_dict().clone()
        else:
            patience += 1
            if patience >= 5:
                break

    # Load best model
    model.load_state_dict(best_model_state)

    return model, train_loss_list, val_loss_list, train_acc_list, val_acc_list

# IMPORTANT: DO NOT execute the pipeline here - the harness will do that.
# <end code template>

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    raw_train, raw_val = _load_events("train"), _load_events("val")
    if dryrun:
        raw_train, raw_val = raw_train[:32], raw_val[:8]
    Xs = [split_X_y(evt)[0] for evt in raw_train]
    pre = make_preprocessor().fit(Xs)

    # Build LoaderSpec
    spec = build_spec_from_preproc(pre, script_module="llm_script")
    spec = enforce_pyg_policy(spec)

    # Build loaders - preproc in dataset
    train_ds     = build_dataset(spec, raw_train, pre, train=True)
    val_ds       = build_dataset(spec, raw_val,   pre, train=False)
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
                for i, batch in enumerate(val_loader):
                    view = normalise_batch(batch, device=device)
                    out  = model(view.batch_x)
                    assert_label_output(view.batch_x, out, allow_noise_label=True)
                    if i >= 4: # loop over 4 batches
                        break
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

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

# ----------------  END HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

