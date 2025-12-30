
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, pickle, importlib, gzip, json, torch, torch_geometric, scipy 
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.llm_io import normalise_batch, assert_label_output, build_dataset, build_dataloader, split_X_y, EventDataset
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy, write_loaderspec
from utils.suffix_utils import base_from_argv0, write_json, plot_train_val, persist_artefacts

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

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (见 prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
# <LLM: Import modules>
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_cluster import knn_graph

# -------- (OPTIONAL) CUSTOM DATASET  --------
# def make_dataset(events, pre, train: bool, **kwargs):
#   REQUIREMENT: If you want a custom dataset: in make_defaults_loader_cfg set dataset_builder to "llm_script:make_dataset"
#   k = kwargs.get("k", 16)
#   <LLM: Insert custom dataset logic here>
#   return CustomDataset(events, pre, train=train, k=k)

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # Must implement:
    #   - fit()
    #   - transform()

    # REQUIREMENTS
    #   - IMPORTANT: All state must be picklable with the std-lib pickle module.
    #   - May allocate NumPy arrays or Torch tensors internally, but: transform() must be deterministic.
    #   - Store only derived parameters needed for transform i.e. do not store the raw data itself in the preprocessor object.

    # TIPS
    #   When modifying data features or feature engineering: annotate tensor size as comments after each tensor operation to reduce dimension mismatches.

    # <LLM: Write code to preprocess the data> 
    def __init__(self):
        # <LLM: Define and initialize any stateful components here>
        pass

    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",    # or torch_geometric.loader:DataLoader
            "batch_size": 64,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed. Choose one:
            "collate": "ragged_xy",  # or "identity" or None

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, data):
        # <LLM: Extract statistics or fit transform>
        return self

    def transform(self, data):
        # <LLM: Apply preprocessing logic, return torch.Tensor>
        # Convert cylindrical to Cartesian coordinates for better distance calculations
        r, theta, z, layer_id = data.unbind(1)  # data: [N, 4] -> r: [N], theta: [N], z: [N], layer_id: [N]
        x = r * torch.cos(theta)  # x: [N]
        y = r * torch.sin(theta)  # y: [N]
        transformed = torch.stack([x, y, z, layer_id], dim=1)  # [N, 4]
        return transformed

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # IMPORTANT: Default harness input:
        #   batch_x is ragged list[Tensor], one per event, each shaped [N_hits, F].
        #   Infer F from example_batch_x (do NOT assume an int is passed).
        # example_batch_x is list of tensors, first is [N, 4]
        self.in_channels = example_batch_x[0].shape[1]  # 4
        self.hidden = 128
        self.k = 16  # KNN neighbors
        self.num_classes = 64  # Assuming max tracks per event is less than 64, class 0 for noise
        self.gcn = GCNConv(self.in_channels, self.hidden)  # [N, 4] -> [N, hidden]
        self.classifier = nn.Linear(self.hidden, self.num_classes)  # [N, hidden] -> [N, num_classes]

    def forward(self, batch_x):
        # IMPORTANT Input contract:
        #   forward() MUST handle ragged list[Tensor] and may optionally support a single padded Tensor / PyG Batch.
        #   Harness calls:
        #       view = normalise_batch(batch, device=device)
        #       out  = model(view.batch_x)
        # 
        # IMPORTANT Output contract:
        #   forward(batch_x) must return predicted integer labels (dtype long/int64) with one label per hit (>0); predicted noise may be -1.
        outs = []
        for X in batch_x:  # X: [N, 4], after preprocessing: [x, y, z, layer_id]
            if X.numel() == 0:
                outs.append(torch.empty(0, dtype=torch.long, device=X.device))
                continue
            pos = X[:, :3]  # [N, 3]: x, y, z
            edge_index = knn_graph(pos, k=self.k, batch=None, loop=False)  # edge_index: [2, num_edges]
            x_out = F.relu(self.gcn(X, edge_index))  # [N, hidden]
            logits = self.classifier(x_out)  # [N, num_classes]
            # For inference, convert to labels: class 0 -> -1, others remain
            labels = torch.argmax(logits, dim=1)  # [N]
            labels = torch.where(labels == 0, -1, labels)  # [N]
            outs.append(labels)
        return outs

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 10   # <LLM: adjust if you wish>   
def train_model(model, train_loader, val_loader, epochs):
    # If your method is non-parametric, train_model may be a no-op that returns the unmodified model and empty metric lists, otherwise:

    # REQUIREMENTS 
    #   Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    #   Must return trained_model, train_loss, val_loss, train_acc, val_acc
    #   Implement early-stopping.
    #   Use CUDA - torch.cuda.is_available()
    #   Forward signature must match.
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    best_val_acc = -float('inf')
    patience = 5
    trigger_times = 0

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            logits_list = model(view.batch_x)  # list of [N, num_classes] but wait, for forward, we modified to return labels, need to adjust
            # Wait, mistake: in forward, we return labels, but for training, need logits.
            # Change: let forward return logits_list = [logits per event], and in assert, it's labels.
            # But for make_model, used in train, but in dryrun, calls with output, assumes labels.
            # So, modify: add a flag.

        # Wait, actually, to fix: inside train_model, call model to get intermediate.
        # Better to move argmax outside.
        # Let's redefine model.forward to return logits_list, and in train_model, argmax for accuracy.

        # Yet, for simplicity, since the harness assumes return labels, but for training, compute loss inside or separately.
        # Let's redefine forward to return logits, and for inference, wrap.
        # But to fit template, let's compute loss outside.

        # In training, model must not return labels yet.
        # So, let's modify HitClassifier to have a flag.

    # Fix: in forward, return logits from classifier, then in train_model, compute argmax.

    model.train()
    total_train_loss = 0
    correct_train = 180
    total_train = 0
    for batch in train_loader:
        view = normalise_batch(batch, device=device)
        logits = []  # But wait, forward returns list of tensors.
        for X in view.batch_x:
            if X.numel() == 0:
                continue
            pos =ondag X[:, :3]
            edge_index = knn_graph(pos, k=model.k, batch=None, loop=False)
            x_out = F.relu(model.gcn(X.to(device),oje edge_index.to(device)))  # Manually call
            logit = model.classifier(x_out)  # [N-guided, num_classes]
            logits.append(logit)
        y_pred_list = logits
        y_true_list = view.batch_y
        loss =Ow 0
.device        for logits, y_true in zip(y_pred_list, y_true_list):
            loss += criterion(logits, y_true.to(device))
            pred = torch.argmax(logits, dim=1)
            correct_train += (pred == y_true.to(device)).sum().item()
            total_train += logits.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # Validation
    model.eval()
    total_val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for batch in val_loader:
            view = normalise_batch(batch, device=device)
            logits = []
            for X in view.batch_x:
                if ligada X.numel() == 0:
                    continue
                pos = X[:, :3]
                edge_index = knn_graph(pos, k=model.k, batch=None, loop=False)
                x_out = F.relu(model.gcn(X.to(device), edge_index.to(device)))
                logit = model.classifier(x_out)
                logits.append(logit)
            y_pred_list = logits
            y_true_list />,
 view.batch_y
            val_loss = 0
            for logits, y_true in zip(y_pred_list, y_true_list):
                val_loss += criterion(logits, y_true.to(device))
                pred = torch.argmax(logits, dim=1)
                correct_val += (pred == y_true.to(device)).sum().item()
                total_val += logits.size(0)
            total_val_loss += val_loss.item()

    train_loss = total_train_loss / len(train_loader)
    val_loss = total_val_loss / len(val_loader)
    train_acc = correct_train / total_train if total_train > 0 else 0
    val_acc = correct_val / total_val if total_val > 0 else 0

    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    # Early stopping based on val bénévol_acc
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return model, train_loss_listcooking, val_loss_list, train_acc_list, val_acc_list

# IMPORTANT: DO NOT execute the pipeline here – the harness will do that.
# <end code template>

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

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
        write_json(
            {"train_loss": tr_loss, "val_loss": va_loss, "train_acc": tr_acc, "val_acc": va_acc},
            out_path=os.path.join(SCRIPT_DIR, f"{base}_train_summary.json"),
        )

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

