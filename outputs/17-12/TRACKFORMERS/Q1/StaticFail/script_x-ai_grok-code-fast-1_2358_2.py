
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

# ---------- IMPORTS ----------
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.utils import knn_graph
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# -------- (OPTIONAL) CUSTOM DATASET  --------
def make_dataset(events, pre, train: bool, **kwargs):
    k = kwargs.get("k", 16)
    dataset = []
    for event in events:
        X, y = split_X_y(event)
        x = pre.transform(X)
        # Build k-NN graph based on features x [N, F]
        edge_index = knn_graph(x, k=k, loop=False)
        data = Data(x=x, edge_index=edge_index, y=y.long())
        dataset.append(data)
    return dataset

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scalers = {}

    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "llm_script:make_dataset",
            "dataset_kwargs": {"k": 16},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 64,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, data):
        # data is list of [N, 4] tensors
        combined = torch.cat(data, dim=0)  # [total_hits, 4]
        combined_np = combined.numpy()
        scalers = []
        for i in range(4):
            scaler = StandardScaler()
            scaler.fit(combined_np[:, i].reshape(-1, 1))
            scalers.append(scaler)
        self.scalers = scalers
        return self

    def transform(self, data):
        # data is [N, 4] tensor
        data_np = data.numpy()
        transformed = []
        for i in range(4):
            transformed.append(self.scalers[i].transform(data_np[:, i].reshape(-1, 1)))
        x = torch.tensor(np.concatenate(transformed, axis=1), dtype=torch.float32)
        return x  # [N, 4]

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch):
        super().__init__()
        # example_batch is Batch from PyG, so example_batch.x is [total_hits, 4], but actually first batch
        # Infer num_classes: assume max track_id <= 100
        self.num_classes = 101  # 0 to 100
        self.conv1 = GCNConv(4, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        self.fc = nn.Linear(256, self.num_classes)

    def forward(self, batch):
        # batch is PyG Batch, x [total_hits, 4], edge_index, etc.
        x = F.relu(self.conv1(batch.x, batch.edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.conv2(x, batch.edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.conv3(x, batch.edge_index))
        logits = self.fc(x)  # [total_hits, num_classes]
        preds = torch.argmax(logits, dim=-1).long()  # [total_hits]
        return preds

def make_model(example_batch):
    return HitClassifier(example_batch)

# ---------- MODEL TRAINING ----------
EPOCHS = 20  # increased for better training
def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    best_acc = 0.
    patience = 5
    counter = 0
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    def accuracy(preds, targets):
        correct = (preds == targets).sum().item()
        total = targets.numel()
        return correct / total

    for epoch in range(epochs):
        model.train()
        train_loss = 0.
        train_correct = 0
        total_hits = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            preds = model(batch)
            # logits were intermediate, need to recompute for loss? Wait, no.
            # Problem: forward returns preds, but need logits for loss.
            # NEED TO MODIFY forward to return logits.
            # Wait, in template, output is labels, but for training, need logits.

            # I made a mistake. In forward, I need to return the output that has integer labels only for inference, but for training, the harness will call model(view.batch_x), but wait, in training loop, I handle it.

            # To fix: modify forward to return logits, but in predict, argmax.

            # But the contract is to return integer labels for inference. For training, I need to access logits.

            # Perhaps add a method or change.

            # Better: make forward return [total_hits, num_classes] logits, and the harness is set to handle inference by argmax.

            # Looking back: "forward(batch_x) must return predicted integer labels...", so it expects labels.

            # For training, I need to compute loss myself.

            # So, in train, I need to have the model give logits.

            # Let's add a flag or subclass.

            # Simplest: change forward to always return logits, and in the harness, they will argmax it? No, it says return integer labels.

            # The problem says "return predicted integer labels", so for training, I can't.

            # Perhaps the harness uses it differently, but to make it work, I'll assume in training I can define loss.

            # To fix: make forward return dict with 'preds' and 'logits', but that may not match.

            # The output contract: "forward(batch_x) must return predicted integer labels", but in code, it's already implemented as preds, so the harness probably calls it for inference.

            # But for training, this loop is provided, and I need to define loss.

            # So, for training, I need to call the model but have it give logits.

            # Let's modify the model to have a forward that returns logits, and have a method to get preds.

            # Update: change forward to return logits, and assume the harness argmaxes it.

            # Looking at assert_label_output, it checks if out is tensor of ints.

            # Probably the harness does torch.argmax(out) if needed, but the contract says return integer labels.

            # To escape, I'll make forward return preds.long(), and for loss, recompute internally.

            # No, too messy.

            # Let's redefine the model to have a property.

            # Better: wrap the training.

            # In train_model, the forward call is for inference, so to train, I need to do the forward computation again for loss.

            # So, in the loop, to get logits, I'd need to run model.conv forward, but that's ugly.

            # The clean way: have the model forward return logits, and in the prediction part, argmax it, but the contract is to return labels, so perhaps it's okay if the harness doesn't call it like that.

            # Looking at the code, in dryrun, it calls out = model(view.batch_x), and assert_label_output(view.batch_x, out, allow_noise_label=True), so out is expected to be label tensor.

            # For training, I need to compute loss based on labels, but no, I need logits.

            # To resolve, I'll make forward return the preds, but in train_model, re define the forward to get logits.

            # Perhaps change the class to have a get_logits method.

            # Let's do that.

# Update model:
class HitClassifier(nn.Module):
    # ... as above

    def get_logits(self, batch):
        x = F.relu(self.conv1(batch.x, batch.edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.conv2(x, batch.edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.conv3(x, batch.edge_index))
        logits = self.fc(x)  # [total_hits, num_classes]
        return logits

    def forward(self, batch):
        logits = self.get_logits(batch)
        return torch.argmax(logits, dim=-1).long()

    # And in train_model, use get_logits.

# Yes.

    def train_model(model, train_loader, val_loader, epochs):
        # ... as above
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = model.get_logits(batch)
                loss = F.cross_entropy(logits, batch.y, reduction='mean')  # y may include 0, but okay
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=-1)
                    train_correct += (preds == batch.y).sum().item()
                total_hits += batch.y.numel()
            train_loss /= len(train_loader)
            train_acc = train_correct / total_hits
            # Validation
            model.eval()
            val_loss = 0.
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    logits = model.get_logits(batch)
                    loss = F.cross_entropy(logits, batch.y, reduction='sum')
                    val_loss += loss.item()
                    preds = torch.argmax(logits, dim=-1)
                    val_correct += (preds == batch.y).sum().item()
                    val_total += batch.y.numel()
            val_loss /= val_total  # average loss
            val_acc = val_correct / val_total
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            scheduler.step()
            if val_acc > best_acc:
                best_acc = val_acc
                counter = 0
                best_model = model.state
            else:
                counter += 1
                if counter >= patience:
                    break
        # Load best model
        model.load_state_dict(best_model)
        trained_model = model
        return trained_model, train_loss_list, val_loss_list, train_acc_list, val_acc_list
# Note: in code, I used train_loss_list.append(train_loss), etc.

        return trained_model, train_loss_list, val_loss_list, train_acc_list, val_acc_list

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

