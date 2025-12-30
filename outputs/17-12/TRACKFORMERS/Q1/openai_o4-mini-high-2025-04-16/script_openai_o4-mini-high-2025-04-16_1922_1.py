
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

import numpy as np
import torch
from sklearn.linear_model import RANSACRegressor
from torch import nn

class MyPreprocessor:
    def __init__(self):
        pass

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 64,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": "ragged_xy",
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, data):
        return self

    def transform(self, data):
        return data

def make_preprocessor():
    return MyPreprocessor()

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Minimum hits to form a valid track
        self.min_hits = 4
        # Factor for dynamic RANSAC inlier threshold
        self.factor = 1.5

    def forward(self, batch_x):
        # batch_x: list of Tensors, each [N_i, 4]
        preds = []
        for x in batch_x:
            # x_cpu: numpy array [N_i,4]
            x_cpu = x.detach().cpu().numpy()
            # radial coordinate r [N_i,1]
            r = x_cpu[:, 0].reshape(-1, 1)
            # z coordinate [N_i]
            z = x_cpu[:, 2]
            # layer ids [N_i]
            layers = x_cpu[:, 3]
            # number of unique layers
            n_layers = np.unique(layers).shape[0]
            # span of z
            z_span = z.max() - z.min()
            # dynamic residual threshold
            if n_layers > 1:
                residual_threshold = self.factor * z_span / (n_layers - 1)
            else:
                residual_threshold = z_span
            # total hits
            N_i = x_cpu.shape[0]
            # initialize labels: -1 means noise/unassigned
            labels = -1 * np.ones(N_i, dtype=np.int64)
            # indices not yet assigned to a cluster
            unassigned = np.arange(N_i)
            cluster_id = 1
            # iterative RANSAC to find linear tracks in r-z plane
            while unassigned.shape[0] >= self.min_hits:
                ransac = RANSACRegressor(
                    min_samples=self.min_hits,
                    residual_threshold=residual_threshold,
                    random_state=42
                )
                try:
                    # Fit z = f(r) on unassigned points
                    ransac.fit(r[unassigned], z[unassigned])
                except Exception:
                    break
                inlier_mask = ransac.inlier_mask_  # boolean mask [len(unassigned)]
                if inlier_mask.sum() < self.min_hits:
                    break
                # map inliers back to original indices
                inlier_idx = unassigned[inlier_mask]
                labels[inlier_idx] = cluster_id
                cluster_id += 1
                # remove inliers from further consideration
                unassigned = unassigned[~inlier_mask]
            # convert to tensor and move to original device
            preds.append(torch.from_numpy(labels).to(x.dtype).long().to(x.device))
        return preds

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

EPOCHS = 10

def train_model(model, train_loader, val_loader, epochs):
    # No trainable parameters, so skip training; return dummy metrics
    train_loss = [0.0 for _ in range(epochs)]
    val_loss = [0.0 for _ in range(epochs)]
    train_acc = [0.0 for _ in range(epochs)]
    val_acc = [0.0 for _ in range(epochs)]
    return model, train_loss, val_loss, train_acc, val_acc

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

