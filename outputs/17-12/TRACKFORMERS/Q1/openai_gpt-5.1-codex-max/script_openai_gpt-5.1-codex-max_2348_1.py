
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
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
import math

# -------- (OPTIONAL) CUSTOM DATASET  --------
# def make_dataset(events, pre, train: bool, **kwargs):
#   REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:make_dataset"
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

    def __init__(self):
        pass

    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",    # or torch_geometric.loader:DataLoader
            "batch_size": 16,
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
        return self

    def transform(self, data):
        return data # must return an indexable, picklable object

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # No learnable parameters; deterministic clustering in forward.
        self.theta_threshold = 0.12
        self.skip_threshold = 0.15
        self.w_z = 0.05

    def forward(self, batch_x):
        # Handle ragged list of events or single tensor.
        if isinstance(batch_x, torch.Tensor):
            batch_list = [batch_x]
            single = True
        elif isinstance(batch_x, list):
            batch_list = batch_x
            single = False
        else:
            # Fallback if PyG Batch is provided
            if hasattr(batch_x, "to_data_list"):
                batch_list = [d.x for d in batch_x.to_data_list()]
                single = False
            else:
                batch_list = [batch_x]
                single = True
        outputs = []
        for x in batch_list:
            if x.numel() == 0:
                outputs.append(torch.empty((0,), dtype=torch.long, device=x.device))
                continue
            device = x.device
            x_cpu = x.detach().cpu()
            r = x_cpu[:, 0].numpy()
            theta = x_cpu[:, 1].numpy()
            z = x_cpu[:, 2].numpy()
            layer = x_cpu[:, 3].numpy().astype(int)
            N = len(theta)
            if N == 0:
                outputs.append(torch.empty((0,), dtype=torch.long, device=device))
                continue
            z_std = float(np.std(z) + 1e-3)
            layers_unique = sorted(list(set(layer.tolist())))
            parent = list(range(N))
            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra == rb:
                    return
                parent[rb] = ra
            layer_to_indices = {l: np.where(layer == l)[0] for l in layers_unique}
            def angular_diff(a, b):
                return np.abs(((a - b + math.pi) % (2 * math.pi)) - math.pi)
            def connect_layers(l1, l2, thr):
                idx1 = layer_to_indices.get(l1, [])
                idx2 = layer_to_indices.get(l2, [])
                if len(idx1) == 0 or len(idx2) == 0:
                    return
                for i in idx1:
                    dt = angular_diff(theta[i], theta[idx2])
                    dz = np.abs(z[i] - z[idx2]) / z_std
                    dist = dt + self.w_z * dz
                    j_sel = idx2[np.argmin(dist)]
                    if dist.min() <= thr:
                        union(i, j_sel)
                for j in idx2:
                    dt = angular_diff(theta[j], theta[idx1])
                    dz = np.abs(z[j] - z[idx1]) / z_std
                    dist = dt + self.w_z * dz
                    i_sel = idx1[np.argmin(dist)]
                    if dist.min() <= thr:
                        union(i_sel, j)
            for li in range(len(layers_unique) - 1):
                l = layers_unique[li]
                ln = layers_unique[li + 1]
                connect_layers(l, ln, self.theta_threshold)
            for li in range(len(layers_unique) - 2):
                l = layers_unique[li]
                ln = layers_unique[li + 2]
                connect_layers(l, ln, self.skip_threshold)
            root_to_indices = {}
            for i in range(N):
                rroot = find(i)
                root_to_indices.setdefault(rroot, []).append(i)
            labels = -torch.ones(N, dtype=torch.long)
            cluster_id = 1
            for inds in root_to_indices.values():
                if len(inds) >= 4:
                    for idx in inds:
                        labels[idx] = cluster_id
                    cluster_id += 1
            outputs.append(labels.to(device))
        if single:
            return outputs[0]
        return outputs

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 1
def train_model(model, train_loader, val_loader, epochs):
    # No training needed for deterministic model; return placeholders.
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

