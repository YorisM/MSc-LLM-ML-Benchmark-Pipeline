# ./challenges/TRACKFORMERS/trackformers.py

from challenges.challenges import Challenge, Question

trackformers_challenge = Challenge(
    name = "TRACKFORMERS",

    version = "v2.4.1",

    dataset = { 
        "REDVID_10-50_linear" : {
            "Train" : "./challenges/TRACKFORMERS/data/train/REDVID_10-50_linear_train.pkl.gz",
            "Val"   : "./challenges/TRACKFORMERS/data/train/REDVID_10-50_linear_val.pkl.gz",
            "Test"  : "./challenges/TRACKFORMERS/data/test/REDVID_10-50_linear_test.pkl.gz"
            },

        "REDVID_10-50_linear_frac0.05" : {
            "Train" : "./challenges/TRACKFORMERS/data/train/REDVID_10-50_linear_frac0.05_train.pkl.gz",
            "Val"   : "./challenges/TRACKFORMERS/data/train/REDVID_10-50_linear_frac0.05_val.pkl.gz",
            "Test"  : "./challenges/TRACKFORMERS/data/test/REDVID_10-50_linear_frac0.05_test.pkl.gz"
            }
        },

    problem_description = r"""** Problem Description **
Efficiently reconstructing particle trajectories from detector hits is crucial for the performance of particle  physics experiments at colliders like the Large Hadron Collider (LHC). With the significant increase in data volumes expected in the High-Luminosity LHC era . So far, traditional algorithms have generally been used yet traditional tracking methods become computationally expensive and increasingly inefficient. Deep Learning is likely to take over trajectory reconstruction in the future. Compared with classical combinatorial tracking the learned approach scales sub-quadratically, copes gracefully with dense environments, and is far better suited for the extreme occupancies anticipated in the High-Luminosity era. In this challenge we thus cast tracking as a supervised classification problem: for every hit the model must predict the track-ID it belongs to.  
""",

    dataset_description = r"""** Dataset Description **
The provided dataset consists of simulated proton-proton collision events, each containing between 10 and 50 particle tracks. With a median of 45 tracks per event. These events are generated using the REDVID simulation framework, which approximates realistic detector geometries resembling those at the LHC. Each event comprises detector hits represented in cylindrical coordinates. The dataset contains 5000 events, split 80% for training, 10% validation and 10% testing.

IMPORTANT: Each sample is one collision event with variable-length hits, with no explicit ordering of hits. Data flow is as follows:

    A) Raw Events loaded as:
        hit_r       : float32 [N_hits]
        hit_theta   : float32 [N_hits]
        hit_z       : float32 [N_hits]
        layer_id    : float32 [N_hits]
        track_id    : int32   [N_hits]   (per-hit truth track id; event-local; 0 = noise/unassigned)

    B) Default dataset sample (what __getitem__ returns if you do NOT override make_dataset)
        The harness converts the raw dict into a tuple (X, y):
        X = [N_hits, 4] float32 tensor ("hit_r", "hit_theta", "hit_z", "layer_id")
        y = [N_hits] int64 tensor ("track_id" per hit)

    C) Default DataLoader batch (what the training loop receives if you do NOT override loader_class)
        The harness uses a ragged collate. A batch is:
        Xs : list of length B, where Xs[i] is a float32 tensor [N_i, 4]
        ys : list of length B, where ys[i] is an int tensor [N_i]
        So the model must handle variable-length events; padding is allowed internally inside the model.         

Here, "hit_r" represents the r coordinate defining the recorded hit point, "hit_theta" represents the theta coordinate defining the recorded hit point, "hit_z" represents the z coordinate defining the recorded hit point, each of which are on the relevant sub-detector. The "layer_id" indicates which sub-detector the hit was recorded on. It is an incremental identifier for different sub-detector layers belonging to a geometry, which is unique within the scope of the geometry. The "track_id" event-local identifiers: the numeric values have no meaning across different events. Track_id == 0 denotes a noise/unassigned hit (if present). Track_id > 0 denotes membership of a true track within that event. The loader yields ragged batches (list of events). Your model must handle variable-length events - padding may be done internally in the model.
""",

    evaluation_metric = r"""** Evaluation Metric **
The evaluation metric for this particle reconstruction challenge is the FitAccuracy. FitAccuracy (REDVID) is computed in the TrackML style: each predicted cluster is matched to the truth track that contributes the most hits to it. A predicted cluster counts as a valid reconstructed track if it contains at least 4 hits and both (i) at least 50% of its hits come from the matched truth track and (ii) it covers at least 50% of that truth track's hits. The FitAccuracy score is the fraction of all non-noise truth hits that are correctly assigned under these rules (with unit hit weights). A perfect reconstruction obtains FitAccuracy = 1, while missing tracks, duplicated tracks or impurity in any reconstructed track lower the score. In other words, FitAccuracy is a hit-weighted reconstruction efficiency that rewards both high recall (many true hits found) and high purity (bad tracks are vetoed).
""",

    prefix = r"""
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
    pkl = os.path.join(DATA_DIR, f"{{TAG}}_{{split}}.pkl.gz")
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

""",

    code_template = r"""** Code Template **
# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
# <LLM: Import modules>

#  -------- (OPTIONAL) CUSTOM DATASET  --------
# class CustomDataset(Dataset):
#   REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:CustomDataset"
#    def __init__(self, events, pre, train: bool = True, **kwargs):
#        X, y = events
#        self.X = pre.transform(X) if pre is not None else X
#        self.y = y
#    def __len__(self):
#        return int(self.y.shape[0])
#    def __getitem__(self, idx):
#        return self.X[idx], self.y[idx]

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
            "dataset_builder": "utils.llm_io:EventDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",    # or torch_geometric.loader:DataLoader
            "batch_size": 64,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed. Choose one: 
            "collate": "ragged_xy",  # or "identity" or None (If loader_class is torch_geometric.loader:DataLoader, set "collate": None)

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
        return X # MUST return torch.FloatTensor [N_hits, F_out] for the default EventDataset path.

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
# Model batch contract:
#   Your DataLoader batch is NOT guaranteed to be a single Tensor.
#   Depending your dataset/loader choice, a batch can be:
#      - (X, y) tuple OR [X, y] list  (common for default PyTorch/PyG collation)
#      - ragged: X is list[Tensor] and y is list[Tensor] (one Tensor per event)
#      - multi-input: (X1, X2, ..., y) OR [X1, X2, ..., y]
#      - dict-like: {"x": X, "y": y} (or inputs/labels variants)
#      - PyG: torch_geometric.data.Data or torch_geometric.data.Batch
#
# ALWAYS adapt the raw batch using:
#     view = normalise_batch(batch, device=device)
#
# normalise_batch returns a BatchView with:
#   view.batch_x : the model inputs (Tensor / list[Tensor] / tuple / dict / PyG Batch)
#   view.batch_y : labels if present, else None
#
# IMPORTANT: normalise_batch(..., device=device) moves ALL contained tensors to device (recursively). Do NOT call .to(device) on the raw batch object.

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # IMPORTANT: Default harness input:
        #   - batch_x is ragged list[Tensor], one per event, each shaped [N_hits, F].

        # <LLM: Define and initialize any stateful components here>

    def forward(self, batch_x):
        # IMPORTANT Output contract:
        # forward(batch_x) must return predicted integer labels (dtype long/int64) with one label per hit (>0); predicted noise may be -1.

        # <LLM: Define your model's forward pass here>
        pass

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 10   # <LLM: adjust if you wish>
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
    return trained_model, train_loss, val_loss, train_acc, val_acc

# IMPORTANT: DO NOT execute the pipeline here - the harness will do that.
# <end code template>
# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
""",

    suffix = r"""
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
""",

    questions = [
        Question("Q1", r""" ** IMPORTANT: Your Challenge **
Write Python code for a model that classifies events into tracks: per-hit labeling. The model must output one integer label per hit (cluster id), per event. Labels may be arbitrary up to permutation. Focus on maximising the FitAccuracy using the code template above. You may freely choose any pre-processing methods and techniques as well as model architecture and training conventions as long as it respects the template and harness. Do absolutely everything in your power to achieve the highest possible FitAccuracy.                        
""")
]
)
