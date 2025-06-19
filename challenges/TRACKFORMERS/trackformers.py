# ./challenges/TRACKFORMERS/trackformers.py

from challenges.challenges import Challenge, Question

# To FIX:
# -------
# Datatype in preproc ambiguity
#       MyPreprocessor.transform(self, events) -> MyPreprocessor.transform(self, hits: torch.Tensor)
# Import PyG

trackformers_challenge = Challenge(
    name = "TRACKFORMERS",

    dataset = { 
        "REDVID_10-50_linear" : {
            "Train" : "./challenges/TRACKFORMERS/data/REDVID_10-50_linear_train.pkl.gz",
            "Val"   : "./challenges/TRACKFORMERS/data/REDVID_10-50_linear_val.pkl.gz",
            "Test"  : "./challenges/TRACKFORMERS/data/REDVID_10-50_linear_test.pkl.gz"
            }
        },

    problem_description = r"""** Problem Description **
Efficiently reconstructing particle trajectories from detector hits is crucial for the performance of
particle physics experiments at colliders like the Large Hadron Collider (LHC). With the significant increase
in data volumes expected in the High-Luminosity LHC era. So far, traditional conventional algorithms have generally 
been used yet traditional tracking methods become computationally expensive and increasingly inefficient. 
Deep Learning is likely to take over trajectory reconstruction in the future. 
""",

# The challenge here is to develop a two-step regression network that efficiently handles events with variable-length
# hit sequences. Your model should first group hits into distinct track clusters and then accurately predict
# track parameters for each identified cluster. Accurate reconstruction of track parameters enables precise
# measurements of fundamental particle properties and event reconstructions.

    dataset_description = r"""** Dataset Description **
The provided dataset consists of simulated proton-proton collision events, each containing between 10 and 50
particle tracks. With a median of 45. These events are generated using the REDVID simulation framework, which approximates realistic
detector geometries resembling those at the LHC. Each event comprises detector hits represented in cylindrical 
coordinates. The dataset contains 100000 events, split 80% for training, 10% evaluation and 10% testing.

IMPORTANT: The specific line format of the dataset is represented as a tensor dictionary as follows:

{
  "hit_r"       : np.array of floats,     # [N_hits,]
  "hit_theta"   : np.array of floats,     # [N_hits,]
  "hit_z"       : np.array of floats,     # [N_hits,]
  "layer_id"    : np.array of ints  ,     # [N_hits,]
}

Here, "hit_r" represents the r coordinate defining the recorded hit point, "hit_theta" represents the theta
coordinate defining the recorded hit point, "hit_z" represents the z coordinate defining the recorded hit point,
each of which are on the relevant sub-detector. The "layer_id" indicates which sub-detector the hit was recorded
on. It is an incremental identifier for different sub-detector layers belonging to a geometry, which is unique within
the scope of the geometry. Three different sub-detector types are available: pixel, short-strip, and long-strip.
The data is variable-length per event, with no explicit ordering of hits.
""",

    evaluation_metric = r"""** Evaluation Metric **
The primary evaluation metric is the accuracy, which measures the fraction of hits that are correctly associated
with their corresponding true particle tracks based on clustering accuracy.
""",

    prefix = r"""
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: Python 3.12, PyTorch 2.6.0, PyG 2.6.1, NumPy 2.2.3, SciPy v1.15.2, SciKit-Learn 1.6.1
import os, sys, pickle, gzip, json, torch, torch_geometric, scipy, numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_DIR = "./challenges/TRACKFORMERS/data"
TAG      = "10_50_linear"

def _load_events(split: str):
    pkl = os.path.join(DATA_DIR, f"REDVID_{{TAG}}_{{split}}.pkl.gz")
    with gzip.open(pkl, "rb") as fh:
        return pickle.load(fh)["events"]

def split_X_y(evt):
    lay = evt["layer_id"].astype(np.float32)
    lay_norm = lay / lay.max()
    X = np.column_stack([evt["hit_r"],
                         evt["hit_theta"],
                         evt["hit_z"],
                         lay_norm])
    t_id = evt["track_id"].astype(np.int32)
    return (torch.from_numpy(X),
            torch.from_numpy(t_id))

class EventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, track_id = split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        return (X, track_id)

def _ragged(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    # batch[i] = (hits_i, track_id_i)      ← shapes: (N_i, F), (N_i,)
    return batch

def make_loaders(batch_size=128, workers=0):
    tr = EventDataset(_load_events("train"), pre=None, train=True)
    va = EventDataset(_load_events("val"),   pre=None, train=False)

    train_ld = DataLoader(tr, batch_size=batch_size, shuffle=True,
                          collate_fn=_ragged, num_workers=workers)
    val_ld   = DataLoader(va, batch_size=batch_size, collate_fn=_ragged)
    return train_ld, val_ld

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

""",

    code_template = r"""** Code Template **
# <start code template>
# 0. ---------- IMPORTS ----------
# NOTE: Some imports (torch, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch, scipy, sklearn (sub-)modules you actually use.
# <LLM: Import modules>

# 1. ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # REQUIREMENTS
    # IMPORTANT: All state must be picklable with the std-lib pickle module.
    # May allocate NumPy arrays or Torch tensors internally, but:
    # transform() must be deterministic.
    # fit(events) receives the *raw* event dicts list, not a tensor batch.
    # Store only derived parameters needed for transform i.e. do not store the raw data
    # itself in the preprocessor object.
    def __init__(self):
        # <LLM: Define and initialize any stateful components here>
        pass

    def fit(self, events):
        # <LLM: Extract statistics or fit transformers>
        return X

    def transform(self, events):
        # <LLM: Apply preprocessing logic, return torch.Tensor>
        return X

    def fit_transform(self, events):
        self.fit(events)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()
    
# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        # <LLM: Define and initialize any stateful components here>

    # <LLM: optionally build extra layers here>
    
    def forward(self, batch):
        # batch : list[torch.Tensor]  or  list[tuple[Tensor, Tensor]]
        #    You decide: either just the hits or the (hits, labels) tuples.

        # <LLM: Define your model's forward pass here>

def make_model(in_features):
    return HitClassifier(in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 10   # <LLM: adjust if you wish>   
def train_model(model, train_loader, val_loader, epochs):
    # PARAMETERS
    # model     : torch.nn.Module   
    # train_loader / val_loader
    # epochs    : int

    # RETURNS
    # trained_model : nn.Module          (same instance, trained in-place)
    # train_loss    : list[float]        (length == epochs)
    # val_loss      : list[float]
    # train_acc     : list[float]
    # val_acc       : list[float]
    
    # REQUIREMENTS 
    # Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    # Implement early-stopping.

    # <LLM: Write code to define training loop>
    return trained_model, train_loss, val_loss, train_acc, val_acc

# IMPORTANT: DO NOT execute the pipeline here - the harness will do that.
# IMPORTANT: Strictly follow this code template, and do not deviate from it.
# <end code template>    
""",

    suffix = r"""
# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    raw_train, raw_val = _load_events("train"), _load_events("val")
    if dryrun:
        raw_train, raw_val = raw_train[:32], raw_val[:8]
    pre = make_preprocessor().fit(raw_train)
    train_ds = EventDataset(raw_train, pre, train=True)
    val_ds   = EventDataset(raw_val , pre, train=False)
    train_ld = DataLoader(train_ds, batch_size=512,
                        shuffle=True, collate_fn=_ragged)
    val_ld   = DataLoader(val_ds,   batch_size=512,
                        collate_fn=_ragged)

    # 2. Build model
    in_features = train_ds[0][0].shape[-1]                   
    model = make_model(in_features)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_ld, val_ld, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* - run a single toy forward pass
    if dryrun:
        toy_event       = torch.zeros(10, in_features)
        toy_transformed = pre.transform(toy_event)
        toy_batch       = [toy_transformed]
        try:
            _ = trained_model(toy_batch)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 5. Persist artefacts
    if not dryrun:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

        pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
        pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
        pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

        torch.save(trained_model.state_dict(), pth_state)
        with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
        with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

        # 6. Save plots
        _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
        _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

    # 7. Write JSON Summary
    if not dryrun: 
        summary = {
            "epochs": n_epochs,
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
""",

    questions = [
        Question("Q1", r""" ** IMPORTANT: Your Challenge **
Write Python code for a classification model that classifies events into tracks. Focus on maximising the accuracy
using the code template above. You may freely choose any pre-processing methods and techniques as well as model
architecture and training conventions. Do everything in your power to get the highest accuracy.                        
""")

#       Question("Q2", r""" ** IMPORTANT: Your Challenge **
#Write Python code for a classification model that classifies events into tracks. Focus on maximising the accuracy
#using the code template above. You may freely choose any pre-processing methods and techniques as well as model
#architecture and training conventions. 
# 
#
# 
# 
#Do everything in your power to get the highest accuracy.      
#"""")


]
)
