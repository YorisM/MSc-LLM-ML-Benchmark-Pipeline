# ./challenges/FOURTOPS/fourtops.py

from challenges.challenges import Challenge, Question

fourtop_challenge = Challenge(
    name = "FOURTOPS",

    version = "v2.4.1",

    dataset = {
    "X_train": "./challenges/FOURTOPS/data/train/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/train/Y_train.csv",
    "X_val":   "./challenges/FOURTOPS/data/train/X_val.csv",
    "Y_val":   "./challenges/FOURTOPS/data/train/Y_val.csv"
    },

    problem_description = r"""** Problem Description **
A major task in particle physics is the measurement of rare signal processes with very small cross-sections. With the unprecedented amount of data provided by the upcoming runs of the Large Hadron Collider (LHC), one can start to measure these processes. An example is the recent observation of four top quarks originating from a single proton-proton collision event. Accurate classification of these events is crucial, as even a small reduction in background noise on the order of a few tens of percent while maintaining the same signal detection efficiency can lead to a profound increase in sensitivity.
""",

    dataset_description = r"""** Dataset Description **
The dataset used for this problem consists of simulated proton-proton collision at a center of mass energy of 13 TeV. The signal process is defined as $pp \rightarrow t \bar{t} t \bar{t}$. The relevant production processes of the backgrounds are $t \bar{t} + X$ where $X = Z, W^+, W^+W^-$. 

The dataset includes 302072 events, of which roughly 50% is signal and 50% are background processes. All background processes have an equal number of events. Maximum objects encoded per event is 18 and there is no specific order. The contents of the datasets (X_train & X_val) are given below. 

IMPORTANT: The specific line format of the dataset is as follows:

$$E_{T}^{miss}, \phi_{E_t}^{miss}, obj_1, E_1, p_{T1}, \eta_1, \phi_1, obj_2, E_2, p_{t2}, \eta_2, \phi_2, ...$$

Such that each object is represented by five consecutive numerical features. The first is an integer identifier "$obj_n$" representing a particular object in the event. The object identifier is followed by its kinematic properties in the form of a four-vector containing the full energy "E" and the transverse momentum "p_T" in units of MeV, as well as the pseudo-rapidity "$\eta$" and the azimuthal angle "$\phi$". The other two quantities are "$E_{T}^{miss}$", the magnitude of the missing transverse energy in units of MeV and "$\phi_{E_T}^{miss}$" is the azimuthal angle of the missing transverse energy.

Since the length of the original events is variable, the data is zero-padded to the largest number of objects found in the events within the entire dataset. The dataset is fairly sparse and not pre-processed.

The relevant datasets are pytorch tensors with the following properties:

Name: X_train, shape: [241657, 92], dtype: torch.float32, 
Name: Y_train, shape: [241657], dtype: torch.int64, 
Name: X_val, shape: [30272, 92], dtype: torch.float32, 
Name: Y_val, shape: [30272], dtype: torch.int64

IMPORTANT: You do not need to read files, the harness provides X_train, Y_train, X_val, Y_val as PyTorch tensors.
""",

    evaluation_metric = r"""** Evaluation Metric **
The evaluation metric for this classification task is the area under the curve (AUC), specified by the area under the receiver operating characteristic (ROC) curve. The AUC summarizes a model's ability to distinguish between positive and negative classes. The higher the score the better.
""",

    prefix = r"""
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts, to_python
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops, dryrun_finite_check_fourtops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
                        
DATASET = {dataset_dict}
                       
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
""",

    code_template = r"""** Code Template **
# -------------------------- START OF LLM BLOCK ------------------------------
# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
# <LLM: Import modules>

#  -------- (OPTIONAL) CUSTOM DATASET  --------
# class CustomDataset(Dataset):
#  REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:CustomDataset"
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

    # <LLM: Write code to preprocess the data> 

    def __init__(self):
        # <LLM: Define and initialize any stateful components here>
        pass

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this. Configure as you please.
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",     # or torch_geometric.loader:DataLoader
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed.
            "collate": None,

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False, 
                                "batch_size": 512} # Or whatever you want
        }

    def fit(self, X, y=None):
        # <LLM: Extract statistics for transform>
        return self

    def transform(self, X):
        # <LLM: Apply pre-processing logic>
        return X # must return an indexable, picklable object

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
# MODEL I/O BATCH CONTRACT (CHOOSE ONE LANE)
# You MUST choose exactly one of the two supported input lanes and keep it consistent:
#
# --- LANE A: Torch dense batch (default) ---
# Loader:
#   - loader_class: "torch.utils.data:DataLoader"
#   - collate: None
# Batch from DataLoader:
#   (Xb, yb) where
#     Xb: FloatTensor[B, F]
#     yb: LongTensor[B] (or [B,1])
# Model forward:
#   out = model(Xb)
#   out must be FloatTensor[B] or FloatTensor[B,1] (logits or probabilities)
#
# --- LANE B: PyTorch Geometric (PyG) graphs ---
# Loader:
#   - loader_class: "torch_geometric.loader:DataLoader"
#   - collate: None
# Dataset samples MUST be torch_geometric.data.Data with at least:
#   data.x : FloatTensor[N_i, F]
#   data.edge_index : LongTensor[2, E_i]   (or equivalent; your model can build edges too)
#   data.y : LongTensor[1]                (GRAPH-LEVEL label for the event!)
# Batch from DataLoader:
#   G : torch_geometric.data.Batch (has G.x, G.edge_index, G.batch, and G.y)
# Model forward:
#   out = model(G)
#   out must be FloatTensor[num_graphs] or FloatTensor[num_graphs,1] (logits or probabilities)
#
# Any other batch shapes are NOT supported.

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # <LLM: Define and initialize any stateful components here>

    # <LLM: optionally build extra layers here>

    def forward(self, batch_x):
        # IMPORTANT output must be logits/probabilities per event
        # <LLM: Define your model's forward pass here>
        pass

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 10   # <LLM: adjust if you wish>
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS
    #   - Must return: trained_model, train_loss, val_loss, train_acc, val_acc
    #   - Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    
    # <LLM: Write code to define training loop, use the code above>
    # <LLM: Implement early stopping if possible>
    return trained_model, train_loss, val_loss, train_acc, val_acc

# DO NOT execute the pipeline here – the harness will do that.
# <end code template>
# ---------------------------  END OF LLM-CODE BLOCK  ---------------------------
""",

    suffix = r"""
# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    X_fit, Y_fit = X_train, Y_train
    if dryrun:
        idx = torch.randperm(X_train.shape[0])[:400]
        X_train, Y_train = X_train[idx], Y_train[idx]
        idx = torch.randperm(X_val.shape[0])[:200]
        X_val, Y_val = X_val[idx], Y_val[idx]
    pre = make_preprocessor().fit(X_fit, Y_fit)
    
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
    n_epochs = 10 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # Dry-run safety check
    if dryrun:
        try:
            dryrun_finite_check_fourtops(trained_model, spec, val_loader, device, batches=10)
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
        summary = to_python(summary)
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 
""",

    questions = [
        Question("Q1", r""" ** IMPORTANT: Your Challenge **
Write Python code for a binary classification model focusing on maximising the AUC using the code template above. You may freely choose any pre-processing methods and techniques as well as model architecture and training conventions. Do absolutely everything in your power to achieve the highest possible AUC.                 
""")
,

       Question("Q2", r"""** IMPORTANT: Your Challenge **
Write Python code for a binary classification model focusing on maximising the AUC using the code template above. You may freely choose any pre-processing methods and techniques as well as model architecture and training conventions.
                
You may optionally leverage the following particle-physics insights (strongly recommended if possible):

Pairwise Particle Features: It has been shown that explicitly computing pairwise particle features, particularly the invariant mass $m_{ij} = \sqrt{(p_i + p_j)^2} = \sqrt{(E_i + E_j)^2 - \abs((p_i^2 + p_j^2))}$ and the angular distance $\delta R_{ij} = \sqrt{(\eta_i - \eta_j)^2 + (\phi_i - \phi_j)^2}$ can significantly enhance the discriminative power of your model.
                
Model Architecture: It has been shown that Transformer models and Graph Neural Networks are particularly well-suited for this task.

Do absolutely everything in your power to achieve the highest possible AUC.
""")
]
)