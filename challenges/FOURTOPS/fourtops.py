# ./challenges/FOURTOPS/fourtops.py

from challenges.challenges import Challenge, Question

fourtop_challenge = Challenge(
    name = "FOURTOPS",

    dataset = {
    "X_train": "./challenges/FOURTOPS/data/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/Y_train.csv",
    "X_val":   "./challenges/FOURTOPS/data/X_val.csv",
    "Y_val":   "./challenges/FOURTOPS/data/Y_val.csv"
    },

    problem_description = r"""** Problem Description **
A major task in particle physics is the measurement of rare signal 
processes with very small cross-sections. With the unprecedented amount of 
data provided by the upcoming runs of the Large Hadron Collider (LHC), 
one can start to measure these processes. An example is the recent 
observation of four top quarks originating from a single proton-proton 
collision event. Accurate classification of these events is crucial, 
as even a small reduction in background noise on the order of a few tens 
of percent while maintaining the same signal detection efficiency can lead 
to a profound increase in sensitivity.
""",

    dataset_description = r"""** Dataset Description **
The dataset used for this problem consists of simulated proton-proton 
collision at a center of mass energy of 13 TeV. The signal process is defined as 
$pp \rightarrow t \bar{t} t \bar{t}$. The relevant production processes of the
backgrounds are $t \bar{t} + X$ where $X = Z, W^+, W^+W^-$.

The dataset includes 302072 events, of which roughly 50\% is signal and 50\%
are background processes. All background processes have an equal number of events. 
There is no cut on the maximum number of objects and there is no order.    

The contents of the datasets (X_train \& X_val) are given below.
IMPORTANT: The specific line format of the dataset is as follows:

$$E_{T}^{miss}, \phi_{E_t}^{miss}, obj_1, E_1, p_{T1}, \eta_1, \phi_1, obj_2, E_2, p_{t2}, \eta_2, \phi_2, ...$$

Such that each object is represented by a string that starts with an identifier "$obj_n$", which is an
integer value representing a particular object in the event. The object identifier is
followed by its kinematic properties in the form of a four-vector containing the full 
energy "E" and the transverse momentum "p\_T" in units of MeV, as well as the pseudo-rapidity 
"$\eta$" and the azimuthal angle "$\phi$". The other three quantities are "weight" given by the cross-section
of the process divided by the total number of events generated. "$E_{T}^{miss}$" is the magnitude of the
missing transverse energy in units of MeV and "$\phi_{E_T}^{miss}$" is the azimuthal angle of the missing
transverse energy.

Since the length of the events is variable, the data is zero-padded to the largest number of objects
found in the events within the entire dataset. The dataset is fairly sparse and not pre-processed.

The relevant datasets are pytorch tensors with the following properties:

Name: X_train, shape: [241657, 92], dtype: torch.float32, 
Name: Y_train, shape: [241657], dtype: torch.int64, 
Name: X_val, shape: [30272, 92], dtype: torch.float32, 
Name: Y_val, shape: [30272], dtype: torch.int64

IMPORTANT: Each of these tensors are pre-loaded and available.
""",

    evaluation_metric = r"""** Evaluation Metric **
The evaluation metric for this classification task is the area under the curve (AUC),
specified by the area under the receiver operating characteristic (ROC) curve. 
The AUC summarizes a model's ability to distinguish between positive and 
negative classes. The higher the score the better.
""",

    prefix = r"""
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: Python 3.12, PyTorch 2.6.0, Torch_Geometric 2.6.1, NumPy 2.2.3, SciPy v1.15.2, SciKit-Learn 1.6.1
import os, sys, pickle, torch, torch_geometric, gc, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

DATASET = {dataset_dict}
                       
def load_data():
    X_train = pd.read_csv('./challenges/FOURTOPS/data/X_train.csv',
                          dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv('./challenges/FOURTOPS/data/Y_train.csv',
                          dtype=np.int64 ).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv('./challenges/FOURTOPS/data/X_val.csv',
                          dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv('./challenges/FOURTOPS/data/Y_val.csv',
                          dtype=np.int64 ).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train),
            torch.from_numpy(Y_train),
            torch.from_numpy(X_val),
            torch.from_numpy(Y_val))

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        if isinstance(self.x, (tuple, list)):
            return (tuple(t[idx] for t in self.x), self.y[idx])
        else:
            return (self.x[idx], self.y[idx])      

def make_loaders(X_train, Y_train, X_val, Y_val, batch=512):
    train_ds = PairDataset(X_train, Y_train)
    val_ds   = PairDataset(X_val , Y_val)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0))

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ----------------                        
# -------------------------- START OF LLM BLOCK ------------------------------
""",

    code_template = r"""** Code Template **
# <start code template>
# 0. ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch, scipy, sklearn (sub-)modules you actually use.
# <LLM: Import modules>

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    #    Must implement:
    #   - fit(...) -> self
    #   - transform(X: Tensor)   -> Tensor  **or**  Tuple[Tensor, Tensor]

    # REQUIREMENTS
    # IMPORTANT: All state must be picklable with the std-lib pickle module.
    # IMPORTANT: Batch first.
    # May allocate NumPy arrays or Torch tensors internally, but:
    # transform() must be deterministic.
    # Store only derived parameters needed for transform i.e. do not store the raw data
    # itself in the preprocessor object.

    # DATA SPECIFICS
    # IMPORTANT: X_train, Y_train, X_val, Y_val are provided as PyTorch tensors in the environment.
    # Total flat length per event (X_train & X_val): 92
    # Index  0 :  missing-ET magnitude  (E_T_miss)
    # Index  1 :  missing-ET azimuth    (phi_Et_miss)
    # Indices  2-6  : object 1  ->  obj_1, E_1, p_T1, eta_1, phi_1
    # Indices  7-11 : object 2  -> obj_2, E_2 , p_T_2 , eta_2 , phi_2
    # ...
    # Indices 88-92 : object 18 -> obj_18, E_18 , p_T_18 , eta_18 , phi_18
    # Per-object slice size = 5
    # Max objects encoded   = 18

    # TIPS
    # When modifying data features or feature engineering: annotate tensor size as comments after 
    # each tensor operation to reduce dimension mismatches.

    # <LLM: Write code to preprocess the data>    
    def __init__(self):
        # <LLM: Define and initialize any stateful components here>
        pass

    def fit(self, X, y=None):
        # <LLM: Extract statistics or fit transformers>
        return self

    def transform(self, X):
        # # Example output options:
        # - return X_new                        # (N, features)
        # - return (X_seq, mask)                # (N,L,F), (N,L)
        # <LLM: Apply preprocessing logic, return torch.Tensor>
        return X 

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    # A UNIVERSAL wrapper.  You *must* keep forward(self, data, mask=None)
    # so it works for BOTH loader paths:
    # (data, label)              -> forward(data)
    # ((data, mask), label)      -> forward(data, mask)

    # input_shape examples
    #   Flat       : (F,)               -> MLP or 1-D CNN
    #   Sequence   : (L, F)             -> Transformer / RNN / DeepSets
    #   2-D image  : (C, H, W)          -> 2-D CNN   (requires reshape upstream!)
    #   Node set   : (N, F_node)         -> GNN / Set Transformer
    
    def __init__(self, input_shape: tuple[int, ...], *, use_mask: bool):
        super().__init__()
        self.use_mask = use_mask
        # <LLM: Define and initialize any stateful components here>

    # <LLM: optionally build extra layers here>

    def forward(self, data: torch.Tensor, mask: torch.Tensor | None = None):
        # data : Tensor
        # Flat  -> (B, F)
        # Seq   -> (B, L, F)
        # 2-D   -> (B, C, H, W)
        # mask : None or BoolTensor (B, L)

        # <LLM: Define your model's forward pass here>

def make_model(input_shape, *, use_mask=False):
    return BinaryClassifier(input_shape, use_mask=use_mask)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 10   # <LLM: adjust if you wish>
def train_model(model, train_loader, val_loader, epochs):
    # PARAMETERS
    # model : torch.nn.Module   
    # train_loader / val_loader yield either
    #   (data,  label)            # single tensor
    #   ((data, mask), label)     # tensor + padding mask
    #   Forward signature must match the chosen format.
    #   epochs: int

    # RETURNS
    # trained_model : nn.Module          (same instance, trained in-place)
    # train_loss    : list[float]        (length == epochs)
    # val_loss      : list[float]
    # train_acc     : list[float]
    # val_acc       : list[float]
    
    # REQUIREMENTS 
    # Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).
    # Implement early-stopping.
    # train_loader / val_loader yield either (data,label) or ((data, mask), label)
    # Forward signature must match.

    # <LLM: Write code to define training loop>
    # <LLM: Implement early stopping if possible>
    return trained_model, train_loss, val_loss, train_acc, val_acc

# IMPORTANT: DO NOT execute the pipeline here – the harness will do that.
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
    X_train, Y_train, X_val, Y_val = load_data()
    if dryrun:
        X_train, Y_train, X_val, Y_val = X_train[:200], Y_train[:200], X_val[:20], Y_val[:20]
    pre = make_preprocessor().fit(X_train, Y_train)
    X_train = pre.transform(X_train)                    # may be Tensor or Tuple
    X_val   = pre.transform(X_val)
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val)

    # 2. Build model
    if isinstance(X_train, torch.Tensor):               # single-tensor case
        temp_ref    = X_train
        input_shape = temp_ref.shape[1:]                # e.g. (F,)
        use_mask    = False
    else:                                               # tuple => (data, mask)
        temp_ref    = X_train
        input_shape = temp_ref[0].shape[1:]             # e.g. (L, F)
        use_mask    = True                              
    model = make_model(input_shape, use_mask=use_mask)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* - run a single toy forward pass
    if dryrun:
        toy_data = torch.zeros(8, *input_shape, dtype=torch.float32)
        if use_mask:
            toy_mask = torch.zeros(8, input_shape[0], dtype=torch.bool)
            toy_batch = (toy_data, toy_mask)
        else:
            toy_batch = toy_data

        toy_transformed = pre.transform(toy_batch)
        try:
            _ = trained_model(*toy_transformed) if isinstance(toy_transformed, (tuple, list)) \
                else trained_model(toy_transformed)
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
Write Python code for a binary classification model focussing on maximising the AUC using the code template above. 
You may freely choose any pre-processing methods and techniques as well as model architecture and training conventions.
Play it safe and implement a simple model you feel confident in.
"""),

        Question("Q2", r""" ** IMPORTANT: Your Challenge **
Write Python code for a binary classification model focussing on maximising the AUC using the code template above. 
You may freely choose any pre-processing methods and techniques as well as model architecture and training conventions.
Do absolutely everything in your power to get the highest score on the AUC.                 
"""),

       Question("Q3", r"""** IMPORTANT: Your Challenge **
Write Python code for a binary classification model focussing on maximising the AUC using the code template above. 
You may freely choose any pre-processing methods and techniques as well as model architecture and training conventions.
You may optionally leverage the following particle-physics insights (strongly recommended if applicable):

Pairwise Particle Features: It has been shown that explicitly computing pairwise particle features, particularly 
the invariant mass $m_{ij}$ and the angular distance $\delta R_{ij} = \sqrt((\eta_i - \eta_j)^2 + (\phi_i - \phi_j))$
​can significantly enhance the discriminative power of your model.

Do absolutely everything in your power to get the highest score on the AUC.
""")
]
)