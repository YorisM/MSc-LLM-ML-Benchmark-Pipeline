
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import normalise_batch, assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts

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
        self.X = pre.transform(X) if pre is not None else X
        self.y = y
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ----------------                        
# -------------------------- START OF LLM BLOCK ------------------------------

{'error': {'message': 'Network connection lost.', 'code': 502, 'metadata': {'provider_name': 'SiliconFlow'}}, 'logprobs': None, 'finish_reason': None, 'native_finish_reason': None, 'index': 0, 'message': {'role': 'assistant', 'content': '', 'refusal': None, 'reasoning': "We are given a dataset of particle physics events, each event is a fixed-length vector of 92 features.\n The first two features are global event features: missing ET magnitude and its azimuthal angle.\n The remaining 90 features represent up to 18 particles, each particle described by 5 features:\n   obj_id (integer type), E, pT, eta, phi.\n\n We are to build a binary classifier to distinguish between signal (four top quarks) and background.\n\n We are allowed to do any pre-processing and use any model architecture, but we must output code that runs.\n\n The evaluation metric is AUC, so we should aim for a model that outputs a probability and then compute the AUC.\n\n Since the dataset is large (241657 training events) and we have up to 18 particles per event, we can consider:\n\n 1. Pre-processing:\n    - The object identifier (obj_id) is categorical. We can one-hot encode it, but note that it's an integer and we don't know the total number of categories.\n    - Alternatively, we can treat the object identifier as a categorical feature and use an embedding layer, but note that the input is fixed-length and we have at most 18 objects.\n\n    However, note that the object identifier might have a physical meaning (like jet, lepton, etc.). We are not given the mapping, so we have to treat it as a categorical integer.\n\n    Alternatively, we can ignore the object identifier and only use the kinematic features? But the problem statement says that the object identifier is important.\n\n    We can also create new features, such as:\n      - For each particle, we can compute the invariant mass of pairs (if we have multiple particles) and angular distances.\n\n    Since the problem suggests that pairwise particle features (invariant mass and delta", 'reasoning_details': [{'format': 'unknown', 'index': 0, 'type': 'reasoning.text', 'text': "We are given a dataset of particle physics events, each event is a fixed-length vector of 92 features.\n The first two features are global event features: missing ET magnitude and its azimuthal angle.\n The remaining 90 features represent up to 18 particles, each particle described by 5 features:\n   obj_id (integer type), E, pT, eta, phi.\n\n We are to build a binary classifier to distinguish between signal (four top quarks) and background.\n\n We are allowed to do any pre-processing and use any model architecture, but we must output code that runs.\n\n The evaluation metric is AUC, so we should aim for a model that outputs a probability and then compute the AUC.\n\n Since the dataset is large (241657 training events) and we have up to 18 particles per event, we can consider:\n\n 1. Pre-processing:\n    - The object identifier (obj_id) is categorical. We can one-hot encode it, but note that it's an integer and we don't know the total number of categories.\n    - Alternatively, we can treat the object identifier as a categorical feature and use an embedding layer, but note that the input is fixed-length and we have at most 18 objects.\n\n    However, note that the object identifier might have a physical meaning (like jet, lepton, etc.). We are not given the mapping, so we have to treat it as a categorical integer.\n\n    Alternatively, we can ignore the object identifier and only use the kinematic features? But the problem statement says that the object identifier is important.\n\n    We can also create new features, such as:\n      - For each particle, we can compute the invariant mass of pairs (if we have multiple particles) and angular distances.\n\n    Since the problem suggests that pairwise particle features (invariant mass and delta"}]}}

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

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
                view = normalise_batch(first_batch, device=device)
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


