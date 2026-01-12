
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

# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler

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

    def __init__(self):
        self.scaler = StandardScaler()

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
        # Fit scaler on features (exclude obj_id which is integer)
        # X: [N, 92], globals at 0-1, then 18*5
        # For each object, obj_id is integer, but E,pT,eta,phi float
        globals_data = X[:, :2].numpy()  # [N,2]
        objects_data = []
        for i in range(18):
            start = 2 + i*5
            obj_features = X[:, start+1:start+5].numpy()  # [N,4] for E,pT,eta,phi
            objects_data.append(obj_features)
        all_features = np.concatenate([globals_data] + objects_data, axis=1)  # [N, 2 + 18*4]
        self.scaler.fit(all_features)
        return self

    def transform(self, X):
        # Apply scaling to the same features
        globals_data = X[:, :2].numpy()  # [N,2]
        objects_data = []
        obj_ids = []
        for i in range(18):
            start = 2 + i*5
            obj_ids.append(X[:, start:start+1].numpy())  # [N,1]
            obj_features = X[:, start+1:start+5].numpy()  # [N,4]
            objects_data.append(obj_features)
        all_features = np.concatenate([globals_data] + objects_data, axis=1)  # [N, 2 + 18*4]
        scaled_features = self.scaler.transform(all_features)
        # Reconstruct X with scaled features but keep obj_ids unchanged
        X_transformed = np.zeros_like(X.numpy())
        X_transformed[:, :2] = scaled_features[:, :2]  # globals
        idx = 2
        for i in range(18):
            X_transformed[:, idx] = obj_ids[i].ravel()  # obj_id
            X_transformed[:, idx+1:idx+5] = scaled_features[:, 2 + i*4 : 2 + (i+1)*4]  # 4 features
            idx += 5
        return torch.from_numpy(X_transformed)

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # sample_object is from make_view_by_lane, for dense it's torch tensor [B, F]
        # Assume dense lane
        self.embed_obj = nn.Embedding(10, 8)  # assume obj_id <=10, embed to 8 dim
        self.rnn = nn.GRU(4 + 8, 32, batch_first=True, num_layers=2, dropout=0.1)
        self.global_fc = nn.Linear(2, 16)
        self.pool_fc = nn.Linear(32, 16)
        self.final_fc = nn.Linear(16 + 16, 1)  # logits for binary

    def forward(self, x):
        # x: [B, 92]
        B = x.size(0)
        globals = x[:, :2]  # [B,2]
        objects = x[:, 2:].view(B, 18, 5)  # [B,18,5]
        obj_ids = objects[:, :, 0].long()  # [B,18]
        kin = objects[:, :, 1:]  # [B,18,4]
        emb = self.embed_obj(obj_ids)  # [B,18,8]
        obj_features = torch.cat([emb, kin], dim=-1)  # [B,18,12]
        # Mask invalid objects
        mask = (obj_ids != 0).unsqueeze(-1)  # [B,18,1]
        obj_features = obj_features * mask.float()  # zero invalid
        # RNN
        out, _ = self.rnn(obj_features)  # [B,18,32]
        # Pool: mean over valid objects
        mask_flat = mask.squeeze(-1)  # [B,18]
        seq_length = mask_flat.sum(dim=1, keepdim=True)  # [B,1]
        pooled = (out * mask_flat.unsqueeze(-1)).sum(dim=1) / (seq_length + 1e-8)  # [B,32]
        g_feat = F.relu(self.global_fc(globals))  # [B,16]
        p_feat = F.relu(self.pool_fc(pooled))  # [B,16]
        concat = torch.cat([g_feat, p_feat], dim=1)  # [B,32]
        logits = self.final_fc(concat).squeeze(1)  # [B]
        return logits  # for BCEWithLogitsLoss

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20   # <LLM: adjust if you wish>
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS
    #   - Must return: trained_model, train_loss, val_loss, train_acc, val_acc
    #   - Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()
    # Early stopping
    best_auc = -np.inf
    best_model_state = None
    patience = 5
    no_improve = 0

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch in range(epochs):
        model.train()
        tr_loss = 0
        tr_correct = 0
        tr_total = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb.float())
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            preds = (torch.sigmoid(out) > 0.5).int()
            tr_correct += (preds == yb).sum().item()
            tr_total += yb.size(0)
        tr_loss /= len(train_loader)
        tr_acc = tr_correct / tr_total
        train_losses.append(tr_loss)
        train_accs.append(tr_acc)

        model.eval()
        va_loss = 0
        va_correct = 0
        va_total = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                loss = criterion(out, yb.float())
                va_loss += loss.item()
                preds = (torch.sigmoid(out) > 0.5).int()
                va_correct += (preds == yb).sum().item()
                va_total += yb.size(0)
        va_loss /= len(val_loader)
        va_acc = va_correct / va_total
        val_losses.append(va_loss)
        val_accs.append(va_acc)

        # Check AUC for early stopping (though harness will compute it)
        # For simplicity, use acc as proxy
        if va_acc > best_auc:
            best_auc = va_acc
            best_model_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
        scheduler.step()

    # Load best model
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, train_accs, val_accs

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

