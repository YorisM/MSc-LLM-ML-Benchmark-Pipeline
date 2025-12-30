
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

# ---------- IMPORTS ----------
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler_glob = StandardScaler()
        self.scaler_kin = StandardScaler()
        self.max_obj_id = 0

    def make_loader_cfg(self):
        return {
            "dataset_builder": "llm_script:FourTopsDataset",   
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",     
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            "collate": None,

            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Extract global and kinematics
        glob = X[:, :2]
        self.scaler_glob.fit(glob.numpy())

        objects = X[:, 2:].view(-1, 18, 5)  # [N, 18, 5]
        kin_flat = objects[:, :, 1:].reshape(-1, 4)  # [N*18, 4]
        self.scaler_kin.fit(kin_flat.numpy())

        self.max_obj_id = int(objects[:, :, 0].max().item()) + 1
        return self

    def transform(self, X):
        # Normalize globals
        glob_norm = torch.tensor(self.scaler_glob.transform(X[:, :2].numpy())).to(X.dtype).float()

        # Process objects
        objects = torch.zeros_like(X[:, 2:])
        obj_kin = X[:, 2:].view(-1, 18, 5)
        obj_ids = obj_kin[:, :, 0]  # [N, 18]
        kinematics = obj_kin[:, :, 1:]  # [N, 18, 4]

        kin_flat = kinematics.reshape(-1, 4)
        kin_norm = torch.tensor(self.scaler_kin.transform(kin_flat.numpy())).reshape(kinematics.shape).to(X.dtype).float()

        # Combine
        obj_new = torch.zeros_like(obj_kin)
        obj_new[:, :, 0] = obj_ids
        obj_new[:, :, 1:] = kin_norm

        objects = obj_new.view(-1, 18*5)

        # Final combined
        X_new = torch.cat([glob_norm, objects], dim=1)
        return X_new

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        objects = sample_object[:, 2:].view(-1, 18, 5)
        self.max_obj = int(objects[:, :, 0].max().item()) + 1
        self.embed_dim = 4
        self.d_model = 16
        self.num_heads = 4
        self.num_layers = 2

        self.obj_emb = nn.Embedding(self.max_obj, self.embed_dim)
        self.proj_kin = nn.Linear(4, self.d_model - self.embed_dim)
        self.pos_emb = nn.Embedding(18, self.d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=self.num_heads, dim_feedforward=32, batch_first=False
            ),
            num_layers=self.num_layers
        )
        self.glob_proj = nn.Linear(2, 8)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model + 8, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, batch_x):
        global_ = batch_x[:, :2]
        objects_flat = batch_x[:, 2:].view(batch_x.size(0), 18, 5)  # [batch, 18, 5]
        obj_ids = objects_flat[:, :, 0].long()
        kinematics = objects_flat[:, :, 1:]

        kin_proj = self.proj_kin(kinematics)  # [batch, 18, d_model - embed_dim]
        obj_emb = self.obj_emb(obj_ids)  # [batch, 18, embed_dim]
        combined = torch.cat([obj_emb, kin_proj], -1)  # [batch, 18, d_model]

        pos = self.pos_emb(torch.arange(18, device=batch_x.device).unsqueeze(0).expand(batch_x.size(0), -1))  # [batch, 18, d_model]
        combined += pos

        # Transformer input (seq, batch, dmodel)
        trans_input = combined.transpose(0, 1)  # [18, batch, d_model]
        trans_out = self.transformer(trans_input)  # [18, batch, d_model]

        # Aggregate
        seq_agg = trans_out.transpose(0, 1).mean(dim=1)  # [batch, d_model]
        glob_agg = self.glob_proj(global_)  # [batch, 8]
        out = self.mlp(torch.cat([seq_agg, glob_agg], -1))
        return out

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50  # Increased for potential better training
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = 0
    early_stop = 10
    epochs_no_improve = 0
    best_model_dict = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        total_train = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).squeeze()
            loss = loss_fn(pred, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += ((pred > 0) == y).sum().item()
            total_train += y.size(0)
        train_loss /= len(train_loader)
        train_acc = train_correct / total_train

        model.eval()
        val_loss = 0
        val_correct = 0
        total_val = 0
        val_preds = []
        val_ys = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).squeeze()
                loss = loss_fn(pred, y.float())
                val_loss += loss.item()
                val_correct += ((pred > 0) == y).sum().item()
                total_val += y.size(0)
                val_preds.extend(pred.sigmoid().cpu().numpy())
                val_ys.extend(y.cpu().numpy())
        val_loss /= len(val_loader)
        val_acc = val_correct / total_val
        auc = roc_auc_score(val_ys, val_preds)

        if auc > best_auc:
            best_auc = auc
            epochs_no_improve = 0
            best_model_dict = model.state_dict().copy()
        else:
            epochs_no_improve += 1

        if epochs_no_improve > early_stop:
            break

        scheduler.step()

    if best_model_dict:
        model.load_state_dict(best_model_dict)
    return model, train_loss, val_loss, train_acc, val_acc

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


