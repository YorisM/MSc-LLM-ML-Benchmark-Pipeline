
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

# ----------- PREPROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.met_log_mean = 0.0
        self.met_log_std = 1.0
        self.energy_log_mean = 0.0
        self.energy_log_std = 1.0
        self.pt_log_mean = 0.0
        self.pt_log_std = 1.0

    def make_loader_cfg(self) -> dict:
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
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def fit(self, X, y=None):
        import numpy as np
        X_np = X.numpy() if torch.is_tensor(X) else X

        met_mag = X_np[:, 0]
        met_mag_valid = met_mag[met_mag > 1]
        if len(met_mag_valid) > 0:
            met_mag_log = np.log(met_mag_valid)
            self.met_log_mean = float(np.mean(met_mag_log))
            self.met_log_std = float(np.std(met_mag_log) + 1e-6)

        obj_features = X_np[:, 2:].reshape(-1, 18, 5)
        mask = obj_features[:, :, 0] != 0

        energy = obj_features[:, :, 1]
        energy_valid = energy[mask]
        energy_valid = energy_valid[energy_valid > 1]
        if len(energy_valid) > 0:
            energy_log = np.log(energy_valid)
            self.energy_log_mean = float(np.mean(energy_log))
            self.energy_log_std = float(np.std(energy_log) + 1e-6)

        pt = obj_features[:, :, 2]
        pt_valid = pt[mask]
        pt_valid = pt_valid[pt_valid > 1]
        if len(pt_valid) > 0:
            pt_log = np.log(pt_valid)
            self.pt_log_mean = float(np.mean(pt_log))
            self.pt_log_std = float(np.std(pt_log) + 1e-6)

        return self

    def transform(self, X):
        X_t = torch.as_tensor(X) if not torch.is_tensor(X) else X
        X_t = X_t.float()

        N = X_t.shape[0]
        output = torch.zeros_like(X_t)

        met_mag = X_t[:, 0]
        met_phi = X_t[:, 1]
        met_mag_log = torch.where(met_mag > 1, torch.log(met_mag), torch.zeros_like(met_mag))
        met_mag_norm = (met_mag_log - self.met_log_mean) / self.met_log_std
        output[:, 0] = met_mag_norm
        output[:, 1] = met_phi

        obj_features = X_t[:, 2:].reshape(N, 18, 5)
        output_obj = output[:, 2:].reshape(N, 18, 5)

        mask = obj_features[:, :, 0] != 0

        output_obj[:, :, 0] = obj_features[:, :, 0]

        energy = obj_features[:, :, 1]
        energy_log = torch.where(energy > 1, torch.log(energy), torch.zeros_like(energy))
        energy_norm = torch.where(mask, (energy_log - self.energy_log_mean) / self.energy_log_std, torch.zeros_like(energy))
        output_obj[:, :, 1] = energy_norm

        pt = obj_features[:, :, 2]
        pt_log = torch.where(pt > 1, torch.log(pt), torch.zeros_like(pt))
        pt_norm = torch.where(mask, (pt_log - self.pt_log_mean) / self.pt_log_std, torch.zeros_like(pt))
        output_obj[:, :, 2] = pt_norm

        output_obj[:, :, 3] = obj_features[:, :, 3]
        output_obj[:, :, 4] = obj_features[:, :, 4]

        output[:, 2:] = output_obj.reshape(N, 90)

        return output

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        input_dim = 4
        hidden_dim = 128
        n_heads = 4
        n_layers = 4

        self.obj_type_embed = nn.Embedding(21, 32, padding_idx=0)

        self.met_embed = nn.Linear(2, hidden_dim)
        self.particle_embed = nn.Linear(input_dim + 32, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch_x):
        B = batch_x.shape[0]

        met = batch_x[:, :2]
        particles = batch_x[:, 2:].reshape(B, 18, 5)

        obj_id = particles[:, :, 0].long()
        obj_kinematics = particles[:, :, 1:]

        mask = obj_id == 0

        obj_type_emb = self.obj_type_embed(torch.clamp(obj_id, min=0, max=20))

        obj_features = torch.cat([obj_type_emb, obj_kinematics], dim=-1)

        met_emb = self.met_embed(met)
        particle_emb = self.particle_embed(obj_features)

        tokens = torch.cat([met_emb.unsqueeze(1), particle_emb], dim=1)
        full_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)

        encoded = self.transformer(tokens, src_key_padding_mask=full_mask)

        encoded_for_max = encoded.clone()
        encoded_for_max[full_mask] = -1e9
        max_pool = torch.max(encoded_for_max, dim=1)[0]

        encoded_for_mean = encoded.clone()
        encoded_for_mean[full_mask] = 0
        sum_pool = torch.sum(encoded_for_mean, dim=1)
        count = (~full_mask).sum(dim=1, keepdim=True).float()
        mean_pool = sum_pool / (count + 1e-6)

        combined = torch.cat([max_pool, mean_pool], dim=1)
        logits = self.classifier(combined).squeeze(-1)

        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 25

def train_model(model, train_loader, val_loader, epochs):
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 7

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_labels = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.shape[0]
            preds = (torch.sigmoid(logits) > 0.5).long()
            train_correct += (preds == batch_y.long()).sum().item()
            train_total += batch_x.shape[0]

            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        train_loss /= train_total
        train_acc = train_correct / train_total
        train_auc = roc_auc_score(train_labels, train_preds)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float()

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * batch_x.shape[0]
                preds = (torch.sigmoid(logits) > 0.5).long()
                val_correct += (preds == batch_y.long()).sum().item()
                val_total += batch_x.shape[0]

                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_labels, val_preds)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        scheduler.step(val_auc)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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

