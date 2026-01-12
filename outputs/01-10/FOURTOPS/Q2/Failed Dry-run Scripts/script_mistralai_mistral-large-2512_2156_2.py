
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
from sklearn.preprocessing import RobustScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.max_objects = 18
        self.obj_slice = 5
        self.global_features = 2

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

            "eval_overrides": {"shuffle": False,
                                "batch_size": 512}
        }

    def fit(self, X, y=None):
        # Extract global features (E_T_miss, phi_Et_miss)
        global_feats = X[:, :self.global_features].numpy()
        self.scaler.fit(global_feats)
        return self

    def transform(self, X):
        # Convert to numpy for sklearn
        X_np = X.numpy() if torch.is_tensor(X) else X

        # Scale global features
        global_feats = X_np[:, :self.global_features]
        global_feats_scaled = self.scaler.transform(global_feats)

        # Process object features
        obj_features = []
        for i in range(self.max_objects):
            start_idx = self.global_features + i * self.obj_slice
            end_idx = start_idx + self.obj_slice
            obj_slice = X_np[:, start_idx:end_idx]

            # Extract kinematic features (E, pT, eta, phi)
            kinematics = obj_slice[:, 1:5]  # Skip obj_id

            # Create pairwise features for each object pair
            if i > 0:
                prev_kinematics = obj_features[-1][:, 1:5]  # Get previous object's kinematics
                # Calculate delta R and invariant mass for each pair
                delta_eta = kinematics[:, 2:3] - prev_kinematics[:, 2:3]  # [N, 1]
                delta_phi = kinematics[:, 3:4] - prev_kinematics[:, 3:4]  # [N, 1]
                delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
                delta_R = np.sqrt(delta_eta**2 + delta_phi**2)  # [N, 1]

                # Invariant mass calculation
                E_i = kinematics[:, 0:1]  # [N, 1]
                E_j = prev_kinematics[:, 0:1]  # [N, 1]
                px_i = kinematics[:, 1:2] * np.cos(kinematics[:, 3:4])  # [N, 1]
                py_i = kinematics[:, 1:2] * np.sin(kinematics[:, 3:4])  # [N, 1]
                pz_i = kinematics[:, 1:2] * np.sinh(kinematics[:, 2:3])  # [N, 1]
                px_j = prev_kinematics[:, 1:2] * np.cos(prev_kinematics[:, 3:4])  # [N, 1]
                py_j = prev_kinematics[:, 1:2] * np.sin(prev_kinematics[:, 3:4])  # [N, 1]
                pz_j = prev_kinematics[:, 1:2] * np.sinh(prev_kinematics[:, 2:3])  # [N, 1]

                inv_mass_sq = (E_i + E_j)**2 - ((px_i + px_j)**2 + (py_i + py_j)**2 + (pz_i + pz_j)**2)
                inv_mass_sq = np.clip(inv_mass_sq, 0, None)  # Ensure non-negative
                inv_mass = np.sqrt(inv_mass_sq)  # [N, 1]

                # Add pairwise features to current object
                pairwise_feats = np.concatenate([delta_R, inv_mass], axis=1)  # [N, 2]
                obj_slice = np.concatenate([obj_slice, pairwise_feats], axis=1)

            obj_features.append(obj_slice)

        # Stack all object features
        obj_features = np.concatenate(obj_features, axis=1)  # [N, 18*(5+2)] = [N, 126]

        # Combine global and object features
        processed_X = np.concatenate([global_feats_scaled, obj_features], axis=1)

        return torch.from_numpy(processed_X).float()

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # Calculate input features
        input_size = sample_object.shape[1]

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )

        # Object embedding
        self.obj_embedding = nn.Linear(7, 128)  # 5 original + 2 pairwise features

        # Global features processing
        self.global_processor = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, 128)
        )

        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(128 * 19, 512),  # 18 objects + 1 global = 19 tokens
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, batch_x):
        # batch_x shape: [B, F] where F = 2 + 18*7 = 128

        # Split global and object features
        global_feats = batch_x[:, :2]  # [B, 2]
        obj_feats = batch_x[:, 2:]  # [B, 18*7]

        # Reshape object features to [B, 18, 7]
        obj_feats = obj_feats.view(-1, 18, 7)  # [B, 18, 7]

        # Embed objects
        obj_emb = self.obj_embedding(obj_feats)  # [B, 18, 128]

        # Process global features
        global_emb = self.global_processor(global_feats)  # [B, 128]
        global_emb = global_emb.unsqueeze(1)  # [B, 1, 128]

        # Combine all tokens
        all_tokens = torch.cat([global_emb, obj_emb], dim=1)  # [B, 19, 128]

        # Transformer encoding
        transformer_out = self.transformer_encoder(all_tokens)  # [B, 19, 128]

        # Flatten and classify
        flat_out = transformer_out.flatten(1)  # [B, 19*128]
        logits = self.classifier(flat_out)  # [B, 1]

        return logits.squeeze(-1)  # [B]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=False)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    best_model = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_targets = []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds.float() == batch_y).sum().item()
            train_total += batch_y.size(0)

            all_train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            all_train_targets.extend(batch_y.detach().cpu().numpy())

        train_loss /= train_total
        train_acc = train_correct / train_total
        train_auc = roc_auc_score(all_train_targets, all_train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.float())

                val_loss += loss.item() * batch_x.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds.float() == batch_y).sum().item()
                val_total += batch_y.size(0)

                all_val_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                all_val_targets.extend(batch_y.detach().cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(all_val_targets, all_val_preds)

        # Update scheduler
        scheduler.step(val_auc)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping if no improvement for 5 epochs
        if epoch > 10 and val_auc <= max(val_accs[-5:]):
            print("Early stopping triggered")
            break

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

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

