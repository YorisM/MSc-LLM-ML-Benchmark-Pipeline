
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
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
        x = self.X[idx]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x, self.y[idx]

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import math

class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.object_scaler = StandardScaler()
        self.global_scaler = StandardScaler()

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Reshape to extract object features
        batch_size = X.shape[0]
        num_objects = 18
        object_features = 5

        # Global features
        global_feats = X[:, :2].numpy()  # (N, 2)
        self.global_scaler.fit(global_feats)

        # Object features (excluding object ID for scaling)
        obj_data = X[:, 2:].reshape(batch_size, num_objects, object_features)  # (N, 18, 5)
        kinematics = obj_data[:, :, 1:].reshape(-1, 4)  # (N*18, 4): E, pT, eta, phi
        self.object_scaler.fit(kinematics)

        return self

    def transform(self, X):
        batch_size = X.shape[0]
        num_objects = 18
        object_features = 5

        # Split into global and object features
        global_feats = X[:, :2].numpy()  # (N, 2)
        obj_data = X[:, 2:].reshape(batch_size, num_objects, object_features)  # (N, 18, 5)

        # Scale global features
        global_feats_scaled = self.global_scaler.transform(global_feats)  # (N, 2)

        # Scale object kinematics
        kinematics = obj_data[:, :, 1:].reshape(-1, 4)  # (N*18, 4)
        kinematics_scaled = self.object_scaler.transform(kinematics)
        kinematics_scaled = kinematics_scaled.reshape(batch_size, num_objects, 4)  # (N, 18, 4)

        # Combine object ID with scaled kinematics
        object_ids = obj_data[:, :, 0:1]  # (N, 18, 1)
        obj_features = np.concatenate([object_ids, kinematics_scaled], axis=2)  # (N, 18, 5)

        # Compute pairwise features: invariant mass and deltaR
        # We'll compute for all object pairs, resulting in (N, 153, 2) features
        batch_pairwise_features = []

        for b in range(batch_size):
            pairwise_features = []
            for i in range(num_objects):
                for j in range(i+1, num_objects):
                    # Get kinematics for objects i and j
                    E1, pT1, eta1, phi1 = obj_features[b, i, 1:5]
                    E2, pT2, eta2, phi2 = obj_features[b, j, 1:5]

                    # Skip if both objects are padded (pT=0)
                    if pT1 == 0 and pT2 == 0:
                        inv_mass = 0.0
                        delta_r = 0.0
                    else:
                        # Compute invariant mass: m² = (E1+E2)² - (p1+p2)²
                        # First compute momentum components
                        px1 = pT1 * np.cos(phi1)
                        py1 = pT1 * np.sin(phi1)
                        pz1 = pT1 * np.sinh(eta1)

                        px2 = pT2 * np.cos(phi2)
                        py2 = pT2 * np.sin(phi2)
                        pz2 = pT2 * np.sinh(eta2)

                        # Total momentum
                        px_tot = px1 + px2
                        py_tot = py1 + py2
                        pz_tot = pz1 + pz2
                        E_tot = E1 + E2

                        # Invariant mass
                        inv_mass_sq = E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)
                        inv_mass = np.sqrt(max(inv_mass_sq, 0))

                        # Compute deltaR
                        delta_eta = eta1 - eta2
                        delta_phi = phi1 - phi2
                        # Normalize delta_phi to [-pi, pi]
                        delta_phi = np.mod(delta_phi + np.pi, 2*np.pi) - np.pi
                        delta_r = np.sqrt(delta_eta**2 + delta_phi**2)

                    pairwise_features.append([inv_mass, delta_r])

            batch_pairwise_features.append(pairwise_features)

        pairwise_features = np.array(batch_pairwise_features)  # (N, 153, 2)

        # Flatten object features
        obj_features_flat = obj_features.reshape(batch_size, -1)  # (N, 90)

        # Combine all features: global + object + pairwise
        pairwise_features_flat = pairwise_features.reshape(batch_size, -1)  # (N, 306)

        # Final feature vector: global (2) + object (90) + pairwise (306) = 398
        combined_features = np.concatenate([
            global_feats_scaled,
            obj_features_flat,
            pairwise_features_flat
        ], axis=1)  # (N, 398)

        return torch.tensor(combined_features, dtype=torch.float32)

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = sample_object.shape[1]

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )

        # Process input to transformer dimension
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_x):
        # batch_x shape: (batch_size, 398)

        # Project to transformer dimension
        x = self.input_projection(batch_x)  # (batch_size, 256)

        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # (batch_size, 1, 256)

        # Apply transformer
        x = self.transformer(x)  # (batch_size, 1, 256)
        x = x.squeeze(1)  # (batch_size, 256)

        # Attention pooling (even though we have single token, this adds flexibility)
        attn_weights = self.attention_pool(x)  # (batch_size, 1)
        context = x * attn_weights  # (batch_size, 256)

        # Classifier
        logits = self.classifier(context)  # (batch_size, 1)
        return logits.squeeze(1)  # (batch_size,)

def make_model(example_object):
    return BinaryClassifier(example_object)

EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=False
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb.float())
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == yb).sum().item()
            train_total += yb.size(0)

        train_acc = train_correct / train_total if train_total > 0 else 0.0
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                outputs = model(xb)
                loss = criterion(outputs, yb.float())

                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == yb).sum().item()
                val_total += yb.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc)

        # Update scheduler
        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

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

