
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

# ---------- IMPORTS ----------
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

# ----------- PREPROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.obj_mean = None
        self.obj_std = None

    def make_loader_cfg(self):
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X

        # Global features: MET magnitude and phi
        global_features = X_np[:, :2]
        self.global_mean = np.mean(global_features, axis=0)
        self.global_std = np.std(global_features, axis=0) + 1e-8

        # Object features: reshape to [N, 18, 5]
        obj_features = X_np[:, 2:].reshape(-1, 18, 5)
        obj_ids = obj_features[:, :, 0]
        valid_mask = obj_ids != 0

        # Kinematics: E, pT, eta, phi
        kinematics = obj_features[:, :, 1:]
        valid_kinematics = kinematics[valid_mask]

        if len(valid_kinematics) > 0:
            self.obj_mean = np.mean(valid_kinematics, axis=0)
            self.obj_std = np.std(valid_kinematics, axis=0) + 1e-8
        else:
            self.obj_mean = np.zeros(4)
            self.obj_std = np.ones(4)

        return self

    def transform(self, X):
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
            was_torch = True
        else:
            X_np = X.copy()
            was_torch = False

        # Normalize global features
        X_np[:, :2] = (X_np[:, :2] - self.global_mean) / self.global_std

        # Normalize object kinematics
        obj_features = X_np[:, 2:].reshape(-1, 18, 5)
        obj_ids = obj_features[:, :, 0]
        valid_mask = obj_ids != 0

        kinematics = obj_features[:, :, 1:]
        kinematics[valid_mask] = (kinematics[valid_mask] - self.obj_mean) / self.obj_std

        obj_features[:, :, 1:] = kinematics
        X_np[:, 2:] = obj_features.reshape(-1, 90)

        if was_torch:
            return torch.from_numpy(X_np).float()
        return X_np

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()

        # MET feature processing
        self.met_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )

        # Object embedding: E, pT, eta, phi -> embedding
        self.obj_embed = nn.Linear(4, 128)

        # Transformer encoder for object interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Pairwise feature network
        self.pairwise_net = nn.Sequential(
            nn.Linear(10, 64),  # delta_R, m_inv for multiple pairs + raw features
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(224, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def compute_pairwise_physics(self, kinematics, mask):
        # kinematics: [B, 18, 4] - E, pT, eta, phi
        # Returns physics-motivated pairwise features

        batch_size = kinematics.shape[0]
        device = kinematics.device

        E = kinematics[:, :, 0]    # [B, 18]
        pT = kinematics[:, :, 1]   # [B, 18]
        eta = kinematics[:, :, 2]  # [B, 18]
        phi = kinematics[:, :, 3]  # [B, 18]

        # Get top-4 objects by pT
        pT_masked = pT.clone()
        pT_masked[~mask] = -1e9
        _, top_idx = torch.topk(pT_masked, k=min(4, 18), dim=1)  # [B, 4]

        features_list = []

        # Compute features for top pairs
        for i in range(min(3, top_idx.shape[1])):
            for j in range(i + 1, min(4, top_idx.shape[1])):
                idx_i = top_idx[:, i]  # [B]
                idx_j = top_idx[:, j]  # [B]

                # Gather features
                pT_i = torch.gather(pT, 1, idx_i.unsqueeze(1)).squeeze(1)    # [B]
                pT_j = torch.gather(pT, 1, idx_j.unsqueeze(1)).squeeze(1)    # [B]
                eta_i = torch.gather(eta, 1, idx_i.unsqueeze(1)).squeeze(1)  # [B]
                eta_j = torch.gather(eta, 1, idx_j.unsqueeze(1)).squeeze(1)  # [B]
                phi_i = torch.gather(phi, 1, idx_i.unsqueeze(1)).squeeze(1)  # [B]
                phi_j = torch.gather(phi, 1, idx_j.unsqueeze(1)).squeeze(1)  # [B]

                # Delta R
                delta_eta = eta_i - eta_j
                delta_phi = phi_i - phi_j
                # Wrap phi to [-pi, pi]
                delta_phi = torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))
                delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)  # [B]

                # Invariant mass (simplified formula for massless particles)
                m_inv = torch.sqrt(torch.clamp(
                    2 * pT_i * pT_j * (torch.cosh(eta_i - eta_j) - torch.cos(phi_i - phi_j)),
                    min=1e-8
                ))  # [B]

                features_list.extend([delta_R, m_inv])

        # Pad if needed
        while len(features_list) < 10:
            features_list.append(torch.zeros(batch_size, device=device))

        features_list = features_list[:10]  # Take first 10
        pairwise_features = torch.stack(features_list, dim=1)  # [B, 10]

        return pairwise_features

    def forward(self, batch_x):
        # batch_x: [B, 92]
        batch_size = batch_x.shape[0]

        # Extract MET features
        met_features = batch_x[:, :2]  # [B, 2]

        # Extract object data
        obj_data = batch_x[:, 2:].view(batch_size, 18, 5)  # [B, 18, 5]
        obj_ids = obj_data[:, :, 0]  # [B, 18]
        obj_kinematics = obj_data[:, :, 1:]  # [B, 18, 4]

        # Valid object mask
        mask = (obj_ids != 0)  # [B, 18]

        # Process MET
        met_emb = self.met_net(met_features)  # [B, 64]

        # Embed objects
        obj_emb = self.obj_embed(obj_kinematics)  # [B, 18, 128]

        # Apply transformer with padding mask
        pad_mask = ~mask  # [B, 18]
        obj_encoded = self.transformer(obj_emb, src_key_padding_mask=pad_mask)  # [B, 18, 128]

        # Global pooling over valid objects
        mask_expanded = mask.unsqueeze(-1).float()  # [B, 18, 1]
        n_valid = mask_expanded.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, 1]
        obj_pooled = (obj_encoded * mask_expanded).sum(dim=1) / n_valid.squeeze(-1)  # [B, 128]

        # Compute pairwise physics features
        pairwise_features = self.compute_pairwise_physics(obj_kinematics, mask)  # [B, 10]
        pairwise_emb = self.pairwise_net(pairwise_features)  # [B, 32]

        # Combine all features
        combined = torch.cat([obj_pooled, met_emb, pairwise_emb], dim=1)  # [B, 224]

        # Final classification
        logits = self.classifier(combined)  # [B, 1]

        return logits.squeeze(-1)  # [B]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    best_auc = 0.0
    best_state = None
    patience = 7
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * yb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)

        avg_train_loss = train_loss_sum / train_total
        avg_train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                logits = model(xb)
                loss = criterion(logits, yb.float())

                val_loss_sum += loss.item() * yb.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        avg_val_loss = val_loss_sum / val_total
        avg_val_acc = val_correct / val_total
        val_auc = roc_auc_score(all_labels, all_probs)

        train_loss_hist.append(avg_train_loss)
        val_loss_hist.append(avg_val_loss)
        train_acc_hist.append(avg_train_acc)
        val_acc_hist.append(avg_val_acc)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"TrainLoss: {avg_train_loss:.4f}, TrainAcc: {avg_train_acc:.4f}, "
              f"ValLoss: {avg_val_loss:.4f}, ValAcc: {avg_val_acc:.4f}, ValAUC: {val_auc:.4f}")

        scheduler.step(val_auc)

        # Early stopping based on validation AUC
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

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

