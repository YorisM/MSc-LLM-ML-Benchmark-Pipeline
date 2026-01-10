
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops

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
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.max_objects = 18
        self.obj_feature_size = 5
        self.global_feature_size = 2
        self.total_features = self.global_feature_size + self.max_objects * self.obj_feature_size

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
        # Extract global features (E_T_miss, phi_Et_miss) and object features separately
        global_features = X[:, :2].numpy()
        self.scaler.fit(global_features)
        return self

    def transform(self, X):
        # X shape: [N, 92]
        batch_size = X.shape[0]

        # Extract and scale global features
        global_features = X[:, :2].numpy()
        global_features = self.scaler.transform(global_features)
        global_features = torch.from_numpy(global_features).float()  # [N, 2]

        # Extract object features (obj_id, E, p_T, eta, phi)
        object_features = X[:, 2:].reshape(batch_size, self.max_objects, self.obj_feature_size)  # [N, 18, 5]

        # Convert object IDs to one-hot encoding (assuming obj_id is categorical)
        obj_ids = object_features[:, :, 0].long()  # [N, 18]
        obj_ids_onehot = F.one_hot(obj_ids, num_classes=6).float()  # [N, 18, 6] (assuming 6 classes)

        # Extract kinematic features (E, p_T, eta, phi)
        kinematic_features = object_features[:, :, 1:].float()  # [N, 18, 4]

        # Combine one-hot and kinematic features
        object_features_processed = torch.cat([obj_ids_onehot, kinematic_features], dim=-1)  # [N, 18, 10]

        # Compute pairwise features (invariant mass and delta R)
        pairwise_features = self._compute_pairwise_features(kinematic_features)  # [N, 18, 18, 2]

        # Flatten pairwise features for each object (take mean over other objects)
        pairwise_features_mean = pairwise_features.mean(dim=2)  # [N, 18, 2]

        # Combine all features
        object_features_final = torch.cat([
            object_features_processed,  # [N, 18, 10]
            pairwise_features_mean      # [N, 18, 2]
        ], dim=-1)  # [N, 18, 12]

        # Create mask for padding (where obj_id == 0)
        mask = (obj_ids != 0).float().unsqueeze(-1)  # [N, 18, 1]

        # Combine global and object features
        features = {
            'global_features': global_features,  # [N, 2]
            'object_features': object_features_final,  # [N, 18, 12]
            'mask': mask  # [N, 18, 1]
        }

        return features

    def _compute_pairwise_features(self, kinematic_features):
        # kinematic_features: [N, 18, 4] (E, p_T, eta, phi)
        batch_size, num_objects, _ = kinematic_features.shape

        # Extract components
        E = kinematic_features[:, :, 0].unsqueeze(-1)  # [N, 18, 1]
        px = kinematic_features[:, :, 1] * torch.cos(kinematic_features[:, :, 3])  # [N, 18]
        py = kinematic_features[:, :, 1] * torch.sin(kinematic_features[:, :, 3])  # [N, 18]
        pz = kinematic_features[:, :, 1] * torch.sinh(kinematic_features[:, :, 2])  # [N, 18]

        # Compute invariant mass m_ij = sqrt((E_i + E_j)^2 - (px_i + px_j)^2 - (py_i + py_j)^2 - (pz_i + pz_j)^2)
        E_i = E.unsqueeze(2)  # [N, 18, 1, 1]
        E_j = E.unsqueeze(1)  # [N, 1, 18, 1]
        px_i = px.unsqueeze(2)  # [N, 18, 1]
        px_j = px.unsqueeze(1)  # [N, 1, 18]
        py_i = py.unsqueeze(2)  # [N, 18, 1]
        py_j = py.unsqueeze(1)  # [N, 1, 18]
        pz_i = pz.unsqueeze(2)  # [N, 18, 1]
        pz_j = pz.unsqueeze(1)  # [N, 1, 18]

        m_ij_squared = (E_i + E_j).pow(2) - (px_i + px_j).pow(2) - (py_i + py_j).pow(2) - (pz_i + pz_j).pow(2)
        m_ij = torch.sqrt(torch.clamp(m_ij_squared, min=0))  # [N, 18, 18]

        # Compute delta R = sqrt((eta_i - eta_j)^2 + (phi_i - phi_j)^2)
        eta_i = kinematic_features[:, :, 2].unsqueeze(2)  # [N, 18, 1]
        eta_j = kinematic_features[:, :, 2].unsqueeze(1)  # [N, 1, 18]
        phi_i = kinematic_features[:, :, 3].unsqueeze(2)  # [N, 18, 1]
        phi_j = kinematic_features[:, :, 3].unsqueeze(1)  # [N, 1, 18]

        delta_eta = eta_i - eta_j
        delta_phi = torch.min(
            torch.abs(phi_i - phi_j),
            2 * math.pi - torch.abs(phi_i - phi_j)
        )
        delta_R = torch.sqrt(delta_eta.pow(2) + delta_phi.pow(2))  # [N, 18, 18]

        # Stack pairwise features
        pairwise_features = torch.stack([m_ij, delta_R], dim=-1)  # [N, 18, 18, 2]

        return pairwise_features

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=18):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [N, seq_len, d_model]
        x = x + self.pe[:x.size(1)]
        return x

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.object_feature_size = 12  # From preprocessor
        self.global_feature_size = 2
        self.d_model = 128
        self.nhead = 8
        self.num_layers = 4
        self.dropout = 0.1

        # Object feature embedding
        self.obj_embed = nn.Linear(self.object_feature_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=4*self.d_model,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=self.num_layers)

        # Global feature embedding
        self.global_embed = nn.Linear(self.global_feature_size, self.d_model)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1)
        )

    def forward(self, batch_x):
        # batch_x is a dictionary from preprocessor
        global_features = batch_x['global_features']  # [B, 2]
        object_features = batch_x['object_features']  # [B, 18, 12]
        mask = batch_x['mask']  # [B, 18, 1]

        # Embed object features
        obj_embedded = self.obj_embed(object_features)  # [B, 18, d_model]
        obj_embedded = self.pos_encoder(obj_embedded)  # [B, 18, d_model]

        # Create attention mask (1 for real objects, 0 for padding)
        key_padding_mask = (mask.squeeze(-1) == 0)  # [B, 18]

        # Transformer encoder
        transformer_out = self.transformer_encoder(
            obj_embedded,
            src_key_padding_mask=key_padding_mask
        )  # [B, 18, d_model]

        # Global average pooling over objects
        mask_expanded = mask.expand_as(transformer_out)  # [B, 18, d_model]
        sum_out = (transformer_out * mask_expanded).sum(dim=1)  # [B, d_model]
        count = mask.sum(dim=1)  # [B, 1]
        pooled_out = sum_out / count.clamp(min=1)  # [B, d_model]

        # Embed global features
        global_embedded = self.global_embed(global_features)  # [B, d_model]

        # Concatenate global and pooled object features
        combined = torch.cat([global_embedded, pooled_out], dim=-1)  # [B, 2*d_model]

        # Classifier
        logits = self.classifier(combined).squeeze(-1)  # [B]

        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch_x, batch_y = batch
            # Move batch_x to device (it's a dict)
            batch_x = {k: v.to(device) for k, v in batch_x.items()}
            batch_y = batch_y.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_y.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch_x, batch_y = batch
                batch_x = {k: v.to(device) for k, v in batch_x.items()}
                batch_y = batch_y.to(device).float()

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_y.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)

                probs = torch.sigmoid(outputs)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Compute AUC
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(all_labels, all_probs)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')

        # Early stopping based on AUC
        scheduler.step(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()

        # Early stopping if learning rate becomes too small
        if optimizer.param_groups[0]['lr'] < 1e-6:
            print("Early stopping due to low learning rate")
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

    # Build batch and check
    first_batch = next(iter(train_loader))
    mode = detect_and_assert_lane_fourtops(spec, first_batch)
    view = make_view_by_lane_fourtops(mode, first_batch, device)

    # Build model
    model = make_model(view.batch_x).to(device)

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
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

