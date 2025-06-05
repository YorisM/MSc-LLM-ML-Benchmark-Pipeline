
import os, sys, pickle, torch, gc, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

DATASET = {
    "X_train": "./challenges/FOURTOPS/data/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/Y_val.csv"
}
                       
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
                        
# ----------------  START OF LLM BLOCK  ----------------

# 0. ---------- IMPORTS ----------
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import lr_scheduler
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.et_miss_scaler = StandardScaler()
        self.E_scaler = StandardScaler()
        self.pT_scaler = StandardScaler()
        self.n_objects = 18  # Max objects encoded
        self.obj_features = 5  # Features per object

    def fit(self, X, y=None):
        # Extract ETmiss features
        et_miss = X[:, 0].cpu().numpy().reshape(-1, 1)

        # Create scalers for ETmiss
        self.et_miss_scaler.fit(et_miss)

        # Collect all non-zero energy and pT values for normalization
        all_E = []
        all_pT = []

        for i in range(self.n_objects):
            start_idx = 2 + i * self.obj_features

            # Extract object features
            obj_id = X[:, start_idx].cpu().numpy()
            obj_E = X[:, start_idx + 1].cpu().numpy()
            obj_pT = X[:, start_idx + 2].cpu().numpy()

            # Only include objects with non-zero obj_id
            valid_mask = (obj_id != 0)

            if np.any(valid_mask):
                all_E.append(obj_E[valid_mask])
                all_pT.append(obj_pT[valid_mask])

        if all_E and all_pT:
            # Fit scalers to energy and pT
            self.E_scaler.fit(np.concatenate(all_E).reshape(-1, 1))
            self.pT_scaler.fit(np.concatenate(all_pT).reshape(-1, 1))

        return self

    def transform(self, X):
        batch_size = X.shape[0]

        # Create tensors for sequence data
        seq_data = torch.zeros((batch_size, self.n_objects, self.obj_features + 1), dtype=torch.float32)
        seq_mask = torch.zeros((batch_size, self.n_objects), dtype=torch.bool)

        # Extract and normalize ETmiss
        et_miss = X[:, 0].clone().reshape(-1, 1)
        phi_et_miss = X[:, 1].clone()

        # Normalize ETmiss
        et_miss_norm = torch.tensor(
            self.et_miss_scaler.transform(et_miss.cpu().numpy()),
            dtype=torch.float32
        ).squeeze()

        # Process objects
        for i in range(self.n_objects):
            start_idx = 2 + i * self.obj_features

            # Extract object features
            obj_id = X[:, start_idx].clone()
            obj_E = X[:, start_idx + 1].clone()
            obj_pT = X[:, start_idx + 2].clone()
            obj_eta = X[:, start_idx + 3].clone()
            obj_phi = X[:, start_idx + 4].clone()

            # Identify valid objects
            valid_mask = (obj_id != 0)
            seq_mask[:, i] = valid_mask

            # Store object ID
            seq_data[:, i, 0] = obj_id

            # Normalize and store energy
            E_norm = torch.zeros_like(obj_E)
            if valid_mask.any():
                E_norm[valid_mask] = torch.tensor(
                    self.E_scaler.transform(obj_E[valid_mask].cpu().numpy().reshape(-1, 1)),
                    dtype=torch.float32
                ).squeeze()
            seq_data[:, i, 1] = E_norm

            # Normalize and store pT
            pT_norm = torch.zeros_like(obj_pT)
            if valid_mask.any():
                pT_norm[valid_mask] = torch.tensor(
                    self.pT_scaler.transform(obj_pT[valid_mask].cpu().numpy().reshape(-1, 1)),
                    dtype=torch.float32
                ).squeeze()
            seq_data[:, i, 2] = pT_norm

            # Store eta and phi (no normalization)
            seq_data[:, i, 3] = obj_eta
            seq_data[:, i, 4] = obj_phi

            # Add ETmiss as a feature for each object
            seq_data[:, i, 5] = et_miss_norm

            # Calculate dR to ETmiss for valid objects
            if valid_mask.any():
                delta_phi = torch.abs(obj_phi[valid_mask] - phi_et_miss[valid_mask])
                delta_phi = torch.min(delta_phi, 2*np.pi - delta_phi)
                # Here we'd add this as an additional feature if needed

        return seq_data, seq_mask

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, input_shape, *, use_mask=True):
        super().__init__()
        self.use_mask = use_mask

        # Extract input dimensions
        self.n_objects, self.features_per_obj = input_shape

        # Model hyperparameters
        self.embed_dim = 128
        self.n_heads = 4
        self.n_layers = 3
        self.dropout = 0.2

        # Object embedding
        self.obj_embed = nn.Linear(self.features_per_obj, self.embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.n_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.Softmax(dim=1)
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, data, mask=None):
        if self.use_mask:
            # Handle input
            if isinstance(data, tuple):
                seq_data, seq_mask = data
            else:
                seq_data = data
                seq_mask = mask

            # Embed object features
            x = self.obj_embed(seq_data)  # (batch, n_objects, embed_dim)

            # Apply transformer
            # Transformer expects padding mask where True values are masked
            transformer_mask = ~seq_mask
            x = self.transformer(x, src_key_padding_mask=transformer_mask)  # (batch, n_objects, embed_dim)

            # Apply attention pooling
            attn_weights = self.attention(x)  # (batch, n_objects, 1)

            # Apply mask to attention weights
            mask_expanded = seq_mask.unsqueeze(-1).float()
            masked_attn = attn_weights * mask_expanded

            # Normalize attention weights
            attn_sum = masked_attn.sum(dim=1, keepdim=True)
            normalized_attn = masked_attn / (attn_sum + 1e-10)

            # Weighted sum
            pooled = (x * normalized_attn).sum(dim=1)  # (batch, embed_dim)

            # Classify
            output = self.classifier(pooled)
            return output.squeeze(-1)
        else:
            # This model is designed to work with masked sequence data
            raise NotImplementedError("This model requires masked sequence data")

def make_model(input_shape, *, use_mask=True):
    return BinaryClassifier(input_shape, use_mask=use_mask)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Define scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Metrics tracking
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    # Early stopping
    best_val_auc = 0
    patience = 5
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        epoch_preds = []
        epoch_labels = []

        for batch_data, batch_labels in train_loader:
            # Move data to device
            if isinstance(batch_data, tuple):
                seq_data, seq_mask = batch_data
                seq_data = seq_data.to(device)
                seq_mask = seq_mask.to(device)
                batch_data = (seq_data, seq_mask)
            else:
                batch_data = batch_data.to(device)

            batch_labels = batch_labels.to(device).float()

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_data)

            # Compute loss
            loss = F.binary_cross_entropy(outputs, batch_labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            epoch_preds.extend(outputs.detach().cpu().numpy())
            epoch_labels.extend(batch_labels.cpu().numpy())

        # Calculate epoch metrics
        epoch_loss /= len(train_loader)
        epoch_auc = roc_auc_score(epoch_labels, epoch_preds)
        epoch_acc = accuracy_score(epoch_labels, np.round(epoch_preds))

        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        # Validation
        model.eval()
        val_epoch_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                # Move data to device
                if isinstance(batch_data, tuple):
                    seq_data, seq_mask = batch_data
                    seq_data = seq_data.to(device)
                    seq_mask = seq_mask.to(device)
                    batch_data = (seq_data, seq_mask)
                else:
                    batch_data = batch_data.to(device)

                batch_labels = batch_labels.to(device).float()

                # Forward pass
                outputs = model(batch_data)

                # Compute loss
                loss = F.binary_cross_entropy(outputs, batch_labels)

                # Track metrics
                val_epoch_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())

        # Calculate validation metrics
        val_epoch_loss /= len(val_loader)
        val_epoch_auc = roc_auc_score(val_labels, val_preds)
        val_epoch_acc = accuracy_score(val_labels, np.round(val_preds))

        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)

        # Update learning rate
        scheduler.step(val_epoch_auc)

        # Early stopping check
        if val_epoch_auc > best_val_auc:
            best_val_auc = val_epoch_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_loss, val_loss, train_acc, val_acc

# ----------------  END OF LLM-CODE BLOCK ----------------
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    pre = make_preprocessor().fit(X_train, Y_train)
    X_train = pre.transform(X_train) # may be Tensor or Tuple
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

