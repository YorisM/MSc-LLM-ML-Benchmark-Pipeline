
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
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.max_objects = 18
        self.obj_size = 5  # obj_id, E, pT, eta, phi

        # Statistics for normalization
        self.feature_means = None
        self.feature_stds = None

    def fit(self, X, y=None):
        # Calculate normalization statistics
        n_features = X.shape[1]
        self.feature_means = torch.zeros(n_features)
        self.feature_stds = torch.ones(n_features)

        # First two features are missing-ET magnitude and phi
        self.feature_means[0] = X[:, 0].mean()
        self.feature_stds[0] = X[:, 0].std() + 1e-8

        # Process object features
        for i in range(self.max_objects):
            start_idx = 2 + i * self.obj_size
            obj_id_idx = start_idx

            # Only consider valid objects (non-zero ID)
            valid_mask = X[:, obj_id_idx] != 0

            if valid_mask.sum() > 0:
                # Process each feature except obj_id and angles
                for j in range(1, self.obj_size):  # Skip object ID
                    feat_idx = start_idx + j
                    valid_values = X[valid_mask, feat_idx]

                    # Skip normalization for angles (eta and phi)
                    if j != 3 and j != 4:  # Not eta or phi
                        self.feature_means[feat_idx] = valid_values.mean()
                        self.feature_stds[feat_idx] = valid_values.std() + 1e-8

        return self

    def _calculate_delta_R(self, eta1, phi1, eta2, phi2):
        # Calculate ΔR = sqrt(Δη² + Δφ²)
        delta_eta = eta1 - eta2
        delta_phi = torch.abs(phi1 - phi2)
        # Handle phi wraparound
        delta_phi = torch.where(delta_phi > torch.pi, 2 * torch.pi - delta_phi, delta_phi)
        return torch.sqrt(delta_eta**2 + delta_phi**2)

    def transform(self, X):
        batch_size = X.shape[0]

        # Normalize features
        X_norm = X.clone()
        for i in range(X.shape[1]):
            # Skip object IDs and angle values (eta, phi)
            is_obj_id = (i - 2) % self.obj_size == 0
            is_eta = (i - 2) % self.obj_size == 3
            is_phi = (i - 2) % self.obj_size == 4

            if not (is_obj_id or is_eta or is_phi):
                X_norm[:, i] = (X[:, i] - self.feature_means[i]) / self.feature_stds[i]

        # Create mask for valid objects
        mask = torch.zeros((batch_size, self.max_objects), dtype=torch.bool)
        for i in range(self.max_objects):
            start_idx = 2 + i * self.obj_size
            obj_id = X[:, start_idx]
            mask[:, i] = (obj_id != 0)

        # Extract object features
        # Each object will have 4 basic features: E, pT, eta, phi
        obj_features = torch.zeros((batch_size, self.max_objects, 4), dtype=torch.float32)

        for i in range(self.max_objects):
            start_idx = 2 + i * self.obj_size
            obj_features[:, i, 0] = X_norm[:, start_idx + 1]  # E
            obj_features[:, i, 1] = X_norm[:, start_idx + 2]  # pT
            obj_features[:, i, 2] = X[:, start_idx + 3]       # eta (unnormalized)
            obj_features[:, i, 3] = X[:, start_idx + 4]       # phi (unnormalized)

        # Extract missing-ET features
        et_miss = X_norm[:, 0]  # Normalized
        phi_et_miss = X[:, 1]   # Unnormalized

        # Global features: missing ET, phi, and derived features
        global_features = torch.zeros((batch_size, 5), dtype=torch.float32)
        global_features[:, 0] = et_miss
        global_features[:, 1] = phi_et_miss

        # Calculate total visible energy and momentum
        valid_E = obj_features[:, :, 0] * mask.float()
        valid_pT = obj_features[:, :, 1] * mask.float()

        global_features[:, 2] = valid_E.sum(dim=1)  # Total visible energy
        global_features[:, 3] = valid_pT.sum(dim=1)  # Total visible pT
        global_features[:, 4] = mask.float().sum(dim=1)  # Number of objects

        return (obj_features, mask, global_features)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, input_shape, *, use_mask=False):
        super().__init__()
        self.use_mask = use_mask

        # Extract dimensions from input_shape
        obj_shape, global_shape = input_shape
        seq_len, feat_dim = obj_shape

        # Embedding dimension
        self.d_model = 64

        # Object features processing
        self.obj_projection = nn.Linear(feat_dim, self.d_model)

        # Transformer for object sequences
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Global features processing
        self.global_fc = nn.Sequential(
            nn.Linear(global_shape, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32)
        )

        # Combined processing
        self.combined_fc = nn.Sequential(
            nn.Linear(self.d_model + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, obj_data, mask=None, global_features=None):
        # Process object features
        x_obj = self.obj_projection(obj_data)  # [batch_size, seq_len, d_model]

        if self.use_mask and mask is not None:
            # Create attention mask for transformer
            attn_mask = ~mask  # Invert mask: True means padding
            x_obj = self.transformer(x_obj, src_key_padding_mask=attn_mask)
        else:
            x_obj = self.transformer(x_obj)

        # Global pooling of object features
        if self.use_mask and mask is not None:
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, self.d_model)
            masked_sum = (x_obj * mask_expanded.float()).sum(dim=1)
            x_obj_pooled = masked_sum / (mask.float().sum(dim=1, keepdim=True) + 1e-8)
        else:
            # Simple mean pooling
            x_obj_pooled = x_obj.mean(dim=1)

        # Process global features
        x_global = self.global_fc(global_features)

        # Combine all features
        x_combined = torch.cat([x_obj_pooled, x_global], dim=1)

        # Final processing
        x_out = self.combined_fc(x_combined)

        return x_out.squeeze(-1)

def make_model(input_shape, *, use_mask=False):
    return BinaryClassifier(input_shape, use_mask=use_mask)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Tracking metrics
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    # Early stopping parameters
    best_val_auc = 0
    patience = 10
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        all_preds = []
        all_labels = []

        for batch in train_loader:
            # Extract data and labels
            if isinstance(batch[0], tuple):
                data, labels = batch
                inputs = tuple(d.to(device) for d in data)
            else:
                inputs, labels = batch[0].to(device), batch[1]

            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
            loss = criterion(outputs, labels.float())

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)

            # Store predictions and labels for AUC calculation
            all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        # Calculate epoch metrics
        epoch_loss /= epoch_total
        epoch_acc = epoch_correct / epoch_total
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        train_auc = roc_auc_score(all_labels, all_preds)

        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        # Validation phase
        model.eval()
        val_epoch_loss = 0
        val_epoch_correct = 0
        val_epoch_total = 0
        val_all_preds = []
        val_all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                # Extract data and labels
                if isinstance(batch[0], tuple):
                    data, labels = batch
                    inputs = tuple(d.to(device) for d in data)
                else:
                    inputs, labels = batch[0].to(device), batch[1]

                labels = labels.to(device)

                # Forward pass
                outputs = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
                loss = criterion(outputs, labels.float())

                # Track metrics
                val_epoch_loss += loss.item() * labels.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_epoch_correct += (preds == labels).sum().item()
                val_epoch_total += labels.size(0)

                # Store predictions and labels for AUC calculation
                val_all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                val_all_labels.append(labels.cpu().numpy())

        # Calculate validation metrics
        val_epoch_loss /= val_epoch_total
        val_epoch_acc = val_epoch_correct / val_epoch_total
        val_all_preds = np.concatenate(val_all_preds)
        val_all_labels = np.concatenate(val_all_labels)
        val_auc = roc_auc_score(val_all_labels, val_all_preds)

        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)

        # Update learning rate
        scheduler.step(val_auc)

        # Check for early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model
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

