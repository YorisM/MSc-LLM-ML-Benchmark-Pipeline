
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.means = None
        self.stds = None
        self.max_objects = 18  # Maximum number of objects (92-2)/5 = 18
        self.object_dim = 5    # Each object has 5 features

    def fit(self, X, y=None):
        # Calculate normalization statistics
        self.means = torch.mean(X, dim=0)
        self.stds = torch.std(X, dim=0)
        # Avoid division by zero
        self.stds[self.stds < 1e-5] = 1.0
        return self

    def transform(self, X):
        # Normalize
        X_norm = (X - self.means) / self.stds  # (batch_size, 92)

        batch_size = X.shape[0]

        # Extract missing ET and phi
        et_miss = X_norm[:, 0].unsqueeze(1)  # (batch_size, 1)
        phi_et_miss = X_norm[:, 1].unsqueeze(1)  # (batch_size, 1)

        # Reshape the object data
        object_data = X_norm[:, 2:].reshape(batch_size, self.max_objects, self.object_dim)  # (batch_size, 18, 5)

        # Create mask for valid objects (where obj_id is non-zero)
        mask = (object_data[:, :, 0] != 0)  # (batch_size, 18)

        # Combine ET_miss and phi_ET_miss as global features
        global_features = torch.cat([et_miss, phi_et_miss], dim=1)  # (batch_size, 2)

        return (object_data, global_features, mask)

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

        # Input shape: (max_objects, object_dim)
        self.max_objects, self.object_dim = input_shape

        # Object embedding
        self.object_embedding = nn.Sequential(
            nn.Linear(self.object_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # Global feature embedding
        self.global_embedding = nn.Sequential(
            nn.Linear(2, 32),  # 2 global features (ET_miss, phi_ET_miss)
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # Transformer for object interaction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 128),  # Concatenated features
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, data, mask=None):
        if isinstance(data, tuple):
            # Unpack data if it's a tuple from the preprocessor
            object_data, global_features, object_mask = data
        else:
            # This case should not happen with our preprocessor
            object_data = data
            batch_size = object_data.shape[0]
            global_features = torch.zeros(batch_size, 2, device=data.device)
            object_mask = mask

        # Process object features
        x_objects = self.object_embedding(object_data)  # (batch_size, max_objects, 128)

        # Apply transformer with masking
        if self.use_mask and object_mask is not None:
            # Transformer expects src_key_padding_mask where True values are ignored
            padding_mask = ~object_mask  # (batch_size, max_objects)
            x_transformed = self.transformer(x_objects, src_key_padding_mask=padding_mask)
        else:
            x_transformed = self.transformer(x_objects)

        # Aggregate object features with attention to masking
        if self.use_mask and object_mask is not None:
            # Apply mask to get valid objects only
            mask_expanded = object_mask.unsqueeze(-1).expand_as(x_transformed)  # (batch_size, max_objects, 128)
            sum_features = (x_transformed * mask_expanded).sum(dim=1)  # (batch_size, 128)
            # Divide by number of valid objects (prevent division by zero)
            count = object_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (batch_size, 1)
            x_agg = sum_features / count  # (batch_size, 128)
        else:
            # Simple average if no mask
            x_agg = x_transformed.mean(dim=1)  # (batch_size, 128)

        # Process global features
        x_global = self.global_embedding(global_features)  # (batch_size, 64)

        # Combine features
        x_combined = torch.cat([x_agg, x_global], dim=1)  # (batch_size, 128+64)

        # Final classification
        logits = self.classifier(x_combined).squeeze(-1)  # (batch_size)

        return logits

def make_model(input_shape, *, use_mask=False):
    return BinaryClassifier(input_shape, use_mask=use_mask)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, threshold=0.001
    )

    # Lists to store metrics
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    # Early stopping variables
    best_val_auc = 0
    best_model = None
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        train_preds = []
        train_targets = []

        for batch in train_loader:
            # Handle data
            if isinstance(batch[0], tuple):
                inputs, labels = batch
                inputs = tuple(i.to(device) for i in inputs)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)

            labels = labels.to(device).float()

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate metrics
            epoch_train_loss += loss.item() * labels.size(0)
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        # Calculate epoch metrics
        epoch_train_loss /= len(train_loader.dataset)
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)
        epoch_train_acc = accuracy_score(train_targets.round(), train_preds.round())

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                # Handle data
                if isinstance(batch[0], tuple):
                    inputs, labels = batch
                    inputs = tuple(i.to(device) for i in inputs)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(device)

                labels = labels.to(device).float()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Accumulate metrics
                epoch_val_loss += loss.item() * labels.size(0)
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        # Calculate epoch metrics
        epoch_val_loss /= len(val_loader.dataset)
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        epoch_val_acc = accuracy_score(val_targets.round(), val_preds.round())
        epoch_val_auc = roc_auc_score(val_targets, val_preds)

        # Store metrics
        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)
        train_acc.append(epoch_train_acc)
        val_acc.append(epoch_val_acc)

        # Update learning rate
        scheduler.step(epoch_val_auc)

        # Early stopping check
        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best Val AUC: {best_val_auc:.4f}")
                break

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

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

