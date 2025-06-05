
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
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.means = None
        self.stds = None
        self.max_objects = 18
        self.features_per_object = 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X, y=None):
        # Calculate statistics for normalization
        self.means = torch.mean(X, dim=0)
        self.stds = torch.std(X, dim=0)

        # Replace zeros in stds to avoid division by zero
        self.stds = torch.where(self.stds == 0, torch.ones_like(self.stds), self.stds)

        return self

    def transform(self, X):
        # Normalize all features
        X_norm = (X - self.means) / self.stds

        batch_size = X.shape[0]

        # Extract missing energy features
        met_magnitude = X_norm[:, 0:1]  # E_T_miss
        met_phi = X_norm[:, 1:2]        # phi_{E_t}_miss

        # Create object features tensor (batch_size, max_objects, features_per_object)
        object_features = torch.zeros((batch_size, self.max_objects, self.features_per_object), device=X.device)

        # Extract object features
        for i in range(self.max_objects):
            start_idx = 2 + i * self.features_per_object
            end_idx = start_idx + self.features_per_object
            if end_idx <= X.shape[1]:
                object_features[:, i, :] = X_norm[:, start_idx:end_idx]

        # Create a mask for valid objects (non-zero object IDs)
        valid_objects = object_features[:, :, 0] != 0

        # Calculate derived physics features

        # Total pT (HT)
        pt_values = object_features[:, :, 2]
        ht = torch.sum(pt_values * valid_objects.float(), dim=1, keepdim=True)

        # MET significance
        met_significance = met_magnitude / (torch.sqrt(ht + 1e-8))

        # Count of objects
        obj_count = torch.sum(valid_objects.float(), dim=1, keepdim=True) / self.max_objects

        # Create a flattened representation with original features and derived ones
        flat_objects = object_features.reshape(batch_size, -1)

        # Add derived features
        enhanced_features = torch.cat([
            met_magnitude, 
            met_phi, 
            ht, 
            met_significance, 
            obj_count, 
            flat_objects
        ], dim=1)

        return enhanced_features

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
def make_model(input_shape, *, use_mask=False):
    # Model architecture parameters
    n_features = input_shape[0]
    hidden_sizes = [512, 256, 128, 64]
    dropout_rate = 0.3

    # Define model
    class HEPClassifier(nn.Module):
        def __init__(self, n_features, hidden_sizes, dropout_rate):
            super().__init__()

            # Build model layers
            layers = []

            # Input layer
            layers.append(nn.Linear(n_features, hidden_sizes[0]))
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

            # Hidden layers
            for i in range(len(hidden_sizes)-1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))

            # Output layer
            layers.append(nn.Linear(hidden_sizes[-1], 1))

            self.model = nn.Sequential(*layers)

        def forward(self, x, mask=None):
            # Ignore mask as we're using a flattened representation
            return torch.sigmoid(self.model(x)).squeeze()

    model = HEPClassifier(n_features, hidden_sizes, dropout_rate)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 100

def train_model(model, train_loader, val_loader, epochs):
    # Define loss function
    criterion = nn.BCELoss()

    # Define optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )

    # Initialize tracking variables
    best_val_auc = 0.0
    best_model_state = None
    patience = 15
    patience_counter = 0

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for data, labels in train_loader:
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels.float())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update stats
            epoch_train_loss += loss.item() * data.size(0)
            train_total += labels.size(0)
            train_correct += ((outputs >= 0.5).float() == labels).sum().item()

        # Calculate epoch statistics
        epoch_train_loss /= train_total
        epoch_train_acc = train_correct / train_total
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_true_labels = []

        with torch.no_grad():
            for data, labels in val_loader:
                # Forward pass
                outputs = model(data)
                loss = criterion(outputs, labels.float())

                # Update stats
                epoch_val_loss += loss.item() * data.size(0)
                val_total += labels.size(0)
                val_correct += ((outputs >= 0.5).float() == labels).sum().item()

                # Store predictions for AUC calculation
                val_predictions.append(outputs.cpu())
                val_true_labels.append(labels.cpu())

        # Calculate validation metrics
        epoch_val_loss /= val_total
        epoch_val_acc = val_correct / val_total
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        # Calculate AUC
        all_predictions = torch.cat(val_predictions).numpy()
        all_labels = torch.cat(val_true_labels).numpy()
        val_auc = roc_auc_score(all_labels, all_predictions)

        # Update learning rate based on validation AUC
        scheduler.step(val_auc)

        # Check for improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_loss, val_loss, train_acc, val_acc

# ----------------  END OF LLM BLOCK ----------------
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("epoch"); plt.legend()
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

    # 4. *Dry-run safety check* – run a single toy forward pass
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

