
import os, sys, json, pickle, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"

DATASET = {
    "X_train": "./challenges/FOURTOPS/data/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/Y_val.csv"
}
                       
def load_data():
    X_train_df = pd.read_csv('./challenges/FOURTOPS/data/X_train.csv')
    Y_train_df = pd.read_csv('./challenges/FOURTOPS/data/Y_train.csv')
    X_val_df   = pd.read_csv('./challenges/FOURTOPS/data/X_val.csv')
    Y_val_df   = pd.read_csv('./challenges/FOURTOPS/data/Y_val.csv')

    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_df.values, dtype=torch.long).squeeze()
    X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
    Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long).squeeze()
    return X_train, Y_train, X_val, Y_val

def make_loaders(X_train, Y_train, X_val, Y_val, batch=1024):
    train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train))
    val = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val))
    return (DataLoader(train, batch_size=batch, shuffle=True),
            DataLoader(val, batch_size=batch))
                        
# ----------------  START OF LLM BLOCK  ----------------
# Imports
import os, sys, json, pickle, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
# Only import extra std-lib modules, torch.nn or sklearn sub-modules you actually use.
from sklearn.preprocessing import StandardScaler
import torch.optim as optim

class MyPreprocessor:
    #    Must implement:
    #   - fit(X: np.ndarray | torch.Tensor, y: np.ndarray | None) -> self
    #   - transform(X: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor

    # REQUIREMENTS
    # IMPORTANT: All state must be picklable with the std-lib pickle module.
    # May allocate NumPy arrays or Torch tensors internally, but:
    # transform() must be deterministic & must output a NumPy array *or* 
    # a torch.Tensor of shape (N, features).

    # DATA SHAPE
    # Total flat length per event: 92
    # Index  0 :  missing-ET magnitude  (E_T_miss)
    # Index  1 :  missing-ET azimuth    (phi_Et_miss)
    # Indices  2–6  : object 1  →  [obj_1, E_1, p_T1, eta_1, phi_1]
    # ... (Indices 88–92 for object 18)
    # Per-object slice size = 5
    # Max objects encoded   = 18

    # Original feature spec
    _IDX_ET_MISS = 0
    _IDX_PHI_ET_MISS = 1
    _OBJ_BASE_IDX_ORIG = 2
    _OBJ_FEATURES_ORIG = 5
    _OBJ_IDX_E_ORIG = 1
    _OBJ_IDX_PT_ORIG = 2
    _OBJ_IDX_ETA_ORIG = 3
    _OBJ_IDX_PHI_ORIG = 4
    _MAX_OBJECTS = 18

    # Processed feature spec
    # Global: E_T_miss_scaled, cos(phi_ET_miss), sin(phi_ET_miss) -> 3 features
    # Per object: obj_id, E_scaled, pT_scaled, eta_scaled, cos(phi), sin(phi) -> 6 features
    _PROC_GLOBAL_FEATURE_COUNT = 3
    _PROC_OBJ_FEATURE_COUNT = 6
    output_dim = _PROC_GLOBAL_FEATURE_COUNT + _MAX_OBJECTS * _PROC_OBJ_FEATURE_COUNT # 3 + 18 * 6 = 111

    def __init__(self):
        self.scaler_ = None
        self.scale_indices_ = [self._IDX_ET_MISS] + \
                              [self._OBJ_BASE_IDX_ORIG + i * self._OBJ_FEATURES_ORIG + j 
                               for i in range(self._MAX_OBJECTS) 
                               for j in [self._OBJ_IDX_E_ORIG, self._OBJ_IDX_PT_ORIG, self._OBJ_IDX_ETA_ORIG]]

    def _to_numpy(self, X):
        if isinstance(X, torch.Tensor):
            return X.cpu().numpy()
        return np.asarray(X) # Ensure it's a NumPy array

    def fit(self, X, y=None):
        X_np = self._to_numpy(X)
        
        X_to_scale = X_np[:, self.scale_indices_]
        
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_to_scale)
        return self

    def transform(self, X_orig):
        if self.scaler_ is None:
            raise RuntimeError("Preprocessor must be fitted before transform is called.")
        
        X_np_orig = self._to_numpy(X_orig)
        num_samples = X_np_orig.shape[0]

        # Initialize the output tensor
        # Using a list of tensors and then torch.cat for assembly
        processed_parts = [] 

        # Scale the relevant columns first
        X_scaled_subset = self.scaler_.transform(X_np_orig[:, self.scale_indices_])
        X_scaled_subset_torch = torch.from_numpy(X_scaled_subset).float()

        # 1. E_T_miss (scaled)
        processed_parts.append(X_scaled_subset_torch[:, 0:1]) # ET_miss is the 0-th column in X_scaled_subset_torch

        # 2. phi_Et_miss (cos, sin)
        phi_Et_miss = torch.from_numpy(X_np_orig[:, self._IDX_PHI_ET_MISS]).float()
        processed_parts.append(torch.cos(phi_Et_miss).unsqueeze(1))
        processed_parts.append(torch.sin(phi_Et_miss).unsqueeze(1))
        
        current_scaled_col_idx = 1 # Index for X_scaled_subset_torch, after ET_miss

        for i in range(self._MAX_OBJECTS):
            obj_orig_start_idx = self._OBJ_BASE_IDX_ORIG + i * self._OBJ_FEATURES_ORIG
            
            # obj_id (original, unscaled)
            # Column 0 of object block: obj_id
            processed_parts.append(torch.from_numpy(X_np_orig[:, obj_orig_start_idx]).float().unsqueeze(1))
            
            # E_i, p_Ti, eta_i (scaled)
            # These correspond to 3 columns in X_scaled_subset_torch for each object
            processed_parts.append(X_scaled_subset_torch[:, current_scaled_col_idx : current_scaled_col_idx+3])
            current_scaled_col_idx += 3
            
            # phi_i (cos, sin)
            # Column 4 of object block: phi_i
            phi_i = torch.from_numpy(X_np_orig[:, obj_orig_start_idx + self._OBJ_IDX_PHI_ORIG]).float()
            processed_parts.append(torch.cos(phi_i).unsqueeze(1))
            processed_parts.append(torch.sin(phi_i).unsqueeze(1))
            
        X_processed = torch.cat(processed_parts, dim=1)
        return X_processed

def make_preprocessor():
    return MyPreprocessor()

def make_model(input_dim: int):
    # PARAMETERS
    # input_dim : int : Number of features per event after preprocessing.

    # RETURNS
    # model : torch.nn.Module : Untrained binary-classifier network.
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1)
    )
    return model

EPOCHS = 30
    
def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                epochs: int):
    # RETURNS
    # trained_model : nn.Module          (same instance, trained in-place)
    # train_loss    : list[float]        (length == epochs)
    # val_loss      : list[float]        (length == epochs)
    # train_acc     : list[float]        (length == epochs)
    # val_acc       : list[float]        (length == epochs)
    
    # REQUIREMENTS 
    # Define training loop clearly including number of epochs
    # Do NOT pass “verbose=” to any PyTorch scheduler (not supported in this image).

    device = torch.device("cpu") # As per constraints
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Scheduler: ReduceLROnPlateau by default has verbose=False. So not passing it is fine.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train() # Set model to training mode
        running_train_loss = 0.0
        correct_train, total_train = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            
            preds = torch.sigmoid(outputs).round()
            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        model.eval() # Set model to evaluation mode
        running_val_loss = 0.0
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                
                preds = torch.sigmoid(outputs).round()
                total_val += labels.size(0)
                correct_val += (preds == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        scheduler.step(epoch_val_loss) # Step scheduler based on validation loss
            
    return model, train_losses, val_losses, train_accs, val_accs
# ----------------  END OF LLM BLOCK ----------------
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_tr, y_tr, X_va, y_va = load_data()
    pre = make_preprocessor();  pre.fit(X_tr, y_tr)
    X_tr = pre.transform(X_tr); X_va = pre.transform(X_va)
    tr_loader, va_loader = make_loaders(X_tr, y_tr, X_va, y_va)

    # 2. Build model
    model = make_model(input_dim=X_tr.shape[1])
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    trained, tr_loss, va_loss, tr_acc, va_acc = train_model(
        model, tr_loader, va_loader, epochs=n_epochs
    )

    # 3. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
        toy = torch.zeros(8, X_tr.shape[1])      # 8 fake events
        try:
            _ = trained(pre.transform(toy))
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return  # no files in dry-run

    # 4. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
    torch.save(trained.state_dict(), f"{base}_state.pt")
    with open(f"{base}_model.pkl", "wb") as f: pickle.dump(trained, f)
    with open(f"{base}_preproc.pkl", "wb") as f: pickle.dump(pre, f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",      f"{base}_loss.png")
    _plot(tr_acc,  va_acc,  "Accuracy",  f"{base}_accuracy.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

