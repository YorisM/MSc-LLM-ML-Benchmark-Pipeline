
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
    # Indices  2-6  : object 1  ->  [obj_1, E_1, p_T1, eta_1, phi_1]
    # Indices  7-11 : object 2  -> [obj_2, E_2 , p_T_2 , eta_2 , phi_2]
    # ...
    # Indices 88-92 : object 18 -> [obj_18, E_18 , p_T_18 , eta_18 , phi_18]
    # Per-object slice size = 5
    # Max objects encoded   = 18

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        # Indices for E_T_miss, and E_k, pT_k for each of the 18 objects
        self.log_transform_indices = [0]  # E_T_miss
        for i in range(18): # 18 objects
            self.log_transform_indices.append(3 + i * 5)  # E_k
            self.log_transform_indices.append(4 + i * 5)  # pT_k
        self.pT_indices = [4 + i * 5 for i in range(18)]

    def _preprocess_features(self, X_np: np.ndarray) -> np.ndarray:
        X_transformed = X_np.copy()
        
        # Apply log transform to specified energy/momentum features
        # log(1 + x/1000) to scale MeV to GeV-like values before log
        # log1p handles x=0 correctly (log1p(0) = 0)
        for idx in self.log_transform_indices:
            X_transformed[:, idx] = np.log1p(X_np[:, idx] / 1000.0)
        
        # Create 'number of active particles' feature
        # Count objects with pT > 1 MeV as active
        num_active_particles = np.sum(X_np[:, self.pT_indices] > 1.0, axis=1, keepdims=True)
        
        # Concatenate original (log-transformed) features with the new engineered feature
        X_full = np.concatenate((X_transformed, num_active_particles), axis=1)
        return X_full

    def fit(self, X, y=None): 
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
        
        X_processed = self._preprocess_features(X_np)
        self.scaler.fit(X_processed)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fit before transforming data.")
        
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
        
        X_processed = self._preprocess_features(X_np)
        X_scaled = self.scaler.transform(X_processed)
        
        return torch.from_numpy(X_scaled).float()

def make_preprocessor():
    return MyPreprocessor()

def make_model(input_dim: int):
    # PARAMETERS
    # input_dim : int : Number of features per event after preprocessing.

    # RETURNS
    # model : torch.nn.Module : Untrained binary-classifier network.
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.3),
        nn.Linear(64, 1)  # Output raw logits for BCEWithLogitsLoss
    )
    return model

EPOCHS = 30
    
def train_model(model, train_loader, val_loader, epochs):
    # PARAMETERS
    # model : torch.nn.Module   
    # train_loader: torch.utils.data.DataLoader
    # val_loader  : torch.utils.data.DataLoader
    # epochs: int

    # RETURNS
    # trained_model : nn.Module          (same instance, trained in-place)
    # train_loss    : list[float]        (length == epochs)
    # val_loss      : list[float]        (length == epochs)
    # train_acc     : list[float]        (length == epochs)
    # val_acc       : list[float]        (length == epochs)
    
    device = torch.device("cpu") # As per constraints
    model.to(device)

    # Using BCEWithLogitsLoss for numerical stability (expects raw logits from model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # Scheduler to adjust learning rate, e.g., Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
        
        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # Step the scheduler
        scheduler.step()
        
        # Printing epoch stats (optional, for local debugging)
        # print(f"Epoch {epoch+1}/{epochs} - "
        #       f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} - "
        #       f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f} - "
        #       f"LR: {scheduler.get_last_lr()[0]:.2e}")

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

