
import os, sys, pickle, torch, gc
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

def make_loaders(X_train, Y_train, X_val, Y_val, batch=512):
    train_ds = TensorDataset(X_train, Y_train)
    val_ds   = TensorDataset(X_val , Y_val)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0))
                        
# ----------------  START OF LLM BLOCK  ----------------

import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler # For pre-processing

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    #    Must implement:
    #   - fit(X: torch.Tensor, y: torch.Tensor) -> self
    #   - transform(X: torch.Tensor) -> torch.Tensor

    # REQUIREMENTS
    # IMPORTANT: All state must be picklable with the std-lib pickle module.
    # May allocate NumPy arrays or Torch tensors internally, but:
    # transform() must be deterministic & output a torch.Tensor of shape (N, features).
    # Store only derived parameters needed for transform i.e. do not store the raw data
    # itself in the preprocessor object.

    # DATA SPECIFICS
    # IMPORTANT: X_train, Y_train, X_val, Y_val are provided as PyTorch tensors in the environment.
    # Total flat length per event (X_train & X_val): 92
    # Index  0 :  missing-ET magnitude  (E_T_miss)
    # Index  1 :  missing-ET azimuth    (phi_Et_miss)
    # Indices  2-6  : object 1  ->  obj_1, E_1, p_T1, eta_1, phi_1
    # Indices  7-11 : object 2  -> obj_2, E_2 , p_T_2 , eta_2 , phi_2
    # ...
    # Indices 88-92 : object 18 -> obj_18, E_18 , p_T_18 , eta_18 , phi_18
    # Per-object slice size = 5
    # Max objects encoded   = 18

    # TIPS
    # When modifying data features or feature engineering: annotate tensor size as comments after 
    # each tensor operation to reduce dimension mismatches.

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False # Ensures fit is called before transform

    def fit(self, X, y=None):
        # X is a torch.Tensor. StandardScaler expects a NumPy array.
        # Fit the scaler on all 92 features directly.
        X_np = X.cpu().numpy() # Shape: (N, 92)
        self.scaler.fit(X_np)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
        # X is a torch.Tensor.
        X_np = X.cpu().numpy() # Shape: (N, 92)
        X_scaled_np = self.scaler.transform(X_np) # Shape: (N, 92), dtype typically float64
        # Convert back to torch.Tensor with original float32 dtype
        return torch.from_numpy(X_scaled_np).float() # Shape: (N, 92), dtype: torch.float32

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
def make_model(input_dim: int):
    # PARAMETERS
    # input_dim : int : Number of features per event after preprocessing.

    # RETURNS
    # model : torch.nn.Module : Untrained binary-classifier network.

    # A simple Multi-Layer Perceptron (MLP)
    # Architecture: Input -> Linear(128) -> ReLU -> Dropout(0.3) -> Linear(64) -> ReLU -> Dropout(0.3) -> Linear(1)
    # Dropout is added for regularization.
    # The final Linear layer outputs logits for BCEWithLogitsLoss.
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1)  # Output layer: 1 logit for binary classification
    )
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 15 # Chosen as a balance between performance and training time constraints.
    
def train_model(model, train_loader, val_loader, epochs):
    # PARAMETERS
    # model : torch.nn.Module   
    # train_loader: torch.utils.data.DataLoader
    # val_loader  : torch.utils.data.DataLoader
    # epochs: int

    # RETURNS
    # trained_model : nn.Module          (same instance, trained in-place)
    # train_loss    : list[float]        (length == epochs)
    # val_loss      : list[float]
    # train_acc     : list[float]
    # val_acc       : list[float]
    
    # REQUIREMENTS 
    # Define training loop clearly including number of epochs
    # Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Standard Adam optimizer.
    criterion = nn.BCEWithLogitsLoss() # Numerically stable loss for binary classification.

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Assuming model and data are on CPU as per problem constraints (4 CPU / 8 GB RAM)

    for epoch in range(epochs):
        model.train() # Set model to training mode
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            # labels are LongTensor (int64), shape (batch_size,). Convert for BCEWithLogitsLoss.
            labels_float = labels.float().unsqueeze(1) # Shape: (batch_size, 1), dtype: float32

            optimizer.zero_grad()
            outputs = model(inputs) # outputs shape: (batch_size, 1), logits
            loss = criterion(outputs, labels_float)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy for logging (primary metric is AUC, handled by caller)
            predicted = (torch.sigmoid(outputs) > 0.5).float() # Apply sigmoid and threshold for accuracy
            total_train += labels_float.size(0)
            correct_train += (predicted == labels_float).sum().item()

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        model.eval() # Set model to evaluation mode
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad(): # Disable gradient calculations for validation
            for inputs, labels in val_loader:
                labels_float = labels.float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels_float)
                running_val_loss += loss.item() * inputs.size(0)

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total_val += labels_float.size(0)
                correct_val += (predicted == labels_float).sum().item()
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
            
    return model, train_losses, val_losses, train_accuracies, val_accuracies

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
    pre = make_preprocessor()
    pre.fit(X_train, Y_train)
    X_train = pre.transform(X_train)
    X_val = pre.transform(X_val)
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val)

    # 2. Build model
    model = make_model(input_dim=X_train.shape[1])
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 3. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
        toy = torch.zeros(8, X_train.shape[1])      # 8 fake events
        try:
            _ = trained_model(pre.transform(toy))
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return  # no files in dry-run

    # 4. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
    torch.save(trained_model.state_dict(), f"{base}_state.pt")
    with open(f"{base}_model.pkl", "wb") as f: pickle.dump(trained_model, f)
    with open(f"{base}_preproc.pkl", "wb") as f: pickle.dump(pre, f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",      f"{base}_loss.png")
    _plot(tr_acc,  va_acc,  "Accuracy",  f"{base}_accuracy.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

