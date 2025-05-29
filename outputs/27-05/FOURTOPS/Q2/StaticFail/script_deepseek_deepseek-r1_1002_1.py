
import os, sys, pickle, torch, gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
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
from sklearn.metrics import roc_auc_score

class MyPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        device = X.device
        n = X.shape[0]
        
        # Split into physics components
        # Process objects [B, 18,5]->[B, 18,4: E,px,py,pz]
        X_obj = X[:,2:].view(-1,18,5)
        
        # Precompute sine/cosine for faster ops
        phi = X_obj[...,4]  # obj_phi
        pt = X_obj[...,3]   # p_T (MeV)
        eta = X_obj[...,2]  # eta
        obj_valid = (X_obj[...,0] != 0).float()
        
        # Convert to Cartesian coordinates (Lorentz symmetry operations)
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)
        E = X_obj[...,1]/1000  # GeV scale
        
        # Normalize each objects four-vector
        four_vectors = torch.stack([E, px/1000, py/1000, pz/1000], dim=-1)
        
        # Create global energy features
        system_px = px/1000.sum(dim=1)
        system_py = py/1000.sum(dim=1)
        
        # Missing ET components
        et_miss = X[:,0]/1000
        et_phi = X[:,1]
        px_miss = et_miss * torch.cos(et_phi)
        py_miss = et_miss * torch.sin(et_phi)
        
        # Pairwise invariant mass computations
        energy = four_vectors[...,0]
        vector_part = four_vectors[...,1:]
        mass_sq = (energy**2 - (vector_part**2).sum(dim=-1)) * obj_valid
        
        # Flatten four-vector and global features
        X_out = torch.cat([
            four_vectors.view(n,-1),
            mass_sq.view(n,-1),
            system_px.unsqueeze(-1),
            system_py.unsqueeze(-1),
            px_miss.unsqueeze(-1),
            py_miss.unsqueeze(-1)
        ], dim=1)
        
        return X_out


def make_model(input_dim: int):
    class LorentzInvariantNN(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Equivariant message passing via Tensor products
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.LayerNorm(64),
                
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.LayerNorm(32),
                
                # Final classifier head
                nn.Linear(32, 1)
            )
            
        def forward(self, x):
            return self.mlp(x)
    
    return LorentzInvariantNN()


EPOCHS = 15

def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            x, y = batch
            y = y.float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
        
        train_loss.append(epoch_loss / len(train_loader))
        train_acc.append(correct / total)
        
        # Validation
        model.eval()
        val_epoch_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                y = y.float().unsqueeze(1)
                outputs = model(x)
                loss = criterion(outputs, y)
                
                val_epoch_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        
        val_loss.append(val_epoch_loss / len(val_loader))
        val_acc.append(val_correct / val_total)
        
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

    pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
    pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
    pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

    torch.save(trained_model.state_dict(), pth_state)
    with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
    with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
    _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

