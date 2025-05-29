
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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

class MyPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X, y=None):
        # Compute mean and std only for non-zero features (E, p_T, eta, phi)
        non_zeros = X != 0
        valid_feats = X[:, 2:].masked_select(non_zeros[:, 2:].bool()).view(-1, 4)
        self.mean = torch.mean(valid_feats, dim=0)
        self.std = torch.std(valid_feats, dim=0)
        # For E_T_miss and phi, use existing values
        et_miss_mean = torch.mean(X[:, 0])
        et_miss_std = torch.std(X[:, 0])
        phi_mean = torch.mean(X[:, 1])
        phi_std = torch.std(X[:, 1])
        full_mean = torch.cat([et_miss_mean.unsqueeze(0), phi_mean.unsqueeze(0), self.mean])
        full_std = torch.cat([et_miss_std.unsqueeze(0), phi_std.unsqueeze(0), self.std])
        self.mean = full_mean
        self.std = full_std
        return self
    
    def transform(self, X):
        # Apply standardization
        X_norm = (X - self.mean) / (self.std + 1e-8)
        # Replace zero-padded objects with zeros (after normalization)
        mask = (X != 0).float()
        X_norm = X_norm * mask
        # Feature engineering: jet mass (assuming E^2 = p_T^2 + m^2)
        # For each object, m = sqrt(E^2 - p_T^2) (ignoring eta/phi for 3D momentum)
        # This approximation is physics-informed
        n_objects = 18
        features = []
        for i in range(n_objects):
            start_idx = 2 + i*5
            E = X[:, start_idx + 1]
            pT = X[:, start_idx + 2]
            m = torch.sqrt(E.abs()**2 - pT.abs()**2).unsqueeze(-1)
            features.append(m)
        mass_feat = torch.cat(features, dim=1)
        return torch.cat([X_norm, mass_feat], dim=1)


def make_model(input_dim: int):
    class SlotAttention(nn.Module):
        def __init__(self, num_slots=4, dim=64, iters=3):
            super().__init__()
            self.num_slots = num_slots
            self.iters = iters
            self.dim = dim
            
            self.to_q = nn.Linear(dim, dim)
            self.to_k = nn.Linear(dim, dim)
            self.to_v = nn.Linear(dim, dim)
            self.gru = nn.GRUCell(dim, dim)
            
            self.norm_input = nn.LayerNorm(dim)
            self.norm_slots = nn.LayerNorm(dim)
            
        def forward(self, inputs):
            b, n, d = inputs.shape
            slots = torch.randn((b, self.num_slots, self.dim)).to(inputs.device)
            
            for _ in range(self.iters):
                slots_prev = slots
                slots = self.norm_slots(slots)
                
                q = self.to_q(slots)  # [b, num_slots, dim]
                k = self.to_k(inputs)  # [b, n, dim]
                v = self.to_v(inputs)  # [b, n, dim]
                
                attn = torch.einsum('bqd,bnd->bqn', q, k) / np.sqrt(self.dim)
                attn = F.softmax(attn, dim=-1)  # [b, num_slots, n]
                
                updates = torch.einsum('bqn,bnd->bqd', attn, v)  # [b, num_slots, dim]
                
                # GRU update
                slots = self.gru(
                    updates.reshape(-1, self.dim),
                    slots_prev.reshape(-1, self.dim)
                ).reshape(b, self.num_slots, self.dim)
            
            return slots
    
    class ParticleTransformer(nn.Module):
        def __init__(self, input_features):
            super().__init__()
            self.embed = nn.Linear(input_features, 64)
            self.slot_attention = SlotAttention(num_slots=4)
            self.mlp = nn.Sequential(
                nn.Linear(64*4, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            orig_shape = x.shape
            # Mask zero-padded particles
            mask = (x[:, :, ::5] != 0).any(dim=-1)  # Check if any feature in the particle is non-zero
            
            # Embed particles
            x_emb = self.embed(x.view(-1, input_dim)).view(orig_shape[0], -1, 64)
            
            slots = self.slot_attention(x_emb)
            
            # Aggregate slot features
            slots_flat = slots.view(orig_shape[0], -1)
            
            return self.mlp(slots_flat)
    
    return ParticleTransformer(input_dim)

EPOCHS = 30

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    
    best_val_auc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.float().to(device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out.squeeze(), y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            preds = (torch.sigmoid(out) > 0.5).float()
            correct_train += (preds.squeeze() == y).sum().item()
            total_train += y.size(0)
        
        train_loss = epoch_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        val_loss = 0
        correct_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.float().to(device)
                out = model(x)
                loss = criterion(out.squeeze(), y)
                val_loss += loss.item()
                preds = (torch.sigmoid(out) > 0.5).float()
                correct_val += (preds.squeeze() == y).sum().item()
                val_preds.append(torch.sigmoid(out).cpu())
                val_true.append(y.cpu())
        val_loss /= len(val_loader)
        val_acc = correct_val / Y_val.size(0)
        
        val_auc = roc_auc_score(torch.cat(val_true).numpy(), torch.cat(val_preds).numpy())
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            # No model saving to adhere to size constraints
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
    
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

