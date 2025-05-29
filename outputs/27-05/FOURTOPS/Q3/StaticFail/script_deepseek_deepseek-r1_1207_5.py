
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
from sklearn.preprocessing import StandardScaler
from functools import lru_cache

class MyPreprocessor:
    def __init__(self):
        self.e_t_miss_scaler = StandardScaler()
        self.obj_scalers = {'E': StandardScaler(),
                           'pT': StandardScaler(),
                           'eta': StandardScaler()}
        self.obj_col_indices = [(2+i*5, 7+i*5) for i in range(18)]
        self.max_obj_type = 0

    def fit(self, X, y=None):
        
        # Fit scalers for ET_miss
        self.e_t_miss_scaler.fit(X[:,0:1].numpy())
        
        # Collect object features
        all_obj_features = []
        obj_types = []
        for evt in X:
            for j in range(len(self.obj_col_indices)):
                start = 2 + j*5
                obj_type = evt[start].item()
                if obj_type > 0: # Valid object
                    features = evt[start+1:start+4].numpy() # E, pT, eta
                    all_obj_features.append(features)
                    obj_types.append(obj_type)
        
        # Fit scalers for object features
        all_obj_features = np.array(all_obj_features)
        for i,key in enumerate(self.obj_scalers.keys()):
            self.obj_scalers[key].fit(all_obj_features[:, [i]])
        
        self.max_obj_type = int(np.max(obj_types)) if obj_types else 0
        return self

    def transform(self, X):
        # Process global features
        et_miss = self.e_t_miss_scaler.transform(X[:,0:1].numpy())
        phi_et_miss = X[:,1]
        
        global_features = np.concatenate([
            et_miss,
            np.sin(phi_et_miss).unsqueeze(-1).numpy(),
            np.cos(phi_et_miss).unsqueeze(-1).numpy()
        ], axis=1)

        # Process object features
        obj_features = []
        for evt in X:
            features = []
            for j in range(len(self.obj_col_indices)):
                start = 2 + j*5
                obj_type = evt[start].item()
                
                # Base features: type + normalized features
                if obj_type ==0:
                    obj_vec = np.zeros(6) # [type,0,0,0,0,0]
                else:
                    E = self.obj_scalers['E'].transform([[evt[start+1].item()]])[0][0]
                    pT = self.obj_scalers['pT'].transform([[evt[start+2].item()]])[0][0]
                    eta = self.obj_scalers['eta'].transform([[evt[start+3].item()]])[0][0]
                    phi = evt[start+4].item()
                    
                    obj_vec = [
                        obj_type,
                        E,
                        pT,
                        eta,
                        np.sin(phi),
                        np.cos(phi)
                    ]
                features.extend(obj_vec)
            obj_features.append(features)
        
        np_global = np.array(global_features).astype(np.float32)
        np_objs = np.array(obj_features).astype(np.float32)
        transformed = np.concatenate([np_global, np_objs], axis=1)
        return torch.from_numpy(transformed)

def make_preprocessor():
    return MyPreprocessor()

class SlotAttention(nn.Module):
    def __init__(self, input_dim, n_slots=4, hidden_dim=64, n_iters=4):
        super().__init__()
        self.n_iters = n_iters
        self.n_slots = n_slots
        
        self.slots_mu = nn.Parameter(torch.randn(1, n_slots, input_dim))
        self.slots_log_var = nn.Parameter(torch.zeros(1, n_slots, input_dim))
        
        self.project_q = nn.Linear(input_dim, hidden_dim)
        self.project_k = nn.Linear(input_dim, hidden_dim)
        self.project_v = nn.Linear(input_dim, hidden_dim)
        
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
    def forward(self, x, mask):
        
        batch_size, n_objects, _ = x.shape
        
        k = self.project_k(x)
        v = self.project_v(x)
        
        # Initialize slots
        slots = self.slots_mu.expand(batch_size, -1, -1)
        
        # Mask states: [B,N]
        for _ in range(self.n_iters):
            slots_prev = slots
            
            # Compute attention
            q = self.project_q(slots)
            attn_logits = torch.einsum('bsd,bnd->bsn', q, k)
            
            # Apply mask: [B,S,N]
            attn_logits = attn_logits.masked_fill(~mask.unsqueeze(1), -1e9)
            attn_weights = torch.softmax(attn_logits, dim=-1)
            
            # Weighted sum of values
            updates = torch.einsum('bsn,bnd->bsd', atten_weights, v)
            
            # Update slots
            slots = self.gru(updates.view(-1, updates.size(-1)),
                            slots_prev.view(-1, slots_prev.size(-1))).view(batch_size, self.n_slts, -1)
        
        return slots

class PhysicsFormer(nn.Module):
    def __init__(self, input_dim, max_obj_type=52):
        super().__init__()
        self.obj_embed = nn.Embedding(max_obj_type+1, 16)
        
        self.global_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        
        self.slot_attention = SlotAttention(input_dim=38, n_slots=4)
        
        self.final_classifier = nn.Sequential(
            nn.Linear(4*64 +64, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        global_feats = x[:, :3]
        obj_feats = x[:,3:].view(x.size(0), 18, 6)
        
        # Extract object details
        obj_types = obj_feats[:,:,0].long()
        nums_mask = (obj_types !=0)
        
        # Object embedding
        embed = self.obj_embed(obj_types) # [B,18,16]
        numeric_feats = obj_feats[:, :,1:] # [B,18,5]
        obj_combined = torch.cat([embed, numeric_feats], dim=-1)
        
        # Slot attention grouping
        slots = self.slot_attention(obj_combined, nums_mask) # [B,4,64]
        slots_agg = slots.flatten(1)
        
        # Global processing
        global_processed = self.global_net(global_feats)
        
        combined = torch.cat([slots_agg, global_processed], dim=-1)
        return self.final_classifier(combined).squeeze(-1)

def make_model(input_dim: int):
    return PhysicsFormer(input_dim)

EPOCHS = 15

def train_model(model, train_loader, val_Loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(torch.float32)
            targets = targets.to(torch.float32)
            
            output = model(x)
            loss = criterion(output, targets.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = (output > 0).long()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        
        train_loss.append(epoch_loss/len(train_loader))
        train_acc.append(correct/total)
        
        # Validation phase
        model.eval()
        val_epoch_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.float()
                targets = targets.float()
                
                output = model(inputs)
                loss = criterion(output, targets)
                
                val_epoch_loss += loss.item()
                preds = (output > 0).long()
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
        
        val_loss.append(val_epoch_loss/len(val_loader))
        val_acc.append(val_correct/val_total)
    
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

