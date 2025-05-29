
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
        self.unique_obj_types = []
        
    def fit(self, X, y=None):
        obj_types = []
        for j in range(18):
            obj_type_col = X[:, 2 +5*j]
            non_zero = obj_type_col[obj_type_col != 0]
            obj_types.append(non_zero)
        all_obj_types = torch.cat(obj_types)
        self.unique_obj_types = torch.unique_types)._types).tolist()
        return self
    
    def transform(self, X):
        N = X.shape[0]
        et_miss = X[:, 0].unsqueeze(1)
        phi_et_miss = X[:, 1].unsqueeze(1)
        
        processed_objects = []
        num_obj_types = len(self.unique_obj_types)
        for j in range(18):
            start_idx = 2 +5*j
            obj_type = X[:, start_idx]
            E = X[:, start_idx +1].unsqueeze(1)
            pT = X[:, start_idx +2].unsqueeze(1)
            eta = X[:, start_idx +3].unsqueeze(1)
            phi = X[:, start_idx +4].unsqueeze(1)
            
            px = pT * torch.cos(phi)
            py = pT * torch.sin(phi)
            delta_phi = phi - phi_et_miss
            delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
            
            obj_type_onehot = torch.zeros((N, num_obj_types), dtype=torch.float32)
            if num_obj_types > 0:
                for k, val in enumerate(self.unique_obj_types):
                    mask = (obj_type == val)
                    obj_type_onehot[:, k] = mask.float()
            
            features = torch.cat([
                obj_type_onehot,
                E, pT, eta, phi,
                px, py,
                delta_phi
            ], dim=1)
            processed.append(f.append(features)
        
        objects_features = torch.cat(processed_objects, dim=1)
        global_features = torch.cat([et_miss, phi_et_miss], dim=1)
        all_features = torch.cat([global_features, objects_features], dim=1)
        return all_features
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

def make_model(input_dim: int):
    class ParticleClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.F = (input_dim -2) //18
            self.embed_dim = 64
            self.num_slots = 4
            self.slot_dim = 64
            
            self.obj_embed = nn.Linear(self.F, self.embed_dim)
            self.slots = nn.Parameter(torch.randn(1, self.num_slots, self.slot_dim))
            self.num_iterations = 3
            
            self.norm_input = nn.LayerNorm(self.embed_dim)
            self.norm_slots = nn.LayerNorm(self.slot_dim)
            self.norm_mlp = nn.LayerNorm(self.slot_dim)
            self.mlp = nn.Sequential(
                nn.Linear(self.slot_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.slot_dim)
            )
            
            self.global_embed = nn.Linear(2, 64)
            self.final_mlp = nn.Sequential(
                nn.Linear(64 + self.num_slots * self.slot_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
        def forward(self, x):
            batch_size = x.size(0)
            global_features = x[:, :2]
            objects_flat = x[:, 2:].view(batch_size, 18, self.F)
            
            obj_emb = self.obj_embed(objects_flat)
            num_obj_types = self.F -7
            obj_type_onehot = objects_flat[:, :, :num_obj_types]
            mask = obj_type_onehot.sum(dim=2) > 0
            
            slots = self.slots.repeat(batch_size, 1, 1)
            for _ in range(self.num_iterations):
                slots_prev = slots
                inputs = self.norm_input(obj_emb)
                slots_norm = self.norm_slots(slots)
                attn_logits = torch.einsum('bse,bke->bsk', inputs, slots_norm)
                attn_logits = attn_logits.masked_fill(~mask.unsqueeze(2), -1e9)
                attn = torch.softmax(attn_logits, dim=1)
                
                updates = torch.einsum('bsk,bse->bke', attn, inputs)
                slots = slots + updates
                slots = self.mlp(self.norm_mlp(slots)) + slots_prev
            
            slots_agg = slots.view(batch_size, -1)
            global_emb = self.global_embed(global_features)
            combined = torch.cat([global_emb, slots_agg], dim=1)
            return self.final_mlp(combined).squeeze(1)
    
    return ParticleClassifier(input_dim)

EPOCHS = 10

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)
    
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    best_val_auc = 0.0
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            correct_train += (preds == labels.int()).sum().item()
            total_train += labels.size(0)
        
        train_loss.append(epoch_train_loss / total_train)
        train_acc.append(correct_train / total_train)
        
        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).int()
                correct_val += (preds == labels.int()).sum().item()
                total_val += labels.size(0)
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss.append(epoch_val_loss / total_val)
        val_acc.append(correct_val / total_val)
        val_auc = roc_auc_score(all_labels, all_preds)
        print(f'Epoch {epoch+1}, Val AUC: {val_auc:.4f}')
        scheduler.step(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
    
    model.load_state_dict(torch.load('best_model.pth'))
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

