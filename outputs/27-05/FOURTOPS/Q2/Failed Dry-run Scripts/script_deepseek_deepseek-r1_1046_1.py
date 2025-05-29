
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
        self.norm_params = None

    def fit(self, X, y=None):
        # Calculate normalization parameters using non-sparse features
        X_flat = X.reshape(-1, 5)
        active = X_flat[:, 0] != 0  # Identify non-zero-padded objects
        means = torch.mean(X_flat[active, 1:], dim=0)
        stds = torch.std(X_flat[active, 1:], dim=0)
        self.norm_params = (means, stds)
        return self

    def transform(self, X):
        # Reshape to (batch, 18 objects, 5 features)
        X_objects = X.view(-1, 18, 5)
        
        # Separate ET_miss and phi
        et_miss = X_objects[:, 0, 0].unsqueeze(1)  # (N, 1)
        phi_miss = X_objects[:, 0, 1].unsqueeze(1)
        
        # Process objects
        objects = X_objects[:, 1:, :]  # Shape (N, 17, 5)
        mask = objects[:, :, 0] != 0  # Non-padded objects
        valid_objects = objects[mask]  # (total_valid, 5)
        
        # Normalize energy-related features
        valid_objects[:, 1:] = (valid_objects[:, 1:] - self.norm_params[0]) / (self.norm_params[1] + 1e-8)
        objects[mask] = valid_objects
        
        # Calculate invariant mass for each pair
        mass_features = []
        for evt in objects:
            valid = evt[:, 0] != 0
            pts = evt[valid, 2]
            etas = evt[valid, 3]
            phis = evt[valid, 4]
            px = pts * torch.cos(phis)
            py = pts * torch.sin(phis)
            pz = pts * torch.sinh(etas)
            energy = torch.sqrt(px**2 + py**2 + pz**2 + 0.938**2)  # Assume proton mass
            
            # Pairwise invariant masses
            indices = torch.combinations(torch.arange(energy.size(0)), 2)
            if indices.size(0) == 0:
                mass = torch.zeros(1)
            else:
                e = energy[indices].sum(dim=1)
                px_sum = px[indices].sum(dim=1)
                py_sum = py[indices].sum(dim=1)
                pz_sum = pz[indices].sum(dim=1)
                mass = torch.sqrt(e**2 - (px_sum**2 + py_sum**2 + pz_sum**2))
            mass_features.append(mass.mean().unsqueeze(0))
        
        mass_tensor = torch.stack(mass_features)
        
        # Combine features
        processed = torch.cat([et_miss, phi_miss, objects.flatten(1), mass_tensor], dim=1)
        return processed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

class LorentzEquivariantBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.message = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        
    def forward(self, x, edges):
        # x shape: (batch, nodes, in_dim)
        # edges: node adjacency indices
        messages = torch.zeros_like(x)
        for src, dst in edges:
            m = self.message(torch.cat([x[:, src], x[:, dst]], dim=-1))
            messages[:, dst] += m
        x = self.linear(x) + messages
        return nn.functional.relu(x)

class FourTopModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embed = nn.Linear(5, 64)
        self.blocks = nn.ModuleList([
            LorentzEquivariantBlock(64, 64) for _ in range(3)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64 + 2, 256),  # Includes global features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        # Split input into objects and global features
        et_miss = x[:, 0]
        phi_miss = x[:, 1]
        objects = x[:, 2:-1].view(x.size(0), 17, 5)
        
        # Embed object features
        x_emb = self.embed(objects)
        
        # Simulate message passing between detected objects
        # Using fully-connected edges for demonstration
        nodes = x_emb.size(1)
        edges = torch.combinations(torch.arange(nodes), 2)
        edges = [(i,j) for i,j in edges] + [(j,i) for i,j in edges]
        
        for block in self.blocks:
            x_emb = block(x_emb, edges)
        
        # Pool per-object features
        global_feat = self.pool(x_emb.permute(0,2,1)).squeeze()
        combined = torch.cat([global_feat, et_miss.unsqueeze(1), phi_miss.unsqueeze(1)], dim=1)
        
        return self.classifier(combined).squeeze()

def make_model(input_dim: int):
    return FourTopModel(input_dim)

EPOCHS = 20
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    
    best_auc = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = (outputs > 0).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_loss.append(epoch_loss / len(train_loader))
        train_acc.append(correct / total)
        
        # Validation phase
        model.eval()
        val_epoch_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.float().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_epoch_loss += loss.item()
                preds = (outputs > 0).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.append(outputs.sigmoid().cpu())
                all_labels.append(labels.cpu())
        
        val_loss.append(val_epoch_loss / len(val_loader))
        val_acc.append(val_correct / val_total)
        auc = roc_auc_score(torch.cat(all_labels), torch.cat(all_preds))
        
        scheduler.step(val_epoch_loss)
        
        # Save best model based on AUC
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'best_model.pth')
    
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
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

