
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
import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F

class MyPreprocessor:
    def __init__(self):
        self.global_means_ = None
        self.global_stds_ = None
        self.obj_means_ = None
        self.obj_stds_ = None

    def fit(self, X, y=None):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        # Global features
        et = X_np[:, 0].flatten()
        phi = X_np[:, 1].flatten()
        # Object features
        all_e, all_pt, all_eta, all_phi = [], [], [], []
        
        for evt in X_np:
            objs = evt[2:].reshape(18,5)
            for obj in objs:
                if np.all(obj == 0): continue
                all_e.append(obj[1])
                all_pt.append(obj[2])
                all_eta.append(obj[3])
                all_phi.append(obj[4])
        
        self.global_means_ = [np.mean(et), np.mean(phi)]
        self.global_stds_ = [np.std(et), np.std(phi)]
        
        self.obj_means_ = [
            np.mean(all_e) if all_e else 0.,
            np.mean(all_pt) if all_pt else 0.,
            np.mean(all_eta) if all_eta else 0.,
            np.mean(all_phi) if all_phi else 0.
        ]
        self.obj_stds_ = [
            max(np.std(all_e),1e-8) if all_e else 1.,
            max(np.std(all_pt),1e-8) if all_pt else 1.,
            max(np.std(all_eta),1e-8) if all_eta else 1.,
            max(np.std(all_phi),1e-8) if all_phi else 1.
        ]
        return self
    
    def transform(self, X):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X.copy()
        X_np = X_np.astype(np.float32)
        # Process global
        X_np[:,0] = (X_np[:,0] - self.global_means_[0]) / self.global_stds_[0]
        X_np[:,1] = (X_np[:,1] - self.global_means_[1]) / self.global_stds_[1]        
        # Process objects
        objs = X_np[:,2:].reshape(-1,18,5)
        for evt in objs:
            for obj in evt:
                if np.all(obj == 0): continue
                obj[1] = (obj[1] - self.obj_means_[0]) / self.obj_stds_[0]
                obj[2] = (obj[2] - self.obj_means_[1]) / self.obj_stds_[1]
                obj[3] = (obj[3] - self.obj_means_[2]) / self.obj_stds_[2]
                obj[4] = (obj[4] - self.obj_means_[3]) / self.obj_stds_[3]
        X_np[:,2:] = objs.reshape(-1,90)
        return torch.from_numpy(X_np) if isinstance(X, torch.Tensor) else X_np

def make_preprocessor():
    return MyPreprocessor()

def make_model(input_dim: int):
    class ParticleClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100,8)  # Assume max obj_type <100
            self.obj_mlp = nn.Sequential(
                nn.Linear(12,64),    # embed(8) + 4 normalized features
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            self.global_mlp = nn.Sequential(
                nn.Linear(64+2,128),
                nn.ReLU(),
                nn.Linear(128,32),
                nn.ReLU(),
                nn.Linear(32,1)
            )
        
        def forward(self, x):
            # Global features
            global_feat = x[:, :2]
            # Objects
            objs = x[:,2:].view(-1,18,5)
            obj_types = objs[:,:,0].long()
            obj_conts = objs[:,:,1:5]
            
            embeds = self.embed(obj_types)
            obj_feats = torch.cat([embeds, obj_conts], dim=-1)
            obj_out = self.obj_mlp(obj_feats)
            agg = obj_out.max(dim=1)[0]
            
            combined = torch.cat([agg, global_feat], dim=1)
            return self.global_mlp(combined).squeeze()
    return ParticleClassifier()

EPOCHS = 15

def train_model(model, train_loader, val_loader, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total =0
        for x, y in train_loader:
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y.float())
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            correct += ((pred.sigmoid() >0.5).long() == y).sum().item()
            total += y.shape[0]
        
        train_loss.append(epoch_loss / len(train_loader))
        train_acc.append(correct / total)
        
        # Validation
        model.eval()
        valid_loss =0
        valid_correct=0
        valid_total=0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                loss = loss_fn(pred, y.float())
                valid_loss += loss.item()
                valid_correct += ((pred.sigmoid()>0.5).long() ==y).sum().item()
                valid_total += y.shape[0]
        val_loss.append(valid_loss / len(val_loader))
        val_acc.append(valid_correct / valid_total)
        
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

