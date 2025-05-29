
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
import os, sys, json, pickle, torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class MyPreprocessor:
    def __init__(self):
        self.etmiss_mean = 0.0
        self.etmiss_std = 1.0
        self.phi_etmiss_mean = 0.0
        self.phi_etmiss_std = 1.0
        self.E_mean = 0.0
        self.E_std = 1.0
        self.pT_mean = 0.0
        self.pT_std = 1.0
        self.eta_mean = 0.0
        self.eta_std = 1.0
        self.phi_mean = 0.0
        self.phi_std = 1.0

    def fit(self, X, y=None):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X.copy()
        
        # Global features
        self.etmiss_mean = X_np[:, 0].mean()
        self.etmiss_std = X_np[:, 0].std()
        self.phi_etmiss_mean = X_np[:, 1].mean()
        self.phi_etmiss_std = X_np[:, 1].std()
        
        # Object features
        X_obj = X_np[:, 2:].reshape(-1, 18, 5)
        pT_values = X_obj[:, :, 2].flatten()
        mask = pT_values > 0
        
        E_all = X_obj[:, :, 1].flatten()[mask]
        pT_all = pT_values[mask]
        eta_all = X_obj[:, :, 3].flatten()[mask]
        phi_all = X_obj[:, :, 4].flatten()[mask]
        
        self.E_mean = E_all.mean()
        self.E_std = E_all.std()
        self.pT_mean = pT_all.mean()
        self.pT_std = pT_all.std()
        self.eta_mean = eta_all.mean()
        self.eta_std = eta_all.std()
        self.phi_mean = phi_all.mean()
        self.phi_std = phi_all.std()
        
        # Avoid division by zero
        for attr in ['etmiss_std', 'phi_etmiss_std', 'E_std', 'pT_std', 'eta_std', 'phi_std']:
            setattr(self, attr, max(getattr(self, attr), 1e-8))
        
        return self

    def transform(self, X):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X.copy()
        
        # Normalize global features
        X_np[:, 0] = (X_np[:, 0] - self.etmiss_mean) / self.etmiss_std
        X_np[:, 1] = (X_np[:, 1] - self.phi_etmiss_mean) / self.phi_etmiss_std
        
        # Normalize object features
        X_obj = X_np[:, 2:].reshape(-1, 18, 5)
        X_obj[:, :, 1] = (X_obj[:, :, 1] - self.E_mean) / self.E_std
        X_obj[:, :, 2] = (X_obj[:, :, 2] - self.pT_mean) / self.pT_std
        X_obj[:, :, 3] = (X_obj[:, :, 3] - self.eta_mean) / self.eta_std
        X_obj[:, :, 4] = (X_obj[:, :, 4] - self.phi_mean) / self.phi_std
        
        X_np[:, 2:] = X_obj.reshape(-1, 90)
        return torch.tensor(X_np, dtype=torch.float32)

def make_preprocessor():
    return MyPreprocessor()

class ParticleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_embedding = nn.Embedding(100, 8)
        self.obj_mlp = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(34, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # Global features
        global_features = x[:, :2]
        
        # Object processing
        objects = x[:, 2:].view(-1, 18, 5)
        obj_type = objects[:, :, 0].long()
        features = objects[:, :, 1:]
        
        mask = objects[:, :, 2] > 0
        obj_embed = self.obj_embedding(obj_type)
        combined = torch.cat([obj_embed, features], dim=-1)
        
        obj_processed = self.obj_mlp(combined)
        obj_processed = obj_processed * mask.unsqueeze(-1)
        aggregated = obj_processed.sum(dim=1)
        
        # Combine and classify
        combined_features = torch.cat([global_features, aggregated], dim=1)
        return self.global_mlp(combined_features).squeeze(-1)

def make_model(input_dim: int):
    return ParticleClassifier()

EPOCHS = 10

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)
        
        train_loss.append(epoch_loss / total)
        train_acc.append(correct / total)
        
        # Validation
        model.eval()
        val_epoch_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_epoch_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).long()
                val_correct += (preds == labels.long()).sum().item()
                val_total += labels.size(0)
        
        val_loss.append(val_epoch_loss / val_total)
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

