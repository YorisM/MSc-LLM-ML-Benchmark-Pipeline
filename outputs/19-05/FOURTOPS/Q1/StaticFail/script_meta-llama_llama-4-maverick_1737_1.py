
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
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

class MyPreprocessor:
    def __init__(self):
        self.obj_cols = [2 + 5*i for i in range(18)]
        self.E_cols = [3 + 5*i for i in range(18)]
        self.pT_cols = [4 + 5*i for i in range(18)]
        self.eta_cols = [5 + 5*i for i in range(18)]
        self.phi_cols = [6 + 5*i for i in range(18)]
        self.E_T_miss_col = 0
        self.phi_E_T_miss_col = 1

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Extract relevant features
        E_T_miss = X[:, self.E_T_miss_col].unsqueeze(1)
        phi_E_T_miss = X[:, self.phi_E_T_miss_col].unsqueeze(1)
        obj_ids = X[:, self.obj_cols]
        energies = X[:, self.E_cols]
        pTs = X[:, self.pT_cols]
        etas = X[:, self.eta_cols]
        phis = X[:, self.phi_cols]
        
        # Create new features
        n_objects = (obj_ids != 0).sum(dim=1, keepdim=True).float()
        sum_E = energies.sum(dim=1, keepdim=True)
        sum_pT = pTs.sum(dim=1, keepdim=True)
        
        # Concatenate features
        features = torch.cat([E_T_miss, phi_E_T_miss, n_objects, sum_E, sum_pT], dim=1)
        return features

def make_preprocessor():
    return MyPreprocessor()

def make_model(input_dim: int):
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    return model

EPOCHS = 10

def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    train_auc = []
    val_auc = []
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        epoch_train_correct = 0
        epoch_train_total = 0
        epoch_train_preds = []
        epoch_train_labels = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            preds = (outputs > 0.5).int()
            epoch_train_correct += (preds == y_batch.unsqueeze(1)).sum().item()
            epoch_train_total += y_batch.size(0)
            epoch_train_preds.extend(outputs.detach().cpu().numpy().flatten())
            epoch_train_labels.extend(y_batch.cpu().numpy())
        epoch_train_loss /= len(train_loader)
        epoch_train_acc = epoch_train_correct / epoch_train_total
        epoch_train_auc = roc_auc_score(epoch_train_labels, epoch_train_preds)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        train_auc.append(epoch_train_auc)
        
        model.eval()
        epoch_val_loss = 0
        epoch_val_correct = 0
        epoch_val_total = 0
        epoch_val_preds = []
        epoch_val_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1).float())
                epoch_val_loss += loss.item()
                preds = (outputs > 0.5).int()
                epoch_val_correct += (preds == y_batch.unsqueeze(1)).sum().item()
                epoch_val_total += y_batch.size(0)
                epoch_val_preds.extend(outputs.detach().cpu().numpy().flatten())
                epoch_val_labels.extend(y_batch.cpu().numpy())
        epoch_val_loss /= len(val_loader)
        epoch_val_acc = epoch_val_correct / epoch_val_total
        epoch_val_auc = roc_auc_score(epoch_val_labels, epoch_val_preds)
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        val_auc.append(epoch_val_auc)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Train AUC: {epoch_train_auc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val AUC: {epoch_val_auc:.4f}')
    return model, train_loss, val_loss, train_acc, val_acc

# Load data
X_train = torch.load('X_train.pt')
Y_train = torch.load('Y_train.pt')
X_val = torch.load('X_val.pt')
Y_val = torch.load('Y_val.pt')

# Preprocess data
preprocessor = make_preprocessor()
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)

# Create data loaders
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataset = TensorDataset(X_val, Y_val)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Train model
model = make_model(X_train.shape[1])
trained_model, train_loss, val_loss, train_acc, val_acc = train_model(model, train_loader, val_loader, EPOCHS)
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

