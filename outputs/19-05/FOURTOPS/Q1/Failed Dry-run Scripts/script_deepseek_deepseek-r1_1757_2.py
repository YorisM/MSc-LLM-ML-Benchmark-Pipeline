
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
from sklearn.metrics import roc_auc_score

class MyPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X, y=None):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        
        # Calculate mean and std for E_T_miss and pT components
        non_obj_features = X[:, :2]
        pT_features = []
        for i in range(0, 90, 5):
            if i+3 < X.shape[1]:
                pT_features.append(X[:, i+3])
        pT_stack = np.concatenate(pT_features, axis=0)
        
        self.mean = np.mean(np.concatenate([non_obj_features.flatten(), pT_stack]), axis=0)
        self.std = np.std(np.concatenate([non_obj_features.flatten(), pT_stack]), axis=0)
        return self
    
    def transform(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        
        X_trans = X.copy()
        X_trans[:, 0] = (X_trans[:, 0] - self.mean) / (self.std + 1e-8)
        X_trans[:, 1] = (X_trans[:, 1] - 0) / (2*np.pi)  # Phi normalization
        
        # Object feature normalization
        for i in range(0, 90, 5):
            if i+3 < X.shape[1]:
                X_trans[:, i+3] = (X_trans[:, i+3] - self.mean) / (self.std + 1e-8)
            if i+4 < X.shape[1]:
                X_trans[:, i+4] = (X_trans[:, i+4] - (-5)) / (5 - (-5))  # Eta normalization [-5,5]
        
        return torch.tensor(X_trans, dtype=torch.float32)

def make_preprocessor():
    return MyPreprocessor()

class TTbarClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding = nn.Embedding(100, 4)  # Object type embedding
        self.rnn = nn.GRU(4+3, 32, batch_first=True)  # +3 features (pT, eta, phi)
        self.classifier = nn.Sequential(
            nn.Linear(32 + 2 + input_dim, 64),  # Combine event-level and RNN features
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # Event-level features (E_T_miss and phi)
        event_feats = x[:, :2]
        
        # Process objects
        objects = x[:, 2:].view(x.size(0), 18, 5)
        
        # Extract object type (long tensor)
        obj_type = objects[:, :, 0].long()
        obj_features = self.embedding(obj_type)
        
        # Concatenate kinematic features (pT, eta, phi)
        kinematic = torch.stack([objects[:, :, 2], objects[:, :, 3], objects[:, :, 4]], dim=2)
        obj_full = torch.cat([obj_features, kinematic], dim=2)
        
        # Process with GRU
        _, hidden = self.rnn(obj_full)
        rnn_out = hidden[-1]
        
        # Combine with event features and other info
        combined = torch.cat([event_feats, rnn_out], dim=1)
        return self.classifier(combined)


def make_model(input_dim: int):
    return TTbarClassifier(input_dim)

EPOCHS = 15

def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.float())
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_train_loss = running_loss / len(train_loader)
        train_acc_val = correct / total
        train_loss.append(avg_train_loss)
        train_acc.append(train_acc_val)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        y_true = []
        y_score = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels.float())
                val_running_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).long()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                y_true.extend(labels.cpu().numpy())
                y_score.extend(torch.sigmoid(outputs).cpu().numpy())
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_acc_val = val_correct / val_total
        val_loss.append(avg_val_loss)
        val_acc.append(val_acc_val)
        
        # Calculate validation AUC
        auc = roc_auc_score(y_true, y_score)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f}, Acc: {train_acc_val:.4f}/{val_acc_val:.4f}, Val AUC: {auc:.4f}')
        
        scheduler.step(avg_val_loss)
    
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

