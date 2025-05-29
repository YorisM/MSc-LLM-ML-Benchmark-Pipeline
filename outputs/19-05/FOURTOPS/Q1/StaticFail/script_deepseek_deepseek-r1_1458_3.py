
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
        self.obj_channels = 5
        self.max_objects = 18
        self.valid_obj_ids = [4,5,6,7,8]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        
        batch_size = X.shape[0]
        processed = []
        
        # Extract global features
        et_miss = X[:, 0]
        phi_et_miss = X[:,1]
        
        # Process object features
        obj_features = []
        for i in range(self.max_objects):
            start = 2 + i*self.obj_channels
            end = start + self.obj_channels
            obj_slice = X[:, start:end]
            
            obj_id = obj_slice[:,0]
            mask = np.isin(obj_id, self.valid_obj_ids).astype(np.float32)
            
            pT = obj_slice[:,2] / 1000.0  # MeV to GeV
            eta = obj_slice[:,3]
            phi = obj_slice[:,4]
            
            # Create features for each object
            obj_features.append(np.column_stack([
                mask * pT,
                mask * eta,
                mask * phi,
                mask * (np.cos(phi - phi_et_miss[:, np.newaxis])),
                mask * (np.sqrt(pT) / (1 + np.abs(eta)))
            ]))
        
        # Combine all object features
        obj_features = np.concatenate(obj_features, axis=1)
        
        # Combine global and object features
        final_features = np.column_stack([
            et_miss / 1000.0,  # MeV to GeV
            np.cos(phi_et_miss),
            np.sin(phi_et_miss),
            obj_features
        ])
        
        return final_features.astype(np.float32)


def make_preprocessor():
    return MyPreprocessor()


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=1)
        return (x * weights).sum(dim=1), weights


class PhysicsModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = AttentionLayer(3)
        
        self.dense = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        pT = x[:, :18]  # First 18 features are pT
        eta = x[:, 18:36]  # Next 18 are eta
        phi = x[:, 36:54]  # Then phi
        
        # Attention on object pT
        attn_input = torch.stack([pT, eta, phi], dim=2)
        attn_output, _ = self.attention(attn_input)
        
        # Combine with global features
        main_input = torch.cat([x[:, 54:], attn_output], dim=1)
        
        return self.dense(main_input)


def make_model(input_dim: int):
    return PhysicsModel(input_dim)

EPOCHS = 12


def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    best_auc = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in train_loader:
            inputs = batch[0].float().to(next(model.parameters()).device)
            labels = batch[1].float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs).detach().cpu())
            all_labels.append(labels.cpu())
        
        # Train metrics
        avg_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_loss)
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        train_auc = roc_auc_score(all_labels.numpy(), all_preds.numpy())
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_epoch_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].float().to(next(model.parameters()).device)
                labels = batch[1].float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_epoch_loss += loss.item()
                val_preds.append(torch.sigmoid(outputs).cpu())
                val_labels.append(labels.cpu())
        
        # Val metrics
        avg_val_loss = val_epoch_loss / len(val_loader)
        val_loss.append(avg_val_loss)
        
        val_preds = torch.cat(val_preds)
        val_labels = torch.cat(val_labels)
        val_auc = roc_auc_score(val_labels.numpy(), val_preds.numpy())
        
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")
    
    # Load best model weights
    model.load_state_dict(torch.load('best_model.pth'))
    return model, train_loss, val_loss, [0]*epochs, [0]*epochs
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

