
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
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Reshape the data to separate objects and their features
        X = X.reshape(X.shape[0], -1, 5)
        # Separate object identifier and its kinematic properties
        obj_ids = X[:, :, 0]
        obj_features = X[:, :, 1:]
        # Concatenate E_T_miss and phi_{E_t}_miss with object features
        E_T_miss = X[:, 0, 0]  # Corrected index for E_T_miss
        phi_E_T_miss = X[:, 0, 1]  # Corrected index for phi_{E_t}_miss
        global_features = torch.stack([E_T_miss, phi_E_T_miss], dim=1)
        obj_features = torch.cat([global_features.unsqueeze(1), obj_features], dim=1)
        # Zero-out the features of padded objects
        obj_features[obj_ids == 0] = 0
        # Flatten the features
        X = obj_features.flatten(start_dim=1)
        return X

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def make_model(input_dim: int):
    model = BinaryClassifier(input_dim)
    return model

EPOCHS = 10

def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0, 0
        total_samples = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            predictions = torch.sigmoid(outputs.squeeze()) > 0.5
            total_correct += (predictions == y_batch).sum().item()
            total_samples += X_batch.size(0)
        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        model.eval()
        val_loss_epoch, val_acc_epoch = 0, 0
        total_samples = 0
        with torch.no_grad():
            total_val_loss, total_val_correct = 0, 0
            for X_val_batch, y_val_batch in val_loader:
                outputs = model(X_val_batch)
                loss = criterion(outputs.squeeze(), y_val_batch.float())
                total_val_loss += loss.item() * X_val_batch.size(0)
                predictions = torch.sigmoid(outputs.squeeze()) > 0.5
                total_val_correct += (predictions == y_val_batch).sum().item()
                total_samples += X_val_batch.size(0)
            val_loss_epoch = total_val_loss / total_samples
            val_acc_epoch = total_val_correct / total_samples
            val_loss.append(val_loss_epoch)
            val_acc.append(val_acc_epoch)
        print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.4f}')
    return model, train_loss, val_loss, train_acc, val_acc

def main():
    X_train = torch.load('X_train.pt')
    Y_train = torch.load('Y_train.pt')
    X_val = torch.load('X_val.pt')
    Y_val = torch.load('Y_val.pt')
    preprocessor = make_preprocessor()
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    model = make_model(X_train.shape[1])
    trained_model, train_loss, val_loss, train_acc, val_acc = train_model(model, train_loader, val_loader, EPOCHS)
    # Evaluate AUC on validation set
    trained_model.eval()
    with torch.no_grad():
        outputs = trained_model(X_val)
        predictions = torch.sigmoid(outputs.squeeze())
        auc = roc_auc_score(Y_val.numpy(), predictions.numpy())
        print(f'Validation AUC: {auc:.4f}')

if __name__ == '__main__':
    main()
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

