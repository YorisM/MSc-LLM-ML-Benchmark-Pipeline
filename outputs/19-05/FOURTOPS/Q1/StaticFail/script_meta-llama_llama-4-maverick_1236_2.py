
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

EPOCHS = 10
                        
def load_data():
    to_np = lambda path: pd.read_csv(path).values
    X_train = to_np(DATASET["X_train"])
    Y_train = to_np(DATASET["Y_train"]).ravel()
    X_val   = to_np(DATASET["X_val"])
    Y_val   = to_np(DATASET["Y_val"]).ravel()
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
from sklearn.metrics import roc_auc_score, accuracy_score

class MyPreprocessor:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Reshape the data to separate objects and their features
        X = X.reshape(X.shape[0], -1, 5)
        # Extract relevant features
        obj_ids = X[:, :, 0]
        energies = X[:, :, 1]
        pts = X[:, :, 2]
        etas = X[:, :, 3]
        phis = X[:, :, 4]
        # Calculate some high-level features
        total_energy = np.sum(energies, axis=1)
        total_pt = np.sum(pts, axis=1)
        missing_et = X[:, 0, 0]  # E_T_miss is at index 0
        missing_et_phi = X[:, 0, 1]  # phi_{E_t}_miss is at index 1
        features = np.stack([total_energy, total_pt, missing_et, missing_et_phi], axis=1)
        return torch.tensor(features, dtype=torch.float32)

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def make_model(input_dim: int):
    return BinaryClassifier(input_dim)

EPOCHS = 10

def train_model(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, epochs: int):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0, 0
        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += ((outputs.squeeze() > 0.5) == labels).sum().item()
        train_loss.append(total_loss / len(train_loader))
        train_acc.append(total_correct / len(train_loader.dataset))
        model.eval()
        val_loss_epoch, val_acc_epoch = 0, 0
        with torch.no_grad():
            total_correct = 0
            for batch in val_loader:
                inputs, labels = batch
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                val_loss_epoch += loss.item()
                total_correct += ((outputs.squeeze() > 0.5) == labels).sum().item()
            val_loss.append(val_loss_epoch / len(val_loader))
            val_acc.append(total_correct / len(val_loader.dataset))
    return model, train_loss, val_loss, train_acc, val_acc

def main(X_train, Y_train, X_val, Y_val):
    preprocessor = make_preprocessor()
    X_train = preprocessor.fit_transform(X_train.numpy())
    X_val = preprocessor.transform(X_val.numpy())
    train_dataset = TensorDataset(X_train, torch.tensor(Y_train))
    val_dataset = TensorDataset(X_val, torch.tensor(Y_val))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    model = make_model(X_train.shape[1])
    trained_model, train_loss, val_loss, train_acc, val_acc = train_model(model, train_loader, val_loader, EPOCHS)
    Y_pred = trained_model(X_val).squeeze().detach().numpy()
    auc = roc_auc_score(Y_val, Y_pred)
    return auc

# Assuming X_train, Y_train, X_val, Y_val are defined and loaded
# auc = main(X_train, Y_train, X_val, Y_val)
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

