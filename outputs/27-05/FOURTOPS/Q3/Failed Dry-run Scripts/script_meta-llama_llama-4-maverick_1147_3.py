
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

# 0. ---------- IMPORTS ----------
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.obj_start_idx = 2  # Starting index of object features
        self.obj_feat_size = 5  # Size of features for each object
        self.max_objects = 18  # Maximum number of objects

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Separate E_T_miss and phi_{E_t}_miss
        E_T_miss = X[:, 0].unsqueeze(1)  # Shape: (N, 1)
        phi_E_T_miss = X[:, 1].unsqueeze(1)  # Shape: (N, 1)
        obj_features = X[:, self.obj_start_idx:]  # Shape: (N, 90)
        obj_features = obj_features.view(-1, self.max_objects, self.obj_feat_size)  # Shape: (N, 18, 5)
        obj_ids = obj_features[:, :, 0].long()  # Shape: (N, 18)
        obj_E = obj_features[:, :, 1].unsqueeze(-1)  # Shape: (N, 18, 1)
        obj_pT = obj_features[:, :, 2].unsqueeze(-1)  # Shape: (N, 18, 1)
        obj_eta = obj_features[:, :, 3].unsqueeze(-1)  # Shape: (N, 18, 1)
        obj_phi = obj_features[:, :, 4].unsqueeze(-1)  # Shape: (N, 18, 1)
        # Create augmented features
        obj_px = obj_pT * torch.cos(obj_phi)  # Shape: (N, 18, 1)
        obj_py = obj_pT * torch.sin(obj_phi)  # Shape: (N, 18, 1)
        obj_pz = obj_pT * torch.sinh(obj_eta)  # Shape: (N, 18, 1)
        obj_mass = torch.sqrt(obj_E**2 - obj_px**2 - obj_py**2 - obj_pz**2)  # Shape: (N, 18, 1)
        # Concatenate the new features
        new_obj_features = torch.cat([obj_E, obj_px, obj_py, obj_pz, obj_mass], dim=-1)  # Shape: (N, 18, 5)
        # Concatenate E_T_miss and phi_{E_t}_miss with the new object features
        E_T_miss_x = E_T_miss * torch.cos(phi_E_T_miss)  # Shape: (N, 1)
        E_T_miss_y = E_T_miss * torch.sin(phi_E_T_miss)  # Shape: (N, 1)
        global_features = torch.cat([E_T_miss_x, E_T_miss_y], dim=1)  # Shape: (N, 2)
        return torch.cat([global_features, new_obj_features.flatten(1)], dim=1)  # Shape: (N, 2 + 18*5)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class SlotAttention(nn.Module):
    def __init__(self, num_slots, input_dim, slot_dim=64, iters=3):
        super().__init__
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.iters = iters
        self.slots_mu = nn.Parameter(torch.randn(1, 1, self.slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, self.slot_dim))
        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(self.slot_dim)
        self.norm_mlp = nn.LayerNorm(self.slot_dim)
        self.project_inputs = nn.Linear(input_dim, self.slot_dim, bias=False)
        self.scale = self.slot_dim ** -0.5
        self.mlp = nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim), nn.ReLU(), nn.Linear(self.slot_dim, self.slot_dim))

    def forward(self, inputs):
        inputs = self.norm_inputs(inputs)
        inputs = self.project_inputs(inputs)
        batch_size, num_inputs, _ = inputs.shape
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            attn_logits = torch.einsum('bid,bjd->bij', inputs, slots) * self.scale
            attn = attn_logits.softmax(dim=1)
            updates = torch.einsum('bid,bij->bjd', inputs, attn)
            slots = slots_prev + self.mlp(self.norm_mlp(updates))
        return slots.flatten(1)

def make_model(input_dim: int):
    return nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.LayerNorm(128),
                         SlotAttention(num_slots=4, input_dim=128, slot_dim=64, iters=3),
                         nn.Linear(256, 128), nn.ReLU(), nn.LayerNorm(128),
                         nn.Linear(128, 2))

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 10

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0, 0
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (outputs.argmax(-1) == labels).sum().item()
        train_loss.append(total_loss / len(train_loader))
        train_acc.append(total_correct / len(train_loader.dataset))
        model.eval()
        total_loss, total_correct = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                total_correct += (outputs.argmax(-1) == labels).sum().item()
        val_loss.append(total_loss / len(val_loader))
        val_acc.append(total_correct / len(val_loader.dataset))
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

