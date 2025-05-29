
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
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        N = X.shape[0]
        global_feats = X[:, :2]
        objects = X[:, 2:].view(N, 18, 5)
        
        pT = objects[:, :, 2]
        eta = objects[:, :, 3]
        phi = objects[:, :, 4]
        
        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)
        log_pT = torch.log(pT + 1.0)
        
        processed_objects = torch.cat([
            objects,
            px.unsqueeze(-1),
            py.unsqueeze(-1),
            pz.unsqueeze(-1),
            log_pT.unsqueeze(-1)
        ], dim=-1)
        
        processed_objects_flat = processed_objects.view(N, -1)
        processed_X = torch.cat([processed_objects_flat, global_feats], dim=1)
        return processed_X

def make_model(input_dim: int):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.obj_embed = nn.Linear(9, 64)
            self.num_slots = 4
            self.slot_dim = 64
            self.slots = nn.Parameter(torch.randn(1, self.num_slots, self.slot_dim))
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=256),
                num_layers=2
            )
            self.gru = nn.GRUCell(64, 64)
            self.mlp = nn.Sequential(
                nn.Linear(4*64 + 2, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.LayerNorm(128),
                nn.Linear(128, 1)
            )
        
        def forward(self, x):
            batch_size = x.size(0)
            objects_flat = x[:, :162]
            global_feats = x[:, 162:]
            objects = objects_flat.view(batch_size, 18, 9)
            obj_id = objects[:, :, 0]
            mask = (obj_id == 0)
            obj_emb = self.obj_embed(objects)
            obj_emb = obj_emb.permute(1, 0, 2)
            encoder_output = self.encoder(obj_emb, src_key_padding_mask=mask)
            encoder_output = encoder_output.permute(1, 0, 2)
            slots = self.slots.repeat(batch_size, 1, 1)
            for _ in range(3):
                attn_logits = torch.einsum('bsd,bod->bso', slots, encoder_output)
                attn = torch.softmax(attn_logits, dim=-1)
                updates = torch.einsum('bso,bod->bsd', attn, encoder_output)
                slots = self.gru(updates.view(-1, 64), slots.view(-1, 64)).view(batch_size, 4, 64)
            slots_agg = slots.view(batch_size, -1)
            combined = torch.cat([slots_agg, global_feats], dim=1)
            output = self.mlp(combined)
            return torch.sigmoid(output.squeeze())
    return Model()

EPOCHS = 20

def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)
    
    best_val_auc = 0
    best_model = model.state_dict()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X_batch.size(0)
            preds = (outputs > 0.5).int()
            correct_train += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        avg_train_loss = total_train_loss / total
        train_loss.append(avg_train_loss)
        train_acc.append(correct_train / total)
        
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                outputs = model(X_val_batch)
                loss = criterion(outputs, y_val_batch.float())
                total_val_loss += loss.item() * X_val_batch.size(0)
                preds = (outputs > 0.5).int()
                correct_val += (preds == y_val_batch).sum().item()
                total_val += y_val_batch.size(0)
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(y_val_batch.cpu().numpy())
        avg_val_loss = total_val_loss / total_val
        val_loss.append(avg_val_loss)
        val_acc.append(correct_val / total_val)
        auc = roc_auc_score(all_labels, all_outputs)
        scheduler.step(auc)
        
        if auc > best_val_auc:
            best_val_auc =_model_model_model = model.state_dict()
        
    model.load_state_dict(best_model)
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

