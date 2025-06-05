
import os, sys, pickle, torch, gc, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
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

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        if isinstance(self.x, (tuple, list)):
            return (tuple(t[idx] for t in self.x), self.y[idx])
        else:
            return (self.x[idx], self.y[idx])      

def make_loaders(X_train, Y_train, X_val, Y_val, batch=512):
    train_ds = PairDataset(X_train, Y_train)
    val_ds   = PairDataset(X_val , Y_val)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0))
                        
# ----------------  START OF LLM BLOCK  ----------------

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

class MyPreprocessor:
    def __init__(self):
        self.obj_id_list = None
        self.obj_id_to_idx = {}
        self.feature_means = None
        self.feature_stds = None

    def fit(self, X, y=None):
        # Collect object IDs and create mapping
        objects = X[:, 2:].view(-1, 18, 5)  # (N, 18, 5)
        obj_ids = objects[:, :, 0].flatten()
        # Filter out padding (0) and collect unique obj_ids
        valid_obj_ids = obj_ids[obj_ids != 0]
        unique_obj_ids = torch.unique(valid_obj_ids).tolist()
        self.obj_id_list = unique_obj_ids
        # Create mapping from obj_id to index starting from 1 (0 is reserved for padding)
        self.obj_id_to_idx = {oid: i+1 for i, oid in enumerate(unique_obj_ids)}

        # Collect all features from real objects (for normalization)
        all_features = []
        batch_size = X.size(0)
        for i in range(batch_size):
            event_objects = objects[i]  # (18, 5)
            event_valid_mask = event_objects[:, 0] != 0
            event_valid = event_objects[event_valid_mask]
            if event_valid.size(0) == 0:
                continue
            # Add missing ET features to each valid object
            missing_et = X[i, :2].unsqueeze(0).expand(event_valid.size(0), 2)
            event_features = torch.cat([event_valid, missing_et], dim=1)  # (n_objs, 7)
            all_features.append(event_features)
        all_features = torch.cat(all_features, dim=0).numpy()  # (M,7)

        # Compute normalization parameters
        self.feature_means = np.mean(all_features, axis=0)
        self.feature_stds = np.std(all_features, axis=0, ddof=1) + 1e-8  # avoid division by zero
        return self

    def transform(self, X):
        # Split missing ET and objects
        missing_et = X[:, :2]
        objects = X[:, 2:].view(-1, 18, 5)
        batch_size = objects.size(0)

        # Map obj_ids to indices
        obj_ids = objects[:, :, 0].flatten().numpy().astype(int)
        mapped_ids = np.array([self.obj_id_to_idx.get(oid, 0) for oid in obj_ids])
        mapped_ids = torch.tensor(mapped_ids).view(batch_size, 18)

        # Replace obj_ids with mapped indices
        objects_mapped = torch.zeros_like(objects)
        objects_mapped[:, :, 0] = mapped_ids
        objects_mapped[:, :, 1:] = objects[:, :, 1:]

        # Add missing ET features to each object
        missing_et_expanded = missing_et.view(batch_size, 1, 2).expand(-1, 18, 2)
        objects_with_et = torch.cat([objects_mapped, missing_et_expanded], dim=2)

        # Normalize features
        objects_with_et = objects_with_et.numpy()
        objects_with_et = (objects_with_et - self.feature_means) / self.feature_stds
        objects_normalized = torch.tensor(objects_with_et, dtype=torch.float32)

        # Create mask (valid objects have mapped_id > 0)
        mask = (mapped_ids > 0).float()  # (batch_size, 18)

        return (objects_normalized, mask)

def make_preprocessor():
    return MyPreprocessor()

class ParticleTransformer(nn.Module):
    def __init__(self, num_obj_ids, emb_dim=32, hidden_dim=128, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(num_obj_ids+1, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim + 6, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x, mask=None):
        # Extract obj_ids and other features
        obj_ids = x[:, :, 0].long()  # (B, 18)
        other_feats = x[:, :, 1:]     # (B, 18, 6)

        # Embed obj_ids and combine
        emb = self.embedding(obj_ids)  # (B, 18, emb_dim)
        combined = torch.cat([emb, other_feats], dim=2)  # (B,18, emb_dim+6)
        projected = self.linear(combined)  # (B,18, hidden_dim)

        # Transformer with padding mask (mask is 1 for valid)
        src_key_padding_mask = (~mask.bool()) if mask is not None else None
        transformed = self.transformer(
            projected, 
            src_key_padding_mask=src_key_padding_mask
        )

        # Pool valid elements
        if mask is not None:
            lengths = mask.sum(dim=1, keepdim=True)  # (B,1)
            sum_emb = (transformed * mask.unsqueeze(-1)).sum(dim=1)  # (B, hidden_dim)
            mean_emb = sum_emb / (lengths + 1e-8)
        else:
            mean_emb = torch.mean(transformed, dim=1)

        return torch.sigmoid(self.classifier(mean_emb).squeeze(-1))

def make_model(input_shape, *, use_mask=False):
    num_obj_ids = 30  # Should be inferred, but set based on observed data
    return ParticleTransformer(num_obj_ids)

EPOCHS = 30
BATCH_SIZE = 512

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    best_auc = 0
    train_loss = []
    val_loss = []
    train_auc = []
    val_auc = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = []
        epoch_preds = []
        epoch_labels = []

        for data in train_loader:
            if len(data) == 3:  # with sample weights
                (seq, mask), labels, weights = data
                weights = weights.to(device)
            else:
                (seq, mask), labels = data
                weights = None

            seq, mask, labels = seq.to(device), mask.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(seq, mask)
            loss = criterion(outputs, labels)
            if weights is not None:
                loss = (loss * weights).mean()
            loss.backward()
            optimizer.step()

            epoch_train_loss.append(loss.item())
            epoch_preds.append(outputs.detach().cpu())
            epoch_labels.append(labels.detach().cpu())

        train_loss_epoch = np.mean(epoch_train_loss)
        train_auc_epoch = roc_auc_score(
            torch.cat(epoch_labels).numpy(), 
            torch.cat(epoch_preds).numpy()
        )
        train_loss.append(train_loss_epoch)
        train_auc.append(train_auc_epoch)

        # Validation
        model.eval()
        epoch_val_loss = []
        epoch_val_preds = []
        epoch_val_labels = []
        with torch.no_grad():
            for data in val_loader:
                (seq, mask), labels = data
                seq, mask, labels = seq.to(device), mask.to(device), labels.to(device).float()

                outputs = model(seq, mask)
                loss = criterion(outputs, labels)

                epoch_val_loss.append(loss.item())
                epoch_val_preds.append(outputs.cpu())
                epoch_val_labels.append(labels.cpu())

        val_loss_epoch = np.mean(epoch_val_loss)
        val_auc_epoch = roc_auc_score(
            torch.cat(epoch_val_labels).numpy(),
            torch.cat(epoch_val_preds).numpy()
        )
        val_loss.append(val_loss_epoch)
        val_auc.append(val_auc_epoch)

        scheduler.step(val_auc_epoch)

        # Early stopping
        if val_auc_epoch > best_auc:
            best_auc = val_auc_epoch
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f} | Train AUC: {train_auc_epoch:.4f} | Val AUC: {val_auc_epoch:.4f}')

    # Load best model
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    return model, train_loss, val_loss, train_auc, val_auc

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
    pre = make_preprocessor().fit(X_train, Y_train)
    X_train = pre.transform(X_train) # may be Tensor or Tuple
    X_val   = pre.transform(X_val)
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val)

    # 2. Build model
    if isinstance(X_train, torch.Tensor):               # single-tensor case
        temp_ref    = X_train
        input_shape = temp_ref.shape[1:]                # e.g. (F,)
        use_mask    = False
    else:                                               # tuple => (data, mask)
        temp_ref    = X_train
        input_shape = temp_ref[0].shape[1:]             # e.g. (L, F)
        use_mask    = True                              
    model = make_model(input_shape, use_mask=use_mask)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
        toy_data = torch.zeros(8, *input_shape, dtype=torch.float32)
        if use_mask:
            toy_mask = torch.zeros(8, input_shape[0], dtype=torch.bool)
            toy_batch = (toy_data, toy_mask)
        else:
            toy_batch = toy_data

        toy_transformed = pre.transform(toy_batch)
        try:
            _ = trained_model(*toy_transformed) if isinstance(toy_transformed, (tuple, list)) \
                else trained_model(toy_transformed)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 5. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

    pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
    pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
    pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

    torch.save(trained_model.state_dict(), pth_state)
    with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
    with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

    # 6. Save plots
    _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
    _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

    # 7. Write JSON Summary
    if not dryrun: 
        summary = {
            "epochs": n_epochs,
            "train_loss": tr_loss   if tr_loss else None,
            "val_loss":   va_loss   if va_loss else None,
            "train_acc":  tr_acc    if tr_acc else None,
            "val_acc":    va_acc    if va_acc else None,
        }
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

