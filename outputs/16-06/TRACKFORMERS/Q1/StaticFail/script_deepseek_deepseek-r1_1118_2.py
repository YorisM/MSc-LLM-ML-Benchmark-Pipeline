
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: Python 3.12, PyTorch 2.6.0, NumPy 2.2.3, SciKit-Learn 1.6.1
import os, sys, pickle, gzip, json, torch, numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_DIR = "./challenges/TRACKFORMERS/data"
TAG      = "10_50_linear"

def _load_events(split: str):
    pkl = os.path.join(DATA_DIR, f"REDVID_{TAG}_{split}.pkl.gz")
    with gzip.open(pkl, "rb") as fh:
        return pickle.load(fh)["events"]

def split_X_y(evt):
    lay = evt["layer_id"].astype(np.float32)
    lay_norm = lay / lay.max()
    X = np.column_stack([evt["hit_r"],
                         evt["hit_theta"],
                         evt["hit_z"],
                         lay_norm])
    t_id = evt["track_id"].astype(np.int32)
    return (torch.from_numpy(X),
            torch.from_numpy(t_id))

class EventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, track_id = split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        return (X, track_id)

def _ragged(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    # batch[i] = (hits_i, track_id_i)      ← shapes: (N_i, F), (N_i,)
    return batch

def make_loaders(batch_size=128, workers=0):
    tr = EventDataset(_load_events("train"), pre=None, train=True)
    va = EventDataset(_load_events("val"),   pre=None, train=False)

    train_ld = DataLoader(tr, batch_size=batch_size, shuffle=True,
                          collate_fn=_ragged, num_workers=workers)
    val_ld   = DataLoader(va, batch_size=batch_size, collate_fn=_ragged)
    return train_ld, val_ld

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# 0. ---------- IMPORTS ----------
from sklearn.cluster import KMeans
from torch.optim import AdamW
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import numpy as np

# 1. ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.r_mean = None
        self.r_std = None
        self.theta_mean = None
        self.theta_std = None
        self.z_mean = None
        self.z_std = None
        self.layer_mean = None
        self.layer_std = None

    def fit(self, events):
        all_r = np.concatenate([evt['hit_r'] for evt in events])
        all_theta = np.concatenate([evt['hit_theta'] for evt in events])
        all_z = np.concatenate([evt['hit_z'] for evt in events])
        all_layer = np.concatenate([evt['layer_id'] for evt in events])

        self.r_mean = np.mean(all_r)
        self.r_std = np.std(all_r)
        self.theta_mean = np.mean(all_theta)
        self.theta_std = np.std(all_theta)
        self.z_mean = np.mean(all_z)
        self.z_std = np.std(all_z)
        self.layer_mean = np.mean(all_layer)
        self.layer_std = np.std(all_layer)

    def transform(self, X):
        X_np = X.numpy()
        X_np[:, 0] = (X_np[:, 0] - self.r_mean) / self.r_std
        X_np[:, 1] = (X_np[:, 1] - self.theta_mean) / self.theta_std
        X_np[:, 2] = (X_np[:, 2] - self.z_mean) / self.z_std
        X_np[:, 3] = (X_np[:, 3] - self.layer_mean) / self.layer_std
        return torch.from_numpy(X_np).float()

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.encoder_layer = TransformerEncoderLayer(
            d_model=in_features, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=3)
        self.projection = nn.Linear(in_features, 64)

    def forward(self, batch):
        embeddings = []
        for event in batch:
            event = event.unsqueeze(0)  # Add batch dimension
            encoded = self.transformer_encoder(event)
            projected = self.projection(encoded)
            embeddings.append(projected.squeeze(0))
        return embeddings

def make_model(in_features):
    return HitClassifier(in_features)

# 3. ---------- MODEL TRAINING ----------
def compute_accuracy(embeddings, track_ids):
    correct = 0
    total = 0
    for emb, tid in zip(embeddings, track_ids):
        if emb.shape[0] == 0:
            continue
        k = len(torch.unique(tid))
        if k == 0:
            continue
        emb_np = emb.detach().cpu().numpy()
        tid_np = tid.cpu().numpy()
        kmeans = KMeans(n_clusters=k, n_init=10)
        pred_labels = kmeans.fit_predict(emb_np)
        # Hungarian algorithm to match clusters to true labels
        from sklearn.utils.linear_assignment_ import linear_assignment
        contingency_matrix = np.zeros((k, k), dtype=np.int64)
        for i in range(len(pred_labels)):
            contingency_matrix[pred_labels[i], tid_np[i]] += 1
        row_ind, col_ind = linear_assignment(-contingency_matrix)
        mapping = {row: col for row, col in zip(row_ind, col_ind)}
        remapped_labels = np.array([mapping[p] for p in pred_labels])
        correct += np.sum(remapped_labels == tid_np)
        total += len(tid_np)
    return correct / total if total > 0 else 0.0

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_model = None
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        all_train_embs, all_train_tids = [], []

        for batch in train_loader:
            optimizer.zero_grad()
            events = [X.to(device) for X, _ in batch]
            track_ids = [tid.to(device) for _, tid in batch]

            embeddings = model(events)
            loss = 0.0
            for emb, tid in zip(embeddings, track_ids):
                sim_matrix = torch.matmul(emb, emb.T)  # (N, N)
                mask = (tid.unsqueeze(1) == tid.unsqueeze(0)).float()
                pos_loss = -torch.log(torch.sigmoid(sim_matrix[mask == 1])).mean()
                neg_loss = -torch.log(1 - torch.sigmoid(sim_matrix[mask == 0])).mean()
                loss += (pos_loss + neg_loss) / 2
            loss /= len(embeddings)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

            all_train_embs.extend([e.detach().cpu() for e in embeddings])
            all_train_tids.extend([t.cpu() for t in track_ids])

        train_acc = compute_accuracy(all_train_embs, all_train_tids)
        train_losses.append(epoch_train_loss / len(train_loader))
        train_accs.append(train_acc)

        model.eval()
        epoch_val_loss = 0.0
        all_val_embs, all_val_tids = [], []
        with torch.no_grad():
            for batch in val_loader:
                events = [X.to(device) for X, _ in batch]
                track_ids = [tid.to(device) for _, tid in batch]

                embeddings = model(events)
                loss = 0.0
                for emb, tid in zip(embeddings, track_ids):
                    sim_matrix = torch.matmul(emb, emb.T)
                    mask = (tid.unsqueeze(1) == tid.unsqueeze(0)).float()
                    pos_loss = -torch.log(torch.sigmoid(sim_matrix[mask == 1])).mean()
                    neg_loss = -torch.log(1 - torch.sigmoid(sim_matrix[mask == 0])).mean()
                    loss += (pos_loss + neg_loss) / 2
                loss /= len(embeddings)
                epoch_val_loss += loss.item()

                all_val_embs.extend([e.cpu() for e in embeddings])
                all_val_tids.extend([t.cpu() for t in track_ids])

        val_acc = compute_accuracy(all_val_embs, all_val_tids)
        val_losses.append(epoch_val_loss / len(val_loader))
        val_accs.append(val_acc)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_model)
    return model, train_losses, val_losses, train_accs, val_accs

EPOCHS = 20

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    raw_train, raw_val = _load_events("train"), _load_events("val")
    if dryrun:
        raw_train, raw_val = raw_train[:32], raw_val[:8]
    pre = make_preprocessor().fit(raw_train)
    train_ds = EventDataset(raw_train, pre, train=True)
    val_ds   = EventDataset(raw_val , pre, train=False)
    train_ld = DataLoader(train_ds, batch_size=512,
                        shuffle=True, collate_fn=_ragged)
    val_ld   = DataLoader(val_ds,   batch_size=512,
                        collate_fn=_ragged)

    # 2. Build model
    in_features = train_ds[0][0].shape[-1]                   
    model = make_model(in_features)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_ld, val_ld, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* - run a single toy forward pass
    if dryrun:
        toy_event       = torch.zeros(10, in_features)
        toy_transformed = pre.transform(toy_event)
        toy_batch       = [toy_transformed]
        try:
            _ = trained_model(toy_batch)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 5. Persist artefacts
    if not dryrun:
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
# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

