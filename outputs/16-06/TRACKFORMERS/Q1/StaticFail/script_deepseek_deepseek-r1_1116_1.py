
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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

# 1. ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, events):
        all_hits = []
        for evt in events:
            X, _ = split_X_y(evt)
            all_hits.append(X.numpy())
        self.scaler.fit(np.concatenate(all_hits))

    def transform(self, X):
        return torch.tensor(self.scaler.transform(X), dtype=torch.float32)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        self.projection = nn.Linear(128, 64)

    def forward(self, batch):
        outputs = []
        for event in batch:
            hits = event[0]
            emb = self.encoder(hits)
            trans_out = self.transformer(emb.unsqueeze(0)).squeeze(0)
            proj_out = self.projection(trans_out)
            outputs.append(proj_out)
        return outputs

def make_model(in_features):
    return HitClassifier(in_features)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20

def nt_xent_loss(embeddings, track_ids, temperature=0.1):
    sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
    sim_matrix = sim_matrix / temperature
    mask = (track_ids.unsqueeze(1) == track_ids.unsqueeze(0)).float()
    mask.fill_diagonal_(0)

    pos_sum = (sim_matrix * mask).sum(dim=1)
    neg_sum = torch.logsumexp(sim_matrix - 1e9 * mask, dim=1)
    loss = - (pos_sum - neg_sum).mean()
    return loss

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_acc = 0
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss, epoch_train_acc = 0, 0

        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = 0
            batch_acc = 0

            for event in batch:
                X, y = event[0].to(device), event[1].to(device)
                embeddings = model([(X, y)])[0]

                # Calculate loss
                loss = nt_xent_loss(embeddings, y)
                batch_loss += loss

            batch_loss /= len(batch)
            batch_loss.backward()
            optimizer.step()
            epoch_train_loss += batch_loss.item()

        # Validation
        model.eval()
        epoch_val_loss, epoch_val_acc = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                for event in batch:
                    X, y = event[0].to(device), event[1].to(device)
                    embeddings = model([(X, y)])[0]

                    # Calculate validation loss
                    loss = nt_xent_loss(embeddings, y)
                    epoch_val_loss += loss.item()

                    # Calculate clustering accuracy
                    emb_np = embeddings.cpu().numpy()
                    y_np = y.cpu().numpy()

                    # DBSCAN clustering
                    clustering = DBSCAN(eps=0.5, min_samples=2).fit(emb_np)
                    pred_labels = clustering.labels_

                    # Hungarian matching
                    unique_pred = np.unique(pred_labels)
                    unique_true = np.unique(y_np)
                    cost_matrix = np.zeros((len(unique_pred), len(unique_true)), dtype=int)

                    for i, u_p in enumerate(unique_pred):
                        for j, u_t in enumerate(unique_true):
                            cost_matrix[i,j] = -np.sum((pred_labels == u_p) & (y_np == u_t))

                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    correct = -cost_matrix[row_ind, col_ind].sum()
                    acc = correct / len(y_np)
                    epoch_val_acc += acc

        # Update metrics
        train_loss.append(epoch_train_loss/len(train_loader))
        val_loss.append(epoch_val_loss/len(val_loader))
        val_acc.append(epoch_val_acc/len(val_loader.dataset))

        # Update best model
        if val_acc[-1] > best_acc:
            best_acc = val_acc[-1]
            torch.save(model.state_dict(), 'best_model.pt')

        scheduler.step()

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    return model, train_loss, val_loss, [], val_acc

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

