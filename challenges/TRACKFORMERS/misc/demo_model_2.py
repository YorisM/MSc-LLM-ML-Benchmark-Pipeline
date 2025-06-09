# challenges\TRACKFORMERS\misc\demo_model_2.py

# Light baseline for the TRACKFORMERS challenge.
# Run: 
# python challenges/TRACKFORMERS/misc/demo_model_2.py --data-dir challenges/TRACKFORMERS/data --tag 10_50_linear


import argparse, gzip, pickle, os
from typing import List, Dict, Any

import numpy as np
import torch, torch.nn as nn
import hdbscan
from tqdm import tqdm


# CLI
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="challenges/TRACKFORMERS/data")
    p.add_argument("--tag", default="10_50_linear")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-3)
    return p.parse_args()


# I/O
def load_split(data_dir: str, tag: str, split: str) -> List[Dict[str, Any]]:
    path = os.path.join(data_dir, f"REDVID_{tag}_{split}.pkl.gz")
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)["events"]


def event_to_tensor(evt):
    layer = evt["layer_id"]
    layer_norm = layer / layer.max()  # scalar 0-1
    feats = np.column_stack([evt["hit_r"],
                             evt["hit_theta"],
                             evt["hit_z"],
                             layer_norm]).astype(np.float32)
    return (torch.from_numpy(feats),
            torch.from_numpy(evt["track_params"].astype(np.float32)),
            torch.from_numpy(evt["track_id"].astype(np.int32)))


# Model
class TinyEncoder(nn.Module):
    def __init__(self, dim=64, heads=4, depth=2, in_dim=4):
        super().__init__()
        self.input = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim,
                                       nhead=heads,
                                       dim_feedforward=dim*2,
                                       batch_first=True)
            for _ in range(depth)
        ])
        self.head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, 6)          # 6 track params
        )

    def forward(self, x):             # x shape [N,4]
        h = self.input(x)
        for enc in self.layers:
            h = enc(h)
        return self.head(h)           # [N,6]


# FitAccuracy
def fit_accuracy(pred_label, true_tid):
    """
    TrackML-style: for each true track find best-matching predicted cluster.
    """
    correct = 0
    for t in np.unique(true_tid):
        mask_true = true_tid == t
        # candidate predicted labels overlapping this true track
        labels, counts = np.unique(pred_label[mask_true], return_counts=True)
        if labels.size == 0:          # all hits flagged noise
            continue
        best_label = labels[counts.argmax()]
        # hits that are true t and predicted best_label
        mask_pred = pred_label == best_label
        correct += (mask_true & mask_pred).sum()
    return correct / len(true_tid)


# Train & Eval
def run_epoch(evts, model, opt=None):
    model.train(bool(opt))
    loss_fn = nn.SmoothL1Loss()
    total_loss, total_hits = 0., 0
    for evt in tqdm(evts, disable=opt is None):
        x, y, _ = event_to_tensor(evt)
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        if opt:
            opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * x.size(0)
        total_hits += x.size(0)
    return total_loss / total_hits


def evaluate(evts, model):
    model.eval(); scores = []
    with torch.no_grad():
        for evt in evts:
            x, _, true_tid = event_to_tensor(evt)
            pred = model(x.to(device)).cpu().numpy()
            # HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
            labels = clusterer.fit_predict(pred)
            score = fit_accuracy(labels, true_tid.numpy())
            scores.append(score)
    return float(np.mean(scores))


# ───────────────────────────────────────────────  Main
if __name__ == "__main__":
    args = parse()
    device = torch.device("cpu")

    train_evts = load_split(args.data_dir, args.tag, "train")
    val_evts   = load_split(args.data_dir, args.tag, "val")
    test_evts  = load_split(args.data_dir, args.tag, "test")

    model = TinyEncoder().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        tr_loss = run_epoch(train_evts, model, opt)
        val_acc = evaluate(val_evts, model)
        print(f"E{ep:02d}  train_loss={tr_loss:.4f}  val_FitAcc={val_acc:.3f}")

    test_acc = evaluate(test_evts, model)
    print(f"\nFINAL  test_FitAccuracy = {test_acc:.3f}")
