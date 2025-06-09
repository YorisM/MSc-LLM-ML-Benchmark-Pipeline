# demo_model.py


# Ultra-light baseline for the TRACKFORMERS challenge.
# Run:
#   python challenges/TRACKFORMERS/misc/demo_model.py \
#        --data-dir challenges/TRACKFORMERS/data \
#        --tag 10_50_linear


import argparse, gzip, pickle, random, os
from typing import List, Dict, Any
import numpy as np
import torch, torch.nn as nn
import hdbscan
from tqdm import tqdm


# CLI
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    default="challenges/TRACKFORMERS/data")
    p.add_argument("--tag",         default="10_50_linear")
    p.add_argument("--lr",          type=float, default=3e-3)
    p.add_argument("--epochs",      type=int,   default=10)
    return p.parse_args()


# Data utils
def load_split(data_dir: str, tag: str, split: str) -> List[Dict[str,Any]]:
    path = os.path.join(data_dir, f"REDVID_{tag}_{split}.pkl.gz")
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)["events"]

def event_to_tensor(evt):
    layer_norm = evt["layer_id"] / evt["layer_id"].max()   # 0-1 scalar
    x = np.column_stack([evt["hit_r"],
                         evt["hit_theta"],
                         evt["hit_z"],
                         layer_norm]).astype(np.float32)
    y = evt["track_params"].astype(np.float32)              # (N,6)
    return torch.as_tensor(x), torch.as_tensor(y), evt["track_id"]


# Tiny model
class TinyReg(nn.Module):
    def __init__(self, in_dim=4, hid=64, out_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, out_dim)
        )
    def forward(self, x): return self.net(x)


# Clustering + metric
def simple_cluster_accuracy(pred_params, true_track_ids):
    """Assign each HDBSCAN cluster the majority true ID, return hit accuracy."""
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    labels = clusterer.fit_predict(pred_params)
    acc_hits, tot = 0, len(labels)
    for lbl in np.unique(labels):
        mask = labels == lbl
        majority = np.bincount(true_track_ids[mask]).argmax()
        acc_hits += (true_track_ids[mask] == majority).sum()
    return acc_hits / tot if tot else 0.0

def main():
    args = parse()
    device = torch.device("cpu")

    # 1. load
    train_evts = load_split(args.data_dir, args.tag, "train")[:1000]  # tiny subset
    test_evts  = load_split(args.data_dir, args.tag, "test") [:200]

    # 2. model
    model = TinyReg().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()

    # 3. training loop
    model.train()
    for ep in range(1, args.epochs+1):
        tot, n = 0., 0
        random.shuffle(train_evts)
        for evt in tqdm(train_evts, desc=f"epoch {ep}"):
            x, y, _ = event_to_tensor(evt)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        print(f"epoch {ep}  avg_loss = {tot/n:.4f}")

    # 4. inference & toy FitAccuracy
    model.eval(); accs = []
    with torch.no_grad():
        for evt in test_evts:
            x, _, true_tid = event_to_tensor(evt)
            pred = model(x).cpu().numpy()
            acc  = simple_cluster_accuracy(pred, true_tid)
            accs.append(acc)
    print(f"\nToy hit-level accuracy (mean over {len(accs)} test events): "
          f"{np.mean(accs):.3f}")

if __name__ == "__main__":
    main()
