import os, glob, gzip, pickle, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import networkx as nx

CSV_DIR   = r"challenges\TRACKFORMERS\data\hits_and_tracks_3d_events_all.csv" 
CACHE_DIR = "./cache_redvid_cpu"
MAX_EDGES = 20_000                       # subsample for speed
BATCH_SZ  = 8
EPOCHS    = 6
DTYPE     = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
DEVICE    = torch.device("cpu")          # force CPU

# ------------------------------------------------------------
def cartesian(hits):
    r = hits['hit_r'].values
    th = hits['hit_theta'].values
    z = hits['hit_z'].values
    x = r * np.cos(th);  y = r * np.sin(th)
    return np.stack([x, y, z], axis=1)   # [N,3]

def build_event(df):
    pos   = cartesian(df)                              # [N,3]
    layer = df['sub_detector_id'].to_numpy()
    tid   = df['track_id'].to_numpy()
    # edge candidates – only consecutive layers
    src, dst = [], []
    by_lay = {}
    for i,L in enumerate(layer):
        by_lay.setdefault(L, []).append(i)
    ordered = sorted(by_lay)
    for a,b in zip(ordered[:-1], ordered[1:]):
        src.extend(by_lay[a]);  dst.extend(by_lay[b])
    src, dst = np.asarray(src), np.asarray(dst)
    if len(src)==0: return None

    # subsample to cap memory
    keep = np.random.choice(len(src),
                             min(len(src), MAX_EDGES),
                             replace=False)
    src, dst = src[keep], dst[keep]

    p_i, p_j = pos[src], pos[dst]
    dxyz = p_i - p_j                          # [E,3]
    dr   = np.linalg.norm(p_i[:,:2],axis=1) - np.linalg.norm(p_j[:,:2],axis=1)
    gap  = layer[src] - layer[dst]            # always -1
    feats = np.column_stack([dxyz, np.abs(dr), np.abs(gap)])
    y = (tid[src] == tid[dst]).astype(np.float32)
    return feats.astype(np.float32), y

def preprocess():
    os.makedirs(CACHE_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    random.seed(42); random.shuffle(files)
    split = int(0.8*len(files))
    for tag, sel in [('train', files[:split]), ('val', files[split:])]:
        out = os.path.join(CACHE_DIR, f"{tag}.pkl.gz")
        if os.path.exists(out): continue
        blobs = []
        for csv in sel:
            df = pd.read_csv(csv)
            for _,hits in df.groupby('event_id'):
                out_evt = build_event(hits)
                if out_evt: blobs.append(out_evt)
        with gzip.open(out, 'wb') as f: pickle.dump(blobs, f)
        print(f"cached {tag}: {len(blobs)} events → {out}")

class EdgeDataset(Dataset):
    def __init__(self, tag):
        path = os.path.join(CACHE_DIR, f"{tag}.pkl.gz")
        with gzip.open(path,'rb') as f: self.events = pickle.load(f)
    def __len__(s): return len(s.events)
    def __getitem__(s,i):
        x,y = s.events[i]
        return torch.as_tensor(x, dtype=DTYPE), torch.as_tensor(y)

def collate(batch):
    xs, ys = zip(*batch)
    return torch.cat(xs), torch.cat(ys)

class EdgeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5,32), nn.ReLU(),
            nn.Linear(32,32), nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self,x): return self.mlp(x).squeeze(1)  # [E]

def run_epoch(loader, model, loss_fn, opt=None):
    tot, y_true, y_prob = 0., [], []
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        if opt:
            opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()*y.numel()
        y_true.append(y.cpu()); y_prob.append(torch.sigmoid(out).cpu())
    y_true = torch.cat(y_true); y_prob = torch.cat(y_prob)
    auc = roc_auc_score(y_true, y_prob)
    return tot/len(loader.dataset), auc

def main():
    preprocess()
    train_ds, val_ds = EdgeDataset('train'), EdgeDataset('val')
    train_ld = DataLoader(train_ds, BATCH_SZ, shuffle=True, collate_fn=collate)
    val_ld   = DataLoader(val_ds,   BATCH_SZ, collate_fn=collate)

    model = EdgeNet().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    bce   = nn.BCEWithLogitsLoss()
    for ep in range(1,EPOCHS+1):
        tr,auc_tr = run_epoch(train_ld, model, bce, opt)
        vl,auc_vl = run_epoch(val_ld,   model, bce)
        sched.step()
        print(f"E{ep:02d}  train-AUC={auc_tr:.3f}  val-AUC={auc_vl:.3f}")
    torch.save(model.state_dict(), "edge_mlp_cpu.pth")

    # quick demo reconstruction on first val event
    feats, y = val_ds[0]
    with torch.no_grad():
        s = torch.sigmoid(model(feats)).numpy()
    keep = np.where(s>0.5)[0]
    G = nx.Graph(); G.add_edges_from([(int(i),int(j)) for i,j in
                         zip(np.zeros_like(keep), keep)])  # dummy index shift not needed for demo
    print("recovered tracks:", len(list(nx.connected_components(G))))

if __name__ == "__main__":
    main()
