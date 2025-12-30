
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, pickle, importlib, gzip, json, torch, torch_geometric, scipy 
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.llm_io import normalise_batch, assert_label_output, build_dataset, build_dataloader, split_X_y, EventDataset
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy, write_loaderspec
from utils.suffix_utils import base_from_argv0, write_json, plot_train_val, persist_artefacts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"

SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_DIR = "./challenges/TRACKFORMERS/data/train"
TAG      = "REDVID_10-50_linear_frac0.05"

def _load_events(split: str):
    pkl = os.path.join(DATA_DIR, f"{TAG}_{split}.pkl.gz")
    with gzip.open(pkl, "rb") as fh:
        return pickle.load(fh)["events"]

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import hdbscan
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv, knn

# -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomPygDataset(Dataset):
    def __init__(self, events, pre, train: bool, **kwargs):
        self.events = events
        self.pre = pre
        self.is_train = train

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        event_dict = self.events[idx]
        # The preprocessor's transform method will convert the event dict to a PyG Data object
        return self.pre.transform(event_dict)

def make_dataset(events, pre, train: bool, **kwargs):
    return CustomPygDataset(events, pre, train=train, **kwargs)

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # Must implement:
    #   - fit()
    #   - transform()

    def __init__(self):
        self.norm_stats = {}
        self.knn_k = 8

    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "llm_script:make_dataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 16, # GNNs can be memory-intensive
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,  # PyG DataLoader handles its own collation
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, data):
        # data is a list of X tensors
        all_hits = torch.cat(data, dim=0) # [Total_hits, 4]

        r, theta, z, layer_id = all_hits.T
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        # Features to normalize: x, y, z, r, layer_id
        features_to_norm = torch.stack([x, y, z, r, layer_id], dim=1)

        mean = features_to_norm.mean(dim=0)
        std = features_to_norm.std(dim=0)
        std[std == 0] = 1.0 # Avoid division by zero

        self.norm_stats['mean'] = mean
        self.norm_stats['std'] = std
        return self

    def transform(self, event_dict):
        # event_dict contains r, theta, z, layer_id, track_id
        r = torch.from_numpy(event_dict['hit_r'])
        theta = torch.from_numpy(event_dict['hit_theta'])
        z = torch.from_numpy(event_dict['hit_z'])
        layer_id = torch.from_numpy(event_dict['layer_id'])
        y = torch.from_numpy(event_dict['track_id']).long()

        # Feature engineering
        x = r * torch.cos(theta)
        y_coord = r * torch.sin(theta) # rename to avoid clash with labels 'y'

        # shape: [N_hits, 5]
        features = torch.stack([x, y_coord, z, r, layer_id], dim=1)

        # Normalization
        mean = self.norm_stats['mean']
        std = self.norm_stats['std']
        norm_features = (features - mean) / std

        # Graph construction
        coords = torch.stack([x, y_coord, z], dim=1)
        unique_layers = torch.unique(layer_id)

        all_edges = []
        # Group hits by layer
        hit_indices_by_layer = {l.item(): (layer_id == l).nonzero(as_tuple=False).squeeze(-1) for l in unique_layers}

        for i in range(len(unique_layers) - 1):
            l1_id = unique_layers[i].item()
            l2_id = unique_layers[i+1].item()

            indices1 = hit_indices_by_layer[l1_id]
            indices2 = hit_indices_by_layer[l2_id]

            if len(indices1) == 0 or len(indices2) == 0:
                continue

            # Find kNN for points in layer l1 from points in layer l2
            # The edge direction should be from l1 to l2
            edge_pairs = knn(x=coords[indices2], y=coords[indices1], k=self.knn_k)

            # knn returns (row, col) where row is index in y (l1) and col is index in x (l2)
            sources = indices1[edge_pairs[0]]
            targets = indices2[edge_pairs[1]]

            all_edges.append(torch.stack([sources, targets]))

        if len(all_edges) > 0:
            edge_index = torch.cat(all_edges, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(x=norm_features, edge_index=edge_index, y=y)

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim=128):
        super().__init__()
        # Note: EdgeConv internally doubles the input feature dimension
        nn1 = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim), 
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.gnn1 = EdgeConv(nn1, aggr='mean')

        nn2 = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.gnn2 = EdgeConv(nn2, aggr='mean')

        nn3 = nn.Sequential(
            nn.Linear(2 * hidden_dim, embedding_dim)
        )
        self.gnn3 = EdgeConv(nn3, aggr='mean')

    def forward(self, x, edge_index):
        h1 = self.gnn1(x, edge_index)
        h2 = self.gnn2(h1, edge_index)
        h3 = self.gnn3(h2, edge_index)
        # Normalize embeddings to lie on a unit hypersphere
        return nn.functional.normalize(h3, p=2, dim=1)

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # Infer input dimension from preprocessed data. My preprocessor creates 5 features.
        self.input_dim = 5
        self.embedding_dim = 16 
        self.hdbscan_min_cluster_size = 4
        self.hdbscan_min_samples = 1

        self.embedding_net = EmbeddingNet(self.input_dim, self.embedding_dim)

    def forward(self, batch):
        # NOTE: This forward pass always performs clustering for inference.
        # Training is handled separately in `train_model` by accessing `self.embedding_net`.

        # self.training is False during inference and validation
        if self.training:
           # The training loop will call embedding_net directly, this is a safeguard
           raise RuntimeError("model.forward() should not be called during training. Access model.embedding_net instead.")

        # `batch` is a PyG Batch object
        embeddings = self.embedding_net(batch.x, batch.edge_index)

        # Split embeddings back into a list, one for each event
        hit_counts_per_event = torch.bincount(batch.batch).cpu().tolist()
        event_embeddings_list = torch.split(embeddings, hit_counts_per_event)

        pred_labels_list = []
        for event_embeds in event_embeddings_list:
            if event_embeds.shape[0] < self.hdbscan_min_cluster_size:
                # Not enough hits to form a cluster, predict all as noise
                pred_labels = torch.zeros(event_embeds.shape[0], dtype=torch.long)
                pred_labels_list.append(pred_labels)
                continue

            event_embeds_np = event_embeds.detach().cpu().numpy()

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.hdbscan_min_cluster_size,
                min_samples=self.hdbscan_min_samples,
                metric='euclidean',
                core_dist_n_jobs=1 # Avoid thread contention
            )
            labels = clusterer.fit_predict(event_embeds_np)

            # Post-processing: remap labels to be contiguous from 1, with 0 for noise
            labels[labels == -1] = 0 # HDBSCAN noise is -1, remap to 0
            unique_ids = np.unique(labels[labels > 0])

            # Make cluster IDs contiguous
            if len(unique_ids) > 0:
                mapper = {old_id: new_id for new_id, old_id in enumerate(unique_ids, 1)}
                final_labels = np.vectorize(mapper.get)(labels, 0) # default to 0 (noise) if not in map
            else:
                final_labels = np.zeros_like(labels) # all noise

            pred_labels_list.append(torch.from_numpy(final_labels).long())

        return pred_labels_list

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 35

def centroid_loss(embeddings, track_ids, batch_idx, push_margin=1.0):
    device = embeddings.device
    total_pull_loss = torch.tensor(0.0, device=device)
    total_push_loss = torch.tensor(0.0, device=device)
    n_events_with_tracks = 0

    for event_id in torch.unique(batch_idx):
        event_mask = (batch_idx == event_id)
        event_embeds = embeddings[event_mask]
        event_tids = track_ids[event_mask]

        # Ignore noise hits for loss calculation
        valid_hits_mask = event_tids > 0
        if not valid_hits_mask.any():
            continue

        unique_tracks = torch.unique(event_tids[valid_hits_mask])
        if len(unique_tracks) == 0:
            continue

        n_events_with_tracks += 1
        centroids = []

        # Pull loss (attractive)
        event_pull_loss = torch.tensor(0.0, device=device)
        for tid in unique_tracks:
            track_mask = (event_tids == tid)
            track_embeds = event_embeds[track_mask]

            centroid = track_embeds.mean(dim=0)
            centroids.append(centroid)

            pull = (track_embeds - centroid).norm(p=2, dim=1).pow(2).mean()
            event_pull_loss += pull

        total_pull_loss += event_pull_loss / len(unique_tracks)

        # Push loss (repulsive)
        if len(centroids) > 1:
            centroids = torch.stack(centroids)
            dists = torch.cdist(centroids, centroids) # [N_tracks, N_tracks]

            # Penalize centroids being too close
            push = torch.clamp(push_margin - dists, min=0).pow(2)
            # Sum over upper triangle, avoiding diagonal
            push_loss = torch.triu(push, diagonal=1).sum()
            num_pairs = len(unique_tracks) * (len(unique_tracks) - 1) / 2
            total_push_loss += push_loss / num_pairs

    if n_events_with_tracks == 0:
        return torch.tensor(0.0, device=device)

    return (total_pull_loss + total_push_loss) / n_events_with_tracks

def train_model(model, train_loader, val_loader, epochs):
    # Training is performed on the internal embedding network
    embedding_net = model.embedding_net.to(device)
    optimizer = optim.Adam(embedding_net.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)

    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 5

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        embedding_net.train()
        model.train() # Make model aware it's in training mode

        epoch_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            embeddings = embedding_net(batch.x, batch.edge_index)
            loss = centroid_loss(embeddings, batch.y, batch.batch)

            # In case of no valid tracks in batch or other issues
            if torch.isnan(loss) or loss.item() == 0.0:
                continue

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        train_loss.append(avg_train_loss)
        train_acc.append(-avg_train_loss) # use negative loss as an accuracy-like metric

        # Validation
        embedding_net.eval()
        model.eval() # Make model aware it's in eval mode
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                embeddings = embedding_net(batch.x, batch.edge_index)
                loss = centroid_loss(embeddings, batch.y, batch.batch)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_loss.append(avg_val_loss)
        val_acc.append(-avg_val_loss)

        # Verbose output for monitoring
        # print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            # print("Early stopping triggered.")
            break

    # Set model back to eval mode for final inference
    model.eval()
    return model, train_loss, val_loss, train_acc, val_acc

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    raw_train, raw_val = _load_events("train"), _load_events("val")
    if dryrun:
        raw_train, raw_val = raw_train[:32], raw_val[:8]
    Xs = [split_X_y(evt)[0] for evt in raw_train]
    pre = make_preprocessor().fit(Xs)

    # Build LoaderSpec
    spec = build_spec_from_preproc(pre, script_module="llm_script")
    spec = enforce_pyg_policy(spec)

    # Build loaders - preproc in dataset
    train_ds     = build_dataset(spec, raw_train, pre, train=True)
    val_ds       = build_dataset(spec, raw_val,   pre, train=False)
    train_loader = build_dataloader(spec, train_ds, is_eval=False)
    val_loader   = build_dataloader(spec, val_ds,   is_eval=True)

    # Build model
    first_batch = next(iter(train_loader))
    view        = normalise_batch(first_batch, device=device)
    model       = make_model(view.batch_x).to(device)

    # Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # Dry-run safety check
    if dryrun:
        try:
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    view = normalise_batch(batch, device=device)
                    out  = model(view.batch_x)
                    assert_label_output(view.batch_x, out, allow_noise_label=True)
                    if i >= 4: # loop over 4 batches
                        break
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    if not dryrun:
        # Persist artefacts
        base = base_from_argv0()
        persist_artefacts(base, SCRIPT_DIR, trained_model, pre, spec)

        # Save plots
        plot_train_val(tr_loss, va_loss, f"{base} Loss", os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
        plot_train_val(tr_acc, va_acc, f"{base} Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))
        
        # Write JSON Summary
        write_json(
            {"train_loss": tr_loss, "val_loss": va_loss, "train_acc": tr_acc, "val_acc": va_acc},
            out_path=os.path.join(SCRIPT_DIR, f"{base}_train_summary.json"),
        )

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

