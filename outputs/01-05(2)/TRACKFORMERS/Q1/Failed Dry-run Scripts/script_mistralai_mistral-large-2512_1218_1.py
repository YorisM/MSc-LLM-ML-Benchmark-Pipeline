
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import detect_and_assert_lane, assert_label_output_by_lane, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts, build_trackformers_model

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

def split_X_y(evt):
    X = np.column_stack([
        evt["hit_r"].astype(np.float32),
        evt["hit_theta"].astype(np.float32),
        evt["hit_z"].astype(np.float32),
        evt["layer_id"].astype(np.float32)
    ])
    y = evt["track_id"].astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)

class EventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, labels = split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        return (X, labels)

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# ---------- IMPORTS ----------
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.layer_mean_r = None
        self.layer_std_r = None
        self.layer_mean_z = None
        self.layer_std_z = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,

            "collate": "ragged_xy",
            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False, "num_workers": 2}
        }

    def fit(self, Xs):
        # Compute global statistics for normalization
        all_X = np.concatenate(Xs, axis=0)
        self.scaler.fit(all_X[:, :3])  # Only scale r, theta, z

        # Compute per-layer statistics
        layer_ids = np.unique(all_X[:, 3])
        self.layer_mean_r = {}
        self.layer_std_r = {}
        self.layer_mean_z = {}
        self.layer_std_z = {}

        for layer in layer_ids:
            mask = all_X[:, 3] == layer
            self.layer_mean_r[layer] = np.mean(all_X[mask, 0])
            self.layer_std_r[layer] = np.std(all_X[mask, 0]) + 1e-6
            self.layer_mean_z[layer] = np.mean(all_X[mask, 2])
            self.layer_std_z[layer] = np.std(all_X[mask, 2]) + 1e-6

        return self

    def transform(self, X):
        # X: [N_hits, 4] - r, theta, z, layer_id
        X = X.clone().numpy() if torch.is_tensor(X) else X.copy()

        # Normalize r, theta, z
        X[:, :3] = self.scaler.transform(X[:, :3])

        # Layer-aware normalization
        layer_ids = X[:, 3]
        for layer in np.unique(layer_ids):
            mask = layer_ids == layer
            X[mask, 0] = (X[mask, 0] - self.layer_mean_r[layer]) / self.layer_std_r[layer]
            X[mask, 2] = (X[mask, 2] - self.layer_mean_z[layer]) / self.layer_std_z[layer]

        return torch.FloatTensor(X)  # [N_hits, 4]

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()

        # Determine input feature dimension from example batch
        self.input_dim = example_batch_x[0].shape[1] if isinstance(example_batch_x, list) else example_batch_x.shape[1]

        # Embedding layers
        self.hit_embed = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Output layers
        self.cluster_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64)
        )

        # Learnable position encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, 128))

        # Track classification head
        self.track_classifier = nn.Linear(128, 1)

    def forward(self, Xs):
        # Xs: list of [N_i, 4] tensors
        device = next(self.parameters()).device

        # Process each event in the batch
        all_embeddings = []
        all_masks = []
        max_len = max(x.size(0) for x in Xs)

        for x in Xs:
            x = x.to(device)
            N = x.size(0)

            # Get hit embeddings
            embed = self.hit_embed(x)  # [N, 128]

            # Add position encoding
            pos_enc = self.pos_encoder[:, :N, :]  # [1, N, 128]
            embed = embed + pos_enc

            # Create attention mask
            mask = torch.zeros(N, N, device=device)
            mask[torch.triu_indices(N, N, offset=1)] = float('-inf')

            all_embeddings.append(embed)
            all_masks.append(mask)

        # Pad sequences for transformer
        padded_embeddings = torch.nn.utils.rnn.pad_sequence(
            all_embeddings, batch_first=True, padding_value=0
        )  # [B, max_len, 128]

        # Pad masks
        padded_masks = torch.nn.utils.rnn.pad_sequence(
            all_masks, batch_first=True, padding_value=0
        )  # [B, max_len, max_len]

        # Transformer processing
        transformer_out = self.transformer(
            padded_embeddings,
            mask=padded_masks
        )  # [B, max_len, 128]

        # Unpad and return
        outputs = []
        for i, x in enumerate(Xs):
            N = x.size(0)
            outputs.append(transformer_out[i, :N, :])  # [N, 128]

        return outputs

    def predict_labels(self, Xs):
        device = next(self.parameters()).device
        embeddings = self.forward(Xs)

        # Get cluster assignments
        all_labels = []
        for embed in embeddings:
            embed = embed.detach().cpu().numpy()
            cluster_feats = self.cluster_head(torch.FloatTensor(embed).to(device)).detach().cpu().numpy()

            # Use HDBSCAN for clustering
            try:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=4,
                    min_samples=2,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )
                labels = clusterer.fit_predict(cluster_feats)
            except:
                # Fallback to connected components if HDBSCAN fails
                tree = cKDTree(cluster_feats)
                adj_matrix = tree.sparse_distance_matrix(tree, max_distance=1.0)
                n_components, labels = connected_components(
                    csr_matrix(adj_matrix), directed=False
                )

            # Convert to torch tensor and adjust labels
            labels = torch.LongTensor(labels)
            unique_labels = torch.unique(labels)
            label_map = {old.item(): new for new, old in enumerate(unique_labels)}
            labels = torch.tensor([label_map.get(l.item(), -1) for l in labels])

            all_labels.append(labels.to(device))

        return all_labels

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for Xs, ys in train_loader:
            Xs = [x.to(device) for x in Xs]
            ys = [y.to(device) for y in ys]

            optimizer.zero_grad()

            # Forward pass
            embeddings = model(Xs)  # List of [N_i, 128]

            # Compute loss
            loss = 0.0
            for embed, y in zip(embeddings, ys):
                # Skip noise hits (track_id = 0)
                mask = y > 0
                if mask.sum() == 0:
                    continue

                embed = embed[mask]
                y = y[mask]

                # Get unique track IDs
                unique_tracks = torch.unique(y)
                track_embeds = []

                for track in unique_tracks:
                    track_embeds.append(embed[y == track].mean(dim=0))

                track_embeds = torch.stack(track_embeds)  # [num_tracks, 128]

                # Compute pairwise distances
                dists = torch.cdist(track_embeds, track_embeds)
                pos_mask = torch.eye(len(unique_tracks), device=device).bool()
                neg_mask = ~pos_mask

                # Contrastive loss
                pos_dists = dists[pos_mask]
                neg_dists = dists[neg_mask]

                loss += torch.logsumexp(neg_dists, dim=0) - pos_dists.mean()

            loss = loss / len(embeddings)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for Xs, ys in val_loader:
                Xs = [x.to(device) for x in Xs]
                ys = [y.to(device) for y in ys]

                embeddings = model(Xs)
                batch_loss = 0.0

                for embed, y in zip(embeddings, ys):
                    mask = y > 0
                    if mask.sum() == 0:
                        continue

                    embed = embed[mask]
                    y = y[mask]

                    unique_tracks = torch.unique(y)
                    track_embeds = []

                    for track in unique_tracks:
                        track_embeds.append(embed[y == track].mean(dim=0))

                    track_embeds = torch.stack(track_embeds)
                    dists = torch.cdist(track_embeds, track_embeds)
                    pos_mask = torch.eye(len(unique_tracks), device=device).bool()
                    neg_mask = ~pos_mask

                    pos_dists = dists[pos_mask]
                    neg_dists = dists[neg_mask]

                    batch_loss += torch.logsumexp(neg_dists, dim=0) - pos_dists.mean()

                val_loss += batch_loss.item() / len(embeddings)

        # Calculate accuracy (simplified - actual FitAccuracy would need full evaluation)
        train_acc = 0.0  # Placeholder
        val_acc = 0.0    # Placeholder

        epoch_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model, train_losses, val_losses, train_accs, val_accs

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

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

    # Build batch and check
    first_batch = next(iter(train_loader))
    mode = detect_and_assert_lane(spec, first_batch)

    # Build model
    model = build_trackformers_model(mode, first_batch, make_model, device)

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
        if not hasattr(trained_model, "predict_labels") or not callable(getattr(trained_model, "predict_labels")):
            raise TypeError("Contract error: trained model must implement predict_labels(batch_x).")

        trained_model.eval()
        try:
            with torch.no_grad():
                mode = None
                for i, batch in enumerate(val_loader):
                    if mode is None:
                        mode = detect_and_assert_lane(spec, batch)

                    if mode == "torch_ragged_xy":
                        Xs, _ys = batch
                        Xs = [x.to(device) for x in Xs]
                        out = trained_model.predict_labels(Xs)
                    elif mode == "pyg_batch":
                        G = batch.to(device)
                        out = trained_model.predict_labels(G)
                    else:
                        raise RuntimeError(f"Unknown lane mode: {mode}")

                    assert_label_output_by_lane(mode, batch, out, allow_noise_label=True)
                    if i >= 3:  # 4 batches
                        break
        except Exception as e:
            raise RuntimeError("Sanity-check predict_labels() failed") from e
        return


    if not dryrun:
        # Persist artefacts
        base = base_from_argv0()
        persist_artefacts(base, SCRIPT_DIR, trained_model, pre, spec)

        # Save plots
        plot_train_val(tr_loss, va_loss, f"{base} Loss", os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
        plot_train_val(tr_acc, va_acc, f"{base} Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))
        
        # Write JSON Summary
        summary = {
            "epochs": n_epochs      if n_epochs else None,
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

# ----------------  END HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

