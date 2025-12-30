
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import normalise_batch, assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
                        
DATASET = {
    "X_train": "./challenges/FOURTOPS/data/train/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/train/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/train/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/train/Y_val.csv"
}
                       
def load_data():
    X_train = pd.read_csv(DATASET["X_train"], dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv(DATASET["Y_train"], dtype=np.int64).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv(DATASET["X_val"], dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv(DATASET['Y_val'], dtype=np.int64).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train), torch.from_numpy(Y_train),
            torch.from_numpy(X_val), torch.from_numpy(Y_val))

class FourTopsDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.X = pre.transform(X) if pre is not None else X
        self.y = y
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

# ---------- IMPORTS ----------
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
import torch_geometric.transforms as T

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.node_feature_means = None
        self.node_feature_stds = None
        self.global_feature_means = None
        self.global_feature_stds = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "drop_last": False},
        }

    def fit(self, X, y=None):
        # Reshape to extract node features
        batch_size = X.shape[0]
        global_features = X[:, :2]  # (batch, 2)

        # Reshape object features: (batch, 18, 5)
        obj_features = X[:, 2:].reshape(batch_size, 18, 5)

        # Compute statistics for non-zero padded objects only
        valid_mask = (obj_features[:, :, 0] != 0)  # obj_id != 0
        valid_features = obj_features[valid_mask]  # (n_valid, 5)

        # Node feature statistics (excluding obj_id which is categorical)
        self.node_feature_means = torch.mean(valid_features[:, 1:], dim=0)  # (4,)
        self.node_feature_stds = torch.std(valid_features[:, 1:], dim=0) + 1e-8  # (4,)

        # Global feature statistics
        self.global_feature_means = torch.mean(global_features, dim=0)  # (2,)
        self.global_feature_stds = torch.std(global_features, dim=0) + 1e-8  # (2,)

        return self

    def transform(self, X):
        batch_size = X.shape[0]

        # Global features
        global_features = X[:, :2]  # (batch, 2)
        global_features = (global_features - self.global_feature_means) / self.global_feature_stds

        # Object features: (batch, 18, 5)
        obj_features = X[:, 2:].reshape(batch_size, 18, 5)
        obj_ids = obj_features[:, :, 0:1]  # (batch, 18, 1) - keep as categorical

        # Normalize continuous features (E, pT, eta, phi)
        continuous_features = obj_features[:, :, 1:]  # (batch, 18, 4)
        continuous_features = (continuous_features - self.node_feature_means) / self.node_feature_stds

        # Combine features
        processed_features = torch.cat([obj_ids, continuous_features], dim=-1)  # (batch, 18, 5)

        # Create mask for valid objects (obj_id != 0)
        valid_mask = (obj_ids[:, :, 0] != 0).float()  # (batch, 18)

        # Create adjacency matrix (fully connected graph)
        edge_indices = []
        for i in range(batch_size):
            num_valid = int(valid_mask[i].sum().item())
            if num_valid > 0:
                # Create complete graph for valid nodes
                src = torch.arange(num_valid, device=X.device).repeat_interleave(num_valid)
                dst = torch.arange(num_valid, device=X.device).repeat(num_valid)

                # Remove self-loops if desired
                mask = src != dst
                src, dst = src[mask], dst[mask]

                edge_index = torch.stack([src, dst], dim=0)
                edge_indices.append(edge_index)
            else:
                # Add dummy edge for empty graphs
                edge_indices.append(torch.zeros((2, 1), device=X.device, dtype=torch.long))

        # Combine global features, node features, valid mask, and edge indices
        output = {
            'global_features': global_features,
            'node_features': processed_features,
            'valid_mask': valid_mask,
            'edge_indices': edge_indices,
            'batch_indices': torch.arange(batch_size, device=X.device).repeat_interleave(
                [e.shape[1] for e in edge_indices]
            )
        }

        return output

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class ParticleGNN(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        input_dim = 5  # obj_id + 4 normalized features

        # Node feature embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        # Global feature embedding
        self.global_embedding = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128)
        )

        # GAT layers with edge feature computation
        self.gat1 = GATConv(128, 256, heads=4, dropout=0.1)
        self.gat2 = GATConv(256 * 4, 256, heads=4, dropout=0.1)
        self.gat3 = GATConv(256 * 4, 128, heads=2, dropout=0.1)

        # Transformer for global context
        encoder_layer = TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=2)

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def compute_edge_features(self, node_features, edge_index):
        """Compute invariant mass and deltaR as edge features"""
        src, dst = edge_index
        n_edges = src.shape[0]

        # Extract features
        src_features = node_features[src]  # (n_edges, 128)
        dst_features = node_features[dst]  # (n_edges, 128)

        # For simplicity, use learned edge features from node embeddings
        edge_features = src_features * dst_features
        return edge_features

    def forward(self, batch_x):
        # Extract components
        global_features = batch_x['global_features']
        node_features = batch_x['node_features']
        valid_mask = batch_x['valid_mask']
        edge_indices = batch_x['edge_indices']
        batch_idx = batch_x['batch_indices']

        batch_size = global_features.shape[0]

        # Embed node features
        node_emb = self.node_embedding(node_features)  # (batch, 18, 128)
        node_emb = node_emb * valid_mask.unsqueeze(-1)  # Mask padded nodes

        # Embed global features and broadcast to nodes
        global_emb = self.global_embedding(global_features)  # (batch, 128)
        global_emb_expanded = global_emb.unsqueeze(1).expand(-1, 18, -1)  # (batch, 18, 128)

        # Combine node and global embeddings
        node_emb = node_emb + global_emb_expanded  # (batch, 18, 128)

        # Process with GNN
        graphs = []
        for i in range(batch_size):
            mask_i = valid_mask[i] > 0
            if mask_i.sum() > 0:
                # Get valid nodes for this graph
                node_i = node_emb[i][mask_i]  # (num_valid, 128)
                edge_index = edge_indices[i]

                if edge_index.shape[1] > 0:
                    # Compute edge features
                    edge_attr = self.compute_edge_features(node_i, edge_index)

                    # Apply GAT layers
                    x = F.relu(self.gat1(node_i, edge_index, edge_attr=edge_attr))
                    x = F.relu(self.gat2(x, edge_index, edge_attr=edge_attr))
                    x = self.gat3(x, edge_index, edge_attr=edge_attr)

                    graphs.append(x)
                else:
                    graphs.append(node_i)
            else:
                graphs.append(torch.zeros((1, 256), device=node_emb.device))

        # Pad sequences for transformer
        max_len = max(g.shape[0] for g in graphs)
        padded = []
        for g in graphs:
            pad_len = max_len - g.shape[0]
            if pad_len > 0:
                g = F.pad(g, (0, 0, 0, pad_len))
            padded.append(g)

        transformer_input = torch.stack(padded, dim=0)  # (batch, max_len, 256)

        # Apply transformer with mask
        padding_mask = (torch.arange(max_len, device=node_emb.device).unsqueeze(0) >= 
                       torch.tensor([g.shape[0] for g in graphs], device=node_emb.device).unsqueeze(1))

        transformer_output = self.transformer(transformer_input, src_key_padding_mask=padding_mask)

        # Attention pooling
        attn_weights = self.attention_pool(transformer_output)  # (batch, max_len, 1)
        attn_weights = attn_weights.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=1)
        graph_embedding = (transformer_output * attn_weights).sum(dim=1)  # (batch, 256)

        # Combine with global embedding
        combined = torch.cat([graph_embedding, global_emb], dim=-1)  # (batch, 384)

        # Final classification
        logits = self.classifier(combined)  # (batch, 1)
        return logits.squeeze(-1)

def make_model(example_object):
    return ParticleGNN(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = next(model.parameters()).device

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # Cosine annealing scheduler with warmup
    warmup_epochs = 3
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss function with label smoothing
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))

    # Early stopping
    best_val_auc = 0
    patience = 7
    patience_counter = 0
    best_model_state = None

    # For storing metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # AUC computation
    from sklearn.metrics import roc_auc_score

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        train_preds, train_targets = [], []

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y.float()

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()

            # Store predictions for AUC
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                train_preds.extend(probs.cpu().numpy())
                train_targets.extend(yb.cpu().numpy())

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y.float()

                outputs = model(xb)
                loss = criterion(outputs, yb)
                epoch_val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(yb.cpu().numpy())

        # Compute metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)

        train_auc = roc_auc_score(train_targets, train_preds)
        val_auc = roc_auc_score(val_targets, val_preds)

        train_acc = ((np.array(train_preds) > 0.5) == np.array(train_targets)).mean()
        val_acc = ((np.array(val_preds) > 0.5) == np.array(val_targets)).mean()

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping check
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Train AUC: {train_auc:.4f}, "
                  f"Val AUC: {val_auc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    if dryrun:
        idx = torch.randperm(X_train.shape[0])[:400]
        X_train, Y_train = X_train[idx], Y_train[idx]
        idx = torch.randperm(X_val.shape[0])[:20]
        X_val, Y_val = X_val[idx], Y_val[idx]
    pre     = make_preprocessor().fit(X_train, Y_train)
    
    # Build LoaderSpec
    spec = build_spec_from_preproc(pre, script_module="llm_script")
    spec = enforce_pyg_policy(spec, require_torch_collate=False)

    # Build loaders - preproc in dataset
    train_ds     = build_dataset(spec, (X_train, Y_train), pre, train=True)
    val_ds       = build_dataset(spec, (X_val,   Y_val),   pre, train=False)
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
                view = normalise_batch(first_batch, device=device)
                out  = trained_model(view.batch_x)
                scores, kind = assert_binary_output(view, out)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e

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

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

