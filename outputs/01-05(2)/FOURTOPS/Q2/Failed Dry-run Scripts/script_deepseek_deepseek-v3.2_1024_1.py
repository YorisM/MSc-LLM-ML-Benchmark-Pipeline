
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops

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
        X2 = pre.transform(X) if pre is not None else X
        if not torch.is_tensor(X2):
            X2 = torch.as_tensor(X2)
        self.X = X2.float()
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        self.y = y.long()
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

# -------------------------- START OF LLM BLOCK ------------------------------
# <start code template>
# ---------- IMPORTS ----------
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

#  -------- CUSTOM DATASET  --------
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        # Preprocessor returns graph data
        self.graph_data = pre.transform(X) if pre is not None else X
        self.y = torch.as_tensor(y).long() if not torch.is_tensor(y) else y.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.graph_data[idx], self.y[idx]

# ----------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.node_feat_mean = None
        self.node_feat_std = None
        self.edge_feat_mean = None
        self.edge_feat_std = None
        self.obj_type_stats = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 512}
        }

    def _extract_particles(self, X_batch):
        """Extract particle features from flat tensor."""
        batch_size = X_batch.shape[0]
        # Global features: E_T^miss and phi_E_T^miss
        global_feats = X_batch[:, :2]  # [B, 2]

        # Reshape object features: [B, 18, 5]
        obj_features = X_batch[:, 2:].reshape(batch_size, 18, 5)

        # Object type (integer identifier)
        obj_types = obj_features[:, :, 0]  # [B, 18]

        # Kinematic features
        E = obj_features[:, :, 1]  # Energy [B, 18]
        pT = obj_features[:, :, 2]  # Transverse momentum [B, 18]
        eta = obj_features[:, :, 3]  # Pseudorapidity [B, 18]
        phi = obj_features[:, :, 4]  # Azimuthal angle [B, 18]

        # Mask for real particles (non-zero padded)
        mask = (obj_types != 0) & (E != 0) & (pT != 0)  # [B, 18]

        return global_feats, obj_types, E, pT, eta, phi, mask

    def _compute_4vectors(self, E, pT, eta, phi):
        """Compute 4-vector components from kinematic features."""
        # Convert to GeV for numerical stability
        E_gev = E / 1000.0
        pT_gev = pT / 1000.0

        # Compute 3-momentum components
        px = pT_gev * torch.cos(phi)  # [B, N]
        py = pT_gev * torch.sin(phi)  # [B, N]
        pz = pT_gev * torch.sinh(eta)  # [B, N]

        return E_gev, px, py, pz

    def _compute_pairwise_features(self, E, pT, eta, phi, mask):
        """Compute invariant mass and deltaR for all particle pairs."""
        batch_size, max_particles = E.shape

        # Convert to 4-vectors
        E_gev, px, py, pz = self._compute_4vectors(E, pT, eta, phi)

        # Initialize edge features
        edge_features_list = []

        for b in range(batch_size):
            # Get indices of real particles in this event
            real_idx = torch.where(mask[b])[0]
            n_real = len(real_idx)

            if n_real < 2:
                # Single particle or no particles - create empty edge features
                edge_features_list.append(torch.zeros((0, 2), dtype=torch.float32))
                continue

            # Create all pairs
            i_idx, j_idx = torch.meshgrid(real_idx, real_idx, indexing='ij')
            i_idx = i_idx.flatten()
            j_idx = j_idx.flatten()

            # Remove self-pairs
            non_self = i_idx != j_idx
            i_idx = i_idx[non_self]
            j_idx = j_idx[non_self]

            # Compute deltaR
            eta_i = eta[b, i_idx]
            eta_j = eta[b, j_idx]
            phi_i = phi[b, i_idx]
            phi_j = phi[b, j_idx]

            delta_eta = eta_i - eta_j
            delta_phi = torch.remainder(phi_i - phi_j + np.pi, 2*np.pi) - np.pi
            deltaR = torch.sqrt(delta_eta**2 + delta_phi**2)

            # Compute invariant mass
            E_i = E_gev[b, i_idx]
            E_j = E_gev[b, j_idx]
            px_i = px[b, i_idx]
            px_j = px[b, j_idx]
            py_i = py[b, i_idx]
            py_j = py[b, j_idx]
            pz_i = pz[b, i_idx]
            pz_j = pz[b, j_idx]

            inv_mass = torch.sqrt(
                (E_i + E_j)**2 - 
                (px_i + px_j)**2 - 
                (py_i + py_j)**2 - 
                (pz_i + pz_j)**2
            )

            # Stack edge features
            edge_feats = torch.stack([deltaR, inv_mass], dim=1)
            edge_features_list.append(edge_feats)

        return edge_features_list

    def fit(self, X, y=None):
        """Compute normalization statistics from training data."""
        # Convert to tensor if needed
        if not torch.is_tensor(X):
            X = torch.as_tensor(X).float()

        # Extract features
        global_feats, obj_types, E, pT, eta, phi, mask = self._extract_particles(X)

        # Collect node features for normalization (only real particles)
        all_node_feats = []
        all_edge_feats = []

        batch_size = X.shape[0]
        for b in range(min(batch_size, 10000)):  # Sample for efficiency
            real_idx = torch.where(mask[b])[0]
            if len(real_idx) > 0:
                # Node features: [pT, eta, phi, E] for real particles
                node_feats_b = torch.stack([
                    pT[b, real_idx] / 1000.0,  # GeV
                    eta[b, real_idx],
                    phi[b, real_idx],
                    E[b, real_idx] / 1000.0   # GeV
                ], dim=1)
                all_node_feats.append(node_feats_b)

                # Compute edge features for this batch
                edge_feats_list = self._compute_pairwise_features(
                    E[b:b+1], pT[b:b+1], eta[b:b+1], phi[b:b+1], mask[b:b+1]
                )
                if len(edge_feats_list) > 0 and edge_feats_list[0].shape[0] > 0:
                    all_edge_feats.append(edge_feats_list[0])

        if all_node_feats:
            all_node_feats = torch.cat(all_node_feats, dim=0)
            self.node_feat_mean = all_node_feats.mean(dim=0, keepdim=True)
            self.node_feat_std = all_node_feats.std(dim=0, keepdim=True) + 1e-8

        if all_edge_feats:
            all_edge_feats = torch.cat(all_edge_feats, dim=0)
            self.edge_feat_mean = all_edge_feats.mean(dim=0, keepdim=True)
            self.edge_feat_std = all_edge_feats.std(dim=0, keepdim=True) + 1e-8

        # Object type statistics (for embedding)
        obj_types_flat = obj_types[mask].long()
        unique_types = torch.unique(obj_types_flat)
        self.num_obj_types = len(unique_types) + 1  # +1 for padding/unknown

        return self

    def transform(self, X):
        """Transform flat tensor into list of PyG Data objects."""
        if not torch.is_tensor(X):
            X = torch.as_tensor(X).float()

        batch_size = X.shape[0]
        graph_data_list = []

        # Extract features
        global_feats, obj_types, E, pT, eta, phi, mask = self._extract_particles(X)

        for b in range(batch_size):
            # Get real particles
            real_idx = torch.where(mask[b])[0]
            n_real = len(real_idx)

            if n_real == 0:
                # Create minimal graph with one dummy node
                x = torch.zeros((1, 16), dtype=torch.float32)  # [1, 16]
                edge_index = torch.zeros((2, 1), dtype=torch.long)
                edge_attr = torch.zeros((1, 2), dtype=torch.float32)
            else:
                # Node features
                obj_types_b = obj_types[b, real_idx].long()  # [N]

                # Normalized kinematic features
                pT_norm = (pT[b, real_idx] / 1000.0 - self.node_feat_mean[0, 0]) / self.node_feat_std[0, 0]
                eta_norm = (eta[b, real_idx] - self.node_feat_mean[0, 1]) / self.node_feat_std[0, 1]
                phi_norm = (phi[b, real_idx] - self.node_feat_mean[0, 2]) / self.node_feat_std[0, 2]
                E_norm = (E[b, real_idx] / 1000.0 - self.node_feat_mean[0, 3]) / self.node_feat_std[0, 3]

                # Global features (repeated for each node)
                global_etmiss = global_feats[b, 0:1].repeat(n_real, 1) / 1000.0  # GeV
                global_phi = global_feats[b, 1:2].repeat(n_real, 1)

                # Combine node features: [kinematics, obj_type_onehot, global_feats]
                obj_type_onehot = F.one_hot(obj_types_b, num_classes=self.num_obj_types).float()

                x = torch.cat([
                    pT_norm.unsqueeze(1),
                    eta_norm.unsqueeze(1),
                    phi_norm.unsqueeze(1),
                    E_norm.unsqueeze(1),
                    obj_type_onehot,
                    global_etmiss,
                    global_phi
                ], dim=1)  # [N, 4 + num_obj_types + 2]

                # Edge construction (fully connected among real particles)
                edge_index = []
                for i in range(n_real):
                    for j in range(n_real):
                        if i != j:  # Exclude self-loops
                            edge_index.append([i, j])

                if edge_index:
                    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # [2, E]

                    # Compute edge features (deltaR, invariant mass)
                    E_batch = E[b:b+1, real_idx].unsqueeze(0)
                    pT_batch = pT[b:b+1, real_idx].unsqueeze(0)
                    eta_batch = eta[b:b+1, real_idx].unsqueeze(0)
                    phi_batch = phi[b:b+1, real_idx].unsqueeze(0)
                    mask_batch = mask[b:b+1, real_idx].unsqueeze(0)

                    edge_feats_list = self._compute_pairwise_features(
                        E_batch, pT_batch, eta_batch, phi_batch, mask_batch
                    )
                    edge_attr = edge_feats_list[0] if edge_feats_list else torch.zeros((0, 2))

                    # Normalize edge features
                    if edge_attr.shape[0] > 0 and self.edge_feat_mean is not None:
                        edge_attr = (edge_attr - self.edge_feat_mean) / self.edge_feat_std
                else:
                    # No edges
                    edge_index = torch.zeros((2, 1), dtype=torch.long)
                    edge_attr = torch.zeros((1, 2), dtype=torch.float32)

            # Create PyG Data object
            from torch_geometric.data import Data
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=x.shape[0]
            )
            graph_data_list.append(data)

        return graph_data_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class ParticleAttention(nn.Module):
    """Multi-head attention for particle features."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [B, N, D]
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        out = self.norm(x + self.dropout(attn_out))
        return out

class EdgeConv(nn.Module):
    """Edge convolution operation for pairwise features."""
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # x: [N, D], edge_index: [2, E], edge_attr: [E, De]
        row, col = edge_index

        # Concatenate node features with edge features
        out = torch.cat([x[row], x[col], edge_attr], dim=-1)  # [E, 2D + De]
        out = self.mlp(out)  # [E, D]

        # Aggregate messages
        out = torch.zeros_like(x).index_add_(0, col, out)  # [N, D]
        return out

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Sample object is a PyG Data from make_view_by_lane_fourtops
        node_dim = sample_object.x.shape[1]  # Get from preprocessor
        edge_dim = sample_object.edge_attr.shape[1] if hasattr(sample_object, 'edge_attr') else 0

        # Node feature embedding
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1)
        )

        # Edge feature embedding
        if edge_dim > 0:
            self.edge_embed = nn.Sequential(
                nn.Linear(edge_dim, 64),
                nn.ReLU(),
                nn.LayerNorm(64)
            )
            edge_hidden = 64
        else:
            self.edge_embed = None
            edge_hidden = 0

        # Graph processing layers
        self.edge_convs = nn.ModuleList([
            EdgeConv(128, edge_hidden, 256),
            EdgeConv(128, edge_hidden, 256)
        ])

        self.attention_layers = nn.ModuleList([
            ParticleAttention(128, num_heads=8, dropout=0.1),
            ParticleAttention(128, num_heads=8, dropout=0.1)
        ])

        # Global pooling and classification
        self.global_pool = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Auxiliary layers
        self.node_norm = nn.LayerNorm(128)
        self.gru = nn.GRU(128, 128, batch_first=True, bidirectional=True)

    def forward(self, batch):
        # batch is a PyG Batch object
        x, edge_index, edge_attr, batch_vec = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        # Embed node features
        x = self.node_embed(x)  # [total_nodes, 128]

        # Embed edge features if available
        if self.edge_embed is not None and edge_attr is not None:
            edge_attr_emb = self.edge_embed(edge_attr)  # [total_edges, 64]
        else:
            edge_attr_emb = None

        # Process through graph layers
        for edge_conv, attn in zip(self.edge_convs, self.attention_layers):
            # Edge convolution
            if edge_attr_emb is not None:
                x_edge = edge_conv(x, edge_index, edge_attr_emb)
                x = x + x_edge  # Residual connection

            # Prepare for attention (group by graph)
            unique_batches = torch.unique(batch_vec)
            x_attn_list = []

            for b in unique_batches:
                mask = batch_vec == b
                x_b = x[mask].unsqueeze(0)  # [1, N_b, 128]

                # Apply attention
                x_b_attn = attn(x_b)
                x_attn_list.append(x_b_attn.squeeze(0))

            # Concatenate back
            x = torch.cat(x_attn_list, dim=0)  # [total_nodes, 128]
            x = self.node_norm(x)

        # Global pooling
        pooled = []
        for b in unique_batches:
            mask = batch_vec == b
            x_b = x[mask]  # [N_b, 128]

            # Multiple pooling strategies
            max_pool = x_b.max(dim=0)[0]
            mean_pool = x_b.mean(dim=0)
            sum_pool = x_b.sum(dim=0)

            # Process through GRU for sequence-aware pooling
            if x_b.shape[0] > 1:
                x_b_seq = x_b.unsqueeze(0)  # [1, N_b, 128]
                _, h_n = self.gru(x_b_seq)
                gru_pool = h_n.mean(dim=0).squeeze(0)  # [128]
            else:
                gru_pool = x_b.squeeze(0)

            # Combine pooling strategies
            combined = max_pool + mean_pool + sum_pool + gru_pool
            pooled.append(combined)

        pooled = torch.stack(pooled, dim=0)  # [batch_size, 128]

        # Final classification
        global_features = self.global_pool(pooled)
        logits = self.classifier(global_features).squeeze(-1)  # [batch_size]

        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # Move model to device
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)

    # Loss function with class weighting
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))

    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_auc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_preds, train_labels = [], []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            logits = model(batch)
            labels = batch.y.float()

            # Compute loss
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item() * batch.num_graphs

            # Store predictions for metrics
            probs = torch.sigmoid(logits).detach().cpu()
            train_preds.extend(probs.numpy())
            train_labels.extend(labels.cpu().numpy())

        # Training metrics
        train_loss = epoch_train_loss / len(train_loader.dataset)
        train_auc = roc_auc_score(train_labels, train_preds)
        train_acc = np.mean((np.array(train_preds) > 0.5) == np.array(train_labels))

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                labels = batch.y.float()

                loss = criterion(logits, labels)
                epoch_val_loss += loss.item() * batch.num_graphs

                probs = torch.sigmoid(logits).cpu()
                val_preds.extend(probs.numpy())
                val_labels.extend(labels.cpu().numpy())

        # Validation metrics
        val_loss = epoch_val_loss / len(val_loader.dataset)
        val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = np.mean((np.array(val_preds) > 0.5) == np.array(val_labels))

        # Update learning rate
        scheduler.step(val_auc)

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

# <end code template>
# ---------------------------  END OF LLM-CODE BLOCK  ---------------------------

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

    # Build batch and check
    first_batch = next(iter(train_loader))
    mode = detect_and_assert_lane_fourtops(spec, first_batch)
    view = make_view_by_lane_fourtops(mode, first_batch, device)

    # Build model
    model = make_model(view.batch_x).to(device)

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
                mode = detect_and_assert_lane_fourtops(spec, first_batch)
                view = make_view_by_lane_fourtops(mode, first_batch, device)
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

