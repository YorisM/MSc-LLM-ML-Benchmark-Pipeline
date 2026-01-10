
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
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.special import logsumexp
from typing import Tuple, List, Optional

#  -------- (OPTIONAL) CUSTOM DATASET  --------
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        self.X = pre.transform(X) if pre is not None else X
        self.y = y
        if isinstance(self.X, list):
            self.X = [x if torch.is_tensor(x) else torch.as_tensor(x) for x in self.X]
        elif not torch.is_tensor(self.X):
            self.X = torch.as_tensor(self.X)
        if not torch.is_tensor(self.y):
            self.y = torch.as_tensor(self.y)

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.global_means = None
        self.global_stds = None
        self.particle_means = None
        self.particle_stds = None
        self.obj_type_means = None
        self.obj_type_stds = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:CustomDataset",
            "dataset_kwargs": {},
            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 128,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True if torch.cuda.is_available() else False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False, "batch_size": 256}
        }

    def fit(self, X, y=None):
        X_np = X.numpy() if torch.is_tensor(X) else X

        # Global features: ET_miss, phi_ET_miss
        self.global_means = np.mean(X_np[:, :2], axis=0)
        self.global_stds = np.std(X_np[:, :2], axis=0) + 1e-8

        # Particle features: Extract all non-zero objects
        particle_features = []
        obj_type_features = []
        for i in range(X_np.shape[0]):
            for j in range(18):  # Max 18 objects
                idx = 2 + j * 5
                obj_type = X_np[i, idx]
                if obj_type > 0:  # Non-zero object
                    particle_features.append(X_np[i, idx+1:idx+5])  # E, pT, eta, phi
                    obj_type_features.append(obj_type)

        particle_features = np.array(particle_features)
        obj_type_features = np.array(obj_type_features)

        self.particle_means = np.mean(particle_features, axis=0)
        self.particle_stds = np.std(particle_features, axis=0) + 1e-8

        # For object types, use one-hot encoding
        self.obj_type_means = np.mean(obj_type_features)
        self.obj_type_stds = np.std(obj_type_features) + 1e-8

        return self

    def transform(self, X):
        X_np = X.numpy() if torch.is_tensor(X) else X
        n_events = X_np.shape[0]
        processed_events = []

        for i in range(n_events):
            # Extract global features and normalize
            global_features = X_np[i, :2]
            global_features = (global_features - self.global_means) / self.global_stds

            # Extract particles
            particles = []
            obj_types = []
            for j in range(18):
                idx = 2 + j * 5
                obj_type = X_np[i, idx]
                if obj_type > 0:  # Non-zero particle
                    # Normalize object type
                    norm_obj_type = (obj_type - self.obj_type_means) / self.obj_type_stds

                    # Extract and normalize kinematic features
                    kin_features = X_np[i, idx+1:idx+5]
                    kin_features = (kin_features - self.particle_means) / self.particle_stds

                    # Add log features for better numerical stability
                    log_E = np.log1p(X_np[i, idx+1]) / 10.0
                    log_pT = np.log1p(X_np[i, idx+2]) / 10.0

                    # Combine features
                    particle_feat = np.concatenate([
                        [norm_obj_type],
                        kin_features,
                        [log_E, log_pT]
                    ])

                    particles.append(particle_feat)
                    obj_types.append(obj_type)

            if len(particles) == 0:
                # Handle events with no particles (shouldn't happen)
                particles = [np.zeros(7)]
                obj_types = [0]

            # Convert to tensors
            particles_tensor = torch.FloatTensor(np.array(particles))  # [n_particles, 7]

            # Create edge features: invariant mass and deltaR
            n_particles = len(particles)
            edge_features = []
            edge_indices = []

            for p1 in range(n_particles):
                for p2 in range(p1+1, n_particles):
                    # Extract original (unnormalized) kinematics for physics calculations
                    idx1 = 2 + (p1 * 5) if p1 < 18 else 0
                    idx2 = 2 + (p2 * 5) if p2 < 18 else 0

                    # Get 4-vectors
                    E1 = X_np[i, idx1+1] if p1 < 18 else 0
                    pT1 = X_np[i, idx1+2] if p1 < 18 else 0
                    eta1 = X_np[i, idx1+3] if p1 < 18 else 0
                    phi1 = X_np[i, idx1+4] if p1 < 18 else 0

                    E2 = X_np[i, idx2+1] if p2 < 18 else 0
                    pT2 = X_np[i, idx2+2] if p2 < 18 else 0
                    eta2 = X_np[i, idx2+3] if p2 < 18 else 0
                    phi2 = X_np[i, idx2+4] if p2 < 18 else 0

                    # Calculate invariant mass (in GeV)
                    # First convert MeV to GeV
                    E1_gev, pT1_gev = E1 / 1000.0, pT1 / 1000.0
                    E2_gev, pT2_gev = E2 / 1000.0, pT2 / 1000.0

                    # Calculate momentum components
                    px1 = pT1_gev * np.cos(phi1)
                    py1 = pT1_gev * np.sin(phi1)
                    pz1 = pT1_gev * np.sinh(eta1)

                    px2 = pT2_gev * np.cos(phi2)
                    py2 = pT2_gev * np.sin(phi2)
                    pz2 = pT2_gev * np.sinh(eta2)

                    # Invariant mass
                    E_sum = E1_gev + E2_gev
                    px_sum = px1 + px2
                    py_sum = py1 + py2
                    pz_sum = pz1 + pz2

                    inv_mass2 = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
                    inv_mass = np.sqrt(max(0, inv_mass2))

                    # DeltaR
                    delta_eta = eta1 - eta2
                    delta_phi = phi1 - phi2
                    # Handle phi periodicity
                    while delta_phi > np.pi:
                        delta_phi -= 2*np.pi
                    while delta_phi < -np.pi:
                        delta_phi += 2*np.pi

                    deltaR = np.sqrt(delta_eta**2 + delta_phi**2)

                    # Normalize features
                    inv_mass_norm = inv_mass / 1000.0  # Normalize by 1 TeV
                    deltaR_norm = deltaR / 10.0

                    edge_features.append([inv_mass_norm, deltaR_norm])
                    edge_indices.append([p1, p2])
                    edge_indices.append([p2, p1])  # Undirected graph

            if len(edge_features) == 0:
                edge_features = [[0, 0]]
                edge_indices = [[0, 0]]

            edge_features_tensor = torch.FloatTensor(np.array(edge_features))  # [n_edges, 2]
            edge_index_tensor = torch.LongTensor(np.array(edge_indices)).t().contiguous()  # [2, n_edges*2]

            # Create graph data structure
            graph_data = {
                'x': particles_tensor,  # [n_particles, 7]
                'edge_index': edge_index_tensor,  # [2, n_edges*2]
                'edge_attr': edge_features_tensor,  # [n_edges, 2]
                'global_features': torch.FloatTensor(global_features),  # [2]
                'num_particles': torch.tensor(n_particles, dtype=torch.long)
            }

            processed_events.append(graph_data)

        return processed_events

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class ParticleTransformer(nn.Module):
    """Transformer for particle sequences"""
    def __init__(self, d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(7, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, 7]
        x = self.input_proj(x)  # [batch_size, seq_len, d_model]

        # Apply transformer
        if mask is not None:
            # Convert to attention mask
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            attn_mask = attn_mask.expand(-1, -1, x.size(1), -1)
        else:
            attn_mask = None

        x = self.transformer(x, src_key_padding_mask=mask)  # [batch_size, seq_len, d_model]
        x = self.output_proj(x)  # [batch_size, seq_len, d_model]
        return x

class EdgeAwareGNN(nn.Module):
    """Graph Neural Network with edge features"""
    def __init__(self, node_dim=7, edge_dim=2, hidden_dim=128, num_layers=4):
        super().__init__()
        self.num_layers = num_layers

        # Input projections
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(EdgeConvLayer(hidden_dim))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index, edge_attr):
        # x: [batch_size*nodes, node_dim]
        # edge_index: [2, batch_size*edges]
        # edge_attr: [batch_size*edges, edge_dim]

        # Project inputs
        h = self.node_proj(x)  # [batch_size*nodes, hidden_dim]
        e = self.edge_proj(edge_attr)  # [batch_size*edges, hidden_dim]

        # Apply GNN layers
        for layer in self.gnn_layers:
            h = layer(h, edge_index, e)  # [batch_size*nodes, hidden_dim]

        # Output projection
        h = self.output_proj(h)  # [batch_size*nodes, hidden_dim]
        return h

class EdgeConvLayer(nn.Module):
    """Single edge convolution layer"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # x: [N, hidden_dim]
        # edge_index: [2, E]
        # edge_attr: [E, hidden_dim]

        row, col = edge_index  # [E], [E]

        # Aggregate messages from neighbors
        msg = torch.cat([x[col], edge_attr], dim=1)  # [E, 2*hidden_dim]
        msg = self.edge_mlp(msg)  # [E, hidden_dim]

        # Aggregate messages to nodes
        out = torch.zeros_like(x)
        out = out.index_add_(0, row, msg)  # [N, hidden_dim]

        # Update node features
        out = self.node_mlp(out)  # [N, hidden_dim]

        # Residual connection
        out = out + x

        return out

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Determine input dimensions from sample
        if isinstance(sample_object, dict):
            # Graph mode
            node_dim = sample_object['x'].shape[-1] if hasattr(sample_object['x'], 'shape') else 7
            edge_dim = sample_object['edge_attr'].shape[-1] if hasattr(sample_object['edge_attr'], 'shape') else 2
            global_dim = sample_object['global_features'].shape[-1] if hasattr(sample_object['global_features'], 'shape') else 2
        else:
            # Dense mode
            node_dim = 7
            edge_dim = 2
            global_dim = 2

        # Model hyperparameters
        hidden_dim = 256
        transformer_dim = 128
        gnn_dim = 128

        # Particle Transformer
        self.transformer = ParticleTransformer(
            d_model=transformer_dim,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.1
        )

        # Edge-aware GNN
        self.gnn = EdgeAwareGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=gnn_dim,
            num_layers=4
        )

        # Global feature processor
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(transformer_dim + gnn_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch_x):
        if isinstance(batch_x, dict):
            # Unpack graph data
            x = batch_x['x']  # [total_nodes, node_dim]
            edge_index = batch_x['edge_index']  # [2, total_edges]
            edge_attr = batch_x['edge_attr']  # [total_edges, edge_dim]
            global_features = batch_x['global_features']  # [batch_size, global_dim]
            num_particles = batch_x['num_particles']  # [batch_size]

            batch_size = global_features.shape[0]

            # Process particles with GNN
            gnn_out = self.gnn(x, edge_index, edge_attr)  # [total_nodes, gnn_dim]

            # Process particles with Transformer (need to reshape)
            # Create batch indices for particles
            batch_indices = []
            for i, n in enumerate(num_particles):
                batch_indices.extend([i] * n.item())
            batch_indices = torch.tensor(batch_indices, device=x.device)

            # Pad sequences for transformer
            max_particles = num_particles.max().item()
            padded_sequences = []
            masks = []

            start_idx = 0
            for n in num_particles:
                n_val = n.item()
                seq = x[start_idx:start_idx+n_val]  # [n, node_dim]

                # Pad sequence
                if n_val < max_particles:
                    pad_len = max_particles - n_val
                    padding = torch.zeros(pad_len, seq.shape[1], device=seq.device)
                    seq = torch.cat([seq, padding], dim=0)

                padded_sequences.append(seq)
                masks.append(torch.cat([
                    torch.zeros(n_val, dtype=torch.bool, device=x.device),
                    torch.ones(max_particles - n_val, dtype=torch.bool, device=x.device)
                ]))
                start_idx += n_val

            transformer_input = torch.stack(padded_sequences)  # [batch_size, max_particles, node_dim]
            mask = torch.stack(masks)  # [batch_size, max_particles]

            transformer_out = self.transformer(transformer_input, mask)  # [batch_size, max_particles, transformer_dim]

            # Attention pooling for transformer output
            trans_pool_weights = self.attention_pool(transformer_out)  # [batch_size, max_particles, 1]
            trans_pool_weights = trans_pool_weights.masked_fill(mask.unsqueeze(-1), float('-inf'))
            trans_pool_weights = F.softmax(trans_pool_weights, dim=1)
            trans_pooled = (transformer_out * trans_pool_weights).sum(dim=1)  # [batch_size, transformer_dim]

            # Pool GNN outputs per graph
            gnn_pooled = []
            start_idx = 0
            for n in num_particles:
                n_val = n.item()
                graph_nodes = gnn_out[start_idx:start_idx+n_val]  # [n, gnn_dim]
                graph_pooled = graph_nodes.mean(dim=0)  # [gnn_dim]
                gnn_pooled.append(graph_pooled)
                start_idx += n_val

            gnn_pooled = torch.stack(gnn_pooled)  # [batch_size, gnn_dim]

            # Process global features
            global_processed = self.global_proj(global_features)  # [batch_size, hidden_dim]

            # Combine all features
            combined = torch.cat([
                trans_pooled,  # [batch_size, transformer_dim]
                gnn_pooled,    # [batch_size, gnn_dim]
                global_processed  # [batch_size, hidden_dim]
            ], dim=1)  # [batch_size, transformer_dim + gnn_dim + hidden_dim]

            # Final classification
            output = self.classifier(combined)  # [batch_size, 1]

        else:
            # Dense mode (backup)
            output = torch.zeros(batch_x.shape[0], 1, device=batch_x.device)

        return output.squeeze(-1)  # [batch_size]

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 100

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Early stopping
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_model_state = None

    # Metrics tracking
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_aucs = []
    val_aucs = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_probs = []
        train_targets = []

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            # Move to device
            if isinstance(batch_x, list):
                # Graph mode
                for key in batch_x[0].keys():
                    if torch.is_tensor(batch_x[0][key]):
                        batch_x[0][key] = batch_x[0][key].to(device)
                batch_y = batch_y.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_x[0])
                loss = criterion(outputs, batch_y.float())

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Calculate accuracy
                preds = (torch.sigmoid(outputs) > 0.5).float()
                train_correct += (preds == batch_y).sum().item()
                train_total += batch_y.size(0)

                # Store for AUC
                train_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                train_targets.extend(batch_y.cpu().numpy())

                train_loss += loss.item() * batch_y.size(0)

        # Calculate training metrics
        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Calculate training AUC
        try:
            train_auc = roc_auc_score(train_targets, train_probs)
        except:
            train_auc = 0.5

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_probs = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                if isinstance(batch_x, list):
                    # Graph mode
                    for key in batch_x[0].keys():
                        if torch.is_tensor(batch_x[0][key]):
                            batch_x[0][key] = batch_x[0][key].to(device)
                    batch_y = batch_y.to(device)

                    # Forward pass
                    outputs = model(batch_x[0])
                    loss = criterion(outputs, batch_y.float())

                    # Calculate accuracy
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    val_correct += (preds == batch_y).sum().item()
                    val_total += batch_y.size(0)

                    # Store for AUC
                    val_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())

                    val_loss += loss.item() * batch_y.size(0)

        # Calculate validation metrics
        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Calculate validation AUC
        try:
            val_auc = roc_auc_score(val_targets, val_probs)
        except:
            val_auc = 0.5

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        # Print progress
        if epoch % 5 == 0:
            print(f'Epoch {epoch:03d}: '
                  f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
                  f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

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

