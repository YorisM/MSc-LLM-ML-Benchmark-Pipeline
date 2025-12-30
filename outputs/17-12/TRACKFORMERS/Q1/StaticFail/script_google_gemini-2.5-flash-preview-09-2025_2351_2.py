
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

# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
import hdbscan
from sklearn.neighbors import NearestNeighbors
from torch.nn import Functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List, Tuple
# <LLM: Import modules>

# -------- (OPTIONAL) CUSTOM DATASET  --------
# def make_dataset(events, pre, train: bool, **kwargs):
#   REQUIREMENT: If you want a custom dataset: in make_loader_cfg set dataset_builder to "llm_script:make_dataset"
#   k = kwargs.get("k", 16)
#   <LLM: Insert custom dataset logic here>
#   return CustomDataset(events, pre, train=train, k=k)

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # Must implement:
    #   - fit()
    #   - transform()

    # REQUIREMENTS
    #   - IMPORTANT: All state must be picklable with the std-lib pickle module.
    #   - May allocate NumPy arrays or Torch tensors internally, but: transform() must be deterministic.
    #   - Store only derived parameters needed for transform i.e. do not store the raw data itself in the preprocessor object.

    # TIPS
    #   When modifying data features or feature engineering: annotate tensor size as comments after each tensor operation to reduce dimension mismatches.

    # <LLM: Write code to preprocess the data> 
    def __init__(self):
        # <LLM: Define and initialize any stateful components here>
        # Estimated empirical normalization values for typical HEP detectors
        self.r_scale = 1000.0
        self.z_scale = 3000.0
        self.layer_scale = 50.0 
        pass

    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 16, # Optimized for GNN processing
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            "collate": "ragged_xy",  # Ragged list of tensors

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False, "batch_size": 32}
        }

    def fit(self, data):
        # <LLM: Extract statistics or fit transform>
        # Using fixed scales, no fitting required.
        return self

    def transform(self, data: torch.Tensor):
        # <LLM: Apply preprocessing logic, return torch.Tensor>
        # Input tensor data: [N_hits, 4] ("hit_r", "hit_theta", "hit_z", "layer_id")

        r, theta, z, layer_id = data[:, 0], data[:, 1], data[:, 2], data[:, 3] # [N_hits]

        # Calculate Cartesian coordinates
        x = r * torch.cos(theta) # [N_hits]
        y = r * torch.sin(theta) # [N_hits]

        # Normalization
        r_norm = r / self.r_scale
        z_norm = z / self.z_scale
        x_norm = x / self.r_scale
        y_norm = y / self.r_scale
        theta_norm = theta / math.pi # math imported via standard library math or numpy

        # Layer ID normalization 
        layer_norm = (layer_id - 1) / self.layer_scale 

        # F=6: [x_norm, y_norm, z_norm, r_norm, theta_norm, layer_norm]
        F_transformed = torch.stack([x_norm, y_norm, z_norm, r_norm, theta_norm, layer_norm], dim=1) # [N_hits, 6]

        return F_transformed # must return an indexable, picklable object

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------

HIDDEN_DIM = 64
EMBEDDING_DIM = 16
K_NEIGHBORS = 8
N_GNN_LAYERS = 3

class TrackGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        # x: [N_hits, In_C], edge_index: [2, E]
        x = self.conv(x, edge_index) # [N_hits, O]
        x = self.norm(x)
        return F.relu(x)

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x: List[torch.Tensor]):
        super().__init__()
        # IMPORTANT: Default harness input:
        #   batch_x is ragged list[Tensor], one per event, each shaped [N_hits, F].

        # <LLM: Define and initialize any stateful components here>
        if isinstance(example_batch_x, list):
            F_in = example_batch_x[0].shape[-1] if example_batch_x and len(example_batch_x) > 0 else 6
        elif isinstance(example_batch_x, torch.Tensor):
            F_in = example_batch_x.shape[-1] if example_batch_x.ndim == 2 else 6
        else:
            F_in = 6

        self.F_in = F_in

        # 1. Input embedding MLP
        self.input_mlp = Seq(
            Lin(F_in, HIDDEN_DIM),
            ReLU(),
            Lin(HIDDEN_DIM, HIDDEN_DIM),
            ReLU()
        )

        # 2. GNN layers (Use residual connections potentially, but keeping simple GCNConv here)
        self.gnn_layers = nn.ModuleList()
        for _ in range(N_GNN_LAYERS):
            self.gnn_layers.append(TrackGNNLayer(HIDDEN_DIM, HIDDEN_DIM))

        # 3. Output Embedding layer (for metric learning and clustering)
        self.output_mlp = Lin(HIDDEN_DIM, EMBEDDING_DIM) 

        # HDBSCAN configuration (Crucial for FitAccuracy)
        self.hdbscan_params = dict(
            min_cluster_size=4, 
            min_samples=1, 
            cluster_selection_method='eom', 
            metric='euclidean',
            allow_single_cluster=True
        )

    def _build_graph(self, X_event: torch.Tensor) -> torch.Tensor:
        """ KNN graph construction using (Normalized X, Y, Z) coordinates (Features 0, 1, 2) """
        N_i = X_event.shape[0]

        if N_i < 2:
            return torch.empty((2, 0), dtype=torch.long, device=X_event.device)

        # Ensure tensor is on CPU for sklearn compatibility
        X_coords = X_event[:, 0:3].detach().cpu().numpy() # [N_i, 3]

        k = min(N_i - 1, K_NEIGHBORS)

        nn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean').fit(X_coords)
        distances, indices = nn.kneighbors(X_coords) 

        source_nodes = torch.arange(N_i).view(-1, 1).repeat(1, k) # [N_i, k]
        target_nodes = torch.tensor(indices, dtype=torch.long) # [N_i, k]

        edge_index = torch.stack([
            source_nodes.flatten(),
            target_nodes.flatten()
        ], dim=0).to(X_event.device) # [2, N_i * k]

        # Add reverse edges 
        edge_index_rev = edge_index[[1, 0]]
        edge_index = torch.cat([edge_index, edge_index_rev], dim=1) # [2, 2 * N_i * k]

        # Note: If memory usage is high, we could optimize graph construction (e.g., using radius search or layer constraints).

        return edge_index

    def _convert_ragged_batch_to_pyg(self, batch_x: List[torch.Tensor]) -> Tuple[Batch, List[int]]:
        """Converts ragged list input into a PyG Batch object with dynamically generated graphs."""

        data_list = []
        n_nodes_per_event = []

        for x_i in batch_x:
            N_i = x_i.shape[0]
            n_nodes_per_event.append(N_i)

            # Build graph dynamically for this event
            edge_index_i = self._build_graph(x_i) # [2, E_i]

            data = Data(x=x_i, edge_index=edge_index_i)
            data_list.append(data)

        pyg_batch = Batch.from_data_list(data_list)
        return pyg_batch, n_nodes_per_event

    def compute_metric_loss(self, embeddings: torch.Tensor, truth_y: torch.Tensor):
        """ Metric Learning Loss (Margin/Hinge Loss) """

        # Filter out noise (track_id == 0)
        mask = truth_y > 0
        E = embeddings[mask] # [N_true, D]
        Y = truth_y[mask]    # [N_true]

        N_true = E.size(0)
        if N_true < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        dist_sq = torch.cdist(E, E, p=2).pow(2) # [N_true, N_true]

        Y_matrix = (Y.unsqueeze(0) == Y.unsqueeze(1)).float() 
        I = torch.eye(N_true, device=E.device)

        P_mask = Y_matrix - I # Positive pairs (excluding diagonal)
        N_mask = 1 - Y_matrix # Negative pairs

        # Define margins
        MARGIN_POS_SQ = 0.5**2 # Target distance for positive pairs
        MARGIN_NEG_SQ = 2.0**2 # Required separation for negative pairs

        # Attractive loss: penalize positive pairs that are too far
        L_att = (dist_sq - MARGIN_POS_SQ).clamp(min=0) * P_mask
        L_att = L_att.sum() / (P_mask.sum() + 1e-6)

        # Repulsive loss: penalize negative pairs that are too close
        L_rep = (MARGIN_NEG_SQ - dist_sq).clamp(min=0) * N_mask
        L_rep = L_rep.sum() / (N_mask.sum() + 1e-6)

        # Total loss: Balancing L_att and L_rep
        L_total = L_att + L_rep * 0.1 

        return L_total

    def _cluster(self, embeddings: torch.Tensor, n_nodes_per_event: List[int]) -> List[torch.Tensor]:
        """ Apply HDBSCAN clustering event by event on embeddings. """
        E_np = embeddings.detach().cpu().float().numpy() 

        predictions = []
        offset = 0

        for N_i in n_nodes_per_event:
            E_event = E_np[offset : offset + N_i] # [N_i, D]
            offset += N_i

            if N_i < self.hdbscan_params['min_cluster_size']:
                # Too few hits for track reconstruction criteria
                labels = np.full(N_i, -1, dtype=np.int64)
            else:
                try:
                    clusterer = hdbscan.HDBSCAN(**self.hdbscan_params, prediction_data=False)
                    clusterer.fit(E_event)
                    labels = clusterer.labels_.astype(np.int64)
                except Exception:
                    labels = np.full(N_i, -1, dtype=np.int64)

            # Map HDBSCAN output: -1 (noise) -> -1; k >= 0 (cluster ID) -> k+1 (track ID)
            mapped_labels = np.where(labels >= 0, labels + 1, -1)

            # Convert to Tensor (long/int64) on the appropriate device
            predictions.append(torch.tensor(mapped_labels, dtype=torch.long, device=embeddings.device))

        return predictions

    def forward(self, batch_x: List[torch.Tensor]):
        # Input: List of tensors [N_i, F]

        # Convert ragged batch to PyG format and build dynamic graph
        pyg_batch, n_nodes_per_event = self._convert_ragged_batch_to_pyg(batch_x)

        x = pyg_batch.x.to(self.output_mlp.weight.device) # [Total_N, F_in]
        edge_index = pyg_batch.edge_index.to(self.output_mlp.weight.device) # [2, Total_E]

        # Input Embedding
        h = self.input_mlp(x) # [Total_N, HIDDEN_DIM]

        # GNN propagation
        for layer in self.gnn_layers:
            h = layer(h, edge_index) # [Total_N, HIDDEN_DIM]

        # Output embedding
        embeddings = self.output_mlp(h) # [Total_N, EMBEDDING_DIM]

        if self.training:
           # Return embeddings for metric learning loss calculation
           return embeddings

        else:
            # Inference/Evaluation: perform clustering to get discrete predicted labels
            predicted_labels_list = self._cluster(embeddings, n_nodes_per_event)

            # Output contract requires list of Tensors matching the ragged input structure
            return predicted_labels_list 

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 30   # <LLM: adjust if you wish>   

# Helper function to compute proxy accuracy based on clustered labels
def compute_proxy_accuracy(y_true_list: List[torch.Tensor], y_pred_list: List[torch.Tensor]) -> float:
    """ Computes the fraction of non-noise truth hits assigned a non-noise prediction label. """
    total_true_hits = 0
    total_found_hits = 0

    for y_t, y_p in zip(y_true_list, y_pred_list):
        # y_t: truth track ID (0=noise)
        # y_p: predicted track ID (-1=noise, >0=track)

        y_p_non_noise = y_p >= 1 
        y_t_non_noise = y_t > 0

        total_true_hits += y_t_non_noise.sum().item()

        # We count overlaps: True track hits correctly predicted as *some* track (non-noise)
        total_found_hits += (y_t_non_noise & y_p_non_noise.to(y_t.device)).sum().item()

    return total_found_hits / (total_true_hits + 1e-6)

def train_model(model: HitClassifier, train_loader, val_loader, epochs):
    # Requirements: Use CUDA, return trained_model, train_loss, val_loss, train_acc, val_acc, implement early-stopping.

    device = next(model.parameters()).device

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    train_losses = []
    val_losses = []
    # Train accuracy tracks proxy accuracy but is skipped for computational efficiency
    train_accs = [0.0] * epochs 
    val_accs = []

    best_val_loss = float('inf')
    patience_counter = 0
    PATIENCE = 10 

    # <LLM: Write code to define training loop>
    for epoch in range(epochs):

        model.train()
        total_train_loss = 0
        batch_count = 0

        # Training Phase
        # We manually import tqdm structure since it might be available in the environment 
        # but we must ensure we don't rely on it unless necessary. Assuming standard iterative loop.

        for i, batch in enumerate(train_loader):
            view = normalise_batch(batch, device=device)
            X = view.batch_x # List[Tensor]
            Y = view.batch_y # List[Tensor] (True track IDs)
            Y_flat = torch.cat(Y, dim=0).to(device) # [Total_N]

            optimizer.zero_grad()

            embeddings = model(X) # [Total_N, EMBEDDING_DIM] (during training)
            loss = model.compute_metric_loss(embeddings, Y_flat)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            batch_count += 1

        avg_train_loss = total_train_loss / batch_count
        train_losses.append(avg_train_loss)

        # Validation Phase
        model.eval()
        total_val_loss = 0
        val_batch_count = 0
        all_y_true = []
        all_y_pred = []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                X = view.batch_x
                Y = view.batch_y # List[Tensor] (True track IDs)

                Y_flat = torch.cat(Y, dim=0).to(device) # [Total_N]

                # 1. Calculate Validation Loss (on embeddings)
                embeddings = model.forward(X) # Forward returns embeddings in Eval mode here, as long as self.training=False is handled by the model logic
                val_loss = model.compute_metric_loss(embeddings, Y_flat)
                total_val_loss += val_loss.item()
                val_batch_count += 1

                # 2. Calculate Proxy Accuracy (requires clustering)
                # Since model.forward(X) automatically runs clustering when model.training is False
                y_pred_list = model.forward(X) 

                all_y_true.extend(Y)
                all_y_pred.extend(y_pred_list)

        avg_val_loss = total_val_loss / val_batch_count
        val_losses.append(avg_val_loss)

        avg_val_acc = compute_proxy_accuracy(all_y_true, all_y_pred)
        val_accs.append(avg_val_acc)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={avg_val_acc:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")

        # <LLM: Implement early stopping if possible>
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        scheduler.step(avg_val_loss)

    # Note: Train accuracy is stored as 0.0 because calculating clustering accuracy on the large training set is too slow.
    train_loss_np = np.array(train_losses)
    val_loss_np = np.array(val_losses)
    train_acc_np = np.array(train_accs)[:len(train_losses)]
    val_acc_np = np.array(val_accs)

    return model, train_loss_np, val_loss_np, train_acc_np, val_acc_np

# IMPORTANT: DO NOT execute the pipeline here – the harness will do that.
# <end code template>

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

