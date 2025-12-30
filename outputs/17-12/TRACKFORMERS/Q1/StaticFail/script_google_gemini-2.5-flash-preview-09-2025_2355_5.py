
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
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import knn_graph
from sklearn.preprocessing import StandardScaler
import numpy as np
import hdbscan
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple 

# Helper function to convert ragged list to PyG Batch object representation
def list_to_pyg_batch(batch_x: List[torch.Tensor], device) -> Tuple[torch.Tensor, torch.Tensor]:
    # batch_x: list of [N_i, F] tensors
    # Concatenate all nodes
    x = torch.cat(batch_x, dim=0).to(device, non_blocking=True) # [N_total, F]

    # Create batch vector
    batch_indices = []
    for i, hits in enumerate(batch_x):
        N_i = hits.size(0)
        batch_indices.append(torch.full((N_i,), i, dtype=torch.long, device=device))

    batch_idx_tensor = torch.cat(batch_indices, dim=0) # [N_total]

    return x, batch_idx_tensor

# Custom Metric Learning Loss: Supervised Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.15):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-6

    def forward(self, embeddings: torch.Tensor, truth_labels: List[torch.Tensor]):

        y_total = torch.cat(truth_labels, dim=0).long().view(-1) # [N_total]

        # L2 Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1) # [N_total, D]

        # Mask for non-noise hits (Q filter, track_id > 0)
        non_noise_mask = y_total > 0

        if torch.sum(non_noise_mask) < 2:
            # If less than 2 track hits in the batch, loss is zero
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        embeddings_Q = embeddings[non_noise_mask] # [N_Q, D]
        y_Q = y_total[non_noise_mask] # [N_Q]

        # Similarity matrix S_QN: [N_Q, N_total] 
        S_QN = torch.matmul(embeddings_Q, embeddings.transpose(0, 1))

        # Logits
        logits = S_QN / self.temperature # [N_Q, N_total]

        # Similarity to positives mask M_P
        y_Q_matrix = y_Q.unsqueeze(1) # [N_Q, 1]
        y_N_matrix = y_total.unsqueeze(0) # [1, N_total]

        M_P = (y_Q_matrix == y_N_matrix).float() # [N_Q, N_total]. Includes self-comparison.


        # Stability normalization
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach() # [N_Q, N_total]
        exp_logits = torch.exp(logits)

        # Denominator Z: Sum over all indices N_total
        Z = torch.sum(exp_logits, dim=1, keepdim=True) # [N_Q, 1]

        # Numerator: Sum over positive pairs P
        numerator_sum = torch.sum(exp_logits * M_P, dim=1) # [N_Q]

        log_prob = torch.log(numerator_sum / (Z.squeeze() + self.eps)) # [N_Q]

        cardinality_P = torch.sum(M_P, dim=1) # [N_Q]

        # SupCon Loss L = - mean( 1/|P(i)| * log_prob )
        weighted_log_prob = -(1.0 / (cardinality_P + self.eps)) * log_prob # [N_Q]

        loss = torch.mean(weighted_log_prob)

        return loss


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

    # <LLM: Write code to preprocess the data> 
    def __init__(self):
        # <LLM: Define and initialize any stateful components here>
        self.scaler = StandardScaler()
        self.fitted = False
        pass

    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",    
            "batch_size": 16, 
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            "collate": "ragged_xy",  

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False, "batch_size": 32} 
        }

    def _convert_and_process(self, data_list):
        # data_list is a list of [N_i, 4] Tensors (r, theta, z, layer_id)
        processed_data = []

        for event_hits in data_list:
            if isinstance(event_hits, np.ndarray):
                event_hits = torch.from_numpy(event_hits)

            # Input: [N, 4] (r, theta, z, L)
            r, theta, z, L = event_hits.T 

            # Convert to Cartesian (x, y, z)
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)

            # Features: (x, y, z, r, L). [N_i, 5]
            features = torch.stack([x, y, z, r, L], dim=1).cpu().numpy()

            processed_data.append(features)

        return processed_data

    def fit(self, data: List[torch.Tensor]):
        # <LLM: Extract statistics or fit transform>
        features_list = self._convert_and_process(data)
        all_features = np.concatenate(features_list, axis=0) # [N_total, 5]

        # Normalize (x, y, z, r)
        self.scaler.fit(all_features[:, :4]) 

        self.fitted = True
        return self

    def transform(self, data: List[torch.Tensor]):
        # <LLM: Apply preprocessing logic, return torch.Tensor>
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transformation.")

        output_list = []
        for event_hits in data:
            if isinstance(event_hits, torch.Tensor) and event_hits.is_cuda:
                event_hits = event_hits.cpu()

            r, theta, z, L = event_hits.T 

            x = r * torch.cos(theta)
            y = r * torch.sin(theta)

            features_unscaled = torch.stack([x, y, z, r, L], dim=1).numpy() # [N, 5] 

            scaled_coords = self.scaler.transform(features_unscaled[:, :4]) # [N, 4]

            L_raw = features_unscaled[:, 4:] # [N, 1]

            processed_features = np.concatenate([scaled_coords, L_raw], axis=1) # [N, 5]

            output_list.append(torch.from_numpy(processed_features).float())

        return output_list

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()

        HIDDEN_DIM = 96 
        OUTPUT_DIM = 16 
        K_NEIGHBORS = 8

        if isinstance(example_batch_x, list) and len(example_batch_x) > 0:
            in_features = example_batch_x[0].size(1) # [N_i, 5]
        else:
            in_features = 5

        self.K_NEIGHBORS = K_NEIGHBORS
        self.OUTPUT_DIM = OUTPUT_DIM

        # 1. Initial Embedding
        self.f_in = nn.Sequential(
            nn.Linear(in_features, HIDDEN_DIM),
            nn.LeakyReLU(), 
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM)
        )
        # H: [N_total, HIDDEN_DIM]

        # 2. Iterative Graph Interaction Layers
        self.num_layers = 4
        self.gnn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.gnn_layers.append(
                pyg_nn.SAGEConv(HIDDEN_DIM, HIDDEN_DIM)
            )

        # 3. Output layer for Embeddings
        self.f_out = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, batch_x: List[torch.Tensor]):

        X_total, batch_indices = list_to_pyg_batch(batch_x, device) # X_total: [N_total, 5], batch_indices: [N_total]

        H = self.f_in(X_total) # H: [N_total, HIDDEN_DIM] 

        pos = X_total[:, :3] # Normalized (x, y, z) coordinates used for graph creation

        # Dynamic Graph Construction: k-NN within event boundaries
        edge_index = knn_graph(pos, k=self.K_NEIGHBORS, batch=batch_indices, dim=-1) # [2, E]

        # GNN Propagation
        for layer in self.gnn_layers:
            H_res = H 
            H = layer(H, edge_index) # H: [N_total, HIDDEN_DIM]
            H = H.relu()
            H = H_res + H 
            H = F.layer_norm(H, H.shape[-1:]) # Layer norm

        embeddings = self.f_out(H) # [N_total, OUTPUT_DIM]

        # --- Output / Clustering ---
        if self.training:
             # Return embeddings for metric learning loss calculation
             return embeddings
        else:
             # Inference mode: Perform clustering on normalized embeddings (CPU operation)
             embeddings_norm = F.normalize(embeddings, dim=1).detach().cpu().numpy()

             predicted_labels = []
             current_idx = 0
             N_thresh = 4 # Requirement: minimum 4 hits per valid track

             for event_x in batch_x:
                 N_i = event_x.size(0)
                 event_embeddings = embeddings_norm[current_idx: current_idx + N_i]

                 if N_i < N_thresh:
                     labels = np.full(N_i, -1, dtype=np.int64)
                 else:
                     clusterer = hdbscan.HDBSCAN(
                         min_cluster_size=N_thresh, 
                         min_samples=1, 
                         core_dist_n_jobs=-1,
                         prediction_data=False 
                     )

                     labels_hdbscan = clusterer.fit_predict(event_embeddings)

                     unique_track_ids = np.unique(labels_hdbscan)

                     # Map HDBSCAN 0, 1, 2... to 1, 2, 3... (skipping -1 for noise)
                     track_id_counter = 1
                     label_map = {}
                     for tid in unique_track_ids:
                         if tid != -1:
                             label_map[tid] = track_id_counter
                             track_id_counter += 1

                     labels = np.array([label_map.get(tid, -1) for tid in labels_hdbscan], dtype=np.int64)

                 predicted_labels.append(labels)
                 current_idx += N_i

             return torch.cat([torch.from_numpy(l) for l in predicted_labels], dim=0).long().to(device)


def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 35   
def train_model(model, train_loader, val_loader, epochs):

    criterion = ContrastiveLoss(temperature=0.15)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = [] 
    val_acc_history = []   

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 8

    # Note: split_X_y is provided by the harness environment.

    for epoch in range(epochs):

        # --- Training Loop ---
        model.train()
        running_loss = 0.0
        N_train = 0

        for batch in train_loader:
            optimizer.zero_grad()

            Xs, Ys = split_X_y(batch)
            Xs_dev = [x.to(device) for x in Xs]

            embeddings = model(Xs_dev) 

            loss = criterion(embeddings, Ys)

            if loss.requires_grad:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * len(Xs)
            N_train += len(Xs)

        epoch_train_loss = running_loss / N_train
        train_loss_history.append(epoch_train_loss)

        scheduler.step(epoch_train_loss)

        # --- Validation Loop (for loss tracking only) ---
        val_running_loss = 0.0
        N_val = 0

        # Temporarily force model into training mode to ensure embeddings are returned for loss calculation
        is_training = model.training
        model.train() 

        with torch.no_grad():
            for batch in val_loader:
                Xs, Ys = split_X_y(batch)
                Xs_dev = [x.to(device) for x in Xs]

                embeddings = model(Xs_dev) 

                loss = criterion(embeddings, Ys)
                val_running_loss += loss.item() * len(Xs)
                N_val += len(Xs)

        # Restore model training state
        if not is_training:
            model.eval()

        epoch_val_loss = val_running_loss / N_val
        val_loss_history.append(epoch_val_loss)

        # Proxy accuracy based on loss convergence
        train_acc_history.append(max(0.0, 1.0 - epoch_train_loss))
        val_acc_history.append(max(0.0, 1.0 - epoch_val_loss))

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Current LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early Stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                break

    # Ensure model is in eval mode before returning for final evaluation by harness
    model.eval()

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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

