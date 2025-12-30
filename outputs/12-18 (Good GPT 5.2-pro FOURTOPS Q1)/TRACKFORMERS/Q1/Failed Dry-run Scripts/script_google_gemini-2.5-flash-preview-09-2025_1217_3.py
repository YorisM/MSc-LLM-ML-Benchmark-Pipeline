
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, pickle, importlib, gzip, json, torch, torch_geometric, scipy 
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.llm_io import normalise_batch, assert_label_output, build_dataset, build_dataloader
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

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# <start code template>
# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
import hdbscan
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn.functional as F

# -------- (OPTIONAL) CUSTOM DATASET  --------

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # <LLM: Write code to preprocess the data> 

    def __init__(self):
        # <LLM: Define and initialize any stateful components here>
        self.scaler_r = StandardScaler()
        self.scaler_z = StandardScaler()
        self.scaler_layer = StandardScaler()

    def make_loader_cfg(self) -> dict: 
        return {
            "dataset_builder": "utils.llm_io:EventDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 32, # Use moderate batch size for efficient metric learning training
            "shuffle": True,
            "num_workers": 4, 
            "pin_memory": True,

            "collate": "ragged_xy",

            "extra_loader_kwargs": {},

            # Evaluation must use B=1 for robust per-event clustering (HDBSCAN)
            "eval_overrides": {"shuffle": False, "batch_size": 1, "num_workers": 0} 
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, 4] (r, theta, z, layer_id)

        R_list = torch.cat([X[:, 0] for X in Xs]).numpy().reshape(-1, 1) # [N_total, 1]
        Z_list = torch.cat([X[:, 2] for X in Xs]).numpy().reshape(-1, 1) # [N_total, 1]
        L_list = torch.cat([X[:, 3] for X in Xs]).numpy().reshape(-1, 1) # [N_total, 1]

        self.scaler_r.fit(R_list)
        self.scaler_z.fit(Z_list)
        self.scaler_layer.fit(L_list)

        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, 4] (r, theta, z, layer_id)

        # Convert to numpy for feature creation and scaling
        X_np = X.numpy()
        N_hits = X_np.shape[0]

        R = X_np[:, 0].reshape(-1, 1)    # [N_hits, 1]
        Theta = X_np[:, 1]               # [N_hits]
        Z = X_np[:, 2].reshape(-1, 1)    # [N_hits, 1]
        L = X_np[:, 3].reshape(-1, 1)    # [N_hits, 1]

        # 1. Coordinate transformation (Cartesian)
        X_cart = R * np.cos(Theta).reshape(-1, 1) # [N_hits, 1]
        Y_cart = R * np.sin(Theta).reshape(-1, 1) # [N_hits, 1]
        Z_cart = Z # [N_hits, 1]

        # 2. Scaling (applying stored parameters)
        R_scaled = self.scaler_r.transform(R)     # [N_hits, 1]
        Z_scaled = self.scaler_z.transform(Z)     # [N_hits, 1]
        L_scaled = self.scaler_layer.transform(L) # [N_hits, 1]

        # 3. Normalize Theta
        Theta_norm = Theta.reshape(-1, 1) / np.pi # [N_hits, 1] (normalizes [-pi, pi] to [-1, 1])

        # Combine features: (X, Y, Z, R_scaled, Theta_norm, Z_scaled, L_scaled)
        F_out = np.concatenate([
            X_cart, Y_cart, Z_cart,               # 3 features (Cartesian)
            R_scaled, Theta_norm, Z_scaled,       # 3 features (Normalized Cylindrical)
            L_scaled                              # 1 feature (Normalized Layer ID)
        ], axis=1) # [N_hits, 7]

        return torch.from_numpy(F_out).float() # [N_hits, 7]

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        F_in = example_batch_x[0].shape[-1] # F_in = 7
        E_dim = 32 # Embedding Dimension
        self.E_dim = E_dim

        # <LLM: Define and initialize any stateful components here>
        self.f = nn.Sequential(
            nn.Linear(F_in, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, E_dim)
        )

    def forward(self, batch_x):
        # batch_x: list[Tensor] [N_i, F]
        current_device = batch_x[0].device

        if self.training:
            # During training, return the embeddings for metric loss calculation
            embeddings = []
            for X in batch_x:
                embeddings.append(self.f(X)) # [N_i, E_dim]
            return embeddings # list of [N_i, E_dim] tensors

        else:
            # During inference/evaluation, perform HDBSCAN clustering.
            # We assume BATCH SIZE = 1 during evaluation due to loader_cfg override.

            if len(batch_x) != 1:
                # Fallback path if B > 1 during inference, process sequentially
                all_predicted_labels = []
                for X_event in batch_x:
                     # 1. Compute embeddings
                    E = self.f(X_event).detach().cpu().numpy() # [N_i, E_dim]

                    # 2. Perform HDBSCAN clustering
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=8, 
                        min_samples=1, 
                        cluster_selection_epsilon=0.0,
                        approx_min_span_tree=True,
                        metric='euclidean'
                    )

                    pred_labels = clusterer.fit_predict(E) # [N_i], Noise is -1

                    # Relabel: HDBSCAN output [-1, 0, 1, 2, ...] -> [-1 (noise), 1, 2, 3, ...]

                    if np.min(pred_labels) == -1:
                        positive_labels = np.unique(pred_labels[pred_labels != -1])
                        label_map = {old_label: new_label + 1 for new_label, old_label in enumerate(positive_labels)}
                        label_map[-1] = -1 
                        mapped_labels = np.vectorize(label_map.get)(pred_labels)
                    else:
                        mapped_labels = pred_labels + 1

                    all_predicted_labels.append(torch.from_numpy(mapped_labels).long().to(current_device))

                return all_predicted_labels

            else:
                # Optimized path: B=1 (X_event is batch_x[0])
                X_event = batch_x[0]
                E = self.f(X_event).detach().cpu().numpy() # [N_i, E_dim]

                # Clustering
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=8, 
                    min_samples=1, 
                    cluster_selection_epsilon=0.0,
                    approx_min_span_tree=True,
                    metric='euclidean'
                )

                pred_labels = clusterer.fit_predict(E) # [N_i]

                # Relabel
                if np.min(pred_labels) == -1:
                    positive_labels = np.unique(pred_labels[pred_labels != -1])
                    label_map = {old_label: new_label + 1 for new_label, old_label in enumerate(positive_labels)}
                    label_map[-1] = -1 
                    mapped_labels = np.vectorize(label_map.get)(pred_labels)
                else:
                    mapped_labels = pred_labels + 1

                return [torch.from_numpy(mapped_labels).long().to(current_device)]

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 30   # Increased epochs for stability and convergence in metric learning 

# Define utility functions for Metric Learning Loss calculation

CONTRASTIVE_MARGIN = 1.0

def compute_contrastive_loss(embeddings_list, targets_list, margin):
    # Calculate loss event by event to respect event-local track IDs
    if not embeddings_list: 
        return torch.tensor(0.0, device=device) # device is global set in prefix code

    total_loss = torch.tensor(0.0, device=embeddings_list[0].device)
    total_pairs = 0

    for E_event, Y_event in zip(embeddings_list, targets_list):
        non_noise_mask_evt = Y_event > 0
        E_evt_clean = E_event[non_noise_mask_evt] # [N_i_clean, E_dim]
        Y_evt_clean = Y_event[non_noise_mask_evt] # [N_i_clean]
        N_i_clean = E_evt_clean.shape[0]

        if N_i_clean < 2:
            continue

        # L2 normalize embeddings
        E_evt_clean = F.normalize(E_evt_clean, p=2, dim=1)

        # Calculate squared distance matrix D_sq [N_i_clean, N_i_clean]
        dot_products_evt = E_evt_clean @ E_evt_clean.T
        D_sq_evt = torch.clamp(2.0 - 2.0 * dot_products_evt, min=0.0)

        # Calculate mask M_pos (Positive Pairs: same track ID)
        Y_evt_clean_reshaped = Y_evt_clean.unsqueeze(0)
        M_pos = (Y_evt_clean_reshaped == Y_evt_clean_reshaped.T).float()
        M_pos.fill_diagonal_(0)

        # M_neg mask (Negative Pairs: different track ID)
        M_neg = 1.0 - M_pos - torch.eye(N_i_clean, device=E_evt_clean.device)
        M_neg = torch.clamp(M_neg, min=0.0)

        N_P = M_pos.sum()
        N_N = M_neg.sum()

        if N_P == 0 or N_N == 0:
             continue

        # Positive loss term (pull together): minimize D^2
        L_pos = (M_pos * D_sq_evt).sum() / N_P

        # Negative loss term (push apart): maximize distance past margin M
        D_neg = torch.sqrt(D_sq_evt)
        L_neg = (M_neg * torch.relu(margin - D_neg)).pow(2).sum() / N_N

        L_evt = L_pos + L_neg

        total_loss += L_evt
        total_pairs += 1

    return total_loss / total_pairs if total_pairs > 0 else torch.tensor(0.0, device=embeddings_list[0].device)

def compute_accuracy_dummy(embeddings_list, targets_list, margin):
    # Proxy metric: Reports fraction of positive pairs correctly pulled closer than margin.
    if not embeddings_list: return 0.0

    total_correct_pos = 0
    total_pos_checks = 0

    with torch.no_grad():
        for E_event, Y_event in zip(embeddings_list, targets_list):
            non_noise_mask_evt = Y_event > 0
            N_i_clean = non_noise_mask_evt.sum().item()
            if N_i_clean < 2: continue

            E_evt_clean = F.normalize(E_event[non_noise_mask_evt], p=2, dim=1)
            Y_evt_clean = Y_event[non_noise_mask_evt] 

            Y_evt_clean_reshaped = Y_evt_clean.unsqueeze(0)
            M_pos = (Y_evt_clean_reshaped == Y_evt_clean_reshaped.T)
            M_pos.fill_diagonal_(0) 

            if M_pos.sum() == 0: continue

            dot_products_evt = E_evt_clean @ E_evt_clean.T
            D_sq_evt = torch.clamp(2.0 - 2.0 * dot_products_evt, min=0.0)
            D_evt = torch.sqrt(D_sq_evt)

            pos_pairs = M_pos
            pos_distances = D_evt[pos_pairs]

            correct_pos = (pos_distances < margin).sum().item()
            total_correct_pos += correct_pos
            total_pos_checks += pos_distances.numel()

    return total_correct_pos / total_pos_checks if total_pos_checks > 0 else 0.0


def train_model(model, train_loader, val_loader, epochs):
    # REQUIREMENTS: Use CUDA, return specific variables, implement early stopping.

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, threshold=1e-4) # Added threshold to prevent verbose

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_val_loss = float('inf')
    patience_counter = 0
    MAX_PATIENCE = 10

    model.to(device)

    margin = CONTRASTIVE_MARGIN

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        N_batches = 0

        for batch_idx, (Xs, Ys) in enumerate(train_loader):

            Xs_d = [X.to(device) for X in Xs]
            Ys_d = [Y.to(device) for Y in Ys]

            optimizer.zero_grad()

            embeddings_list = model(Xs_d) 

            loss = compute_contrastive_loss(embeddings_list, Ys_d, margin=margin)

            if loss.requires_grad and loss.item() != 0.0:
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_acc += compute_accuracy_dummy(embeddings_list, Ys_d, margin=margin)
                N_batches += 1

        train_loss = running_loss / N_batches if N_batches > 0 else 0.0
        train_acc = running_acc / N_batches if N_batches > 0 else 0.0
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_running_acc = 0.0
        N_val_batches = 0

        with torch.no_grad():
            for batch_idx, (Xs_val, Ys_val) in enumerate(val_loader):
                Xs_d = [X.to(device) for X in Xs_val]
                Ys_d = [Y.to(device) for Y in Ys_val]

                embeddings_list = model(Xs_d)

                val_loss = compute_contrastive_loss(embeddings_list, Ys_d, margin=margin)

                val_running_loss += val_loss.item()
                val_running_acc += compute_accuracy_dummy(embeddings_list, Ys_d, margin=margin)
                N_val_batches += 1

        val_loss = val_running_loss / N_val_batches if N_val_batches > 0 else 0.0
        val_acc = val_running_acc / N_val_batches if N_val_batches > 0 else 0.0
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        scheduler.step(val_loss)

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= MAX_PATIENCE and epoch > 15: # Wait at least 15 epochs before early stopping
                break

        # Note: Printing output is suppressed by the harness if not explicitly required, 
        # but including it here for visibility during local testing.
        # print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

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

