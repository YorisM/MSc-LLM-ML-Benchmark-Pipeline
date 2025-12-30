
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

# ---------- IMPORTS ----------
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import TransformerConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
import math

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.global_scaler = StandardScaler()
        self.obj_scaler = StandardScaler()

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},
            "loader_class": "torch_geometric.loader:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "collate": None,
            "extra_loader_kwargs": {},
            "eval_overrides": {"shuffle": False},
        }

    def fit(self, X, y=None):
        # Extract global features (first 2 columns)
        global_features = X[:, :2].numpy()
        self.global_scaler.fit(global_features)

        # Extract object features (reshape to [n_samples * n_objects, 5])
        obj_features = X[:, 2:].reshape(-1, 5).numpy()
        self.obj_scaler.fit(obj_features)
        return self

    def transform(self, X):
        # Scale global features
        global_features = X[:, :2]
        global_scaled = torch.from_numpy(self.global_scaler.transform(global_features.numpy())).float()

        # Scale object features
        obj_features = X[:, 2:].reshape(-1, 5)
        obj_scaled = torch.from_numpy(self.obj_scaler.transform(obj_features.numpy())).float()
        obj_scaled = obj_scaled.reshape(X.shape[0], -1)

        # Combine scaled features
        X_scaled = torch.cat([global_scaled, obj_scaled], dim=1)
        return X_scaled

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        # Extract dimensions from sample
        if isinstance(sample_object, torch.Tensor):
            # Flat tensor case
            self.global_dim = 2
            self.obj_dim = 5
            self.max_objects = 18
            self.hidden_dim = 128
        else:
            # PyG Data case
            self.global_dim = sample_object.x.shape[1] - 5  # Assuming first 2 are global
            self.obj_dim = 5
            self.max_objects = sample_object.num_nodes
            self.hidden_dim = 128

        # Global feature processing
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Object feature processing
        self.obj_encoder = nn.Sequential(
            nn.Linear(self.obj_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Transformer layers for object interactions
        self.transformer_conv1 = TransformerConv(self.hidden_dim, self.hidden_dim // 2, heads=4)
        self.transformer_conv2 = TransformerConv(self.hidden_dim, self.hidden_dim // 2, heads=4)

        # Attention for global-object interaction
        self.global_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, 1)
        )

    def compute_pairwise_features(self, obj_features):
        # obj_features: [batch_size * max_objects, hidden_dim]
        n_objects = obj_features.shape[0]

        # Compute pairwise features
        pairwise = []
        for i in range(n_objects):
            for j in range(i+1, n_objects):
                # Get features for object i and j
                feat_i = obj_features[i]
                feat_j = obj_features[j]

                # Compute invariant mass (simplified)
                E_i, pt_i, eta_i, phi_i = feat_i[0], feat_i[1], feat_i[2], feat_i[3]
                E_j, pt_j, eta_j, phi_j = feat_j[0], feat_j[1], feat_j[2], feat_j[3]

                # Simplified invariant mass (using E and pt)
                m_ij = torch.sqrt(2 * E_i * E_j * (1 - torch.cos(phi_i - phi_j)))

                # Angular distance
                delta_eta = eta_i - eta_j
                delta_phi = phi_i - phi_j
                delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)

                # Combine features
                pairwise.append(torch.cat([feat_i, feat_j, m_ij.unsqueeze(0), delta_R.unsqueeze(0)]))

        if len(pairwise) == 0:
            return torch.zeros(1, self.hidden_dim * 2 + 2).to(obj_features.device)
        return torch.stack(pairwise)

    def forward(self, batch_x):
        if isinstance(batch_x, torch.Tensor):
            # Flat tensor case - convert to PyG format
            global_feat = batch_x[:, :2]  # [batch_size, 2]
            obj_feat = batch_x[:, 2:].reshape(-1, self.max_objects, 5)  # [batch_size, max_objects, 5]

            # Create PyG data objects
            data_list = []
            for i in range(obj_feat.shape[0]):
                x = obj_feat[i]  # [max_objects, 5]
                edge_index = self.create_complete_graph(x.shape[0]).to(x.device)
                data = Data(x=x, edge_index=edge_index)
                data_list.append(data)

            batch = Batch.from_data_list(data_list)
            batch.global_feat = global_feat
        else:
            # Already PyG batch
            batch = batch_x
            global_feat = batch.global_feat

        # Process global features
        global_encoded = self.global_encoder(global_feat)  # [batch_size, hidden_dim]

        # Process object features
        obj_encoded = self.obj_encoder(batch.x)  # [num_objects, hidden_dim]

        # Transformer layers
        obj_encoded = F.relu(self.transformer_conv1(obj_encoded, batch.edge_index))
        obj_encoded = F.relu(self.transformer_conv2(obj_encoded, batch.edge_index))

        # Global pooling of object features
        obj_pooled = global_mean_pool(obj_encoded, batch.batch)  # [batch_size, hidden_dim]

        # Compute pairwise features
        pairwise_feats = []
        for i in range(batch.num_graphs):
            start_idx = batch.ptr[i]
            end_idx = batch.ptr[i+1]
            obj_feats = obj_encoded[start_idx:end_idx]
            pairwise = self.compute_pairwise_features(obj_feats)
            if pairwise.shape[0] > 0:
                pairwise = pairwise.mean(dim=0)  # Average pairwise features
            else:
                pairwise = torch.zeros(self.hidden_dim * 2 + 2).to(obj_encoded.device)
            pairwise_feats.append(pairwise)

        pairwise_pooled = torch.stack(pairwise_feats)  # [batch_size, hidden_dim*2+2]

        # Combine all features
        combined = torch.cat([global_encoded, obj_pooled, pairwise_pooled], dim=1)

        # Classifier
        logits = self.classifier(combined).squeeze(-1)
        return logits

    def create_complete_graph(self, n_nodes):
        # Create edge_index for complete graph
        edge_index = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_index.append([i, j])
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

def make_model(example_object):
    return BinaryClassifier(example_object)

# ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0
    best_model = None

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0
        train_samples = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb.float())

            # Compute accuracy
            preds = (logits > 0).float()
            acc = (preds == yb.float()).float().mean()

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * yb.shape[0]
            epoch_train_acc += acc.item() * yb.shape[0]
            train_samples += yb.shape[0]

        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_acc = 0
        val_samples = 0
        val_probs = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y

                logits = model(xb)
                loss = criterion(logits, yb.float())

                # Compute accuracy
                preds = (logits > 0).float()
                acc = (preds == yb.float()).float().mean()

                # Store for AUC calculation
                probs = torch.sigmoid(logits)
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(yb.cpu().numpy())

                epoch_val_loss += loss.item() * yb.shape[0]
                epoch_val_acc += acc.item() * yb.shape[0]
                val_samples += yb.shape[0]

        # Calculate metrics
        train_loss = epoch_train_loss / train_samples
        train_acc = epoch_train_acc / train_samples
        val_loss = epoch_val_loss / val_samples
        val_acc = epoch_val_acc / val_samples

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_labels, val_probs)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping and model saving
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                print(f"Early stopping at epoch {epoch}")
                break

        scheduler.step(val_auc)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

    # Load best model
    model.load_state_dict(best_model)

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

