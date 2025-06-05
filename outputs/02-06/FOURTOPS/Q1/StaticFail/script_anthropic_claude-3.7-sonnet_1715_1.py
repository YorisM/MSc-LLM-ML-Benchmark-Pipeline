
import os, sys, pickle, torch, gc, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

DATASET = {
    "X_train": "./challenges/FOURTOPS/data/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/Y_val.csv"
}
                       
def load_data():
    X_train = pd.read_csv('./challenges/FOURTOPS/data/X_train.csv',
                          dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv('./challenges/FOURTOPS/data/Y_train.csv',
                          dtype=np.int64 ).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv('./challenges/FOURTOPS/data/X_val.csv',
                          dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv('./challenges/FOURTOPS/data/Y_val.csv',
                          dtype=np.int64 ).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train),
            torch.from_numpy(Y_train),
            torch.from_numpy(X_val),
            torch.from_numpy(Y_val))

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        if isinstance(self.x, (tuple, list)):
            return (tuple(t[idx] for t in self.x), self.y[idx])
        else:
            return (self.x[idx], self.y[idx])      

def make_loaders(X_train, Y_train, X_val, Y_val, batch=512):
    train_ds = PairDataset(X_train, Y_train)
    val_ds   = PairDataset(X_val , Y_val)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0))
                        
# ----------------  START OF LLM BLOCK  ----------------

# 0. ---------- IMPORTS ----------
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # Statistics for feature normalization
        self.et_miss_mean = None
        self.et_miss_std = None
        self.e_mean = None
        self.e_std = None
        self.pt_mean = None
        self.pt_std = None
        self.eta_mean = None
        self.eta_std = None
        self.max_objects = 18

    def fit(self, X, y=None):
        # Calculate statistics for missing energy
        self.et_miss_mean = torch.mean(X[:, 0])
        self.et_miss_std = torch.std(X[:, 0])

        # Collect valid object features for normalization
        energies = []
        pts = []
        etas = []

        # Extract features from valid objects (non-zero object IDs)
        for i in range(self.max_objects):
            base_idx = 2 + i * 5
            obj_id_idx = base_idx
            e_idx = base_idx + 1
            pt_idx = base_idx + 2
            eta_idx = base_idx + 3

            valid_mask = X[:, obj_id_idx] != 0
            if torch.any(valid_mask):
                energies.append(X[valid_mask, e_idx])
                pts.append(X[valid_mask, pt_idx])
                etas.append(X[valid_mask, eta_idx])

        # Compute statistics for normalization
        if energies:
            all_energies = torch.cat(energies)
            self.e_mean = torch.mean(all_energies)
            self.e_std = torch.std(all_energies)
        else:
            self.e_mean = torch.tensor(0.0)
            self.e_std = torch.tensor(1.0)

        if pts:
            all_pts = torch.cat(pts)
            self.pt_mean = torch.mean(all_pts)
            self.pt_std = torch.std(all_pts)
        else:
            self.pt_mean = torch.tensor(0.0)
            self.pt_std = torch.tensor(1.0)

        if etas:
            all_etas = torch.cat(etas)
            self.eta_mean = torch.mean(all_etas)
            self.eta_std = torch.std(all_etas)
        else:
            self.eta_mean = torch.tensor(0.0)
            self.eta_std = torch.tensor(1.0)

        return self

    def transform(self, X):
        batch_size = X.shape[0]

        # Create tensors for processed data
        # 7 features per object: [obj_id, E_norm, pT_norm, eta_norm, phi, 
        #                        delta_phi_to_miss, log_mass]
        object_features = torch.zeros((batch_size, self.max_objects, 7), dtype=torch.float32)
        object_mask = torch.zeros((batch_size, self.max_objects), dtype=torch.bool)

        # Process missing energy
        et_miss_norm = (X[:, 0] - self.et_miss_mean) / self.et_miss_std
        phi_miss = X[:, 1]

        # Global event features
        total_energy = torch.zeros(batch_size)
        total_pt = torch.zeros(batch_size)
        ht = torch.zeros(batch_size)  # Scalar sum of pT
        object_counts = torch.zeros((batch_size, 10), dtype=torch.float32)  # Count of each object type

        # Extract and process object features
        for i in range(self.max_objects):
            base_idx = 2 + i * 5
            obj_id_idx = base_idx
            e_idx = base_idx + 1
            pt_idx = base_idx + 2
            eta_idx = base_idx + 3
            phi_idx = base_idx + 4

            # Get object ID and create mask for valid objects
            obj_id = X[:, obj_id_idx]
            valid_mask = obj_id != 0
            object_mask[:, i] = valid_mask

            if torch.any(valid_mask):
                # Store object ID
                object_features[:, i, 0] = obj_id

                # Extract raw features for valid objects
                e = X[valid_mask, e_idx]
                pt = X[valid_mask, pt_idx]
                eta = X[valid_mask, eta_idx]
                phi = X[valid_mask, phi_idx]

                # Normalize features
                object_features[valid_mask, i, 1] = (e - self.e_mean) / self.e_std
                object_features[valid_mask, i, 2] = (pt - self.pt_mean) / self.pt_std
                object_features[valid_mask, i, 3] = (eta - self.eta_mean) / self.eta_std
                object_features[valid_mask, i, 4] = phi

                # Calculate delta phi to missing energy (physics-inspired feature)
                delta_phi = torch.abs(phi - phi_miss[valid_mask])
                # Handle circular nature of phi
                delta_phi = torch.min(delta_phi, 2*np.pi - delta_phi)
                object_features[valid_mask, i, 5] = delta_phi

                # Calculate approximate mass (E^2 - p^2 = m^2)
                p = pt * torch.cosh(eta)  # Approximate momentum
                mass_squared = e**2 - p**2
                # Use log(1 + mass) to handle negative values and emphasize small masses
                mass_feature = torch.log1p(torch.abs(mass_squared) + 1e-6)
                object_features[valid_mask, i, 6] = mass_feature

                # Update global features
                total_energy[valid_mask] += e
                total_pt[valid_mask] += pt
                ht[valid_mask] += pt

                # Count object types (assuming obj_id is 1-indexed and < 10)
                for j in range(1, 10):  # Object types 1-9
                    obj_count = (obj_id[valid_mask] == j).float()
                    if obj_count.numel() > 0:
                        for idx, is_obj in enumerate(obj_count):
                            if is_obj:
                                valid_indices = torch.where(valid_mask)[0]
                                object_counts[valid_indices[idx], j-1] += 1

        # Create global features tensor
        # [ET_miss_norm, phi_miss, total_E_norm, total_pT_norm, HT_norm, num_objects, obj_type_counts]
        global_features = torch.zeros((batch_size, 15), dtype=torch.float32)
        global_features[:, 0] = et_miss_norm
        global_features[:, 1] = phi_miss
        global_features[:, 2] = (total_energy - self.e_mean * 5) / (self.e_std * 5)
        global_features[:, 3] = (total_pt - self.pt_mean * 5) / (self.pt_std * 5)
        global_features[:, 4] = (ht - self.pt_mean * 5) / (self.pt_std * 5)
        global_features[:, 5] = object_mask.sum(dim=1).float() / self.max_objects
        global_features[:, 6:15] = object_counts

        return (object_features, object_mask, global_features)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
def make_model(input_shape, *, use_mask=True):
    class PhysicsGNN(nn.Module):
        def __init__(self, d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.2):
            super().__init__()

            # Process object features (7 features per object)
            self.obj_embedding = nn.Sequential(
                nn.Linear(7, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # Process global features (15 features)
            self.global_embedding = nn.Sequential(
                nn.Linear(15, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # Transformer encoder for processing object sequences
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Self-attention for combining object features with global features
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

            # Final classification layers
            self.classifier = nn.Sequential(
                nn.Linear(d_model * 2, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )

            # Object type embeddings for improved representation
            self.obj_type_embedding = nn.Embedding(10, 16)  # 10 possible object types, 16-dim embedding

        def forward(self, x):
            if use_mask:
                obj_features, obj_mask, global_features = x

                # Embed object types
                obj_types = obj_features[:, :, 0].long()
                obj_type_emb = self.obj_type_embedding(obj_types)

                # Combine object type embeddings with other object features
                obj_features_combined = torch.cat([
                    obj_type_emb,
                    obj_features[:, :, 1:],  # All features except object ID
                ], dim=2)

                # Process object features
                obj_embedded = self.obj_embedding(obj_features_combined)  # [batch_size, max_objects, d_model]

                # Create attention mask for transformer
                attn_mask = ~obj_mask

                # Apply transformer with attention mask
                obj_encoded = self.transformer_encoder(obj_embedded, src_key_padding_mask=attn_mask)

                # Process global features
                global_embedded = self.global_embedding(global_features)  # [batch_size, d_model]

                # Global pooling over objects (mean of valid objects)
                expanded_mask = obj_mask.unsqueeze(-1).float()  # [batch_size, max_objects, 1]
                masked_encoding = obj_encoded * expanded_mask
                obj_pooled = masked_encoding.sum(dim=1) / (expanded_mask.sum(dim=1) + 1e-8)  # [batch_size, d_model]

                # Self-attention between object representation and global features
                global_expanded = global_embedded.unsqueeze(1)  # [batch_size, 1, d_model]
                combined = torch.cat([global_expanded, obj_pooled.unsqueeze(1)], dim=1)  # [batch_size, 2, d_model]
                attn_output, _ = self.self_attn(combined, combined, combined)

                # Flatten and pass through classifier
                attn_output_flat = attn_output.reshape(attn_output.size(0), -1)  # [batch_size, 2*d_model]
                logits = self.classifier(attn_output_flat)

                return logits.squeeze(-1)
            else:
                # For compatibility with non-mask models
                # Extract and restructure data
                batch_size = x.shape[0]

                # Extract missing energy and global features
                et_miss = x[:, 0]
                phi_miss = x[:, 1]

                # Create tensors for objects
                obj_features = torch.zeros((batch_size, 18, 7), dtype=torch.float32)
                obj_mask = torch.zeros((batch_size, 18), dtype=torch.bool)

                # Extract object features and create mask
                for i in range(18):
                    start_idx = 2 + i * 5
                    obj_id = x[:, start_idx]
                    valid = obj_id != 0
                    obj_mask[:, i] = valid

                    # Basic object features
                    obj_features[:, i, 0] = obj_id
                    obj_features[:, i, 1:5] = x[:, start_idx+1:start_idx+5]

                    # Calculate delta phi to missing energy
                    phi = x[:, start_idx+4]
                    delta_phi = torch.abs(phi - phi_miss)
                    delta_phi = torch.min(delta_phi, 2*np.pi - delta_phi)
                    obj_features[:, i, 5] = delta_phi

                    # Simple placeholder for mass
                    obj_features[:, i, 6] = 0.0

                # Create global features placeholder
                global_feat = torch.zeros((batch_size, 15), dtype=torch.float32)
                global_feat[:, 0] = et_miss
                global_feat[:, 1] = phi_miss
                global_feat[:, 5] = obj_mask.sum(dim=1).float() / 18

                # Process with masked input
                return self.forward((obj_features, obj_mask, global_feat))

    return PhysicsGNN()

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50    
def train_model(model, train_loader, val_loader, epochs):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, threshold=0.001, min_lr=1e-6
    )

    # Track metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Early stopping
    best_val_auc = 0
    patience = 8
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for data, labels in train_loader:
            # Move data to device
            if isinstance(data, tuple):
                data = tuple(d.to(device) for d in data)
            else:
                data = data.to(device)
            labels = labels.to(device).float()

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track metrics
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for data, labels in val_loader:
                # Move data to device
                if isinstance(data, tuple):
                    data = tuple(d.to(device) for d in data)
                else:
                    data = data.to(device)
                labels = labels.to(device).float()

                # Forward pass
                outputs = model(data)
                loss = criterion(outputs, labels)

                # Track metrics
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_outputs.append(outputs)
                all_labels.append(labels)

        # Calculate AUC
        all_outputs = torch.cat(all_outputs).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        val_auc = roc_auc_score(all_labels, torch.sigmoid(torch.tensor(all_outputs)).numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        # Update learning rate
        scheduler.step(val_auc)

        # Early stopping check
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs

# ----------------  END OF LLM BLOCK ----------------
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    pre = make_preprocessor()
    pre.fit(X_train, Y_train)
    X_train = pre.transform(X_train)
    X_val = pre.transform(X_val)
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val)

    # 2. Build model
    if torch.is_tensor(X_train):                 # single-tensor
        tensor_ref  = X_train
        use_mask    = False
    else:                                        # tuple => (data, mask)
        tensor_ref  = X_train[0]
        use_mask    = True
    input_shape = tensor_ref.shape[1:]
    model = make_model(input_shape, use_mask=use_mask)

    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)

    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 3. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
              # 8 fake events
        if isinstance(X_train, torch.Tensor):           # single-tensor case
            toy = torch.zeros(8, *input_shape, dtype=torch.float32)
            toy_transformed = pre.transform(toy)
        else:                                           # tuple case
            toy = torch.zeros(8, *input_shape, dtype=torch.float32)
            mask = torch.zeros(8, input_shape[0], dtype=torch.bool)
            toy_transformed = (toy, mask)
        try: 
            _ = trained_model(*toy_transformed) if use_mask else trained_model(toy_transformed)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 4. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

    pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
    pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
    pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

    torch.save(trained_model.state_dict(), pth_state)
    with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
    with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
    _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

    # 6. Write JSON Summary
    if not dryrun: 
        summary = {
            "epochs": n_epochs,
            "train_loss": tr_loss,
            "val_loss":   va_loss,
            "train_acc":  tr_acc,
            "val_acc":    va_acc,
            "best_train_loss": min(tr_loss),
            "best_train_loss_epoch": tr_loss.index(min(tr_loss))+1,
            "best_train_acc":  max(tr_acc),
            "best_train_acc_epoch": tr_acc.index(max(tr_acc))+1,
            "best_val_loss": min(va_loss),
            "best_val_loss_epoch": va_loss.index(min(va_loss))+1,
            "best_val_acc":  max(va_acc),
            "best_val_acc_epoch": va_acc.index(max(va_acc))+1,
        }
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

