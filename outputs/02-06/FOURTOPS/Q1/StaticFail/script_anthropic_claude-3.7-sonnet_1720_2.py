
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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # Define constants
        self.max_objects = 18  # Max number of objects per event
        self.features_per_object = 5  # Each object has 5 features

        # Statistics for normalization
        self.et_miss_mean = None
        self.et_miss_std = None
        self.phi_et_miss_mean = None
        self.phi_et_miss_std = None
        self.obj_mean = None
        self.obj_std = None

    def fit(self, X, y=None):
        """Extract statistics for normalization from training data"""
        # Extract missing ET and phi
        et_miss = X[:, 0]
        phi_et_miss = X[:, 1]

        # Store statistics for ET_miss and phi_ET_miss
        self.et_miss_mean = et_miss.mean()
        self.et_miss_std = et_miss.std() + 1e-8  # Add epsilon to avoid division by zero
        self.phi_et_miss_mean = phi_et_miss.mean()
        self.phi_et_miss_std = phi_et_miss.std() + 1e-8

        # Reshape object data for easier processing
        objects_flat = X[:, 2:]  # All object data
        objects = objects_flat.view(X.shape[0], self.max_objects, self.features_per_object)

        # Create mask for valid objects (non-zero object ID)
        obj_ids = objects[:, :, 0]
        valid_mask = obj_ids != 0

        # Get only valid objects for computing statistics
        valid_objects = objects[valid_mask]

        # Compute mean and std for object features (excluding object ID)
        # We'll normalize features 1-4 (E, pT, eta, phi)
        self.obj_mean = torch.mean(valid_objects[:, 1:], dim=0)
        self.obj_std = torch.std(valid_objects[:, 1:], dim=0) + 1e-8

        return self

    def transform(self, X):
        """Transform the data with normalization and feature engineering"""
        batch_size = X.shape[0]

        # Extract and normalize ET_miss and phi_ET_miss
        et_miss = (X[:, 0:1] - self.et_miss_mean) / self.et_miss_std
        phi_et_miss = (X[:, 1:2] - self.phi_et_miss_mean) / self.phi_et_miss_std

        # Reshape object data
        objects_flat = X[:, 2:]
        objects = objects_flat.view(batch_size, self.max_objects, self.features_per_object)

        # Create mask for valid objects
        obj_ids = objects[:, :, 0]
        valid_mask = (obj_ids != 0)

        # Normalize object features (keeping object ID as is)
        norm_objects = torch.zeros_like(objects)
        norm_objects[:, :, 0] = objects[:, :, 0]  # Keep object ID unchanged
        norm_objects[:, :, 1:] = (objects[:, :, 1:] - self.obj_mean) / self.obj_std

        # Feature engineering - Physics motivated features

        # 1. Scalar sum of pT (HT)
        HT = torch.sum(objects[:, :, 2] * valid_mask, dim=1, keepdim=True)
        norm_HT = (HT - HT.mean()) / (HT.std() + 1e-8)

        # 2. Number of objects by type
        unique_obj_ids = torch.unique(obj_ids[valid_mask])
        unique_obj_ids = unique_obj_ids[unique_obj_ids != 0]  # Remove padding

        obj_type_counts = []
        for obj_type in unique_obj_ids:
            count = torch.sum((obj_ids == obj_type).float(), dim=1, keepdim=True)
            obj_type_counts.append(count)

        if obj_type_counts:
            obj_counts = torch.cat(obj_type_counts, dim=1)
        else:
            obj_counts = torch.zeros((batch_size, 0), device=X.device)

        # 3. Calculate deltaR between high-pT objects
        # Sort by pT
        pt_values = objects[:, :, 2]
        _, sorted_indices = torch.sort(pt_values, dim=1, descending=True)

        # Get top 4 objects or fewer if not enough valid objects
        top_k = min(4, torch.sum(valid_mask[0]).item())

        if top_k > 1:  # Need at least 2 objects to calculate deltaR
            deltaR_features = []

            for i in range(top_k):
                for j in range(i+1, top_k):
                    # Get indices of objects to compare
                    idx_i = sorted_indices[:, i]
                    idx_j = sorted_indices[:, j]

                    # Extract eta and phi for these objects
                    batch_indices = torch.arange(batch_size, device=X.device)

                    eta_i = objects[batch_indices, idx_i, 3]
                    phi_i = objects[batch_indices, idx_i, 4]
                    eta_j = objects[batch_indices, idx_j, 3]
                    phi_j = objects[batch_indices, idx_j, 4]

                    # Calculate deltaR
                    deta = eta_i - eta_j
                    dphi = torch.abs(phi_i - phi_j)
                    # Handle phi wrapping
                    dphi = torch.min(dphi, 2*np.pi - dphi)
                    dr = torch.sqrt(deta**2 + dphi**2).unsqueeze(1)

                    deltaR_features.append(dr)

            if deltaR_features:
                deltaR = torch.cat(deltaR_features, dim=1)
            else:
                deltaR = torch.zeros((batch_size, 0), device=X.device)
        else:
            deltaR = torch.zeros((batch_size, 0), device=X.device)

        # 4. Missing ET to HT ratio
        et_miss_to_HT = X[:, 0:1] / (HT + 1e-8)
        norm_et_miss_to_HT = (et_miss_to_HT - et_miss_to_HT.mean()) / (et_miss_to_HT.std() + 1e-8)

        # Combine all features
        global_features = torch.cat([
            et_miss, 
            phi_et_miss,
            norm_HT,
            norm_et_miss_to_HT,
            obj_counts,
            deltaR
        ], dim=1)

        # Return both normalized objects and mask
        return norm_objects, valid_mask, global_features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class HEPModel(nn.Module):
    def __init__(self, input_shape, *, use_mask=True):
        super().__init__()
        self.use_mask = use_mask
        self.max_objects = input_shape[0]
        self.obj_features = input_shape[1]
        self.global_features = input_shape[2]

        # Embedding for object types - estimate max types
        self.max_types = 100  # Generous estimate
        self.type_embedding = nn.Embedding(self.max_types, 16)

        # Object encoder
        self.object_encoder = nn.Sequential(
            nn.Linear(self.obj_features - 1 + 16, 128),  # -1 for type, +16 for embedding
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Self-attention for objects
        self.attention_query = nn.Linear(128, 64)
        self.attention_key = nn.Linear(128, 64)
        self.attention_value = nn.Linear(128, 128)

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if self.use_mask:
            objects, mask, global_features = x
        else:
            objects, _, global_features = x
            # Create mask based on object ID
            mask = objects[:, :, 0] != 0

        # Process object features
        obj_types = objects[:, :, 0].long()
        obj_features = objects[:, :, 1:]

        # Embed object types
        type_embeddings = self.type_embedding(obj_types)

        # Combine type embedding with object features
        combined_features = torch.cat([type_embeddings, obj_features], dim=2)

        # Encode objects
        encoded_objects = self.object_encoder(combined_features)

        # Apply mask
        encoded_objects = encoded_objects * mask.unsqueeze(-1)

        # Self-attention mechanism
        query = self.attention_query(encoded_objects)
        key = self.attention_key(encoded_objects)
        value = self.attention_value(encoded_objects)

        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(64)

        # Apply mask to attention scores
        attention_scores = attention_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        # Softmax attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention to values
        context = torch.matmul(attention_weights, value)

        # Pool attended features (average over sequence length)
        object_features = torch.sum(context * mask.unsqueeze(-1).unsqueeze(1), dim=2) / \
                         (torch.sum(mask, dim=1, keepdim=True).unsqueeze(1) + 1e-8)

        # Reshape to remove the singleton dimension
        object_features = object_features.squeeze(1)

        # Process global features
        global_encoding = self.global_encoder(global_features)

        # Combine object and global features
        combined = torch.cat([object_features, global_encoding], dim=1)

        # Classify
        output = self.classifier(combined)

        return output

def make_model(input_shape, *, use_mask=True):
    model = HEPModel(input_shape, use_mask=use_mask)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    # Metrics tracking
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    # Early stopping
    best_val_auc = 0.0
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            # Move data to device
            if isinstance(inputs, tuple):
                inputs = tuple(x.to(device) for x in inputs)
            else:
                inputs = inputs.to(device)
            targets = targets.to(device).float()

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track metrics
            running_loss += loss.item() * targets.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        # Calculate epoch metrics
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move data to device
                if isinstance(inputs, tuple):
                    inputs = tuple(x.to(device) for x in inputs)
                else:
                    inputs = inputs.to(device)
                targets = targets.to(device).float()

                # Forward pass
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)

                # Track metrics
                running_loss += loss.item() * targets.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

                # Store for AUC calculation
                all_targets.append(targets.cpu().numpy())
                all_predictions.append(torch.sigmoid(outputs).cpu().numpy())

        # Calculate epoch metrics
        val_loss = running_loss / total
        val_acc = correct / total

        # Calculate AUC
        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)
        val_auc = roc_auc_score(all_targets, all_predictions)

        # Update learning rate
        scheduler.step(val_auc)

        # Update history
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Early stopping check
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # Restore best model
                model.load_state_dict(best_model_state)
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

    # Load best model if we completed all epochs
    if epoch == epochs - 1 and best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

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

