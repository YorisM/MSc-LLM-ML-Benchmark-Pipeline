
import os, sys, pickle, torch, gc, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import cast

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
import math
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        # Store normalization statistics
        self.feature_means = {}
        self.feature_stds = {}
        # For identifying and counting particle types
        self.object_types = {}
        # Constants for data structure
        self.max_objects = 18
        self.obj_feature_size = 5

    def fit(self, X, y=None):
        # Normalize missing energy and phi
        self.feature_means['et_miss'] = X[:, 0].mean().item()
        self.feature_stds['et_miss'] = X[:, 0].std().item() or 1.0

        self.feature_means['phi_et_miss'] = X[:, 1].mean().item()
        self.feature_stds['phi_et_miss'] = X[:, 1].std().item() or 1.0

        # Collect statistics for object features
        energy_vals = []
        pt_vals = []
        eta_vals = []
        phi_vals = []

        # Process all events and objects
        for i in range(X.shape[0]):
            for j in range(2, X.shape[1], self.obj_feature_size):
                if j + 4 < X.shape[1] and X[i, j] != 0:  # Valid object
                    obj_id = int(X[i, j].item())

                    # Count object types
                    if obj_id not in self.object_types:
                        self.object_types[obj_id] = 0
                    self.object_types[obj_id] += 1

                    # Collect feature values for normalization
                    energy_vals.append(X[i, j+1].item())
                    pt_vals.append(X[i, j+2].item())
                    eta_vals.append(X[i, j+3].item())
                    phi_vals.append(X[i, j+4].item())

        # Calculate normalization parameters
        self.feature_means['energy'] = np.mean(energy_vals) if energy_vals else 0
        self.feature_stds['energy'] = np.std(energy_vals) if energy_vals else 1

        self.feature_means['pt'] = np.mean(pt_vals) if pt_vals else 0
        self.feature_stds['pt'] = np.std(pt_vals) if pt_vals else 1

        self.feature_means['eta'] = np.mean(eta_vals) if eta_vals else 0
        self.feature_stds['eta'] = np.std(eta_vals) if eta_vals else 1

        self.feature_means['phi'] = np.mean(phi_vals) if phi_vals else 0
        self.feature_stds['phi'] = np.std(phi_vals) if phi_vals else 1

        return self

    def transform(self, X):
        batch_size = X.shape[0]

        # Create normalized tensor
        X_processed = torch.zeros_like(X)

        # Normalize ET_miss and phi_ET_miss
        X_processed[:, 0] = (X[:, 0] - self.feature_means['et_miss']) / self.feature_stds['et_miss']
        X_processed[:, 1] = (X[:, 1] - self.feature_means['phi_et_miss']) / self.feature_stds['phi_et_miss']

        # Process object features (normalize and preserve object IDs)
        for i in range(batch_size):
            for j in range(2, X.shape[1], self.obj_feature_size):
                if j + 4 < X.shape[1] and X[i, j] != 0:  # Valid object
                    # Keep object ID as is
                    X_processed[i, j] = X[i, j]

                    # Normalize energy, pt, eta, phi
                    X_processed[i, j+1] = (X[i, j+1] - self.feature_means['energy']) / self.feature_stds['energy']
                    X_processed[i, j+2] = (X[i, j+2] - self.feature_means['pt']) / self.feature_stds['pt']
                    X_processed[i, j+3] = (X[i, j+3] - self.feature_means['eta']) / self.feature_stds['eta']
                    X_processed[i, j+4] = (X[i, j+4] - self.feature_means['phi']) / self.feature_stds['phi']

        # Create a mask for valid objects (to handle variable-length sequences)
        mask = torch.zeros((batch_size, (X.shape[1] - 2) // self.obj_feature_size), dtype=torch.bool)

        for i in range(batch_size):
            obj_idx = 0
            for j in range(2, X.shape[1], self.obj_feature_size):
                if j + 4 < X.shape[1] and X[i, j] != 0:  # Valid object
                    mask[i, obj_idx] = True
                    obj_idx += 1

        # Extract additional physics-inspired features
        physics_features = self._calculate_physics_features(X)

        return (X_processed, mask, physics_features)

    def _calculate_physics_features(self, X):
        batch_size = X.shape[0]
        n_physics_features = 30  # Adjust based on features created

        physics_features = torch.zeros((batch_size, n_physics_features), dtype=torch.float32)

        for i in range(batch_size):
            # Extract objects in this event
            objects = []
            for j in range(2, X.shape[1], self.obj_feature_size):
                if j + 4 < X.shape[1] and X[i, j] != 0:  # Valid object
                    obj_id = int(X[i, j].item())
                    energy = X[i, j+1].item()
                    pt = X[i, j+2].item()
                    eta = X[i, j+3].item()
                    phi = X[i, j+4].item()
                    objects.append((obj_id, energy, pt, eta, phi))

            # 1. Total object count
            physics_features[i, 0] = len(objects)

            # 2. Count objects by type
            obj_counts = {}
            for obj in objects:
                obj_id = int(obj[0])
                if obj_id not in obj_counts:
                    obj_counts[obj_id] = 0
                obj_counts[obj_id] += 1

            # Store counts for most frequent object types
            feature_idx = 1
            for obj_type in sorted(self.object_types.keys())[:8]:
                if feature_idx < n_physics_features:
                    physics_features[i, feature_idx] = obj_counts.get(obj_type, 0)
                    feature_idx += 1

            # 3. Energy and momentum sums
            total_energy = sum(obj[1] for obj in objects)
            total_pt = sum(obj[2] for obj in objects)

            physics_features[i, 9] = total_energy
            physics_features[i, 10] = total_pt

            # 4. Missing ET / sqrt(HT) ratio (useful for distinguishing signal)
            if total_pt > 0:
                physics_features[i, 11] = X[i, 0].item() / math.sqrt(total_pt)

            # 5. Sort objects by pT (descending) for further calculations
            if objects:
                objects.sort(key=lambda x: x[2], reverse=True)

                # 6. Leading object properties
                if len(objects) >= 1:
                    physics_features[i, 12] = objects[0][1]  # Energy
                    physics_features[i, 13] = objects[0][2]  # pT

                # 7. Invariant mass combinations
                if len(objects) >= 2:
                    for j in range(min(4, len(objects))):
                        for k in range(j+1, min(4, len(objects))):
                            pair_idx = (j * 3 + k - j - 1)
                            if 14 + pair_idx < n_physics_features:
                                m_inv = self._invariant_mass(
                                    objects[j][1], objects[j][2], objects[j][3], objects[j][4],
                                    objects[k][1], objects[k][2], objects[k][3], objects[k][4]
                                )
                                physics_features[i, 14 + pair_idx] = m_inv

                # 8. Angular separations (important for jet identification)
                if len(objects) >= 2:
                    for j in range(min(3, len(objects))):
                        for k in range(j+1, min(3, len(objects))):
                            pair_idx = (j * 2 + k - j - 1)
                            if 20 + pair_idx < n_physics_features:
                                dR = self._delta_R(
                                    objects[j][3], objects[j][4],
                                    objects[k][3], objects[k][4]
                                )
                                physics_features[i, 20 + pair_idx] = dR

                # 9. Sphericity or aplanarity (simplified)
                if len(objects) >= 3:
                    physics_features[i, 23] = self._calculate_sphericity(objects)

                # 10. Sum of invariant masses of all pairs
                if len(objects) >= 2:
                    sum_m_inv = 0
                    for j in range(len(objects)):
                        for k in range(j+1, len(objects)):
                            sum_m_inv += self._invariant_mass(
                                objects[j][1], objects[j][2], objects[j][3], objects[j][4],
                                objects[k][1], objects[k][2], objects[k][3], objects[k][4]
                            )
                    physics_features[i, 24] = sum_m_inv

                # 11. Average angular separation
                if len(objects) >= 2:
                    sum_dR = 0
                    count = 0
                    for j in range(len(objects)):
                        for k in range(j+1, len(objects)):
                            sum_dR += self._delta_R(
                                objects[j][3], objects[j][4],
                                objects[k][3], objects[k][4]
                            )
                            count += 1
                    if count > 0:
                        physics_features[i, 25] = sum_dR / count

                # 12. Missing ET significance (MET / sqrt(sum pT))
                if total_pt > 0:
                    physics_features[i, 26] = X[i, 0].item() / math.sqrt(total_pt)

                # 13. Centrality
                if total_energy > 0:
                    physics_features[i, 27] = total_pt / total_energy

                # 14. HT (scalar sum of pT)
                physics_features[i, 28] = total_pt

                # 15. Ratio of leading pT to total pT
                if total_pt > 0 and len(objects) > 0:
                    physics_features[i, 29] = objects[0][2] / total_pt

        return physics_features

    def _invariant_mass(self, E1, pt1, eta1, phi1, E2, pt2, eta2, phi2):
        # Calculate 4-vectors
        px1 = pt1 * math.cos(phi1)
        py1 = pt1 * math.sin(phi1)
        pz1 = pt1 * math.sinh(eta1)

        px2 = pt2 * math.cos(phi2)
        py2 = pt2 * math.sin(phi2)
        pz2 = pt2 * math.sinh(eta2)

        # Calculate invariant mass
        E_sum = E1 + E2
        px_sum = px1 + px2
        py_sum = py1 + py2
        pz_sum = pz1 + pz2

        m_squared = E_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2
        return math.sqrt(max(0, m_squared))

    def _delta_R(self, eta1, phi1, eta2, phi2):
        # Calculate angular separation in eta-phi space
        deta = eta1 - eta2
        dphi = abs(phi1 - phi2)
        # Adjust dphi to be in [0, pi]
        dphi = min(dphi, 2 * math.pi - dphi) if dphi > math.pi else dphi
        return math.sqrt(deta**2 + dphi**2)

    def _calculate_sphericity(self, objects):
        # Simplified sphericity calculation
        momentum_tensor = np.zeros((3, 3))
        p_squared_sum = 0

        for obj in objects:
            pt = obj[2]
            eta = obj[3]
            phi = obj[4]

            px = pt * math.cos(phi)
            py = pt * math.sin(phi)
            pz = pt * math.sinh(eta)

            momentum_tensor[0, 0] += px * px
            momentum_tensor[0, 1] += px * py
            momentum_tensor[0, 2] += px * pz
            momentum_tensor[1, 0] += py * px
            momentum_tensor[1, 1] += py * py
            momentum_tensor[1, 2] += py * pz
            momentum_tensor[2, 0] += pz * px
            momentum_tensor[2, 1] += pz * py
            momentum_tensor[2, 2] += pz * pz

            p_squared_sum += px*px + py*py + pz*pz

        if p_squared_sum > 0:
            momentum_tensor /= p_squared_sum

            # Calculate eigenvalues
            try:
                eigenvalues = np.linalg.eigvalsh(momentum_tensor)
                eigenvalues.sort()  # Sort in ascending order

                # Sphericity = 3/2 * (λ2 + λ1)
                return 1.5 * (eigenvalues[0] + eigenvalues[1])
            except:
                return 0
        return 0

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
def make_model(input_shape, *, use_mask=False):
    class ParticleClassifier(nn.Module):
        def __init__(self, input_shape, use_mask):
            super().__init__()
            self.use_mask = use_mask

            if use_mask:
                # We expect three inputs: data, mask, physics_features
                self.object_dim = 5  # obj_id, E, pT, eta, phi
                max_objects = (input_shape[0] - 2) // 5
                n_physics_features = 30  # Match preprocessor

                # Event features (ET_miss, phi_ET_miss)
                self.event_projection = nn.Sequential(
                    nn.Linear(2, 32),
                    nn.ReLU(),
                    nn.Linear(32, 64),
                    nn.ReLU()
                )

                # Object embedding
                self.object_embedding = nn.Sequential(
                    nn.Linear(self.object_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                )

                # Transformer for object sequence processing
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=128,
                    nhead=8,
                    dim_feedforward=512,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

                # Physics features processing
                self.physics_projection = nn.Sequential(
                    nn.Linear(n_physics_features, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU()
                )

                # Combined features size
                combined_size = 64 + 128 + 128  # event + object + physics
            else:
                # Simple model for flattened input
                combined_size = input_shape[0]

            # Classifier head
            self.classifier = nn.Sequential(
                nn.Linear(combined_size, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            if self.use_mask:
                # Unpack the input: (data, mask, physics_features)
                data, mask, physics_features = x

                # Process event features (ET_miss, phi_ET_miss)
                event_features = data[:, :2]
                event_output = self.event_projection(event_features)  # [batch, 64]

                # Extract and process object features
                batch_size = data.shape[0]
                max_objects = (data.shape[1] - 2) // 5

                # Reshape object features
                object_features = torch.zeros((batch_size, max_objects, self.object_dim), device=data.device)

                for i in range(batch_size):
                    obj_idx = 0
                    for j in range(2, data.shape[1], 5):
                        if j + 4 < data.shape[1] and data[i, j] != 0:  # Valid object
                            object_features[i, obj_idx] = data[i, j:j+5]
                            obj_idx += 1

                # Embed object features
                embedded_objects = self.object_embedding(object_features)  # [batch, max_objects, 128]

                # Apply transformer (with masking)
                # TransformerEncoder expects mask to be False for valid positions, True for padding
                transformer_mask = ~mask

                transformer_output = self.transformer(
                    embedded_objects,
                    src_key_padding_mask=transformer_mask
                )  # [batch, max_objects, 128]

                # Global pooling over valid objects
                valid_counts = mask.sum(dim=1, keepdim=True)
                valid_counts = torch.clamp(valid_counts, min=1)  # Avoid division by zero

                # Apply mask and pool
                masked_output = transformer_output * mask.unsqueeze(-1)
                object_output = masked_output.sum(dim=1) / valid_counts  # [batch, 128]

                # Process physics features
                physics_output = self.physics_projection(physics_features)  # [batch, 128]

                # Combine all features
                combined = torch.cat([event_output, object_output, physics_output], dim=1)
            else:
                # Simple model for flattened input
                combined = x

            # Final classification
            output = self.classifier(combined)
            return output.squeeze(-1)  # [batch]

    return ParticleClassifier(input_shape, use_mask)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 100

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        # Linear warmup for 5 epochs, then cosine decay
        if epoch < 5:
            return epoch / 5
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - 5) / (epochs - 5)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initialize tracking variables
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_auc = 0
    best_model_state = None
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        epoch_train_correct = 0
        epoch_train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            if isinstance(data, tuple):
                data = tuple(d.to(device) for d in data)
            else:
                data = data.to(device)
            target = target.to(device).float()

            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backward pass and optimization
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Calculate accuracy
            predicted = (output > 0.5).float()
            epoch_train_correct += (predicted == target).sum().item()
            epoch_train_total += target.size(0)
            epoch_train_loss += loss.item()

        # Calculate epoch metrics
        train_loss = epoch_train_loss / len(train_loader)
        train_acc = epoch_train_correct / epoch_train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        epoch_val_correct = 0
        epoch_val_total = 0
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                # Move data to device
                if isinstance(data, tuple):
                    data = tuple(d.to(device) for d in data)
                else:
                    data = data.to(device)
                target = target.to(device).float()

                # Forward pass
                output = model(data)
                loss = criterion(output, target)

                # Calculate accuracy
                predicted = (output > 0.5).float()
                epoch_val_correct += (predicted == target).sum().item()
                epoch_val_total += target.size(0)
                epoch_val_loss += loss.item()

                # Store outputs and targets for AUC calculation
                all_targets.append(target.cpu().numpy())
                all_outputs.append(output.cpu().numpy())

        # Calculate epoch metrics
        val_loss = epoch_val_loss / len(val_loader)
        val_acc = epoch_val_correct / epoch_val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Calculate AUC
        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        val_auc = roc_auc_score(all_targets, all_outputs)

        # Update learning rate
        scheduler.step()

        # Early stopping based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model
    if best_model_state:
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
    pre = make_preprocessor().fit(X_train, Y_train)
    X_train = pre.transform(X_train) # may be Tensor or Tuple
    X_val   = pre.transform(X_val)
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val)

    # 2. Build model
    if isinstance(X_train, torch.Tensor):               # single-tensor case
        data_tensor = cast(torch.Tensor, X_train)       # pylint: disable=no-member
        input_shape = data_tensor.shape[1:]                # e.g. (F,)
        use_mask    = False
    else:                                               # tuple => (data, mask)
        data_tensor = cast(torch.Tensor, X_train[0])
        input_shape = data_tensor.shape[1:]             # e.g. (L, F)
        use_mask    = True                              
    model = make_model(input_shape, use_mask=use_mask)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
        toy_data = torch.zeros(8, *input_shape, dtype=torch.float32)
        if use_mask:
            toy_mask = torch.zeros(8, input_shape[0], dtype=torch.bool)
            toy_batch = (toy_data, toy_mask)
        else:
            toy_batch = toy_data

        toy_transformed = pre.transform(toy_batch)
        try:
            _ = trained_model(*toy_transformed) if isinstance(toy_transformed, (tuple, list)) \
                else trained_model(toy_transformed)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 5. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

    pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
    pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
    pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

    torch.save(trained_model.state_dict(), pth_state)
    with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
    with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

    # 6. Save plots
    _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
    _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

    # 7. Write JSON Summary
    if not dryrun: 
        summary = {
            "epochs": n_epochs,
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

