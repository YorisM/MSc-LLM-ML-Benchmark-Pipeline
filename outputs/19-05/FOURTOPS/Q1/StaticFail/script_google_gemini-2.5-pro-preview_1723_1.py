
import os, sys, json, pickle, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"

DATASET = {
    "X_train": "./challenges/FOURTOPS/data/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/Y_val.csv"
}
                       
def load_data():
    X_train_df = pd.read_csv('./challenges/FOURTOPS/data/X_train.csv')
    Y_train_df = pd.read_csv('./challenges/FOURTOPS/data/Y_train.csv')
    X_val_df   = pd.read_csv('./challenges/FOURTOPS/data/X_val.csv')
    Y_val_df   = pd.read_csv('./challenges/FOURTOPS/data/Y_val.csv')

    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_df.values, dtype=torch.long).squeeze()
    X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
    Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long).squeeze()
    return X_train, Y_train, X_val, Y_val

def make_loaders(X_train, Y_train, X_val, Y_val, batch=1024):
    train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train))
    val = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val))
    return (DataLoader(train, batch_size=batch, shuffle=True),
            DataLoader(val, batch_size=batch))
                        
# ----------------  START OF LLM BLOCK  ----------------
# Imports
import os, sys, json, pickle, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
# Only import extra std-lib modules, torch.nn or sklearn sub-modules you actually use.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

class MyPreprocessor:
    def __init__(self):
        self.scaler_etmiss = StandardScaler()
        self.scaler_n_objects = StandardScaler()
        self.scaler_ht = StandardScaler()
        self.scaler_obj_E = StandardScaler()
        self.scaler_obj_pT = StandardScaler()
        self.scaler_obj_eta = StandardScaler()
        self.fitted = False
        self.num_objects = 18
        self.features_per_object = 5

    def fit(self, X, y=None):
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = np.asarray(X)

        N = X_np.shape[0]
        E_T_miss = X_np[:, 0]
        # phi_Et_miss = X_np[:, 1] # Not needed for fitting scalers, used in transform
        objects_flat = X_np[:, 2:]
        # obj_id, E, p_T, eta, phi
        objects = objects_flat.reshape(N, self.num_objects, self.features_per_object)

        # Use p_T > 0 to identify valid objects, as physical objects must have p_T > 0.
        # Padded objects will have p_T = 0.
        is_valid_object = objects[:, :, 2] > 1e-6 # Add epsilon for float comparisons

        # Log transform relevant features (add 1 for stability, log1p(x) = log(1+x))
        E_T_miss_log = np.log1p(E_T_miss)
        
        obj_E = objects[:, :, 1]
        obj_pT = objects[:, :, 2]
        obj_eta = objects[:, :, 3]

        obj_E_log = np.log1p(obj_E)
        obj_pT_log = np.log1p(obj_pT)

        # Derived global features for fitting scalers
        n_objects_feat = is_valid_object.sum(axis=1)
        ht_feat = obj_pT.sum(axis=1) # Sum of original pT for HT
        ht_log_feat = np.log1p(ht_feat)

        # Fit scalers
        self.scaler_etmiss.fit(E_T_miss_log.reshape(-1, 1))
        self.scaler_n_objects.fit(n_objects_feat.reshape(-1, 1))
        self.scaler_ht.fit(ht_log_feat.reshape(-1, 1))

        # Collect all valid object features for fitting object scalers
        # These will be 1D arrays of all valid values across all objects and events
        valid_obj_E_log = obj_E_log[is_valid_object]
        valid_obj_pT_log = obj_pT_log[is_valid_object]
        valid_obj_eta = obj_eta[is_valid_object]

        # Ensure there's data to fit on (at least one valid object in training set)
        if valid_obj_E_log.size > 0:
            self.scaler_obj_E.fit(valid_obj_E_log.reshape(-1, 1))
        if valid_obj_pT_log.size > 0:
            self.scaler_obj_pT.fit(valid_obj_pT_log.reshape(-1, 1))
        if valid_obj_eta.size > 0:
            self.scaler_obj_eta.fit(valid_obj_eta.reshape(-1, 1))
        
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform!")

        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = np.asarray(X)

        N = X_np.shape[0]
        E_T_miss = X_np[:, 0]
        phi_Et_miss = X_np[:, 1]
        objects_flat = X_np[:, 2:]
        raw_objects = objects_flat.reshape(N, self.num_objects, self.features_per_object)

        # Sort objects by p_T descending. Padded objects (p_T=0) go to end.
        # raw_objects structure: [obj_id, E, p_T, eta, phi]
        obj_pT_for_sorting = raw_objects[:, :, 2]
        # Create an array of indices that sort pT in descending order for each event
        # Adding a small epsilon from object index to break pT ties consistently
        stable_sort_key = obj_pT_for_sorting - np.arange(self.num_objects)[np.newaxis, :] * 1e-9
        sort_indices = np.argsort(-stable_sort_key, axis=1)

        # Apply sorting using advanced indexing
        I = np.arange(N)[:, np.newaxis]
        sorted_objects = raw_objects[I, sort_indices]

        # Global features processing
        E_T_miss_log = np.log1p(E_T_miss)
        scaled_E_T_miss_log = self.scaler_etmiss.transform(E_T_miss_log.reshape(-1, 1))
        cos_phi_Et_miss = np.cos(phi_Et_miss).reshape(-1, 1)
        sin_phi_Et_miss = np.sin(phi_Et_miss).reshape(-1, 1)

        # Derived global features
        # Use p_T > 0 from sorted objects to count
        is_valid_object_sorted = sorted_objects[:, :, 2] > 1e-6
        n_objects_feat = is_valid_object_sorted.sum(axis=1)
        scaled_n_objects = self.scaler_n_objects.transform(n_objects_feat.reshape(-1, 1))

        ht_feat = sorted_objects[:, :, 2].sum(axis=1) # Sum of pT of sorted objects
        ht_log_feat = np.log1p(ht_feat)
        scaled_ht_log = self.scaler_ht.transform(ht_log_feat.reshape(-1, 1))

        processed_global_features_list = [
            scaled_E_T_miss_log, cos_phi_Et_miss, sin_phi_Et_miss,
            scaled_n_objects, scaled_ht_log
        ]

        # Object features processing from sorted_objects
        obj_E = sorted_objects[:, :, 1]
        obj_pT = sorted_objects[:, :, 2]
        obj_eta = sorted_objects[:, :, 3]
        obj_phi = sorted_objects[:, :, 4]

        obj_E_log = np.log1p(obj_E)
        obj_pT_log = np.log1p(obj_pT)

        # Apply scaling. Reshape for scaler, then reshape back.
        # Scalers handle zero inputs (e.g. for padded objects log1p(0)=0) appropriately if they saw such values during fit.
        scaled_obj_E_log = self.scaler_obj_E.transform(obj_E_log.reshape(-1,1)).reshape(N, self.num_objects, 1) if valid_obj_E_log.size > 0 else np.zeros_like(obj_E_log.reshape(N,self.num_objects,1))
        scaled_obj_pT_log = self.scaler_obj_pT.transform(obj_pT_log.reshape(-1,1)).reshape(N, self.num_objects, 1) if valid_obj_pT_log.size > 0 else np.zeros_like(obj_pT_log.reshape(N,self.num_objects,1))
        scaled_obj_eta = self.scaler_obj_eta.transform(obj_eta.reshape(-1,1)).reshape(N, self.num_objects, 1) if valid_obj_eta.size > 0 else np.zeros_like(obj_eta.reshape(N,self.num_objects,1))

        cos_obj_phi = np.cos(obj_phi).reshape(N, self.num_objects, 1)
        sin_obj_phi = np.sin(obj_phi).reshape(N, self.num_objects, 1)

        # Concatenate features for each object: (E_log_scaled, pT_log_scaled, eta_scaled, cos_phi, sin_phi)
        # Original obj_id is dropped.
        processed_object_features = np.concatenate(
            [scaled_obj_E_log, scaled_obj_pT_log, scaled_obj_eta, cos_obj_phi, sin_obj_phi],
            axis=2 # Concatenate along feature dimension for each object
        )

        # Flatten object features: (N, num_objects * features_per_transformed_object)
        processed_object_features_flat = processed_object_features.reshape(N, -1)

        # Concatenate all features: global_features + flattened_object_features
        final_features_list = processed_global_features_list + [processed_object_features_flat]
        final_features_np = np.concatenate(final_features_list, axis=1)

        return torch.from_numpy(final_features_np.astype(np.float32))

def make_preprocessor():
    return MyPreprocessor()

def make_model(input_dim: int):
    # Total global features = 1 (ETmiss) + 2 (ETmiss_phi) + 1 (n_obj) + 1 (HT) = 5
    # Total object features = 18 objects * (1(E) + 1(pT) + 1(eta) + 2(phi)) = 18 * 5 = 90
    # Expected input_dim = 5 + 90 = 95
    model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.4),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.4),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.4),
        nn.Linear(64, 1)  # Output for BCEWithLogitsLoss
    )
    return model

EPOCHS = 25 # Adjusted for CPU training within 2h.

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    # Scheduler can be useful, CosineAnnealingLR is simple and doesn't require val_loss monitoring for step.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []
    # train_auc_hist, val_auc_hist = [], [] # Not in template, but good to track

    print(f"Starting training for {epochs} epochs on {device}.")

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        # train_preds_for_auc, train_labels_for_auc = [], [] # For epoch AUC

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)
            # For AUC calculation:
            # train_preds_for_auc.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            # train_labels_for_auc.extend(y_batch.cpu().numpy())

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        train_loss_hist.append(epoch_train_loss)
        train_acc_hist.append(epoch_train_acc)
        # epoch_train_auc = roc_auc_score(train_labels_for_auc, train_preds_for_auc)
        # train_auc_hist.append(epoch_train_auc)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_preds_for_auc, val_labels_for_auc = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).float().unsqueeze(1)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
                
                val_preds_for_auc.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels_for_auc.extend(y_batch.cpu().numpy())

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        val_loss_hist.append(epoch_val_loss)
        val_acc_hist.append(epoch_val_acc)
        
        epoch_val_auc = 0.0
        if len(val_labels_for_auc) > 0 and len(np.unique(val_labels_for_auc)) > 1: # Check for trivial cases
             epoch_val_auc = roc_auc_score(val_labels_for_auc, val_preds_for_auc)
        # val_auc_hist.append(epoch_val_auc) # Not in template

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val AUC: {epoch_val_auc:.4f}")

        scheduler.step() # For CosineAnnealingLR, step per epoch. For ReduceLROnPlateau, use scheduler.step(epoch_val_loss)

    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist
# ----------------  END OF LLM BLOCK ----------------
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_tr, y_tr, X_va, y_va = load_data()
    pre = make_preprocessor();  pre.fit(X_tr, y_tr)
    X_tr = pre.transform(X_tr); X_va = pre.transform(X_va)
    tr_loader, va_loader = make_loaders(X_tr, y_tr, X_va, y_va)

    # 2. Build model
    model = make_model(input_dim=X_tr.shape[1])
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    trained, tr_loss, va_loss, tr_acc, va_acc = train_model(
        model, tr_loader, va_loader, epochs=n_epochs
    )

    # 3. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
        toy = torch.zeros(8, X_tr.shape[1])      # 8 fake events
        try:
            _ = trained(pre.transform(toy))
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return  # no files in dry-run

    # 4. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
    torch.save(trained.state_dict(), f"{base}_state.pt")
    with open(f"{base}_model.pkl", "wb") as f: pickle.dump(trained, f)
    with open(f"{base}_preproc.pkl", "wb") as f: pickle.dump(pre, f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",      f"{base}_loss.png")
    _plot(tr_acc,  va_acc,  "Accuracy",  f"{base}_accuracy.png")

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

