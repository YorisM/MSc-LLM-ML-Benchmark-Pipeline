
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
import os, sys, json, pickle, torch
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # Not used in the final model code
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.optim as optim

class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        self.epsilon = 1e-7
        self.N_OBJECTS = 18
        self.OBJ_FEATURES_PER_OBJECT = 5 # type, E, pT, eta, phi
        self.pt_cutoff = 1e-3 # Min pT in MeV to be considered an active particle
        self.output_dim_ = -1 # To be set after fitting

    def _process_features(self, X_data_np: np.ndarray) -> np.ndarray:
        N = X_data_np.shape[0]
        
        met = X_data_np[:, 0]
        met_phi = X_data_np[:, 1]
        
        log_met = np.log(np.maximum(met, 0) + self.epsilon)
        met_x = met * np.cos(met_phi)
        met_y = met * np.sin(met_phi)

        obj_data_flat = X_data_np[:, 2:]
        obj_data = obj_data_flat.reshape(N, self.N_OBJECTS, self.OBJ_FEATURES_PER_OBJECT)
        
        obj_ids_orig = obj_data[:, :, 0]
        E_orig = obj_data[:, :, 1]
        pT_orig = obj_data[:, :, 2]
        eta_orig = obj_data[:, :, 3]
        phi_orig = obj_data[:, :, 4]

        active_mask = (pT_orig > self.pt_cutoff).astype(np.float32)
        
        num_active_objects = np.sum(active_mask, axis=1)
        HT = np.sum(pT_orig * active_mask, axis=1)

        log_E = np.zeros_like(E_orig)
        log_pT = np.zeros_like(pT_orig)
        
        # Apply log transform only to active objects with positive E/pT
        # Epsilon added for numerical stability, though E_orig/pT_orig > 0 here
        active_pos_E_mask = (E_orig > 0) & (active_mask > 0)
        log_E[active_pos_E_mask] = np.log(E_orig[active_pos_E_mask] + self.epsilon)
        
        active_pos_pT_mask = (pT_orig > 0) & (active_mask > 0)
        log_pT[active_pos_pT_mask] = np.log(pT_orig[active_pos_pT_mask] + self.epsilon)

        px = pT_orig * np.cos(phi_orig) * active_mask
        py = pT_orig * np.sin(phi_orig) * active_mask
        eta_masked = eta_orig * active_mask
        # obj_ids are sorted but not otherwise transformed; their original values are preserved for active objects.
        # For inactive objects, their features (logE, logPt, px, py, eta_masked) become 0.

        sort_indices = np.argsort(-pT_orig, axis=1) # Sort by pT descending

        obj_ids_sorted = np.take_along_axis(obj_ids_orig, sort_indices, axis=1)
        log_E_sorted = np.take_along_axis(log_E, sort_indices, axis=1)
        log_pT_sorted = np.take_along_axis(log_pT, sort_indices, axis=1)
        px_sorted = np.take_along_axis(px, sort_indices, axis=1)
        py_sorted = np.take_along_axis(py, sort_indices, axis=1)
        eta_masked_sorted = np.take_along_axis(eta_masked, sort_indices, axis=1)

        flat_obj_ids = obj_ids_sorted.reshape(N, -1)
        flat_log_E = log_E_sorted.reshape(N, -1)
        flat_log_pT = log_pT_sorted.reshape(N, -1)
        flat_eta_masked = eta_masked_sorted.reshape(N, -1)
        flat_px = px_sorted.reshape(N, -1)
        flat_py = py_sorted.reshape(N, -1)
        
        processed_features = np.concatenate([
            log_met[:, np.newaxis], 
            met_x[:, np.newaxis], 
            met_y[:, np.newaxis],
            num_active_objects[:, np.newaxis], 
            HT[:, np.newaxis],
            flat_obj_ids, 
            flat_log_E, 
            flat_log_pT, 
            flat_eta_masked, 
            flat_px, 
            flat_py
        ], axis=1)
        
        return processed_features

    def fit(self, X, y=None):
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X
            
        processed_X_np = self._process_features(X_np)
        self.scaler.fit(processed_X_np)
        self.fitted = True
        self.output_dim_ = processed_X_np.shape[1]
        return self

    def transform(self, X):
        if not self.fitted:
            # This case should ideally be handled by fitting on train data first.
            # For robustness in case fit wasn't called (e.g. loading a pickled preprocessor without fitting state)
            # but the challenge implies fit will always be called on training data.
            raise RuntimeError("Preprocessor must be fitted before transform.")
        
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X
            
        processed_X_np = self._process_features(X_np)
        scaled_X_np = self.scaler.transform(processed_X_np)
        return torch.from_numpy(scaled_X_np).float()

def make_preprocessor():
    return MyPreprocessor()

def make_model(input_dim: int):
    model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.35), # Adjusted dropout slightly
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.35),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.35),
        nn.Linear(64, 1) 
    )
    return model

EPOCHS = 40
    
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.2, verbose=False)

    train_losses, val_losses_list = [], [] # Renamed val_losses to val_losses_list to avoid conflict with loop var
    train_accs, val_accs = [], []

    for epoch_idx in range(epochs): # Renamed epoch to epoch_idx
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        epoch_train_loss = running_train_loss / train_total if train_total > 0 else 0.0
        epoch_train_acc = train_correct / train_total if train_total > 0 else 0.0
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels) # Use new var name for loop loss
                
                running_val_loss += val_loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = running_val_loss / val_total if val_total > 0 else 0.0
        epoch_val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_losses_list.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        scheduler.step(epoch_val_loss)
        
        # A print statement for monitoring, will be commented out or removed in final submission for contests
        # if (epoch_idx + 1) % 1 == 0: # Print every epoch
        #    print(f"Epoch {epoch_idx+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.1e}")

    return model, train_losses, val_losses_list, train_accs, val_accs
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

