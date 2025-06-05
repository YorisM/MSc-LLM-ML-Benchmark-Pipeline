
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

# <start code template>
# 0. ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules, torch or sklearn (sub-)modules you actually use.
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy # For deepcopying model state dict for early stopping

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    #    Must implement:
    #   - fit(...) -> self
    #   - transform(X: Tensor)   -> Tensor  **or**  Tuple[Tensor, Tensor]

    # REQUIREMENTS
    # IMPORTANT: All state must be picklable with the std-lib pickle module.
    # IMPORTANT: Batch first.
    # May allocate NumPy arrays or Torch tensors internally, but:
    # transform() must be deterministic.
    # Store only derived parameters needed for transform i.e. do not store the raw data
    # itself in the preprocessor object.

    # DATA SPECIFICS
    # IMPORTANT: X_train, Y_train, X_val, Y_val are provided as PyTorch tensors in the environment.
    # Total flat length per event (X_train & X_val): 92
    # Index  0 :  missing-ET magnitude  (E_T_miss)
    # Index  1 :  missing-ET azimuth    (phi_Et_miss)
    # Indices  2-6  : object 1  ->  obj_1, E_1, p_T1, eta_1, phi_1
    # Indices  7-11 : object 2  -> obj_2, E_2 , p_T_2 , eta_2 , phi_2
    # ...
    # Indices 88-92 : object 18 -> obj_18, E_18 , p_T_18 , eta_18 , phi_18
    # Per-object slice size = 5
    # Max objects encoded   = 18

    # TIPS
    # When modifying data features or feature engineering: annotate tensor size as comments after 
    # each tensor operation to reduce dimension mismatches.

    def __init__(self):
        # Initialize mean and std tensors. These will be updated in fit().
        # Using torch.tensor ensures they are picklable and can be moved to devices if needed.
        # Global features
        self.global_E_T_miss_mean = torch.tensor(0.0, dtype=torch.float32)
        self.global_E_T_miss_std = torch.tensor(1.0, dtype=torch.float32)

        # Object features (E, pT, eta)
        self.obj_E_mean = torch.tensor(0.0, dtype=torch.float32)
        self.obj_E_std = torch.tensor(1.0, dtype=torch.float32)
        self.obj_pT_mean = torch.tensor(0.0, dtype=torch.float32)
        self.obj_pT_std = torch.tensor(1.0, dtype=torch.float32)
        self.obj_eta_mean = torch.tensor(0.0, dtype=torch.float32)
        self.obj_eta_std = torch.tensor(1.0, dtype=torch.float32)

        self.num_objects = 18
        self.num_object_features_original = 5 # obj_id, E, pT, eta, phi

    def fit(self, X, y=None):
        # X is a torch.Tensor (N, 92)

        # Global E_T_miss (feature 0)
        global_E_T_miss_raw = X[:, 0:1] # (N, 1)
        log_global_E_T_miss = torch.log1p(global_E_T_miss_raw) # log(1+x)
        self.global_E_T_miss_mean = log_global_E_T_miss.mean().float()
        self.global_E_T_miss_std = log_global_E_T_miss.std().clamp(min=1e-6).float()

        # Object features
        objects_flat = X[:, 2:] # (N, 90)
        objects = objects_flat.reshape(X.shape[0], self.num_objects, self.num_object_features_original) # (N, 18, 5)

        obj_E_raw   = objects[:, :, 1] # (N, 18) E is at index 1 of object features
        obj_pT_raw  = objects[:, :, 2] # (N, 18) pT is at index 2
        obj_eta_raw = objects[:, :, 3] # (N, 18) eta is at index 3

        # Create a mask for valid objects (pT > 0). Use a small epsilon for float comparisons.
        valid_obj_mask = obj_pT_raw > 1e-9 # (N, 18)

        if valid_obj_mask.any(): # Ensure there are valid objects to compute stats
            # Log-transform E and pT for valid objects
            log_obj_E = torch.log1p(obj_E_raw[valid_obj_mask])
            log_obj_pT = torch.log1p(obj_pT_raw[valid_obj_mask])

            self.obj_E_mean = log_obj_E.mean().float()
            self.obj_E_std  = log_obj_E.std().clamp(min=1e-6).float()

            self.obj_pT_mean = log_obj_pT.mean().float()
            self.obj_pT_std  = log_obj_pT.std().clamp(min=1e-6).float()

            # Eta is not log-transformed
            self.obj_eta_mean = obj_eta_raw[valid_obj_mask].mean().float()
            self.obj_eta_std  = obj_eta_raw[valid_obj_mask].std().clamp(min=1e-6).float()
        # If no valid objects, means/stds remain 0/1 as initialized.

        return self

    def transform(self, X):
        # X is a torch.Tensor (B, 92) or (N, 92)
        batch_size = X.shape[0]

        # 1. Global features
        global_E_T_miss_raw = X[:, 0:1] # (B, 1)
        global_phi_miss_raw = X[:, 1:2] # (B, 1)

        # Scale E_T_miss
        scaled_global_E_T_miss = (torch.log1p(global_E_T_miss_raw) - self.global_E_T_miss_mean) / self.global_E_T_miss_std # (B, 1)

        # Trigonometric encoding for phi_miss
        global_phi_miss_cos = torch.cos(global_phi_miss_raw) # (B, 1)
        global_phi_miss_sin = torch.sin(global_phi_miss_raw) # (B, 1)

        processed_global_features = torch.cat([scaled_global_E_T_miss, global_phi_miss_cos, global_phi_miss_sin], dim=1) # (B, 3)

        # 2. Object features
        objects_flat = X[:, 2:] # (B, 90)
        objects = objects_flat.reshape(batch_size, self.num_objects, self.num_object_features_original) # (B, 18, 5)

        # obj_id = objects[:, :, 0:1] # (B, 18, 1) - not used as feature
        obj_E_raw   = objects[:, :, 1:2] # (B, 18, 1)
        obj_pT_raw  = objects[:, :, 2:3] # (B, 18, 1)
        obj_eta_raw = objects[:, :, 3:4] # (B, 18, 1)
        obj_phi_raw = objects[:, :, 4:5] # (B, 18, 1)

        # Create mask based on pT > 0
        # Squeeze to make mask (B, 18). Clamp pT to avoid issues with very small values if not using epsilon.
        # Here, pT_raw is (B, 18, 1). Squeezing it results in (B, 18).
        padding_mask = (obj_pT_raw.squeeze(-1) > 1e-9) # (B, 18), True for valid objects

        # Scale object features
        scaled_obj_E = (torch.log1p(obj_E_raw) - self.obj_E_mean) / self.obj_E_std       # (B, 18, 1)
        scaled_obj_pT = (torch.log1p(obj_pT_raw) - self.obj_pT_mean) / self.obj_pT_std   # (B, 18, 1)
        scaled_obj_eta = (obj_eta_raw - self.obj_eta_mean) / self.obj_eta_std           # (B, 18, 1)

        # Trigonometric encoding for object phi
        obj_phi_cos = torch.cos(obj_phi_raw) # (B, 18, 1)
        obj_phi_sin = torch.sin(obj_phi_raw) # (B, 18, 1)

        processed_obj_features = torch.cat([scaled_obj_E, scaled_obj_pT, scaled_obj_eta, obj_phi_cos, obj_phi_sin], dim=2) # (B, 18, 5)

        # 3. Combine features
        # Expand global features to be concatenated with each object
        expanded_global_features = processed_global_features.unsqueeze(1).repeat(1, self.num_objects, 1) # (B, 18, 3)

        final_features = torch.cat([processed_obj_features, expanded_global_features], dim=2) # (B, 18, 5+3=8)

        # Apply mask to zero out features of padded objects (optional, good practice if not using attention mask properly elsewhere)
        # final_features = final_features * padding_mask.unsqueeze(-1).float() # Not strictly necessary if using src_key_padding_mask

        return final_features, padding_mask # (B, 18, 8), (B, 18)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class BinaryClassifier(nn.Module):
    # A UNIVERSAL wrapper.  You *must* keep forward(self, data, mask=None)
    # so it works for BOTH loader paths:
    # (data, label)              -> forward(data)
    # ((data, mask), label)      -> forward(data, mask)

    def __init__(self, input_shape: tuple[int, ...], *, use_mask: bool):
        super().__init__()
        self.use_mask = use_mask # True if input_shape is (L,F) for sequence

        # input_shape expected to be (L, F_in) for sequence, e.g., (18, 8)
        # F_in is num_features_per_object after preprocessing
        L_seq, F_in = input_shape 

        d_model = 128 # Dimension of transformer
        nhead = 8      # Number of attention heads
        num_encoder_layers = 4 # Number of transformer encoder layers
        dim_feedforward = 512 # Dimension of feedforward network in transformer
        dropout_rate = 0.1

        self.input_projection = nn.Linear(F_in, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True # IMPORTANT: data is (Batch, Seq, Feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, data: torch.Tensor, mask: torch.Tensor | None = None):
        # data : Tensor (B, L, F_in) e.g. (B, 18, 8)
        # mask : BoolTensor (B, L) e.g. (B, 18), True for valid elements

        x = self.input_projection(data) # (B, L, F_in) -> (B, L, d_model)

        if mask is not None:
            # TransformerEncoder src_key_padding_mask: True for PAD positions
            src_key_padding_mask = ~mask # Invert mask
            encoded_seq = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask) # (B, L, d_model)

            # Masked average pooling
            # Zero out pad positions before sum, use mask for count
            masked_encoded_seq = encoded_seq.masked_fill(~mask.unsqueeze(-1), 0.0) # (B, L, d_model)
            summed_tokens = masked_encoded_seq.sum(dim=1) # (B, d_model)
            num_valid_tokens = mask.sum(dim=1, keepdim=True).clamp(min=1e-9) # (B, 1)
            pooled_representation = summed_tokens / num_valid_tokens # (B, d_model)
        else:
            # This case should not happen if use_mask is True and preprocessor returns mask
            # If no mask, assume all elements are valid (e.g. for flat features)
            encoded_seq = self.transformer_encoder(x) # (B, L, d_model)
            pooled_representation = encoded_seq.mean(dim=1) # (B, d_model)

        logits = self.output_mlp(pooled_representation) # (B, 1)
        return logits.squeeze(-1) # (B,)

def make_model(input_shape, *, use_mask=False):
    return BinaryClassifier(input_shape, use_mask=use_mask)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 100
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5) # Removed verbose as requested

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience_early_stopping = 15 # Number of epochs to wait for improvement before stopping
    best_model_state_dict = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, batch in enumerate(train_loader):
            if model.use_mask:
                (data_tuple, labels) = batch
                data_seq, M_mask = data_tuple
                data_seq, M_mask, labels = data_seq.to(device), M_mask.to(device), labels.to(device).float()
                outputs = model(data_seq, M_mask)
            else: # Should not happen with this preprocessor
                (data, labels) = batch
                data, labels = data.to(device), labels.to(device).float()
                outputs = model(data)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0) # loss.item() is mean loss for batch

            preds_classes = (torch.sigmoid(outputs) > 0.5).long()
            correct_train += (preds_classes == labels.long()).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                if model.use_mask:
                    (data_tuple, labels) = batch
                    data_seq, M_mask = data_tuple
                    data_seq, M_mask, labels = data_seq.to(device), M_mask.to(device), labels.to(device).float()
                    outputs = model(data_seq, M_mask)
                else:
                    (data, labels) = batch
                    data, labels = data.to(device), labels.to(device).float()
                    outputs = model(data)

                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * labels.size(0)

                preds_classes = (torch.sigmoid(outputs) > 0.5).long()
                correct_val += (preds_classes == labels.long()).sum().item()
                total_val += labels.size(0)

                # Store predictions and labels for AUC (optional, if needed for deeper analysis)
                # all_val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                # all_val_labels.extend(labels.cpu().numpy())


        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        # roc_auc = roc_auc_score(np.array(all_val_labels), np.array(all_val_preds))
        # print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val AUC: {roc_auc:.4f}")
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")


        scheduler.step(epoch_val_loss)

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_model_state_dict = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience_early_stopping:
            print(f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation loss.")
            if best_model_state_dict:
                model.load_state_dict(best_model_state_dict)
            break

    if epochs_no_improve < patience_early_stopping and best_model_state_dict: # Ensure best model is loaded if training finished regularly
         model.load_state_dict(best_model_state_dict)

    return model, train_losses, val_losses, train_accs, val_accs

# <end code template>

# ----------------  END OF LLM-CODE BLOCK ----------------
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
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
        temp_ref    = X_train
        input_shape = temp_ref.shape[1:]                # e.g. (F,)
        use_mask    = False
    else:                                               # tuple => (data, mask)
        temp_ref    = X_train
        input_shape = temp_ref[0].shape[1:]             # e.g. (L, F)
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

    # 4. *Dry-run safety check* - run a single toy forward pass
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

