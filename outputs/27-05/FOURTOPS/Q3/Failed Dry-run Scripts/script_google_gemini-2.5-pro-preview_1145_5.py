
import os, sys, pickle, torch, gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
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

def make_loaders(X_train, Y_train, X_val, Y_val, batch=512):
    train_ds = TensorDataset(X_train, Y_train)
    val_ds   = TensorDataset(X_val , Y_val)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0))
                        
# ----------------  START OF LLM BLOCK  ----------------

import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# 0. ---------- IMPORTS ----------
# (Already covered by the template and above)

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.means = torch.empty(4) # E_T_miss, obj_E, obj_pT, obj_eta
        self.stds = torch.empty(4)
        # Constants for feature structure
        self.global_feature_indices = [0, 1] # E_T_miss, phi_E_T_miss
        self.num_objects = 18
        self.object_feature_size = 5 # type, E, pT, eta, phi

    def fit(self, X, y=None):
        # Global features: E_T_miss
        e_t_miss = X[:, 0]
        self.means[0] = torch.mean(e_t_miss)
        self.stds[0] = torch.std(e_t_miss)

        # Object features: E, pT, eta
        # Reshape to (N, num_objects, object_feature_size)
        objects = X[:, 2:].reshape(X.shape[0], self.num_objects, self.object_feature_size)
        
        obj_type = objects[:, :, 0]
        # Mask for valid (non-padded) objects. Assuming type 0 is padding.
        valid_obj_mask = (obj_type != 0)

        # Extract E, pT, eta for valid objects only
        obj_e = objects[:, :, 1][valid_obj_mask]
        obj_pt = objects[:, :, 2][valid_obj_mask]
        obj_eta = objects[:, :, 3][valid_obj_mask]

        if len(obj_e) > 0: # Ensure there are valid objects to compute stats
            self.means[1] = torch.mean(obj_e)
            self.stds[1] = torch.std(obj_e)
            self.means[2] = torch.mean(obj_pt)
            self.stds[2] = torch.std(obj_pt)
            self.means[3] = torch.mean(obj_eta)
            self.stds[3] = torch.std(obj_eta)
        else: # Fallback if no valid objects in training data (highly unlikely)
            self.means[1:] = 0.0
            self.stds[1:] = 1.0

        # Prevent division by zero if std is 0 for any feature
        self.stds[self.stds == 0] = 1.0
        return self

    def transform(self, X):
        N = X.shape[0]

        # Process global features
        e_t_miss = X[:, self.global_feature_indices[0]]
        phi_e_t_miss = X[:, self.global_feature_indices[1]]

        scaled_e_t_miss = (e_t_miss - self.means[0]) / self.stds[0]
        cos_phi_met = torch.cos(phi_e_t_miss)
        sin_phi_met = torch.sin(phi_e_t_miss)
        
        global_feats = torch.stack([scaled_e_t_miss, cos_phi_met, sin_phi_met], dim=1) # Shape: (N, 3)

        # Process object features
        objects_in = X[:, 2:].reshape(N, self.num_objects, self.object_feature_size)
        
        obj_type = objects_in[:, :, 0].clone() # Shape: (N, 18)
        obj_e = objects_in[:, :, 1]
        obj_pt = objects_in[:, :, 2]
        obj_eta = objects_in[:, :, 3]
        obj_phi = objects_in[:, :, 4]

        # Scale kinematic variables
        scaled_e = (obj_e - self.means[1]) / self.stds[1]
        scaled_pt = (obj_pt - self.means[2]) / self.stds[2]
        scaled_eta = (obj_eta - self.means[3]) / self.stds[3]
        
        cos_phi = torch.cos(obj_phi)
        sin_phi = torch.sin(obj_phi)

        # Create mask for active (non-padded) objects based on original obj_type
        # Padded objects have obj_type == 0. Their E,pT,eta,phi are also 0.
        active_obj_mask_unsqueeze = (obj_type != 0).float().unsqueeze(-1) # Shape: (N, 18, 1)

        # Apply mask: features of padded objects should be 0 (except obj_type itself which is already 0)
        scaled_e_masked = scaled_e * active_obj_mask_unsqueeze.squeeze(-1)
        scaled_pt_masked = scaled_pt * active_obj_mask_unsqueeze.squeeze(-1)
        scaled_eta_masked = scaled_eta * active_obj_mask_unsqueeze.squeeze(-1)
        cos_phi_masked = cos_phi * active_obj_mask_unsqueeze.squeeze(-1) 
        sin_phi_masked = sin_phi * active_obj_mask_unsqueeze.squeeze(-1)

        # Assemble object features: [type, E, pT, eta, cos_phi, sin_phi]
        # Each component is (N, 18, 1)
        object_features_list = [
            obj_type.unsqueeze(-1), 
            scaled_e_masked.unsqueeze(-1),
            scaled_pt_masked.unsqueeze(-1),
            scaled_eta_masked.unsqueeze(-1),
            cos_phi_masked.unsqueeze(-1),
            sin_phi_masked.unsqueeze(-1)
        ]
        object_feats = torch.cat(object_features_list, dim=-1) # Shape: (N, 18, 6)
        
        flat_object_feats = object_feats.reshape(N, -1) # Shape: (N, 18 * 6 = 108)

        # Concatenate global and object features
        final_features = torch.cat([global_feats, flat_object_feats], dim=1) # Shape: (N, 3 + 108 = 111)
        return final_features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class SlotAttentionModel(nn.Module):
    def __init__(self, input_dim_flat, d_model=64, n_slots=8, n_iters=3, obj_type_embedding_dim=8, max_obj_types=7, num_objects=18, raw_obj_feat_dim=6, global_feat_dim=3):
        super().__init__()
        self.d_model = d_model
        self.n_slots = n_slots
        self.n_iters = n_iters
        self.num_objects = num_objects
        self.raw_obj_feat_dim = raw_obj_feat_dim # Dim of preprocessed obj vector (type, E, pT, eta, cos_phi, sin_phi)
        self.global_feat_dim = global_feat_dim

        self.obj_type_embedding = nn.Embedding(max_obj_types, obj_type_embedding_dim) 
        
        # After embedding type: obj_type_embedding_dim + (raw_obj_feat_dim - 1 for type)
        particle_feature_dim_after_embed = obj_type_embedding_dim + (self.raw_obj_feat_dim - 1)
        
        self.particle_mlp = nn.Sequential(
            nn.Linear(particle_feature_dim_after_embed, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.slots_init = nn.Parameter(torch.randn(1, self.n_slots, self.d_model))
        
        self.norm_inputs = nn.LayerNorm(d_model)
        self.norm_slots = nn.LayerNorm(d_model)

        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.to_q_slots = nn.Linear(d_model, d_model, bias=False)

        self.slot_update_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(self.global_feat_dim, d_model // 2),
            nn.ReLU()
        )

        # Classifier: takes flattened slot features and processed global features
        classifier_input_dim = self.n_slots * self.d_model + (d_model // 2)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        N, _ = x.shape

        # Split input into global and object features
        global_features_flat = x[:, :self.global_feat_dim]
        object_features_flat = x[:, self.global_feat_dim:]
        
        object_features = object_features_flat.reshape(N, self.num_objects, self.raw_obj_feat_dim)

        obj_type = object_features[:, :, 0].long() # (N, num_objects)
        obj_kinematics = object_features[:, :, 1:]  # (N, num_objects, raw_obj_feat_dim-1)

        # Create padding mask for object attention (True for padded objects)
        obj_padding_mask = (obj_type == 0) # (N, num_objects)

        embedded_obj_type = self.obj_type_embedding(obj_type) # (N, num_objects, obj_type_embedding_dim)
        full_particle_features = torch.cat([embedded_obj_type, obj_kinematics], dim=-1)
        particle_embeddings = self.particle_mlp(full_particle_features) # (N, num_objects, d_model)

        # Slot Attention
        slots = self.slots_init.expand(N, -1, -1) # (N, n_slots, d_model)

        particle_embeddings_norm = self.norm_inputs(particle_embeddings)
        k = self.to_k(particle_embeddings_norm) # (N, num_objects, d_model)
        v = self.to_v(particle_embeddings_norm) # (N, num_objects, d_model)

        for _ in range(self.n_iters):
            slots_norm = self.norm_slots(slots)
            q_slots = self.to_q_slots(slots_norm) # (N, n_slots, d_model)

            attn_scores = torch.matmul(q_slots, k.transpose(-1, -2)) / (self.d_model ** 0.5) # (N, n_slots, num_objects)
            
            # Apply padding mask to attention scores before softmax
            # Mask is (N,num_objects), needs to be (N,n_slots,num_objects)
            # True in obj_padding_mask means it's a padded object, should have -inf score
            attn_scores = attn_scores.masked_fill(obj_padding_mask.unsqueeze(1), float('-inf'))
            
            attn_dist = F.softmax(attn_scores, dim=-1) # (N, n_slots, num_objects)

            updates = torch.matmul(attn_dist, v) # (N, n_slots, d_model)
            slots = slots + self.slot_update_mlp(updates) # Apply MLP update

        # Process global features
        processed_global_features = self.global_mlp(global_features_flat) # (N, d_model // 2)

        # Combine slot features and global features for classification
        slot_features_flat = slots.reshape(N, -1) # (N, n_slots * d_model)
        combined_features = torch.cat([slot_features_flat, processed_global_features], dim=-1)
        
        logits = self.classifier(combined_features) # (N, 1)
        return logits

def make_model(input_dim: int):
    # input_dim is the total number of features from the preprocessor (e.g., 111)
    # Model parameters can be adjusted here if needed.
    model = SlotAttentionModel(input_dim_flat=input_dim)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 20 # Adjusted for potentially faster training
BATCH_SIZE = 256 # Adjusted for memory and speed
LEARNING_RATE = 1e-3

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                total_val += labels.size(0)
                correct_val += (preds == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        # This print statement is for local debugging, should be removed for submission usually.
        # print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

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
    model = make_model(input_dim=X_train.shape[1])
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 3. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
        toy = torch.zeros(8, X_train.shape[1])      # 8 fake events
        try:
            _ = trained_model(pre.transform(toy))
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return  # no files in dry-run

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

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

