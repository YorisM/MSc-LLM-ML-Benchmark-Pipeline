
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
from collections import OrderedDict
import math
from torch.nn import functional as F

# Define device (CPU given constraints)
device = torch.device("cpu")

# Constants
NUM_OBJECTS = 18
ORIG_OBJ_FEATURES = 5  # obj_id, E, pT, eta, phi
FLAT_GLOBAL_FEATURES = 2 # E_T_miss, phi_E_T_miss
EPS = 1e-7 # Small epsilon for log stability and division

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.is_fitted = False
        self.global_means = None
        self.global_stds = None
        self.particle_feature_means = None
        self.particle_feature_stds = None
        self.obj_id_map = {}
        self.num_obj_categories = 0
        self.engineered_global_dim = 3
        self.engineered_particle_dim_no_mask = 0
        self.final_feature_dim = 0

    def _get_particle_data_flat(self, X_event, particle_idx):
        start_idx = FLAT_GLOBAL_FEATURES + particle_idx * ORIG_OBJ_FEATURES
        return X_event[start_idx : start_idx + ORIG_OBJ_FEATURES]

    def fit(self, X, y=None):
        et_miss = X[:, 0]
        phi_et_miss = X[:, 1]
        log_et_miss_vals = torch.log(et_miss + EPS)
        cos_phi_et_miss_vals = torch.cos(phi_et_miss)
        sin_phi_et_miss_vals = torch.sin(phi_et_miss)
        
        stacked_global_features = torch.stack([log_et_miss_vals, cos_phi_et_miss_vals, sin_phi_et_miss_vals], dim=1)
        self.global_means = torch.mean(stacked_global_features, dim=0)
        self.global_stds = torch.std(stacked_global_features, dim=0)
        self.global_stds[self.global_stds < EPS] = 1.0

        all_present_obj_ids = []
        collected_log_E, collected_log_pT, collected_eta, collected_cos_phi, collected_sin_phi = [], [], [], [], []

        for i in range(X.shape[0]):
            for j in range(NUM_OBJECTS):
                particle_data = self._get_particle_data_flat(X[i], j)
                obj_id, E, pT, eta, phi = particle_data[0], particle_data[1], particle_data[2], particle_data[3], particle_data[4]

                if E > EPS:
                    all_present_obj_ids.append(int(obj_id.item()))
                    collected_log_E.append(torch.log(E + EPS))
                    collected_log_pT.append(torch.log(pT + EPS) if pT > EPS else torch.log(torch.tensor(EPS, device=X.device))) # Ensure pT is also positive for log
                    collected_eta.append(eta)
                    collected_cos_phi.append(torch.cos(phi))
                    collected_sin_phi.append(torch.sin(phi))

        unique_obj_ids = sorted(list(set(all_present_obj_ids)))
        self.obj_id_map = {val: i for i, val in enumerate(unique_obj_ids)}
        self.num_obj_categories = len(unique_obj_ids)
        if self.num_obj_categories == 0:
             self.obj_id_map = {0:0} # Dummy category if no objects found
             self.num_obj_categories = 1

        if len(collected_log_E) > 0:
            self.particle_feature_means = torch.tensor([
                torch.mean(torch.stack(collected_log_E)),
                torch.mean(torch.stack(collected_log_pT)),
                torch.mean(torch.stack(collected_eta)),
                torch.mean(torch.stack(collected_cos_phi)),
                torch.mean(torch.stack(collected_sin_phi))
            ])
            self.particle_feature_stds = torch.tensor([
                torch.std(torch.stack(collected_log_E)),
                torch.std(torch.stack(collected_log_pT)),
                torch.std(torch.stack(collected_eta)),
                torch.std(torch.stack(collected_cos_phi)),
                torch.std(torch.stack(collected_sin_phi))
            ])
            self.particle_feature_stds[self.particle_feature_stds < EPS] = 1.0
        else:
            self.particle_feature_means = torch.zeros(5)
            self.particle_feature_stds = torch.ones(5)

        self.engineered_particle_dim_no_mask = self.num_obj_categories + 5
        self.final_feature_dim = self.engineered_global_dim + NUM_OBJECTS * (self.engineered_particle_dim_no_mask + 1)
        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
        
        N_events = X.shape[0]
        processed_X = torch.zeros((N_events, self.final_feature_dim), dtype=torch.float32, device=X.device)

        et_miss = X[:, 0]
        phi_et_miss = X[:, 1]
        processed_X[:, 0] = (torch.log(et_miss + EPS) - self.global_means[0]) / self.global_stds[0]
        processed_X[:, 1] = (torch.cos(phi_et_miss) - self.global_means[1]) / self.global_stds[1]
        processed_X[:, 2] = (torch.sin(phi_et_miss) - self.global_means[2]) / self.global_stds[2]

        current_col_idx = self.engineered_global_dim
        for i in range(NUM_OBJECTS):
            particle_original_data = X[:, FLAT_GLOBAL_FEATURES + i*ORIG_OBJ_FEATURES : FLAT_GLOBAL_FEATURES + (i+1)*ORIG_OBJ_FEATURES]
            obj_id_orig, E_orig, pT_orig, eta_orig, phi_orig = particle_original_data[:,0], particle_original_data[:,1], particle_original_data[:,2], particle_original_data[:,3], particle_original_data[:,4]
            is_present_mask = (E_orig > EPS)

            obj_id_one_hot = torch.zeros((N_events, self.num_obj_categories), device=X.device)
            for k in range(N_events):
                if is_present_mask[k]:
                    obj_id_val = int(obj_id_orig[k].item())
                    if obj_id_val in self.obj_id_map:
                        obj_id_one_hot[k, self.obj_id_map[obj_id_val]] = 1.0
            processed_X[:, current_col_idx : current_col_idx + self.num_obj_categories] = obj_id_one_hot
            current_col_idx += self.num_obj_categories

            log_E, log_pT, eta_scaled, cos_phi_scaled, sin_phi_scaled = (torch.zeros_like(E_orig) for _ in range(5))
            if torch.any(is_present_mask):
                valid_E = E_orig[is_present_mask]
                valid_pT = pT_orig[is_present_mask]
                valid_eta = eta_orig[is_present_mask]
                valid_phi = phi_orig[is_present_mask]

                log_E[is_present_mask] = (torch.log(valid_E + EPS) - self.particle_feature_means[0]) / self.particle_feature_stds[0]
                # Handle pT=0 for log separately if it can occur with E>0, else assume pT>EPS if_present_mask
                pT_for_log = torch.where(valid_pT > EPS, valid_pT, torch.tensor(EPS, device=X.device))
                log_pT[is_present_mask] = (torch.log(pT_for_log + EPS) - self.particle_feature_means[1]) / self.particle_feature_stds[1]
                eta_scaled[is_present_mask] = (valid_eta - self.particle_feature_means[2]) / self.particle_feature_stds[2]
                cos_phi_scaled[is_present_mask] = (torch.cos(valid_phi) - self.particle_feature_means[3]) / self.particle_feature_stds[3]
                sin_phi_scaled[is_present_mask] = (torch.sin(valid_phi) - self.particle_feature_means[4]) / self.particle_feature_stds[4]

            kin_features = [log_E, log_pT, eta_scaled, cos_phi_scaled, sin_phi_scaled]
            for feat_tensor in kin_features:
                processed_X[:, current_col_idx] = feat_tensor
                current_col_idx += 1
            
            processed_X[:, current_col_idx] = is_present_mask.float()
            current_col_idx += 1
            
        return processed_X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn=nn.ReLU):
        super().__init__()
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(activation_fn())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim_inputs, dim_slot, num_iterations=3, mlp_hidden_size=128, epsilon=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.dim_slot = dim_slot

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim_slot))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, dim_slot))
        
        self.to_k = nn.Linear(dim_inputs, dim_slot, bias=False)
        self.to_v = nn.Linear(dim_inputs, dim_slot, bias=False)
        self.to_q = nn.Linear(dim_slot, dim_slot, bias=False)

        self.gru = nn.GRUCell(dim_slot, dim_slot)

        self.mlp = nn.Sequential(
            nn.Linear(dim_slot, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, dim_slot)
        )
        self.norm_inputs = nn.LayerNorm(dim_inputs)
        self.norm_slots  = nn.LayerNorm(dim_slot)
        self.norm_mlp = nn.LayerNorm(dim_slot)

    def forward(self, inputs, pad_mask=None):
        batch_size, num_inputs, _ = inputs.shape
        
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        inputs = self.norm_inputs(inputs)        
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q_slots = self.to_q(slots)

            attn_logits = torch.einsum('bsd,bnd->bsn', q_slots, k) / math.sqrt(self.dim_slot)
            
            if pad_mask is not None:
                attn_logits.masked_fill_(~pad_mask.unsqueeze(1), float('-inf'))

            attn = F.softmax(attn_logits, dim=-1)
            
            weights = attn + self.epsilon
            weights = weights / torch.sum(weights, dim=-1, keepdim=True)
            
            updates = torch.einsum('bsn,bnd->bsd', weights, v)

            slots = self.gru(
                updates.reshape(-1, self.dim_slot),
                slots_prev.reshape(-1, self.dim_slot)
            )
            slots = slots.reshape(batch_size, self.num_slots, self.dim_slot)
            slots = slots + self.mlp(self.norm_mlp(slots))
            
        return slots

class MyModel(nn.Module):
    def __init__(self, input_dim, engineered_global_dim, engineered_particle_dim_no_mask, num_obj_categories):
        super().__init__()
        self.engineered_global_dim = engineered_global_dim
        self.engineered_particle_dim_no_mask = engineered_particle_dim_no_mask
        self.num_obj_categories = num_obj_categories
        self.particle_input_feature_dim = engineered_particle_dim_no_mask

        self.particle_embed_dim = 128
        self.global_embed_dim = 64
        self.num_slots = 8
        self.slot_iterations = 3
        self.slot_mlp_hidden = 128
        self.transformer_nhead = 4
        self.transformer_ff_dim = self.particle_embed_dim * 4
        self.transformer_layers = 2
        
        self.particle_encoder = MLP(self.particle_input_feature_dim, 
                                    [self.particle_embed_dim, self.particle_embed_dim], 
                                    self.particle_embed_dim)
        
        self.global_encoder = MLP(self.engineered_global_dim, 
                                  [self.global_embed_dim], 
                                  self.global_embed_dim)

        self.slot_attention = SlotAttention(num_slots=self.num_slots,
                                            dim_inputs=self.particle_embed_dim,
                                            dim_slot=self.particle_embed_dim,
                                            num_iterations=self.slot_iterations,
                                            mlp_hidden_size=self.slot_mlp_hidden)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.particle_embed_dim, 
                                                   nhead=self.transformer_nhead,
                                                   dim_feedforward=self.transformer_ff_dim,
                                                   activation='relu',
                                                   batch_first=True,
                                                   norm_first=True)
        self.transformer_encoder_on_slots = nn.TransformerEncoder(encoder_layer, 
                                                                  num_layers=self.transformer_layers)
        
        classifier_input_dim = self.particle_embed_dim + self.global_embed_dim
        self.classifier_mlp = MLP(classifier_input_dim, 
                                  [128, 64], 
                                  1)

    def forward(self, x):
        batch_size = x.shape[0]
        global_features_flat = x[:, :self.engineered_global_dim]
        
        particle_block_size = self.engineered_particle_dim_no_mask + 1
        particles_flat = x[:, self.engineered_global_dim:]
        
        particles_structured = particles_flat.reshape(batch_size, NUM_OBJECTS, particle_block_size)
        
        particle_features_to_embed = particles_structured[:, :, :-1]
        particle_is_present_mask = particles_structured[:, :, -1].bool()

        embedded_particles = self.particle_encoder(particle_features_to_embed)
        embedded_global = self.global_encoder(global_features_flat)

        slots = self.slot_attention(embedded_particles, pad_mask=particle_is_present_mask)
        
        encoded_slots = self.transformer_encoder_on_slots(slots)

        aggregated_slots = encoded_slots.mean(dim=1)
        
        final_representation = torch.cat([aggregated_slots, embedded_global], dim=1)
        
        logits = self.classifier_mlp(final_representation)
        return logits.squeeze(-1)

def make_model(input_dim: int, preprocessor: MyPreprocessor):
    return MyModel(input_dim=input_dim,
                   engineered_global_dim=preprocessor.engineered_global_dim,
                   engineered_particle_dim_no_mask=preprocessor.engineered_particle_dim_no_mask,
                   num_obj_categories=preprocessor.num_obj_categories).to(device)

EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * X_batch.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct_train += (preds == y_batch.bool()).sum().item()
            total_train += y_batch.size(0)
        
        avg_train_loss = epoch_train_loss / total_train if total_train > 0 else 0
        avg_train_acc = correct_train / total_train if total_train > 0 else 0
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                epoch_val_loss += loss.item() * X_batch.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                correct_val += (preds == y_batch.bool()).sum().item()
                total_val += y_batch.size(0)

        avg_val_loss = epoch_val_loss / total_val if total_val > 0 else 0
        avg_val_acc = correct_val / total_val if total_val > 0 else 0
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)
        
        # print(f"Epoch {epoch+1}/{epochs} => Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        scheduler.step()

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

