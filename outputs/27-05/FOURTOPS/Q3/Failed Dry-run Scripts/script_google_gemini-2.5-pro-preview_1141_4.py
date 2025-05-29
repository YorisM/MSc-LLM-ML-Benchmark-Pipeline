
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
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# 0. ---------- IMPORTS ----------
# (already listed above)

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self, num_particle_features_original=5, max_particles=18):
        self.eps = 1e-5  # For log transformations
        self.eps_std = 1e-7  # For stable division by std
        self.stats_ = {}
        self.is_fit = False
        self.num_particle_features_original = num_particle_features_original
        self.max_particles = max_particles

    def fit(self, X, y=None):
        N = X.shape[0]

        # Global features
        et_miss = X[:, 0]
        self.stats_['et_miss_mean'] = et_miss.mean()
        self.stats_['et_miss_std'] = et_miss.std()

        # Particle features
        particles_original = X[:, 2:].reshape(N, self.max_particles, self.num_particle_features_original)
        
        # Mask for actual particles (pT > 0, assuming pT is at offset 2)
        # obj_type (offset 0), E (1), pT (2), eta (3), phi (4)
        pT_original = particles_original[:, :, 2]
        actual_particle_mask = pT_original > self.eps # use small eps to define active an particle

        # E (log-normalized)
        E_actual = particles_original[:, :, 1][actual_particle_mask]
        log_E_actual = torch.log(E_actual + self.eps)
        self.stats_['particle_logE_mean'] = log_E_actual.mean()
        self.stats_['particle_logE_std'] = log_E_actual.std()

        # pT (log-normalized)
        pT_actual = pT_original[actual_particle_mask] # Already have pT_original
        log_pT_actual = torch.log(pT_actual + self.eps)
        self.stats_['particle_logpT_mean'] = log_pT_actual.mean()
        self.stats_['particle_logpT_std'] = log_pT_actual.std()

        # eta (normalized)
        eta_actual = particles_original[:, :, 3][actual_particle_mask]
        self.stats_['particle_eta_mean'] = eta_actual.mean()
        self.stats_['particle_eta_std'] = eta_actual.std()
        
        self.is_fit = True
        return self

    def transform(self, X):
        if not self.is_fit:
            raise RuntimeError("Preprocessor must be fit before transform.")
        
        N = X.shape[0]
        device = X.device

        # Global features (1 E_T_miss -> 1 norm + 1 phi_miss -> 2 trig = 3 total)
        et_miss = X[:, 0:1]
        phi_et_miss = X[:, 1:2]

        norm_et_miss = (et_miss - self.stats_['et_miss_mean']) / (self.stats_['et_miss_std'] + self.eps_std)
        cos_phi_et_miss = torch.cos(phi_et_miss)
        sin_phi_et_miss = torch.sin(phi_et_miss)
        processed_global_feats = torch.cat([norm_et_miss, cos_phi_et_miss, sin_phi_et_miss], dim=1) # (N, 3)

        # Particle features (each 5 features -> 6 features: type, logE, logpT, eta, cosPhi, sinPhi)
        particles_original = X[:, 2:].reshape(N, self.max_particles, self.num_particle_features_original)
        
        obj_types = particles_original[:, :, 0:1] # (N, 18, 1)
        E_p = particles_original[:, :, 1:2]
        pT_p = particles_original[:, :, 2:3]
        eta_p = particles_original[:, :, 3:4]
        phi_p = particles_original[:, :, 4:5]

        # Create mask for operations ( N, 18, 1) for broadcasting
        actual_particle_mask_expanded = (pT_p > self.eps)

        # log(E)
        norm_log_E_p = torch.zeros_like(E_p, device=device)
        if actual_particle_mask_expanded.any(): # Avoid operations on empty tensors if all are padded
            log_E_p_active = torch.log(E_p[actual_particle_mask_expanded] + self.eps)
            norm_log_E_p[actual_particle_mask_expanded] = (log_E_p_active - self.stats_['particle_logE_mean']) / (self.stats_['particle_logE_std'] + self.eps_std)

        # log(pT)
        norm_log_pT_p = torch.zeros_like(pT_p, device=device)
        if actual_particle_mask_expanded.any():
            log_pT_p_active = torch.log(pT_p[actual_particle_mask_expanded] + self.eps)
            norm_log_pT_p[actual_particle_mask_expanded] = (log_pT_p_active - self.stats_['particle_logpT_mean']) / (self.stats_['particle_logpT_std'] + self.eps_std)
        
        # eta
        norm_eta_p = torch.zeros_like(eta_p, device=device)
        if actual_particle_mask_expanded.any():
            norm_eta_p[actual_particle_mask_expanded] = (eta_p[actual_particle_mask_expanded] - self.stats_['particle_eta_mean']) / (self.stats_['particle_eta_std'] + self.eps_std)

        # phi
        cos_phi_p = torch.cos(phi_p)
        sin_phi_p = torch.sin(phi_p)
        # For padded particles, phi might be 0. cos(0)=1, sin(0)=0. This is fine. obj_type=0 will mask them in model.

        processed_particles = torch.cat([obj_types, norm_log_E_p, norm_log_pT_p, norm_eta_p, cos_phi_p, sin_phi_p], dim=2) # (N, 18, 6)
        flat_processed_particles = processed_particles.reshape(N, -1) # (N, 18 * 6)

        final_X = torch.cat([processed_global_feats, flat_processed_particles], dim=1) # (N, 3 + 18*6) = (N, 111)
        return final_X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class SlotAttentionClassifier(nn.Module):
    def __init__(self, input_dim_flat, 
                 num_obj_types=20, obj_type_embedding_dim=10, # Max obj_type_id is 19
                 d_model=64, num_slots=6, num_slot_attn_iter=2, 
                 nhead_transformer=4, num_transformer_layers=1, dim_feedforward=128,
                 dropout_val=0.1):
        super().__init__()
        self.num_total_features_flat = input_dim_flat # Should be 111
        self.num_global_features_proc = 3
        self.num_particle_features_proc = 6 # type, E, pT, eta, cosPhi, sinPhi
        self.max_particles = (input_dim_flat - self.num_global_features_proc) // self.num_particle_features_proc # Should be 18

        self.obj_type_embedding = nn.Embedding(num_obj_types, obj_type_embedding_dim, padding_idx=0) # Specify padding_idx if 0 is padding

        particle_feat_dim_after_emb = obj_type_embedding_dim + (self.num_particle_features_proc - 1) # -1 for obj_type itself
        self.particle_feature_projector = nn.Linear(particle_feat_dim_after_emb, d_model)

        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, d_model)) 
        
        self.norm_particles_in = nn.LayerNorm(d_model)
        self.norm_slots_in = nn.LayerNorm(d_model) 
        
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.to_q = nn.Linear(d_model, d_model, bias=False) 

        self.gru = nn.GRUCell(d_model, d_model) 
        self.norm_slots_gru_out = nn.LayerNorm(d_model) 

        if num_transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead_transformer, dim_feedforward, dropout_val, batch_first=True, activation='gelu')
            self.slot_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_transformer_layers)
        else:
            self.slot_transformer_encoder = nn.Identity()
        
        self.global_feature_projector = nn.Linear(self.num_global_features_proc, d_model // 2)

        self.mlp_hidden_dim = d_model 
        self.fc = nn.Sequential(
            nn.Linear(d_model + d_model // 2, self.mlp_hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Linear(self.mlp_hidden_dim // 2, 1)
        )
        self.d_model = d_model
        self.num_slots = num_slots
        self.num_slot_attn_iter = num_slot_attn_iter

    def forward(self, x_flat):
        N = x_flat.shape[0]
        
        global_features_proc = x_flat[:, :self.num_global_features_proc]
        particle_features_proc_flat = x_flat[:, self.num_global_features_proc:]
        particle_features_proc = particle_features_proc_flat.reshape(N, self.max_particles, self.num_particle_features_proc)

        obj_types_int = particle_features_proc[:, :, 0].long()
        continuous_particle_features = particle_features_proc[:, :, 1:]

        particle_padding_mask = (obj_types_int == 0) 

        embedded_obj_types = self.obj_type_embedding(obj_types_int)
        raw_particle_embeddings = torch.cat([embedded_obj_types, continuous_particle_features], dim=2)
        
        x_particles = self.particle_feature_projector(raw_particle_embeddings) 

        slots = self.slots_mu.expand(N, -1, -1).clone() 
        
        particles_norm = self.norm_particles_in(x_particles)
        k_particles = self.to_k(particles_norm) 
        v_particles = self.to_v(particles_norm) 

        for _ in range(self.num_slot_attn_iter):
            slots_prev = slots
            slots_norm = self.norm_slots_in(slots)
            q_slots = self.to_q(slots_norm) 

            dots = torch.matmul(q_slots, k_particles.transpose(1, 2)) / (self.d_model**0.5) 
            dots.masked_fill_(particle_padding_mask.unsqueeze(1), float('-inf')) 
            attn_weights = dots.softmax(dim=-1) 

            updates = torch.matmul(attn_weights, v_particles) 
            
            slots = self.gru(
                updates.reshape(-1, self.d_model),
                slots_prev.reshape(-1, self.d_model)
            )
            slots = self.norm_slots_gru_out(slots.reshape(N, self.num_slots, self.d_model))

        slot_representations = self.slot_transformer_encoder(slots) 
        
        aggregated_slots = slot_representations.mean(dim=1) 

        processed_global_features = self.global_feature_projector(global_features_proc) 

        combined_representation = torch.cat([aggregated_slots, processed_global_features], dim=1)
        
        logits = self.fc(combined_representation)
        return logits.squeeze(-1)

def make_model(input_dim: int):
    model = SlotAttentionClassifier(input_dim_flat=input_dim,
                                    num_obj_types=20, obj_type_embedding_dim=8,
                                    d_model=64, num_slots=6, num_slot_attn_iter=2,
                                    nhead_transformer=4, num_transformer_layers=1, dim_feedforward=128,
                                    dropout_val=0.1)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 10
BATCH_SIZE = 512 
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SCHEDULER_STEP_SIZE = 3 
SCHEDULER_GAMMA = 0.5

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predicted = torch.sigmoid(outputs) > 0.5
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                
                predicted = torch.sigmoid(outputs) > 0.5
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # Example print statement (optional, can be commented out)
        # print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
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

