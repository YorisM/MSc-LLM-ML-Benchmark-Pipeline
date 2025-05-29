
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
import math
from torch.nn import functional as F


# Constants for preprocessor and model based on analysis
NUM_MAX_OBJECTS = 18
NUM_RAW_OBJ_FEATURES = 5 # obj_id, E, pT, eta, phi
# Engineered particle features: obj_id, log_E, log_pT, eta, phi, px, py, pz, m_inv
# Plus one raw_pT for masking
NUM_ENGINEERED_PARTICLE_FEATURES_NET = 9 # These go into network computations
NUM_PARTICLE_FEATURES_PREPROC = NUM_ENGINEERED_PARTICLE_FEATURES_NET + 1 # Includes raw_pT for mask
NUM_GLOBAL_FEATURES = 2 # E_T_miss, phi_E_t_miss

# Derived input dimension for the flattened tensor
TOTAL_FLAT_INPUT_DIM = NUM_MAX_OBJECTS * NUM_PARTICLE_FEATURES_PREPROC + NUM_GLOBAL_FEATURES # 18*10 + 2 = 182

# Model hyperparameters (chosen to be modest for CPU and memory constraints)
D_MODEL = 64  # Dimension for embeddings and transformer
N_HEAD_TX = 4      # Heads for transformer encoder on slots
NUM_SLOTS = 6      # Number of slots for Slot Attention
SA_ITERATIONS = 3  # Iterations for Slot Attention refinement
TX_ENCODER_LAYERS = 2 # Number of layers in Transformer encoder for slots
TX_DIM_FEEDFORWARD = D_MODEL * 4 # Feedforward dimension in Transformer
DROPOUT_RATE = 0.1

# Training Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 128

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.particle_means = None
        self.particle_stds = None
        self.global_means = None
        self.global_stds = None
        self.eps = 1e-7 # For stable division
        
    def _engineer_features(self, X):
        # X shape: (N, 92)
        X_global_raw = X[:, :NUM_GLOBAL_FEATURES] # (N, 2)
        X_particles_raw = X[:, NUM_GLOBAL_FEATURES:].reshape(
            -1, NUM_MAX_OBJECTS, NUM_RAW_OBJ_FEATURES
        ) # (N, 18, 5)

        obj_ids = X_particles_raw[..., 0]
        E = X_particles_raw[..., 1]
        pT = X_particles_raw[..., 2] # Original pT for feature engineering
        eta = X_particles_raw[..., 3]
        phi = X_particles_raw[..., 4]

        log_E = torch.log1p(torch.relu(E))
        log_pT = torch.log1p(torch.relu(pT))
        log_E_T_miss = torch.log1p(torch.relu(X_global_raw[:, 0]))

        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)

        m2 = E.square() - px.square() - py.square() - pz.square()
        m_inv = torch.sqrt(torch.relu(m2))

        processed_particles_net_features = torch.stack([
            obj_ids, log_E, log_pT, eta, phi, px, py, pz, m_inv
        ], dim=-1) # (N, 18, 9)
        
        mask_pT_unscaled = X_particles_raw[..., 2].clone() # (N, 18), for mask generation
        
        final_particle_block = torch.cat(
            (processed_particles_net_features, mask_pT_unscaled.unsqueeze(-1)), dim=-1
        ) # (N, 18, 10)

        phi_E_t_miss = X_global_raw[:, 1]
        processed_global = torch.stack(
            [log_E_T_miss, phi_E_t_miss], dim=-1
        ) # (N, 2)
        
        return final_particle_block, processed_global

    def fit(self, X, y=None):
        particle_features_full, global_features = self._engineer_features(X)
        particles_to_scale = particle_features_full[..., :NUM_ENGINEERED_PARTICLE_FEATURES_NET] # (N, 18, 9)
        
        actual_particle_mask = (particle_features_full[..., NUM_ENGINEERED_PARTICLE_FEATURES_NET] > 0) # (N, 18)
        masked_particles = particles_to_scale[actual_particle_mask] # (num_actual_particles, 9)
        
        if masked_particles.shape[0] == 0:
            self.particle_means = torch.zeros(NUM_ENGINEERED_PARTICLE_FEATURES_NET, device=X.device)
            self.particle_stds = torch.ones(NUM_ENGINEERED_PARTICLE_FEATURES_NET, device=X.device)
        else:
            self.particle_means = torch.mean(masked_particles, dim=0)
            self.particle_stds = torch.std(masked_particles, dim=0)

        self.global_means = torch.mean(global_features, dim=0)
        self.global_stds = torch.std(global_features, dim=0)
        
        self.particle_stds[self.particle_stds < self.eps] = 1.0
        self.global_stds[self.global_stds < self.eps] = 1.0
        
        return self

    def transform(self, X):
        if self.particle_means is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")

        particle_features_full, global_features = self._engineer_features(X)
        
        particles_to_scale = particle_features_full[..., :NUM_ENGINEERED_PARTICLE_FEATURES_NET]
        scaled_particles_block = (particles_to_scale - self.particle_means.to(X.device)) / self.particle_stds.to(X.device)

        raw_pT_for_mask = particle_features_full[..., NUM_ENGINEERED_PARTICLE_FEATURES_NET].unsqueeze(-1)
        
        final_particles_processed = torch.cat((scaled_particles_block, raw_pT_for_mask), dim=-1)

        scaled_global = (global_features - self.global_means.to(X.device)) / self.global_stds.to(X.device)
        
        flat_particles = final_particles_processed.reshape(X.shape[0], -1)
        output = torch.cat((flat_particles, scaled_global), dim=1)
        return output

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim_slots, dim_input, iters, eps=1e-8, hidden_dim_mlp=128):
        super().__init__()
        self.num_slots = num_slots
        self.dim_slots = dim_slots
        self.dim_input = dim_input
        self.iters = iters
        self.eps = eps
        self.scale = dim_slots ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim_slots))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim_slots))

        self.to_k = nn.Linear(dim_input, dim_slots)
        self.to_v = nn.Linear(dim_input, dim_slots)

        self.gru = nn.GRUCell(dim_slots, dim_slots)

        self.norm_input  = nn.LayerNorm(dim_input)
        self.norm_slots  = nn.LayerNorm(dim_slots)

    def forward(self, inputs, attention_mask=None):
        batch_size, num_inputs, _ = inputs.shape

        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = torch.exp(self.slots_log_sigma).expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            q_slots = slots
            attn_logits = torch.einsum('bnd,bmd->bnm', q_slots, k) * self.scale

            if attention_mask is not None:
                attn_logits.masked_fill_(~attention_mask.unsqueeze(1).bool(), -float('inf'))

            attn = F.softmax(attn_logits, dim=-1)
            attn = attn + self.eps / num_inputs 
            attn = attn / torch.sum(attn, dim=-1, keepdim=True)

            updates = torch.einsum('bnm,bmd->bnd', attn, v)

            slots = self.gru(
                updates.reshape(-1, self.dim_slots),
                slots_prev.reshape(-1, self.dim_slots)
            )
            slots = slots.reshape(batch_size, self.num_slots, self.dim_slots)
        return slots

class TopTaggerTransformer(nn.Module):
    def __init__(self, input_dim_flat,
                 num_particle_features_preproc = NUM_PARTICLE_FEATURES_PREPROC,
                 num_particle_features_net = NUM_ENGINEERED_PARTICLE_FEATURES_NET,
                 num_global_features = NUM_GLOBAL_FEATURES,
                 d_model=D_MODEL, 
                 nhead_tx=N_HEAD_TX, 
                 num_encoder_layers=TX_ENCODER_LAYERS,
                 dim_feedforward_tx=TX_DIM_FEEDFORWARD, 
                 dropout=DROPOUT_RATE,
                 num_slots=NUM_SLOTS, 
                 sa_iters=SA_ITERATIONS):
        super().__init__()
        
        self.num_particle_features_preproc = num_particle_features_preproc
        self.num_particle_features_net = num_particle_features_net
        self.num_global_features = num_global_features
        self.d_model = d_model

        self.particle_embed = nn.Linear(self.num_particle_features_net, d_model)
        self.particle_pos_encoder = nn.Embedding(NUM_MAX_OBJECTS, d_model)
        self.global_embed = nn.Linear(self.num_global_features, d_model)

        self.slot_attention = SlotAttention(num_slots, d_model, d_model, sa_iters, hidden_dim_mlp=d_model*2)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead_tx, dim_feedforward_tx, dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x_flat):
        batch_size = x_flat.shape[0]

        particle_features_idx_end = NUM_MAX_OBJECTS * self.num_particle_features_preproc
        particles_data_flat = x_flat[:, :particle_features_idx_end]
        global_features_data = x_flat[:, particle_features_idx_end:]

        particles_full = particles_data_flat.view(batch_size, NUM_MAX_OBJECTS, self.num_particle_features_preproc)
        particles_net_input = particles_full[..., :self.num_particle_features_net]
        raw_pT_for_mask = particles_full[..., self.num_particle_features_net]
        
        particle_attention_mask = (raw_pT_for_mask > 0)

        particle_embeddings = self.particle_embed(particles_net_input)
        particle_indices = torch.arange(NUM_MAX_OBJECTS, device=x_flat.device).unsqueeze(0).expand(batch_size, -1)
        pos_enc = self.particle_pos_encoder(particle_indices)
        particle_embeddings = particle_embeddings + pos_enc
        # No activation after pos_enc typically, but can add if desired or part of a norm+add block

        slots = self.slot_attention(particle_embeddings, attention_mask=particle_attention_mask)
        transformer_output = self.transformer_encoder(slots) # Slot mask not needed if all slots are meaningful

        aggregated_slots = torch.mean(transformer_output, dim=1)
        global_embedded = self.global_embed(global_features_data)
        # Activation for global_embedded can be added if useful e.g. F.relu(global_embedded)
        
        combined_representation = torch.cat((aggregated_slots, global_embedded), dim=1)
        logits = self.classifier(combined_representation)
        return logits

# 2. ---------- MODEL DEFINITION ----------
def make_model(input_dim: int):
    assert input_dim == TOTAL_FLAT_INPUT_DIM, \
        f"Input dimension mismatch. Expected {TOTAL_FLAT_INPUT_DIM}, got {input_dim}"
    
    model = TopTaggerTransformer(input_dim_flat=input_dim)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 30

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch_idx in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5)
            total_train += labels.size(0)
            correct_train += (predicted == labels.byte()).sum().item()
        
        epoch_loss_train = running_loss / len(train_loader.dataset)
        epoch_acc_train = correct_train / total_train
        train_losses.append(epoch_loss_train)
        train_accs.append(epoch_acc_train)

        model.eval()
        running_loss_val = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs_val, labels_val in val_loader:
                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device).float().unsqueeze(1)
                outputs_val = model(inputs_val)
                loss_val = criterion(outputs_val, labels_val)
                
                running_loss_val += loss_val.item() * inputs_val.size(0)
                predicted_val = (torch.sigmoid(outputs_val) > 0.5)
                total_val += labels_val.size(0)
                correct_val += (predicted_val == labels_val.byte()).sum().item()

        epoch_loss_val = running_loss_val / len(val_loader.dataset)
        epoch_acc_val = correct_val / total_val
        val_losses.append(epoch_loss_val)
        val_accs.append(epoch_acc_val)
        
        # Example print, can be removed or conditional based on a verbose flag if needed.
        # print(f"Epoch {epoch_idx+1}/{epochs} - "
        #       f"Train Loss: {epoch_loss_train:.4f}, Acc: {epoch_acc_train:.4f} - "
        #       f"Val Loss: {epoch_loss_val:.4f}, Acc: {epoch_acc_val:.4f} - "
        #       f"LR: {scheduler.get_last_lr()[0]:.2e}")

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

