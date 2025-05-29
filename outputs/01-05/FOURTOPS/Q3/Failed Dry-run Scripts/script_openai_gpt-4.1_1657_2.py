# ----- FIXED SECTION: Import Libraries -----
import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

# ----- FIXED SECTION: Data Loading -----
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


# ----- FREE SECTION: Data Preprocessing -----
class PreprocessModule(torch.nn.Module):
    def __init__(self, obj_start, obj_stride, n_slots, obj_mask, means, stds):
        super().__init__()
        self.obj_start = obj_start
        self.obj_stride = obj_stride
        self.n_slots = n_slots
        self.register_buffer('obj_mask', obj_mask)  # [n_objects] mask, 1 for real obj, 0 for padding
        self.register_buffer('means', means)
        self.register_buffer('stds', stds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        # Normalize
        x = (x - self.means) / (self.stds + 1e-6)
        return x

# Helper function: Identify objects in batch (mask nonzero rows)
def build_object_mask(X):
    # Each event: [F=105], structure: E_T_miss (0), phi_ETmiss (1), then [obj_id, E, pt, eta, phi]*n (102 left)
    # Step size 5, first obj at 2, so positions 2,7,12,...
    n_obj = (X.shape[1] - 2)//5
    batch_obj_ids = X[:, 2::5]  # shape [B, n_obj]
    mask = (batch_obj_ids > 0).float()  # 1 if real, 0 if padded
    return mask

# Physics feature extraction: for each particle
# - Reconstruct per-particle rapidity, px, py, pz, mass (where possible)
# - For each event: number of b-jets, leptons, etc.
# Particle type encoding (obj_id):
# According to HEP convention, let's one-hot encode: [jet, b-jet, lepton+, lepton-, photon, others]
# But here, let's just use obj_id as 'type', and calculate per-event sums.

def augment_features(x):
    # x: [B, 105]
    B = x.shape[0]
    device = x.device
    # Original features: [E_T_miss, phi_ETmiss, obj_1, E1, pt1, eta1, phi1, obj_2,..]
    # Nobj = 102//5 = 20
    n_obj = (x.shape[1] - 2)//5
    E_T_miss = x[:, 0:1]  # (B,1)
    phi_Et = x[:, 1:2]    # (B,1)
    # Stack particle properties
    obj_id = x[:, 2::5]   # (B, n_obj)
    E      = x[:, 3::5]
    pt     = x[:, 4::5]
    eta    = x[:, 5::5]
    phi    = x[:, 6::5]

    mask = (obj_id > 0).float()  # (B, n_obj)

    # Extract auxiliary features
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    # mass^2 = E^2 - (pt^2 + pz^2), but set to 0 if inf or nan.
    m2 = E**2 - (pt**2 + pz**2)
    m2 = torch.where(m2>=0, m2, torch.zeros_like(m2))
    m = torch.sqrt(m2+1e-4)  # (B, n_obj)

    # Per-event counts
    n_particles = mask.sum(dim=1, keepdim=True)  # total number of reco'ed objects
    # Physics-motivated: top mass is ~173 GeV, W ~80 GeV
    # Compute all possible trijets in event (exclude padded objects)
    # Only take up to first 10 objects for combinatorics (speed)
    n_comb = 10
    obj_mask = mask[:, :n_comb]
    E_t = E[:, :n_comb]
    px_t = px[:, :n_comb]
    py_t = py[:, :n_comb]
    pz_t = pz[:, :n_comb]
    # All distinct trijet combinations
    idx = torch.combinations(torch.arange(n_comb, device=device), r=3)  # [N_c, 3]
    def trijet_mass(E,p_x,p_y,p_z):
        sum_E  = E[:, idx[:,0]]+E[:, idx[:,1]]+E[:, idx[:,2]]
        sum_px = p_x[:, idx[:,0]]+p_x[:, idx[:,1]]+p_x[:, idx[:,2]]
        sum_py = p_y[:, idx[:,0]]+p_y[:, idx[:,1]]+p_y[:, idx[:,2]]
        sum_pz = p_z[:, idx[:,0]]+p_z[:, idx[:,1]]+p_z[:, idx[:,2]]
        sum_p2 = sum_px**2 + sum_py**2 + sum_pz**2
        m2 = sum_E**2 - sum_p2
        m2 = torch.clamp(m2, min=0.0)
        m = torch.sqrt(m2+1e-4)
        return m  # shape [B, N_c]
    trijet_masses = trijet_mass(E_t,px_t,py_t,pz_t)
    # for each event: best mass match to top (173 GeV=173000 MeV)
    top_mass = 173000.0
    min_top_mdiff = torch.min(torch.abs(trijet_masses - top_mass), dim=1, keepdim=True)[0]

    # Select best W mass from all dijets
    def dijet_mass(E,px,py,pz):
        # all i<j pairs
        N = E.shape[1]
        idx2 = torch.combinations(torch.arange(N, device=device), r=2)
        sum_E = E[:, idx2[:,0]]+E[:, idx2[:,1]]
        sum_px = px[:, idx2[:,0]]+px[:, idx2[:,1]]
        sum_py = py[:, idx2[:,0]]+py[:, idx2[:,1]]
        sum_pz = pz[:, idx2[:,0]]+pz[:, idx2[:,1]]
        sum_p2 = sum_px**2 + sum_py**2 + sum_pz**2
        m2 = sum_E**2 - sum_p2
        m2 = torch.clamp(m2, min=0.0)
        m = torch.sqrt(m2 + 1e-4)
        return m  # (B, N_pair)
    dijet_masses = dijet_mass(E_t,px_t,py_t,pz_t)
    W_mass = 80380.0
    min_W_mdiff = torch.min(torch.abs(dijet_masses - W_mass), dim=1, keepdim=True)[0]

    # Per-event kinematic sums
    sum_pt = (pt*mask).sum(dim=1, keepdim=True)
    sum_E = (E*mask).sum(dim=1, keepdim=True)

    # stack per-event physics features
    feats = [E_T_miss, phi_Et, n_particles, sum_pt, sum_E, min_top_mdiff, min_W_mdiff]
    event_feats = torch.cat(feats,dim=1)  # (B,7)

    # Per-particle physics features
    # 1-hot obj-id: for known types
    obj_types = (obj_id*mask).clone().int()  # padded zeros stay as 0
    # Discover most frequent types (over full dataset: let's assume [0=padded, 1=jet, 2=b-jet, 3=lep+, 4=lep-, 5=photon,...], otherwise just embed as integer)
    # For this challenge: encode obj_id/20 (for first 20 objects), as a feature.
    type_norm = (obj_id / 20.0)
    # Stack per-particle features
    # [obj_id_norm, pt, eta, phi, px, py, pz, mass, mask]
    particle_feats = [type_norm, pt, eta, phi, px, py, pz, m, mask]
    part_feats = torch.stack(particle_feats,dim=2)  # [B, n_obj, F=9]
    part_feats = part_feats.reshape(B,-1)  # [B, n_obj*9]

    out = torch.cat([event_feats, part_feats],dim=1)  # [B, 7+n_obj*9]
    return out


def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=256):
    # Derive statistics
    with torch.no_grad():
        # Physics-motivated feature augmentation
        X_train_aug = augment_features(X_train)
        X_val_aug   = augment_features(X_val)
        means = X_train_aug.mean(dim=0)
        stds = X_train_aug.std(dim=0) + 1e-6
        # For masking
        n_obj = (X_train.shape[1] - 2)//5
        obj_mask = build_object_mask(X_train).max(dim=0)[0]  # any event has real obj at slot
    obj_start = 2
    obj_stride = 5
    n_slots = n_obj
    # Register constants
    preproc = PreprocessModule(obj_start, obj_stride, n_slots, obj_mask, means, stds)
    X_train_p = preproc(augment_features(X_train))
    X_val_p   = preproc(augment_features(X_val))
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p, Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    return train_loader, val_loader, preproc

# ========== Slot Attention Core =============
class SlotAttention(nn.Module):
    def __init__(self, dim, n_slots, iters=3):
        super().__init__()
        self.n_slots = n_slots
        self.iters = iters
        self.dim = dim
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim*2, bias=False)
        self.slots_mu = nn.Parameter(torch.randn(1, n_slots, dim))
        self.slots_sigma = nn.Parameter(torch.ones(1, n_slots, dim))
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.eps = 1e-8
    def forward(self, x, mask):
        # x: [B, N, D]; mask: [B,N]  1 real, 0 pad
        B, N, D = x.shape
        # Normalize
        x = self.norm_inputs(x)
        # Init slots
        slots = self.slots_mu + self.slots_sigma * torch.randn(B, self.n_slots, D, device=x.device)
        for it in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.to_q(slots_norm)  # [B, n_slots, D]
            k, v = self.to_kv(x).chunk(2, dim=-1)  # [B,N,D],[B,N,D]
            q = q.unsqueeze(2)  # [B,n_slots,1,D]
            k = k.unsqueeze(1)  # [B,1,N,D]
            attn_logits = torch.sum(q * k, dim=-1)/np.sqrt(D)  # [B, n_slots, N]
            # Mask
            mask_pad = (1-mask).unsqueeze(1).expand_as(attn_logits) # mask padded,
            attn_logits = attn_logits.masked_fill(mask_pad.bool(), float('-inf'))
            attn = torch.softmax(attn_logits, dim=1)  # [B, n_slots,N], sum_slots=1
            # Normalize over slots as in slot attention
            attn = attn + self.eps
            attn = attn/(attn.sum(dim=1, keepdim=True) + self.eps)
            updates = torch.einsum('bkn,bnd->bkd', attn, v)   # [B, n_slots, D]
            # Slot GRU
            slots = self.gru(
                updates.reshape(B*self.n_slots, D),
                slots_prev.reshape(B*self.n_slots, D)
            ).reshape(B,self.n_slots,D)
            slots = slots + self.mlp(self.norm_slots(slots))
        return slots, attn  # slot outputs, attention weights [B, n_slots, N]


# ============ Transformer block for per-particle ===============
class ParticleTransformer(nn.Module):
    def __init__(self, in_dim, embed_dim, n_heads, n_blocks, dropout):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, embed_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=embed_dim*2,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_blocks)
        ])
    def forward(self, x, mask):
        # x: [B,N,F_in], mask: [B,N]
        x = self.fc_in(x)  # [B,N,embed_dim]
        for block in self.blocks:
            x = block(x, src_key_padding_mask=(mask==0))
        return x

# ============ Main Classifier Model ============
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # input_dim is of preprocessed feature vector length (as generated in preprocess_data)
        # We need to: split into event-level [7] and per-object [n_obj,9]
        self.n_obj = 20  # as in data structure
        self.part_feat_dim = 9
        self.event_feat_dim = 7
        # Particle encoder
        self.embed_dim = 32
        self.particle_embed = ParticleTransformer(
            in_dim=self.part_feat_dim,
            embed_dim=self.embed_dim,
            n_heads=4,
            n_blocks=2,
            dropout=0.1
        )
        # Slot Attention
        self.n_slots = 4  # Physics: group objects to 4 'tops'
        self.slot_attention = SlotAttention(
            dim=self.embed_dim,
            n_slots=self.n_slots,
            iters=3
        )
        # Event MLP
        self.event_mlp = nn.Sequential(
            nn.Linear(self.event_feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32,32)
        )
        # Classifier head: fuse [slots + event encoding]
        self.head = nn.Sequential(
            nn.Linear(self.n_slots*self.embed_dim + 32, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        # x: [B, D]
        B = x.shape[0]
        # Split features
        event_feats = x[:, :self.event_feat_dim]          # (B,7)
        part_feats  = x[:, self.event_feat_dim:]
        part_feats  = part_feats.reshape(B, self.n_obj, self.part_feat_dim)  # (B,20,9)
        # Create mask: padded objects have part_feats[:,:,8]==0 (last col is mask)
        mask = part_feats[:,:,8]  # (B, n_obj)
        part_input = part_feats[:,:,:8]  # drop mask col for encode
        # Transformer encode per-particle
        part_emb = self.particle_embed(part_input, mask)  # (B,n_obj,embed)
        # Slot Attention: group into n_slots ('tops')
        slots, attn = self.slot_attention(part_emb, mask)
        slots_flat = slots.reshape(B, -1)
        event_emb = self.event_mlp(event_feats)
        x = torch.cat([slots_flat, event_emb], dim=1)
        logits = self.head(x).squeeze(-1)  # [B]
        return logits

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    for ep in range(epochs):
        model.train()
        total, correct, lsum = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            lsum += loss.item()*xb.size(0)
            preds = torch.sigmoid(out)>0.5
            correct += (preds==yb.bool()).sum().item()
            total += xb.size(0)
        training_loss.append(lsum/total)
        training_acc.append(correct/total)
        # Validation
        model.eval()
        with torch.no_grad():
            vtotal, vcorrect, vsum = 0, 0, 0.0
            ytrues = []
            yprobs = []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                out = model(xb)
                vsum += loss_fn(out, yb).item()*xb.size(0)
                probs = torch.sigmoid(out)
                preds = (probs > 0.5)
                yprobs.append(probs.detach().cpu())
                ytrues.append(yb.detach().cpu())
                vcorrect += (preds==yb.bool()).sum().item()
                vtotal += xb.size(0)
            all_probs = torch.cat(yprobs).numpy().flatten()
            all_trues = torch.cat(ytrues).numpy().flatten()
            val_auc = roc_auc_score(all_trues, all_probs)
            validation_loss.append(vsum/vtotal)
            validation_acc.append(val_auc)
        scheduler.step()
        print(f"Epoch {ep+1}/{epochs} Train loss {training_loss[-1]:.4f} acc {training_acc[-1]:.4f}  Val loss {validation_loss[-1]:.4f} valAUC {validation_acc[-1]:.4f}")
    return model, training_loss, validation_loss, training_acc, validation_acc

# ----- FIXED SECTION: Plotting and Saving Outputs -----
def plot_and_save(metric_train, metric_val, metric_name, filename):
    plt.figure()
    plt.plot(metric_train, label=f'Training {metric_name}')
    plt.plot(metric_val, label=f'Validation {metric_name}')
    plt.title(f'{metric_name} per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(filename)
    plt.close()

# ----- FIXED SECTION: Main Function -----
def main(dryrun=False):
    # Data Loading
    X_train, Y_train, X_val, Y_val = load_data()
    # Preprocessing
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val)
    # Model Initialization
    sample_X, _, = next(iter(train_loader))
    model = Classifier(input_dim=sample_X.shape[1])
    # Training
    epochs = 1 if dryrun else 10
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs)
    if not dryrun:
        # determine base name & script directory
        base       = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        os.makedirs(script_dir, exist_ok=True)
        # save model
        model_path = os.path.join(script_dir, f"{base}_model.pth")
        torch.save(trained_model.state_dict(), model_path)
        # save scripted model
        scripted_path = os.path.join(script_dir, f"{base}_scripted.pt")
        torch.jit.script(trained_model).save(scripted_path)
        # save preprocessor
        scripted_preproc = torch.jit.script(preproc)
        scripted_preproc.save(os.path.join(script_dir, f"{base}_preproc.pt"))
        # Plot and Save Metrics
        plot_and_save(training_loss, validation_loss, f"Loss - {base}", os.path.join(script_dir, f"{base}_loss.png"))
        plot_and_save(training_acc, validation_acc, f"Accuracy - {base}", os.path.join(script_dir, f"{base}_accuracy.png"))

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)