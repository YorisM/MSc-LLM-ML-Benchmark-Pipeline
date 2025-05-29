# ----- FIXED SECTION: Import Libraries -----
import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
# <FREE: You may only import python and torch native modules here. NO OTHER MODULES.>

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
    def __init__(self, means, stds, obj_map, mask_idx):
        super().__init__()
        self.register_buffer('means', means)
        self.register_buffer('stds', stds)
        # For object type encoding
        self.obj_map = obj_map  # Dictionary mapping id to index
        # For masking paddings
        self.register_buffer('mask_idx', mask_idx)
        # Saved for vectorized index conversion
        idxs = torch.zeros(20, dtype=torch.long)
        for k, v in obj_map.items():
            if k<idxs.shape[0]:
                idxs[k]=v
        self.register_buffer('id_encs', idxs)  # id->vector index
    def forward(self, x):
        # x: [batch, 105] flat
        batch = x.shape[0]
        # unpack
        Etmiss   = x[:, 0:1]  # [B, 1]
        Phimiss  = x[:, 1:2]

        nobj = (x.shape[1] - 2) // 5
        obj_raw  = x[:, 2:].reshape(batch, nobj, 5)  # [B, nobj, 5]
        obj_id   = obj_raw[..., 0].long()            # [B, nobj]
        obj_e    = obj_raw[..., 1]
        obj_pt   = obj_raw[..., 2]
        obj_eta  = obj_raw[..., 3]
        obj_phi  = obj_raw[..., 4]
        pad_mask = (obj_id==self.mask_idx)           # [B, nobj]

        # Masked where obj_id==mask_idx.
        # Object type onehot: for physical info
        typevec = torch.zeros(batch, nobj, len(self.obj_map), device=x.device)
        for idval, idx in self.obj_map.items():
            typevec[:,:,idx] = (obj_id==idval).float()
            
        # Physics-inspired features:
        #  - Invariant mass w/ Etmiss
        #  - Object E/pT ratios
        #  - Azimuthal delta phi to Etmiss
        delta_phi_miss = (obj_phi - Phimiss).remainder(2.*np.pi)
        delta_phi_miss = torch.where(delta_phi_miss>np.pi, delta_phi_miss-2.*np.pi, delta_phi_miss)
        E_div_pt = torch.where(obj_pt.abs()>1e-6, obj_e/(obj_pt+1e-6), torch.zeros_like(obj_e))
        # For triple/invariant mass clusters: set up for downstream module
        # For now, these are not added at this step; the transformer will learn combinations.

        # Standardize numeric features
        f_obj = torch.stack([obj_e, obj_pt, obj_eta, obj_phi, E_div_pt, delta_phi_miss], dim=-1)
        # shape: [B, nobj, 6]
        mask_float = (~pad_mask).unsqueeze(-1).float()  # [B, nobj, 1]
        f_obj = (f_obj - self.means[None,None,:])/(self.stds[None,None,:]+1e-8) * mask_float

        feat_obj = torch.cat([f_obj, typevec], dim=-1)  # [B, nobj, 6+typenum]

        # Also preprocess Etmiss and Phimiss
        Etmiss_std  = (Etmiss - self.means[-2])/self.stds[-2]
        Phimiss_std = (Phimiss - self.means[-1])/self.stds[-1]
        # Stack 
        out_feat = [feat_obj.reshape(batch,-1), Etmiss_std, Phimiss_std]
        out = torch.cat(out_feat, dim=1)
        return out

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=256):
    # decode object ids in train
    obj_ids = X_train[:,2::5].long()  # shape: [N, 21]
    unique_ids = set(obj_ids.flatten().tolist())
    if 0 in unique_ids:
        unique_ids.discard(0) # 0 for padding
    unique_ids = sorted([v for v in unique_ids if v>0])
    # Map object ids to slot types for onehot
    obj_map = {oid:i for i,oid in enumerate(unique_ids)}   # id -> slot
    ntype = len(obj_map)
    mask_idx = torch.tensor(0)

    # Compute obj features: [E, pT, eta, phi], and physics features
    nobj = (X_train.shape[1]-2)//5
    obj_e_train   = X_train[:,2+1::5]
    obj_pt_train  = X_train[:,2+2::5]
    obj_eta_train = X_train[:,2+3::5]
    obj_phi_train = X_train[:,2+4::5]
    # Only compute normalization on non-padding objects
    mask = X_train[:,2::5]!=0
    # Make E_div_pt and delta_phi_miss for training set
    E_div_pt_train = torch.zeros_like(obj_e_train)
    valid_mask = obj_pt_train.abs()>1e-6
    E_div_pt_train[valid_mask] = obj_e_train[valid_mask]/(obj_pt_train[valid_mask]+1e-6)
    Phimiss_train = X_train[:,1:2]
    delta_phi_train = (obj_phi_train - Phimiss_train)
    delta_phi_train = delta_phi_train.remainder(2.*np.pi)
    delta_phi_train = torch.where(delta_phi_train>np.pi, delta_phi_train-2.*np.pi, delta_phi_train)
    # Gather all features: [E, pt, eta, phi, E_div_pt, delta_phi_miss]
    all_obj_feats = torch.stack([obj_e_train, obj_pt_train, obj_eta_train, obj_phi_train, E_div_pt_train, delta_phi_train], dim=-1)
    # Only for valid objects
    valid_obj_feats = all_obj_feats[mask]
    means = valid_obj_feats.mean(dim=0)
    stds = valid_obj_feats.std(dim=0)
    # Compose means and stds for missing ET
    means_all = torch.cat([means, X_train[:,0:1].mean(dim=0), X_train[:,1:2].mean(dim=0)])
    stds_all = torch.cat([stds, X_train[:,0:1].std(dim=0), X_train[:,1:2].std(dim=0)])

    preproc = PreprocessModule(means_all, stds_all, obj_map, mask_idx)
    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)

    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p,   Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    return train_loader, val_loader, preproc

# ----- FREE SECTION: Slot Attention + Transformer Encoder -----
class SlotAttentionModule(nn.Module):
    def __init__(self, feat_dim, slot_dim, n_slots=4, iters=3):
        super().__init__()
        self.n_slots = n_slots
        self.iters = iters
        self.slot_dim = slot_dim
        # slot init
        self.slots_mu = nn.Parameter(torch.randn(1, n_slots, slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, n_slots, slot_dim))
        layernorm = lambda d: nn.LayerNorm(d, elementwise_affine=True)
        self.norm_in = layernorm(feat_dim)
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_kv = nn.Linear(feat_dim, 2*slot_dim, bias=False)
        self.gru   = nn.GRUCell(slot_dim, slot_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim)
        )
        self.norm_slots  = layernorm(slot_dim)
        self.norm_pre_ff = layernorm(slot_dim)

    def forward(self, x, mask):
        # x: [B, N, D]
        B, N, D = x.shape
        mask = mask.float()
        x_in = self.norm_in(x)
        # slots: [B, n_slots, slot_dim]
        mu = self.slots_mu.expand(B, -1, -1)
        sigma = (self.slots_logsigma.exp()+1e-5).expand(B, -1, -1)
        slots = mu + torch.randn_like(mu)*sigma
        
        for _ in range(self.iters):
            # attend
            slot_q = self.to_q(self.norm_slots(slots)) # [B, n_slots, d]
            kv     = self.to_kv(x_in)                  # [B, N, 2d]
            k,v    = kv.split(self.slot_dim, dim=-1)
            attn_logits = (slot_q @ k.transpose(-2,-1))/np.sqrt(self.slot_dim) # [B, n_slots, N]
            attn_logits = attn_logits.masked_fill(mask[:,None,:]==0, -9e9)
            attn = attn_logits.softmax(dim=-1) + 1e-8 # [B, n_slots, N]
            attn = attn/(attn.sum(dim=-1, keepdim=True)+1e-8)
            updates = attn @ v  # [B, n_slots, d]
            # GRU update (per slot)
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots.reshape(-1, self.slot_dim)
            ).reshape(B, self.n_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots, attn # [B, n_slots, slot_dim], attn [B, n_slots, N]

class ParticleTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.ReLU(),
            nn.Linear(dim*2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x, mask=None):
        x2, _ = self.attn(x, x, x, key_padding_mask=(~mask) if mask is not None else None)
        x = x + x2
        x = self.norm1(x)
        x2 = self.mlp(x)
        x = x + x2
        x = self.norm2(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Infer object count and features
        nobj = 21
        # feature layout: [nobj*(base+phy+type), Etmiss, Phimiss]
        # from preprocess: base(4) + 2phy + typevec (variable)
        # Compute ntype
        # n_feat = nobj*(6+ntype)+2
        # Assume preprocess outputs [B, nobj*(6+ntype)+2]
        # 
        # Determine based on input_dim
        feat_dim = (input_dim-2)//nobj
        ntype = feat_dim - 6
        self.nobj = nobj
        self.feat_dim = feat_dim
        self.ntype = ntype
        # Slot Attention to extract up to 4 groupings
        self.slot_dim = 64
        self.n_slots = 4
        self.slot_attention = SlotAttentionModule(feat_dim, self.slot_dim, n_slots=self.n_slots, iters=3)
        # Particle Transformer for refined per-particle processing
        self.embed_proj = nn.Linear(feat_dim, self.slot_dim)
        self.pt_blocks = nn.ModuleList([
            ParticleTransformerBlock(self.slot_dim, num_heads=4, dropout=0.15) for _ in range(2)
        ])
        # Global pooling, then dense classifier
        self.fc = nn.Sequential(
            nn.Linear(self.n_slots*self.slot_dim+2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        # x: [B, nobj*feat + 2]
        batch = x.shape[0]
        feat_particles = x[:,:self.nobj*self.feat_dim].reshape(batch, self.nobj, self.feat_dim) # [B, nobj, feat_dim]
        Etmiss = x[:,-2:]
        # mask: paddings are all zero features
        mask = (feat_particles.abs().sum(dim=-1)>0)
        # Pass through particle transformer
        xobj = self.embed_proj(feat_particles)
        for block in self.pt_blocks:
            xobj = block(xobj, mask)
        # Slot Attention
        slots, _ = self.slot_attention(xobj, mask)
        # Flatten slot outputs
        slots_flat = slots.reshape(batch, -1)
        # Concatenate Etmiss features
        final_in = torch.cat([slots_flat, Etmiss], dim=-1)
        out = self.fc(final_in)
        return out.squeeze(-1)

# ----- TRAINING LOOP -----
def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    training_loss, validation_loss = [], []
    training_acc, validation_acc   = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss, n, correct = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float()
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            # Metrics
            with torch.no_grad():
                preds = (logits.sigmoid()>0.5).long()
                correct += (preds==yb.long()).sum().item()
            epoch_loss += float(loss.item())*xb.size(0)
            n += xb.size(0)
        training_loss.append(epoch_loss/n)
        training_acc.append(correct/n)

        # Validation
        model.eval()
        val_loss, nval, correct_val = 0.0, 0, 0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).float()
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += float(loss.item())*xb.size(0)
                nval += xb.size(0)
                preds = (logits.sigmoid()>0.5).long()
                correct_val += (preds==yb.long()).sum().item()
                all_logits.append(logits.cpu())
                all_labels.append(yb.cpu())
        validation_loss.append(val_loss/nval)
        validation_acc.append(correct_val/nval)
        # Print AUC for curiosity
        try:
            scores = torch.cat(all_logits).sigmoid().numpy()
            labels = torch.cat(all_labels).numpy()
            auc = roc_auc_score(labels, scores)
        except Exception:
            auc = -1
        print(f"Epoch {epoch+1}: train_loss={training_loss[-1]:.4f} val_loss={validation_loss[-1]:.4f} val_acc={validation_acc[-1]:.4f} val_auc={auc:.4f}")

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

    # Train the model
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