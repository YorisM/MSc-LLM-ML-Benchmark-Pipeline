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
    def __init__(self, obj_idx, e_idx, pt_idx, eta_idx, phi_idx, mask_idx, means, stds, nobj):
        super().__init__()
        self.register_buffer('means', means)
        self.register_buffer('stds', stds)
        self.register_buffer('mask_idx', mask_idx)
        self.obj_idx = obj_idx
        self.e_idx = e_idx
        self.pt_idx = pt_idx
        self.eta_idx = eta_idx
        self.phi_idx = phi_idx
        self.nobj = nobj

    def forward(self, x):
        # x: (bsz, 105)
        xnorm = (x - self.means) / self.stds
        bsz = x.shape[0]
        out = xnorm.clone()

        # Compute augmented features for each slot, and add after each particle
        # (E, pT, eta, phi) per object; feature augmentation for each object
        obj_feats = []
        for i in range(self.nobj):
            o_idx = self.obj_idx + 5*i
            e_idxi = o_idx + self.e_idx
            pt_idxi = o_idx + self.pt_idx
            eta_idxi = o_idx + self.eta_idx
            phi_idxi = o_idx + self.phi_idx
            # (bsz,)
            E   = x[:, e_idxi]
            pT  = x[:, pt_idxi]
            eta = x[:, eta_idxi]
            phi = x[:, phi_idxi]
            # Compose 4-vector (pt*cos(phi), pt*sin(phi), pt*sinh(eta), E)
            px = pT * torch.cos(phi)
            py = pT * torch.sin(phi)
            pz = pT * torch.sinh(eta)
            # m2 = E^2 - px^2 - py^2 - pz^2, mass:=0 if negative
            m2 = E**2 - px**2 - py**2 - pz**2
            mass = torch.sqrt(torch.clamp(m2, min=0))
            # R=\sqrt(eta^2+phi^2)
            R = torch.sqrt(eta**2 + phi**2)
            # Append as new features: mass, R
            obj_feats.append(torch.stack([mass, R], -1)) #(bsz,2)
        # Concatenate the new features for all objects: (bsz, nobj*2)
        feat_ext = torch.cat(obj_feats, dim=1)
        # Concatenate new features after last column
        out = torch.cat([out, feat_ext], dim=1)
        return out

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=256):
    # Indices
    # col0=E_T_miss (0), col1=phi_Etmiss (1),
    # then blocks of (obj_idx, E, pT, eta, phi) for obj1...obj21, so 5*21=105-2=103 unused in last slot(s)
    nobj = (X_train.shape[1]-2)//5
    obj_idx=2; e_idx=1; pt_idx=2; eta_idx=3; phi_idx=4
    # Compute mask: where E==0 and pT==0 and |eta|==0 and |phi|==0 for objects, set to mask=1 else 0
    mask = torch.zeros_like(X_train)
    for i in range(nobj):
        idx_base = obj_idx + 5*i
        e_col = idx_base + e_idx
        pt_col= idx_base + pt_idx
        eta_col= idx_base + eta_idx
        phi_col= idx_base + phi_idx
        is_mask = (
            (X_train[:, e_col] == 0.) &
            (X_train[:, pt_col] == 0.) &
            (X_train[:, eta_col].abs() == 0.) &
            (X_train[:, phi_col] == 0.)
        )
        mask[:, eta_col] = is_mask.float()  # mark mask at eta column (arbitrary slot)
    # Standardization over real/unmasked elements
    valid = 1-mask
    means = (X_train*valid).sum(0) / torch.clamp(valid.sum(0),min=1)
    stds = torch.sqrt(((X_train-means)**2 * valid).sum(0) / torch.clamp(valid.sum(0),min=1))
    stds[torch.isnan(stds)] = 1.
    stds[stds==0] = 1.
    means = means.float()
    stds = stds.float()
    mask_idx=(obj_idx+3,)

    preproc = PreprocessModule(obj_idx, e_idx, pt_idx, eta_idx, phi_idx, mask_idx, means, stds, nobj)

    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)

    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p,   Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    return train_loader, val_loader, preproc

# ----- FREE SECTION: Slot Attention -----
# Implemented as a torch native nn.Module block supporting batching
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.scale = dim ** -0.5
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))    # Learnable mu
        self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, dim)) + 1e-4) # Learnable sigma
        self.project_q = nn.Linear(dim, dim, bias=False)
        self.project_k = nn.Linear(dim, dim, bias=False)
        self.project_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim)
        )
        self.norm_in    = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, mask=None):
        # inputs: (bsz, n_obj, dim)
        bsz, n_obj, dim = inputs.shape
        # Slot initialization
        mu    = self.slots_mu.expand(bsz, self.num_slots, -1)
        sigma = self.slots_sigma.expand(bsz, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu, device=inputs.device)  # (bsz, n_slots, dim)
        inputs = self.norm_in(inputs)
        for _ in range(self.iters):
            slots_prev = slots
            q = self.project_q(self.norm_slots(slots))      # (bsz, n_slots, dim)
            k = self.project_k(inputs)                      # (bsz, n_obj, dim)
            v = self.project_v(inputs)                      # (bsz, n_obj, dim)
            attn_logits = torch.einsum('bnd,bmd->bnm', q, k) * self.scale  # bsz, n_slots, n_obj
            if mask is not None:
                attn_logits = attn_logits.masked_fill(mask.unsqueeze(1).bool(), float('-inf'))
            attn = attn_logits.softmax(-1)   # (bsz, n_slots, n_obj)
            updates = torch.einsum('bnm,bmd->bnd', attn, v)
            # GRU-style slot update
            slots = self.gru(
                updates.reshape(-1, dim),
                slots_prev.reshape(-1, dim)
            ).reshape(bsz, self.num_slots, dim)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots, attn

# ----- FREE SECTION: Particle Transformer Layer -----
class ParticleTransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_hidden):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model)
        )
    def forward(self, x, mask=None):
        # x: (bsz, n_obj, d_model)
        residual = x
        x = self.ln1(x)
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = attn_output + residual
        residual2 = x
        x = self.ln2(x)
        x = self.ff(x) + residual2
        return x

# ----- FREE SECTION: Classifier -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Format input: [E_Tmiss, phi_ETmiss, 21*(5 fields), 21*aug] -> total input_dim
        # Names
        self.nobj = 21
        self.input_dim = input_dim
        # Arrange: the input vector is [E_Tmiss, phi_Etmiss, obj1_obj, obj1_E, obj1_pT, ...]

        # Construct input split/embedding
        # We'll process per-particle records: extract nobj blocks + global features
        self.obj_offset = 2
        self.fields_per_obj = 5
        # After preprocessing, we appened 2 features per object (mass,R): new_dim = input_dim+2*nobj
        obj_feature_dim = self.fields_per_obj + 2 # E, pT, eta, phi, objidx, mass, R
        self.global_dim = 2  # E_Tmiss, phi_Etmiss

        # Embeddings
        # Per-object initial MLP
        self.particle_embedding = nn.Sequential(
            nn.Linear(obj_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        # Positional encoding: embed objidx
        self.pos_emb = nn.Embedding(self.nobj, 8)
        # Projected particle dim
        self.d_model = 40
        # Combine embedding and pos into (nobj, d_model)
        self.comb_proj = nn.Linear(32+8, self.d_model)

        # Particle Transformer
        self.tr_block1 = ParticleTransformerBlock(d_model=self.d_model, n_head=4, d_hidden=64)
        self.tr_block2 = ParticleTransformerBlock(d_model=self.d_model, n_head=4, d_hidden=64)

        # Slot attention to group for 4-top structure: use NUM_SLOTS=4 (for 4 top quarks)
        self.slot_attention = SlotAttention(num_slots=4, dim=self.d_model, iters=3, hidden_dim=64)

        # Output heads: aggregate slot outputs with attention to global features
        self.global_head = nn.Sequential(
            nn.Linear(self.global_dim, 12), nn.ReLU(), nn.Linear(12, 12)
        )

        # Fuse slot-attended output and global MET features
        self.final_head = nn.Sequential(
            nn.Linear(self.d_model*4+12, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: [bsz, 105 + nobj*2]
        bsz = x.shape[0]
        nobj = self.nobj
        obj_features = []
        # Compose mask for zero-padded objects
        # Objects in input: each 5 columns (objtype, E, pT, eta, phi); new mass/R appended at end of sample
        ogcols = self.obj_offset + nobj*self.fields_per_obj
        for i in range(nobj):
            off = self.obj_offset + i*self.fields_per_obj
            base_feats = x[:, off:off+self.fields_per_obj]  # (bsz,5)
            # mass, R are appended at the end; index = 105 + 2*i, 105+2*i+1
            mass = x[:, ogcols + 2*i].unsqueeze(1)
            radius = x[:, ogcols + 2*i+1].unsqueeze(1)
            feats = torch.cat([base_feats, mass, radius], dim=1) #(bsz,7)
            obj_features.append(feats)
        particles = torch.stack(obj_features, dim=1) #(bsz, nobj, 7)
        # Mask: padded object if (E==pT==0 & abs(eta)==0 & abs(phi)==0)
        mask = (particles[...,1]==0) & (particles[...,2]==0) & (particles[...,3].abs()==0) & (particles[...,4].abs()==0)
        # Particle embedding
        xpart = self.particle_embedding(particles)
        # Positional encoding (obj idx)
        idx = torch.arange(nobj, device=x.device).unsqueeze(0).repeat(bsz,1) #(bsz,nobj)
        xpos = self.pos_emb(idx)
        xcat = torch.cat([xpart, xpos],-1)
        xpt = self.comb_proj(xcat) #(bsz, nobj, d_model)
        # Particle Transformer blocks
        xpt = self.tr_block1(xpt, mask)
        xpt = self.tr_block2(xpt, mask)
        # Slot Attention: groups objects into 4 tops
        slots, slot_attn = self.slot_attention(xpt, mask)
        # slots: (bsz, 4, d_model)
        # Aggregate over slots (flatten)
        slots_flat = slots.view(bsz,-1)
        # Global features:
        global_feats = x[:,0:2] #(bsz,2)
        global_emb = self.global_head(global_feats)
        finalvec = torch.cat([slots_flat, global_emb],dim=1)
        logits = self.final_head(finalvec).squeeze(-1)
        return logits

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    lossfn = nn.BCEWithLogitsLoss()
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    best_auc = 0
    for epoch in range(epochs):
        model.train()
        tr_loss_sum = 0
        tr_preds = []
        tr_targets = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            yfloat = yb.float()
            loss = lossfn(logits, yfloat)
            loss.backward()
            optimizer.step()
            tr_loss_sum += loss.item() * xb.shape[0]
            prob = torch.sigmoid(logits)
            tr_preds.append(prob.detach().cpu())
            tr_targets.append(yb.cpu())
        tr_pred_all = torch.cat(tr_preds)
        tr_target_all = torch.cat(tr_targets)
        tr_loss_mean = tr_loss_sum/len(train_loader.dataset)
        tr_acc = accuracy_score(tr_target_all, (tr_pred_all>0.5).long())
        training_loss.append(tr_loss_mean)
        training_acc.append(tr_acc)
        # =============== VALIDATION ===============
        model.eval()
        val_loss_sum = 0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                yfloat = yb.float()
                loss = lossfn(logits, yfloat)
                val_loss_sum += loss.item() * xb.shape[0]
                prob = torch.sigmoid(logits)
                val_preds.append(prob.detach().cpu())
                val_targets.append(yb.cpu())
        val_pred_all = torch.cat(val_preds)
        val_target_all = torch.cat(val_targets)
        val_loss_mean = val_loss_sum/len(val_loader.dataset)
        val_acc = accuracy_score(val_target_all, (val_pred_all>0.5).long())
        validation_loss.append(val_loss_mean)
        validation_acc.append(val_acc)
        # Compute ROC-AUC if on last epoch
        if epoch == epochs-1:
            auc_val = roc_auc_score(val_target_all, val_pred_all)
            if auc_val>best_auc:
                best_auc = auc_val
        print(f'Epoch {epoch+1}/{epochs}: Loss: {tr_loss_mean:.4f}/{val_loss_mean:.4f} Acc: {tr_acc:.4f}/{val_acc:.4f}  [AUC:{best_auc:.4f}]')
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
    X_train, Y_train, X_val, Y_val = load_data()
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val)
    sample_X, _, = next(iter(train_loader))
    model = Classifier(input_dim=sample_X.shape[1])
    epochs = 1 if dryrun else 10
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs)
    if not dryrun:
        base       = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        os.makedirs(script_dir, exist_ok=True)
        model_path = os.path.join(script_dir, f"{base}_model.pth")
        torch.save(trained_model.state_dict(), model_path)
        scripted_path = os.path.join(script_dir, f"{base}_scripted.pt")
        torch.jit.script(trained_model).save(scripted_path)
        scripted_preproc = torch.jit.script(preproc)
        scripted_preproc.save(os.path.join(script_dir, f"{base}_preproc.pt"))
        plot_and_save(training_loss, validation_loss, f"Loss - {base}", os.path.join(script_dir, f"{base}_loss.png"))
        plot_and_save(training_acc, validation_acc, f"Accuracy - {base}", os.path.join(script_dir, f"{base}_accuracy.png"))

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)