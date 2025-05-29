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
    def __init__(self, obj_mask, feature_means, feature_stds, n_max_objs, obj_id_map):
        super().__init__()
        # Padding mask: [feature_idx], bool, True if it's an object feature
        self.register_buffer('obj_mask', obj_mask)
        self.register_buffer('feature_means', feature_means)
        self.register_buffer('feature_stds', feature_stds)
        self.n_max_objs = n_max_objs
        # (for embedding object IDs)
        self.register_buffer('obj_id_map', obj_id_map)

    def forward(self, x):
        # x: [batch, 105]
        x = (x - self.feature_means) / (self.feature_stds + 1e-8)
        return x

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=512):
    # --- 1. Identify structure: parse X, find object slot indices & event-level feature indices ---
    # By template: [E_T_miss, phi_et_miss, obj_1, E1, pt1, eta1, phi1, obj2, ...]
    # There can be up to N objects, padded to reach 105 columns.

    total_cols = X_train.shape[1]
    n_evt_feat = 2 # E_T_miss, phi_{E_t}_miss
    n_obj_feat = 5 # obj_id, E, pt, eta, phi
    n_max_objs = (total_cols - n_evt_feat)//n_obj_feat
    
    evt_idx = [0, 1]
    obj_offset = n_evt_feat
    obj_idx = [] # list of (start, end) for object slots
    for i in range(n_max_objs):
        start = obj_offset + i*n_obj_feat
        end = start + n_obj_feat
        obj_idx.append((start, end))
    obj_mask = torch.zeros(total_cols, dtype=torch.bool)
    for s, e in obj_idx:
        obj_mask[s:e] = True
    # Gather object_id values in all events
    obj_ids = []
    for s, e in obj_idx:
        ids = X_train[:, s]
        obj_ids.append(ids)
    obj_ids = torch.cat(obj_ids)
    unique_obj_ids = torch.unique(obj_ids[(obj_ids != 0)]) # padding = 0
    n_obj_classes = len(unique_obj_ids)
    obj_id_map = torch.zeros(100, dtype=torch.long)
    for i, v in enumerate(unique_obj_ids):
        obj_id_map[int(v)] = i+1 # padding=0 stays zero
    # --- 2. Feature Engineering ---
    def particle_feature_augment(X):
        # X shape: [N,105]
        batch = X.clone()
        particles = []
        for i, (s, e) in enumerate(obj_idx):
            slot = batch[:, s:e]    # [N, 5]
            particles.append(slot)
        particles = torch.stack(particles, dim=1) # [N, n_max_objs, 5]
        obj_id  = particles[:,:,0] # [N, n_max_objs]
        mask = (obj_id > 0)
        # Create per-object augmented features: mass (will be zero due to 4vec incomplete),
        # charge (based on obj_id, if known), |eta|, pt/E ratio
        E     = particles[:,:,1]
        pt    = particles[:,:,2]
        eta   = particles[:,:,3]
        phi   = particles[:,:,4]
        # Augmentation 1: abs(eta)
        abs_eta = torch.abs(eta)
        # Augmentation 2: pt/E
        pt_over_E = torch.where(E > 0, pt/(E+1e-6), torch.zeros_like(pt))
        # Augmentation 3: object type embedding (categorical)
        # Augmentation 4: deltaR(missET, object)
        evt_metphi = batch[:,1].unsqueeze(1) # [N,1]
        dphi = phi - evt_metphi
        dphi = (dphi + torch.pi) % (2*torch.pi) - torch.pi
        # define (eta, phi) for met as (0, evt_metphi) for rough deltaR
        deta = eta - 0
        deltaR_met = torch.sqrt(deta**2 + dphi**2)
        # Augmentation 5: 1 for existence mask
        exists = mask.float()
        # concatenate all per-object features: [obj_id, E, pt, eta, phi, abs_eta, pt_over_E, deltaR_met, exists]
        aug_features = [obj_id, E, pt, eta, phi, abs_eta, pt_over_E, deltaR_met, exists]
        perobj = torch.stack(aug_features, dim=-1) # [N, n_max_objs, F=9]
        # Flat back to event level
        perobj = perobj.view(X.shape[0], n_max_objs*9)
        # Add event-level features: [E_T_miss, phi_met]
        evt_feats = batch[:,:2]
        X_aug = torch.cat([evt_feats, perobj], dim=-1)
        return X_aug
    X_train_aug = particle_feature_augment(X_train)
    X_val_aug   = particle_feature_augment(X_val)
    # --- 3. Compute Normalization Stats using non-padded values ---
    is_particle = X_train_aug[:,2::9][:,0::1] > 0 # [N, n_max_objs] (obj_id > 0)
    mask2d = is_particle.reshape(X_train_aug.shape[0], n_max_objs)
    means = X_train_aug.mean(dim=0)
    stds  = X_train_aug.std(dim=0) + 1e-5
    means[2:] = (X_train_aug[:,2:][X_train_aug[:,2:] !=0].mean(dim=0))
    stds[2:]  = (X_train_aug[:,2:][X_train_aug[:,2:] !=0].std(dim=0)) + 1e-5
    means = means.float(); stds = stds.float()
    X_train_aug = (X_train_aug - means)/(stds)
    X_val_aug   = (X_val_aug   - means)/(stds)
    #
    preproc = PreprocessModule(obj_mask=torch.ones_like(means, dtype=torch.bool),
                              feature_means=means,
                              feature_stds=stds,
                              n_max_objs=n_max_objs,
                              obj_id_map=obj_id_map)
    X_train_p = preproc(X_train_aug)
    X_val_p   = preproc(X_val_aug)
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds = TensorDataset(X_val_p, Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    return train_loader, val_loader, preproc

# ----- FREE SECTION: Slot Attention Implementation -----
class SlotAttention(nn.Module):
    def __init__(self, n_slots, dim_in, dim_slot, n_iter=3):
        super().__init__()
        self.n_slots = n_slots
        self.n_iter = n_iter
        self.dim_in = dim_in
        self.dim_slot = dim_slot
        self.slots_mu = nn.Parameter(torch.randn(1,n_slots,dim_slot))
        self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1,n_slots,dim_slot)))
        self.project_q = nn.Linear(dim_slot, dim_slot, bias=False)
        self.project_k = nn.Linear(dim_in, dim_slot, bias=False)
        self.project_v = nn.Linear(dim_in, dim_slot, bias=False)
        self.gru = nn.GRUCell(dim_slot, dim_slot)
        self.mlp = nn.Sequential(
            nn.Linear(dim_slot, dim_slot), nn.ReLU(),
            nn.Linear(dim_slot, dim_slot)
        )
        self.norm_input = nn.LayerNorm(dim_in)
        self.norm_slots = nn.LayerNorm(dim_slot)
        self.norm_pre_ff = nn.LayerNorm(dim_slot)
    def forward(self, x):
        # x: [B, n_obj, dim_in]
        B, n_obj, D = x.shape
        mu = self.slots_mu.expand(B,self.n_slots,self.dim_slot)
        sigma = self.slots_sigma.expand(B,self.n_slots,self.dim_slot)
        slots = mu + torch.randn_like(sigma)*sigma if self.training else mu
        x = self.norm_input(x)
        for _ in range(self.n_iter):
            slots_prev = slots
            q = self.project_q(self.norm_slots(slots)) # [B,n_slots,dim_slot]
            k = self.project_k(x)                     # [B,n_obj,dim_slot]
            attn_logits = torch.einsum('bid,bjd->bij', q, k) # [B, n_slots, n_obj]
            attn = torch.softmax(attn_logits/torch.sqrt(torch.tensor(self.dim_slot, dtype=x.dtype, device=x.device)), dim=1) # over slots
            attn = attn + 1e-8
            attn = attn/attn.sum(dim=2, keepdim=True)
            v = self.project_v(x)
            updates = torch.einsum('bij,bjd->bid', attn, v)
            slots = self.gru(updates.reshape(-1,self.dim_slot), slots_prev.reshape(-1,self.dim_slot))
            slots = slots.reshape(B, self.n_slots, self.dim_slot)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots, attn # slots: [B, n_slots, dim_slot], attn: [B, n_slots, n_obj]

# ----- FREE SECTION: Transformer Encoder (Physics-informed) -----
class ParticleEncoder(nn.Module):
    def __init__(self, in_feats, emb_dim, n_heads, n_layers, n_max_objs):
        super().__init__()
        # Input is [B, n_obj, in_feats]
        self.n_max_objs = n_max_objs
        self.obj_embedding = nn.Embedding(20, 8, padding_idx=0)
        self.feat_proj = nn.Linear(in_feats-1, emb_dim-8) # All features except obj_id
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads,
                                                   dim_feedforward=emb_dim*2, batch_first=True, dropout=0.15)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    def forward(self, obj_feats):
        # obj_feats: [B, n_obj, obj_in_feats]; first column is obj_id
        obj_id = obj_feats[:,:,0].long().clamp(min=0, max=19)
        embed = self.obj_embedding(obj_id) # [B, n_obj, 8]
        rest = obj_feats[:,:,1:]
        continuous = self.feat_proj(rest) # [B, n_obj, emb_dim-8]
        x = torch.cat([embed, continuous], dim=-1) # [B, n_obj, emb_dim]
        x = self.encoder(x)
        return x

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        # Event features: [E_T_miss, phi_met]; augment + 36 objects x 9 features = 326
        # Split event/global features and object features
        n_max_objs = 36
        obj_dim = 9
        evt_feat = 2
        self.n_max_objs = n_max_objs
        # Transformer encoding
        # Encoder embeds each particle slot.
        self.particle_encoder = ParticleEncoder(in_feats=obj_dim, emb_dim=32, n_heads=4, n_layers=2, n_max_objs=n_max_objs)
        # Slot Attention: 4 slots (since 4 tops per event), slot_dim=32
        self.slot_attention = SlotAttention(n_slots=4, dim_in=32, dim_slot=32, n_iter=3)
        # Event features projection
        self.event_proj = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU()
        )
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(4*32+16, 64), nn.ReLU(),
            nn.Dropout(0.18),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        # x: [B, 2 + n_max_objs*9]
        evt_feats = x[:, :2] # [B,2]
        particle_feats = x[:, 2:].view(-1, self.n_max_objs, 9)
        # Physics-informed masking: remove padded items
        mask = (particle_feats[:,:,0] > 0) # obj_id > 0
        # Encoder: [B, n_max_objs, obj_dim=9] -> [B, n_max_objs, 32]
        enc_out = self.particle_encoder(particle_feats)
        # For padded slots, zero out
        enc_out = enc_out * mask.unsqueeze(-1)
        # Slot Attention
        slot_repr, slot_attn = self.slot_attention(enc_out)
        slot_repr = slot_repr.view(x.shape[0], -1) # [B, 4*32]
        evt_emb = self.event_proj(evt_feats)
        feats = torch.cat([slot_repr, evt_emb], dim=-1)
        logits = self.classifier(feats)
        return logits.squeeze(-1)

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=2)
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    best_auc = 0.
    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        tr_preds = []
        tr_targets = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float()
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.)
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
            tr_preds.append(outputs.detach().cpu())
            tr_targets.append(yb.cpu())
        avg_tr_loss = tr_loss/len(train_loader.dataset)
        training_loss.append(avg_tr_loss)
        trainallpred = torch.cat(tr_preds).numpy()
        trainalltarget = torch.cat(tr_targets).numpy()
        trainauc = roc_auc_score(trainalltarget, trainallpred)
        training_acc.append(trainauc)
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).float()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item()*xb.size(0)
                val_preds.append(outputs.cpu())
                val_targets.append(yb.cpu())
        avg_val_loss = val_loss/len(val_loader.dataset)
        validation_loss.append(avg_val_loss)
        val_all_preds = torch.cat(val_preds).numpy()
        val_all_targets = torch.cat(val_targets).numpy()
        val_auc = roc_auc_score(val_all_targets, val_all_preds)
        validation_acc.append(val_auc)
        # Scheduler on val auc (AUC up = good)
        scheduler.step(val_auc)
        if val_auc > best_auc:
            best_auc = val_auc
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