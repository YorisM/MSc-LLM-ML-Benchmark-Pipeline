import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

# --- Lorentz Utilities ---
def fourvec_from_event(x):
    '''
    Given [batch, 105]: Extracts [batch, n_obj, 4] tensor of E, p_x, p_y, p_z for all valid objects (i.e., mask==1)
    The structure is: [E_{T,miss}, phi_{Et_miss}, obj_1, E_1, p_T1, eta_1, phi_1, ..., obj_n, E_n, p_Tn, eta_n, phi_n, ...]
    Where E_T, phi_Et are at indexes 0,1, not used as objects.
    For all objects: group every 4 after index=2.
    '''
    batch_size = x.shape[0]
    obj_start = 2
    obj_features = 4
    max_obj = (x.shape[1] - 2)//obj_features
    objs = x[:, obj_start:].reshape(batch_size, max_obj, obj_features) # [E, p_T, eta, phi]
    # Mask: If (E,p_T,eta,phi) all==0 -> padding
    mask = (objs.abs().sum(dim=2) > 0).float()  # [batch, max_obj]
    # Compute px, py, pz
    E      = objs[...,0]
    p_T    = objs[...,1]
    eta    = objs[...,2]
    phi    = objs[...,3]
    px     = p_T * torch.cos(phi)
    py     = p_T * torch.sin(phi)
    pz     = p_T * torch.sinh(eta)
    fours  = torch.stack([E, px, py, pz], dim=-1) # [batch, max_obj, 4]
    return fours, mask  # mask: nonzero = valid, 0 = pad

def lorentz_metric():
    '''Return (-1,1,1,1) diag metric tensor.'''
    return torch.tensor([-1, 1, 1, 1], dtype=torch.float32)  # Not registered as buffer intentionally.

def lorentz_inner(a, b):
    '''
    Compute Lorentz inner product between [...,4], [...,4] tensors using metric (-1,1,1,1):
    (-E1*E2 + px1*px2 + py1*py2 + pz1*pz2)
    '''
    return -a[...,0]*b[...,0] + (a[...,1:]*b[...,1:]).sum(-1)

def batch_outer(x, y):
    '''For [...,4] x [...,4], return [...,4,4] (outer product per vector in batch).'''
    return x.unsqueeze(-1) * y.unsqueeze(-2)

def invariant_mass2(four):
    '''Computes Lorentz-invariant mass squared for a [batch, N, 4] tensor.'''
    return lorentz_inner(four, four)

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
    def __init__(self, mean, std, obj_mask):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("obj_mask", obj_mask)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standardize all except the 0-padded objects using mask
        # Use obj_mask to mask out objects (for zero-padded rows)
        x = (x - self.mean) / (self.std + 1e-8)
        x = x * self.obj_mask
        return x

def derive_mean_std_mask(X):
    # Derive mask: 0 for padded object features, 1 otherwise
    obj_pad = (X[:,2:].reshape(X.size(0), -1, 4).abs().sum(2) > 0).float() # [batch, n_obj]
    # Mask: map all valid objects to 1, pads to 0
    obj_mask = obj_pad.unsqueeze(-1).repeat(1,1,4).reshape(X.size(0), -1) # [batch, nb_obj*4]
    obj_mask = torch.cat([torch.ones(X.size(0),2), obj_mask], dim=1)
    # Compute mean/std ignoring pads:
    obj_feat = X[obj_mask.bool()].view(-1)
    mean     = X.mean(dim=0, keepdim=True)
    std      = X.std(dim=0, keepdim=True) + 1e-6
    # For features that are *always* zero (padding), set std=1 to avoid nan
    std[std==0]=1.
    obj_mask = torch.where(torch.std(X,dim=0,keepdim=True)==0, torch.zeros_like(obj_mask[:,:1].T), torch.ones_like(obj_mask[:,:1].T)).T
    return mean, std, obj_mask[0:1]

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=1024):
    mean, std, obj_mask = derive_mean_std_mask(X_train)
    preproc = PreprocessModule(mean, std, obj_mask)
    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p,   Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    return train_loader, val_loader, preproc

# ----- FREE SECTION: Lorentz-Equivariant GNN -----
class LorentzSelfAttention(nn.Module):
    '''A message passing layer over Lorentz 4-vectors using tensor operations and invariant functions.'''
    def __init__(self, n_input, n_hidden, heads=4):
        super().__init__()
        self.n_input = n_input # features per object
        self.n_hidden = n_hidden
        self.heads = heads
        # Each head transforms 4-vec -> vector features
        self.query = nn.Linear(4, n_hidden*heads, bias=False)
        self.key   = nn.Linear(4, n_hidden*heads, bias=False)
        self.value = nn.Linear(4, n_hidden*heads, bias=False)
        self.out   = nn.Linear(n_hidden*heads, n_hidden)
        self.layer_norm = nn.LayerNorm([n_hidden])
    def forward(self, fourvecs, mask):
        # fourvecs: [batch, n_obj, 4], mask: [batch, n_obj]
        batch, n_obj, _ = fourvecs.shape
        Q = self.query(fourvecs).view(batch, n_obj, self.heads, self.n_hidden) # [B,N,H,F]
        K = self.key(fourvecs).view(batch, n_obj, self.heads, self.n_hidden)
        V = self.value(fourvecs).view(batch, n_obj, self.heads, self.n_hidden)

        # Lorentz inner products [B,N,N]
        metric = lorentz_metric().to(fourvecs.device)
        dot = -fourvecs[:,:,0].unsqueeze(2)*fourvecs[:,:,0].unsqueeze(1)
        dot = dot + (fourvecs[:,:,1:].unsqueeze(2)*fourvecs[:,:,1:].unsqueeze(1)).sum(-1)
        # Use dot product as attention logits (since Lorentz inner is invariant)
        attn_logits = dot / (1. + dot.abs().mean()) # [B,N,N]
        # mask: don't attend to padded objects
        mask1 = mask.unsqueeze(2).expand_as(attn_logits)
        mask2 = mask.unsqueeze(1).expand_as(attn_logits)
        attn_logits = attn_logits.masked_fill((mask1*mask2)==0, float('-inf'))
        attn_scores = torch.softmax(attn_logits, dim=-1) # [B,N,N]

        # Aggregate messages (apply to each head+feature)
        out_heads = []
        for h in range(self.heads):
            # Q: [B,N,F], K: [B,N,F], V: [B,N,F]
            messages = torch.matmul(attn_scores, V[:,:,h,:]) # [B,N,F]
            out_heads.append(messages)
        out = torch.cat(out_heads, dim=-1) # [B,N,H*F]
        out = self.out(out)              # [B,N,n_hidden]
        out = self.layer_norm(out)
        out = torch.relu(out)
        # Zero out pad objs
do     ut = out * mask.unsqueeze(-1)
        return out

class LorentzGNNBlock(nn.Module):
    '''A block that passes Lorentz 4-vectors, allows higher-order invariants.'''
    def __init__(self, in_features, out_features):
        super().__init__()
        self.attn = LorentzSelfAttention(4, out_features)
        # Add non-linear MLP over invariant features
        self.invar_mlp = nn.Sequential(
            nn.Linear(8, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.ReLU(),
        )
    def forward(self, fourvecs, mask):
        # fourvecs: [B,N,4], mask: [B,N]
        x_attn = self.attn(fourvecs, mask) # out: [B,N,F]
        # Compute all-pairs invariant masses for each object (sum over connections)
        pair_m2 = invariant_mass2(fourvecs.unsqueeze(2)+fourvecs.unsqueeze(1)) # [B,N,N]
        feat1 = fourvecs.norm(dim=-1, keepdim=True)
        feat2 = invariant_mass2(fourvecs).unsqueeze(-1) # mass^2
        feat3 = pair_m2.sum(dim=2, keepdim=True) # sum_i(m^2(obj, obj_i))
        features = torch.cat([feat1, feat2, feat3, x_attn], dim=-1) # [B,N,1+1+1+F]
        # Map to next four-vector (project down)
        stats = torch.cat([
            features.mean(dim=1),
            features.max(dim=1)[0],
            features.min(dim=1)[0],
            ],dim=-1)
        x_new = self.invar_mlp(stats)
        return x_new

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        # Estimate max number of objects
        self.n_obj = (input_dim - 2)//4
        self.obj_pad = input_dim
        # Final stat features: [mean, max, min] of features
        self.block1 = LorentzGNNBlock(4, 32)
        self.block2 = LorentzGNNBlock(32, 32)
        self.block3 = LorentzGNNBlock(32, 32)
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Sequential(
            nn.Linear(32*3, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(48,1),
        )
    def forward(self, x):
        # x: [batch, 105]
        fourvecs, mask = fourvec_from_event(x) # [B, N, 4], [B,N]
        # Apply Lorentz GNN blocks
        x1 = self.block1(fourvecs, mask) # [B, features]
        x2 = self.block2(fourvecs, mask) # [B, features]
        x3 = self.block3(fourvecs, mask)
        # Summary Pooling: 
        pooled = torch.cat([x3, x2, x1], dim=-1)
        pooled = self.dropout(pooled)
        # Optional: add E_Tmiss, phi_Etmiss
        global_feats = x[:,:2]
        embed = torch.cat([pooled, global_feats], dim=-1)
        out = self.fc(embed)
        return out.squeeze(-1)

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    training_loss, validation_loss = [], []
    training_acc, validation_acc = [], []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0; n_train = 0; n_corr= 0
        train_probs=[]; train_target=[]
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = (torch.sigmoid(logits)>0.5).long()
            n_corr += (preds==yb).sum().item()
            epoch_loss += loss.item()*xb.size(0)
            train_probs.append(torch.sigmoid(logits).detach().cpu())
            train_target.append(yb.detach().cpu())
            n_train += xb.size(0)
        training_loss.append(epoch_loss/n_train)
        training_acc.append(n_corr/n_train)
        scheduler.step()
        # ---- Validation ----
        model.eval()
        val_loss=0; val_corr=0; n_val=0
        val_probs=[]; val_targets=[]
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb.float())
                preds = (torch.sigmoid(logits)>0.5).long()
                val_corr+=(preds==yb).sum().item()
                val_loss += loss.item()*xb.size(0)
                val_probs.append(torch.sigmoid(logits).cpu())
                val_targets.append(yb.cpu())
                n_val += xb.size(0)
        validation_loss.append(val_loss/n_val)
        validation_acc.append(val_corr/n_val)
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