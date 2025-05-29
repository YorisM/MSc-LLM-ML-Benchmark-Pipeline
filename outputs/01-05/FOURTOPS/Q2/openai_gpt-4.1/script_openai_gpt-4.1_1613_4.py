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
    def __init__(self, means=None, stds=None, obj_mask=None, nobj=24):
        super().__init__()
        if means is not None and stds is not None:
            self.register_buffer("means", means)
            self.register_buffer("stds", stds)
        else:
            self.means = None
            self.stds = None
        # mask: 1 if element is object data, 0 if global (MET)
        if obj_mask is not None:
            self.register_buffer("obj_mask", obj_mask)
        else:
            self.obj_mask = None
        self.nobj = nobj
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply Z-score normalization only to object features
        # x shape: [batch, 105]
        if (self.means is not None) and (self.stds is not None):
            x = (x - self.means) / self.stds.clamp(min=1e-6)
        return x

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=512):
    nfeat = X_train.shape[1] # 105
    # The first two features are MET, phi_MET
    # Remaining: 103 = 24 objects * 4 features + 1 (padding/weight?)
    nobj = (nfeat - 1) // 4 # Actually (105-1)/4 = 26 so that cannot be. Instead, (105-2) is 103, 103/4 = 25.75. But in format, after MET + PHI, each object: [obj_n, E, pT, eta, phi] is 5 features, so (105-2)/5=20.6, so probably 20 objects max.
    # But from input, the order is: [E_T_miss, phi_Et_miss, obj_1, E1, pT1, eta1, phi1, obj_2, ...]
    # Each object: (id, E, pT, eta, phi): 5, so nobj = (105-2)//5 = 20.6 ~ 20
    nobj = (X_train.shape[1] - 2) // 5
    # Prepare normalization (means and stds) only for object features (excluding object id column, which is int), apply to kinematic quantities.
    means = torch.zeros((X_train.shape[1],), dtype=torch.float32)
    stds = torch.ones((X_train.shape[1],), dtype=torch.float32)
    # Mask: 1 for features to normalize, 0 otherwise
    mask = torch.zeros((X_train.shape[1],), dtype=torch.float32)
    # MET features (first 2) should be normalized
    mask[0] = 1; mask[1] = 1
    # For each object, columns: [id, E, pT, eta, phi] (5 fields)
    for i in range(nobj):
        offs = 2 + i * 5
        # mask id col as 0 (categorical).
        mask[offs] = 0
        # The next four are continuous
        mask[offs+1:offs+5] = 1
    obj_features = (mask == 1)
    means[obj_features] = X_train[:,obj_features].mean(dim=0)
    stds[obj_features] = X_train[:,obj_features].std(dim=0)
    # Register everything
    preproc = PreprocessModule(means, stds, mask, nobj)

    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p,   Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    return train_loader, val_loader, preproc

# ----- FREE SECTION: Lorentz Symmetry-Empowered Network -----
# Utilities for Lorentz structure parsing.
def parse_objects(x, nobj):
    """
    x: [B, 105], return: ids: [B, nobj], feat: [B, nobj, 4] (E, pT, eta, phi), mask: [B, nobj]
    """
    # x: [B, 105]
    B = x.shape[0]
    o = []
    obj_mask = []
    obj_ids = []
    for i in range(nobj):
        offs = 2 + i*5
        # Each: id (int), E, pT, eta, phi
        id_col = x[:, offs:offs+1]              # [B, 1]
        feat_col = x[:, offs+1:offs+5]          # [B, 4]
        # If all zeros: pad
        mask = (feat_col.abs().sum(dim=1, keepdim=True) > 0).float() # [B,1]
        o.append(feat_col)                      # [B,4]
        obj_mask.append(mask)                   # [B,1]
        obj_ids.append(id_col)
    feats = torch.stack(o, dim=1)              # [B, nobj, 4]
    masks = torch.cat(obj_mask, dim=1)         # [B, nobj]
    ids   = torch.cat(obj_ids,  dim=1)         # [B, nobj]
    return ids, feats, masks

def fourvec_to_cartesian(E, pT, eta, phi):
    # E, pT, eta, phi: (..., 4)
    # Output: (..., 4): (E, px, py, pz)
    px = pT * torch.cos(phi)
    py = pT * torch.sin(phi)
    pz = pT * torch.sinh(eta)
    return torch.stack((E, px, py, pz), dim=-1)

def cartesian_outer(f):
    # f: [B, nobj, 4]
    # Compute all pairwise Minkowski contractions
    # Outer product: [B, nobj, nobj, 4, 4]
    B, N, _ = f.size()
    f1 = f.unsqueeze(2)           # [B, nobj,1,4]
    f2 = f.unsqueeze(1)           # [B,1,nobj,4]
    # outer: [B, N, N, 4, 4]
    outer = f1.unsqueeze(-1) * f2.unsqueeze(-2)
    return outer

def minkowski_dot(v1, v2):
    # v1, v2: (...,4);  metric (+,-,-,-)
    m = torch.tensor([1, -1, -1, -1], dtype=v1.dtype, device=v1.device)
    return (v1 * m) * v2

def pairwise_invariant_mass(f):
    # f: [B, nobj, 4] (E, px, py, pz)
    B, N, D = f.size()
    f1 = f.unsqueeze(2) # B,N,1,4
    f2 = f.unsqueeze(1) # B,1,N,4
    metric = torch.tensor([1,-1,-1,-1], dtype=f.dtype, device=f.device)
    dots = ((f1*metric)*f2).sum(dim=-1) # B,N,N
    # Square root of diag only; off-diag for pairs
    return dots

class LorentzMPNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=32):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, out_dim)
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*in_dim+1, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, node_feat, mask, invmass):
        # node_feat: [B, N, D], mask: [B, N], invmass: [B,N,N]
        B, N, D = node_feat.shape
        # Expand node pairs
        fi = node_feat.unsqueeze(2).expand(B,N,N,D)   # B,N,N,D
        fj = node_feat.unsqueeze(1).expand(B,N,N,D)   # B,N,N,D
        mij = invmass
        # concatenate
        edge_input = torch.cat([fi, fj, mij.unsqueeze(-1)], dim=-1) # B,N,N,2D+1
        edge_feat = self.edge_mlp(edge_input) * (mask.unsqueeze(1).unsqueeze(-1)) # B,N,N,K
        msg = edge_feat.sum(dim=2)  # aggregate from neighbors [B,N,K]
        nodes = self.node_mlp(node_feat)
        out = self.norm(nodes + msg)
        out = out * mask.unsqueeze(-1)
        return out

class Classifier(nn.Module):
    def __init__(self, input_dim, nobj=20, hidden_dim=32, n_layers=3):
        super().__init__()
        # Split input: MET features (2), per-object (20*5)
        self.input_dim = input_dim
        self.nobj = nobj
        self.hidden_dim = hidden_dim
        # MLP for MET features
        self.met_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        # Initial embedding for per-object: id (1-hot for first 18 types), E, pT, eta, phi (4). Use embedding for id.
        self.n_id = 18 # Assume up to 18 object types
        self.id_embed = nn.Embedding(30, 4) # More than any possible id
        self.init_obj_mlp = nn.Sequential(
            nn.Linear(4+4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        # Lorentz invariant MPNN layers
        self.lorentz_layers = nn.ModuleList([
            LorentzMPNLayer(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        # Final aggregation
        self.node_agg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        # x: [B, 105]
        B = x.shape[0]
        met = x[:, :2] # [B,2]
        met_feat = self.met_mlp(met) # [B,H]
        nobj = (x.shape[1]-2)//5
        ids, feats, mask = parse_objects(x, nobj)
        # feats: [B, nobj, 4]
        # id embedding
        ids = ids.clamp(0, self.id_embed.num_embeddings-1).long()
        id_emb = self.id_embed(ids) # [B, nobj, 4]
        obj_x = torch.cat([feats, id_emb], dim=-1) # [B,nobj,8]
        obj_feat = self.init_obj_mlp(obj_x) # [B,nobj,H]
        # Transform 4-vector: (E, pT, eta, phi) to (E, px, py, pz)
        f_cart = fourvec_to_cartesian(feats[...,0], feats[...,1], feats[...,2], feats[...,3]) # [B, nobj,4]
        # Pairwise inv mass (invariant scalar)
        invmass = pairwise_invariant_mass(f_cart) # [B,nobj,nobj]
        out = obj_feat
        for layer in self.lorentz_layers:
            out = layer(out, mask, invmass)
        # Masked sum as event-level rep
        node_mask = mask.unsqueeze(-1)
        event_emb = (out * node_mask).sum(dim=1) / (node_mask.sum(dim=1)+1e-6)
        agg = self.node_agg(event_emb)  # [B,H]
        tot = torch.cat([agg, met_feat], dim=-1)
        y = self.final_mlp(tot).squeeze(-1) # [B]
        return y

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    best_auc = 0.
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    training_loss, validation_loss = [], []
    training_acc, validation_acc = [], []
    for ep in range(epochs):
        model.train()
        epoch_loss = 0.
        n_correct, n_total = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            ylogits = model(xb)
            loss = criterion(ylogits, yb.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*xb.size(0)
            ypred = (torch.sigmoid(ylogits)>0.5).long()
            n_correct += (ypred==yb).sum().item()
            n_total  += yb.size(0)
        training_loss.append(epoch_loss/n_total)
        training_acc.append(n_correct/n_total)
        model.eval()
        val_loss = 0.; val_corr=0; val_total=0
        y_true_list = []; y_score_list = []
        with torch.no_grad():
            for xb,yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                ylogits = model(xb)
                loss = criterion(ylogits, yb.float())
                val_loss += loss.item()*xb.size(0)
                yprob = torch.sigmoid(ylogits)
                ypred = (yprob>0.5).long()
                val_corr += (ypred==yb).sum().item()
                val_total += yb.size(0)
                y_true_list.append(yb.cpu())
                y_score_list.append(yprob.cpu())
        validation_loss.append(val_loss/val_total)
        validation_acc.append(val_corr/val_total)
        # Compute AUC
        y_true = torch.cat(y_true_list).numpy()
        y_score = torch.cat(y_score_list).numpy()
        auc = 0.5
        try:
            auc = roc_auc_score(y_true, y_score)
        except:
            pass
        if auc > best_auc:
            best_auc = auc
        print(f"Epoch {ep+1}/{epochs} | loss {training_loss[-1]:.4f}/{validation_loss[-1]:.4f} | acc {training_acc[-1]:.4f}/{validation_acc[-1]:.4f} | AUC={auc:.4f}")
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