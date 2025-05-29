import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

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

# ===================================
# === Lorentz Equivariant Modules ===
# ===================================
# Object parsing: helper function
# Each object: (obj_n, E, pT, eta, phi)
def parse_event_objects(x):
    # x: [batch, 105]
    # Returns:
    #  - E_Tmiss, phi_E_Tmiss: [batch, 2]
    #  - obj_m: [batch, maxobj] int
    #  - E, pT, eta, phi: [batch, maxobj]
    E_Tmiss     = x[:,0] # [B]
    phi_E_Tmiss = x[:,1]
    objects = []
    for i in range(0, 104, 5):
        o = x[:,2+i:2+i+5] # (B, 5)
        objects.append(o)
    objects = torch.stack(objects, dim=1) # (B, maxobj, 5)
    obj_mask = (objects[:,:,0]!=0) # padding=0
    obj_ids = objects[:,:,0]
    Es   = objects[:,:,1]
    pTs  = objects[:,:,2]
    etas = objects[:,:,3]
    phis = objects[:,:,4]
    return E_Tmiss, phi_E_Tmiss, obj_ids, Es, pTs, etas, phis, obj_mask

# Convert (E,pt,eta,phi) -> 4-vector (E,px,py,pz)
def fourvec_from_Eptetaphi(E,pT,eta,phi):
    px = pT * torch.cos(phi)
    py = pT * torch.sin(phi)
    # pz = pT*sinh(eta)
    pz = pT * torch.sinh(eta)
    return torch.stack([E, px, py, pz], dim=-1) # (...,4)

# Lorentz invariant dot product: v1,v2 = (...,4), (...,4) => (...,)
def lorentz_dot(a,b):
    # (-,+,+,+) metric
    return a[...,0]*b[...,0] - torch.sum(a[...,1:]*b[...,1:],dim=-1)

# Construct pairwise Lorentz invariants and Edge Features
def pairwise_lorentz(vectors, mask):
    # vectors: (B, N, 4), mask: (B,N)
    B,N,_ = vectors.shape
    vv1 = vectors.unsqueeze(2).expand(-1,-1,N,-1)  # (B,N,N,4)
    vv2 = vectors.unsqueeze(1).expand(-1,N,-1,-1)  # (B,N,N,4)
    msk1 = mask.unsqueeze(2).expand(-1,-1,N)
    msk2 = mask.unsqueeze(1).expand(-1,N,-1)
    valid = msk1 & msk2 # (B,N,N)
    mass2 = lorentz_dot(vv1+vv2, vv1+vv2)
    dot   = lorentz_dot(vv1, vv2)
    # For edge features: invariant mass squared, dot product
    edge_feat = torch.stack([mass2, dot], dim=-1) # (B,N,N,2)
    edge_feat = edge_feat * valid.unsqueeze(-1)
    return edge_feat, valid

# ===========
# Preprocess : Lorentz 4-vector standardization
# ===========
class PreprocessModule(torch.nn.Module):
    def __init__(self, means=None, stds=None, **kwargs):
        super().__init__()
        if means is None:
            means = torch.zeros(7)
        if stds is None:
            stds = torch.ones(7)
        self.register_buffer('means', means)
        self.register_buffer('stds', stds)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply per-feature zscoring (only on nonzero padded, so leave maskable features unaltered)
        # 0: E_Tmiss, 1: phi_E_Tmiss, then per-object (objid, E, pt, eta, phi)x21
        out = x.clone()
        # scale global ETmiss and phi
        out[:,0] = (x[:,0]-self.means[0])/self.stds[0]
        out[:,1] = (x[:,1]-self.means[1])/self.stds[1]
        for j in range(21):
            b = 2+5*j
            if b+1<out.shape[1]:
                # E, pt, eta, phi
                for c in range(4):
                    out[:,b+1+c] = (x[:,b+1+c]-self.means[c+2])/self.stds[c+2]
        return out

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=1024):
    # Fit means/stds on X_train, skipping zero padded features
    feats = []
    feats.append(X_train[:,0][X_train[:,0]!=0]) # ETmiss
    feats.append(X_train[:,1][X_train[:,1]!=0]) # phi_ETmiss
    for j in range(21):
        b=2+5*j
        if b+1<X_train.shape[1]:
            for c in range(4):
                feats.append(X_train[:,b+1+c][X_train[:,b]!=0])
    means = torch.tensor([f.mean().item() if len(f)>0 else 0 for f in feats],dtype=torch.float32)
    stds  = torch.tensor([f.std().item()  if len(f)>0 else 1 for f in feats],dtype=torch.float32)
    preproc = PreprocessModule(means=means, stds=stds)
    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p,   Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    return train_loader, val_loader, preproc

# ===================================
# === Lorentz Message Passing Layer ===
# ===================================
class LorentzMP(nn.Module):
    def __init__(self, feat_dim, edge_dim, hidden_dim):
        super().__init__()
        # Attention + edge update with Lorentz invariant edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim+2*feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim,1),
            nn.Sigmoid()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(feat_dim+hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,feat_dim),
        )
    def forward(self, h, edge_feat, mask):
        # h: (B,N,F), edge_feat:(B,N,N,E), mask:(B,N)
        B,N,F = h.shape
        msk1 = mask.unsqueeze(2).expand(-1,-1,N)
        msk2 = mask.unsqueeze(1).expand(-1,N,-1)
        valid = msk1 & msk2
        h1 = h.unsqueeze(2).expand(-1,-1,N,-1) # (B,N,N,F)
        h2 = h.unsqueeze(1).expand(-1,N,-1,-1) # (B,N,N,F)
        edge_in = torch.cat([h1,h2,edge_feat],dim=-1) # (B,N,N,2F+edge_dim)
        edge_msg = self.edge_mlp(edge_in) # (B,N,N,H)
        attn    = self.attn(edge_msg).squeeze(-1) # (B,N,N)
        attn = attn * valid.float()
        normed_attn = attn/(attn.sum(dim=-1,keepdim=True)+1e-6)
        msg_aggr = (edge_msg*normed_attn.unsqueeze(-1)).sum(dim=2) # (B,N,H)
        nodein = torch.cat([h, msg_aggr], dim=-1)
        hout = self.node_mlp(nodein) # (B,N,F)
        hout = hout*mask.unsqueeze(-1)
        return hout

# =========================
# The Main Model
# =========================
class Classifier(nn.Module):
    def __init__(self, input_dim, nobj=21, hidden=32, nmp=3):
        super().__init__()
        self.maxobj = nobj
        # Object type embedding (if desired)
        self.obj_embed = nn.Embedding(20, 4) # 20 object types
        # Initial node features: obj_emb (4), E, pt, eta, phi (4), fourvec (E,px,py,pz)(4)
        self.nf = 4+4+4 # 12 per-object
        self.input_proj = nn.Sequential(
            nn.Linear(self.nf, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.gnn_layers = nn.ModuleList([
            LorentzMP(hidden, edge_dim=2, hidden_dim=hidden) for _ in range(nmp)
        ])
        self.event_mlp = nn.Sequential(
            nn.Linear(hidden+2, 32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
    def forward(self, x):
        # x: [B,105]
        B = x.shape[0]
        E_Tmiss, phi_E_Tmiss, obj_ids, Es, pTs, etas, phis, mask = parse_event_objects(x)
        # Clamp obj_ids for embedding lookup
        obj_ids_clamped = torch.clamp(obj_ids.long(),0,self.obj_embed.num_embeddings-1)
        obj_emb = self.obj_embed(obj_ids_clamped) # (B,N,4)
        fourvec = fourvec_from_Eptetaphi(Es,pTs,etas,phis) # (B,N,4)
        # Initial node feat: [obj_emb, E, pt, eta, phi, fourvec]
        node_raw = torch.cat([
            obj_emb,
            Es.unsqueeze(-1), pTs.unsqueeze(-1), etas.unsqueeze(-1), phis.unsqueeze(-1),
            fourvec
        ],dim=-1) # (B,N,12)
        node = self.input_proj(node_raw) # (B,N,H)
        edge_feat, edge_mask = pairwise_lorentz(fourvec, mask)
        for gnn in self.gnn_layers:
            node = gnn(node, edge_feat, mask)
        # Pool to event (sum, mask)
        node_sum = (node*mask.unsqueeze(-1)).sum(dim=1) # (B,H)
        # Global features: E_Tmiss, phi_E_Tmiss (normalized in preproc)
        global_feat = torch.stack([E_Tmiss, phi_E_Tmiss],dim=-1)
        event_feat = torch.cat([node_sum, global_feat],dim=-1)
        out = self.event_mlp(event_feat).squeeze(-1) # (B,)
        return out

# ======================
# Training Function
# ======================
def train_model(model, train_loader, val_loader, epochs=10):
    model = model.cuda() if torch.cuda.is_available() else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    training_loss=[]
    validation_loss=[]
    training_acc = []
    validation_acc = []
    for epoch in range(epochs):
        model.train()
        ep_loss = []
        ep_acc  = []
        for Xb, yb in train_loader:
            Xb, yb = Xb, yb.float()
            if torch.cuda.is_available():
                Xb, yb = Xb.cuda(), yb.cuda()
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            ep_loss.append(loss.item())
            yhat = torch.sigmoid(logits).detach().cpu().numpy()>0.5
            acc  = (yhat==yb.detach().cpu().numpy()).mean()
            ep_acc.append(acc)
        training_loss.append(np.mean(ep_loss))
        training_acc.append(np.mean(ep_acc))
        model.eval()
        with torch.no_grad():
            val_losses=[]
            val_accs=[]
            all_logits=[]
            all_targets=[]
            for Xb, yb in val_loader:
                Xb, yb = Xb, yb.float()
                if torch.cuda.is_available():
                    Xb, yb = Xb.cuda(), yb.cuda()
                logits = model(Xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                all_logits.append(logits.cpu())
                all_targets.append(yb.cpu())
                yhat = torch.sigmoid(logits).cpu().numpy()>0.5
                acc = (yhat==yb.cpu().numpy()).mean()
                val_accs.append(acc)
            validation_loss.append(np.mean(val_losses))
            validation_acc.append(np.mean(val_accs))
        # AUC at every epoch (replace acc list with AUC if desired)
        # (opt: Could add ROC curve plotting as well)
    return model, training_loss, validation_loss, training_acc, validation_acc

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

def main(dryrun=False):
    X_train, Y_train, X_val, Y_val = load_data()
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val)
    sample_X, _ = next(iter(train_loader))
    model = Classifier(input_dim=sample_X.shape[1])
    epochs = 1 if dryrun else 10
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs)
    if not dryrun:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
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

if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)