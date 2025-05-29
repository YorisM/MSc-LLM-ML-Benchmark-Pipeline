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
    def __init__(self, mask, means, stds, obj_map):
        super().__init__()
        self.register_buffer('mask', mask)
        self.register_buffer('means', means)
        self.register_buffer('stds', stds)
        self.register_buffer('obj_map', obj_map)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mask padded objects
        # Remove padded objects (mask==0)
        x = x * self.mask  # mask out padded region
        # Apply standardization to all non-ohe features (leave OHE as is)
        x = (x - self.means) / (self.stds + 1e-6)
        return x

def get_object_id_slices(dim=105):
    # Helper: returns [(start,end), ...] for each object slot
    slices = []
    idx = 2
    while idx + 5 <= dim:
        slices.append((idx, idx+5))
        idx += 5
    return slices

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=256):
    # --- Find mask of real objects (1) vs padded (0) -----
    N = X_train.shape[0]
    dim = X_train.shape[1]
    device = X_train.device
    # padded objects: assume if all entries in a slot are zero
    slices = get_object_id_slices(dim)
    mask = torch.zeros((1, dim), dtype=torch.float32, device=device)
    mask[0,0:2] = 1.0 # always keep E_Tmiss, phi_MET
    for (s,e) in slices:
        if torch.any(X_train[:,s:e]!=0):
            mask[0,s:e] = 1.0
    mask = mask # [1, 105]
    # --- Standardize (robust Median/IQR or mean/std) ---
    # Only over non-padded entries
    vals = X_train[mask[0]==1]
    means = torch.zeros(dim, dtype=torch.float32)
    stds = torch.ones(dim, dtype=torch.float32)
    for i in range(dim):
        msk_i = mask[0,i]==1
        if msk_i:
            dat = X_train[:,i][X_train[:,i]!=0]
            if dat.numel()>0:
                means[i] = dat.mean()
                stds[i] = dat.std()
    # --- Encode object ids as one-hot (if desired) ---
    obj_ids = []
    for k, (s,e) in enumerate(slices):
        vals = X_train[:,s].unique()
        for v in vals:
            if v not in obj_ids and v != 0:
                obj_ids.append(v.item())
    obj_map = torch.zeros(20, dtype=torch.float32)  # max 20 object types for safety
    for ix,v in enumerate(obj_ids):
        obj_map[int(v)] = ix+1
    # --- Create module ---
    preproc = PreprocessModule(mask, means, stds, obj_map)
    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p,   Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    return train_loader, val_loader, preproc

# ----- FREE SECTION: Lorentz Equivariant Message Passing -----
def to_fourvec(xi):
    # Given [obj_id, E, pt, eta, phi], return [E, px, py, pz]
    E = xi[...,1]
    pt = xi[...,2]
    eta = xi[...,3]
    phi = xi[...,4]
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    return torch.stack([E, px, py, pz], dim=-1)

def lorentz_invariant_feats(fourvecs):
    # fourvecs: [batch, nobj, 4]
    batch, nobj, _ = fourvecs.shape
    # (i) object m^2 = E^2 - (px^2+py^2+pz^2)
    msq = fourvecs[...,0]**2 - (fourvecs[...,1:]**2).sum(dim=-1)
    # (ii) sum of four-vectors (event sum)
    vec_sum = fourvecs.sum(dim=1)  # [batch, 4]
    # (iii) largest pT, sum pT
    pt = torch.sqrt(fourvecs[...,1]**2 + fourvecs[...,2]**2)
    max_pt = pt.max(dim=1)[0]
    sum_pt = pt.sum(dim=1)
    # (iv) pairwise inv mass
    i,j = torch.triu_indices(nobj,nobj,offset=1)
    pairs = fourvecs[:,i,:]+fourvecs[:,j,:]
    # mass squared of sum
    pair_m2 = pairs[...,0]**2 - (pairs[...,1:]**2).sum(dim=-1)
    # min, max, mean pair_m2
    # ignore nan/neg: replace <0 with 0
    valid = (pair_m2>0).float()
    mean_pairmass = (pair_m2*valid).sum(dim=1)/(valid.sum(dim=1)+1e-6)
    max_pairmass = pair_m2.max(dim=1)[0]
    min_pairmass = (pair_m2+(1-valid)*float('inf')).min(dim=1)[0]
    min_pairmass[min_pairmass>1e8]=0.0
    # same for all batch
    return torch.cat([
        msq.mean(dim=1,keepdim=True),
        msq.max(dim=1,keepdim=True),
        msq.min(dim=1,keepdim=True),
        vec_sum,
        max_pt.unsqueeze(1), sum_pt.unsqueeze(1),
        mean_pairmass.unsqueeze(1), max_pairmass.unsqueeze(1), min_pairmass.unsqueeze(1)
    ],dim=1)

def get_obj_tensor(x):
    # Given x: [batch, 105], returns objects tensor: [batch, nobj, 5]
    # obj n starts at 2+5*i
    bsz = x.shape[0]
    nobj = (x.shape[1]-2)//5
    obj_list=[]
    for i in range(nobj):
        obj = x[:,2+5*i:2+5*(i+1)]
        obj_list.append(obj.unsqueeze(1))
    objs = torch.cat(obj_list,dim=1) # [batch, nobj, 5]
    return objs

class LorentzMPNLayer(nn.Module):
    def __init__(self, input_dim, msg_dim, hidden_dim):
        super().__init__()
        # Input_dim = 8 (eg: [E, px, py, pz, pt, eta, phi, orig_id]) 
        self.f_node = nn.Linear(input_dim, hidden_dim)
        self.f_msg = nn.Linear(2*input_dim, msg_dim) # edge: i-j
        self.f_agg = nn.Linear(msg_dim, hidden_dim)
        self.f_update = nn.GRUCell(hidden_dim, input_dim)
    def forward(self, X):
        # X: [B, nobj, input_dim]
        B, nobj, featd = X.shape
        device = X.device
        # For each node, get message from every other (dense graph)
        h_node = self.f_node(X)      # [B,nobj,h]
        # Broadcasting pair-wise concat
        Xi = X.unsqueeze(2).repeat(1,1,nobj,1) # [B,nobj,nobj, input_dim]
        Xj = X.unsqueeze(1).repeat(1,nobj,1,1) # [B,nobj,nobj, input_dim]
        pair = torch.cat([Xi, Xj], dim=-1)
        msg = self.f_msg(pair) # [B,nobj,nobj,msgd]
        msg = msg*torch.eye(nobj,device=device).unsqueeze(0).unsqueeze(-1).logical_not() # zero diagonal self-messages
        agg = msg.sum(dim=2) # sum over neighbors [B,nobj,msgd]
        agg = torch.relu(self.f_agg(agg))  # [B,nobj,hidden]
        # update node
        xnew = self.f_update(agg.view(-1,agg.shape[-1]), X.view(-1, X.shape[-1]))
        xnew = xnew.view(B,nobj,-1)
        return xnew

class LorentzNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # --- Constants ---
        self.input_dim = input_dim
        nobj = (input_dim-2)//5
        self.nobj = nobj
        # --- Feature extraction ---
        # After Preprocess: everything standardized. input=[ETmiss,phimet,obj1..objN]
        mpn_input_dim = 8 # [E, px, py, pz, pt, eta, phi, orig_id]
        # --- Input encode ---
        self.Etmiss_norm = nn.LayerNorm(2)
        self.obj_embed = nn.Embedding(21, 2) # object id embedding
        # --- Message Passing ---
        self.mpn1 = LorentzMPNLayer(mpn_input_dim, 32, 32)
        self.mpn2 = LorentzMPNLayer(mpn_input_dim, 32, 32)
        # --- Invariant Feats ---
        self.inv_proj = nn.Sequential(nn.Linear(18, 16), nn.ReLU())
        # --- Pooling/output ---
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.final = nn.Sequential(
            nn.Linear(34, 32), nn.ReLU(),
            nn.Linear(32, 8), nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        # x: [B, 105]
        bsz = x.shape[0]
        device = x.device
        nobj = self.nobj
        # 1. Get MET
        Etmiss = x[:,:2]     # [B,2]
        Etmiss = self.Etmiss_norm(Etmiss)
        # 2. Get objects as [B, nobj, 5]
        objs = get_obj_tensor(x) # [B, nobj, 5]
        # 3. Store id as int & embed
        obj_ids = objs[:,:,0].long().clamp(0,20) # (max 20 types)
        obj_emb = self.obj_embed(obj_ids) # [B,nobj,2]
        # 4. Build initial node features
        fvec = to_fourvec(objs)
        pt = objs[...,2]
        eta = objs[...,3]
        phi = objs[...,4]
        node_in = torch.cat([fvec, pt.unsqueeze(-1), eta.unsqueeze(-1), phi.unsqueeze(-1), obj_emb],dim=-1) # [B,nobj,8]
        # 5. Message passing
        node_h = self.mpn1(node_in)
        node_h2 = self.mpn2(node_h)
        # 6. Global Lorentz-invariant features
        inv_in = torch.cat([
            lorentz_invariant_feats(fvec),
            lorentz_invariant_feats(fvec+1e-6*torch.randn_like(fvec)), # slight noise
        ], dim=1) # [B,18]
        inv_vec = self.inv_proj(inv_in) # [B,16]
        # 7. Pool node_h2
        pooled = self.pool(node_h2.permute(0,2,1)).squeeze(-1) # [B,h=32]
        # 8. Concat to global
        feat = torch.cat([Etmiss, inv_vec, pooled],dim=1) # [B,34]
        # 9. Final output
        x = self.final(feat)  # [B,1]
        return x.squeeze(-1)

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = LorentzNet(input_dim)
    def forward(self, x):
        return self.net(x)

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.BCEWithLogitsLoss()
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    for ep in range(epochs):
        model.train()
        tloss = 0
        tacc = 0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.float().to(device)
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = crit(y_pred, yb)
            loss.backward()
            optimizer.step()
            tloss += loss.item()*xb.size(0)
            y_out = torch.sigmoid(y_pred)>0.5
            tacc  += (y_out==yb.bool()).float().sum().item()
            n += xb.size(0)
        training_loss.append(tloss/n)
        training_acc.append(tacc/n)
        # Validation
        model.eval()
        vloss=0
        vacc=0
        vpreds=[]
        vlabs=[]
        with torch.no_grad():
            for xvb, yvb in val_loader:
                xvb = xvb.to(device)
                yvb = yvb.float().to(device)
                yv_pred = model(xvb)
                loss = crit(yv_pred, yvb)
                vloss += loss.item()*xvb.size(0)
                yup = torch.sigmoid(yv_pred)>0.5
                vacc += (yup==yvb.bool()).float().sum().item()
                vpreds.append(yv_pred.cpu())
                vlabs.append(yvb.cpu())
        validation_loss.append(vloss/len(val_loader.dataset))
        validation_acc.append(vacc/len(val_loader.dataset))
        # AUC eval (print)
        vpreds = torch.cat(vpreds)
        vlabs = torch.cat(vlabs)
        auc = roc_auc_score(vlabs.numpy(), torch.sigmoid(vpreds).numpy())
        print(f'Epoch {ep+1:2d}: loss={training_loss[-1]:.4f}, val_loss={validation_loss[-1]:.4f}, val_acc={validation_acc[-1]:.4f}, val_auc={auc:.4f}')
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