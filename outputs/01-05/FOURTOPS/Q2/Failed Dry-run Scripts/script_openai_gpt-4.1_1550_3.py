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
def make_masks(X):
    """Returns a tensor mask [N_events, N_objects] where valid objects are 1, else 0"""
    N_obj = (X.shape[1]-2)//5
    mask = torch.zeros((X.shape[0], N_obj), dtype=torch.float32)
    for obj in range(N_obj):
        # The object id is at offset+0: if it's nonzero, the object is present
        id_idx = 2+obj*5
        mask[:, obj] = (X[:, id_idx]!=0).float()
    return mask

def find_object_stats(X):
    N_obj = (X.shape[1]-2)//5
    feats = []
    for offset in range(N_obj):
        idx_start = 2+offset*5
        # Columns: id, E, pT, eta, phi
        feats.append((X[:, idx_start],
                      X[:, idx_start+1],
                      X[:, idx_start+2],
                      X[:, idx_start+3],
                      X[:, idx_start+4]) )
    # Construct E, pT, eta, phi for valid objects
    E = torch.stack([f[1] for f in feats],1)
    pT = torch.stack([f[2] for f in feats],1)
    eta = torch.stack([f[3] for f in feats],1)
    phi = torch.stack([f[4] for f in feats],1)
    id_ = torch.stack([f[0] for f in feats],1)
    return id_, E, pT, eta, phi

class PreprocessModule(torch.nn.Module):
    def __init__(self,
                 obj_mean=None,
                 obj_std=None,
                 obj_mask=None):
        super().__init__()
        if obj_mean is not None and obj_std is not None:
            self.register_buffer('obj_mean', obj_mean)
            self.register_buffer('obj_std', obj_std)
        else:
            self.obj_mean = None
            self.obj_std  = None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize each object's E, pT, eta, phi (not id!), excluding padded zeros
        N_obj = (x.shape[1]-2)//5
        # x: [batch, 2 + 5*N_obj]
        batch = x.shape[0]
        # Retain MET as-is
        met = x[:,:2]
        # Objects [batch, N_obj, 5]
        x_ = x[:,2:].reshape(batch,N_obj,5)
        # id = x_[:,:,0] (leave as is)
        out = torch.zeros_like(x_)
        out[:,:,0]=x_[:,:,0]
        # Normalize E, pT, eta, phi
        for k in range(4):
            idx = k+1
            vals = x_[:,:,idx]
            # Use obj_mean/obj_std as [N_obj,4] shape
            mean = self.obj_mean[:,k]
            std = self.obj_std[:,k]
            out[:,:,idx]= (vals-mean)/std
        out = out.reshape(batch,-1)
        # Concatenate MET unchanged
        return torch.cat([met,out],dim=1)

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size):
    # Get object stats for normalization (ignoring padding)
    N_obj = (X_train.shape[1]-2)//5
    id_, E, pT, eta, phi = find_object_stats(X_train)
    mask = (id_!=0)
    obj_vals=[]
    # Per object statistics (mean/std per slot, ignoring masked-out paddings)
    for f in [E,pT,eta,phi]:
        vals = torch.where(mask,f,torch.nan)
        mean = torch.nanmean(vals,dim=0)
        std = torch.nanstd(vals,dim=0)+1e-6
        obj_vals.append((mean,std))
    # Stack per-object, shape: [N_obj, 4]
    obj_mean = torch.stack([t[0] for t in obj_vals],dim=1)  # [N_obj,4]
    obj_std  = torch.stack([t[1] for t in obj_vals],dim=1)
    obj_mean = obj_mean.transpose(0,1) # [N_obj,4]
    obj_std  = obj_std.transpose(0,1)
    preproc = PreprocessModule(obj_mean=obj_mean, obj_std=obj_std)
    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p,Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader, preproc

# ======== PARTICLE PHYSICS EQUIVARIANT NN DEFINITION ========

def batch_lorentz_invariant_features(x):
    # x: [batch, 2 + 5*N_obj]
    batch = x.shape[0]
    N_obj = (x.shape[1]-2)//5
    # MET: [batch,2]
    met  = x[:,:2]
    # Objects: [batch, N_obj, 5]
    obj_x = x[:,2:].reshape(batch,N_obj,5)
    obj_id = obj_x[:,:,0] # [batch,N_obj]
    E     = obj_x[:,:,1]
    pT    = obj_x[:,:,2]
    eta   = obj_x[:,:,3]
    phi   = obj_x[:,:,4]
    mask  = (obj_id!=0).float() # [batch,N_obj]
    # Compute px, py, pz per object
    px = pT*torch.cos(phi)
    py = pT*torch.sin(phi)
    pz = pT*torch.sinh(eta)
    # Stack four-vectors: [batch, N_obj, 4]
    pvec = torch.stack([E, px, py, pz],-1)
    # Set zero for padded
    pvec = pvec * mask.unsqueeze(-1)
    # MET fourvector
    MET_px = met[:,0]*torch.cos(met[:,1])
    MET_py = met[:,0]*torch.sin(met[:,1])
    MET_vec = torch.stack([torch.zeros_like(MET_px), MET_px, MET_py, torch.zeros_like(MET_px)],-1) # E=0, pz=0
    # Lorentz invariants: pairwise masses
    # m^2 = (p1+p2)^2 = E^2 - (px^2+py^2+pz^2)
    pvec_sum = pvec.unsqueeze(2) + pvec.unsqueeze(1) #[b,N_obj,N_obj,4]
    m2 = (pvec_sum[:,:,:,0])**2 - (pvec_sum[:,:,:,1]**2+pvec_sum[:,:,:,2]**2+pvec_sum[:,:,:,3]**2)
    m2 = m2 * mask.unsqueeze(2) * mask.unsqueeze(1)
    m2 = torch.nan_to_num(m2,nan=0.0,posinf=0.0,neginf=0.0)
    m_invariant = torch.sqrt(torch.relu(m2)+1e-8) # only valid when both are real objects
    # Sum over i<j, mean/moments
    N_pairs = torch.clamp(torch.sum(mask,dim=1)*(torch.sum(mask,dim=1)-1)/2,min=1)
    pair_mask = (torch.triu(torch.ones(N_obj,N_obj),1).to(x.device)).unsqueeze(0) # [1,N_obj,N_obj]
    m_invariant_masked = m_invariant*pair_mask
    mean_minv = torch.sum(m_invariant_masked,dim=[1,2])/N_pairs
    std_minv = torch.sqrt(torch.sum((m_invariant_masked-mean_minv.unsqueeze(-1).unsqueeze(-1))**2,dim=[1,2])/(N_pairs+1e-6))

    # Sum total four-vector (sum of all objects)
    total_pvec = torch.sum(pvec,dim=1) # [batch,4]
    # M_total
    m2_total = total_pvec[:,0]**2 - (total_pvec[:,1]**2+total_pvec[:,2]**2+total_pvec[:,3]**2)
    m2_total = torch.nan_to_num(m2_total,nan=0.0,posinf=0.0,neginf=0.0)
    M_total = torch.sqrt(torch.relu(m2_total)+1e-8)

    # Scalar sum/mean/std of pT
    sum_pT = torch.sum(pT*mask,dim=1)
    mean_pT = sum_pT/(torch.sum(mask,dim=1)+1e-3)
    std_pT = torch.sqrt(torch.sum(((pT-mean_pT.unsqueeze(-1))**2)*mask,dim=1)/(torch.sum(mask,dim=1)+1e-3))

    # Number of leptons/jets: assume id encodes classes:    
    n_obj = torch.sum(mask,dim=1)

    features = torch.stack([mean_minv,std_minv,M_total,sum_pT,mean_pT,std_pT,n_obj],-1)
    return features # [batch,7]

# Message-passing block with Lorentz symmetry: message is a function of Lorentz invariants
class LorentzEquivariantLayer(nn.Module):
    def __init__(self, obj_in_dim, message_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(obj_in_dim*2+1,message_dim), # object_i, object_j, & inv mass
            nn.ReLU(),
            nn.Linear(message_dim,message_dim),
            nn.ReLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(obj_in_dim+message_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,obj_in_dim),
        )
    def forward(self, x, mask, pvec):
        # x: [B,N,feat], mask: [B,N], pvec: [B,N,4]
        B,N,D = x.shape
        device = x.device
        # Build messages (i->j):
        xi = x.unsqueeze(2).expand(B,N,N,D) # [B,N,N,D]
        xj = x.unsqueeze(1).expand(B,N,N,D) # [B,N,N,D]
        # Lorentz inv mass
        pvec_sum = pvec.unsqueeze(2) + pvec.unsqueeze(1)
        m2 = (pvec_sum[:,:,:,0])**2 - (pvec_sum[:,:,:,1]**2 + pvec_sum[:,:,:,2]**2 + pvec_sum[:,:,:,3]**2 )
        m2 = m2 * mask.unsqueeze(2) * mask.unsqueeze(1)
        m = torch.sqrt(torch.relu(m2)+1e-7)
        edge_in = torch.cat([xi,xj,m.unsqueeze(-1)],-1) # [B,N,N,2D+1]
        msg_ij = self.edge_mlp(edge_in) # [B,N,N,message_dim]
        # Mask to valid
        msg_ij = msg_ij*mask.unsqueeze(1).unsqueeze(-1)*mask.unsqueeze(2).unsqueeze(-1)
        msg_agg = torch.sum(msg_ij,dim=2) # [B,N,message_dim] (aggregate over neighbors)
        # Node update
        node_in = torch.cat([x,msg_agg],-1)
        x_new = self.node_mlp(node_in)
        # Apply mask
        x_new = x_new*mask.unsqueeze(-1)
        return x_new

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        N_obj = (input_dim-2)//5
        self.N_obj = N_obj
        self.embedding = nn.Sequential(
            nn.Linear(5,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU()
        )
        # Two layers of Lorentz equivariant message passing
        self.equiv1 = LorentzEquivariantLayer(obj_in_dim=16,message_dim=16,hidden_dim=32)
        self.equiv2 = LorentzEquivariantLayer(obj_in_dim=16,message_dim=16,hidden_dim=32)
        # Output head on pooled outputs
        self.fc = nn.Sequential(
            nn.Linear(16+7+2,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
    def forward(self, x):
        # x: [B,input_dim], same as preprocess output
        B = x.shape[0]
        N_obj = self.N_obj
        met = x[:,:2]
        objs = x[:,2:].reshape(B,N_obj,5)
        id_ = objs[:,:,0]
        obj_mask = (id_!=0).float()
        # pvec construction (same as preprocess)
        E     = objs[:,:,1]
        pT    = objs[:,:,2]
        eta   = objs[:,:,3]
        phi   = objs[:,:,4]
        px = pT*torch.cos(phi)
        py = pT*torch.sin(phi)
        pz = pT*torch.sinh(eta)
        pvec = torch.stack([E,px,py,pz],-1)
        # Object embedding
        x_embed = self.embedding(objs)
        x_embed = x_embed*obj_mask.unsqueeze(-1)
        # Message passing
        x1 = self.equiv1(x_embed, obj_mask, pvec)
        x2 = self.equiv2(x1, obj_mask, pvec)
        # Pool across valid objects: mean pooling
        x_pool = torch.sum(x2*obj_mask.unsqueeze(-1),dim=1)/(torch.sum(obj_mask,dim=1,keepdim=True)+1e-6)
        # Lorentz-invariant event features
        lorentz_feats = batch_lorentz_invariant_features(x)
        # Final classifier head: [mean_embed | lorentz_feats | MET]
        out = torch.cat([x_pool,lorentz_feats,met],dim=1)
        pred = self.fc(out)
        return pred.squeeze(-1)

# ======== TRAINING LOOP IMPLEMENTATION ========
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    training_loss=[]
    validation_loss=[]
    training_acc=[]
    validation_acc=[]
    for epoch in range(epochs):
        model.train()
        train_losses=[]; train_accs=[]
        for xb,yb in train_loader:
            xb = xb.to(device)
            yb = yb.float().to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            # Accuracy: sigmoid
            preds = (torch.sigmoid(out)>0.5).float()
            acc = (preds==yb).float().mean().item()
            train_accs.append(acc)
        training_loss.append(np.mean(train_losses))
        training_acc.append(np.mean(train_accs))
        model.eval()
        val_losses=[]; val_accs=[]
        y_true_all=[]; y_pred_all=[]
        with torch.no_grad():
            for xb,yb in val_loader:
                xb = xb.to(device)
                yb = yb.float().to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_losses.append(loss.item())
                preds = (torch.sigmoid(out)>0.5).float()
                acc = (preds==yb).float().mean().item()
                val_accs.append(acc)
                y_true_all.append(yb.detach().cpu())
                y_pred_all.append(torch.sigmoid(out).detach().cpu())
        validation_loss.append(np.mean(val_losses))
        validation_acc.append(np.mean(val_accs))
        # Optional: print AUC per epoch
        if epoch==epochs-1 or epoch%1==0:
            y_true = torch.cat(y_true_all).numpy()
            y_pred = torch.cat(y_pred_all).numpy()
            auc = roc_auc_score(y_true,y_pred)
            print(f"Epoch {epoch+1}/{epochs} | Train loss {training_loss[-1]:.4f} | Val loss {validation_loss[-1]:.4f} | Train acc {training_acc[-1]:.4f} | Val acc {validation_acc[-1]:.4f} | VAL AUC {auc:.4f}")
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
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=512 if not dryrun else 32)
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

if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)