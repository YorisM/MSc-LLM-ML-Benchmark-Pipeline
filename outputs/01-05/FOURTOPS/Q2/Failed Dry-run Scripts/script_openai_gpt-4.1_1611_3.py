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
    def __init__(self, obj_mask, mean, std):
        super().__init__()
        self.register_buffer("obj_mask", obj_mask)  # [nobj]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Set all object-identifier-padded objects (id==0) to zero everywhere,
        # except the mask will flag all quantities except Etmiss,phiEtmiss.
        x_norm = (x - self.mean) / self.std
        # Zero out all obj slots for obj_id==0 (padding), except Etmiss block
        x_norm[:, 2:] = x_norm[:, 2:] * self.obj_mask  # broadcast
        return x_norm

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=512):
    device = 'cpu'
    # Decompose structure: [Etmiss, phiEtmiss] + 26 objects x [id, E, pt, eta, phi] = 2+ 26*4=105
    # obj_1_id at index 2, obj_2_id at index 7, etc.
    n_features = X_train.shape[1]
    nobj = (n_features - 2) // 4
    # Per field statistics for normalization (ignoring masked/padded values)
    # Find object mask: 1 where valid, 0 where padded (obj id == 0)
    mask_mat = torch.zeros_like(X_train)
    for i in range(nobj):
        obj_idx = 2 + 4*i
        obj_mask = (X_train[:, obj_idx] != 0).float().unsqueeze(1)
        mask_mat[:, obj_idx:obj_idx+4] = obj_mask
    # except Etmiss block always present
    mask_mat[:, :2] = 1
    # Compute mean/std only over non-masked entries
    mu = (X_train*mask_mat).sum(0) / (mask_mat.sum(0) + 1e-8)
    v = (((X_train - mu)**2)*mask_mat).sum(0) / (mask_mat.sum(0) + 1e-8)
    std = torch.sqrt(v + 1e-8)
    # Register mask for all but Etmiss block
    preproc = PreprocessModule(mask_mat[0:1].clone().detach(), mu, std)
    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p,   Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    return train_loader, val_loader, preproc

# ----- FREE SECTION: Lorentz-Equivariant Message Passing Layer -----
class LorentzTensorLayer(nn.Module):
    def __init__(self, f_in, f_out):
        super().__init__()
        self.fc = nn.Linear(f_in, f_out)
        self.bn = nn.BatchNorm1d(f_out)
    def forward(self, ten4):
        # ten4: [B, nobj, 4] (E, px, py, pz)
        # Process all four-vector slots linearly, 
        # preserving Lorentz symmetry via only linear transformation.
        B,N,F = ten4.shape
        x = ten4.view(-1, F)
        x = self.fc(x)
        x = torch.relu(self.bn(x))
        x = x.view(B,N,-1)
        return x

class LorentzPairwiseInvariant(nn.Module):
    # Returns a set of invariant features for each object pair, e.g., m^2 = (p1+p2)^2, deltaR, etc
    def __init__(self):
        super().__init__()
    def forward(self, fourvecs):
        # fourvecs: [B, nobj, 4], padded with zeros for non-existent
        B,N,_ = fourvecs.shape
        # Compute mask for real objects
        mask = (fourvecs.abs().sum(-1) > 0).float()
        output = []
        # Compute object-wise m^2 = E^2 - px^2 - py^2 - pz^2
        E,p_x,p_y,p_z = fourvecs[...,0], fourvecs[...,1], fourvecs[...,2], fourvecs[...,3]
        m2 = (E**2 - p_x**2 - p_y**2 - p_z**2)
        # Compute object-wise pt, eta, phi
        pt = torch.sqrt(p_x ** 2 + p_y ** 2 + 1e-5)
        phi = torch.atan2(p_y, p_x)
        p = torch.sqrt(pt**2 + p_z**2)
        
        eta = 0.5 * torch.log((p + p_z + 1e-8)/(p - p_z + 1e-8))
        # m2 is not strictly >=0, so .clamp
        m = m2.clamp(min=0).sqrt()
        # per object [pt, eta, phi, m]
        obj_feats = torch.stack([pt, eta, phi, m], dim=-1) * mask.unsqueeze(-1)
        output.append(obj_feats)
        # Sum all mass (global scalar)
        m_sum = m.sum(1,keepdim=True)
        output.append(m_sum)
        # Pairwise invariant masses
        m2_pairs = []
        for i in range(N):
            for j in range(i+1, N):
                m2ij = (fourvecs[:,i,0]+fourvecs[:,j,0])**2 - \
                       (fourvecs[:,i,1]+fourvecs[:,j,1])**2 - \
                       (fourvecs[:,i,2]+fourvecs[:,j,2])**2 - \
                       (fourvecs[:,i,3]+fourvecs[:,j,3])**2
                m2_pairs.append(m2ij.unsqueeze(1))
        if len(m2_pairs)>0:
            m2_pairs = torch.cat(m2_pairs, dim=1)
            m_pairs = m2_pairs.clamp(min=0).sqrt()
            # Sum/mean or pool
            pools = [m2_pairs.mean(1, keepdim=True),m_pairs.mean(1,keepdim=True),
                     m_pairs.max(1, keepdim=True)[0],m_pairs.min(1, keepdim=True)[0]]
            output += pools
        # Global features: number of objects
        n_real = mask.sum(1,keepdim=True)
        output.append(n_real)
        # Concatenate all scalar features
        return torch.cat([x.reshape(x.shape[0],-1) for x in output], dim=1)


# ----- FREE SECTION: Lorentz-Equivariant Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim, nobj=26):
        super(Classifier, self).__init__()
        # Input: [batch, 105]
        # 0:E_Tmiss 1:phi_{E_Tmiss} 2:obj_1_id 3:E_1 4:pt_1 5:eta_1 6:phi_1 ...
        self.nobj = nobj
        four_idx = []
        for i in range(self.nobj):
            # idx for [E, pt, eta, phi], id slot ignored
            base = 2 + 4*i
            four_idx.append([base+1, base+2, base+3, base+4])
        flat_idx = sum(four_idx,[])
        self.register_buffer("four_idx", torch.LongTensor(flat_idx))
        # MLP for Etmiss block
        self.etmiss_net = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU()
        )
        # Map from [E,pt,eta,phi] (for each object) -> 4-vector inputs [E,px,py,pz]
        # to construct Lorentz layers
        self.lorentz = LorentzTensorLayer(f_in=4, f_out=32)
        self.lorentz2 = LorentzTensorLayer(f_in=32, f_out=32)
        self.lorentz3 = LorentzTensorLayer(f_in=32, f_out=16)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pair_invariants = LorentzPairwiseInvariant()
        # Final classifier
        fc_in = self.nobj*16 + 16 + 36  # pooled lorentz + etmiss + invariant features
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self,x):
        B = x.shape[0]
        nobj = self.nobj
        device = x.device
        # Etmiss [B,2]
        etmiss = x[:,:2]
        em_1 = self.etmiss_net(etmiss)
        # Objects
        # [B, nobj, id+E+pt+eta+phi] (each 5d)
        feat = x[:,2:].reshape(B, nobj, 4)
        # Four-vector: input is (E, pt, eta, phi)->(E, px, py, pz)
        E, pt, eta, phi = feat[...,0], feat[...,1], feat[...,2], feat[...,3]
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)
        v4 = torch.stack([E, px, py, pz], dim=-1)
        # [B,nobj,4] -> GNN type equivariant update
        x1 = self.lorentz(v4)
        x2 = self.lorentz2(x1)
        x3 = self.lorentz3(x2)
        # Pool: [B,nobj,16]
        x3_flat = x3.reshape(B, -1)
        # Invariants (scalar for each event)
        inv = self.pair_invariants(v4)
        # Concat: pooled features + Etmiss + invariants
        x_tot = torch.cat([x3_flat, em_1, inv], dim=1)
        logits = self.classifier(x_tot).squeeze(-1)
        return logits

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    for ep in range(epochs):
        model.train()
        train_loss = 0
        train_logits = []
        train_targets = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_logits.append(logits.detach().cpu())
            train_targets.append(yb.detach().cpu())
        train_loss /= len(train_loader.dataset)
        training_loss.append(train_loss)
        # Calculate train acc/roc
        train_logits = torch.cat(train_logits).numpy()
        train_targets = torch.cat(train_targets).numpy()
        pred_probs = torch.sigmoid(torch.from_numpy(train_logits)).numpy()
        train_pred = (pred_probs > 0.5).astype(int)
        tr_acc = accuracy_score(train_targets, train_pred)
        training_acc.append(tr_acc)
        # Validation
        model.eval()
        val_loss = 0
        val_logits = []
        val_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_logits.append(logits.cpu())
                val_targets.append(yb.cpu())
        val_loss /= len(val_loader.dataset)
        validation_loss.append(val_loss)
        val_logits = torch.cat(val_logits).numpy()
        val_targets = torch.cat(val_targets).numpy()
        pred_probs = torch.sigmoid(torch.from_numpy(val_logits)).numpy()
        val_pred = (pred_probs > 0.5).astype(int)
        val_acc = accuracy_score(val_targets, val_pred)
        validation_acc.append(val_acc)
        val_auc = roc_auc_score(val_targets, pred_probs)
        print(f"Epoch {ep+1}: TL={train_loss:.4f} TA={tr_acc:.3f}, VL={val_loss:.4f} VA={val_acc:.3f} AUROC={val_auc:.4f}")
        scheduler.step()
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