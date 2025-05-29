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
    def __init__(self, means=None, stds=None, object_mask=None):
        super().__init__()
        if means is not None:
            self.register_buffer('means', means)
            self.register_buffer('stds', stds)
        else:
            self.means = None
            self.stds = None
        if object_mask is not None:
            self.register_buffer('object_mask', object_mask)
        else:
            self.object_mask = None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standardize except paddings
        if self.means is not None:
            x = (x - self.means) / self.stds.clamp(min=1e-6)
        # Optionally mask padded features? (No-op since all batch events are zero padded at same location)
        return x

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=512):
    # Identify padding (zero-padded events)
    # Assume padded entries are all exact zero for all features of the object
    mask = (X_train != 0).float()
    # Compute mean/std only over non-padded entries, feature-wise
    mask_nonzero = (mask.sum(dim=0) > 0)  # keep columns which are not all padded
    feature_means = torch.where(mask_nonzero, (X_train*mask).sum(dim=0)/(mask.sum(dim=0)+1e-8), torch.zeros_like(mask_nonzero, dtype=X_train.dtype))
    feature_stds  = torch.where(mask_nonzero, torch.sqrt(((X_train - feature_means)**2*mask).sum(dim=0)/(mask.sum(dim=0)+1e-8)), torch.ones_like(mask_nonzero, dtype=X_train.dtype))
    # Object mask: where in each event the object slots start; E_Tmiss (1), phi(1), then objects, possibly weight (last col? From prompt not totally clear, we don't standardize last col)
    preproc = PreprocessModule(means=feature_means, stds=feature_stds, object_mask=None)
    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p,   Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, drop_last=False)
    return train_loader, val_loader, preproc

# ----------------------------------------
# LORENTZ EQUIVARIANT BLOCKS AND CLASSIFIER
# ----------------------------------------
def unsqueeze_ragged(x, max_objs, obj_feature_dim):
    '''
    x: [batch, 105]
    Split each row into vector of [batch, max_objs, obj_feature_dim]
    obj_feature_dim=5: (type, E, pT, eta, phi)
    '''
    bsz = x.shape[0]
    # E_Tmiss, phi_ETmiss, obj_1_type, E_1, pT_1, eta_1, phi_1, obj_2_type...
    # Each object feature: (type, E, pT, eta, phi), 5 features per object
    n_header = 2 # ETmiss, phi
    # objects -> (105-2)//5 = 20
    objects = x[:, n_header:-1]  # dropping weight (last col)
    obj_feats = objects.reshape(bsz, max_objs, obj_feature_dim)
    # The weight (xsig/N) is last feature in x: x[:, -1].unsqueeze(1)
    return x[:, 0:1], x[:, 1:2], obj_feats, x[:, -1].unsqueeze(1)

def lorentz_product(p1, p2):
    # Inputs: [batch, n_obj, 4] for (E, px, py, pz)
    # Lorentzian metric signature (+,-,-,-)
    # returns: [batch, n_obj, n_obj] matrix
    # p1, p2: [batch, n_obj, 4]
    # p1[...,0]*p2[...,0] - sum_{i=1..3} p1[...,i]*p2[...,i]
    prod = p1[...,0]*p2[...,0] - (p1[...,1]*p2[...,1] + p1[...,2]*p2[...,2] + p1[...,3]*p2[...,3])
    return prod

def pT_eta_phi_E_to_Epxpypz(x):
    # x: [batch, n, 4]: [E, pT, eta, phi]
    E = x[...,0]
    pT = x[...,1]
    eta = x[...,2]
    phi = x[...,3]
    px = pT * torch.cos(phi)
    py = pT * torch.sin(phi)
    pz = pT * torch.sinh(eta)
    return torch.stack([E, px, py, pz], dim=-1)

def object_mask(obj_arr):
    # obj_arr: [batch, n_obj, obj_feature_dim]
    # Returns batch, n_obj (1 if not all zeros)
    return (obj_arr.abs().sum(-1) > 0).float()

class LorentzTensorLayer(nn.Module):
    '''Message passing layer with Lorentz symmetry-awareness.'''
    def __init__(self, obj_in_dim, obj_out_dim, hidden_dim=32):
        super().__init__()
        self.obj_in_dim = obj_in_dim
        self.obj_out_dim = obj_out_dim
        self.hidden_dim = hidden_dim
        # Aggregate info from neighbors, symmetric under permutations
        self.neigh_embed = nn.Linear(obj_in_dim+1, hidden_dim) # +Lorentz prod
        self.self_embed  = nn.Linear(obj_in_dim, hidden_dim)
        self.lorentz_agg = nn.Linear(hidden_dim, obj_out_dim)
        self.act = nn.GELU()
    def forward(self, obj, obj_mask):
        # obj: [batch, n_obj, obj_in_dim]
        # obj_mask: [batch, n_obj] (1: exists, 0: padded)
        bsz, n_obj, fdim = obj.shape
        # Compute Lorentz four-vectors
        # input fea: [type, E, pT, eta, phi] (0,1,2,3,4)
        E_pT_eta_phi = obj[...,1:5] # shape [bsz, n_obj, 4]
        # Convert to (E, px, py, pz)
        fourv = pT_eta_phi_E_to_Epxpypz(E_pT_eta_phi)
        # Compute all pairwise Lorentz products
        f1 = fourv.unsqueeze(2)         # [batch, n_obj, 1, 4]
        f2 = fourv.unsqueeze(1)         # [batch, 1, n_obj, 4]
        lprod = lorentz_product(f1, f2) # [batch, n_obj, n_obj]
        # Add information from type label as well as four-vector and Lorentz product
        out = []
        for i in range(n_obj):
            # mask for existing neighbors (exclude self? include for symmetry)
            # for each object, aggregate info from others
            neigh_feat = torch.cat([obj, lprod[:,i].unsqueeze(-1)], dim=-1)   # [batch, n_obj, obj_in_dim+1]
            agg = (self.neigh_embed(neigh_feat) * obj_mask.unsqueeze(-1)).sum(dim=1) / (obj_mask.sum(dim=1, keepdim=True)+1e-6) # mean
            self_fea = self.self_embed(obj[:,i])
            res = self.act(agg + self_fea)
            out.append(res)
        stacked = torch.stack(out, dim=1) # [batch, n_obj, hidden]
        out2 = self.lorentz_agg(stacked)  # [batch, n_obj, obj_out_dim]
        return out2 * obj_mask.unsqueeze(-1)  # zero out paddings

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        # Parse number of objects and features from input_dim
        # x: [E_Tmiss, phi_ETmiss, obj_1_type, E, pT, eta, phi, ...], up to (nobj*5)
        self.n_header = 2
        self.weight_idx = input_dim - 1 # last col
        self.object_region = input_dim - 1 - self.n_header
        self.nobj = self.object_region // 5
        self.obj_feature_dim = 5
        # Build Lorentz tensor layers
        self.lt1 = LorentzTensorLayer(self.obj_feature_dim, 16, hidden_dim=32)
        self.lt2 = LorentzTensorLayer(16, 32, hidden_dim=32)
        self.lt3 = LorentzTensorLayer(32, 32, hidden_dim=32)
        # Final aggregate: sum/mean of all objects
        self.agg_fc = nn.Linear(32, 32)
        # Also process E_Tmiss (x0), phi (x1) and weight - concatenate later
        self.aux_fc = nn.Linear(2, 8)
        # Output
        self.final = nn.Sequential(
            nn.Linear(32+8, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        # Split features into ETmiss, phi, objects, and weight
        ETmiss, phimet, obj_feat, weights = unsqueeze_ragged(x, self.nobj, self.obj_feature_dim)
        # Mask for valid objects (all zeros = not present)
        mask = object_mask(obj_feat)
        # Layer 1: raw features
        h = self.lt1(obj_feat, mask)
        h = self.lt2(h, mask)
        h = self.lt3(h, mask)
        # aggregate objects: mean pool (avoid scale dependence on nobj)
        agg = (h*mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True)+1e-6)  # [batch, features]
        # process aux: ETmiss, phiETmiss
        aux = torch.cat([ETmiss, phimet], -1)
        aux_h = torch.relu(self.aux_fc(aux))  # [batch, 8]
        # Concatenate
        xcat = torch.cat([agg, aux_h], dim=-1) # [batch, 40]
        out = self.final(xcat)
        return out.squeeze(-1) # [batch]

# ----------------------------------------
# Training Loop
# ----------------------------------------
def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs//2,1), gamma=0.5)
    model.train()
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    for ep in range(epochs):
        ep_loss = 0.0
        ep_n = 0
        ep_logits = []
        ep_labels = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb.float())
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * xb.size(0)
            ep_n += xb.size(0)
            ep_logits.append(logits.detach().cpu())
            ep_labels.append(yb.detach().cpu())
        training_loss.append(ep_loss / ep_n)
        all_logits = torch.cat(ep_logits)
        all_labels = torch.cat(ep_labels)
        preds = (all_logits.sigmoid() > 0.5).int()
        train_acc = (preds == all_labels.int()).float().mean().item()
        training_acc.append(train_acc)
        # Validation
        model.eval()
        val_loss = 0.0
        val_n = 0
        val_logits = []
        val_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb.float())
                val_loss += loss.item() * xb.size(0)
                val_n += xb.size(0)
                val_logits.append(logits.detach().cpu())
                val_labels.append(yb.detach().cpu())
        validation_loss.append(val_loss / val_n)
        v_logits = torch.cat(val_logits).sigmoid().numpy()
        v_labels = torch.cat(val_labels).numpy()
        if v_labels.min() != v_labels.max():
            val_auc = roc_auc_score(v_labels, v_logits)
        else:
            val_auc = 0.5 # no signal
        preds = (v_logits > 0.5).astype(int)
        val_acc = (preds == v_labels).mean()
        validation_acc.append(val_acc)
        sched.step()
        model.train()
        print(f"Epoch {ep+1}/{epochs} | Train Loss: {training_loss[-1]:.4f}, Acc: {train_acc:.4f} | Val Loss: {validation_loss[-1]:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
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