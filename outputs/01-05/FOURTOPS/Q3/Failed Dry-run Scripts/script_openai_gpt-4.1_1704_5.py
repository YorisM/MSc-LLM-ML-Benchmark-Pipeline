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
    def __init__(self, mask, means, stds, obj_indices, batch_objects, add_features_params):
        super().__init__()
        self.register_buffer("mask", mask)
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.register_buffer("obj_indices", obj_indices)
        self.batch_objects = batch_objects
        # Parameters for physics-inspired features
        for k, v in add_features_params.items():
            if isinstance(v, torch.Tensor):
                self.register_buffer(k, v)
            else:
                setattr(self, k, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 105]
        x_masked = x * self.mask
        # Standardize only valid entries
        x_norm = (x_masked - self.means) / (self.stds + 1e-6)
        # Objects batching: [batch, nobj, 5] (type, E, pt, eta, phi)
        batch = x_norm[:, self.obj_indices]        # [batch, nobj*5]
        batch_objects = batch.view(-1, self.batch_objects, 5)
        # Add physics-inspired features for each object
        obj_types = batch_objects[:,:,0:1] # integer index (not one-hot)
        E = batch_objects[:,:,1]
        pt = batch_objects[:,:,2]
        eta = batch_objects[:,:,3]
        phi = batch_objects[:,:,4]
        # Augment with one hot for object type -- assume at most 10 types
        obj_type_onehot = torch.nn.functional.one_hot(obj_types.long().squeeze(-1), num_classes=10).float()
        # Mass from E,pt,eta,phi (for massless, E^2 = pt^2 cosh^2(eta)), but input E may have small mass, so use M^2=E^2 - pt^2 cosh^2(eta)
        mass2 = E**2 - pt**2 * torch.cosh(eta)**2
        mass = torch.sqrt(torch.clamp(mass2, min=0.))
        # (pt/E), (E - pt)
        pt_over_E = pt / (E+1e-5)
        E_minus_pt = E - pt
        # Stack all object-wise features
        obj_features = torch.cat([
            obj_type_onehot,      # [batch, nobj, 10]
            batch_objects[:,:,1:],# [batch, nobj, 4]
            mass.unsqueeze(-1),  # [batch, nobj, 1]
            pt_over_E.unsqueeze(-1),
            E_minus_pt.unsqueeze(-1)
        ], dim=-1) # [batch, nobj, k]
        # Now, global event features: missing ET, phi_MET, weight (first 3 columns)
        global_feats = x_norm[:, :3]  # [batch, 3]
        # For downstream, return (obj_features, global_feats)
        return obj_features, global_feats


def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=128):
    B, F = X_train.shape
    # Field map:
    # 0: E_T_miss
    # 1: phi_{E_t}_miss
    # 2: weight
    # then, for object_n: [type(int), E, pt, eta, phi] x 20 objects (since (105-3)//5 = 20)
    nobj = (F-3)//5    # 20 particles padded to max
    obj_indices = torch.arange(3, F)
    batch_objects = nobj
    # 1) Mask for zero-padding -- identify which objects are real using type!=0
    type_mask = (X_train[:, 3:F:5] != 0).float()  # [batch, nobj]
    mask = torch.ones_like(X_train)
    # For each event, all features corresponding to padded objects (type==0) set to 0
    for i in range(nobj):
        col_start = 3 + 5*i
        mask[type_mask[:,i]==0, col_start:col_start+5] = 0.
    # 2) Means/vars (ignore zero-padding)
    valid = mask.bool()
    means = torch.zeros(F)
    stds  = torch.ones(F)
    for i in range(F):
        vals = X_train[:,i][valid[:,i]]
        means[i] = vals.mean() if vals.numel()>0 else 0.
        stds[i]  = vals.std()  if vals.numel()>0 else 1.
    # Physics features hyperparams (none needed here)
    add_features_params = {}
    preproc = PreprocessModule(mask=mask[0], means=means, stds=stds,
                              obj_indices=obj_indices, batch_objects=batch_objects,
                              add_features_params=add_features_params)
    # Transform data
    Xu = X_train
    X_train_p = preproc(Xu)
    X_val_p = preproc(X_val)
    # Wrap for DataLoader (event objects, event globals) --> label
    class EventDataset(torch.utils.data.Dataset):
        def __init__(self, Xp, Y):
            self.Xobjs, self.G = Xp
            self.Y = Y
        def __getitem__(self,idx):
            return self.Xobjs[idx], self.G[idx], self.Y[idx]
        def __len__(self):
            return self.Y.shape[0]
    train_ds = EventDataset(X_train_p, Y_train)
    val_ds   = EventDataset(X_val_p,   Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    return train_loader, val_loader, preproc

# ----- FREE SECTION: Slot Attention -----
class SlotAttention(nn.Module):
    """
    Slot Attention module for grouping objects into K sets (ideally mapping to top quark decay objects).
    """
    def __init__(self, n_slots, dim, iters=3, slot_dim=64):
        super().__init__()
        self.n_slots = n_slots
        self.iters = iters
        self.scale = dim ** -0.5
        # Parameters
        self.slot_mu = nn.Parameter(torch.randn(1, n_slots, slot_dim))
        self.slot_logsigma = nn.Parameter(torch.zeros(1, n_slots, slot_dim))
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(dim, slot_dim, bias=False)
        self.project_v = nn.Linear(dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim), nn.ReLU(),
            nn.Linear(slot_dim, slot_dim))
    def forward(self, inputs):
        # inputs: [B, N_obj, dim]
        B, N, D = inputs.shape
        # Initialize slots
        mu = self.slot_mu.expand(B, -1, -1)             # [B, n_slots, slot_dim]
        sigma = torch.exp(self.slot_logsigma).expand(B, self.n_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)  # [B, n_slots, slot_dim]
        x = self.norm_inputs(inputs)
        for _ in range(self.iters):
            slots_prev = slots
            # Attention: slots as queries, objects as keys/values
            k = self.project_k(x)  # [B,N,slot_dim]
            v = self.project_v(x)  # [B,N,slot_dim]
            q = self.project_q(self.norm_slots(slots)) # [B, n_slots, slot_dim]
            dots = torch.einsum('bid,bjd->bij', k, q) * self.scale   # [B,N,n_slots]
            attn = torch.softmax(dots, dim=2) + 1e-8  # N obj per slot (soft assignment)
            attn = attn / attn.sum(dim=1, keepdim=True)  # normalize slots over N
            updates = torch.einsum('bjn,bjd->bnd', attn.transpose(1,2), v)  # [B, n_slots, slot_dim]
            # GRU update (flatten for torch)
            slots = self.gru(
                updates.reshape(B*self.n_slots, -1),
                slots_prev.reshape(B*self.n_slots, -1)
            ).reshape(B, self.n_slots, -1)
            # MLP
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots  # [B, n_slots, slot_dim]

# ----- FREE SECTION: Transformer-based Event Encoder -----
class ParticleTransformer(nn.Module):
    def __init__(self, obj_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_layer = nn.Linear(obj_dim, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True),
            num_layers)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        # x: [batch, nobj, obj_dim]
        x = self.input_layer(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x       # [batch, nobj, d_model]

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        # Determine input shape: We know our input is (obj_features, global_feats)
        # One-hot type (10d) + 4 obj features + 3 augment = 17
        self.nobj = (input_dim-3)//5
        obj_dim = 10+4+3  # as constructed in PreprocessModule
        self.obj_dim = obj_dim
        self.transformer = ParticleTransformer(obj_dim=obj_dim, d_model=64, nhead=4, num_layers=2)
        self.slot_attention = SlotAttention(n_slots=4, dim=64, iters=3, slot_dim=64)
        # process slot output for classification
        self.global_mlp = nn.Sequential(
            nn.Linear(4*64+3, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x_tuple):
        # Unpack tuple: (obj_features, global_feats)
        x, global_feats = x_tuple      # x: [batch, nobj, obj_dim], global_feats: [batch,3]
        obj_embed = self.transformer(x)    # [batch, nobj, 64]
        slots = self.slot_attention(obj_embed) # [batch, 4, 64]
        slots_flat = slots.flatten(1)  # [batch, 4*64]
        feats = torch.cat([slots_flat, global_feats], dim=-1)
        y = self.global_mlp(feats)              # [batch,1]
        return y.squeeze(-1)                    # [batch]

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    training_loss, validation_loss = [], []
    training_acc, validation_acc = [], []
    best_auc = 0.0
    for ep in range(epochs):
        model.train()
        ep_losses = []
        preds_train, targets_train = [], []
        for X_obj, X_glob, y in train_loader:
            X_obj, X_glob, y = X_obj.to(device), X_glob.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model((X_obj, X_glob))
            loss = criterion(logits, y.float())
            loss.backward()
            optimizer.step()
            ep_losses.append(loss.item())
            preds_train.append(torch.sigmoid(logits).detach().cpu())
            targets_train.append(y.detach().cpu())
        training_loss.append(np.mean(ep_losses))
        # Compute training accuracy
        p_train = torch.cat(preds_train,0)
        t_train = torch.cat(targets_train,0)
        training_acc.append(accuracy_score(t_train, (p_train>0.5).int()))
        # Validation
        model.eval()
        val_losses = []
        preds_val, targets_val = [], []
        with torch.no_grad():
            for X_obj, X_glob, y in val_loader:
                X_obj, X_glob, y = X_obj.to(device), X_glob.to(device), y.to(device)
                logits = model((X_obj, X_glob))
                vloss = criterion(logits, y.float())
                val_losses.append(vloss.item())
                preds_val.append(torch.sigmoid(logits).cpu())
                targets_val.append(y.cpu())
        validation_loss.append(np.mean(val_losses))
        p_val = torch.cat(preds_val,0)
        t_val = torch.cat(targets_val,0)
        validation_acc.append(accuracy_score(t_val, (p_val>0.5).int()))
        try:
            auc = roc_auc_score(t_val.numpy(), p_val.numpy())
        except Exception:
            auc = 0.0
        print(f"Epoch {ep+1}: train loss={training_loss[-1]:.4f}, val loss={validation_loss[-1]:.4f}, val AUC={auc:.4f}")
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
    sample_Xobj, sample_G, _ = next(iter(train_loader))
    model = Classifier(input_dim=X_train.shape[1])

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