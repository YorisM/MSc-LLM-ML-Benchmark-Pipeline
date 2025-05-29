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


class PreprocessModule(torch.nn.Module):
    def __init__(self, w_mean, w_std, etm_mean, etm_std, 
                 e_mean, e_std, pt_mean, pt_std, eta_mean, eta_std,
                 phis_mean, phis_std, phic_mean, phic_std):
        super().__init__()
        self.register_buffer("w_mean", w_mean)
        self.register_buffer("w_std", w_std)
        self.register_buffer("etm_mean", etm_mean)
        self.register_buffer("etm_std", etm_std)
        self.register_buffer("e_mean", e_mean)
        self.register_buffer("e_std", e_std)
        self.register_buffer("pt_mean", pt_mean)
        self.register_buffer("pt_std", pt_std)
        self.register_buffer("eta_mean", eta_mean)
        self.register_buffer("eta_std", eta_std)
        self.register_buffer("phis_mean", phis_mean)
        self.register_buffer("phis_std", phis_std)
        self.register_buffer("phic_mean", phic_mean)
        self.register_buffer("phic_std", phic_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process global
        log_w = (torch.log(x[:,0]) - self.w_mean) / self.w_std
        etm = (x[:,1] - self.etm_mean) / self.etm_std
        phis_global = torch.sin(x[:,2])
        phic_global = torch.cos(x[:,2])
        global_feat = torch.stack([log_w, etm, phis_global, phic_global], dim=1)
        
        # Process objects
        obj_feat = []
        for i in range(20):
            start = 3 + i*5
            obj_id = x[:,start]
            mask = obj_id != 0
            
            e = (x[:,start+1] - self.e_mean)/self.e_std
            pt = (x[:,start+2] - self.pt_mean)/self.pt_std
            eta = (x[:,start+3] - self.eta_mean)/self.eta_std
            phi = x[:,start+4]
            phis = (torch.sin(phi) - self.phis_mean)/self.phis_std
            phic = (torch.cos(phi) - self.phic_mean)/self.phic_std
            
            e = torch.where(mask, e, 0)
            pt = torch.where(mask, pt, 0)
            eta = torch.where(mask, eta, 0)
            phis = torch.where(mask, phis, 0)
            phic = torch.where(mask, phic, 0)
            obj_feat.append(torch.stack([obj_id, e, pt, eta, phis, phic], dim=1))
        
        obj_feat = torch.cat(obj_feat, dim=1)
        return torch.cat([global_feat, obj_feat], dim=1)


def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=512):
    # Compute stats
    with torch.no_grad():
        # Global features
        w_mean = torch.log(X_train[:,0]).mean()
        w_std = torch.log(X_train[:,0]).std()
        etm_mean = X_train[:,1].mean()
        etm_std = X_train[:,1].std()
        
        # Object features
        e_vals, pt_vals, eta_vals, phi_vals = [], [], [], []
        for i in range(20):
            start = 3 + i*5
            mask = X_train[:,start] != 0
            e_vals.append(X_train[mask, start+1])
            pt_vals.append(X_train[mask, start+2])
            eta_vals.append(X_train[mask, start+3])
            phi_vals.append(X_train[mask, start+4])
        
        e_vals = torch.cat(e_vals)
        e_mean, e_std = e_vals.mean(), e_vals.std()
        pt_vals = torch.cat(pt_vals)
        pt_mean, pt_std = pt_vals.mean(), pt_vals.std()
        eta_vals = torch.cat(eta_vals)
        eta_mean, eta_std = eta_vals.mean(), eta_vals.std()
        phi_vals = torch.cat(phi_vals)
        phis_mean = torch.sin(phi_vals).mean()
        phis_std = torch.sin(phi_vals).std()
        phic_mean = torch.cos(phi_vals).mean()
        phic_std = torch.cos(phi_vals).std()
        
    preproc = PreprocessModule(w_mean, w_std, etm_mean, etm_std,
                              e_mean, e_std, pt_mean, pt_std,
                              eta_mean, eta_std, phis_mean, phis_std,
                              phic_mean, phic_std)
    
    # Apply preprocessing
    X_train_p = preproc(X_train)
    X_val_p = preproc(X_val)
    
    # Create datasets
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds = TensorDataset(X_val_p, Y_val)
    
    # Create loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader, preproc


class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(model, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    train_loss, val_loss = [], []
    train_auc, val_auc = [], []
    
    for epoch in range(epochs):
        # Training
        model.train()
        preds, targets = [], []
        epoch_loss = 0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device).float()
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item() * len(x)
            preds.append(out.sigmoid().detach().cpu().numpy())
            targets.append(y.cpu().numpy())
        
        train_loss.append(epoch_loss / len(train_loader.dataset))
        train_auc.append(roc_auc_score(np.concatenate(targets), np.concatenate(preds)))
        
        # Validation
        model.eval()
        preds, targets = [], []
        epoch_vloss = 0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device).float()
                out = model(x)
                epoch_vloss += criterion(out, y).item() * len(x)
                preds.append(out.sigmoid().cpu().numpy())
                targets.append(y.cpu().numpy())
        
        val_loss.append(epoch_vloss / len(val_loader.dataset))
        val_auc.append(roc_auc_score(np.concatenate(targets), np.concatenate(preds)))
        
        print(f'Epoch {epoch+1}: Train AUC: {train_auc[-1]:.4f}, Val AUC: {val_auc[-1]:.4f}')
    
    return model, train_loss, val_loss, train_auc, val_auc


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
    # Data Loading
    X_train, Y_train, X_val, Y_val = load_data()

    # Preprocessing
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=512)

    # Model init
    sample_X, _ = next(iter(train_loader))
    model = Classifier(input_dim=sample_X.shape[1])

    # Training
    epochs = 1 if dryrun else 50
    trained_model, t_loss, v_loss, t_auc, v_auc = train_model(model, train_loader, val_loader, epochs)

    if not dryrun:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        os.makedirs(script_dir, exist_ok=True)
        
        torch.save(trained_model.state_dict(), os.path.join(script_dir, f"{base}_model.pth"))
        torch.jit.script(trained_model).save(os.path.join(script_dir, f"{base}_scripted.pt"))
        torch.jit.script(preproc).save(os.path.join(script_dir, f"{base}_preproc.pt"))
        
        plot_and_save(t_loss, v_loss, "Loss", os.path.join(script_dir, f"{base}_loss.png"))
        plot_and_save(t_auc, v_auc, "AUC", os.path.join(script_dir, f"{base}_auc.png"))


if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)