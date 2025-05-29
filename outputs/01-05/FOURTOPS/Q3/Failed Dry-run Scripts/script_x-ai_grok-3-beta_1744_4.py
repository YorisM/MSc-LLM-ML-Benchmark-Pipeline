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
    def __init__(self, means, stds, max_energy, max_pt):
        super().__init__()
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.register_buffer("max_energy", max_energy)
        self.register_buffer("max_pt", max_pt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # Extract missing ET and phi
        et_miss = x[:, 0:1] / self.max_pt
        phi_et_miss = x[:, 1:2]
        # Normalize kinematic features for objects
        obj_features = x[:, 2:].reshape(batch_size, -1, 5)
        mask = (obj_features[:, :, 0] > 0).float().unsqueeze(-1)  # Valid object indicator
        kinematics = obj_features[:, :, 1:]
        # Normalize energy and pT
        kinematics[:, :, 0:1] = kinematics[:, :, 0:1] / self.max_energy
        kinematics[:, :, 1:2] = kinematics[:, :, 1:2] / self.max_pt
        # Standardize eta and phi
        kinematics[:, :, 2:3] = (kinematics[:, :, 2:3] - self.means[0]) / self.stds[0]
        kinematics[:, :, 3:4] = (kinematics[:, :, 3:4] - self.means[1]) / self.stds[1]
        # Compute augmented features (e.g., mass approximation)
        pt = kinematics[:, :, 1:2]
        eta = kinematics[:, :, 2:3]
        phi = kinematics[:, :, 3:4]
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)
        energy = kinematics[:, :, 0:1]
        mass_approx = torch.sqrt(energy**2 - px**2 - py**2 - pz**2 + 1e-6)
        mass_approx = torch.clamp(mass_approx, min=0.0, max=1.0)
        augmented_features = torch.cat([kinematics, mass_approx], dim=-1)
        final_features = torch.cat([augmented_features, mask], dim=-1)
        # Reshape to include global features
        global_features = torch.cat([et_miss, phi_et_miss], dim=-1)
        global_features = global_features.unsqueeze(1).expand(batch_size, final_features.shape[1], 2)
        final_features = torch.cat([final_features, global_features], dim=-1)
        return final_features

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size):
    # Compute statistics for normalization
    obj_data = X_train[:, 2:].reshape(-1, 5)
    valid_mask = obj_data[:, 0] > 0
    eta_phi = obj_data[valid_mask, 2:4]
    means = torch.mean(eta_phi, dim=0)
    stds = torch.std(eta_phi, dim=0) + 1e-6
    max_energy = torch.max(obj_data[valid_mask, 1]) + 1e-6
    max_pt = torch.max(obj_data[valid_mask, 2]) + 1e-6

    preproc = PreprocessModule(means, stds, max_energy, max_pt)

    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)

    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p,   Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, num_slots, dim))
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_logsigma)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
        slots = mu + sigma * torch.randn_like(sigma)
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            dots = torch.einsum('bid,bjd->bij', q, k) * (1.0 / np.sqrt(d))
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', v, attn)
            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, attn

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=9, num_slots=4, slot_dim=64, num_heads=4, dim_feedforward=128, num_layers=2):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.input_proj = nn.Linear(input_dim, slot_dim)
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=slot_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=slot_dim, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp_head = nn.Sequential(
            nn.Linear(num_slots * slot_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        b, n, d = x.shape
        x = self.input_proj(x)
        slots, _ = self.slot_attention(x)
        slots_encoded = self.transformer_encoder(slots)
        slots_flat = slots_encoded.reshape(b, -1)
        logits = self.mlp_head(slots_flat)
        return logits

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs, learning_rate=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    validation_auc = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())
        train_loss_avg = train_loss / len(train_loader)
        train_acc_epoch = accuracy_score(train_labels, train_preds)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        val_loss_avg = val_loss / len(val_loader)
        val_acc_epoch = accuracy_score(val_labels, val_preds)
        val_auc_epoch = roc_auc_score(val_labels, val_probs)

        scheduler.step(val_auc_epoch)

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc_epoch:.4f}')
        print(f'Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc_epoch:.4f}, Val AUC: {val_auc_epoch:.4f}')

        training_loss.append(train_loss_avg)
        validation_loss.append(val_loss_avg)
        training_acc.append(train_acc_epoch)
        validation_acc.append(val_acc_epoch)
        validation_auc.append(val_auc_epoch)

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
    batch_size = 128
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size)

    # Model Initialization
    sample_X, _ = next(iter(train_loader))
    model = TransformerClassifier(input_dim=sample_X.shape[-1])

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