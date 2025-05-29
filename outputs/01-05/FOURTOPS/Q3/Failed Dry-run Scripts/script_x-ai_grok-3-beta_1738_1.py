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
    def __init__(self, means, stds, max_objects=26):
        super().__init__()
        self.register_buffer("means", torch.tensor(means, dtype=torch.float32))
        self.register_buffer("stds", torch.tensor(stds, dtype=torch.float32))
        self.max_objects = max_objects

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # Separate missing ET and object features
        met = x[:, :2]  # E_T_miss, phi_E_T_miss
        objects = x[:, 2:].view(batch_size, self.max_objects, 4)  # Reshape to (batch, objects, features)
        # Normalize MET
        met = (met - self.means[:2]) / (self.stds[:2] + 1e-8)
        # Normalize object features (E, p_T, eta, phi)
        objects = (objects - self.means[2:6].view(1, 1, -1)) / (self.stds[2:6].view(1, 1, -1) + 1e-8)
        # Augment features with physics-inspired quantities
        E = objects[:, :, 0]
        p_T = objects[:, :, 1]
        eta = objects[:, :, 2]
        phi = objects[:, :, 3]
        # Compute mass proxy (assuming massless particles for simplicity)
        mass_proxy = torch.sqrt(torch.clamp(E**2 - p_T**2, min=0.0))
        # Compute delta phi with MET
        delta_phi_met = torch.abs(phi - met[:, 1].view(-1, 1))
        delta_phi_met = torch.min(delta_phi_met, 2 * torch.pi - delta_phi_met)
        # Stack augmented features
        objects_aug = torch.stack([E, p_T, eta, phi, mass_proxy, delta_phi_met], dim=-1)
        return objects_aug.view(batch_size, -1)  # Flatten for model input

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=64):
    # Compute normalization statistics from training data
    means = X_train.mean(dim=0).numpy()
    stds = X_train.std(dim=0).numpy()
    preproc = PreprocessModule(means, stds)

    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)

    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p,   Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----
class SlotAttention(nn.Module):
    def __init__(self, num_slots=4, dim=64, eps=1e-8, iters=3):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.eps = eps
        self.iters = iters
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, num_slots, dim))
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        slots = self.slots_mu + torch.exp(self.slots_log_sigma) * torch.randn(b, n_s, self.dim, device=inputs.device)
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            dots = torch.einsum('bid,bjd->bij', q, k) * self.dim ** -0.5
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', v, attn)
            slots = self.gru(updates.reshape(-1, self.dim), slots_prev.reshape(-1, self.dim))
            slots = slots.reshape(b, -1, self.dim)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_objects=26, d_model=64, nhead=8, num_layers=3, num_slots=4):
        super().__init__()
        self.num_objects = num_objects
        self.d_model = d_model
        self.input_proj = nn.Linear(6, d_model)  # 6 features per object (E, p_T, eta, phi, mass_proxy, delta_phi_met)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_slots * d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_objects, -1)  # Shape: (batch, objects, features)
        x = self.input_proj(x)  # Project to d_model dimension
        x = self.transformer(x)  # Apply transformer encoder
        slots = self.slot_attention(x)  # Apply slot attention to group objects into slots
        x = slots.view(batch_size, -1)  # Flatten slots for classification
        x = self.classifier(x)
        return x

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    training_auc = []
    validation_auc = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        train_preds = []
        train_labels = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_auc_score = roc_auc_score(train_labels, train_preds)
        training_loss.append(avg_train_loss)
        training_acc.append(train_acc)
        training_auc.append(train_auc_score)

        model.eval()
        epoch_val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_auc_score = roc_auc_score(val_labels, val_preds)
        validation_loss.append(avg_val_loss)
    validation_acc.append(val_acc)
    validation_auc.append(val_auc_score)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc_score:.4f}")
        print(f"                 Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc_score:.4f}")

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
    sample_X, _ = next(iter(train_loader))
    model = TransformerClassifier(input_dim=sample_X.shape[1])

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